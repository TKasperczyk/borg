import type { LLMCompleteOptions, TokenUsageSink } from "./index.js";
import { addLlmUsage, attachLlmErrorUsage, reportLlmErrorUsage, type LLMUsage } from "./usage.js";
import { normalizeLlmStopReason } from "./stop-reason.js";

function parseClaudeModel(model: string): { family: string; version: number[] } | null {
  const match = /^(claude-[a-z]+)-(\d+(?:[-.]\d+)*)(?:[-._].+)?$/i.exec(model.trim());
  if (match === null) return null;
  const components = match[2]!.split(/[-.]/);
  // Checkpoint dates are suffixes, not minor/patch versions, even after a major-only id.
  const dateIndex = components.findIndex((component) => /^\d{8}$/.test(component));
  return {
    family: match[1]!.toLowerCase(),
    version: (dateIndex < 0 ? components : components.slice(0, dateIndex)).map(Number),
  };
}

export function isOpusModel(model: string): boolean {
  return parseClaudeModel(model)?.family === "claude-opus";
}

export function supportsForcedToolUse(model: string): boolean {
  const parsed = parseClaudeModel(model);
  if (parsed === null) return true;
  const firstUnsupportedMinor =
    parsed.family === "claude-opus" || parsed.family === "claude-sonnet"
      ? 5
      : parsed.family === "claude-fable" || parsed.family === "claude-mythos"
        ? 1
        : undefined;
  if (firstUnsupportedMinor === undefined) return true;
  const [major = 0, minor = 0] = parsed.version;
  return major < 5 || (major === 5 && minor < firstUnsupportedMinor);
}

export type LLMToolChoiceTranslationEvent = {
  label: string;
  model: string;
  expected_tool: string | null;
  retry_fired: boolean;
  attempt: number;
  // Present when retry_fired: the first response, which the caller never receives
  // but the model reads back as its own turn before the reminder.
  first_stop_reason?: string | null;
  first_text_chars?: number;
};

/**
 * Sampling parameters (`temperature`, `top_p`, `top_k`) are rejected on every Opus
 * generation this harness runs, and at any non-default value on Sonnet >= 5.5 and
 * Haiku >= 5.5. Callers keep passing their value; the transport drops it for these
 * models instead of each call site branching on the model.
 */
export function rejectsSamplingParameters(model: string): boolean {
  const parsed = parseClaudeModel(model);
  if (parsed === null) return false;
  if (parsed.family === "claude-opus") return true;
  if (parsed.family !== "claude-sonnet" && parsed.family !== "claude-haiku") return false;
  const [major = 0, minor = 0] = parsed.version;
  return major > 5 || (major === 5 && minor >= 5);
}

/**
 * Whether a request that omits `effort` must send the shared default explicitly.
 * The tier that dropped forced tool use (Opus >= 5.5, Sonnet >= 5.5, Fable >= 5.1,
 * Mythos >= 5.1) gets it: Opus 5.5 changed its omitted-effort default to medium, and
 * Sonnet 5.5 already defaults to high, so sending high there changes nothing. Haiku
 * never receives the default: Haiku 4.5 rejects the parameter outright ("This model
 * does not support the effort parameter"), and Haiku 5.5 runs its small structured
 * calls at its own default.
 */
export function needsExplicitEffortDefault(model: string): boolean {
  return !supportsForcedToolUse(model);
}

export function toolUseReminder(expectedTool: string | null): string {
  return expectedTool === null
    ? "Answer only by calling one of the provided tools."
    : `Answer only by calling the ${expectedTool} tool.`;
}

// Both wire protocols use the same one-shot policy. Each backend preserves its
// native assistant response when constructing the reminder conversation.
export function translateForcedToolUse<Response>(args: {
  options: Pick<
    LLMCompleteOptions,
    | "model"
    | "tool_choice"
    | "tools"
    | "budget"
    | "signal"
    | "onToolChoiceTranslated"
    | "outputLimitRepairBudget"
  >;
  provider?: "anthropic" | "openai";
  send: (context: {
    previous: Response | undefined;
    signal: AbortSignal | null | undefined;
    expectedTool: string | null;
  }) => Promise<Response>;
  hasToolUse: (response: Response) => boolean;
  describe: (response: Response) => { stop_reason: string | null; text_chars: number };
  getUsage: (response: Response) => LLMUsage;
  withUsage: (response: Response, usage: LLMUsage) => Response;
  run?: (operation: (signal?: AbortSignal | null) => Promise<Response>) => Promise<Response>;
  usageSink?: TokenUsageSink;
}): Promise<Response> | undefined {
  const { options } = args;
  const choice = options.tool_choice;
  if (supportsForcedToolUse(options.model) || (choice?.type !== "any" && choice?.type !== "tool")) {
    return undefined;
  }
  const expectedTool =
    choice.type === "tool"
      ? choice.name
      : options.tools?.length === 1
        ? options.tools[0]!.name
        : null;

  return (async () => {
    let retryFired = false;
    let firstShape: { stop_reason: string | null; text_chars: number } | undefined;
    let usage: LLMUsage | undefined;
    const run = async (signal?: AbortSignal | null): Promise<Response> => {
      const first = await args.send({ previous: undefined, signal, expectedTool });
      if (options.outputLimitRepairBudget !== undefined) {
        options.outputLimitRepairBudget.remainingCalls -= 1;
      }
      usage = args.getUsage(first);
      const cut =
        options.outputLimitRepairBudget !== undefined &&
        normalizeLlmStopReason(args.describe(first).stop_reason, undefined, args.provider).kind ===
          "max_tokens";
      if (cut && options.outputLimitRepairBudget !== undefined) {
        options.outputLimitRepairBudget.enforceLimit = true;
      }
      if (args.hasToolUse(first)) return first;
      if (
        options.outputLimitRepairBudget !== undefined &&
        (cut ||
          (options.outputLimitRepairBudget.enforceLimit !== false &&
            options.outputLimitRepairBudget.remainingCalls <= 0))
      )
        return first;
      retryFired = true;
      firstShape = args.describe(first);
      const second = await args.send({ previous: first, signal, expectedTool });
      if (options.outputLimitRepairBudget !== undefined) {
        options.outputLimitRepairBudget.remainingCalls -= 1;
      }
      usage = addLlmUsage(usage, args.getUsage(second));
      return args.withUsage(second, usage);
    };
    try {
      return await (args.run === undefined ? run(options.signal) : args.run(run));
    } catch (error) {
      // Outside the deadline race: accounting must reach callers before rejection,
      // even if an aborted SDK operation takes longer to unwind.
      if (usage !== undefined) attachLlmErrorUsage(error, usage);
      await reportLlmErrorUsage(error, args.usageSink, {
        budget: options.budget,
        model: options.model,
      });
      throw error;
    } finally {
      const event: LLMToolChoiceTranslationEvent = {
        label: options.budget,
        model: options.model,
        expected_tool: expectedTool,
        retry_fired: retryFired,
        attempt: retryFired ? 2 : 1,
        ...(firstShape === undefined
          ? {}
          : { first_stop_reason: firstShape.stop_reason, first_text_chars: firstShape.text_chars }),
      };
      console.warn("llm_call.tool_choice_translated", event);
      options.onToolChoiceTranslated?.(event);
    }
  })();
}
