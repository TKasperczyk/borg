import type { LLMCompleteOptions, TokenUsageSink } from "./index.js";
import { addLlmUsage, attachLlmErrorUsage, reportLlmErrorUsage, type LLMUsage } from "./usage.js";

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
    parsed.family === "claude-opus"
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
};

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
    "model" | "tool_choice" | "tools" | "budget" | "signal" | "onToolChoiceTranslated"
  >;
  send: (context: {
    previous: Response | undefined;
    signal: AbortSignal | null | undefined;
    expectedTool: string | null;
  }) => Promise<Response>;
  hasToolUse: (response: Response) => boolean;
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
    let usage: LLMUsage | undefined;
    const run = async (signal?: AbortSignal | null): Promise<Response> => {
      const first = await args.send({ previous: undefined, signal, expectedTool });
      usage = args.getUsage(first);
      if (args.hasToolUse(first)) return first;
      retryFired = true;
      const second = await args.send({ previous: first, signal, expectedTool });
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
      const event = {
        label: options.budget,
        model: options.model,
        expected_tool: expectedTool,
        retry_fired: retryFired,
        attempt: retryFired ? 2 : 1,
      };
      console.warn("llm_call.tool_choice_translated", event);
      options.onToolChoiceTranslated?.(event);
    }
  })();
}
