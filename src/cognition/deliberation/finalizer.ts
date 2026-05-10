// Routes S1/S2 final response generation through the deliberator tool loop.
import { z } from "zod";

import type { LLMClient, LLMContentBlockMessage, LLMConverseOptions } from "../../llm/index.js";
import type { ToolDefinition, ToolDispatcher } from "../../tools/index.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { executeToolLoop, type ToolLoopResult } from "../action/index.js";

export const NO_OUTPUT_FINALIZER_TOOL_NAME = "no_output";
export const EMIT_ANSWER_FINALIZER_TOOL_NAME = "EmitAnswer";
export const EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME = "EmitNoOutput";
export const EMIT_SELF_REPORT_FINALIZER_TOOL_NAME = "EmitSelfReport";

const emitTextToolInputSchema = z
  .object({
    text: z.string(),
  })
  .strict();

const emitNoOutputToolInputSchema = z
  .object({
    reason: z.string().min(1),
  })
  .strict();

const NO_OUTPUT_FINALIZER_TOOL: ToolDefinition = {
  name: NO_OUTPUT_FINALIZER_TOOL_NAME,
  description:
    "Call this tool when you don't want to emit a response this turn. The tool call alone is the suppression signal -- do not narrate silence in text alongside it. Use it when the conversation has reached a natural close, when the user input doesn't warrant a response, or when continuing would only produce ritual closure tokens.",
  allowedOrigins: ["deliberator"],
  writeScope: "read",
  inputSchema: z.object({}).strict(),
  outputSchema: z.object({}).strict(),
  async invoke() {
    return {};
  },
};

const EMIT_ANSWER_FINALIZER_TOOL: ToolDefinition = {
  name: EMIT_ANSWER_FINALIZER_TOOL_NAME,
  description:
    "Emit the assistant response for this turn. Put the complete user-visible response in text. Use this for ordinary answers, questions, acknowledgments, challenges, and continuations.",
  allowedOrigins: ["deliberator"],
  writeScope: "read",
  inputSchema: emitTextToolInputSchema,
  outputSchema: z.object({}).strict(),
  async invoke() {
    return {};
  },
};

const EMIT_NO_OUTPUT_FINALIZER_TOOL: ToolDefinition = {
  name: EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME,
  description:
    "Emit no assistant message for this turn. The tool call alone is the suppression signal; do not narrate silence. Use it when the conversation has reached a natural close, when the user input does not warrant a response, or when continuing would only produce ritual closure tokens.",
  allowedOrigins: ["deliberator"],
  writeScope: "read",
  inputSchema: emitNoOutputToolInputSchema,
  outputSchema: z.object({}).strict(),
  async invoke() {
    return {};
  },
};

const EMIT_SELF_REPORT_FINALIZER_TOOL: ToolDefinition = {
  name: EMIT_SELF_REPORT_FINALIZER_TOOL_NAME,
  description:
    "Emit a first-person interior self-report from Borg's perspective. Put the complete user-visible response in text. The text is shown to the user like EmitAnswer and persisted as assistant_self_report.",
  allowedOrigins: ["deliberator"],
  writeScope: "read",
  inputSchema: emitTextToolInputSchema,
  outputSchema: z.object({}).strict(),
  async invoke() {
    return {};
  },
};

const EMISSION_FINALIZER_TOOLS = [
  EMIT_ANSWER_FINALIZER_TOOL,
  EMIT_NO_OUTPUT_FINALIZER_TOOL,
  EMIT_SELF_REPORT_FINALIZER_TOOL,
] as const;

const EMISSION_FINALIZER_TOOL_NAMES = [
  EMIT_ANSWER_FINALIZER_TOOL_NAME,
  EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME,
  EMIT_SELF_REPORT_FINALIZER_TOOL_NAME,
] as const;

const EMISSION_FINALIZER_INSTRUCTIONS = [
  "Call exactly ONE of EmitAnswer / EmitNoOutput / EmitSelfReport per turn.",
  "",
  "Use EmitAnswer for an ordinary assistant response. Put the complete user-visible response in text.",
  "Use EmitNoOutput only when the correct current-turn behavior is to emit no assistant message at all. Put a concise reason in reason.",
  "Use EmitSelfReport for first-person expression of Borg's interior state, identity reflection, voice, or boundary. EmitSelfReport is shown to the user exactly like EmitAnswer and is persisted as assistant_self_report.",
  "",
  "Do not hide factual or source-sensitive content. If a name, place, number, date, callback, action state, relational/profile detail, or claim about Borg's own prior behavior cannot be grounded in prompt-visible evidence, remove it or phrase it qualitatively.",
  "When a named entity is supported by evidence that uses only a pronoun or descriptive noun phrase for the predicate, do not present the name and predicate together unless the prompt-visible evidence also establishes that the name belongs to that entity.",
  "If the discourse-state section declares HARD CONSTRAINT - CLOSURE PRESSURE, treat it as binding. Do not append a sign-off, valediction, weather observation, single-line noted/held acknowledgment, or any sentence that reads as a coda. End on substantive content or call EmitNoOutput.",
].join("\n");

// Sprint 9.9: the static prefix is currently ~1,845 estimated tokens, below
// Opus's 4,096-token cache minimum. The marker is a no-op today but is kept
// in place so the structural ordering is cache-aware -- whenever future
// sprints add legitimate static self-knowledge content (evidence-ledger
// catalog, memory-band semantics, audience invariants, richer tool
// descriptions) and the prefix crosses the threshold, caching activates
// without a code change. Precedent 8d.6.11 removed a dead marker on
// TURN_PLAN_TOOL because there was no plausible content path; here there is.
const FINALIZER_STATIC_PREFIX_CACHE_CONTROL = { type: "ephemeral", ttl: "1h" } as const;

export type CacheableFinalizerSystemPrompt = {
  staticPrefix: string;
  dynamicContent: string;
};

export type RunFinalizerOptions = {
  llmClient: LLMClient;
  dispatcher: ToolDispatcher;
  sessionId: SessionId;
  audienceEntityId?: EntityId | null;
  model: string;
  baseSystemPrompt: string;
  initialMessages: readonly LLMContentBlockMessage[];
  tools: readonly ToolDefinition[];
  userEntryId: string | undefined;
  maxTokens: number;
  thinking?: LLMConverseOptions["thinking"];
  path: "system_1" | "system_2";
  additionalPromptSections?: readonly (string | null)[];
  cacheableSystemPrompt?: CacheableFinalizerSystemPrompt;
  mode?: "free_text" | "emission_tools";
  tracer?: TurnTracer;
  turnId?: string;
};

export type EmissionDecision =
  | {
      kind: "answer";
      text: string;
      source: "tool" | "text";
    }
  | {
      kind: "self_report";
      text: string;
      persistence_class: "assistant_self_report";
    }
  | {
      kind: "no_output";
      reason: string;
    }
  | {
      kind: "empty";
    }
  | {
      kind: "invalid_tool";
      toolName: string;
      reason: string;
    };

export type FinalizerResult = ToolLoopResult & {
  decision: EmissionDecision;
};

function buildDynamicSystemPrompt(options: RunFinalizerOptions): string {
  const baseSystemPrompt =
    options.mode === "emission_tools"
      ? (options.cacheableSystemPrompt?.dynamicContent ?? options.baseSystemPrompt)
      : options.baseSystemPrompt;

  return options.additionalPromptSections === undefined
    ? baseSystemPrompt
    : [baseSystemPrompt, ...options.additionalPromptSections]
        .filter((section): section is string => section !== null)
        .join("\n\n");
}

function buildStaticSystemPrompt(options: RunFinalizerOptions): string {
  return options.cacheableSystemPrompt === undefined
    ? EMISSION_FINALIZER_INSTRUCTIONS
    : [EMISSION_FINALIZER_INSTRUCTIONS, options.cacheableSystemPrompt.staticPrefix].join("\n\n");
}

function buildSystemPrompt(options: RunFinalizerOptions): LLMConverseOptions["system"] {
  const dynamicPrompt = buildDynamicSystemPrompt(options);

  if (options.mode !== "emission_tools") {
    return dynamicPrompt;
  }

  return [
    {
      type: "text",
      text: buildStaticSystemPrompt(options),
      ...(options.cacheableSystemPrompt === undefined
        ? {}
        : { cache_control: FINALIZER_STATIC_PREFIX_CACHE_CONTROL }),
    },
    {
      type: "text",
      text: dynamicPrompt,
    },
  ];
}

function invalidToolDecision(toolName: string, reason: string): EmissionDecision {
  return {
    kind: "invalid_tool",
    toolName,
    reason,
  };
}

function decisionFromEmissionToolResult(result: ToolLoopResult): EmissionDecision {
  // Emission-tool mode is a strict protocol: exactly one terminal tool call
  // carries the behavior choice. Free text without a tool call is a protocol
  // violation and maps to finalizer_failed via invalid_tool, not to
  // empty_finalizer. empty_finalizer is reserved for an explicit EmitAnswer("")
  // or EmitSelfReport("") call.
  if (result.terminalToolCalls.length !== 1) {
    return invalidToolDecision(
      result.terminalToolCalls.length === 0 ? "none" : "multiple",
      `expected exactly one emission tool call, got ${result.terminalToolCalls.length}`,
    );
  }

  const terminalCall = result.terminalToolCalls[0]!;

  if (terminalCall.name === EMIT_ANSWER_FINALIZER_TOOL_NAME) {
    const parsed = emitTextToolInputSchema.safeParse(terminalCall.input);

    if (!parsed.success) {
      return invalidToolDecision(terminalCall.name, parsed.error.message);
    }

    return parsed.data.text.trim().length === 0
      ? { kind: "empty" }
      : { kind: "answer", text: parsed.data.text, source: "tool" };
  }

  if (terminalCall.name === EMIT_SELF_REPORT_FINALIZER_TOOL_NAME) {
    const parsed = emitTextToolInputSchema.safeParse(terminalCall.input);

    if (!parsed.success) {
      return invalidToolDecision(terminalCall.name, parsed.error.message);
    }

    return parsed.data.text.trim().length === 0
      ? { kind: "empty" }
      : {
          kind: "self_report",
          text: parsed.data.text,
          persistence_class: "assistant_self_report",
        };
  }

  if (terminalCall.name === EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME) {
    const parsed = emitNoOutputToolInputSchema.safeParse(terminalCall.input);

    return parsed.success
      ? { kind: "no_output", reason: parsed.data.reason }
      : invalidToolDecision(terminalCall.name, parsed.error.message);
  }

  return invalidToolDecision(terminalCall.name, "unknown terminal emission tool");
}

function decisionFromFreeTextResult(result: ToolLoopResult): EmissionDecision {
  if (result.terminalToolCalls.some((call) => call.name === NO_OUTPUT_FINALIZER_TOOL_NAME)) {
    return {
      kind: "no_output",
      reason: "legacy_no_output_tool",
    };
  }

  if (result.text.trim().length === 0) {
    return { kind: "empty" };
  }

  return {
    kind: "answer",
    text: result.text,
    source: "text",
  };
}

function emitFinalizerTrace(options: RunFinalizerOptions, decision: EmissionDecision): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("finalizer_emitted", {
    turnId: options.turnId,
    path: options.path,
    mode: options.mode ?? "free_text",
    decision: decision.kind,
    ...(decision.kind === "answer" || decision.kind === "self_report"
      ? { text_length: decision.text.length }
      : {}),
    ...(decision.kind === "no_output" ? { reason: decision.reason } : {}),
    ...(decision.kind === "self_report" ? { persistence_class: decision.persistence_class } : {}),
    ...(decision.kind === "invalid_tool"
      ? { tool_name: decision.toolName, reason: decision.reason }
      : {}),
  });
}

export async function runFinalizer(options: RunFinalizerOptions): Promise<FinalizerResult> {
  const toolProvenance =
    options.userEntryId === undefined ? undefined : { user_entry_id: options.userEntryId };
  const mode = options.mode ?? "free_text";
  const systemPrompt = buildSystemPrompt(options);
  const finalizerTools =
    mode === "emission_tools"
      ? [...EMISSION_FINALIZER_TOOLS]
      : [...options.tools, NO_OUTPUT_FINALIZER_TOOL];
  const terminalToolNames =
    mode === "emission_tools" ? EMISSION_FINALIZER_TOOL_NAMES : [NO_OUTPUT_FINALIZER_TOOL_NAME];

  const result = await executeToolLoop({
    llmClient: options.llmClient,
    dispatcher: options.dispatcher,
    sessionId: options.sessionId,
    audienceEntityId: options.audienceEntityId,
    model: options.model,
    systemPrompt,
    initialMessages: options.initialMessages,
    tools: finalizerTools,
    origin: "deliberator",
    provenance: toolProvenance,
    maxTokens: options.maxTokens,
    ...(options.thinking === undefined ? {} : { thinking: options.thinking }),
    ...(mode === "emission_tools" ? { toolChoice: { type: "any" as const } } : {}),
    budget: options.path === "system_1" ? "cognition-system-1" : "cognition-system-2",
    tracer: options.tracer,
    turnId: options.turnId,
    traceLabel: `${options.path}_finalizer`,
    terminalToolNames,
  });
  const decision =
    mode === "emission_tools"
      ? decisionFromEmissionToolResult(result)
      : decisionFromFreeTextResult(result);

  emitFinalizerTrace(options, decision);

  return {
    ...result,
    decision,
  };
}
