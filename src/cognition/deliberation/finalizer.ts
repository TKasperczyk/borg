// Routes S1/S2 final response generation through the deliberator tool loop.
import { z } from "zod";

import type { LLMClient, LLMContentBlockMessage, LLMConverseOptions } from "../../llm/index.js";
import type { ToolDefinition, ToolDispatcher } from "../../tools/index.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { executeToolLoop, type ToolLoopResult } from "../action/index.js";
import { replyTargetSchema, type ReplyTarget } from "../generation/types.js";

export const EMIT_ANSWER_FINALIZER_TOOL_NAME = "EmitAnswer";
export const EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME = "EmitNoOutput";
export const EMIT_SELF_REPORT_FINALIZER_TOOL_NAME = "EmitSelfReport";

const emitTextToolInputSchema = z
  .object({
    text: z.string(),
    reply_target: replyTargetSchema.optional(),
  })
  .strict();

const emitNoOutputToolInputSchema = z
  .object({
    reason: z.string().min(1),
  })
  .strict();

const emitSelfReportToolInputSchema = z
  .object({
    kind: z.literal("self_report"),
    text: z.string(),
    persistence_class: z.literal("assistant_self_report"),
  })
  .strict();

const EMIT_ANSWER_FINALIZER_TOOL: ToolDefinition = {
  name: EMIT_ANSWER_FINALIZER_TOOL_NAME,
  description:
    "Emit the assistant response for this turn. Put the complete user-visible response in text. Use this for ordinary answers, questions, acknowledgments, challenges, and continuations. When the response is primarily addressed to one named participant, also set reply_target to kind=entity with their prompt-visible entity_id; default kind=audience (or omit) when speaking to the whole channel or multiple participants.",
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
    "Emit a first-person interior self-report from Borg's perspective. Set kind to self_report, persistence_class to assistant_self_report, and put the complete user-visible response in text. The text is shown to the user like EmitAnswer.",
  allowedOrigins: ["deliberator"],
  writeScope: "read",
  inputSchema: emitSelfReportToolInputSchema,
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
  "Use EmitAnswer for an ordinary assistant response. Put the complete user-visible response in text. Use reply_target.kind=entity with a prompt-visible entity_id when the response is primarily addressed to a single named participant -- including when answering a question from a specific speaker, when addressing one person by name, or when a participant has asked to be addressed directly. Use reply_target.kind=audience (or omit) when the response speaks to the channel as a whole. For example, if Alice asks Borg a question and Borg's response begins 'Alice -- ...', reply_target.kind should be entity with Alice's entity_id.",
  "Use EmitNoOutput only when the correct current-turn behavior is to emit no assistant message at all. Put a concise reason in reason.",
  "Use EmitSelfReport for first-person expression of Borg's interior state, identity reflection, voice, or boundary. EmitSelfReport must include kind=self_report, persistence_class=assistant_self_report, and text. It is shown to the user exactly like EmitAnswer and persisted as assistant_self_report.",
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
  userEntryId: string | undefined;
  maxTokens: number;
  thinking?: LLMConverseOptions["thinking"];
  path: "system_1" | "system_2";
  additionalPromptSections?: readonly (string | null)[];
  cacheableSystemPrompt?: CacheableFinalizerSystemPrompt;
  tracer?: TurnTracer;
  turnId?: string;
};

export type EmissionDecision =
  | {
      kind: "answer";
      text: string;
      source: "tool" | "text";
      reply_target?: ReplyTarget;
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
    options.cacheableSystemPrompt?.dynamicContent ?? options.baseSystemPrompt;

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
      : {
          kind: "answer",
          text: parsed.data.text,
          source: "tool",
          ...(parsed.data.reply_target === undefined
            ? {}
            : { reply_target: parsed.data.reply_target }),
        };
  }

  if (terminalCall.name === EMIT_SELF_REPORT_FINALIZER_TOOL_NAME) {
    const parsed = emitSelfReportToolInputSchema.safeParse(terminalCall.input);

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

function emitFinalizerTrace(options: RunFinalizerOptions, decision: EmissionDecision): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("finalizer_emitted", {
    turnId: options.turnId,
    path: options.path,
    mode: "emission_tools",
    decision: decision.kind,
    ...(decision.kind === "answer" || decision.kind === "self_report"
      ? { text_length: decision.text.length }
      : {}),
    ...(decision.kind === "answer" && decision.reply_target !== undefined
      ? { reply_target: decision.reply_target }
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
  const systemPrompt = buildSystemPrompt(options);

  const result = await executeToolLoop({
    llmClient: options.llmClient,
    dispatcher: options.dispatcher,
    sessionId: options.sessionId,
    audienceEntityId: options.audienceEntityId,
    model: options.model,
    systemPrompt,
    initialMessages: options.initialMessages,
    tools: [...EMISSION_FINALIZER_TOOLS],
    origin: "deliberator",
    provenance: toolProvenance,
    maxTokens: options.maxTokens,
    ...(options.thinking === undefined ? {} : { thinking: options.thinking }),
    toolChoice: { type: "any" as const },
    budget: options.path === "system_1" ? "cognition-system-1" : "cognition-system-2",
    tracer: options.tracer,
    turnId: options.turnId,
    traceLabel: `${options.path}_finalizer`,
    terminalToolNames: EMISSION_FINALIZER_TOOL_NAMES,
  });
  const decision = decisionFromEmissionToolResult(result);

  emitFinalizerTrace(options, decision);

  return {
    ...result,
    decision,
  };
}
