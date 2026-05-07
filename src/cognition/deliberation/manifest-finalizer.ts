// One-call final response generation that emits prose plus a structured claim manifest.
import {
  type LLMClient,
  type LLMCompleteOptions,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolCall,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { EvidenceLedger } from "../evidence-ledger/index.js";
import { toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";
import type { JsonValue } from "../../util/json-value.js";
import { LLMError } from "../../util/errors.js";
import type { DeliberationUsage } from "./types.js";
import {
  emitManifestResponseSchema,
  type EmitManifestResponse,
  type ManifestClaim,
} from "./manifest-schema.js";

export const EMIT_MANIFEST_RESPONSE_TOOL_NAME = "EmitManifestResponse";

const EMIT_MANIFEST_RESPONSE_TOOL: LLMToolDefinition = {
  name: EMIT_MANIFEST_RESPONSE_TOOL_NAME,
  description:
    "Emit the final assistant text and the structured claim manifest for that exact text.",
  inputSchema: toToolInputSchema(emitManifestResponseSchema),
};

const MANIFEST_COVERAGE_INSTRUCTIONS = [
  "The claim manifest is the source contract for final_text.",
  "",
  "Every final_text span that asserts a user-specific fact, exact value, proper name, place, number, date, callback, action state, relational/profile slot, or claim about Borg's own prior behavior must have a matching claim.",
  "",
  "Do not hide factual or source-sensitive content under discourse_only, hedge, interpretation, or self_report.",
  "",
  "Use discourse_only only for connective tissue, acknowledgments, non-factual conversational moves, and imperatives without factual content.",
  "",
  "Use self_report for first-person expression of interior states, identity reflection, voice, or boundary -- the model's own perspective. Self-reports are accepted as expression and persisted with persistence_class: assistant_self_report. They are not factual evidence about the external world.",
  "",
  "Use user_fact for exact user-specific details: names, places, dates, numbers, itinerary items, counts, durations, concrete preferences, and other sourced specifics.",
  "",
  'Use prior_callback for phrases like "you said", "you mentioned", "earlier", "last time", or "as you put it".',
  "",
  "Use action_state for claims that something is considering, committed_to_do, scheduled, completed, not_done, or unknown.",
  "",
  "Use slot_fact for established relational/profile slot values.",
  "",
  "Use agent_self_provenance only for claims about Borg's own prior behavior, authorship, role, system state, or conversation-frame history.",
  "",
  "If an exact value cannot be cited from the EvidenceLedger, remove it from final_text or phrase it qualitatively.",
  "",
  "When the entity is referenced by pronoun (she/he/they/it) or descriptive noun phrase (the tutor, your partner) in the supporting evidence, cite BOTH (1) the evidence that establishes the entity's name, AND (2) the evidence that contains the pronoun/descriptive reference and the predicate. Do not cite only the pronoun-bearing evidence for a named claim.",
  "",
  "A manifest with too few claims is invalid even if the prose sounds good.",
].join("\n");

const MANIFEST_FINALIZER_INSTRUCTIONS = [
  `You must call the ${EMIT_MANIFEST_RESPONSE_TOOL_NAME} tool exactly once. Do not answer in free text outside the tool call.`,
  "Put the complete assistant response in final_text.",
  "Set discourse_act to no_output only when the correct current-turn behavior is to emit no assistant message at all. When discourse_act is no_output, populate no_output_reason.",
  "For every claim evidence ref, cite EvidenceLedger entry IDs verbatim exactly as they appear in id=... metadata inside <borg_evidence_ledger>.",
  "For every claim evidence ref, set source_type to the cited ledger entry's source_type value.",
  "Do not invent evidence IDs. If a rendered span has no evidence requirement, use a claim kind that does not require evidence.",
  "",
  MANIFEST_COVERAGE_INSTRUCTIONS,
].join("\n");

export type RunManifestFinalizerOptions = {
  llmClient: LLMClient;
  model: string;
  baseSystemPrompt: string;
  dialogueMessages: readonly LLMMessage[];
  evidenceLedger: EvidenceLedger | null;
  maxTokens: number;
  thinking?: LLMCompleteOptions["thinking"];
  path: "system_1" | "system_2";
  additionalPromptSections?: readonly (string | null)[];
  tracer?: TurnTracer;
  turnId?: string;
};

export type ManifestFinalizerResult = {
  manifest: EmitManifestResponse;
  usage: DeliberationUsage;
};

type ManifestExtractionResult =
  | {
      ok: true;
      manifest: EmitManifestResponse;
      rawToolInput: unknown;
      unwrapped: boolean;
    }
  | {
      ok: false;
      error: string;
      issues?: unknown;
      rawToolInput?: unknown;
      rawToolCalls: readonly LLMToolCall[];
    };

function wrappedManifestInput(input: unknown): unknown | null {
  if (input === null || typeof input !== "object" || Array.isArray(input)) {
    return null;
  }

  const record = input as Record<string, unknown>;
  const keys = Object.keys(record);

  if (keys.length !== 1 || (keys[0] !== "input" && keys[0] !== "arguments")) {
    return null;
  }

  const value = record[keys[0] as "input" | "arguments"];

  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  return value;
}

function buildManifestSystemPrompt(options: RunManifestFinalizerOptions): string {
  const sections =
    options.additionalPromptSections === undefined
      ? [options.baseSystemPrompt, MANIFEST_FINALIZER_INSTRUCTIONS]
      : [
          options.baseSystemPrompt,
          ...options.additionalPromptSections.filter(
            (section): section is string => section !== null,
          ),
          MANIFEST_FINALIZER_INSTRUCTIONS,
        ];

  return sections.join("\n\n");
}

function countCompletePromptChars(systemPrompt: string, messages: readonly LLMMessage[]): number {
  return (
    systemPrompt.length +
    messages.reduce((sum, message) => sum + message.role.length + message.content.length, 0)
  );
}

function summarizeToolSchemas(tools: readonly LLMToolDefinition[]): JsonValue {
  return tools.map((tool) => ({
    name: tool.name,
    propertyCount:
      tool.inputSchema.properties === undefined
        ? 0
        : Object.keys(tool.inputSchema.properties).length,
    required: Array.isArray(tool.inputSchema.required) ? tool.inputSchema.required.map(String) : [],
  }));
}

function summarizeResponseShape(result: LLMCompleteResult): JsonValue {
  return {
    textLength: result.text.length,
    toolUseBlocks: result.tool_calls.map((call) => ({
      id: call.id,
      name: call.name,
    })),
  };
}

function extractManifestResponse(toolCalls: readonly LLMToolCall[]): ManifestExtractionResult {
  if (toolCalls.length !== 1) {
    return {
      ok: false,
      error: `Manifest finalizer must emit exactly one ${EMIT_MANIFEST_RESPONSE_TOOL_NAME} tool call; received ${toolCalls.length}`,
      rawToolCalls: toolCalls,
    };
  }

  const call = toolCalls[0] as LLMToolCall;

  if (call.name !== EMIT_MANIFEST_RESPONSE_TOOL_NAME) {
    return {
      ok: false,
      error: `Manifest finalizer emitted unexpected tool ${call.name}; expected ${EMIT_MANIFEST_RESPONSE_TOOL_NAME}`,
      rawToolInput: call.input,
      rawToolCalls: toolCalls,
    };
  }

  const wrapperInput = wrappedManifestInput(call.input);

  if (wrapperInput !== null) {
    const unwrapped = emitManifestResponseSchema.safeParse(wrapperInput);

    if (unwrapped.success) {
      return {
        ok: true,
        manifest: unwrapped.data,
        rawToolInput: call.input,
        unwrapped: true,
      };
    }
  }

  const parsed = emitManifestResponseSchema.safeParse(call.input);

  if (!parsed.success) {
    return {
      ok: false,
      error: parsed.error.message,
      issues: parsed.error.issues,
      rawToolInput: call.input,
      rawToolCalls: toolCalls,
    };
  }

  return {
    ok: true,
    manifest: parsed.data,
    rawToolInput: call.input,
    unwrapped: false,
  };
}

function summarizeClaimKinds(claims: readonly ManifestClaim[]): string[] {
  const counts = new Map<string, number>();

  for (const claim of claims) {
    counts.set(claim.kind, (counts.get(claim.kind) ?? 0) + 1);
  }

  return [...counts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([kind, count]) => `${kind}:${count}`);
}

function claimCountsByKind(claims: readonly ManifestClaim[]): JsonValue {
  const counts: Record<string, number> = {};

  for (const claim of claims) {
    counts[claim.kind] = (counts[claim.kind] ?? 0) + 1;
  }

  return counts;
}

function emitParseFailedTrace(
  options: RunManifestFinalizerOptions,
  extraction: ManifestExtractionResult,
): void {
  if (extraction.ok || options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("manifest_finalizer_parse_failed", {
    turnId: options.turnId,
    parsed: false,
    error: extraction.error,
    ...(extraction.issues === undefined ? {} : { issues: toTraceJsonValue(extraction.issues) }),
    ...(extraction.rawToolInput === undefined
      ? {}
      : { raw_tool_input: toTraceJsonValue(extraction.rawToolInput) }),
    raw_tool_calls: toTraceJsonValue(extraction.rawToolCalls),
  });
}

function emitManifestTrace(
  options: RunManifestFinalizerOptions,
  manifest: EmitManifestResponse,
  unwrapped: boolean,
): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("manifest_finalizer_emitted", {
    turnId: options.turnId,
    final_text_length: manifest.final_text.length,
    discourse_act: manifest.discourse_act,
    claim_count: manifest.claims.length,
    claim_kinds: summarizeClaimKinds(manifest.claims),
    claim_counts_by_kind: claimCountsByKind(manifest.claims),
    parsed: true,
    ...(unwrapped ? { manifest_finalizer_unwrapped: true } : {}),
    ...(manifest.no_output_reason === undefined
      ? {}
      : { no_output_reason: manifest.no_output_reason }),
    ...(options.tracer.includePayloads
      ? {
          claims: toTraceJsonValue(manifest.claims),
        }
      : {}),
  });
}

export async function runManifestFinalizer(
  options: RunManifestFinalizerOptions,
): Promise<ManifestFinalizerResult> {
  if (options.evidenceLedger === null) {
    throw new LLMError("Manifest finalizer requires an EvidenceLedger", {
      code: "MANIFEST_FINALIZER_LEDGER_MISSING",
    });
  }

  const systemPrompt = buildManifestSystemPrompt(options);
  const tools = [EMIT_MANIFEST_RESPONSE_TOOL];
  const traceEnabled = options.tracer?.enabled === true && options.turnId !== undefined;
  const traceLabel = `${options.path}_manifest_finalizer`;

  if (traceEnabled && options.turnId !== undefined) {
    options.tracer?.emit("llm_call_started", {
      turnId: options.turnId,
      label: traceLabel,
      model: options.model,
      promptCharCount: countCompletePromptChars(systemPrompt, options.dialogueMessages),
      toolSchemas: summarizeToolSchemas(tools),
      ...(options.tracer?.includePayloads
        ? {
            prompt: toTraceJsonValue({
              system: systemPrompt,
              messages: options.dialogueMessages,
              tools,
            }),
          }
        : {}),
    });
  }

  const result = await options.llmClient.complete({
    model: options.model,
    system: systemPrompt,
    messages: options.dialogueMessages,
    tools,
    tool_choice: { type: "tool", name: EMIT_MANIFEST_RESPONSE_TOOL_NAME },
    max_tokens: options.maxTokens,
    ...(options.thinking === undefined ? {} : { thinking: options.thinking }),
    budget: options.path === "system_1" ? "cognition-system-1" : "cognition-system-2",
  });

  if (traceEnabled && options.turnId !== undefined) {
    options.tracer?.emit("llm_call_response", {
      turnId: options.turnId,
      label: traceLabel,
      responseShape: summarizeResponseShape(result),
      stopReason: result.stop_reason,
      usage: {
        inputTokens: result.input_tokens,
        outputTokens: result.output_tokens,
      },
      ...(options.tracer?.includePayloads
        ? {
            response: toTraceJsonValue({
              text: result.text,
              toolCalls: result.tool_calls,
            }),
          }
        : {}),
    });
  }

  const extraction = extractManifestResponse(result.tool_calls);

  if (!extraction.ok) {
    emitParseFailedTrace(options, extraction);
    throw new LLMError("Manifest finalizer returned invalid tool output", {
      cause: extraction.error,
      code: "MANIFEST_FINALIZER_OUTPUT_INVALID",
    });
  }

  emitManifestTrace(options, extraction.manifest, extraction.unwrapped);

  return {
    manifest: extraction.manifest,
    usage: {
      input_tokens: result.input_tokens,
      output_tokens: result.output_tokens,
      stop_reason: result.stop_reason,
    },
  };
}
