// One-call final response generation that emits prose plus a structured claim manifest.
import {
  type LLMClient,
  type LLMCompleteOptions,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMOutputConfig,
} from "../../llm/index.js";
import type { EvidenceLedger } from "../evidence-ledger/index.js";
import { toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";
import type { JsonValue } from "../../util/json-value.js";
import { LLMError } from "../../util/errors.js";
import type { DeliberationUsage } from "./types.js";
import {
  flatEmitManifestResponseSchema,
  MANIFEST_STRUCTURED_OUTPUT_FORMAT,
  tightenManifestResponse,
  type EmitManifestResponse,
  type FlatManifestClaim,
  type ManifestClaim,
} from "./manifest-schema.js";

// The wire schema uses a flat parent property bag plus allOf+if/then
// conditionals so Anthropic enforces per-kind required fields without
// blowing up the compiled grammar (see manifest-schema.ts for why).
// tightenManifestResponse runs as defense-in-depth: it strips kind-irrelevant
// fields the parent bag still allows and re-parses with the strict per-kind
// schemas before downstream code sees the manifest.
const MANIFEST_RESPONSE_OUTPUT_CONFIG = {
  format: MANIFEST_STRUCTURED_OUTPUT_FORMAT,
} satisfies LLMOutputConfig;

const MANIFEST_COVERAGE_INSTRUCTIONS = [
  "The claim manifest is the source contract for final_text.",
  "",
  "Every final_text span that asserts a user-specific fact, exact value, proper name, place, number, date, callback, action state, relational/profile slot, or claim about Borg's own prior behavior must have a matching claim.",
  "",
  "Do not hide factual or source-sensitive content under discourse_only, hedge, interpretation, or self_report.",
  "",
  "Use discourse_only only for connective tissue, acknowledgments, non-factual conversational moves, and imperatives without factual content.",
  "",
  "When final_text addresses the user vocatively by name (e.g., 'Goodnight, Tom', 'Tom, what do you think?'), set addresses_audience_by_name: true on the claim containing that span. Do NOT set this flag for topic mentions, third-person references, or quotes (e.g., 'Tom is the name we discussed', 'Tom Bombadil from Tolkien', 'You said \"Tom is here\"').",
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
  "Return exactly one structured response matching the provided schema.",
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
      rawStructuredOutput: unknown;
    }
  | {
      ok: false;
      phase: "wire" | "tighten";
      error: string;
      issues?: unknown;
      offendingClaimIndex?: number | null;
      offendingClaim?: FlatManifestClaim | null;
      rawStructuredOutput: unknown;
    };

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isStructuredOutputParseFailure(error: unknown): error is LLMError & { rawText?: string } {
  return error instanceof LLMError && error.code === "LLM_STRUCTURED_OUTPUT_PARSE_FAILED";
}

function structuredOutputParseCauseMessage(error: LLMError): string {
  return error.cause === undefined ? error.message : errorMessage(error.cause);
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

function summarizeOutputConfig(outputConfig: LLMOutputConfig): JsonValue {
  const schema = outputConfig.format.schema;
  const properties = schema.properties;

  return {
    format: outputConfig.format.type,
    propertyCount:
      properties !== null && typeof properties === "object" && !Array.isArray(properties)
        ? Object.keys(properties).length
        : 0,
    required: Array.isArray(schema.required) ? schema.required.map(String) : [],
  };
}

function summarizeResponseShape(result: LLMCompleteResult): JsonValue {
  return {
    textLength: result.text.length,
    structuredOutputPresent: result.structured_output !== undefined,
  };
}

function extractManifestResponse(structuredOutput: unknown): ManifestExtractionResult {
  const wireParse = flatEmitManifestResponseSchema.safeParse(structuredOutput);

  if (!wireParse.success) {
    return {
      ok: false,
      phase: "wire",
      error: wireParse.error.message,
      issues: wireParse.error.issues,
      rawStructuredOutput: structuredOutput,
    };
  }

  const tightened = tightenManifestResponse(wireParse.data);

  if (!tightened.ok) {
    return {
      ok: false,
      phase: "tighten",
      error: tightened.error,
      issues: tightened.issues,
      offendingClaimIndex: tightened.offending_claim_index,
      offendingClaim: tightened.offending_claim,
      rawStructuredOutput: structuredOutput,
    };
  }

  return {
    ok: true,
    manifest: tightened.manifest,
    rawStructuredOutput: structuredOutput,
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

function emitStructuredOutputParseFailedTrace(
  options: RunManifestFinalizerOptions,
  error: LLMError & { rawText?: string },
): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("manifest_finalizer_parse_failed", {
    turnId: options.turnId,
    parsed: false,
    error: structuredOutputParseCauseMessage(error),
    ...(error.rawText === undefined ? {} : { raw_text: error.rawText }),
  });
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
    phase: extraction.phase,
    error: extraction.error,
    ...(extraction.issues === undefined ? {} : { issues: toTraceJsonValue(extraction.issues) }),
    ...(extraction.offendingClaimIndex === undefined
      ? {}
      : {
          offending_claim_index:
            extraction.offendingClaimIndex === null ? null : extraction.offendingClaimIndex,
        }),
    ...(extraction.offendingClaim === undefined || extraction.offendingClaim === null
      ? {}
      : { offending_claim: toTraceJsonValue(extraction.offendingClaim) }),
    raw_structured_output: toTraceJsonValue(extraction.rawStructuredOutput),
  });
}

function emitManifestTrace(
  options: RunManifestFinalizerOptions,
  manifest: EmitManifestResponse,
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
  const traceEnabled = options.tracer?.enabled === true && options.turnId !== undefined;
  const traceLabel = `${options.path}_manifest_finalizer`;

  if (traceEnabled && options.turnId !== undefined) {
    options.tracer?.emit("llm_call_started", {
      turnId: options.turnId,
      label: traceLabel,
      model: options.model,
      promptCharCount: countCompletePromptChars(systemPrompt, options.dialogueMessages),
      outputConfig: summarizeOutputConfig(MANIFEST_RESPONSE_OUTPUT_CONFIG),
      ...(options.tracer?.includePayloads
        ? {
            prompt: toTraceJsonValue({
              system: systemPrompt,
              messages: options.dialogueMessages,
              output_config: MANIFEST_RESPONSE_OUTPUT_CONFIG,
            }),
          }
        : {}),
    });
  }

  let result: LLMCompleteResult;

  try {
    result = await options.llmClient.complete({
      model: options.model,
      system: systemPrompt,
      messages: options.dialogueMessages,
      output_config: MANIFEST_RESPONSE_OUTPUT_CONFIG,
      max_tokens: options.maxTokens,
      ...(options.thinking === undefined ? {} : { thinking: options.thinking }),
      budget: options.path === "system_1" ? "cognition-system-1" : "cognition-system-2",
    });
  } catch (error) {
    if (!isStructuredOutputParseFailure(error)) {
      throw error;
    }

    emitStructuredOutputParseFailedTrace(options, error);
    throw new LLMError("Manifest finalizer returned invalid structured output", {
      cause: structuredOutputParseCauseMessage(error),
      code: "MANIFEST_FINALIZER_OUTPUT_INVALID",
    });
  }

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
              structuredOutput: result.structured_output,
            }),
          }
        : {}),
    });
  }

  const extraction = extractManifestResponse(result.structured_output);

  if (!extraction.ok) {
    emitParseFailedTrace(options, extraction);
    throw new LLMError("Manifest finalizer returned invalid structured output", {
      cause: extraction.error,
      code: "MANIFEST_FINALIZER_OUTPUT_INVALID",
    });
  }

  emitManifestTrace(options, extraction.manifest);

  return {
    manifest: extraction.manifest,
    usage: {
      input_tokens: result.input_tokens,
      output_tokens: result.output_tokens,
      stop_reason: result.stop_reason,
    },
  };
}
