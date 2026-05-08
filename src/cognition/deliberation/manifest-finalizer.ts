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
  type ManifestClaimDemotion,
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
  "Use user_fact ONLY when ALL of the following are true: (1) you can cite at least one ledger entry as evidence, AND (2) you can enumerate the specific values you are claiming as a list of literal strings in exact_values, AND (3) you can pick a confidence in {direct, inferred, uncertain}. If any of those is missing, use discourse_only or hedge instead. Do NOT emit user_fact with missing exact_values; the rendered_span is not a substitute -- exact_values is a separate array of the literal extracted items.",
  "",
  "Example user_fact claim:",
  "  rendered_span: \"flying to Berlin on June 12 with two carry-ons\"",
  "  exact_values: [\"Berlin\", \"June 12\", \"two carry-ons\"]",
  "  evidence: [{id: ..., source_type: ...}]",
  "  confidence: \"direct\"",
  "Negative example -- when you cannot enumerate exact_values, do NOT use user_fact: a span like \"your tutor cancels sometimes\" has no concrete extracted values to list, so emit it as discourse_only or hedge.",
  "",
  'Use prior_callback ONLY when ALL of: (1) you can cite at least one ledger entry as evidence for the prior content, AND (2) the cited evidence\'s source_type matches the callback_scope you select.',
  "",
  "callback_scope must match the cited evidence:",
  "  - callback_scope: \"current_turn\" REQUIRES evidence with source_type: \"current_user_message\"",
  "  - callback_scope: \"current_session_prior\" REQUIRES evidence with source_type: \"current_session_stream\" (or \"assistant_stream\" within the same session)",
  "  - callback_scope: \"prior_session\" REQUIRES evidence with source_type: \"prior_session_stream\" (and you must include scope_disclosure_span identifying the prior-session reference)",
  "If the scope and the cited evidence do not match, do NOT emit prior_callback -- pick the scope that matches the evidence you actually have, or use discourse_only if you cannot ground a specific prior reference.",
  "",
  "Example prior_callback (current_session_prior):",
  "  rendered_span: \"You mentioned earlier you wanted to revisit the Atlas migration\"",
  "  callback_scope: \"current_session_prior\"",
  "  evidence: [{id: \"current_session_transcript:strm_xxxxxxxxxxxxxxxx\", source_type: \"current_session_stream\"}]",
  "Example prior_callback (prior_session, with disclosure span):",
  "  rendered_span: \"In our last session you mentioned the migration\"",
  "  callback_scope: \"prior_session\"",
  "  scope_disclosure_span: \"In our last session\"",
  "  evidence: [{id: \"prior_session_memory:strm_xxxxxxxxxxxxxxxx\", source_type: \"prior_session_stream\"}]",
  "Negative example: do NOT use callback_scope: \"prior_session\" if all your evidence is from the current session -- the scope and the evidence must agree.",
  "",
  "Use action_state ONLY when ALL of: (1) you have an action_record_id from the action_records ledger section, AND (2) you can cite the entry that establishes the record's state, AND (3) you can pick an asserted_state. If you don't have an action_record_id, do NOT use action_state -- a generic 'I'll get back to you' is discourse_only, not action_state.",
  "",
  "Example action_state:",
  "  rendered_span: \"I have completed the Atlas deployment check\"",
  "  action_record_id: \"act_xxxxxxxxxxxxxxxx\"",
  "  asserted_state: \"completed\"",
  "  evidence: [{id: ..., source_type: ...}]",
  "",
  "Use slot_fact ONLY when ALL of: (1) you have a slot_id from the relational_slots ledger section, AND (2) at least one ledger entry as evidence, AND (3) at least one literal value in exact_values for the slot's content. If you don't have a slot_id, do NOT use slot_fact -- consider user_fact (if it grounds fully) or discourse_only.",
  "",
  "Example slot_fact:",
  "  rendered_span: \"your partner Marta\"",
  "  slot_id: \"slot_xxxxxxxxxxxxxxxx\"",
  "  exact_values: [\"Marta\"]",
  "  evidence: [{id: ..., source_type: ...}]",
  "",
  "Use agent_self_provenance only for claims about Borg's own prior behavior, authorship, role, system state, or conversation-frame history. If you are saying you have NO record / NO retrieval / NO memory of something, that is NOT agent_self_provenance -- use discourse_only or self_report. agent_self_provenance always cites at least one evidence entry.",
  "",
  "If an exact value cannot be cited from the EvidenceLedger, remove it from final_text or phrase it qualitatively.",
  "",
  "When the entity is referenced by pronoun (she/he/they/it) or descriptive noun phrase (the tutor, your partner) in the supporting evidence, cite BOTH (1) the evidence that establishes the entity's name, AND (2) the evidence that contains the pronoun/descriptive reference and the predicate. Do not cite only the pronoun-bearing evidence for a named claim.",
  "",
  "A manifest with too few claims is invalid even if the prose sounds good.",
  "",
  "Closure-beat preemption: if the discourse-state section of this prompt declares a HARD CONSTRAINT - CLOSURE PRESSURE, treat that as binding on final_text. Do not append a sign-off, valediction, weather observation, single-line 'noted/held' acknowledgment, or any sentence that reads as a coda. End on substantive content or set discourse_act to no_output. This rule overrides any natural-conversation tendency to wind down a turn.",
  "",
  "Required fields per kind. If you cannot provide every required field for a kind, use a permissive kind (discourse_only or hedge) instead. Do NOT emit a kind with missing required fields.",
  "  - discourse_only: kind, rendered_span (no other fields required)",
  "  - hedge: kind, rendered_span (no other fields required)",
  "  - self_report: kind, rendered_span, persistence_class: \"assistant_self_report\"",
  "  - user_fact: kind, rendered_span, evidence (>=1 ledger entry), exact_values (>=1 string), confidence in {direct, inferred, uncertain}",
  "  - prior_callback: kind, rendered_span, evidence (>=1 ledger entry), callback_scope in {current_turn, current_session_prior, prior_session}",
  "  - action_state: kind, rendered_span, evidence (>=1 ledger entry), action_record_id, asserted_state in {considering, committed_to_do, scheduled, completed, not_done, unknown}",
  "  - slot_fact: kind, rendered_span, evidence (>=1 ledger entry), exact_values (>=1 string), slot_id",
  "  - agent_self_provenance: kind, rendered_span, evidence (>=1 ledger entry)",
  "  - interpretation: kind, rendered_span, evidence (any length), confidence in {low, medium, high}, persistence_allowed: false",
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
      demotions: readonly ManifestClaimDemotion[];
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
    demotions: tightened.demotions,
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
  demotions: readonly ManifestClaimDemotion[],
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
    demotion_count: demotions.length,
    ...(demotions.length === 0
      ? {}
      : {
          demotions: toTraceJsonValue(
            demotions.map((demotion) => ({
              index: demotion.index,
              original_kind: demotion.original_kind,
              reason: demotion.reason,
              ...(options.tracer?.includePayloads
                ? { issues: toTraceJsonValue(demotion.issues) }
                : {}),
            })),
          ),
        }),
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

  emitManifestTrace(options, extraction.manifest, extraction.demotions);

  return {
    manifest: extraction.manifest,
    usage: {
      input_tokens: result.input_tokens,
      output_tokens: result.output_tokens,
      stop_reason: result.stop_reason,
    },
  };
}
