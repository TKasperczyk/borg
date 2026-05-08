import { z } from "zod";

import { evidenceLedgerSourceTypeSchema } from "../evidence-ledger/index.js";

export const discourseActSchema = z.enum([
  "answer",
  "clarify",
  "challenge_frame",
  "acknowledge",
  "continue_task",
  "boundary",
  "no_output",
]);

export type DiscourseAct = z.infer<typeof discourseActSchema>;

export const evidenceRefSchema = z
  .object({
    id: z.string(),
    source_type: evidenceLedgerSourceTypeSchema,
  })
  .strict();

export type EvidenceRef = z.infer<typeof evidenceRefSchema>;

const renderedSpanSchema = z.object({
  rendered_span: z.string(),
  addresses_audience_by_name: z.boolean().optional(),
});

const evidenceArraySchema = z.array(evidenceRefSchema);
const groundingEvidenceArraySchema = evidenceArraySchema.min(1);
const exactValuesSchema = z.array(z.string()).min(1);

export const discourseOnlyClaimSchema = renderedSpanSchema
  .extend({
    kind: z.literal("discourse_only"),
  })
  .strict();

export const userFactClaimSchema = renderedSpanSchema
  .extend({
    kind: z.literal("user_fact"),
    exact_values: exactValuesSchema,
    evidence: groundingEvidenceArraySchema,
    confidence: z.enum(["direct", "inferred", "uncertain"]),
    scope_disclosure_span: z.string().optional(),
  })
  .strict();

export const priorCallbackClaimSchema = renderedSpanSchema
  .extend({
    kind: z.literal("prior_callback"),
    callback_scope: z.enum(["current_turn", "current_session_prior", "prior_session"]),
    evidence: groundingEvidenceArraySchema,
    scope_disclosure_span: z.string().optional(),
  })
  .strict();

export const actionStateClaimSchema = renderedSpanSchema
  .extend({
    kind: z.literal("action_state"),
    action_record_id: z.string(),
    asserted_state: z.enum([
      "considering",
      "committed_to_do",
      "scheduled",
      "completed",
      "not_done",
      "unknown",
    ]),
    evidence: groundingEvidenceArraySchema,
  })
  .strict();

export const slotFactClaimSchema = renderedSpanSchema
  .extend({
    kind: z.literal("slot_fact"),
    slot_id: z.string(),
    exact_values: exactValuesSchema,
    evidence: groundingEvidenceArraySchema,
  })
  .strict();

export const agentSelfProvenanceClaimSchema = renderedSpanSchema
  .extend({
    kind: z.literal("agent_self_provenance"),
    evidence: groundingEvidenceArraySchema,
  })
  .strict();

export const selfReportClaimSchema = z
  .object({
    kind: z.literal("self_report"),
    rendered_span: z.string().min(1),
    addresses_audience_by_name: z.boolean().optional(),
    persistence_class: z.literal("assistant_self_report"),
  })
  .strict();

export const interpretationClaimSchema = renderedSpanSchema
  .extend({
    kind: z.literal("interpretation"),
    evidence: evidenceArraySchema,
    confidence: z.enum(["low", "medium", "high"]),
    persistence_allowed: z.literal(false),
  })
  .strict();

export const hedgeClaimSchema = renderedSpanSchema
  .extend({
    kind: z.literal("hedge"),
  })
  .strict();

export const manifestClaimSchema = z.discriminatedUnion("kind", [
  discourseOnlyClaimSchema,
  userFactClaimSchema,
  priorCallbackClaimSchema,
  actionStateClaimSchema,
  slotFactClaimSchema,
  agentSelfProvenanceClaimSchema,
  selfReportClaimSchema,
  interpretationClaimSchema,
  hedgeClaimSchema,
]);

export type ManifestClaim = z.infer<typeof manifestClaimSchema>;

export const emitManifestResponseSchema = z
  .object({
    final_text: z.string(),
    discourse_act: discourseActSchema,
    claims: z.array(manifestClaimSchema),
    no_output_reason: z.string().optional(),
  })
  .strict();

export type EmitManifestResponse = z.infer<typeof emitManifestResponseSchema>;

// Anthropic's structured-outputs API rejects schemas with an oversized compiled
// grammar ("compiled grammar is too large"). The discriminated union of 9
// strict per-kind claim shapes exceeds that limit, so the wire schema sent to
// the API is a single flat object that allows every per-kind field as
// optional. The response is then tightened back to the strict per-kind union
// locally via tightenManifestResponse before downstream code sees it.
const claimKindSchema = z.enum([
  "discourse_only",
  "user_fact",
  "prior_callback",
  "action_state",
  "slot_fact",
  "agent_self_provenance",
  "self_report",
  "interpretation",
  "hedge",
]);

const flatConfidenceSchema = z.enum([
  "direct",
  "inferred",
  "uncertain",
  "low",
  "medium",
  "high",
]);

const flatAssertedStateSchema = z.enum([
  "considering",
  "committed_to_do",
  "scheduled",
  "completed",
  "not_done",
  "unknown",
]);

const flatCallbackScopeSchema = z.enum([
  "current_turn",
  "current_session_prior",
  "prior_session",
]);

export const flatManifestClaimSchema = z.object({
  kind: claimKindSchema,
  rendered_span: z.string(),
  addresses_audience_by_name: z.boolean().optional(),
  exact_values: z.array(z.string()).optional(),
  evidence: z.array(evidenceRefSchema).optional(),
  confidence: flatConfidenceSchema.optional(),
  scope_disclosure_span: z.string().optional(),
  callback_scope: flatCallbackScopeSchema.optional(),
  action_record_id: z.string().optional(),
  asserted_state: flatAssertedStateSchema.optional(),
  slot_id: z.string().optional(),
  persistence_class: z.literal("assistant_self_report").optional(),
  persistence_allowed: z.literal(false).optional(),
});

export type FlatManifestClaim = z.infer<typeof flatManifestClaimSchema>;

export const flatEmitManifestResponseSchema = z.object({
  final_text: z.string(),
  discourse_act: discourseActSchema,
  claims: z.array(flatManifestClaimSchema),
  no_output_reason: z.string().optional(),
});

export type FlatEmitManifestResponse = z.infer<typeof flatEmitManifestResponseSchema>;

function pickStrictClaimInput(flat: FlatManifestClaim): Record<string, unknown> {
  const base: Record<string, unknown> = {
    kind: flat.kind,
    rendered_span: flat.rendered_span,
  };

  if (flat.addresses_audience_by_name !== undefined) {
    base.addresses_audience_by_name = flat.addresses_audience_by_name;
  }

  switch (flat.kind) {
    case "discourse_only":
    case "hedge":
      return base;
    case "user_fact":
      return {
        ...base,
        exact_values: flat.exact_values,
        evidence: flat.evidence,
        confidence: flat.confidence,
        ...(flat.scope_disclosure_span !== undefined
          ? { scope_disclosure_span: flat.scope_disclosure_span }
          : {}),
      };
    case "prior_callback":
      return {
        ...base,
        callback_scope: flat.callback_scope,
        evidence: flat.evidence,
        ...(flat.scope_disclosure_span !== undefined
          ? { scope_disclosure_span: flat.scope_disclosure_span }
          : {}),
      };
    case "action_state":
      return {
        ...base,
        action_record_id: flat.action_record_id,
        asserted_state: flat.asserted_state,
        evidence: flat.evidence,
      };
    case "slot_fact":
      return {
        ...base,
        slot_id: flat.slot_id,
        exact_values: flat.exact_values,
        evidence: flat.evidence,
      };
    case "agent_self_provenance":
      return {
        ...base,
        evidence: flat.evidence,
      };
    case "self_report":
      return {
        ...base,
        persistence_class: flat.persistence_class,
      };
    case "interpretation":
      return {
        ...base,
        evidence: flat.evidence,
        confidence: flat.confidence,
        persistence_allowed: flat.persistence_allowed,
      };
  }
}

export type TightenManifestResult =
  | { ok: true; manifest: EmitManifestResponse }
  | {
      ok: false;
      error: string;
      issues: unknown;
      offending_claim_index: number | null;
      offending_claim: FlatManifestClaim | null;
    };

export function tightenManifestResponse(flat: FlatEmitManifestResponse): TightenManifestResult {
  const tightenedClaims: ManifestClaim[] = [];

  for (let index = 0; index < flat.claims.length; index += 1) {
    const flatClaim = flat.claims[index];
    if (flatClaim === undefined) {
      continue;
    }

    const strictInput = pickStrictClaimInput(flatClaim);
    const parsed = manifestClaimSchema.safeParse(strictInput);

    if (!parsed.success) {
      return {
        ok: false,
        error: parsed.error.message,
        issues: parsed.error.issues,
        offending_claim_index: index,
        offending_claim: flatClaim,
      };
    }

    tightenedClaims.push(parsed.data);
  }

  const candidate: EmitManifestResponse = {
    final_text: flat.final_text,
    discourse_act: flat.discourse_act,
    claims: tightenedClaims,
    ...(flat.no_output_reason !== undefined ? { no_output_reason: flat.no_output_reason } : {}),
  };

  const parsed = emitManifestResponseSchema.safeParse(candidate);

  if (!parsed.success) {
    return {
      ok: false,
      error: parsed.error.message,
      issues: parsed.error.issues,
      offending_claim_index: null,
      offending_claim: null,
    };
  }

  return { ok: true, manifest: parsed.data };
}
