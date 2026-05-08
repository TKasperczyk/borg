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
