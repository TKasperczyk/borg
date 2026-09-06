import { z } from "zod";

import { memoryDisclosureLabelSchema } from "../common/disclosure-label.js";
import {
  entityIdHelpers,
  autobiographicalPeriodIdHelpers,
  sharedStateEntryIdHelpers,
  episodeIdHelpers,
  goalIdHelpers,
  openQuestionIdHelpers,
  streamEntryIdHelpers,
  traitIdHelpers,
  valueIdHelpers,
  type AutobiographicalPeriodId,
  type EntityId,
  type SharedStateEntryId,
  type EpisodeId,
  type GoalId,
  type OpenQuestionId,
  type StreamEntryId,
  type TraitId,
  type ValueId,
} from "../../util/ids.js";
import { provenanceSchema, type Provenance } from "../common/provenance.js";
import { episodeIdSchema } from "../episodic/types.js";
import { semanticNodeIdSchema } from "../semantic/types.js";
import { goalBlockRecordSchema } from "./goal-blocks.js";

export const goalStatusSchema = z.enum(["active", "done", "abandoned", "blocked"]);
export const identityStateSchema = z.enum(["candidate", "established"]);
export const OPEN_QUESTION_STATUSES = ["open", "resolved", "abandoned"] as const;
export const OPEN_QUESTION_SOURCES = [
  "user",
  "reflection",
  "contradiction",
  "ruminator",
  "overseer",
  "associator",
  "autonomy",
  "deliberator",
] as const;

export const valueIdSchema = z
  .string()
  .refine((value) => valueIdHelpers.is(value), {
    message: "Invalid value id",
  })
  .transform((value) => value as ValueId);

export const goalIdSchema = z
  .string()
  .refine((value) => goalIdHelpers.is(value), {
    message: "Invalid goal id",
  })
  .transform((value) => value as GoalId);

export const openQuestionIdSchema = z
  .string()
  .refine((value) => openQuestionIdHelpers.is(value), {
    message: "Invalid open question id",
  })
  .transform((value) => value as OpenQuestionId);

export const goalAudienceEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid goal audience entity id",
  })
  .transform((value) => value as EntityId);

export const goalOwnerEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid goal owner entity id",
  })
  .transform((value) => value as EntityId);

export const goalCounterpartyEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid goal counterparty entity id",
  })
  .transform((value) => value as EntityId);

export const goalSourceStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid goal source stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const goalCanonicalizedByArtifactEntryIdSchema = z
  .string()
  .refine((value) => sharedStateEntryIdHelpers.is(value), {
    message: "Invalid goal canonicalized shared state entry id",
  })
  .transform((value) => value as SharedStateEntryId);

export const openQuestionAudienceEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid open question audience entity id",
  })
  .transform((value) => value as EntityId);

export const openQuestionStatusSchema = z.enum(OPEN_QUESTION_STATUSES);
export const openQuestionSourceSchema = z.enum(OPEN_QUESTION_SOURCES);

export const openQuestionResolutionStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid open question resolution stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const openQuestionResolvedByArtifactEntryIdSchema = z
  .string()
  .refine((value) => sharedStateEntryIdHelpers.is(value), {
    message: "Invalid open question resolved shared state entry id",
  })
  .transform((value) => value as SharedStateEntryId);

export const traitIdSchema = z
  .string()
  .refine((value) => traitIdHelpers.is(value), {
    message: "Invalid trait id",
  })
  .transform((value) => value as TraitId);

export const valueSourceEpisodeIdSchema = z
  .string()
  .refine((value) => episodeIdHelpers.is(value), {
    message: "Invalid episode id",
  })
  .transform((value) => value as EpisodeId);

export const autobiographicalPeriodIdSchema = z
  .string()
  .refine((value) => autobiographicalPeriodIdHelpers.is(value), {
    message: "Invalid autobiographical period id",
  })
  .transform((value) => value as AutobiographicalPeriodId);

export const autobiographicalPeriodSchema = z
  .object({
    id: autobiographicalPeriodIdSchema,
    record_version: z.number().int().positive().optional(),
    label: z.string().min(1),
    start_ts: z.number().finite(),
    end_ts: z.number().finite().nullable(),
    narrative: z.string(),
    key_episode_ids: z.array(valueSourceEpisodeIdSchema),
    disclosure_label: memoryDisclosureLabelSchema.optional(),
    themes: z.array(z.string().min(1)),
    provenance: provenanceSchema,
    created_at: z.number().finite(),
    last_updated: z.number().finite(),
  })
  .refine((value) => value.end_ts === null || value.end_ts >= value.start_ts, {
    message: "Autobiographical period end_ts must be after start_ts",
    path: ["end_ts"],
  });

export const autobiographicalPeriodPatchSchema = z
  .object({
    label: z.string().min(1).optional(),
    start_ts: z.number().finite().optional(),
    end_ts: z.number().finite().nullable().optional(),
    narrative: z.string().optional(),
    key_episode_ids: z.array(valueSourceEpisodeIdSchema).optional(),
    disclosure_label: memoryDisclosureLabelSchema.optional(),
    themes: z.array(z.string().min(1)).optional(),
    provenance: provenanceSchema.optional(),
  })
  .strict();

export const valueSchema = z.object({
  id: valueIdSchema,
  record_version: z.number().int().positive().optional(),
  label: z.string().min(1),
  description: z.string().min(1),
  priority: z.number().finite(),
  created_at: z.number().finite(),
  last_affirmed: z.number().finite().nullable(),
  state: identityStateSchema,
  established_at: z.number().finite().nullable(),
  confidence: z.number().min(0).max(1),
  last_tested_at: z.number().finite().nullable(),
  last_contradicted_at: z.number().finite().nullable(),
  support_count: z.number().int().min(0),
  contradiction_count: z.number().int().min(0),
  evidence_episode_ids: z.array(valueSourceEpisodeIdSchema).max(3),
  provenance: provenanceSchema,
});

export const valuePatchSchema = valueSchema
  .omit({
    id: true,
    record_version: true,
    created_at: true,
    confidence: true,
    last_tested_at: true,
    last_contradicted_at: true,
    support_count: true,
    contradiction_count: true,
    evidence_episode_ids: true,
  })
  .partial()
  .strict();

export const goalSchema = z.object({
  id: goalIdSchema,
  record_version: z.number().int().positive().optional(),
  description: z.string().min(1),
  terminal_condition: z.string().min(1).nullable().default(null),
  priority: z.number().finite(),
  parent_goal_id: goalIdSchema.nullable(),
  status: goalStatusSchema,
  block_history: z.array(goalBlockRecordSchema).optional(),
  progress_notes: z.string().nullable(),
  last_progress_ts: z.number().finite().nullable(),
  created_at: z.number().finite(),
  target_at: z.number().finite().nullable(),
  audience_entity_id: goalAudienceEntityIdSchema.nullable().default(null),
  owner_entity_id: goalOwnerEntityIdSchema.nullable().optional(),
  counterparty_entity_id: goalCounterpartyEntityIdSchema.nullable().optional(),
  source_stream_entry_ids: z.array(goalSourceStreamEntryIdSchema).min(1).optional(),
  canonicalized_by_artifact_entry_id: goalCanonicalizedByArtifactEntryIdSchema
    .nullable()
    .optional(),
  provenance: provenanceSchema,
});

export const traitSchema = z.object({
  id: traitIdSchema,
  record_version: z.number().int().positive().optional(),
  label: z.string().min(1),
  strength: z.number().min(0).max(1),
  last_reinforced: z.number().finite(),
  last_decayed: z.number().finite().nullable(),
  state: identityStateSchema,
  established_at: z.number().finite().nullable(),
  confidence: z.number().min(0).max(1),
  last_tested_at: z.number().finite().nullable(),
  last_contradicted_at: z.number().finite().nullable(),
  support_count: z.number().int().min(0),
  contradiction_count: z.number().int().min(0),
  evidence_episode_ids: z.array(valueSourceEpisodeIdSchema).max(3),
  provenance: provenanceSchema,
});

export const traitPatchSchema = traitSchema
  .omit({
    id: true,
    record_version: true,
    confidence: true,
    last_tested_at: true,
    last_contradicted_at: true,
    support_count: true,
    contradiction_count: true,
    evidence_episode_ids: true,
  })
  .partial()
  .strict();

export const goalPatchSchema = goalSchema
  .omit({
    id: true,
    record_version: true,
    created_at: true,
    block_history: true,
    audience_entity_id: true,
    owner_entity_id: true,
    counterparty_entity_id: true,
    source_stream_entry_ids: true,
  })
  .extend({
    terminal_condition: z.string().min(1).nullable().optional(),
    audience_entity_id: goalAudienceEntityIdSchema.nullable().optional(),
    owner_entity_id: goalOwnerEntityIdSchema.nullable().optional(),
    counterparty_entity_id: goalCounterpartyEntityIdSchema.nullable().optional(),
    source_stream_entry_ids: z.array(goalSourceStreamEntryIdSchema).min(1).optional(),
  })
  .partial()
  .strict();

export const openQuestionSchema = z.object({
  id: openQuestionIdSchema,
  record_version: z.number().int().positive().optional(),
  question: z.string().min(1),
  urgency: z.number().min(0).max(1),
  status: openQuestionStatusSchema,
  goal_id: goalIdSchema.nullable().default(null),
  audience_entity_id: openQuestionAudienceEntityIdSchema.nullable().default(null),
  related_episode_ids: z.array(episodeIdSchema),
  related_semantic_node_ids: z.array(semanticNodeIdSchema),
  disclosure_label: memoryDisclosureLabelSchema.optional(),
  provenance: provenanceSchema.nullable(),
  source: openQuestionSourceSchema,
  created_at: z.number().finite(),
  last_touched: z.number().finite(),
  resolution_evidence_episode_ids: z.array(episodeIdSchema),
  resolution_evidence_stream_entry_ids: z.array(openQuestionResolutionStreamEntryIdSchema),
  resolution_disclosure_label: memoryDisclosureLabelSchema.optional(),
  resolution_note: z.string().nullable(),
  resolved_at: z.number().finite().nullable(),
  abandoned_reason: z.string().nullable(),
  abandoned_at: z.number().finite().nullable(),
  resolved_by_artifact_entry_id: openQuestionResolvedByArtifactEntryIdSchema.nullable().optional(),
  unresolved_rumination_ticks: z.number().int().nonnegative().default(0),
  last_ruminated_at: z.number().finite().nullable().default(null),
});

export const openQuestionRecordSchema = openQuestionSchema
  .refine(
    (value) =>
      value.related_episode_ids.length > 0 ||
      value.related_semantic_node_ids.length > 0 ||
      value.provenance !== null,
    {
      message:
        "Open question requires related_episode_ids, related_semantic_node_ids, or explicit provenance",
      path: ["provenance"],
    },
  )
  .refine(
    (value) =>
      value.status !== "resolved" ||
      value.resolution_evidence_episode_ids.length > 0 ||
      value.resolution_evidence_stream_entry_ids.length > 0,
    {
      message: "Resolved open question requires episode or stream evidence",
      path: ["resolution_evidence_episode_ids"],
    },
  );

export const openQuestionPatchSchema = z.object({
  question: z.string().min(1).optional(),
  urgency: z.number().min(0).max(1).optional(),
  status: openQuestionStatusSchema.optional(),
  goal_id: goalIdSchema.nullable().optional(),
  audience_entity_id: openQuestionAudienceEntityIdSchema.nullable().optional(),
  related_episode_ids: z.array(episodeIdSchema).optional(),
  related_semantic_node_ids: z.array(semanticNodeIdSchema).optional(),
  disclosure_label: memoryDisclosureLabelSchema.optional(),
  provenance: provenanceSchema.nullable().optional(),
  source: openQuestionSourceSchema.optional(),
  last_touched: z.number().finite().optional(),
  resolution_evidence_episode_ids: z.array(episodeIdSchema).optional(),
  resolution_evidence_stream_entry_ids: z
    .array(openQuestionResolutionStreamEntryIdSchema)
    .optional(),
  resolution_disclosure_label: memoryDisclosureLabelSchema.optional(),
  resolution_note: z.string().nullable().optional(),
  resolved_at: z.number().finite().nullable().optional(),
  abandoned_reason: z.string().nullable().optional(),
  abandoned_at: z.number().finite().nullable().optional(),
  resolved_by_artifact_entry_id: openQuestionResolvedByArtifactEntryIdSchema.nullable().optional(),
});

export type ValueRecord = z.infer<typeof valueSchema>;
export type ValuePatch = z.infer<typeof valuePatchSchema>;
export type GoalRecord = z.infer<typeof goalSchema>;
export type GoalPatch = z.infer<typeof goalPatchSchema>;
export type GoalStatus = z.infer<typeof goalStatusSchema>;
export type OpenQuestion = z.infer<typeof openQuestionSchema>;
export type OpenQuestionPatch = z.infer<typeof openQuestionPatchSchema>;
export type OpenQuestionStatus = z.infer<typeof openQuestionStatusSchema>;
export type OpenQuestionSource = z.infer<typeof openQuestionSourceSchema>;
export type TraitRecord = z.infer<typeof traitSchema>;
export type TraitPatch = z.infer<typeof traitPatchSchema>;
export type AutobiographicalPeriod = z.infer<typeof autobiographicalPeriodSchema>;
export type AutobiographicalPeriodPatch = z.infer<typeof autobiographicalPeriodPatchSchema>;
export type SelfProvenance = Provenance;
export type IdentityState = z.infer<typeof identityStateSchema>;

export type GoalTreeNode = GoalRecord & {
  children: GoalTreeNode[];
};

export type OpenQuestionListOptions = {
  status?: OpenQuestionStatus;
  source?: OpenQuestionSource;
  minUrgency?: number;
  visibleToAudienceEntityId?: EntityId | null;
  limit?: number;
};

export type OpenQuestionSearchCandidate = {
  question: OpenQuestion;
  similarity: number;
};

export type OpenQuestionHandleLookupOptions = {
  streamEntryIds?: readonly string[];
  episodeIds?: readonly string[];
  statuses?: readonly OpenQuestionStatus[];
  visibleToAudienceEntityId?: EntityId | null;
  limit?: number;
};

export type OpenQuestionGoalLookupOptions = {
  goalId: GoalId;
  statuses?: readonly OpenQuestionStatus[];
  limit?: number;
};

export type OpenQuestionSimilarLookupOptions = {
  question: string;
  /** Retained for API compatibility; duplicate lookup is intentionally global. */
  audienceEntityId?: EntityId | null;
};
