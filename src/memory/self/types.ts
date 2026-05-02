import { z } from "zod";

import {
  entityIdHelpers,
  episodeIdHelpers,
  goalIdHelpers,
  streamEntryIdHelpers,
  traitIdHelpers,
  valueIdHelpers,
  type EntityId,
  type EpisodeId,
  type GoalId,
  type StreamEntryId,
  type TraitId,
  type ValueId,
} from "../../util/ids.js";
import { provenanceSchema, type Provenance } from "../common/provenance.js";

export const goalStatusSchema = z.enum(["active", "done", "abandoned", "blocked"]);
export const identityStateSchema = z.enum(["candidate", "established"]);

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

export const goalAudienceEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid goal audience entity id",
  })
  .transform((value) => value as EntityId);

export const goalSourceStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid goal source stream entry id",
  })
  .transform((value) => value as StreamEntryId);

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

export const valueSchema = z.object({
  id: valueIdSchema,
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
  description: z.string().min(1),
  priority: z.number().finite(),
  parent_goal_id: goalIdSchema.nullable(),
  status: goalStatusSchema,
  progress_notes: z.string().nullable(),
  last_progress_ts: z.number().finite().nullable(),
  created_at: z.number().finite(),
  target_at: z.number().finite().nullable(),
  audience_entity_id: goalAudienceEntityIdSchema.nullable().default(null),
  source_stream_entry_ids: z.array(goalSourceStreamEntryIdSchema).min(1).optional(),
  provenance: provenanceSchema,
});

export const traitSchema = z.object({
  id: traitIdSchema,
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
    created_at: true,
    audience_entity_id: true,
    source_stream_entry_ids: true,
  })
  .extend({
    audience_entity_id: goalAudienceEntityIdSchema.nullable().optional(),
    source_stream_entry_ids: z.array(goalSourceStreamEntryIdSchema).min(1).optional(),
  })
  .partial()
  .strict();

export type ValueRecord = z.infer<typeof valueSchema>;
export type ValuePatch = z.infer<typeof valuePatchSchema>;
export type GoalRecord = z.infer<typeof goalSchema>;
export type GoalPatch = z.infer<typeof goalPatchSchema>;
export type GoalStatus = z.infer<typeof goalStatusSchema>;
export type TraitRecord = z.infer<typeof traitSchema>;
export type TraitPatch = z.infer<typeof traitPatchSchema>;
export type SelfProvenance = Provenance;
export type IdentityState = z.infer<typeof identityStateSchema>;

export type GoalTreeNode = GoalRecord & {
  children: GoalTreeNode[];
};
