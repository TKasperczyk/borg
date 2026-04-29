import { z } from "zod";

import { type EpisodeId, type SkillId, skillIdHelpers } from "../../util/ids.js";
import {
  proceduralEvidenceIdHelpers,
  type EntityId,
  type ProceduralEvidenceId,
} from "../../util/ids.js";
import { episodeIdSchema } from "../episodic/types.js";
import {
  pendingProceduralAttemptSchema,
  workingEntityIdSchema,
  type PendingProceduralAttempt,
} from "../working/types.js";
import {
  proceduralContextMetadataSchema,
  proceduralContextSchema,
  type ProceduralContext,
  type ProceduralContextMetadata,
} from "./context.js";

export const skillIdSchema = z
  .string()
  .refine((value) => skillIdHelpers.is(value), {
    message: "Invalid skill id",
  })
  .transform((value) => value as SkillId);

export const skillSchema = z.object({
  id: skillIdSchema,
  applies_when: z.string().min(1),
  approach: z.string().min(1),
  status: z.enum(["active", "superseded"]).default("active"),
  alpha: z.number().positive(),
  beta: z.number().positive(),
  attempts: z.number().int().nonnegative(),
  successes: z.number().int().nonnegative(),
  failures: z.number().int().nonnegative(),
  alternatives: z.array(skillIdSchema),
  superseded_by: z.array(skillIdSchema).default([]),
  superseded_at: z.number().finite().nullable().default(null),
  splitting_at: z.number().finite().nullable().default(null),
  last_split_attempt_at: z.number().finite().nullable().optional(),
  split_failure_count: z.number().int().nonnegative().default(0),
  last_split_error: z.string().min(1).nullable().default(null),
  requires_manual_review: z.boolean().default(false),
  source_episode_ids: z.array(episodeIdSchema).min(1),
  last_used: z.number().finite().nullable(),
  last_successful: z.number().finite().nullable(),
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
});

export type SkillRecord = z.infer<typeof skillSchema>;

export const skillInsertSchema = skillSchema;

export const skillStatsSchema = z.object({
  mean: z.number().min(0).max(1),
  mode: z.number().min(0).max(1).optional(),
  ci_95: z.tuple([z.number().min(0).max(1), z.number().min(0).max(1)]),
});

export type SkillStats = z.infer<typeof skillStatsSchema>;

export type SkillSearchCandidate = {
  skill: SkillRecord;
  similarity: number;
};

export type SkillSelectionCandidate = SkillSearchCandidate & {
  sampledValue: number;
  stats: SkillStats;
  contextStats?: SkillContextStatsRecord | null;
  sampledAlpha?: number;
  sampledBeta?: number;
};

export type SkillSelectionResult = {
  skill: SkillRecord;
  sampledValue: number;
  evaluatedCandidates: SkillSelectionCandidate[];
  proceduralContext?: ProceduralContext | null;
};

export const proceduralOutcomeClassificationSchema = z.enum(["success", "failure", "unclear"]);

export const proceduralEvidenceIdSchema = z
  .string()
  .refine((value) => proceduralEvidenceIdHelpers.is(value), {
    message: "Invalid procedural evidence id",
  })
  .transform((value) => value as ProceduralEvidenceId);

export const proceduralEvidenceSchema = z.object({
  id: proceduralEvidenceIdSchema,
  pending_attempt_snapshot: pendingProceduralAttemptSchema,
  classification: proceduralOutcomeClassificationSchema,
  evidence_text: z.string().min(1),
  grounded: z.boolean().default(true),
  skill_actually_applied: z.boolean().default(true),
  procedural_context: proceduralContextSchema.nullable().optional(),
  resolved_episode_ids: z.array(episodeIdSchema),
  audience_entity_id: workingEntityIdSchema.nullable(),
  consumed_at: z.number().finite().nullable(),
  created_at: z.number().finite(),
});

export type ProceduralOutcomeClassification = z.infer<typeof proceduralOutcomeClassificationSchema>;
export type ProceduralEvidenceRecord = z.infer<typeof proceduralEvidenceSchema>;

export const skillContextStatsSchema = z.object({
  skill_id: skillIdSchema,
  context_key: z.string().min(1),
  procedural_context: proceduralContextMetadataSchema.nullable().optional(),
  alpha: z.number().positive(),
  beta: z.number().positive(),
  attempts: z.number().int().nonnegative(),
  successes: z.number().int().nonnegative(),
  failures: z.number().int().nonnegative(),
  last_used: z.number().finite().nullable(),
  last_successful: z.number().finite().nullable(),
  updated_at: z.number().finite(),
});

export type SkillContextStatsRecord = z.infer<typeof skillContextStatsSchema>;

export type SkillIdValue = SkillId;
export type EpisodeIdValue = EpisodeId;
export type EntityIdValue = EntityId;
export type PendingProceduralAttemptValue = PendingProceduralAttempt;
export type ProceduralEvidenceIdValue = ProceduralEvidenceId;
export type ProceduralContextValue = ProceduralContext;
export type ProceduralContextMetadataValue = ProceduralContextMetadata;
