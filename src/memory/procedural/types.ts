import { z } from "zod";

import { type EpisodeId, type SkillId, skillIdHelpers } from "../../util/ids.js";
import { episodeIdSchema } from "../episodic/types.js";

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
  alpha: z.number().positive(),
  beta: z.number().positive(),
  attempts: z.number().int().nonnegative(),
  successes: z.number().int().nonnegative(),
  failures: z.number().int().nonnegative(),
  alternatives: z.array(skillIdSchema),
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
};

export type SkillSelectionResult = {
  skill: SkillRecord;
  sampledValue: number;
  evaluatedCandidates: SkillSelectionCandidate[];
};

export type SkillIdValue = SkillId;
export type EpisodeIdValue = EpisodeId;
