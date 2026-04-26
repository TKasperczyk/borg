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
  resolved_episode_ids: z.array(episodeIdSchema),
  audience_entity_id: workingEntityIdSchema.nullable(),
  consumed_at: z.number().finite().nullable(),
  created_at: z.number().finite(),
});

export type ProceduralOutcomeClassification = z.infer<
  typeof proceduralOutcomeClassificationSchema
>;
export type ProceduralEvidenceRecord = z.infer<typeof proceduralEvidenceSchema>;

const ASSISTANT_ONLY_EVIDENCE_PATTERN =
  /\b(assistant|agent|model|borg|response|answer|i said|i suggested|we suggested|the suggestion)\b/i;
const USER_SIGNAL_EVIDENCE_PATTERN =
  /\b(user|they|them|their|reply|follow-up|message|reported|reports|confirmed|confirms)\b/i;

export function isProceduralOutcomeEvidenceGrounded(input: {
  classification: ProceduralOutcomeClassification;
  evidence_text: string;
}): boolean {
  if (input.classification === "unclear") {
    return true;
  }

  const evidence = input.evidence_text.trim();

  if (evidence.length === 0) {
    return false;
  }

  if (
    ASSISTANT_ONLY_EVIDENCE_PATTERN.test(evidence) &&
    !USER_SIGNAL_EVIDENCE_PATTERN.test(evidence)
  ) {
    return false;
  }

  return USER_SIGNAL_EVIDENCE_PATTERN.test(evidence);
}

export type SkillIdValue = SkillId;
export type EpisodeIdValue = EpisodeId;
export type EntityIdValue = EntityId;
export type PendingProceduralAttemptValue = PendingProceduralAttempt;
export type ProceduralEvidenceIdValue = ProceduralEvidenceId;
