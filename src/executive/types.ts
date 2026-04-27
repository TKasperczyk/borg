import { z } from "zod";

import type { GoalRecord } from "../memory/self/index.js";
import { provenanceSchema, type Provenance } from "../memory/common/provenance.js";
import {
  executiveStepIdHelpers,
  goalIdHelpers,
  type ExecutiveStepId,
  type GoalId,
} from "../util/ids.js";

export const executiveStepIdSchema = z
  .string()
  .refine((value) => executiveStepIdHelpers.is(value), {
    message: "Invalid executive step id",
  })
  .transform((value) => value as ExecutiveStepId);

export const executiveStepGoalIdSchema = z
  .string()
  .refine((value) => goalIdHelpers.is(value), {
    message: "Invalid goal id",
  })
  .transform((value) => value as GoalId);

export const executiveStepStatusSchema = z.enum([
  "queued",
  "doing",
  "done",
  "blocked",
  "abandoned",
]);

export const executiveStepKindSchema = z.enum(["think", "ask_user", "research", "act", "wait"]);

export const executiveStepSchema = z.object({
  id: executiveStepIdSchema,
  goal_id: executiveStepGoalIdSchema,
  description: z.string().min(1),
  status: executiveStepStatusSchema,
  kind: executiveStepKindSchema,
  due_at: z.number().finite().nullable(),
  last_attempt_ts: z.number().finite().nullable(),
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
  provenance: provenanceSchema,
});

export const executiveStepPatchSchema = executiveStepSchema
  .omit({
    id: true,
    goal_id: true,
    created_at: true,
    updated_at: true,
  })
  .partial()
  .strict();

export type ExecutiveStep = z.infer<typeof executiveStepSchema>;
export type ExecutiveStepPatch = z.infer<typeof executiveStepPatchSchema>;
export type ExecutiveStepStatus = z.infer<typeof executiveStepStatusSchema>;
export type ExecutiveStepKind = z.infer<typeof executiveStepKindSchema>;
export type ExecutiveStepProvenance = Provenance;

export type ExecutiveGoalScoreComponents = {
  priority: number;
  deadline_pressure: number;
  context_fit: number;
  progress_debt: number;
};

export type ExecutiveGoalScore = {
  goal_id: GoalRecord["id"];
  goal: GoalRecord;
  score: number;
  components: ExecutiveGoalScoreComponents;
  reason: string;
};

export type ExecutiveFocus = {
  selected_goal: GoalRecord | null;
  selected_score: ExecutiveGoalScore | null;
  next_step?: ExecutiveStep | null;
  candidates: ExecutiveGoalScore[];
  threshold: number;
};
