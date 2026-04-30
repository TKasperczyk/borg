import { z } from "zod";

export { affectiveSignalSchema } from "../memory/affective/types.js";
import { affectiveSignalSchema } from "../memory/affective/types.js";

export const COGNITIVE_MODES = ["problem_solving", "relational", "reflective", "idle"] as const;

export const cognitiveModeSchema = z.enum(COGNITIVE_MODES);

export const temporalCueSchema = z.object({
  sinceTs: z.number().finite().optional(),
  untilTs: z.number().finite().optional(),
  label: z.string().min(1).optional(),
});

export const factualChallengeSignalSchema = z.object({
  disputed_entity: z.string().trim().min(1).nullable(),
  disputed_property: z.string().trim().min(1).nullable(),
  user_position: z.string().trim().min(1),
});

export const intentRecordSchema = z.object({
  description: z.string().min(1),
  next_action: z.string().min(1).nullable(),
});

export const attentionWeightsSchema = z.object({
  semantic: z.number().min(0),
  goal_relevance: z.number().min(0),
  value_alignment: z.number().min(0),
  mood: z.number().min(0),
  time: z.number().min(0),
  social: z.number().min(0),
  entity: z.number().min(0),
  heat: z.number().min(0),
  suppression_penalty: z.number().min(0),
});

export const perceptionResultSchema = z.object({
  entities: z.array(z.string().min(1)),
  mode: cognitiveModeSchema,
  affectiveSignal: affectiveSignalSchema,
  affectiveSignalDegraded: z.boolean().optional(),
  temporalCue: temporalCueSchema.nullable(),
  factualChallenge: factualChallengeSignalSchema.nullable().optional(),
});

export type CognitiveMode = z.infer<typeof cognitiveModeSchema>;
export type AffectiveSignal = z.infer<typeof affectiveSignalSchema>;
export type TemporalCue = z.infer<typeof temporalCueSchema>;
export type FactualChallengeSignal = z.infer<typeof factualChallengeSignalSchema>;
export type IntentRecord = z.infer<typeof intentRecordSchema>;
export type AttentionWeights = z.infer<typeof attentionWeightsSchema>;
export type PerceptionResult = z.infer<typeof perceptionResultSchema>;
