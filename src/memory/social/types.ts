import { z } from "zod";

import { entityIdSchema } from "../commitments/types.js";
import { provenanceSchema } from "../common/provenance.js";

export const socialSentimentPointSchema = z.object({
  ts: z.number().finite(),
  valence: z.number().min(-1).max(1),
});

export const socialEventKindSchema = z.enum(["interaction", "trust_adjustment", "baseline"]);

export const socialProfileSchema = z.object({
  entity_id: entityIdSchema,
  trust: z.number().min(0).max(1),
  attachment: z.number().min(0).max(1),
  communication_style: z.string().min(1).nullable(),
  shared_history_summary: z.string().min(1).nullable(),
  last_interaction_at: z.number().finite().nullable(),
  interaction_count: z.number().int().nonnegative(),
  commitment_count: z.number().int().nonnegative(),
  sentiment_history: z.array(socialSentimentPointSchema).max(50),
  notes: z.string().min(1).nullable(),
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
});

export const socialEventSchema = z.object({
  id: z.number().int().positive(),
  entity_id: entityIdSchema,
  ts: z.number().finite(),
  kind: socialEventKindSchema,
  provenance: provenanceSchema,
  trust_delta: z.number().finite(),
  attachment_delta: z.number().finite(),
  interaction_delta: z.number().int(),
  valence: z.number().min(-1).max(1).nullable(),
});

export type SocialSentimentPoint = z.infer<typeof socialSentimentPointSchema>;
export type SocialProfile = z.infer<typeof socialProfileSchema>;
export type SocialEvent = z.infer<typeof socialEventSchema>;
