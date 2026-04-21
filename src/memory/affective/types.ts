import { z } from "zod";

import {
  DEFAULT_SESSION_ID,
  episodeIdHelpers,
  sessionIdHelpers,
  type EpisodeId,
  type SessionId,
} from "../../util/ids.js";

export const affectiveSessionIdSchema = z
  .string()
  .refine((value) => value === DEFAULT_SESSION_ID || sessionIdHelpers.is(value), {
    message: "Invalid session id",
  })
  .transform((value) => value as SessionId);

export const affectiveEpisodeIdSchema = z
  .string()
  .refine((value) => episodeIdHelpers.is(value), {
    message: "Invalid episode id",
  })
  .transform((value) => value as EpisodeId);

export const DOMINANT_EMOTIONS = [
  "joy",
  "sadness",
  "fear",
  "anger",
  "surprise",
  "curiosity",
  "neutral",
] as const;

export const dominantEmotionSchema = z.enum(DOMINANT_EMOTIONS);

export const affectiveSignalSchema = z.object({
  valence: z.number().min(-1).max(1),
  arousal: z.number().min(0).max(1),
  dominant_emotion: dominantEmotionSchema.nullable().default(null),
});

export const emotionalArcSchema = z.object({
  start: affectiveSignalSchema.omit({
    dominant_emotion: true,
  }),
  peak: affectiveSignalSchema.omit({
    dominant_emotion: true,
  }),
  end: affectiveSignalSchema.omit({
    dominant_emotion: true,
  }),
  dominant_emotion: dominantEmotionSchema.nullable(),
});

export const moodStateSchema = z.object({
  session_id: affectiveSessionIdSchema,
  valence: z.number().min(-1).max(1),
  arousal: z.number().min(0).max(1),
  updated_at: z.number().finite(),
  half_life_hours: z.number().positive(),
  recent_triggers: z.array(z.string().min(1)),
});

export const moodHistoryEntrySchema = z.object({
  id: z.number().int().positive(),
  session_id: affectiveSessionIdSchema,
  ts: z.number().finite(),
  valence: z.number().min(-1).max(1),
  arousal: z.number().min(0).max(1),
  trigger_episode_id: affectiveEpisodeIdSchema.nullable(),
  trigger_reason: z.string().min(1).nullable(),
});

export type DominantEmotion = z.infer<typeof dominantEmotionSchema>;
export type AffectiveSignal = z.infer<typeof affectiveSignalSchema>;
export type EmotionalArc = z.infer<typeof emotionalArcSchema>;
export type MoodState = z.infer<typeof moodStateSchema>;
export type MoodHistoryEntry = z.infer<typeof moodHistoryEntrySchema>;

export function createNeutralAffectiveSignal(): AffectiveSignal {
  return {
    valence: 0,
    arousal: 0,
    dominant_emotion: null,
  };
}

export function createNeutralEmotionalArc(): EmotionalArc {
  return {
    start: {
      valence: 0,
      arousal: 0,
    },
    peak: {
      valence: 0,
      arousal: 0,
    },
    end: {
      valence: 0,
      arousal: 0,
    },
    dominant_emotion: null,
  };
}
