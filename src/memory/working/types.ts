import { z } from "zod";

import {
  cognitiveModeSchema,
  intentRecordSchema,
  type CognitiveMode,
  type IntentRecord,
} from "../../cognition/types.js";
import { affectiveSignalSchema } from "../affective/types.js";
import {
  entityIdHelpers,
  episodeIdHelpers,
  isSessionId,
  parseSessionId,
  skillIdHelpers,
  streamEntryIdHelpers,
  type EntityId,
  type EpisodeId,
  type SessionId,
  type SkillId,
  type StreamEntryId,
} from "../../util/ids.js";

export const workingSessionIdSchema = z
  .string()
  .refine((value) => isSessionId(value), {
    message: "Invalid session id",
  })
  .transform((value) => parseSessionId(value));

export const suppressedEntrySchema = z.object({
  id: z.string().min(1),
  reason: z.string().min(1),
  until_turn: z.number().int().nonnegative(),
});

export const workingSkillIdSchema = z
  .string()
  .refine((value) => skillIdHelpers.is(value), {
    message: "Invalid skill id",
  })
  .transform((value) => value as SkillId);

export const workingEpisodeIdSchema = z
  .string()
  .refine((value) => episodeIdHelpers.is(value), {
    message: "Invalid episode id",
  })
  .transform((value) => value as EpisodeId);

export const workingEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid entity id",
  })
  .transform((value) => value as EntityId);

export const workingStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const pendingSocialAttributionSchema = z.object({
  entity_id: z.string().min(1),
  interaction_id: z.number().int().positive(),
  agent_response_summary: z.string().min(1).nullable(),
  turn_completed_ts: z.number().finite(),
});

export const pendingTraitAttributionSchema = z.object({
  trait_label: z.string().min(1),
  strength_delta: z.number().min(0).max(0.2).default(0.05),
  source_episode_ids: z.array(workingEpisodeIdSchema).min(1),
  turn_completed_ts: z.number().finite(),
  audience_entity_id: workingEntityIdSchema.nullable(),
});

export const pendingProceduralAttemptSchema = z.object({
  problem_text: z.string().min(1),
  approach_summary: z.string().min(1),
  selected_skill_id: workingSkillIdSchema.nullable(),
  source_stream_ids: z.array(workingStreamEntryIdSchema).min(1),
  turn_counter: z.number().int().nonnegative(),
  audience_entity_id: workingEntityIdSchema.nullable(),
});

export const workingMemorySchema = z.object({
  session_id: workingSessionIdSchema,
  turn_counter: z.number().int().nonnegative(),
  current_focus: z.string().min(1).nullable(),
  hot_entities: z.array(z.string().min(1)),
  pending_intents: z.array(intentRecordSchema),
  pending_social_attribution: pendingSocialAttributionSchema.nullable().default(null),
  pending_trait_attribution: pendingTraitAttributionSchema.nullable().default(null),
  suppressed: z.array(suppressedEntrySchema).default([]),
  mood: affectiveSignalSchema.nullable().default(null),
  last_selected_skill_id: workingSkillIdSchema.nullable().default(null),
  last_selected_skill_turn: z.number().int().nonnegative().nullable().default(null),
  pending_procedural_attempt: pendingProceduralAttemptSchema.nullable().default(null),
  mode: cognitiveModeSchema.nullable(),
  updated_at: z.number().finite(),
});

export type WorkingMemory = z.infer<typeof workingMemorySchema>;
export type PendingSocialAttribution = z.infer<typeof pendingSocialAttributionSchema>;
export type PendingTraitAttribution = z.infer<typeof pendingTraitAttributionSchema>;
export type PendingProceduralAttempt = z.infer<typeof pendingProceduralAttemptSchema>;

/**
 * Derived live-state only. Phase E removed `scratchpad` (S2 planner output
 * -- the stream now persists that as a structured `plan:` thought entry)
 * and `recent_thoughts` (agent self-talk, redundant with the stream's
 * `thought` entries and never a source of recent dialogue). If cognition
 * needs discourse state later (thread summary, active threads), derive it
 * from the stream per-turn rather than caching it here.
 */
export function createWorkingMemory(sessionId: SessionId, timestamp: number): WorkingMemory {
  return {
    session_id: sessionId,
    turn_counter: 0,
    current_focus: null,
    hot_entities: [],
    pending_intents: [],
    pending_social_attribution: null,
    pending_trait_attribution: null,
    suppressed: [],
    mood: null,
    last_selected_skill_id: null,
    last_selected_skill_turn: null,
    pending_procedural_attempt: null,
    mode: null,
    updated_at: timestamp,
  };
}

export type SuppressedEntry = z.infer<typeof suppressedEntrySchema>;

export type { CognitiveMode, IntentRecord, SessionId };
