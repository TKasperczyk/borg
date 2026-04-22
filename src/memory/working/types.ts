import { z } from "zod";

import {
  cognitiveModeSchema,
  intentRecordSchema,
  type CognitiveMode,
  type IntentRecord,
} from "../../cognition/types.js";
import { affectiveSignalSchema } from "../affective/types.js";
import {
  isSessionId,
  parseSessionId,
  skillIdHelpers,
  type SessionId,
  type SkillId,
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

export const workingMemorySchema = z.object({
  session_id: workingSessionIdSchema,
  turn_counter: z.number().int().nonnegative(),
  current_focus: z.string().min(1).nullable(),
  hot_entities: z.array(z.string().min(1)),
  pending_intents: z.array(intentRecordSchema),
  suppressed: z.array(suppressedEntrySchema).default([]),
  mood: affectiveSignalSchema.nullable().default(null),
  last_selected_skill_id: workingSkillIdSchema.nullable().default(null),
  last_selected_skill_turn: z.number().int().nonnegative().nullable().default(null),
  mode: cognitiveModeSchema.nullable(),
  updated_at: z.number().finite(),
});

export type WorkingMemory = z.infer<typeof workingMemorySchema>;

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
    suppressed: [],
    mood: null,
    last_selected_skill_id: null,
    last_selected_skill_turn: null,
    mode: null,
    updated_at: timestamp,
  };
}

export type SuppressedEntry = z.infer<typeof suppressedEntrySchema>;

export type { CognitiveMode, IntentRecord, SessionId };
