import { z } from "zod";

import {
  cognitiveModeSchema,
  intentRecordSchema,
  type CognitiveMode,
  type IntentRecord,
} from "../../cognition/types.js";
import { isSessionId, parseSessionId, type SessionId } from "../../util/ids.js";

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

export const workingMemorySchema = z.object({
  session_id: workingSessionIdSchema,
  turn_counter: z.number().int().nonnegative(),
  scratchpad: z.string(),
  current_focus: z.string().min(1).nullable(),
  recent_thoughts: z.array(z.string()),
  hot_entities: z.array(z.string().min(1)),
  pending_intents: z.array(intentRecordSchema),
  suppressed: z.array(suppressedEntrySchema).default([]),
  mode: cognitiveModeSchema.nullable(),
  updated_at: z.number().finite(),
});

export type WorkingMemory = z.infer<typeof workingMemorySchema>;

export function createWorkingMemory(sessionId: SessionId, timestamp: number): WorkingMemory {
  return {
    session_id: sessionId,
    turn_counter: 0,
    scratchpad: "",
    current_focus: null,
    recent_thoughts: [],
    hot_entities: [],
    pending_intents: [],
    suppressed: [],
    mode: null,
    updated_at: timestamp,
  };
}

export function pushRecentThought(
  workingMemory: WorkingMemory,
  thought: string,
  limit = 10,
): WorkingMemory {
  const trimmed = thought.trim();

  if (trimmed.length === 0) {
    return workingMemory;
  }

  const recentThoughts = [...workingMemory.recent_thoughts, trimmed].slice(-limit);

  return {
    ...workingMemory,
    recent_thoughts: recentThoughts,
  };
}

export type SuppressedEntry = z.infer<typeof suppressedEntrySchema>;

export type { CognitiveMode, IntentRecord, SessionId };
