import { z } from "zod";

import {
  cognitiveModeSchema,
  intentRecordSchema,
  type CognitiveMode,
  type IntentRecord,
} from "../../cognition/types.js";
import { affectiveSignalSchema } from "../affective/types.js";
import { proceduralContextSchema } from "../procedural/context.js";
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

export const suppressedEntrySchema = z
  .object({
    id: z.string().min(1),
    reason: z.string().min(1),
    until_turn: z.number().int().nonnegative(),
  })
  .strict();

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

export const pendingSocialAttributionSchema = z
  .object({
    entity_id: z.string().min(1),
    interaction_id: z.number().int().positive(),
    agent_response_summary: z.string().min(1).nullable(),
    turn_completed_ts: z.number().finite(),
  })
  .strict();

export const pendingTraitAttributionSchema = z
  .object({
    trait_label: z.string().min(1),
    strength_delta: z.number().min(0).max(0.2),
    // Sprint 56: trait demonstration is evidenced by the assistant turn
    // that actually displayed it -- captured here as the user_msg/agent_msg
    // stream entries from the demonstrating turn. The orchestrator resolves
    // these to the extracted episode at consumption time.
    source_stream_entry_ids: z.array(workingStreamEntryIdSchema).min(1),
    turn_completed_ts: z.number().finite(),
    audience_entity_id: workingEntityIdSchema.nullable(),
  })
  .strict();

export const pendingProceduralAttemptSchema = z
  .object({
    problem_text: z.string().min(1),
    approach_summary: z.string().min(1),
    selected_skill_id: workingSkillIdSchema.nullable(),
    source_stream_ids: z.array(workingStreamEntryIdSchema).min(1),
    turn_counter: z.number().int().nonnegative(),
    audience_entity_id: workingEntityIdSchema.nullable(),
    procedural_context: proceduralContextSchema.nullable().optional(),
  })
  .strict();

export const discourseStopProvenanceSchema = z.enum([
  "generation_gate",
  "self_commitment_extractor",
  "no_output_tool",
  "s2_planner_no_output",
  "commitment_guard",
  "relational_guard",
]);

export const stopUntilSubstantiveContentSchema = z
  .object({
    provenance: discourseStopProvenanceSchema,
    source_stream_entry_id: workingStreamEntryIdSchema.optional(),
    reason: z.string().min(1),
    since_turn: z.number().int().nonnegative(),
  })
  .strict();

export const discourseStateSchema = z
  .object({
    stop_until_substantive_content: stopUntilSubstantiveContentSchema.nullable(),
  })
  .strict();

export const pendingActionRecordSchema = intentRecordSchema.extend({
  created_at: z.number().finite().optional(),
});

// Sprint 53: pending procedural attempts are now a bounded list, not one
// slot. Multi-step debugging or delayed-feedback work needs to track each
// attempt independently; reflection retires only grounded success/failure
// outcomes, leaving unclear ones pending until they age out via TTL.
export const PENDING_PROCEDURAL_ATTEMPTS_LIMIT = 5;
export const PENDING_PROCEDURAL_ATTEMPT_TTL_TURNS = 8;

const workingMemoryObjectSchema = z
  .object({
    session_id: workingSessionIdSchema,
    turn_counter: z.number().int().nonnegative(),
    hot_entities: z.array(z.string().min(1)),
    pending_actions: z.array(pendingActionRecordSchema),
    pending_social_attribution: pendingSocialAttributionSchema.nullable(),
    pending_trait_attribution: pendingTraitAttributionSchema.nullable(),
    suppressed: z.array(suppressedEntrySchema),
    mood: affectiveSignalSchema.nullable(),
    pending_procedural_attempts: z.array(pendingProceduralAttemptSchema),
    discourse_state: discourseStateSchema,
    mode: cognitiveModeSchema.nullable(),
    updated_at: z.number().finite(),
  })
  .strict();

export const workingMemorySchema = workingMemoryObjectSchema;

export type WorkingMemory = z.infer<typeof workingMemorySchema>;
export type PendingActionRecord = z.infer<typeof pendingActionRecordSchema>;
export type PendingSocialAttribution = z.infer<typeof pendingSocialAttributionSchema>;
export type PendingTraitAttribution = z.infer<typeof pendingTraitAttributionSchema>;
export type PendingProceduralAttempt = z.infer<typeof pendingProceduralAttemptSchema>;
export type DiscourseState = z.infer<typeof discourseStateSchema>;
export type DiscourseStopProvenance = z.infer<typeof discourseStopProvenanceSchema>;
export type StopUntilSubstantiveContent = z.infer<typeof stopUntilSubstantiveContentSchema>;

/**
 * Derived live-state only. Phase E removed `scratchpad` (S2 planner output
 * -- the stream now persists that as a structured `plan:` thought entry)
 * and `recent_thoughts` (agent self-talk, redundant with the stream's
 * `thought` entries and never a source of recent dialogue). Discourse
 * control state is the one durable turn-control exception: it gates future
 * generation rather than summarizing conversation content.
 */
export function createWorkingMemory(sessionId: SessionId, timestamp: number): WorkingMemory {
  return {
    session_id: sessionId,
    turn_counter: 0,
    hot_entities: [],
    pending_actions: [],
    pending_social_attribution: null,
    pending_trait_attribution: null,
    suppressed: [],
    mood: null,
    pending_procedural_attempts: [],
    discourse_state: {
      stop_until_substantive_content: null,
    },
    mode: null,
    updated_at: timestamp,
  };
}

export type SuppressedEntry = z.infer<typeof suppressedEntrySchema>;

export type { CognitiveMode, IntentRecord, SessionId };
