import { z } from "zod";

import {
  actionIdHelpers,
  sharedStateEntryIdHelpers,
  entityIdHelpers,
  episodeIdHelpers,
  streamEntryIdHelpers,
  type ActionId,
  type SharedStateEntryId,
  type EntityId,
  type EpisodeId,
  type GoalId,
  type OpenQuestionId,
  type SessionId,
  type StreamEntryId,
  isSessionId,
  parseSessionId,
} from "../../util/ids.js";
import { goalIdSchema, openQuestionIdSchema } from "../self/types.js";

export const ACTION_STATES = [
  "considering",
  "committed_to_do",
  "scheduled",
  "completed",
  "not_done",
  "expired",
  "archived",
  "unknown",
] as const;

export type ActionState = (typeof ACTION_STATES)[number];

export type ActionStateTimestampField =
  | "considering_at"
  | "committed_at"
  | "scheduled_at"
  | "completed_at"
  | "not_done_at"
  | "expired_at"
  | "archived_at"
  | "unknown_at";

export type ActionStateMetadata = {
  timestamp_field: ActionStateTimestampField;
  active: boolean;
  terminal: boolean;
};

export const ACTION_STATE_METADATA: Record<ActionState, ActionStateMetadata> = {
  considering: {
    timestamp_field: "considering_at",
    active: true,
    terminal: false,
  },
  committed_to_do: {
    timestamp_field: "committed_at",
    active: true,
    terminal: false,
  },
  scheduled: {
    timestamp_field: "scheduled_at",
    active: true,
    terminal: false,
  },
  completed: {
    timestamp_field: "completed_at",
    active: false,
    terminal: true,
  },
  not_done: {
    timestamp_field: "not_done_at",
    active: false,
    terminal: true,
  },
  expired: {
    timestamp_field: "expired_at",
    active: false,
    terminal: false,
  },
  archived: {
    timestamp_field: "archived_at",
    active: false,
    terminal: false,
  },
  unknown: {
    timestamp_field: "unknown_at",
    active: true,
    terminal: false,
  },
};

// Active means not terminal/expired/archived; pressure for canonicalization/action-bloat observability.
export const ACTIVE_ACTION_STATES: readonly ActionState[] = ACTION_STATES.filter(
  (state) => ACTION_STATE_METADATA[state].active,
);

export const ACTION_SESSION_SCOPES = ["current_session", "next_session"] as const;

export const actionIdSchema = z
  .string()
  .refine((value) => actionIdHelpers.is(value), {
    message: "Invalid action id",
  })
  .transform((value) => value as ActionId);

export const actionEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid action entity id",
  })
  .transform((value) => value as EntityId);

export const actionActorSchema = z.union([z.enum(["user", "borg"]), actionEntityIdSchema]);

export const actionStateSchema = z.enum(ACTION_STATES);

export const actionSessionScopeSchema = z.enum(ACTION_SESSION_SCOPES);

export const actionSessionAnchorIdSchema = z
  .string()
  .refine((value) => isSessionId(value), {
    message: "Invalid action session anchor id",
  })
  .transform((value) => parseSessionId(value));

export const actionEpisodeIdSchema = z
  .string()
  .refine((value) => episodeIdHelpers.is(value), {
    message: "Invalid action provenance episode id",
  })
  .transform((value) => value as EpisodeId);

export const actionStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid action provenance stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const actionCanonicalizedByArtifactEntryIdSchema = z
  .string()
  .refine((value) => sharedStateEntryIdHelpers.is(value), {
    message: "Invalid action canonicalized shared state entry id",
  })
  .transform((value) => value as SharedStateEntryId);

const actionRecordShape = z.object({
  id: actionIdSchema,
  description: z.string().min(1),
  actor: actionActorSchema,
  audience_entity_id: actionEntityIdSchema.nullable(),
  goal_id: goalIdSchema.nullable().default(null),
  open_question_id: openQuestionIdSchema.nullable().default(null),
  state: actionStateSchema,
  confidence: z.number().min(0).max(1),
  provenance_episode_ids: z.array(actionEpisodeIdSchema),
  provenance_stream_entry_ids: z.array(actionStreamEntryIdSchema),
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
  considering_at: z.number().finite().nullable(),
  committed_at: z.number().finite().nullable(),
  scheduled_at: z.number().finite().nullable(),
  completed_at: z.number().finite().nullable(),
  not_done_at: z.number().finite().nullable(),
  expired_at: z.number().finite().nullable().default(null),
  archived_at: z.number().finite().nullable().default(null),
  unknown_at: z.number().finite().nullable(),
  canonicalized_by_artifact_entry_id: actionCanonicalizedByArtifactEntryIdSchema
    .nullable()
    .optional(),
  session_scope: actionSessionScopeSchema.nullable().default(null),
  session_anchor_id: actionSessionAnchorIdSchema.nullable().default(null),
  last_referenced_at_ms: z.number().finite().nullable().default(null),
  last_referenced_turn_counter: z.number().int().nonnegative().nullable().default(null),
  last_referenced_turn_global: z.number().int().nonnegative().nullable().optional(),
});

export const actionRecordSchema = actionRecordShape
  .refine((value) => value.updated_at >= value.created_at, {
    message: "Action updated_at must be greater than or equal to created_at",
    path: ["updated_at"],
  })
  .refine(
    (value) =>
      value.provenance_episode_ids.length > 0 || value.provenance_stream_entry_ids.length > 0,
    {
      message: "Action record requires episode or stream provenance",
      path: ["provenance_stream_entry_ids"],
    },
  );

export const actionRecordPatchSchema = actionRecordShape
  .omit({
    id: true,
    created_at: true,
  })
  .partial()
  .strict();

export type ActionActor = z.infer<typeof actionActorSchema>;
export type ActionSessionScope = z.infer<typeof actionSessionScopeSchema>;
export type ActionSessionAnchorId = SessionId;
export type ActionRecord = z.infer<typeof actionRecordSchema>;
export type ActionRecordPatch = z.infer<typeof actionRecordPatchSchema>;
export type ActionGoalId = GoalId;
export type ActionOpenQuestionId = OpenQuestionId;
