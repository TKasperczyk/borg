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
import { memoryDisclosureLabelSchema } from "../common/disclosure-label.js";

export const ACTION_STATES = [
  "considering",
  "committed_to_do",
  "scheduled",
  "completed",
  "not_done",
  "declined",
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
  | "declined_at"
  | "expired_at"
  | "archived_at"
  | "unknown_at";

export type ActionStateMetadata = {
  timestamp_field: ActionStateTimestampField;
  active: boolean;
  terminal: boolean;
  terminal_closure_guard: boolean;
};

export const ACTION_STATE_METADATA: Record<ActionState, ActionStateMetadata> = {
  considering: {
    timestamp_field: "considering_at",
    active: true,
    terminal: false,
    terminal_closure_guard: false,
  },
  committed_to_do: {
    timestamp_field: "committed_at",
    active: true,
    terminal: false,
    terminal_closure_guard: false,
  },
  scheduled: {
    timestamp_field: "scheduled_at",
    active: true,
    terminal: false,
    terminal_closure_guard: false,
  },
  completed: {
    timestamp_field: "completed_at",
    active: false,
    terminal: true,
    terminal_closure_guard: true,
  },
  not_done: {
    timestamp_field: "not_done_at",
    active: false,
    terminal: true,
    terminal_closure_guard: true,
  },
  declined: {
    timestamp_field: "declined_at",
    active: false,
    terminal: true,
    terminal_closure_guard: true,
  },
  expired: {
    timestamp_field: "expired_at",
    active: false,
    terminal: false,
    terminal_closure_guard: true,
  },
  archived: {
    timestamp_field: "archived_at",
    active: false,
    terminal: false,
    terminal_closure_guard: true,
  },
  unknown: {
    timestamp_field: "unknown_at",
    active: true,
    terminal: false,
    terminal_closure_guard: false,
  },
};

// Active means not terminal/expired/archived; pressure for canonicalization/action-bloat observability.
export const ACTIVE_ACTION_STATES: readonly ActionState[] = ACTION_STATES.filter(
  (state) => ACTION_STATE_METADATA[state].active,
);

// terminal is the final outcome vocabulary emitted by action extraction; terminal_closure_guard
// also includes lifecycle retirements that must not be overwritten by later closure evidence.
export function isActionTerminalClosureGuardState(state: ActionState): boolean {
  return ACTION_STATE_METADATA[state].terminal_closure_guard;
}

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

export const actionMatchRejectionSchema = z.enum([
  "not_offered",
  "missing",
  "terminal",
  "actor_mismatch",
]);

export const actionSelectionAuditSchema = z
  .object({
    turn_id: z.string().nullable(),
    candidate_index: z.number().int().nonnegative(),
    candidate_ids: z.array(actionIdSchema),
    candidate_omitted_count: z.number().int().nonnegative(),
    chosen_action_id: actionIdSchema.nullable(),
    rejection_reason: actionMatchRejectionSchema.nullable(),
    supplied_goal_ids: z.array(goalIdSchema),
    chosen_goal_id: goalIdSchema.nullable(),
    goal_rejection_reason: z.literal("not_offered").nullable(),
    referenced_action_ids: z.array(actionIdSchema).optional(),
    applied_reference_ids: z.array(actionIdSchema).optional(),
    shared_state_entry_id: actionCanonicalizedByArtifactEntryIdSchema.optional(),
  })
  .strict();

export const actionRecordUpdateSchema = z
  .object({
    id: z.number().int().positive(),
    action_id: actionIdSchema,
    text: z.string().min(1),
    from_state: actionStateSchema,
    to_state: actionStateSchema,
    turn_id: z.string().nullable(),
    source_stream_entry_ids: z.array(actionStreamEntryIdSchema),
    timestamp: z.number().finite(),
    selection_audit: actionSelectionAuditSchema,
    disclosure_label: memoryDisclosureLabelSchema,
  })
  .strict();

export type ActionMatchRejection = z.infer<typeof actionMatchRejectionSchema>;
export type ActionSelectionAudit = z.infer<typeof actionSelectionAuditSchema>;
export type ActionRecordUpdate = z.infer<typeof actionRecordUpdateSchema>;
export type ActionUpdateHistory = { updates: ActionRecordUpdate[]; total: number };

export const actionExtractionDecisionSchema = z
  .object({
    turn_id: z.string().nullable(),
    candidate_ids: z.array(actionIdSchema),
    candidate_omitted_count: z.number().int().nonnegative(),
    referenced_action_ids: z.array(actionIdSchema),
    applied_reference_ids: z.array(actionIdSchema),
    ignored_reference_ids: z.array(actionIdSchema),
    decisions: z.array(actionSelectionAuditSchema),
  })
  .strict();
export type ActionExtractionDecision = z.infer<typeof actionExtractionDecisionSchema>;

const actionRecordShape = z.object({
  id: actionIdSchema,
  description: z.string().min(1),
  actor: actionActorSchema,
  audience_entity_id: actionEntityIdSchema.nullable(),
  goal_id: goalIdSchema.nullable().default(null),
  // Null/absent marks pre-model-selection links; migrations never rewrite their goal_id.
  goal_link_basis: z.enum(["extractor", "unlinked"]).nullable().optional(),
  selection_audit: actionSelectionAuditSchema.nullable().optional(),
  disclosure_label: memoryDisclosureLabelSchema.nullable().optional(),
  origin_stream_entry_ids: z.array(actionStreamEntryIdSchema).nullable().optional(),
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
  declined_at: z.number().finite().nullable().default(null),
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
    selection_audit: true,
    origin_stream_entry_ids: true,
  })
  .partial()
  // A missing patch field must remain absent so lifecycle transitions stamp it.
  .extend({ declined_at: actionRecordShape.shape.declined_at.removeDefault().optional() })
  .strict();

export type ActionActor = z.infer<typeof actionActorSchema>;
export type ActionSessionScope = z.infer<typeof actionSessionScopeSchema>;
export type ActionSessionAnchorId = SessionId;
export type ActionRecord = z.infer<typeof actionRecordSchema>;
export type ActionRecordPatch = z.infer<typeof actionRecordPatchSchema>;
export type ActionGoalId = GoalId;
export type ActionOpenQuestionId = OpenQuestionId;
