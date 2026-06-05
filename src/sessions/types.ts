import { z } from "zod";

import { sessionIdSchema } from "../util/id-schemas.js";
import { entityIdHelpers, type EntityId } from "../util/ids.js";

export { sessionIdSchema };

export const SESSION_SOURCE_TYPES = ["demo", "slack", "discord", "imessage", "autonomy"] as const;
export const CONVERSATION_KINDS = ["dm", "channel", "thread", "demo"] as const;
export const SESSION_STATUSES = ["active", "idle", "archived"] as const;
export const SESSION_PRIVACY_LEVELS = ["payload_off", "payload_on"] as const;
export const SESSION_PARTICIPATION_POLICIES = ["active", "paused", "observing", "muted"] as const;
export const SESSION_AUDIENCE_ROLES = ["participant", "operator"] as const;

// source_type is an opaque routing/label key for the outbound connector registry, not a
// closed taxonomy: connectors register their own source_type and borg never branches on a
// specific value. Validate the SHAPE (a lowercase slug) rather than membership, so adding a
// connector never requires editing this file. SESSION_SOURCE_TYPES stays as the set of
// borg-known built-ins (autocomplete + internal reference), not an allow-list.
export const sessionSourceTypeSchema = z.string().regex(/^[a-z][a-z0-9_]*$/, {
  message: "source_type must be a lowercase slug matching /^[a-z][a-z0-9_]*$/",
});
export const conversationKindSchema = z.enum(CONVERSATION_KINDS);
export const sessionStatusSchema = z.enum(SESSION_STATUSES);
export const sessionPrivacyLevelSchema = z.enum(SESSION_PRIVACY_LEVELS);
export const sessionParticipationPolicySchema = z.enum(SESSION_PARTICIPATION_POLICIES);
export const sessionAudienceRoleSchema = z.enum(SESSION_AUDIENCE_ROLES);

export const sessionEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid session audience entity id",
  })
  .transform((value) => value as EntityId);

export const sessionRecordSchema = z.object({
  session_id: sessionIdSchema,
  source_type: sessionSourceTypeSchema,
  source_external_id: z.string().nullable(),
  source_url: z.string().nullable(),
  label: z.string(),
  audience_label: z.string(),
  audience_entity_id: sessionEntityIdSchema.nullable(),
  conversation_kind: conversationKindSchema,
  created_at: z.number().int().finite(),
  last_activity_at: z.number().int().finite(),
  last_turn_id: z.string().nullable(),
  message_count: z.number().int().nonnegative(),
  status: sessionStatusSchema,
  privacy_level: sessionPrivacyLevelSchema,
  participation_policy: sessionParticipationPolicySchema,
  audience_role: sessionAudienceRoleSchema,
});

export const sessionEnsureInputSchema = z.object({
  session_id: sessionIdSchema,
  source_type: sessionSourceTypeSchema,
  source_external_id: z.string().min(1).nullable().optional(),
  source_url: z.string().min(1).nullable().optional(),
  label: z.string().min(1),
  audience_label: z.string().min(1),
  audience_entity_id: sessionEntityIdSchema.nullable().optional(),
  conversation_kind: conversationKindSchema,
  created_at: z.number().int().finite().optional(),
  last_activity_at: z.number().int().finite().optional(),
  last_turn_id: z.string().min(1).nullable().optional(),
  status: sessionStatusSchema.optional(),
  privacy_level: sessionPrivacyLevelSchema.optional(),
  audience_role: sessionAudienceRoleSchema.optional(),
});

export const sessionTouchUpdateSchema = z.object({
  at: z.number().int().finite().optional(),
  lastTurnId: z.string().min(1).nullable().optional(),
  messageCountDelta: z.number().int().nonnegative().optional(),
});

export const sessionQueryOptionsSchema = z.object({
  activeSince: z.number().int().finite().optional(),
  sourceType: sessionSourceTypeSchema.optional(),
  status: sessionStatusSchema.optional(),
  excludeSessionId: sessionIdSchema.optional(),
});

export const sessionListOptionsSchema = sessionQueryOptionsSchema.extend({
  limit: z.number().int().positive().optional(),
});

// Known built-ins keep autocomplete; `& {}` lets any valid slug a connector registers be
// assignable without widening to bare `string`. source_type is an open routing key.
export type SessionSourceType = (typeof SESSION_SOURCE_TYPES)[number] | (string & {});
export type ConversationKind = z.infer<typeof conversationKindSchema>;
export type SessionStatus = z.infer<typeof sessionStatusSchema>;
export type SessionPrivacyLevel = z.infer<typeof sessionPrivacyLevelSchema>;
export type SessionParticipationPolicy = z.infer<typeof sessionParticipationPolicySchema>;
export type SessionAudienceRole = z.infer<typeof sessionAudienceRoleSchema>;
export type SessionRecord = z.infer<typeof sessionRecordSchema>;
export type SessionEnsureInput = z.infer<typeof sessionEnsureInputSchema>;
export type SessionTouchUpdate = z.infer<typeof sessionTouchUpdateSchema>;
export type SessionQueryOptions = z.infer<typeof sessionQueryOptionsSchema>;
export type SessionListOptions = z.infer<typeof sessionListOptionsSchema>;
