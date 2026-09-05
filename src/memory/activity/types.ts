import { z } from "zod";

import type { ConversationKind } from "../../sessions/types.js";
import {
  activityEventIdHelpers,
  entityIdHelpers,
  isSessionId,
  streamEntryIdHelpers,
  type ActivityEventId,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";

export const ACTIVITY_EVENT_KINDS = ["user_contact", "borg_replied", "turn_completed"] as const;
export const ACTIVITY_EVENT_STATUSES = ["active", "inactive"] as const;

export const activityEventIdSchema = z
  .string()
  .refine((value) => activityEventIdHelpers.is(value), {
    message: "Invalid activity event id",
  })
  .transform((value) => value as ActivityEventId);

export const activitySessionIdSchema = z
  .string()
  .refine((value) => isSessionId(value), {
    message: "Invalid activity session id",
  })
  .transform((value) => value as SessionId);

export const activityEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid activity entity id",
  })
  .transform((value) => value as EntityId);

export const activityStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid activity stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const activityEventKindSchema = z.enum(ACTIVITY_EVENT_KINDS);
export const activityEventStatusSchema = z.enum(ACTIVITY_EVENT_STATUSES);

export const activityEventSchema = z
  .object({
    id: activityEventIdSchema,
    kind: activityEventKindSchema,
    occurred_at: z.number().int().finite(),
    session_id: activitySessionIdSchema,
    turn_id: z.string().min(1).nullable(),
    speaker_entity_id: activityEntityIdSchema.nullable(),
    actor_entity_id: activityEntityIdSchema.nullable(),
    audience_entity_id: activityEntityIdSchema.nullable(),
    participant_entity_ids: z.array(activityEntityIdSchema),
    source_stream_entry_ids: z.array(activityStreamEntryIdSchema).min(1),
    status: activityEventStatusSchema,
    created_at: z.number().int().finite(),
    updated_at: z.number().int().finite(),
  })
  .strict();

export type ActivityEventKind = z.infer<typeof activityEventKindSchema>;
export type ActivityEventStatus = z.infer<typeof activityEventStatusSchema>;
export type ActivityEvent = z.infer<typeof activityEventSchema>;

export type ActivityEventRecordInput = {
  id?: ActivityEventId;
  kind: ActivityEventKind;
  occurredAt: number;
  sessionId: SessionId;
  turnId?: string | null;
  speakerEntityId?: EntityId | null;
  actorEntityId?: EntityId | null;
  audienceEntityId?: EntityId | null;
  participantEntityIds?: readonly EntityId[];
  sourceStreamEntryIds: readonly StreamEntryId[];
  status?: ActivityEventStatus;
  now?: number;
};

export type ActivityVisibleSessionEvent = {
  kind: "user_contact" | "borg_replied";
  occurredAt: number;
  sessionId: SessionId;
  audienceEntityId: EntityId;
  conversationKind: ConversationKind;
  conversationName: string;
  participantLabel: string;
  sourceStreamEntryIds: readonly StreamEntryId[];
};
