import type { SessionEnsureInput, SessionRecord } from "../../sessions/index.js";
import { dedupePreservingOrder } from "../../util/collections.js";
import type { EntityId, SessionId, StreamEntryId } from "../../util/ids.js";
import type { ActivityEventRecordInput } from "./types.js";

// The memory owner's own reply on an inbox session, projected as a borg_replied activity event
// exactly like /memory/append-turn's reply-only mode: self speaks and acts, the session's audience
// is the audience, the batch senders are the other participants, the terminal entry is the source.
// Shared by the live inbox runner and the reconcile/backfill pass so the two cannot drift.

export type InboxReplyActivitySkipReason =
  | "session_missing"
  | "self_missing"
  | "audience_missing"
  | "session_record_incomplete";

export type InboxReplyActivityProjectionInput = {
  session: SessionEnsureInput;
  borgReplied: ActivityEventRecordInput & { kind: "borg_replied" };
  touch: { at: number; messageCountDelta: number };
};

export type InboxReplyActivityProjection =
  | { kind: "project"; input: InboxReplyActivityProjectionInput }
  | { kind: "skip"; reason: InboxReplyActivitySkipReason };

export function buildInboxReplyActivityProjection(input: {
  session: SessionRecord | null;
  selfEntityId: EntityId | null;
  terminal: { id: StreamEntryId; sessionId: SessionId; timestamp: number };
  senderEntityIds: readonly EntityId[];
}): InboxReplyActivityProjection {
  if (input.session === null) {
    return { kind: "skip", reason: "session_missing" };
  }
  if (input.selfEntityId === null) {
    return { kind: "skip", reason: "self_missing" };
  }
  const audienceEntityId = input.session.audience_entity_id ?? null;
  if (audienceEntityId === null) {
    return { kind: "skip", reason: "audience_missing" };
  }
  if (!isEnsurableSessionRecord(input.session)) {
    return { kind: "skip", reason: "session_record_incomplete" };
  }
  return {
    kind: "project",
    input: {
      session: sessionEnsureInputFromRecord(input.session),
      borgReplied: {
        kind: "borg_replied",
        occurredAt: input.terminal.timestamp,
        sessionId: input.terminal.sessionId,
        speakerEntityId: input.selfEntityId,
        actorEntityId: input.selfEntityId,
        audienceEntityId,
        participantEntityIds: dedupePreservingOrder([
          input.selfEntityId,
          ...input.senderEntityIds,
          audienceEntityId,
        ]),
        sourceStreamEntryIds: [input.terminal.id],
      },
      touch: { at: input.terminal.timestamp, messageCountDelta: 1 },
    },
  };
}

// sessionEnsureInputSchema requires non-empty strings where a stored record may still hold
// legacy empty values; such a record is left alone instead of failing inside the projection.
export function isEnsurableSessionRecord(record: SessionRecord): boolean {
  const optionalNonEmpty = (value: string | null | undefined) =>
    value === null || value === undefined || value.length > 0;
  return (
    record.label.length > 0 &&
    record.audience_label.length > 0 &&
    optionalNonEmpty(record.source_external_id) &&
    optionalNonEmpty(record.source_url) &&
    optionalNonEmpty(record.last_turn_id)
  );
}

export function sessionEnsureInputFromRecord(record: SessionRecord): SessionEnsureInput {
  return {
    session_id: record.session_id,
    source_type: record.source_type,
    source_external_id: record.source_external_id,
    source_url: record.source_url,
    label: record.label,
    audience_label: record.audience_label,
    audience_entity_id: record.audience_entity_id,
    conversation_kind: record.conversation_kind,
    created_at: record.created_at,
    last_activity_at: record.last_activity_at,
    last_turn_id: record.last_turn_id,
    status: record.status,
    privacy_level: record.privacy_level,
    audience_role: record.audience_role,
  };
}
