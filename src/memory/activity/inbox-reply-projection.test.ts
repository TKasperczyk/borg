import { describe, expect, it } from "vitest";
import type { SessionRecord } from "../../sessions/index.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../../util/ids.js";
import {
  buildInboxReplyActivityProjection,
  isEnsurableSessionRecord,
  sessionEnsureInputFromRecord,
} from "./inbox-reply-projection.js";

function sessionRecord(overrides: Partial<SessionRecord> = {}): SessionRecord {
  return {
    session_id: createSessionId(),
    source_type: "teams_inbox",
    source_external_id: "conversation",
    source_url: null,
    label: "AI Ninjas",
    audience_label: "AI Ninjas",
    audience_entity_id: createEntityId(),
    conversation_kind: "channel",
    created_at: 900,
    last_activity_at: 900,
    last_turn_id: null,
    message_count: 3,
    status: "active",
    privacy_level: "payload_on",
    participation_policy: "active",
    audience_role: "participant",
    ...overrides,
  };
}

describe("buildInboxReplyActivityProjection", () => {
  it("projects the owner reply with self, batch senders and the session audience", () => {
    const session = sessionRecord();
    const self = createEntityId();
    const sender = createEntityId();
    const terminal = { id: createStreamEntryId(), sessionId: session.session_id, timestamp: 1_000 };

    const projection = buildInboxReplyActivityProjection({
      session,
      selfEntityId: self,
      terminal,
      senderEntityIds: [sender, sender],
    });

    expect(projection).toEqual({
      kind: "project",
      input: {
        session: sessionEnsureInputFromRecord(session),
        borgReplied: {
          kind: "borg_replied",
          occurredAt: 1_000,
          sessionId: session.session_id,
          speakerEntityId: self,
          actorEntityId: self,
          audienceEntityId: session.audience_entity_id,
          participantEntityIds: [self, sender, session.audience_entity_id],
          sourceStreamEntryIds: [terminal.id],
        },
        touch: { at: 1_000, messageCountDelta: 1 },
      },
    });
    expect(sessionEnsureInputFromRecord(session)).not.toHaveProperty("message_count");
    expect(sessionEnsureInputFromRecord(session)).not.toHaveProperty("participation_policy");
  });

  it("dedupes a DM sender who is also the audience", () => {
    const session = sessionRecord({ conversation_kind: "dm" });
    const self = createEntityId();
    const projection = buildInboxReplyActivityProjection({
      session,
      selfEntityId: self,
      terminal: { id: createStreamEntryId(), sessionId: session.session_id, timestamp: 5 },
      senderEntityIds: [session.audience_entity_id!],
    });
    expect(projection.kind).toBe("project");
    if (projection.kind === "project") {
      expect(projection.input.borgReplied.participantEntityIds).toEqual([
        self,
        session.audience_entity_id,
      ]);
    }
  });

  it("skips with a reason instead of building an invalid projection", () => {
    const terminal = { id: createStreamEntryId(), sessionId: createSessionId(), timestamp: 5 };
    const self = createEntityId();
    expect(
      buildInboxReplyActivityProjection({
        session: null,
        selfEntityId: self,
        terminal,
        senderEntityIds: [],
      }),
    ).toEqual({ kind: "skip", reason: "session_missing" });
    expect(
      buildInboxReplyActivityProjection({
        session: sessionRecord(),
        selfEntityId: null,
        terminal,
        senderEntityIds: [],
      }),
    ).toEqual({ kind: "skip", reason: "self_missing" });
    expect(
      buildInboxReplyActivityProjection({
        session: sessionRecord({ audience_entity_id: null }),
        selfEntityId: self,
        terminal,
        senderEntityIds: [],
      }),
    ).toEqual({ kind: "skip", reason: "audience_missing" });
    expect(
      buildInboxReplyActivityProjection({
        session: sessionRecord({ label: "" }),
        selfEntityId: self,
        terminal,
        senderEntityIds: [],
      }),
    ).toEqual({ kind: "skip", reason: "session_record_incomplete" });
  });

  it("treats legacy empty strings as unensurable and explicit nulls as fine", () => {
    expect(isEnsurableSessionRecord(sessionRecord())).toBe(true);
    expect(isEnsurableSessionRecord(sessionRecord({ source_url: null, last_turn_id: null }))).toBe(
      true,
    );
    expect(isEnsurableSessionRecord(sessionRecord({ source_external_id: "" }))).toBe(false);
    expect(isEnsurableSessionRecord(sessionRecord({ audience_label: "" }))).toBe(false);
    expect(isEnsurableSessionRecord(sessionRecord({ last_turn_id: "" }))).toBe(false);
  });
});
