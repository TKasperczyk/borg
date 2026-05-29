import { describe, expect, it } from "vitest";

import type { SessionRecord } from "../../../sessions/index.js";
import { DEFAULT_SESSION_ID, createSessionId } from "../../../util/ids.js";

import { buildOperatorSessionSnapshot } from "./session-snapshot.js";

const NOW_MS = 1_700_000_000_000;

function makeSession(input: Partial<SessionRecord> = {}): SessionRecord {
  const sessionId = input.session_id ?? createSessionId();

  return {
    session_id: sessionId,
    source_type: "demo",
    source_external_id: null,
    source_url: null,
    label: `session ${sessionId}`,
    audience_label: "Alice",
    audience_entity_id: null,
    conversation_kind: "demo",
    created_at: NOW_MS - 10_000,
    last_activity_at: NOW_MS - 5 * 60_000,
    last_turn_id: "turn_last",
    message_count: 1,
    status: "active",
    privacy_level: "payload_off",
    participation_policy: "active",
    audience_role: "participant",
    ...input,
  };
}

describe("buildOperatorSessionSnapshot", () => {
  it("aliases active other sessions in recency order and filters current or non-active sessions", () => {
    const firstSessionId = createSessionId();
    const secondSessionId = createSessionId();
    const snapshot = buildOperatorSessionSnapshot({
      currentSessionId: DEFAULT_SESSION_ID,
      nowMs: NOW_MS,
      cap: 2,
      sessions: [
        makeSession({
          session_id: DEFAULT_SESSION_ID,
          audience_label: "Current",
          last_activity_at: NOW_MS,
        }),
        makeSession({
          session_id: createSessionId(),
          audience_label: "Archived",
          status: "archived",
          last_activity_at: NOW_MS - 10 * 60_000,
        }),
        makeSession({
          session_id: secondSessionId,
          audience_label: "Bob",
          conversation_kind: "channel",
          participation_policy: "observing",
          last_activity_at: NOW_MS - 2 * 60 * 60_000,
          last_turn_id: null,
          message_count: 7,
        }),
        makeSession({
          session_id: firstSessionId,
          audience_label: "Alice",
          conversation_kind: "dm",
          participation_policy: "paused",
          last_activity_at: NOW_MS - 5 * 60_000,
          message_count: 42,
        }),
      ],
    });

    expect(snapshot).toEqual({
      generated_at: new Date(NOW_MS).toISOString(),
      sessions: [
        {
          alias: "session_1",
          session_id: firstSessionId,
          outbound_targetable: false,
          audience_label: "Alice",
          conversation_kind: "dm",
          participation_policy: "paused",
          last_activity: "5m ago",
          message_count: 42,
          recent_state: "last_turn_available",
        },
        {
          alias: "session_2",
          session_id: secondSessionId,
          outbound_targetable: false,
          audience_label: "Bob",
          conversation_kind: "channel",
          participation_policy: "observing",
          last_activity: "2h ago",
          message_count: 7,
          recent_state: "no_recent_turn",
        },
      ],
    });
  });

  it("caps visible sessions and reports the omitted tail", () => {
    const sessions = Array.from({ length: 20 }, (_, index) =>
      makeSession({
        session_id: createSessionId(),
        audience_label: `Person ${index + 1}`,
        last_activity_at: NOW_MS - index * 60_000,
      }),
    );

    const snapshot = buildOperatorSessionSnapshot({
      sessions,
      currentSessionId: DEFAULT_SESSION_ID,
      nowMs: NOW_MS,
      cap: 12,
    });

    expect(snapshot?.sessions).toHaveLength(12);
    expect(snapshot?.sessions[0]?.alias).toBe("session_1");
    expect(snapshot?.sessions[11]?.alias).toBe("session_12");
    expect(snapshot?.omitted_count).toBe(8);
  });

  it("uses an explicit total count when the input rows are already limited", () => {
    const sessions = Array.from({ length: 12 }, () =>
      makeSession({
        session_id: createSessionId(),
      }),
    );

    const snapshot = buildOperatorSessionSnapshot({
      sessions,
      currentSessionId: DEFAULT_SESSION_ID,
      nowMs: NOW_MS,
      cap: 12,
      totalActiveOtherSessionCount: 20,
    });

    expect(snapshot?.sessions).toHaveLength(12);
    expect(snapshot?.omitted_count).toBe(8);
  });

  it("returns an empty live snapshot when no other active sessions exist", () => {
    const snapshot = buildOperatorSessionSnapshot({
      sessions: [],
      currentSessionId: DEFAULT_SESSION_ID,
      nowMs: NOW_MS,
    });

    expect(snapshot).toEqual({
      generated_at: new Date(NOW_MS).toISOString(),
      sessions: [],
    });
  });

  it("returns null for an invalid generation time", () => {
    const snapshot = buildOperatorSessionSnapshot({
      sessions: [makeSession()],
      currentSessionId: DEFAULT_SESSION_ID,
      nowMs: Number.NaN,
    });

    expect(snapshot).toBeNull();
  });
});
