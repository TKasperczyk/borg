import { afterEach, describe, expect, it } from "vitest";

import { openDatabase, type SqliteDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID, createSessionId, createEntityId } from "../util/ids.js";

import { sessionMigrations } from "./migrations.js";
import { SessionsRepository } from "./repository.js";

function openRepo(clock = new ManualClock(1_000)): {
  db: SqliteDatabase;
  repo: SessionsRepository;
  clock: ManualClock;
} {
  const db = openDatabase(":memory:", { migrations: sessionMigrations });
  return {
    db,
    repo: new SessionsRepository({ db, clock }),
    clock,
  };
}

describe("SessionsRepository", () => {
  let db: SqliteDatabase | undefined;

  afterEach(() => {
    db?.close();
    db = undefined;
  });

  it("ensures, gets, and idempotently updates a session without resetting counters", () => {
    const harness = openRepo();
    db = harness.db;
    const audienceEntityId = createEntityId();

    const created = harness.repo.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      source_external_id: null,
      source_url: null,
      label: "demo (default)",
      audience_label: "alice",
      audience_entity_id: audienceEntityId,
      conversation_kind: "demo",
    });

    expect(created).toMatchObject({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      label: "demo (default)",
      audience_label: "alice",
      audience_entity_id: audienceEntityId,
      conversation_kind: "demo",
      created_at: 1_000,
      last_activity_at: 1_000,
      last_turn_id: null,
      message_count: 0,
      status: "active",
      privacy_level: "payload_off",
      participation_policy: "active",
      audience_role: "participant",
    });

    harness.repo.touch(DEFAULT_SESSION_ID, { at: 2_000, lastTurnId: "turn_1" });
    const updated = harness.repo.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      source_external_id: null,
      source_url: null,
      label: "demo renamed",
      audience_label: "alice",
      conversation_kind: "demo",
      last_activity_at: 1_500,
      privacy_level: "payload_on",
    });

    expect(updated).toMatchObject({
      session_id: DEFAULT_SESSION_ID,
      label: "demo renamed",
      last_activity_at: 2_000,
      last_turn_id: "turn_1",
      message_count: 1,
      privacy_level: "payload_on",
      participation_policy: "active",
      audience_role: "participant",
    });
    expect(harness.repo.get(DEFAULT_SESSION_ID)).toEqual(updated);
  });

  it("preserves audience role when ensuring an existing session without one", () => {
    const harness = openRepo();
    db = harness.db;

    harness.repo.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      label: "operator",
      audience_label: "Tom",
      conversation_kind: "demo",
      audience_role: "operator",
    });

    const ensured = harness.repo.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      label: "operator renamed",
      audience_label: "Tom",
      conversation_kind: "demo",
    });

    expect(ensured).toMatchObject({
      label: "operator renamed",
      audience_role: "operator",
    });

    const downgraded = harness.repo.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      label: "ordinary",
      audience_label: "alice",
      conversation_kind: "demo",
      audience_role: "participant",
    });

    expect(downgraded.audience_role).toBe("participant");
  });

  it("sets and reads participation policy", () => {
    const harness = openRepo();
    db = harness.db;

    harness.repo.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      label: "demo",
      audience_label: "alice",
      conversation_kind: "demo",
    });

    expect(harness.repo.setParticipationPolicy(DEFAULT_SESSION_ID, "observing")).toMatchObject({
      session_id: DEFAULT_SESSION_ID,
      participation_policy: "observing",
    });
    expect(harness.repo.get(DEFAULT_SESSION_ID)?.participation_policy).toBe("observing");
  });

  it("preserves participation policy when ensuring an existing session", () => {
    const harness = openRepo();
    db = harness.db;

    harness.repo.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      label: "demo",
      audience_label: "alice",
      conversation_kind: "demo",
    });
    harness.repo.setParticipationPolicy(DEFAULT_SESSION_ID, "muted");

    const ensured = harness.repo.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      label: "demo renamed",
      audience_label: "alice",
      conversation_kind: "demo",
    });

    expect(ensured).toMatchObject({
      label: "demo renamed",
      participation_policy: "muted",
    });
  });

  it("returns null when setting policy for a missing session", () => {
    const harness = openRepo();
    db = harness.db;

    expect(harness.repo.setParticipationPolicy(createSessionId(), "paused")).toBeNull();
  });

  it("touches activity, turn id, and message count", () => {
    const harness = openRepo();
    db = harness.db;

    harness.repo.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      label: "demo",
      audience_label: "alice",
      conversation_kind: "demo",
    });

    expect(
      harness.repo.touch(DEFAULT_SESSION_ID, {
        at: 3_000,
        lastTurnId: "turn_a",
        messageCountDelta: 2,
      }),
    ).toMatchObject({
      last_activity_at: 3_000,
      last_turn_id: "turn_a",
      message_count: 2,
    });
    expect(harness.repo.touch(createSessionId())).toBeNull();
  });

  it("preserves newer activity while applying touch side effects", () => {
    const harness = openRepo();
    db = harness.db;

    harness.repo.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      label: "demo",
      audience_label: "alice",
      conversation_kind: "demo",
    });

    harness.repo.touch(DEFAULT_SESSION_ID, { at: 3_000 });
    expect(
      harness.repo.touch(DEFAULT_SESSION_ID, {
        at: 2_000,
        lastTurnId: "turn_b",
        messageCountDelta: 2,
      }),
    ).toMatchObject({
      last_activity_at: 3_000,
      last_turn_id: "turn_b",
      message_count: 3,
    });
  });

  it("lists by activity, source type, and bounded recency order", () => {
    const harness = openRepo();
    db = harness.db;
    const slackSession = createSessionId();

    harness.repo.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      label: "demo",
      audience_label: "alice",
      conversation_kind: "demo",
      last_activity_at: 1_000,
    });
    harness.repo.ensure({
      session_id: slackSession,
      source_type: "slack",
      source_external_id: "thread-1",
      source_url: "https://slack.example/thread-1",
      label: "Slack #planning",
      audience_label: "#planning",
      conversation_kind: "thread",
      status: "idle",
      last_activity_at: 2_000,
    });

    expect(harness.repo.list().map((session) => session.session_id)).toEqual([
      slackSession,
      DEFAULT_SESSION_ID,
    ]);
    expect(harness.repo.list({ activeSince: 1_500 }).map((session) => session.session_id)).toEqual([
      slackSession,
    ]);
    expect(harness.repo.list({ sourceType: "demo" }).map((session) => session.session_id)).toEqual([
      DEFAULT_SESSION_ID,
    ]);
    expect(harness.repo.list({ limit: 1 })).toHaveLength(1);
  });
});
