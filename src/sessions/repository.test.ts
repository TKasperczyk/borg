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

  it("gets a deduplicated session batch in requested order and omits missing rows", () => {
    const harness = openRepo();
    db = harness.db;
    const firstId = createSessionId();
    const secondId = createSessionId();

    for (const [sessionId, label] of [
      [firstId, "first"],
      [secondId, "second"],
    ] as const) {
      harness.repo.ensure({
        session_id: sessionId,
        source_type: "demo",
        label,
        audience_label: label,
        conversation_kind: "demo",
      });
    }

    expect(
      harness.repo
        .getMany([secondId, createSessionId(), firstId, secondId])
        .map((row) => row.session_id),
    ).toEqual([secondId, firstId]);
  });

  it("migration 2 rebuilds a legacy closed-CHECK sessions table to accept connector source types", () => {
    const db = openDatabase(":memory:", { migrations: [] });
    // Simulate a legacy DB: the original baseline table WITH the closed source_type CHECK.
    db.exec(`
      CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY,
        source_type TEXT NOT NULL CHECK (
          source_type IN ('demo', 'slack', 'discord', 'imessage', 'autonomy')
        ),
        source_external_id TEXT, source_url TEXT, label TEXT NOT NULL, audience_label TEXT NOT NULL,
        audience_entity_id TEXT,
        conversation_kind TEXT NOT NULL CHECK (conversation_kind IN ('dm','channel','thread','demo')),
        created_at INTEGER NOT NULL, last_activity_at INTEGER NOT NULL, last_turn_id TEXT,
        message_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL CHECK (status IN ('active','idle','archived')),
        privacy_level TEXT NOT NULL DEFAULT 'payload_off' CHECK (privacy_level IN ('payload_off','payload_on')),
        participation_policy TEXT NOT NULL DEFAULT 'active' CHECK (participation_policy IN ('active','paused','observing','muted')),
        audience_role TEXT NOT NULL DEFAULT 'participant' CHECK (audience_role IN ('participant','operator'))
      );
      INSERT INTO sessions (session_id, source_type, label, audience_label, conversation_kind, created_at, last_activity_at, status)
        VALUES ('sess_legacy', 'demo', 'legacy', 'a', 'demo', 1, 1, 'active');
    `);
    // Before: the closed CHECK rejects a connector source type.
    expect(() =>
      db.exec(
        "INSERT INTO sessions (session_id, source_type, label, audience_label, conversation_kind, created_at, last_activity_at, status) VALUES ('x','botarena','l','a','thread',1,1,'active')",
      ),
    ).toThrow();

    const migration2 = sessionMigrations.find((m) => m.id === 2);
    expect(migration2).toBeDefined();
    (migration2?.up as (database: SqliteDatabase) => void)(db);

    // After: connector source types are accepted and legacy rows are preserved.
    db.exec(
      "INSERT INTO sessions (session_id, source_type, label, audience_label, conversation_kind, created_at, last_activity_at, status) VALUES ('sess_ba','botarena','l','a','thread',1,1,'active')",
    );
    const legacy = db
      .prepare("SELECT source_type FROM sessions WHERE session_id = 'sess_legacy'")
      .get() as { source_type: string } | undefined;
    const ba = db.prepare("SELECT source_type FROM sessions WHERE session_id = 'sess_ba'").get() as
      | { source_type: string }
      | undefined;
    expect(legacy?.source_type).toBe("demo");
    expect(ba?.source_type).toBe("botarena");
    db.close();
  });

  it("persists a session with a connector-defined source_type slug (open source_type)", () => {
    const harness = openRepo();
    db = harness.db;
    const sessionId = createSessionId();

    const created = harness.repo.ensure({
      session_id: sessionId,
      source_type: "botarena",
      source_external_id: "thread-uuid",
      source_url: null,
      label: "Z zycia bota",
      audience_label: "botarena_thread:thread-uuid",
      conversation_kind: "thread",
    });

    expect(created.source_type).toBe("botarena");
    expect(harness.repo.get(sessionId)?.source_type).toBe("botarena");
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
    const archivedSession = createSessionId();

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
    harness.repo.ensure({
      session_id: archivedSession,
      source_type: "demo",
      label: "archived",
      audience_label: "old",
      conversation_kind: "demo",
      status: "archived",
      last_activity_at: 3_000,
    });

    expect(harness.repo.list().map((session) => session.session_id)).toEqual([
      archivedSession,
      slackSession,
      DEFAULT_SESSION_ID,
    ]);
    expect(harness.repo.list({ activeSince: 1_500 }).map((session) => session.session_id)).toEqual([
      archivedSession,
      slackSession,
    ]);
    expect(harness.repo.list({ sourceType: "demo" }).map((session) => session.session_id)).toEqual([
      archivedSession,
      DEFAULT_SESSION_ID,
    ]);
    expect(harness.repo.list({ status: "active" }).map((session) => session.session_id)).toEqual([
      DEFAULT_SESSION_ID,
    ]);
    expect(
      harness.repo.list({ excludeSessionId: archivedSession }).map((session) => session.session_id),
    ).toEqual([slackSession, DEFAULT_SESSION_ID]);
    expect(harness.repo.list({ limit: 1 })).toHaveLength(1);
    expect(harness.repo.count({ status: "active" })).toBe(1);
    expect(harness.repo.count({ excludeSessionId: DEFAULT_SESSION_ID })).toBe(2);
  });
});
