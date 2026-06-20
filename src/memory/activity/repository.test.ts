import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { commitmentMigrations } from "../commitments/index.js";
import { sessionMigrations, SessionsRepository } from "../../sessions/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  createEntityId,
  createSessionId,
  createStreamEntryId,
  type ActivityEventId,
} from "../../util/ids.js";
import { activityMigrations } from "./migrations.js";
import { ActivityRepository } from "./repository.js";

const tempDirs: string[] = [];

afterEach(() => {
  for (const tempDir of tempDirs.splice(0)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

describe("ActivityRepository", () => {
  it("round-trips activity events and keeps source/kind writes idempotent", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-activity-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "activity.db"), {
      migrations: activityMigrations,
    });
    const repository = new ActivityRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const sessionId = createSessionId();
    const speakerEntityId = createEntityId();
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();

    const created = repository.record({
      kind: "user_contact",
      occurredAt: 1_000,
      sessionId,
      turnId: "turn-1",
      speakerEntityId,
      actorEntityId: speakerEntityId,
      audienceEntityId,
      participantEntityIds: [speakerEntityId, audienceEntityId, speakerEntityId],
      sourceStreamEntryIds: [sourceStreamEntryId],
    });
    const updated = repository.record({
      kind: "user_contact",
      occurredAt: 1_500,
      sessionId,
      turnId: "turn-1",
      speakerEntityId,
      actorEntityId: speakerEntityId,
      audienceEntityId,
      participantEntityIds: [audienceEntityId, speakerEntityId],
      sourceStreamEntryIds: [sourceStreamEntryId],
      status: "inactive",
      now: 3_000,
    });

    expect(repository.get(created.id as ActivityEventId)).toEqual(updated);
    expect(updated).toMatchObject({
      id: created.id,
      kind: "user_contact",
      occurred_at: 1_500,
      session_id: sessionId,
      turn_id: "turn-1",
      speaker_entity_id: speakerEntityId,
      actor_entity_id: speakerEntityId,
      audience_entity_id: audienceEntityId,
      participant_entity_ids: [audienceEntityId, speakerEntityId],
      source_stream_entry_ids: [sourceStreamEntryId],
      status: "inactive",
      created_at: 2_000,
      updated_at: 3_000,
    });
    expect(db.prepare("SELECT COUNT(*) AS count FROM activity_events").get()).toEqual({
      count: 1,
    });

    db.close();
  });

  it("lists global recent activity with session source metadata for autobiographical recall", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-activity-autobiographical-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "activity.db"), {
      migrations: composeMigrations(sessionMigrations, activityMigrations, commitmentMigrations),
    });
    const sessions = new SessionsRepository({
      db,
      clock: new FixedClock(5_000),
    });
    const repository = new ActivityRepository({
      db,
      clock: new FixedClock(5_000),
    });
    const arenaSessionId = createSessionId();
    const oldSessionId = createSessionId();
    const audienceEntityId = createEntityId();
    const recentSourceStreamEntryId = createStreamEntryId();

    sessions.ensure({
      session_id: arenaSessionId,
      source_type: "botarena",
      label: "Arena thread",
      audience_label: "Arena",
      audience_entity_id: audienceEntityId,
      conversation_kind: "thread",
      audience_role: "participant",
      status: "active",
      created_at: 1_000,
      last_activity_at: 4_000,
    });
    sessions.ensure({
      session_id: oldSessionId,
      source_type: "demo",
      label: "Old thread",
      audience_label: "Old",
      conversation_kind: "demo",
      audience_role: "operator",
      status: "active",
      created_at: 1_000,
      last_activity_at: 1_500,
    });
    const recent = repository.record({
      kind: "turn_completed",
      occurredAt: 4_000,
      sessionId: arenaSessionId,
      turnId: "turn-arena",
      audienceEntityId,
      participantEntityIds: [audienceEntityId],
      sourceStreamEntryIds: [recentSourceStreamEntryId],
    });
    repository.record({
      kind: "turn_completed",
      occurredAt: 1_500,
      sessionId: oldSessionId,
      turnId: "turn-old",
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    const rows = repository.listRecentGlobalEvents({
      sinceMs: 3_000,
      untilMs: 4_500,
      limit: 10,
    });

    expect(rows).toEqual([
      expect.objectContaining({
        id: recent.id,
        kind: "turn_completed",
        occurredAt: 4_000,
        sessionId: arenaSessionId,
        sessionSourceType: "botarena",
        sessionAudienceRole: "participant",
        sessionLabel: "Arena thread",
        participantLabel: "Arena",
        audienceEntityId,
        participantEntityIds: [audienceEntityId],
        sourceStreamEntryIds: [recentSourceStreamEntryId],
      }),
    ]);

    db.close();
  });

  it("aggregates daily density for other active sessions and returns true latest activity", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-activity-density-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "activity.db"), {
      migrations: composeMigrations(sessionMigrations, activityMigrations, commitmentMigrations),
    });
    const sessions = new SessionsRepository({
      db,
      clock: new FixedClock(10_000),
    });
    const repository = new ActivityRepository({
      db,
      clock: new FixedClock(10_000),
    });
    const currentSessionId = createSessionId();
    const arenaSessionId = createSessionId();
    const autonomousSessionId = createSessionId();
    const archivedSessionId = createSessionId();
    const audienceEntityId = createEntityId();
    const autonomousAudienceEntityId = createEntityId();
    const firstAt = Date.UTC(2026, 5, 15, 10, 0, 0);
    const secondAt = Date.UTC(2026, 5, 15, 11, 30, 0);
    const latestAt = Date.UTC(2026, 5, 15, 12, 0, 0);

    sessions.ensure({
      session_id: currentSessionId,
      source_type: "demo",
      label: "Current",
      audience_label: "Tom",
      conversation_kind: "demo",
      audience_role: "operator",
      status: "active",
      created_at: firstAt,
      last_activity_at: latestAt,
    });
    sessions.ensure({
      session_id: arenaSessionId,
      source_type: "botarena",
      label: "Arena thread",
      audience_label: "BotArena group",
      audience_entity_id: audienceEntityId,
      conversation_kind: "thread",
      audience_role: "participant",
      status: "active",
      created_at: firstAt,
      last_activity_at: latestAt,
    });
    sessions.ensure({
      session_id: autonomousSessionId,
      source_type: "daemon",
      label: "Autonomous tick",
      audience_label: "Alice",
      audience_entity_id: autonomousAudienceEntityId,
      conversation_kind: "demo",
      audience_role: "participant",
      status: "active",
      created_at: firstAt,
      last_activity_at: latestAt,
    });
    sessions.ensure({
      session_id: archivedSessionId,
      source_type: "demo",
      label: "Archived",
      audience_label: "Archived",
      conversation_kind: "demo",
      audience_role: "participant",
      status: "archived",
      created_at: firstAt,
      last_activity_at: latestAt,
    });

    repository.record({
      kind: "user_contact",
      occurredAt: firstAt,
      sessionId: arenaSessionId,
      turnId: "arena-1",
      audienceEntityId,
      sourceStreamEntryIds: [createStreamEntryId()],
    });
    repository.record({
      kind: "borg_replied",
      occurredAt: secondAt,
      sessionId: arenaSessionId,
      turnId: "arena-1",
      audienceEntityId,
      sourceStreamEntryIds: [createStreamEntryId()],
    });
    repository.record({
      kind: "turn_completed",
      occurredAt: latestAt,
      sessionId: arenaSessionId,
      turnId: "arena-1",
      audienceEntityId,
      sourceStreamEntryIds: [createStreamEntryId()],
    });
    repository.record({
      kind: "turn_completed",
      occurredAt: latestAt - 1,
      sessionId: autonomousSessionId,
      turnId: "autonomous-1",
      audienceEntityId: autonomousAudienceEntityId,
      sourceStreamEntryIds: [createStreamEntryId()],
    });
    repository.record({
      kind: "turn_completed",
      occurredAt: latestAt + 1,
      sessionId: currentSessionId,
      turnId: "current-1",
      sourceStreamEntryIds: [createStreamEntryId()],
    });
    repository.record({
      kind: "turn_completed",
      occurredAt: latestAt + 2,
      sessionId: archivedSessionId,
      turnId: "archived-1",
      sourceStreamEntryIds: [createStreamEntryId()],
    });
    repository.record({
      kind: "turn_completed",
      occurredAt: latestAt + 3,
      sessionId: arenaSessionId,
      turnId: "arena-inactive",
      audienceEntityId,
      status: "inactive",
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    expect(
      repository.getMostRecentOtherActiveSessionEventOccurredAt({
        currentSessionId,
        sinceMs: firstAt - 1,
      }),
    ).toBe(latestAt);
    expect(
      repository
        .listRecentOtherActiveSessionEvents({
          currentSessionId,
          sinceMs: firstAt - 1,
          limit: 10,
        })
        .map((row) => row.sessionId),
    ).toEqual([arenaSessionId, arenaSessionId, arenaSessionId]);
    expect(
      repository.listDailyOtherActiveSessionDensity({
        currentSessionId,
        sinceMs: firstAt - 1,
        untilMs: latestAt + 10,
        limit: 10,
      }),
    ).toEqual([
      {
        dayKey: "2026-06-15",
        dayStartMs: Date.UTC(2026, 5, 15),
        sessionId: arenaSessionId,
        sessionLabel: "Arena thread",
        audienceLabel: "BotArena group",
        audienceEntityId,
        eventCount: 3,
        conversationTurnCount: 1,
        kindCounts: {
          userContact: 1,
          borgReplied: 1,
          turnCompleted: 1,
        },
        firstOccurredAt: firstAt,
        lastOccurredAt: latestAt,
      },
    ]);

    db.close();
  });

  it("counts full-window conversational turns inside one capped density day", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-activity-density-flood-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "activity.db"), {
      migrations: composeMigrations(sessionMigrations, activityMigrations, commitmentMigrations),
    });
    const sessions = new SessionsRepository({
      db,
      clock: new FixedClock(10_000),
    });
    const repository = new ActivityRepository({
      db,
      clock: new FixedClock(10_000),
    });
    const currentSessionId = createSessionId();
    const arenaSessionId = createSessionId();
    const audienceEntityId = createEntityId();
    const firstAt = Date.UTC(2026, 5, 15, 9, 0, 0);

    sessions.ensure({
      session_id: currentSessionId,
      source_type: "demo",
      label: "Current",
      audience_label: "Tom",
      conversation_kind: "demo",
      audience_role: "operator",
      status: "active",
      created_at: firstAt,
      last_activity_at: firstAt,
    });
    sessions.ensure({
      session_id: arenaSessionId,
      source_type: "botarena",
      label: "Arena thread",
      audience_label: "BotArena group",
      audience_entity_id: audienceEntityId,
      conversation_kind: "thread",
      audience_role: "participant",
      status: "active",
      created_at: firstAt,
      last_activity_at: firstAt + 80 * 60_000,
    });

    for (let index = 0; index < 80; index += 1) {
      const occurredAt = firstAt + index * 60_000;
      const turnId = `arena-${index}`;

      repository.record({
        kind: "user_contact",
        occurredAt,
        sessionId: arenaSessionId,
        turnId,
        audienceEntityId,
        sourceStreamEntryIds: [createStreamEntryId()],
      });
      repository.record({
        kind: "borg_replied",
        occurredAt: occurredAt + 10_000,
        sessionId: arenaSessionId,
        turnId,
        audienceEntityId,
        sourceStreamEntryIds: [createStreamEntryId()],
      });
      repository.record({
        kind: "turn_completed",
        occurredAt: occurredAt + 20_000,
        sessionId: arenaSessionId,
        turnId,
        audienceEntityId,
        sourceStreamEntryIds: [createStreamEntryId()],
      });
    }

    expect(
      repository.listDailyOtherActiveSessionDensity({
        currentSessionId,
        sinceMs: firstAt - 1,
        untilMs: firstAt + 90 * 60_000,
        limit: 1,
      }),
    ).toEqual([
      expect.objectContaining({
        dayKey: "2026-06-15",
        sessionId: arenaSessionId,
        eventCount: 240,
        conversationTurnCount: 80,
        kindCounts: {
          userContact: 80,
          borgReplied: 80,
          turnCompleted: 80,
        },
      }),
    ]);

    db.close();
  });

  it("aggregates global daily active density without current-session filtering", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-activity-global-density-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "activity.db"), {
      migrations: composeMigrations(sessionMigrations, activityMigrations, commitmentMigrations),
    });
    const sessions = new SessionsRepository({
      db,
      clock: new FixedClock(10_000),
    });
    const repository = new ActivityRepository({
      db,
      clock: new FixedClock(10_000),
    });
    const firstSessionId = createSessionId();
    const secondSessionId = createSessionId();
    const archivedSessionId = createSessionId();
    const dayStart = Date.UTC(2026, 5, 15);
    const firstAt = dayStart + 60_000;
    const secondAt = dayStart + 120_000;
    const nextDayAt = Date.UTC(2026, 5, 16, 1, 0, 0);

    for (const [sessionId, status] of [
      [firstSessionId, "active"],
      [secondSessionId, "active"],
      [archivedSessionId, "archived"],
    ] as const) {
      sessions.ensure({
        session_id: sessionId,
        source_type: "demo",
        label: "Session",
        audience_label: "Audience",
        conversation_kind: "demo",
        audience_role: "participant",
        status,
        created_at: dayStart,
        last_activity_at: nextDayAt,
      });
    }

    repository.record({
      kind: "user_contact",
      occurredAt: firstAt,
      sessionId: firstSessionId,
      turnId: "turn-1",
      sourceStreamEntryIds: [createStreamEntryId()],
    });
    repository.record({
      kind: "borg_replied",
      occurredAt: secondAt,
      sessionId: secondSessionId,
      turnId: "turn-2",
      sourceStreamEntryIds: [createStreamEntryId()],
    });
    repository.record({
      kind: "turn_completed",
      occurredAt: secondAt + 1,
      sessionId: secondSessionId,
      turnId: "turn-2",
      sourceStreamEntryIds: [createStreamEntryId()],
    });
    repository.record({
      kind: "user_contact",
      occurredAt: secondAt + 2,
      sessionId: archivedSessionId,
      turnId: "archived",
      sourceStreamEntryIds: [createStreamEntryId()],
    });
    repository.record({
      kind: "user_contact",
      occurredAt: nextDayAt,
      sessionId: firstSessionId,
      turnId: "next-day",
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    expect(
      repository.listDailyGlobalActiveDensity({
        sinceMs: dayStart,
        untilMs: nextDayAt - 1,
        limit: 10,
      }),
    ).toEqual([
      {
        dayKey: "2026-06-15",
        dayStartMs: dayStart,
        eventCount: 3,
        conversationTurnCount: 2,
        distinctSessionCount: 2,
        kindCounts: {
          userContact: 1,
          borgReplied: 1,
          turnCompleted: 1,
        },
        firstOccurredAt: firstAt,
        lastOccurredAt: secondAt + 1,
      },
    ]);

    db.close();
  });
});
