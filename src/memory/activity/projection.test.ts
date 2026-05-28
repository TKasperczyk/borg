import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { SessionsRepository, sessionMigrations } from "../../sessions/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../../util/ids.js";
import { activityMigrations } from "./migrations.js";
import { selectCrossSessionSelfActivity } from "./projection.js";
import { ActivityRepository } from "./repository.js";

const NOW_MS = 1_000_000;
const tempDirs: string[] = [];

afterEach(() => {
  for (const tempDir of tempDirs.splice(0)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

describe("selectCrossSessionSelfActivity", () => {
  it("shows other active-session contact only to operator sessions from the creator", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-activity-projection-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "activity.db"), {
      migrations: composeMigrations(sessionMigrations, activityMigrations),
    });
    const sessions = new SessionsRepository({
      db,
      clock: new FixedClock(NOW_MS),
    });
    const activity = new ActivityRepository({
      db,
      clock: new FixedClock(NOW_MS),
    });
    const currentSessionId = createSessionId();
    const otherSessionId = createSessionId();
    const operatorEntityId = createEntityId();
    const aliceEntityId = createEntityId();

    sessions.ensure({
      session_id: currentSessionId,
      source_type: "demo",
      label: "Operator control",
      audience_label: "Tom",
      audience_entity_id: operatorEntityId,
      conversation_kind: "demo",
      audience_role: "operator",
      status: "active",
      created_at: NOW_MS - 120_000,
      last_activity_at: NOW_MS,
    });
    sessions.ensure({
      session_id: otherSessionId,
      source_type: "demo",
      label: "Alice demo",
      audience_label: "Alice",
      audience_entity_id: null,
      conversation_kind: "demo",
      audience_role: "participant",
      status: "active",
      created_at: NOW_MS - 120_000,
      last_activity_at: NOW_MS - 41_000,
    });
    activity.record({
      kind: "user_contact",
      occurredAt: NOW_MS - 41_000,
      sessionId: otherSessionId,
      turnId: "alice-turn",
      speakerEntityId: aliceEntityId,
      actorEntityId: aliceEntityId,
      audienceEntityId: aliceEntityId,
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    const visible = selectCrossSessionSelfActivity({
      repository: activity,
      currentSessionId,
      currentAudienceEntityId: operatorEntityId,
      sessionAudienceRole: "operator",
      currentSenderBorgRole: "creator",
      nowMs: NOW_MS,
      recencyWindowMs: 60_000,
      cap: 5,
    });
    const hidden = selectCrossSessionSelfActivity({
      repository: activity,
      currentSessionId,
      currentAudienceEntityId: aliceEntityId,
      sessionAudienceRole: "participant",
      currentSenderBorgRole: null,
      nowMs: NOW_MS,
      recencyWindowMs: 60_000,
      cap: 5,
    });

    expect(visible).toEqual([
      {
        kind: "user_contact",
        occurredAt: NOW_MS - 41_000,
        relativeAge: "~41s ago",
        text: "Alice contacted Borg ~41s ago in another active session.",
      },
    ]);
    expect(hidden).toEqual([]);

    db.close();
  });
});
