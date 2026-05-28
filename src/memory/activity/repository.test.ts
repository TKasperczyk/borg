import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
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
});
