import { describe, expect, it } from "vitest";

import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { streamEntryIndexMigrations } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../../util/ids.js";
import { trainOfThoughtMigrations } from "./migrations.js";
import { TrainOfThoughtRepository } from "./repository.js";

describe("TrainOfThoughtRepository", () => {
  it("appends self-private journal entries and preserves get as the latest thought view", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", {
      migrations: trainOfThoughtMigrations,
    });
    const repository = new TrainOfThoughtRepository({
      db,
      clock,
    });
    const selfEntityId = createEntityId();

    try {
      expect(repository.get()).toBeNull();

      const first = repository.upsert({
        text: "First private thought.",
        selfEntityId,
        sourceTurnId: "turn-first",
      });

      expect(first).toMatchObject({
        text: "First private thought.",
        self_entity_id: selfEntityId,
        disclosure_class: "self_private",
        created_at: 1_000,
        updated_at: 1_000,
      });

      clock.set(2_000);
      const second = repository.append({
        text: "Second private thought.",
        selfEntityId,
        sourceTurnId: "turn-second",
      });

      expect(second).toMatchObject({
        id: 2,
        text: "Second private thought.",
        self_entity_id: selfEntityId,
        disclosure_class: "self_private",
        created_at: 2_000,
        updated_at: 2_000,
        source_turn_id: "turn-second",
        marker_stream_entry_id: null,
      });
      expect(repository.latest()).toEqual(second);
      expect(repository.get()).toEqual({
        text: "Second private thought.",
        self_entity_id: selfEntityId,
        disclosure_class: "self_private",
        created_at: 2_000,
        updated_at: 2_000,
      });
      expect(repository.list({ limit: 2 }).map((entry) => entry.text)).toEqual([
        "Second private thought.",
        "First private thought.",
      ]);
      expect(
        db.prepare("SELECT COUNT(*) AS count FROM train_of_thought_journal_entries").get() as {
          count: number;
        },
      ).toMatchObject({ count: 2 });
    } finally {
      db.close();
    }
  });

  it("migrates a non-empty singleton row into the append journal", () => {
    const db = openDatabase(":memory:", {
      migrations: trainOfThoughtMigrations.slice(0, 1),
    });
    const selfEntityId = createEntityId();

    try {
      db.prepare(
        `
          INSERT INTO train_of_thought (
            id, self_entity_id, text, disclosure_class, created_at, updated_at
          ) VALUES (1, ?, 'Legacy thought.', 'self_private', 1_000, 2_000)
        `,
      ).run(selfEntityId);

      trainOfThoughtMigrations[1]?.up(db);

      const repository = new TrainOfThoughtRepository({ db });

      expect(repository.latest()).toMatchObject({
        id: 1,
        text: "Legacy thought.",
        self_entity_id: selfEntityId,
        disclosure_class: "self_private",
        created_at: 1_000,
        updated_at: 2_000,
      });
      expect(
        db
          .prepare(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'train_of_thought'",
          )
          .get(),
      ).toBeUndefined();
    } finally {
      db.close();
    }
  });

  it("lists inclusive created-at ranges with stable cursors and explicit session anchors", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(streamEntryIndexMigrations, trainOfThoughtMigrations),
    });
    const repository = new TrainOfThoughtRepository({ db, clock });
    const selfEntityId = createEntityId();
    const firstSessionId = createSessionId();
    const secondSessionId = createSessionId();
    const firstMarkerId = createStreamEntryId();
    const secondMarkerId = createStreamEntryId();

    try {
      const first = repository.append({
        text: "first equal-time journal entry",
        selfEntityId,
        sourceTurnId: "turn-first-session",
      });
      const second = repository.append({
        text: "second equal-time journal entry",
        selfEntityId,
        markerStreamEntryId: secondMarkerId,
      });
      const unanchored = repository.append({
        text: "nullable journal anchors",
        selfEntityId,
        sourceTurnId: null,
        markerStreamEntryId: null,
      });

      db.prepare(
        `INSERT INTO stream_entry_index (
           entry_id, session_id, byte_offset, timestamp, kind, turn_id, active
         ) VALUES (?, ?, 0, 1_000, 'thought', ?, 1)`,
      ).run(firstMarkerId, firstSessionId, first.source_turn_id);
      db.prepare(
        `INSERT INTO stream_entry_index (
           entry_id, session_id, byte_offset, timestamp, kind, turn_id, active
         ) VALUES (?, ?, 0, 1_000, 'thought', NULL, 1)`,
      ).run(secondMarkerId, secondSessionId);

      const global = repository.listForRange({
        sinceMs: 1_000,
        untilMs: 1_000,
        limit: 10,
      });

      expect(global.map((entry) => entry.id)).toEqual([unanchored.id, second.id, first.id]);
      expect(
        repository
          .listForRange({
            sinceMs: 1_000,
            untilMs: 1_000,
            limit: 10,
            cursor: { createdAt: 1_000, id: unanchored.id },
          })
          .map((entry) => entry.id),
      ).toEqual([second.id, first.id]);
      expect(
        repository
          .listForRange({
            sinceMs: 1_000,
            untilMs: 1_000,
            limit: 10,
            sessionId: firstSessionId,
          })
          .map((entry) => entry.id),
      ).toEqual([first.id]);
      expect(
        repository
          .listForRange({
            sinceMs: 1_000,
            untilMs: 1_000,
            limit: 10,
            sessionId: secondSessionId,
          })
          .map((entry) => entry.id),
      ).toEqual([second.id]);
      expect(
        db
          .prepare("PRAGMA index_info('idx_train_of_thought_journal_created_at')")
          .all()
          .map((row) => (row as { name: string }).name),
      ).toEqual(["created_at", "id"]);
    } finally {
      db.close();
    }
  });
});
