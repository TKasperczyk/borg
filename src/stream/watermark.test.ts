import { describe, expect, it } from "vitest";

import { openDatabase, runMigrations } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import { createSessionId, DEFAULT_SESSION_ID } from "../util/ids.js";

import { streamWatermarkMigrations, StreamWatermarkRepository } from "./watermark.js";

describe("StreamWatermarkRepository", () => {
  it("baselines prior presentations without replacing existing executive acknowledgements", () => {
    const db = openDatabase(":memory:", {
      migrations: streamWatermarkMigrations.filter((migration) => migration.id < 3),
    });
    const clock = new ManualClock(1_000);
    const repo = new StreamWatermarkRepository({ db, clock });
    const prefix = "autonomy:goal-presentation:";
    const otherSession = createSessionId();
    const presented = { progressAnchor: 123.5, inputGeneration: 7, turnId: "prior-turn" };
    try {
      for (const session of [DEFAULT_SESSION_ID, otherSession]) {
        repo.set(`${prefix}goal_a`, session, {
          lastTs: 9,
          lastEntryId: "goal_a",
          metadata: { version: 1, inputGeneration: 9, lastProgressVersion: null, presented },
        });
      }
      repo.set(`${prefix}goal_null`, DEFAULT_SESSION_ID, {
        lastTs: 0,
        lastEntryId: "goal_null",
        metadata: { presented: null },
      });
      repo.set(`${prefix}goal_without_metadata`, DEFAULT_SESSION_ID, {
        lastTs: 0,
        lastEntryId: "goal_without_metadata",
      });
      repo.set("unrelated", DEFAULT_SESSION_ID, {
        lastTs: 0,
        lastEntryId: "unrelated",
        metadata: { presented },
      });
      repo.set(`${prefix}goal_existing`, DEFAULT_SESSION_ID, {
        lastTs: 7,
        lastEntryId: "goal_existing",
        metadata: { presented },
      });
      const existing = repo.set(`${prefix}executive:goal_existing`, DEFAULT_SESSION_ID, {
        lastTs: 3,
        lastEntryId: "goal_existing",
        metadata: { progressAnchor: 50, inputGeneration: 3, turnId: "own-turn" },
      });
      clock.advance(500);
      runMigrations(db, streamWatermarkMigrations, clock.now());
      for (const session of [DEFAULT_SESSION_ID, otherSession]) {
        expect(repo.get(`${prefix}executive:goal_a`, session)).toEqual({
          processName: `${prefix}executive:goal_a`,
          sessionId: session,
          lastTs: 7,
          lastEntryId: "goal_a",
          updatedAt: 1_000,
          metadata: presented,
        });
      }
      expect(repo.get(`${prefix}executive:goal_existing`, DEFAULT_SESSION_ID)).toEqual(existing);
      expect(repo.get(`${prefix}executive:goal_null`, DEFAULT_SESSION_ID)).toBeNull();
      expect(repo.get(`${prefix}executive:goal_without_metadata`, DEFAULT_SESSION_ID)).toBeNull();
      expect(repo.get(`${prefix}executive:unrelated`, DEFAULT_SESSION_ID)).toBeNull();
      const before = db
        .prepare("SELECT * FROM stream_watermarks ORDER BY process_name, session_id")
        .all();
      const changes = db.prepare("SELECT total_changes() AS count").get();
      runMigrations(db, streamWatermarkMigrations, clock.now());
      // Replaying the SQL itself must also preserve the existing acknowledgements.
      const baseline = streamWatermarkMigrations.find((migration) => migration.id === 3)!;
      expect(baseline.name).toBe("executive_goal_presentation_baseline");
      expect(typeof baseline.up).toBe("string");
      db.exec(baseline.up as string);
      expect(
        db.prepare("SELECT * FROM stream_watermarks ORDER BY process_name, session_id").all(),
      ).toEqual(before);
      expect(db.prepare("SELECT total_changes() AS count").get()).toEqual(changes);
    } finally {
      db.close();
    }
  });

  it("leaves a fresh store empty when baselining executive presentations", () => {
    const db = openDatabase(":memory:", { migrations: streamWatermarkMigrations });
    try {
      expect(db.prepare("SELECT * FROM stream_watermarks").all()).toEqual([]);
      expect(db.listAppliedMigrations().map(({ id, name }) => ({ id, name }))).toEqual([
        { id: 1, name: "stream_watermark_baseline" },
        { id: 2, name: "stream_watermark_metadata_json" },
        { id: 3, name: "executive_goal_presentation_baseline" },
      ]);
    } finally {
      db.close();
    }
  });

  function openRepo(nowMs = 1_000): {
    repo: StreamWatermarkRepository;
    clock: ManualClock;
    close: () => void;
  } {
    const db = openDatabase(":memory:", {
      migrations: streamWatermarkMigrations,
    });
    const clock = new ManualClock(nowMs);
    const repo = new StreamWatermarkRepository({ db, clock });
    return { repo, clock, close: () => db.close() };
  }

  it("returns null for unknown (process, session) pairs", () => {
    const { repo, close } = openRepo();

    try {
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)).toBeNull();
    } finally {
      close();
    }
  });

  it("upserts watermarks and returns them on subsequent reads", () => {
    const { repo, clock, close } = openRepo(1_000);

    try {
      const first = repo.set("episodic-extractor", DEFAULT_SESSION_ID, {
        lastTs: 123,
        lastEntryId: "strm_aaaaaaaaaaaaaaaa",
      });

      expect(first.lastTs).toBe(123);
      expect(first.lastEntryId).toBe("strm_aaaaaaaaaaaaaaaa");
      expect(first.updatedAt).toBe(1_000);
      expect(first.metadata).toBeNull();

      clock.advance(50);
      const second = repo.set("episodic-extractor", DEFAULT_SESSION_ID, {
        lastTs: 456,
        lastEntryId: "strm_bbbbbbbbbbbbbbbb",
      });

      expect(second.lastTs).toBe(456);
      expect(second.updatedAt).toBe(1_050);

      const fetched = repo.get("episodic-extractor", DEFAULT_SESSION_ID);
      expect(fetched?.lastTs).toBe(456);
      expect(fetched?.lastEntryId).toBe("strm_bbbbbbbbbbbbbbbb");
      expect(fetched?.metadata).toBeNull();
    } finally {
      close();
    }
  });

  it("round-trips optional metadata without requiring existing callers to provide it", () => {
    const { repo, close } = openRepo(1_000);

    try {
      const stored = repo.set(
        "autonomy:executive-focus-due:goal-stale-backoff:goal_a",
        DEFAULT_SESSION_ID,
        {
          lastTs: 123,
          lastEntryId: "goal-stale-event",
          metadata: {
            empty_count: 2,
            source: "goal_stale",
          },
        },
      );

      expect(stored.metadata).toEqual({
        empty_count: 2,
        source: "goal_stale",
      });
      expect(
        repo.get("autonomy:executive-focus-due:goal-stale-backoff:goal_a", DEFAULT_SESSION_ID)
          ?.metadata,
      ).toEqual({
        empty_count: 2,
        source: "goal_stale",
      });

      repo.set("autonomy:executive-focus-due:goal-stale-backoff:goal_a", DEFAULT_SESSION_ID, {
        lastTs: 456,
        lastEntryId: "goal-stale-event-2",
      });

      expect(
        repo.get("autonomy:executive-focus-due:goal-stale-backoff:goal_a", DEFAULT_SESSION_ID)
          ?.metadata,
      ).toBeNull();
    } finally {
      close();
    }
  });

  it("fails loudly when a persisted watermark has no cursor entry id", () => {
    const db = openDatabase(":memory:");
    const repo = new StreamWatermarkRepository({ db, clock: new ManualClock(1_000) });

    try {
      db.exec(`
        CREATE TABLE stream_watermarks (
          process_name TEXT NOT NULL,
          session_id TEXT NOT NULL,
          last_ts INTEGER NOT NULL,
          last_entry_id TEXT NULL,
          updated_at INTEGER NOT NULL,
          metadata_json TEXT,
          PRIMARY KEY (process_name, session_id)
        )
      `);
      db.prepare(
        `INSERT INTO stream_watermarks (
          process_name,
          session_id,
          last_ts,
          last_entry_id,
          updated_at,
          metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?)`,
      ).run("episodic-extractor", DEFAULT_SESSION_ID, 123, null, 1_000, null);

      let thrown: unknown;
      try {
        repo.get("episodic-extractor", DEFAULT_SESSION_ID);
      } catch (error) {
        thrown = error;
      }

      expect(thrown).toBeInstanceOf(StorageError);
      expect((thrown as StorageError).code).toBe("STREAM_WATERMARK_INVALID_CURSOR");
      expect((thrown as StorageError).message).toContain("invalid last_entry_id (null)");
    } finally {
      db.close();
    }
  });

  it("keeps watermarks scoped per (process, session)", () => {
    const { repo, close } = openRepo();
    const otherSession = createSessionId();

    try {
      repo.set("episodic-extractor", DEFAULT_SESSION_ID, { lastTs: 100, lastEntryId: "watermark" });
      repo.set("episodic-extractor", otherSession, { lastTs: 200, lastEntryId: "watermark" });
      repo.set("semantic-extractor", DEFAULT_SESSION_ID, { lastTs: 300, lastEntryId: "watermark" });

      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(100);
      expect(repo.get("episodic-extractor", otherSession)?.lastTs).toBe(200);
      expect(repo.get("semantic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(300);
    } finally {
      close();
    }
  });

  it("reset removes the watermark", () => {
    const { repo, close } = openRepo();

    try {
      repo.set("episodic-extractor", DEFAULT_SESSION_ID, { lastTs: 100, lastEntryId: "watermark" });
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)).not.toBeNull();

      repo.reset("episodic-extractor", DEFAULT_SESSION_ID);
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)).toBeNull();
    } finally {
      close();
    }
  });
});
