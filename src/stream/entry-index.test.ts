import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase, type Migration } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { createEntityId, createSessionId } from "../util/ids.js";

import {
  DEFAULT_SESSION_ID,
  StreamEntryIndexRepository,
  StreamWriter,
  streamEntryIndexMigrations,
} from "./index.js";

describe("stream entry index", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("backfills missing rows from the middle of a session stream", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(100),
      entryIndex,
    });

    try {
      const first = await writer.append({
        kind: "user_msg",
        content: "alpha",
      });
      const middle = await writer.append({
        kind: "agent_msg",
        content: "beta",
      });
      const last = await writer.append({
        kind: "internal_event",
        content: "omega",
      });
      const middleRecord = entryIndex.lookup(middle.id);

      expect(middleRecord).not.toBeNull();

      db.prepare("DELETE FROM stream_entry_index WHERE entry_id = ?").run(middle.id);

      expect(entryIndex.lookup(middle.id)).toBeNull();
      await expect(entryIndex.backfillSession(DEFAULT_SESSION_ID)).resolves.toEqual({
        inserted: 1,
      });
      expect(entryIndex.lookup(first.id)).not.toBeNull();
      expect(entryIndex.lookup(middle.id)).toEqual(middleRecord);
      expect(entryIndex.lookup(last.id)).not.toBeNull();
    } finally {
      writer.close();
      db.close();
    }
  });

  it("records sender entity ids and backfills legacy null senders", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const senderEntityId = createEntityId();
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(100),
      entryIndex,
    });

    try {
      const tagged = await writer.append({
        kind: "user_msg",
        content: "alpha",
        sender_entity_id: senderEntityId,
      });
      const legacy = await writer.append({
        kind: "agent_msg",
        content: "beta",
      });

      expect(entryIndex.lookup(tagged.id)?.sender_entity_id).toBe(senderEntityId);
      expect(entryIndex.lookup(legacy.id)?.sender_entity_id).toBeNull();

      db.prepare("DELETE FROM stream_entry_index").run();

      await expect(entryIndex.backfillSession(DEFAULT_SESSION_ID)).resolves.toEqual({
        inserted: 2,
      });
      expect(entryIndex.lookup(tagged.id)?.sender_entity_id).toBe(senderEntityId);
      expect(entryIndex.lookup(legacy.id)?.sender_entity_id).toBeNull();
    } finally {
      writer.close();
      db.close();
    }
  });

  it("counts user messages by session from the stream entry index", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const otherSessionId = createSessionId();
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const clock = new ManualClock(100);
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
      entryIndex,
    });
    const otherWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: otherSessionId,
      clock,
      entryIndex,
    });

    try {
      expect(
        entryIndex.countSessionEntriesByKind({
          sessionId: DEFAULT_SESSION_ID,
          kind: "user_msg",
        }),
      ).toBe(0);

      const first = await writer.append({ kind: "user_msg", content: "one" });
      await writer.append({ kind: "agent_msg", content: "assistant" });
      await writer.append({ kind: "user_msg", content: "two" });
      await otherWriter.append({ kind: "user_msg", content: "other session" });

      expect(
        entryIndex.countSessionEntriesByKind({
          sessionId: DEFAULT_SESSION_ID,
          kind: "user_msg",
        }),
      ).toBe(2);
      expect(
        entryIndex.countSessionEntriesByKind({
          sessionId: DEFAULT_SESSION_ID,
          kind: "user_msg",
          excludeEntryId: first.id,
        }),
      ).toBe(1);
      expect(
        entryIndex.countSessionEntriesByKind({
          sessionId: otherSessionId,
          kind: "user_msg",
        }),
      ).toBe(1);
    } finally {
      writer.close();
      otherWriter.close();
      db.close();
    }
  });

  it("looks up indexed facts by id across hits, misses, and sessions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const otherSessionId = createSessionId();
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const clock = new ManualClock(100);
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
      entryIndex,
    });
    const otherWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: otherSessionId,
      clock,
      entryIndex,
    });

    try {
      const first = await writer.append({
        kind: "user_msg",
        content: "one",
        turn_id: "turn-one",
      });
      const other = await otherWriter.append({
        kind: "agent_msg",
        content: "other",
        turn_id: "turn-other",
      });

      const facts = entryIndex.lookupEntriesById([first.id, "strm_0000000000000000", other.id]);

      expect(new Set(facts.keys())).toEqual(new Set([first.id, other.id]));
      expect(facts.get(first.id)).toMatchObject({
        entry_id: first.id,
        session_id: DEFAULT_SESSION_ID,
        kind: "user_msg",
        turn_id: "turn-one",
        active: true,
      });
      expect(facts.get(other.id)).toMatchObject({
        entry_id: other.id,
        session_id: otherSessionId,
        kind: "agent_msg",
        turn_id: "turn-other",
        active: true,
      });
      expect(facts.has("strm_0000000000000000")).toBe(false);
    } finally {
      writer.close();
      otherWriter.close();
      db.close();
    }
  });

  it("records and backfills active trust facts for aborted turns", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(100),
      entryIndex,
    });

    try {
      const userEntry = await writer.append({
        kind: "user_msg",
        content: "aborted",
        turn_id: "turn-aborted",
      });
      const marker = await writer.append({
        kind: "internal_event",
        content: {
          event: "aborted_turn",
          turn_id: "turn-aborted",
          aborted_stream_entry_ids: [userEntry.id],
        },
      });

      expect(entryIndex.lookupEntriesById([userEntry.id]).get(userEntry.id)?.active).toBe(false);
      expect(entryIndex.lookupEntriesById([marker.id]).get(marker.id)?.active).toBe(false);

      db.prepare(
        "UPDATE stream_entry_index SET active = 1, turn_id = NULL, turn_status = NULL",
      ).run();
      await entryIndex.backfillSession(DEFAULT_SESSION_ID);

      expect(entryIndex.lookupEntriesById([userEntry.id]).get(userEntry.id)).toMatchObject({
        active: false,
        turn_id: "turn-aborted",
      });
      expect(entryIndex.lookupEntriesById([marker.id]).get(marker.id)?.active).toBe(false);
    } finally {
      writer.close();
      db.close();
    }
  });

  it("backfills kind for legacy stream index rows", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(100),
      entryIndex,
    });

    try {
      const entry = await writer.append({ kind: "user_msg", content: "legacy kind" });

      db.prepare("UPDATE stream_entry_index SET kind = NULL WHERE entry_id = ?").run(entry.id);
      expect(entryIndex.lookup(entry.id)?.kind).toBeNull();

      await entryIndex.backfillSession(DEFAULT_SESSION_ID);

      expect(entryIndex.lookup(entry.id)?.kind).toBe("user_msg");
      expect(
        entryIndex.countSessionEntriesByKind({
          sessionId: DEFAULT_SESSION_ID,
          kind: "user_msg",
        }),
      ).toBe(1);
    } finally {
      writer.close();
      db.close();
    }
  });

  it("adds sender entity ids to an existing stream index as a nullable column", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const senderMigration = streamEntryIndexMigrations.find(
      (migration) => migration.name === "add-stream-entry-sender-entity-id",
    ) as Migration | undefined;

    expect(senderMigration).toBeDefined();

    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [
        {
          id: 1,
          name: "legacy-stream-entry-index",
          up: `
            CREATE TABLE stream_entry_index (
              entry_id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              byte_offset INTEGER NOT NULL,
              timestamp INTEGER NOT NULL
            );
            INSERT INTO stream_entry_index (entry_id, session_id, byte_offset, timestamp)
            VALUES ('strm_abcdefghijklmnop', 'default', 0, 100);
          `,
        },
        senderMigration as Migration,
      ],
    });

    try {
      const row = db
        .prepare("SELECT sender_entity_id FROM stream_entry_index WHERE entry_id = ?")
        .get("strm_abcdefghijklmnop") as { sender_entity_id: string | null } | undefined;

      expect(row?.sender_entity_id).toBeNull();
    } finally {
      db.close();
    }
  });

  it("adds kind to an existing stream index as a nullable indexed column", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const kindMigration = streamEntryIndexMigrations.find(
      (migration) => migration.name === "add-stream-entry-kind",
    ) as Migration | undefined;

    expect(kindMigration).toBeDefined();

    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [
        {
          id: 1,
          name: "legacy-stream-entry-index",
          up: `
            CREATE TABLE stream_entry_index (
              entry_id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              byte_offset INTEGER NOT NULL,
              timestamp INTEGER NOT NULL
            );
            INSERT INTO stream_entry_index (entry_id, session_id, byte_offset, timestamp)
            VALUES ('strm_abcdefghijklmnop', 'default', 0, 100);
          `,
        },
        kindMigration as Migration,
      ],
    });

    try {
      const row = db
        .prepare("SELECT kind FROM stream_entry_index WHERE entry_id = ?")
        .get("strm_abcdefghijklmnop") as { kind: string | null } | undefined;
      const indexRow = db
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?")
        .get("idx_stream_entry_session_kind") as { name: string } | undefined;

      expect(row?.kind).toBeNull();
      expect(indexRow?.name).toBe("idx_stream_entry_session_kind");
    } finally {
      db.close();
    }
  });

  it("adds trust facts to an existing stream index with active legacy rows", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const trustFactsMigration = streamEntryIndexMigrations.find(
      (migration) => migration.name === "add-stream-entry-trust-facts",
    ) as Migration | undefined;

    expect(trustFactsMigration).toBeDefined();

    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [
        {
          id: 1,
          name: "legacy-stream-entry-index",
          up: `
            CREATE TABLE stream_entry_index (
              entry_id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              byte_offset INTEGER NOT NULL,
              timestamp INTEGER NOT NULL,
              kind TEXT NULL,
              sender_entity_id TEXT NULL
            );
            INSERT INTO stream_entry_index (
              entry_id, session_id, byte_offset, timestamp, kind, sender_entity_id
            )
            VALUES ('strm_abcdefghijklmnop', 'default', 0, 100, 'user_msg', NULL);
          `,
        },
        trustFactsMigration as Migration,
      ],
    });

    try {
      const row = db
        .prepare(
          `SELECT turn_id, turn_status, active
           FROM stream_entry_index
           WHERE entry_id = ?`,
        )
        .get("strm_abcdefghijklmnop") as
        | { turn_id: string | null; turn_status: string | null; active: number }
        | undefined;
      const indexRow = db
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?")
        .get("idx_stream_entry_active") as { name: string } | undefined;

      expect(row).toEqual({
        turn_id: null,
        turn_status: null,
        active: 1,
      });
      expect(indexRow?.name).toBe("idx_stream_entry_active");
    } finally {
      db.close();
    }
  });
});
