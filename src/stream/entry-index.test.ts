import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase, type Migration } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { createEntityId } from "../util/ids.js";

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
});
