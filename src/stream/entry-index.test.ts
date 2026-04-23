import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";

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
      await expect(entryIndex.backfillSession(DEFAULT_SESSION_ID)).resolves.toEqual({ inserted: 1 });
      expect(entryIndex.lookup(first.id)).not.toBeNull();
      expect(entryIndex.lookup(middle.id)).toEqual(middleRecord);
      expect(entryIndex.lookup(last.id)).not.toBeNull();
    } finally {
      writer.close();
      db.close();
    }
  });
});
