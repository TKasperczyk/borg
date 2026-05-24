import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  QUARANTINED_USER_ENTRY_EVENT,
  StreamEntryIndexRepository,
  StreamReader,
  StreamWriter,
  streamEntryIndexMigrations,
} from "../stream/index.js";
import { openDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";

import { CitationResolver } from "./citations.js";

describe("citation resolver", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("filters citation status markers through the stream entry index", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-citations-"));
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
      const trustedEntry = await writer.append({
        kind: "user_msg",
        content: "trusted citation",
      });
      const quarantinedEntry = await writer.append({
        kind: "user_msg",
        content: "quarantined citation",
      });
      await writer.append({
        kind: "internal_event",
        content: {
          event: QUARANTINED_USER_ENTRY_EVENT,
          source_stream_entry_id: quarantinedEntry.id,
          cited_stream_entry_ids: [quarantinedEntry.id],
        },
      });

      const lookupStatusMarkersSpy = vi.spyOn(entryIndex, "lookupSessionEntriesByKind");
      const iterateSpy = vi.spyOn(StreamReader.prototype, "iterate");
      const indexedResolver = new CitationResolver({
        dataDir: tempDir,
        entryIndex,
      });
      const scanResolver = new CitationResolver({
        dataDir: tempDir,
      });

      const indexed = await indexedResolver.resolveCitationEntries([
        trustedEntry.id,
        quarantinedEntry.id,
      ]);
      const scanned = await scanResolver.resolveCitationEntries([
        trustedEntry.id,
        quarantinedEntry.id,
      ]);

      expect([...indexed.keys()]).toEqual([trustedEntry.id]);
      expect(indexed).toEqual(scanned);
      expect(lookupStatusMarkersSpy).toHaveBeenCalledWith({
        sessionId: trustedEntry.session_id,
        kind: "internal_event",
      });
      expect(iterateSpy).toHaveBeenCalledTimes(1);
    } finally {
      writer.close();
      db.close();
    }
  });
});
