import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { openDatabase, type Migration } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../util/ids.js";

import {
  DEFAULT_SESSION_ID,
  QUARANTINED_USER_ENTRY_EVENT,
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

  it("looks up session entries by kind with byte offsets", async () => {
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
      const firstMarker = await writer.append({
        kind: "internal_event",
        content: { event: "first" },
      });
      await writer.append({
        kind: "user_msg",
        content: "not a marker",
      });
      const secondMarker = await writer.append({
        kind: "internal_event",
        content: { event: "second" },
      });
      await otherWriter.append({
        kind: "internal_event",
        content: { event: "other" },
      });

      const records = entryIndex.lookupSessionEntriesByKind({
        sessionId: DEFAULT_SESSION_ID,
        kind: "internal_event",
      });

      expect(records.map((record) => record.entry_id)).toEqual([firstMarker.id, secondMarker.id]);
      expect(records.map((record) => record.kind)).toEqual(["internal_event", "internal_event"]);
      expect(records.every((record) => record.session_id === DEFAULT_SESSION_ID)).toBe(true);
      expect(records.every((record) => record.byte_offset >= 0)).toBe(true);
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

  it("looks up quarantined shared-state artifact refs across sessions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const otherSessionId = createSessionId();
    const quarantinedSource = createStreamEntryId();
    const quarantinedCitation = createStreamEntryId();
    const otherQuarantinedSource = createStreamEntryId();
    const trustedSource = createStreamEntryId();
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
      await writer.append({
        kind: "internal_event",
        content: {
          event: QUARANTINED_USER_ENTRY_EVENT,
          source_stream_entry_id: quarantinedSource,
          cited_stream_entry_ids: [quarantinedCitation, quarantinedSource, "not-a-stream-id"],
        },
      });
      await otherWriter.append({
        kind: "internal_event",
        content: {
          event: QUARANTINED_USER_ENTRY_EVENT,
          source_stream_entry_id: otherQuarantinedSource,
          cited_stream_entry_ids: [],
        },
      });
      await otherWriter.append({
        kind: "internal_event",
        content: {
          event: "aborted_turn",
          aborted_stream_entry_ids: [trustedSource],
        },
      });

      expect(entryIndex.quarantinedSharedStateArtifactRefs()).toEqual(
        new Set([quarantinedCitation, quarantinedSource, otherQuarantinedSource]),
      );
      expect(entryIndex.quarantinedSharedStateArtifactRefs().has(trustedSource)).toBe(false);
    } finally {
      writer.close();
      otherWriter.close();
      db.close();
    }
  });

  it("backfills quarantined shared-state artifact refs for legacy rows", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const quarantinedSource = createStreamEntryId();
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
      await writer.append({
        kind: "internal_event",
        content: {
          event: QUARANTINED_USER_ENTRY_EVENT,
          source_stream_entry_id: quarantinedSource,
          cited_stream_entry_ids: [quarantinedSource],
        },
      });

      db.prepare("DELETE FROM stream_quarantine_refs").run();
      expect(entryIndex.quarantinedSharedStateArtifactRefs()).toEqual(new Set());

      await expect(entryIndex.backfillSession(DEFAULT_SESSION_ID)).resolves.toEqual({
        inserted: 0,
      });
      expect(entryIndex.quarantinedSharedStateArtifactRefs()).toEqual(new Set([quarantinedSource]));
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

  it("warns when startup backfill leaves legacy rows with null kind", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const logger = {
      error: vi.fn(),
      warn: vi.fn(),
    };
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
      logger,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(100),
      entryIndex,
    });

    try {
      const backfillableEntry = await writer.append({
        kind: "user_msg",
        content: "legacy kind",
      });
      const orphanedLegacyEntryId = createStreamEntryId();

      db.prepare("UPDATE stream_entry_index SET kind = NULL WHERE entry_id = ?").run(
        backfillableEntry.id,
      );
      db.prepare(
        `INSERT INTO stream_entry_index (entry_id, session_id, byte_offset, timestamp, kind, sender_entity_id)
         VALUES (?, ?, ?, ?, NULL, NULL)`,
      ).run(orphanedLegacyEntryId, DEFAULT_SESSION_ID, 999, 50);

      await entryIndex.backfillSession(DEFAULT_SESSION_ID);
      const report = entryIndex.warnLegacyRowsMissingKind();

      expect(entryIndex.lookup(backfillableEntry.id)?.kind).toBe("user_msg");
      expect(report).toEqual({
        count: 1,
        sampleEntryIds: [orphanedLegacyEntryId],
      });
      expect(logger.warn).toHaveBeenCalledWith(
        `Stream entry index has 1 legacy rows with kind IS NULL after startup backfill; sample_entry_ids=${orphanedLegacyEntryId}`,
      );
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

  it("creates the quarantine refs table and indexes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const quarantineRefsMigration = streamEntryIndexMigrations.find(
      (migration) => migration.name === "create-stream-quarantine-refs",
    ) as Migration | undefined;

    expect(quarantineRefsMigration).toBeDefined();
    expect(quarantineRefsMigration?.id).toBe(205);

    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [quarantineRefsMigration as Migration],
    });

    try {
      const tableRow = db
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
        .get("stream_quarantine_refs") as { name: string } | undefined;
      const referencedIndexRow = db
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?")
        .get("idx_stream_quarantine_refs_referenced") as { name: string } | undefined;
      const sessionIndexRow = db
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?")
        .get("idx_stream_quarantine_refs_session") as { name: string } | undefined;

      expect(tableRow?.name).toBe("stream_quarantine_refs");
      expect(referencedIndexRow?.name).toBe("idx_stream_quarantine_refs_referenced");
      expect(sessionIndexRow?.name).toBe("idx_stream_quarantine_refs_session");
    } finally {
      db.close();
    }
  });
});
