import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { openDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../util/ids.js";

import {
  DEFAULT_SESSION_ID,
  getSessionStreamPath,
  getStreamDirectory,
  QUARANTINED_USER_ENTRY_EVENT,
  StreamEntryIndexRepository,
  StreamReader,
  StreamWriter,
  streamEntryIndexMigrations,
  type StreamEntry,
} from "./index.js";

type StreamEntryStampRow = {
  source_message_key_source_type: string | null;
  source_message_key_source_external_id: string | null;
  source_message_key_external_message_id: string | null;
  response_to_kind: string | null;
  response_to_from_cursor_ts: number | null;
  response_to_from_cursor_entry_id: string | null;
  response_to_through_cursor_ts: number | null;
  response_to_through_cursor_entry_id: string | null;
  response_to_source_entry_ids: string | null;
  response_to_count: number | null;
};

describe("stream entry index", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("upgrades a populated v2 database when the v3 index already exists", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-own-record-index-upgrade-"));
    tempDirs.push(tempDir);
    const databasePath = join(tempDir, "borg.db");
    const v2Db = openDatabase(databasePath, {
      migrations: streamEntryIndexMigrations.slice(0, 2),
    });
    const v2Index = new StreamEntryIndexRepository({ db: v2Db, dataDir: tempDir });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(100),
      entryIndex: v2Index,
    });
    const thought = await writer.append({
      kind: "thought",
      content: "record present before the v3 migration",
    });

    v2Db.exec(`
      CREATE INDEX idx_stream_entry_kind_active_time
      ON stream_entry_index(kind, active, timestamp DESC, entry_id DESC)
    `);
    expect(v2Db.listAppliedMigrations().map((migration) => migration.id)).toEqual([1, 2]);
    writer.close();
    v2Db.close();

    const upgradedDb = openDatabase(databasePath, {
      migrations: [...streamEntryIndexMigrations],
    });

    try {
      const upgradedIndex = new StreamEntryIndexRepository({
        db: upgradedDb,
        dataDir: tempDir,
      });
      const indexes = upgradedDb.prepare("PRAGMA index_list('stream_entry_index')").all();

      expect(upgradedDb.listAppliedMigrations().map((migration) => migration.id)).toEqual([
        1, 2, 3, 4,
      ]);
      expect(indexes).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ name: "idx_stream_entry_kind_active_time" }),
          expect.objectContaining({ name: "idx_stream_entry_response_lane" }),
        ]),
      );
      expect(upgradedIndex.lookup(thought.id)).toMatchObject({
        entry_id: thought.id,
        kind: "thought",
        active: true,
      });
    } finally {
      upgradedDb.close();
    }
  });

  it("indexes and backfills task stamps without introducing user backlog stamp fields", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-task-stamp-"));
    tempDirs.push(dataDir);
    const db = openDatabase(join(dataDir, "borg.db"), { migrations: streamEntryIndexMigrations });
    const entryIndex = new StreamEntryIndexRepository({ db, dataDir });
    const writer = new StreamWriter({
      dataDir,
      clock: new ManualClock(100),
      entryIndex,
      taskEventsEnabled: true,
    });
    try {
      const event = await writer.append({ kind: "internal_event", content: "Task completed" });
      const responseTo = {
        kind: "task_event" as const,
        event_id: "event",
        event_entry_id: event.id,
        task_id: "task",
        task_version: 1,
      };
      const terminal = await writer.append({
        kind: "agent_msg",
        content: "Result",
        response_to: responseTo,
      });
      expect(entryIndex.listSessionIdsWithTaskEvents()).toEqual([DEFAULT_SESSION_ID]);
      expect(entryIndex.lookupSessionTaskEventResponseStamps(DEFAULT_SESSION_ID)).toEqual([
        expect.objectContaining({
          entry_id: terminal.id,
          response_to_kind: "task_event",
          response_to_from_cursor_ts: null,
          response_to_through_cursor_ts: null,
          response_to_source_entry_ids: null,
          response_to_count: null,
        }),
      ]);
      expect(
        entryIndex.lookupSessionStreamBacklogResponseStamps({
          sessionId: DEFAULT_SESSION_ID,
          terminalKinds: ["agent_msg"],
        }),
      ).toEqual([]);
      db.prepare("UPDATE stream_entry_index SET response_to_kind = NULL WHERE entry_id = ?").run(
        terminal.id,
      );
      await entryIndex.backfillSession(DEFAULT_SESSION_ID);
      expect(entryIndex.lookupSessionTaskEventResponseStamps(DEFAULT_SESSION_ID)).toHaveLength(1);
    } finally {
      writer.close();
      db.close();
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

  it("backfills entry_index from file order when embedded entry indexes collide", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const first: StreamEntry = {
      id: createStreamEntryId(),
      timestamp: 100,
      entry_index: 7,
      kind: "user_msg",
      content: "first duplicate embedded index",
      sender_entity_id: null,
      reply_target_entity_id: null,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
    };
    const second: StreamEntry = {
      ...first,
      id: createStreamEntryId(),
      timestamp: 101,
      content: "second duplicate embedded index",
    };

    try {
      mkdirSync(getStreamDirectory(tempDir), { recursive: true });
      writeFileSync(
        getSessionStreamPath(tempDir, DEFAULT_SESSION_ID),
        `${JSON.stringify(first)}\n${JSON.stringify(second)}\n`,
      );

      await entryIndex.backfillSession(DEFAULT_SESSION_ID);

      expect(entryIndex.lookup(first.id)?.entry_index).toBe(0);
      expect(entryIndex.lookup(second.id)?.entry_index).toBe(1);
      expect(entryIndex.nextEntryIndex(DEFAULT_SESSION_ID)).toBe(2);
    } finally {
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

  it("persists source message and response stamp columns", async () => {
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
      const sourceMessageKey = {
        source_type: "demo",
        source_external_id: "conversation-1",
        external_message_id: "message-1",
      };
      const source = await writer.append({
        kind: "user_msg",
        content: "source keyed",
        source_message_key: sourceMessageKey,
      });
      const responseTo = {
        kind: "stream_backlog" as const,
        from_cursor_exclusive: null,
        through_cursor_inclusive: {
          ts: source.timestamp,
          entryId: source.id,
        },
        source_entry_ids: [source.id],
        count: 1,
      };
      const response = await writer.append({
        kind: "agent_msg",
        content: "processed backlog",
        response_to: responseTo,
      });
      const selectStampRow = db.prepare(
        `SELECT source_message_key_source_type, source_message_key_source_external_id,
                source_message_key_external_message_id, response_to_kind,
                response_to_from_cursor_ts, response_to_from_cursor_entry_id,
                response_to_through_cursor_ts, response_to_through_cursor_entry_id,
                response_to_source_entry_ids, response_to_count
         FROM stream_entry_index
         WHERE entry_id = ?`,
      );

      expect(selectStampRow.get(source.id) as StreamEntryStampRow).toEqual({
        source_message_key_source_type: sourceMessageKey.source_type,
        source_message_key_source_external_id: sourceMessageKey.source_external_id,
        source_message_key_external_message_id: sourceMessageKey.external_message_id,
        response_to_kind: null,
        response_to_from_cursor_ts: null,
        response_to_from_cursor_entry_id: null,
        response_to_through_cursor_ts: null,
        response_to_through_cursor_entry_id: null,
        response_to_source_entry_ids: null,
        response_to_count: null,
      });
      expect(selectStampRow.get(response.id) as StreamEntryStampRow).toEqual({
        source_message_key_source_type: null,
        source_message_key_source_external_id: null,
        source_message_key_external_message_id: null,
        response_to_kind: responseTo.kind,
        response_to_from_cursor_ts: null,
        response_to_from_cursor_entry_id: null,
        response_to_through_cursor_ts: responseTo.through_cursor_inclusive.ts,
        response_to_through_cursor_entry_id: responseTo.through_cursor_inclusive.entryId,
        response_to_source_entry_ids: JSON.stringify(responseTo.source_entry_ids),
        response_to_count: responseTo.count,
      });
      expect(entryIndex.lookupBySourceMessageKey(sourceMessageKey)?.entry_id).toBe(source.id);
    } finally {
      writer.close();
      db.close();
    }
  });

  it("lists uncapped sessions with pending response backlog", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const otherSessionId = createSessionId();
    const receiptOnlySessionId = createSessionId();
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
    const receiptOnlyWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: receiptOnlySessionId,
      clock,
      entryIndex,
    });

    try {
      const pending = await writer.append({
        kind: "user_msg",
        content: "pending",
      });
      const receiptPending = await receiptOnlyWriter.append({
        kind: "user_msg",
        content: "waiting for image receipt",
        receipt_pending: true,
      });
      await writer.append({
        kind: "user_msg",
        content: "already turn-bound",
        turn_id: "turn-answered",
      });
      await writer.append({
        kind: "agent_msg",
        content: "not pending",
      });
      const inactivePending = await otherWriter.append({
        kind: "user_msg",
        content: "pending even if inactive",
      });

      db.prepare("UPDATE stream_entry_index SET active = 0 WHERE entry_id = ?").run(
        inactivePending.id,
      );

      expect(new Set(entryIndex.listSessionIdsWithPendingResponseBacklog())).toEqual(
        new Set([pending.session_id, otherSessionId, receiptOnlySessionId]),
      );
      expect(entryIndex.lookup(receiptPending.id)?.receipt_pending).toBe(true);

      entryIndex.setReceiptPending(receiptPending.id, false);

      expect(new Set(entryIndex.listSessionIdsWithPendingResponseBacklog())).toEqual(
        new Set([pending.session_id, otherSessionId, receiptOnlySessionId]),
      );
    } finally {
      writer.close();
      otherWriter.close();
      receiptOnlyWriter.close();
      db.close();
    }
  });

  it("preserves a cleared receipt-pending flag during stream backfill", async () => {
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
      const entry = await writer.append({
        kind: "user_msg",
        content: "image receipt started",
        receipt_pending: true,
      });

      entryIndex.setReceiptPending(entry.id, false);
      await entryIndex.backfillSession(DEFAULT_SESSION_ID);

      expect(entryIndex.lookup(entry.id)?.receipt_pending).toBe(false);

      db.prepare("DELETE FROM stream_entry_index WHERE entry_id = ?").run(entry.id);
      await entryIndex.backfillSession(DEFAULT_SESSION_ID);

      expect(entryIndex.lookup(entry.id)?.receipt_pending).toBe(true);
    } finally {
      writer.close();
      db.close();
    }
  });

  it("allows non-user source keys without shadowing user-message duplicates", async () => {
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
      const sourceMessageKey = {
        source_type: "demo",
        source_external_id: "conversation-1",
        external_message_id: "message-1",
      };

      await writer.append({
        kind: "agent_msg",
        content: "not an inbound message",
        source_message_key: sourceMessageKey,
      });

      expect(entryIndex.lookupBySourceMessageKey(sourceMessageKey)).toBeNull();

      const user = await writer.append({
        kind: "user_msg",
        content: "inbound message",
        source_message_key: sourceMessageKey,
      });

      expect(entryIndex.lookupBySourceMessageKey(sourceMessageKey)?.entry_id).toBe(user.id);
    } finally {
      writer.close();
      db.close();
    }
  });

  it("rejects duplicate source message keys", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const sourceMessageKey = {
      source_type: "demo",
      source_external_id: "conversation-1",
      external_message_id: "message-1",
    };
    const first: StreamEntry = {
      id: createStreamEntryId(),
      timestamp: 100,
      entry_index: 0,
      kind: "user_msg",
      content: "first",
      sender_entity_id: null,
      reply_target_entity_id: null,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
      source_message_key: sourceMessageKey,
    };
    const duplicate: StreamEntry = {
      ...first,
      id: createStreamEntryId(),
      timestamp: 101,
      entry_index: 1,
      content: "duplicate",
    };

    try {
      entryIndex.recordEntry(first, 0);

      expect(() => entryIndex.recordEntry(duplicate, 100)).toThrow();
    } finally {
      db.close();
    }
  });

  it("backfills source message and response stamp columns", async () => {
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
      const sourceMessageKey = {
        source_type: "demo",
        source_external_id: "conversation-1",
        external_message_id: "message-1",
      };
      const source = await writer.append({
        kind: "user_msg",
        content: "source keyed",
        source_message_key: sourceMessageKey,
      });
      const responseTo = {
        kind: "stream_backlog" as const,
        from_cursor_exclusive: null,
        through_cursor_inclusive: {
          ts: source.timestamp,
          entryId: source.id,
        },
        source_entry_ids: [source.id],
        count: 1,
      };
      const response = await writer.append({
        kind: "agent_msg",
        content: "processed backlog",
        response_to: responseTo,
      });

      db.prepare(
        `UPDATE stream_entry_index
         SET source_message_key_source_type = NULL,
             source_message_key_source_external_id = NULL,
             source_message_key_external_message_id = NULL,
             response_to_kind = NULL,
             response_to_from_cursor_ts = NULL,
             response_to_from_cursor_entry_id = NULL,
             response_to_through_cursor_ts = NULL,
             response_to_through_cursor_entry_id = NULL,
             response_to_source_entry_ids = NULL,
             response_to_count = NULL`,
      ).run();

      await expect(entryIndex.backfillSession(DEFAULT_SESSION_ID)).resolves.toEqual({
        inserted: 2,
      });

      expect(entryIndex.lookup(source.id)).toMatchObject({
        source_message_key_source_type: sourceMessageKey.source_type,
        source_message_key_source_external_id: sourceMessageKey.source_external_id,
        source_message_key_external_message_id: sourceMessageKey.external_message_id,
      });
      expect(entryIndex.lookup(response.id)).toMatchObject({
        response_to_kind: responseTo.kind,
        response_to_from_cursor_ts: null,
        response_to_from_cursor_entry_id: null,
        response_to_through_cursor_ts: responseTo.through_cursor_inclusive.ts,
        response_to_through_cursor_entry_id: responseTo.through_cursor_inclusive.entryId,
        response_to_source_entry_ids: JSON.stringify(responseTo.source_entry_ids),
        response_to_count: responseTo.count,
      });
    } finally {
      writer.close();
      db.close();
    }
  });

  it("self-repairs source message and response stamp columns after committed index update failure", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
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
    let failingWriter: StreamWriter | null = null;

    try {
      const sourceMessageKey = {
        source_type: "demo",
        source_external_id: "conversation-1",
        external_message_id: "message-1",
      };
      const source = await writer.append({
        kind: "user_msg",
        content: "source keyed before index outage",
        source_message_key: sourceMessageKey,
      });
      const responseTo = {
        kind: "stream_backlog" as const,
        from_cursor_exclusive: null,
        through_cursor_inclusive: {
          ts: source.timestamp,
          entryId: source.id,
        },
        source_entry_ids: [source.id],
        count: 1,
      };
      const failingIndex = {
        isPoisoned: (sessionId: typeof DEFAULT_SESSION_ID) => entryIndex.isPoisoned(sessionId),
        markPoisoned: (sessionId: typeof DEFAULT_SESSION_ID) => entryIndex.markPoisoned(sessionId),
        nextEntryIndex: (sessionId: typeof DEFAULT_SESSION_ID) =>
          entryIndex.nextEntryIndex(sessionId),
        recordEntry: vi.fn(() => {
          throw new Error("index update unavailable after fsync");
        }),
        backfillSession: vi.fn((sessionId: typeof DEFAULT_SESSION_ID) =>
          entryIndex.backfillSession(sessionId),
        ),
      };

      db.prepare(
        `UPDATE stream_entry_index
         SET source_message_key_source_type = NULL,
             source_message_key_source_external_id = NULL,
             source_message_key_external_message_id = NULL
         WHERE entry_id = ?`,
      ).run(source.id);

      failingWriter = new StreamWriter({
        dataDir: tempDir,
        clock,
        logger: { error: vi.fn() },
        entryIndex: failingIndex as never,
      });

      const appendedResponse = await failingWriter.append({
        kind: "agent_msg",
        content: "durable response before index failure",
        response_to: responseTo,
      });

      const durableEntries = new StreamReader({ dataDir: tempDir }).tail(10);
      const durableResponse = durableEntries.find(
        (entry) => entry.content === "durable response before index failure",
      );

      expect(durableEntries.map((entry) => entry.kind)).toEqual(["user_msg", "agent_msg"]);
      expect(durableResponse).toMatchObject({
        kind: "agent_msg",
        response_to: responseTo,
      });
      expect(durableResponse!.id).toBe(appendedResponse.id);
      expect(failingIndex.backfillSession).toHaveBeenCalledTimes(1);
      expect(entryIndex.lookupBySourceMessageKey(sourceMessageKey)?.entry_id).toBe(source.id);
      expect(entryIndex.lookup(durableResponse!.id)).toMatchObject({
        response_to_kind: responseTo.kind,
        response_to_from_cursor_ts: null,
        response_to_from_cursor_entry_id: null,
        response_to_through_cursor_ts: responseTo.through_cursor_inclusive.ts,
        response_to_through_cursor_entry_id: responseTo.through_cursor_inclusive.entryId,
        response_to_source_entry_ids: JSON.stringify(responseTo.source_entry_ids),
        response_to_count: responseTo.count,
      });
    } finally {
      failingWriter?.close();
      writer.close();
      db.close();
    }
  });

  it("looks up stream backlog response stamps in append order for terminal kinds", async () => {
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
      const source = await writer.append({
        kind: "user_msg",
        content: "source keyed",
      });
      const responseTo = {
        kind: "stream_backlog" as const,
        from_cursor_exclusive: null,
        through_cursor_inclusive: {
          ts: source.timestamp,
          entryId: source.id,
        },
        source_entry_ids: [source.id],
        count: 1,
      };
      const observed = await writer.append({
        kind: "agent_observed",
        content: { observation: "terminal" },
        response_to: responseTo,
      });
      await writer.append({
        kind: "thought",
        content: { note: "not terminal" },
        response_to: responseTo,
      });
      const suppressed = await writer.append({
        kind: "agent_suppressed",
        content: { reason: "terminal" },
        response_to: responseTo,
      });
      const agent = await writer.append({
        kind: "agent_msg",
        content: "terminal",
        response_to: responseTo,
      });

      expect(
        entryIndex
          .lookupSessionStreamBacklogResponseStamps({
            sessionId: DEFAULT_SESSION_ID,
            terminalKinds: ["agent_msg", "agent_suppressed", "agent_observed"],
          })
          .map((record) => record.entry_id),
      ).toEqual([observed.id, suppressed.id, agent.id]);
    } finally {
      writer.close();
      db.close();
    }
  });

  it("looks up exact stream backlog response stamps and rejects identity mismatches", async () => {
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
        content: "first",
      });
      const second = await writer.append({
        kind: "user_msg",
        content: "second",
      });
      const fromCursor = {
        ts: first.timestamp,
        entryId: first.id,
      };
      const throughCursor = {
        ts: second.timestamp,
        entryId: second.id,
      };
      const terminalKinds = ["agent_msg", "agent_suppressed", "agent_observed"] as const;
      const stamps = [];

      for (const kind of terminalKinds) {
        stamps.push(
          await writer.append({
            kind,
            content: { terminal: kind },
            response_to: {
              kind: "stream_backlog",
              from_cursor_exclusive: fromCursor,
              through_cursor_inclusive: throughCursor,
              source_entry_ids: [first.id, second.id],
              count: 2,
            },
          }),
        );
      }

      for (const [index, kind] of terminalKinds.entries()) {
        expect(
          entryIndex.lookupExactStreamBacklogResponseStamp({
            sessionId: DEFAULT_SESSION_ID,
            terminalKinds: [kind],
            fromCursorExclusive: fromCursor,
            throughCursorInclusive: throughCursor,
            sourceEntryIds: [first.id, second.id],
            count: 2,
          })?.entry_id,
        ).toBe(stamps[index]?.id);
      }

      expect(
        entryIndex.lookupExactStreamBacklogResponseStamp({
          sessionId: DEFAULT_SESSION_ID,
          terminalKinds,
          fromCursorExclusive: fromCursor,
          throughCursorInclusive: throughCursor,
          sourceEntryIds: [second.id, first.id],
          count: 2,
        }),
      ).toBeNull();
      expect(
        entryIndex.lookupExactStreamBacklogResponseStamp({
          sessionId: DEFAULT_SESSION_ID,
          terminalKinds,
          fromCursorExclusive: fromCursor,
          throughCursorInclusive: throughCursor,
          sourceEntryIds: [first.id, second.id],
          count: 1,
        }),
      ).toBeNull();
      expect(
        entryIndex.lookupExactStreamBacklogResponseStamp({
          sessionId: DEFAULT_SESSION_ID,
          terminalKinds,
          fromCursorExclusive: null,
          throughCursorInclusive: throughCursor,
          sourceEntryIds: [first.id, second.id],
          count: 2,
        }),
      ).toBeNull();
    } finally {
      writer.close();
      db.close();
    }
  });

  it("uses existing row count for next entry index when legacy rows are null", async () => {
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
      await writer.appendMany([
        { kind: "user_msg", content: "alpha" },
        { kind: "agent_msg", content: "beta" },
        { kind: "internal_event", content: "omega" },
      ]);

      db.prepare("UPDATE stream_entry_index SET entry_index = NULL WHERE entry_index = 2").run();

      expect(entryIndex.nextEntryIndex(DEFAULT_SESSION_ID)).toBe(3);
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

  it("lists active thought records globally with inclusive bounds and stable equal-time cursors", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-own-record-index-"));
    tempDirs.push(tempDir);
    const otherSessionId = createSessionId();
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({ db, dataDir: tempDir });
    const clock = new ManualClock(100);
    const writer = new StreamWriter({ dataDir: tempDir, clock, entryIndex });
    const otherWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: otherSessionId,
      clock,
      entryIndex,
    });

    try {
      const localThought = await writer.append({
        kind: "thought",
        content: "local active thought",
        turn_id: "turn-local-active",
      });
      const abortedThought = await writer.append({
        kind: "thought",
        content: "aborted thought",
        turn_id: "turn-aborted-thought",
      });
      await writer.append({
        kind: "internal_event",
        content: {
          event: "aborted_turn",
          turn_id: "turn-aborted-thought",
          aborted_stream_entry_ids: [abortedThought.id],
        },
      });
      const otherThought = await otherWriter.append({
        kind: "thought",
        content: "cross-session active thought",
        turn_id: "turn-other-active",
      });
      await writer.append({ kind: "user_msg", content: "not an own thought" });

      const global = entryIndex.listActiveEntriesByKindRange({
        kinds: ["thought"],
        sinceTs: 100,
        untilTs: 100,
        limit: 10,
      });
      const expectedGlobalIds = [localThought.id, otherThought.id].sort().reverse();

      expect(global.map((record) => record.entry_id)).toEqual(expectedGlobalIds);
      expect(global.map((record) => record.session_id)).toEqual(
        expectedGlobalIds.map((id) =>
          id === localThought.id ? DEFAULT_SESSION_ID : otherSessionId,
        ),
      );
      expect(global.map((record) => record.entry_id)).not.toContain(abortedThought.id);

      expect(
        entryIndex
          .listActiveEntriesByKindRange({
            kinds: ["thought"],
            sinceTs: 100,
            untilTs: 100,
            sessionId: DEFAULT_SESSION_ID,
            limit: 10,
          })
          .map((record) => record.entry_id),
      ).toEqual([localThought.id]);

      const firstPage = entryIndex.listActiveEntriesByKindRange({
        kinds: ["thought"],
        sinceTs: 100,
        untilTs: 100,
        limit: 1,
      });
      const secondPage = entryIndex.listActiveEntriesByKindRange({
        kinds: ["thought"],
        sinceTs: 100,
        untilTs: 100,
        cursor: {
          timestamp: firstPage[0]!.timestamp,
          entryId: firstPage[0]!.entry_id,
        },
        limit: 1,
      });

      expect([...firstPage, ...secondPage].map((record) => record.entry_id)).toEqual(
        expectedGlobalIds,
      );
      expect(
        db
          .prepare("PRAGMA index_info('idx_stream_entry_kind_active_time')")
          .all()
          .map((row) => (row as { name: string }).name),
      ).toEqual(["kind", "active", "timestamp", "entry_id"]);
    } finally {
      writer.close();
      otherWriter.close();
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

  it("does not warn when startup backfill leaves no legacy rows with null kind", async () => {
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
      await writer.append({
        kind: "user_msg",
        content: "backfilled kind",
      });

      await entryIndex.backfillSession(DEFAULT_SESSION_ID);
      const report = entryIndex.warnLegacyRowsMissingKind();

      expect(report).toEqual({
        count: 0,
        sampleEntryIds: [],
      });
      expect(logger.warn).not.toHaveBeenCalled();
    } finally {
      writer.close();
      db.close();
    }
  });

  it("persists corrective-preference retry and dead-letter receipts across reopen", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-receipts-"));
    tempDirs.push(tempDir);
    const dbPath = join(tempDir, "borg.db");
    const sessionId = createSessionId();
    const sourceEntryId = createStreamEntryId();
    const firstDb = openDatabase(dbPath, {
      migrations: [...streamEntryIndexMigrations],
    });
    const firstIndex = new StreamEntryIndexRepository({ db: firstDb, dataDir: tempDir });

    firstIndex.recordCorrectivePreferenceIngestionFailure({
      sourceEntryId,
      sessionId,
      error: "provider unavailable",
      updatedAt: 100,
      maxFailures: 3,
    });
    firstIndex.recordCorrectivePreferenceIngestionFailure({
      sourceEntryId,
      sessionId,
      error: "provider still unavailable",
      updatedAt: 200,
      maxFailures: 3,
    });
    firstDb.close();

    const reopenedDb = openDatabase(dbPath, {
      migrations: [...streamEntryIndexMigrations],
    });
    const reopenedIndex = new StreamEntryIndexRepository({
      db: reopenedDb,
      dataDir: tempDir,
    });

    try {
      expect(reopenedIndex.getCorrectivePreferenceIngestionReceipt(sourceEntryId)).toEqual({
        source_entry_id: sourceEntryId,
        session_id: sessionId,
        status: "retryable",
        failure_count: 2,
        last_error: "provider still unavailable",
        updated_at: 200,
      });
      expect(
        reopenedIndex.recordCorrectivePreferenceIngestionFailure({
          sourceEntryId,
          sessionId,
          error: "persistent poison",
          updatedAt: 300,
          maxFailures: 3,
        }),
      ).toMatchObject({ status: "dead_letter", failure_count: 3 });

      reopenedIndex.recordCorrectivePreferenceIngestionProcessed({
        sourceEntryId,
        sessionId,
        updatedAt: 400,
      });
      expect(reopenedIndex.getCorrectivePreferenceIngestionReceipt(sourceEntryId)).toMatchObject({
        status: "dead_letter",
        failure_count: 3,
        last_error: "persistent poison",
      });
    } finally {
      reopenedDb.close();
    }
  });
});
