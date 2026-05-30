import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  composeMigrations,
  openDatabase,
  type SqliteDatabase,
} from "../../storage/sqlite/index.js";
import {
  DEFAULT_SESSION_ID,
  QUARANTINED_USER_ENTRY_EVENT,
  StreamEntryIndexRepository,
  StreamWatermarkRepository,
  StreamWriter,
  streamEntryIndexMigrations,
  streamWatermarkMigrations,
  type StreamCursor,
  type StreamEntry,
  type StreamEntryKind,
} from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { CognitionError } from "../../util/errors.js";
import { parseStreamEntryId, type StreamEntryId } from "../../util/ids.js";

import {
  CHAT_RESPONSE_PROCESS_NAME,
  ChatResponseWatermarkCoordinator,
  type ChatResponseTerminalKind,
} from "./index.js";

type Harness = {
  db: SqliteDatabase;
  entryIndex: StreamEntryIndexRepository;
  watermarkRepository: StreamWatermarkRepository;
  writer: StreamWriter;
  coordinator: ChatResponseWatermarkCoordinator;
  close: () => void;
};

function cursorFor(entry: Pick<StreamEntry, "id" | "timestamp">): StreamCursor {
  return {
    ts: entry.timestamp,
    entryId: entry.id,
  };
}

async function appendTerminalStamp(
  writer: StreamWriter,
  input: {
    kind?: ChatResponseTerminalKind;
    fromCursorExclusive: StreamCursor | null;
    throughCursorInclusive: StreamCursor;
    sourceEntryIds: readonly StreamEntryId[];
    count?: number;
  },
): Promise<StreamEntry> {
  return writer.append({
    kind: input.kind ?? "agent_msg",
    content: { terminal: true },
    response_to: {
      kind: "stream_backlog",
      from_cursor_exclusive: input.fromCursorExclusive,
      through_cursor_inclusive: input.throughCursorInclusive,
      source_entry_ids: [...input.sourceEntryIds],
      count: input.count ?? input.sourceEntryIds.length,
    },
  });
}

async function appendQuarantineMarker(
  writer: StreamWriter,
  entries: readonly Pick<StreamEntry, "id">[],
): Promise<StreamEntry> {
  const source = entries[0];

  return writer.append({
    kind: "internal_event",
    content: {
      event: QUARANTINED_USER_ENTRY_EVENT,
      source_stream_entry_id: source?.id ?? null,
      cited_stream_entry_ids: entries.map((entry) => entry.id),
    },
  });
}

function indexedEntry(input: {
  id: string;
  entryIndex: number;
  kind?: StreamEntryKind;
}): StreamEntry {
  return {
    id: parseStreamEntryId(input.id),
    timestamp: 100,
    entry_index: input.entryIndex,
    kind: input.kind ?? "user_msg",
    content: input.id,
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: DEFAULT_SESSION_ID,
    compressed: false,
  };
}

describe("ChatResponseWatermarkCoordinator", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  function openHarness(): Harness {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-chat-response-watermark-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(streamEntryIndexMigrations, streamWatermarkMigrations),
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const watermarkRepository = new StreamWatermarkRepository({
      db,
      clock: new ManualClock(1_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      entryIndex,
      clock: new ManualClock(100),
    });
    const coordinator = new ChatResponseWatermarkCoordinator({
      watermarkRepository,
      entryIndex,
      logger: { warn: () => undefined },
    });

    return {
      db,
      entryIndex,
      watermarkRepository,
      writer,
      coordinator,
      close: () => {
        writer.close();
        db.close();
      },
    };
  }

  it("treats a missing watermark row as start-from-beginning", () => {
    const harness = openHarness();

    try {
      expect(harness.coordinator.getWatermark(DEFAULT_SESSION_ID)).toBeNull();
      expect(harness.coordinator.reconcile(DEFAULT_SESSION_ID)).toEqual({
        watermark: null,
        advancedThrough: null,
        appliedStamps: 0,
      });
      expect(
        harness.watermarkRepository.get(CHAT_RESPONSE_PROCESS_NAME, DEFAULT_SESSION_ID),
      ).toBeNull();
    } finally {
      harness.close();
    }
  });

  it("reconciles a single terminal stamp from a null watermark", async () => {
    const harness = openHarness();

    try {
      const source = await harness.writer.append({ kind: "user_msg", content: "A" });
      const throughCursor = cursorFor(source);

      await appendTerminalStamp(harness.writer, {
        fromCursorExclusive: null,
        throughCursorInclusive: throughCursor,
        sourceEntryIds: [source.id],
      });

      expect(harness.coordinator.reconcile(DEFAULT_SESSION_ID)).toEqual({
        watermark: throughCursor,
        advancedThrough: throughCursor,
        appliedStamps: 1,
      });
      expect(harness.coordinator.getWatermark(DEFAULT_SESSION_ID)).toEqual(throughCursor);
    } finally {
      harness.close();
    }
  });

  it.each<ChatResponseTerminalKind>(["agent_msg", "agent_observed", "agent_suppressed"])(
    "reconciles a %s terminal stamp for a quarantined inactive source entry",
    async (kind) => {
      const harness = openHarness();

      try {
        const source = await harness.writer.append({ kind: "user_msg", content: "A" });
        const throughCursor = cursorFor(source);

        await appendQuarantineMarker(harness.writer, [source]);
        expect(harness.entryIndex.lookup(source.id)?.active).toBe(false);

        await appendTerminalStamp(harness.writer, {
          kind,
          fromCursorExclusive: null,
          throughCursorInclusive: throughCursor,
          sourceEntryIds: [source.id],
        });

        expect(harness.coordinator.reconcile(DEFAULT_SESSION_ID)).toEqual({
          watermark: throughCursor,
          advancedThrough: throughCursor,
          appliedStamps: 1,
        });
      } finally {
        harness.close();
      }
    },
  );

  it("reconciles chained terminal stamps by exact cursor equality", async () => {
    const harness = openHarness();

    try {
      const first = await harness.writer.append({ kind: "user_msg", content: "A" });
      const firstCursor = cursorFor(first);
      await appendTerminalStamp(harness.writer, {
        fromCursorExclusive: null,
        throughCursorInclusive: firstCursor,
        sourceEntryIds: [first.id],
      });

      const second = await harness.writer.append({ kind: "user_msg", content: "B" });
      const secondCursor = cursorFor(second);
      await appendTerminalStamp(harness.writer, {
        fromCursorExclusive: firstCursor,
        throughCursorInclusive: secondCursor,
        sourceEntryIds: [second.id],
      });

      expect(harness.coordinator.reconcile(DEFAULT_SESSION_ID)).toEqual({
        watermark: secondCursor,
        advancedThrough: secondCursor,
        appliedStamps: 2,
      });
    } finally {
      harness.close();
    }
  });

  it("skips a non-contiguous stamp whose from cursor is not the current watermark", async () => {
    const harness = openHarness();

    try {
      const source = await harness.writer.append({ kind: "user_msg", content: "A" });
      await appendTerminalStamp(harness.writer, {
        fromCursorExclusive: {
          ts: 100,
          entryId: parseStreamEntryId("strm_aaaaaaaaaaaaaaaa"),
        },
        throughCursorInclusive: cursorFor(source),
        sourceEntryIds: [source.id],
      });

      expect(harness.coordinator.reconcile(DEFAULT_SESSION_ID)).toEqual({
        watermark: null,
        advancedThrough: null,
        appliedStamps: 0,
      });
    } finally {
      harness.close();
    }
  });

  it("rejects malformed stamps whose count mismatches source id length", async () => {
    const harness = openHarness();

    try {
      const source = await harness.writer.append({ kind: "user_msg", content: "A" });
      await appendTerminalStamp(harness.writer, {
        fromCursorExclusive: null,
        throughCursorInclusive: cursorFor(source),
        sourceEntryIds: [source.id],
        count: 2,
      });

      expect(harness.coordinator.reconcile(DEFAULT_SESSION_ID)).toEqual({
        watermark: null,
        advancedThrough: null,
        appliedStamps: 0,
      });
    } finally {
      harness.close();
    }
  });

  it.each<ChatResponseTerminalKind>(["agent_msg", "agent_observed", "agent_suppressed"])(
    "skips an unusable %s stamp and applies a later valid stamp with the same from cursor",
    async (kind) => {
      const harness = openHarness();

      try {
        const first = await harness.writer.append({ kind: "user_msg", content: "A" });
        await appendQuarantineMarker(harness.writer, [first]);
        const second = await harness.writer.append({ kind: "user_msg", content: "B" });
        const secondCursor = cursorFor(second);

        await appendTerminalStamp(harness.writer, {
          kind,
          fromCursorExclusive: null,
          throughCursorInclusive: cursorFor(first),
          sourceEntryIds: [first.id],
          count: 2,
        });
        await appendTerminalStamp(harness.writer, {
          kind,
          fromCursorExclusive: null,
          throughCursorInclusive: secondCursor,
          sourceEntryIds: [first.id, second.id],
        });

        expect(harness.coordinator.reconcile(DEFAULT_SESSION_ID)).toEqual({
          watermark: secondCursor,
          advancedThrough: secondCursor,
          appliedStamps: 1,
        });
      } finally {
        harness.close();
      }
    },
  );

  it("rejects a stamp whose source ids do not match the contiguous queued prefix", async () => {
    const harness = openHarness();

    try {
      const first = await harness.writer.append({ kind: "user_msg", content: "A" });
      const second = await harness.writer.append({ kind: "user_msg", content: "B" });

      await appendTerminalStamp(harness.writer, {
        fromCursorExclusive: null,
        throughCursorInclusive: cursorFor(second),
        sourceEntryIds: [second.id],
      });

      expect(harness.coordinator.reconcile(DEFAULT_SESSION_ID)).toEqual({
        watermark: null,
        advancedThrough: null,
        appliedStamps: 0,
      });
      expect(
        harness.watermarkRepository.get(CHAT_RESPONSE_PROCESS_NAME, DEFAULT_SESSION_ID),
      ).toBeNull();
      expect(harness.entryIndex.lookup(first.id)?.turn_id).toBeNull();
      expect(harness.entryIndex.lookup(second.id)?.turn_id).toBeNull();
    } finally {
      harness.close();
    }
  });

  it("rejects a stamp whose through cursor timestamp mismatches the indexed entry", async () => {
    const harness = openHarness();

    try {
      const source = await harness.writer.append({ kind: "user_msg", content: "A" });
      await appendTerminalStamp(harness.writer, {
        fromCursorExclusive: null,
        throughCursorInclusive: {
          ts: source.timestamp + 1,
          entryId: source.id,
        },
        sourceEntryIds: [source.id],
      });

      let thrown: unknown;
      try {
        harness.coordinator.reconcile(DEFAULT_SESSION_ID);
      } catch (error) {
        thrown = error;
      }

      expect(thrown).toBeInstanceOf(CognitionError);
      expect((thrown as CognitionError).code).toBe("CHAT_RESPONSE_CURSOR_TS_MISMATCH");
      expect(
        harness.watermarkRepository.get(CHAT_RESPONSE_PROCESS_NAME, DEFAULT_SESSION_ID),
      ).toBeNull();
    } finally {
      harness.close();
    }
  });

  it("finds a terminal stamp for an exact batch identity", async () => {
    const harness = openHarness();

    try {
      const first = await harness.writer.append({ kind: "user_msg", content: "A" });
      const second = await harness.writer.append({ kind: "user_msg", content: "B" });
      const stamp = await appendTerminalStamp(harness.writer, {
        kind: "agent_observed",
        fromCursorExclusive: cursorFor(first),
        throughCursorInclusive: cursorFor(second),
        sourceEntryIds: [first.id, second.id],
      });

      expect(
        harness.coordinator.findTerminalStampForBatch({
          sessionId: DEFAULT_SESSION_ID,
          fromCursorExclusive: cursorFor(first),
          throughCursorInclusive: cursorFor(second),
          sourceEntryIds: [first.id, second.id],
          count: 2,
        })?.entry_id,
      ).toBe(stamp.id);
    } finally {
      harness.close();
    }
  });

  it("does not advance to equal or older targets by same-millisecond entry id ordering", () => {
    const harness = openHarness();

    try {
      const older = indexedEntry({
        id: "strm_zzzzzzzzzzzzzzzz",
        entryIndex: 0,
      });
      const current = indexedEntry({
        id: "strm_aaaaaaaaaaaaaaaa",
        entryIndex: 1,
      });
      const newer = indexedEntry({
        id: "strm_mmmmmmmmmmmmmmmm",
        entryIndex: 2,
      });

      harness.entryIndex.recordEntry(older, 0);
      harness.entryIndex.recordEntry(current, 100);
      harness.entryIndex.recordEntry(newer, 200);

      expect(harness.coordinator.advanceThrough(DEFAULT_SESSION_ID, cursorFor(current))).toEqual({
        advanced: true,
        watermark: cursorFor(current),
      });
      expect(harness.coordinator.advanceThrough(DEFAULT_SESSION_ID, cursorFor(current))).toEqual({
        advanced: false,
        watermark: cursorFor(current),
      });
      expect(harness.coordinator.advanceThrough(DEFAULT_SESSION_ID, cursorFor(older))).toEqual({
        advanced: false,
        watermark: cursorFor(current),
      });
      expect(harness.coordinator.advanceThrough(DEFAULT_SESSION_ID, cursorFor(newer))).toEqual({
        advanced: true,
        watermark: cursorFor(newer),
      });
    } finally {
      harness.close();
    }
  });

  it("reconciles only through the stamped cursor after crash-before-advance", async () => {
    const harness = openHarness();

    try {
      const first = await harness.writer.append({ kind: "user_msg", content: "A" });
      const second = await harness.writer.append({ kind: "user_msg", content: "B" });
      const third = await harness.writer.append({ kind: "user_msg", content: "C" });
      const thirdCursor = cursorFor(third);
      await appendTerminalStamp(harness.writer, {
        fromCursorExclusive: null,
        throughCursorInclusive: thirdCursor,
        sourceEntryIds: [first.id, second.id, third.id],
      });
      const fourth = await harness.writer.append({ kind: "user_msg", content: "D" });
      const fifth = await harness.writer.append({ kind: "user_msg", content: "E" });

      expect(harness.coordinator.reconcile(DEFAULT_SESSION_ID)).toEqual({
        watermark: thirdCursor,
        advancedThrough: thirdCursor,
        appliedStamps: 1,
      });
      expect(harness.coordinator.getWatermark(DEFAULT_SESSION_ID)).toEqual(thirdCursor);
      expect(harness.coordinator.getWatermark(DEFAULT_SESSION_ID)).not.toEqual(cursorFor(fifth));
      const thirdEntryIndex = harness.entryIndex.lookup(third.id)?.entry_index;
      const fourthEntryIndex = harness.entryIndex.lookup(fourth.id)?.entry_index;
      const fifthEntryIndex = harness.entryIndex.lookup(fifth.id)?.entry_index;

      if (
        thirdEntryIndex === null ||
        thirdEntryIndex === undefined ||
        fourthEntryIndex === null ||
        fourthEntryIndex === undefined ||
        fifthEntryIndex === null ||
        fifthEntryIndex === undefined
      ) {
        throw new Error("Expected indexed entries for pending-tail assertion");
      }

      expect(fourthEntryIndex).toBeGreaterThan(thirdEntryIndex);
      expect(fifthEntryIndex).toBeGreaterThan(thirdEntryIndex);
    } finally {
      harness.close();
    }
  });
});
