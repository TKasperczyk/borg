import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { openDatabase } from "../storage/sqlite/index.js";
import { relationshipPrivateMemoryDisclosureLabel } from "../memory/common/disclosure-label.js";
import { memoryDisclosurePayloadFields } from "../memory/common/disclosure-serializers.js";
import { createSessionId, createStreamEntryId } from "../util/ids.js";
import {
  StreamEntryIndexRepository,
  StreamReader,
  StreamWriter,
  getSessionStreamPath,
  readStreamEntryAtOffset,
  streamEntryIndexMigrations,
} from "./index.js";
import type { StreamEntryInput } from "./types.js";
import { FRAME_ANOMALY_ANNOTATION_EVENT } from "./frame-anomaly.js";

const cleanups: Array<() => void> = [];
afterEach(() => {
  while (cleanups.length) cleanups.pop()!();
});

function harness() {
  const dataDir = mkdtempSync(join(tmpdir(), "borg-reader-compat-"));
  const sessionId = createSessionId();
  const writer = new StreamWriter({ dataDir, sessionId });
  cleanups.push(() => {
    writer.close();
    rmSync(dataDir, { recursive: true, force: true });
  });
  return {
    dataDir,
    sessionId,
    writer,
    reader: new StreamReader({ dataDir, sessionId }),
    path: getSessionStreamPath(dataDir, sessionId),
  };
}

describe("response stamp reader compatibility", () => {
  it.each([false, true])(
    "retains entries and annotation labels after SQLite closes (legacy entries: %s)",
    async (legacy) => {
      const h = harness();
      const disclosure = memoryDisclosurePayloadFields(
        relationshipPrivateMemoryDisclosureLabel([]),
      );
      const source = await h.writer.append({
        kind: "user_msg",
        content: "通常の報告です。",
        metadata: disclosure,
      });
      const annotation = await h.writer.append({
        kind: "internal_event",
        content: {
          event: FRAME_ANOMALY_ANNOTATION_EVENT,
          version: 1,
          source_stream_entry_id: source.id,
          source_session_id: source.session_id,
          source_type: "kira",
          source_stream_entry_ids: [source.id],
          kind: "frame_assignment_claim",
          confidence: 0.92,
          rationale: "疑わしいフレーム。",
          classifier_model: null,
          classifier_turn_id: null,
          classifier_source_event_id: null,
          restores_quarantine_marker_ids: [],
          audience: null,
          sender_entity_id: null,
          ...disclosure,
          source_disclosure_label: disclosure.disclosure_label,
        },
      });
      const failure = await h.writer.append({
        kind: "internal_event",
        content: "Live stream ingestion failed: boom",
      });
      if (legacy) {
        writeFileSync(
          h.path,
          [source, annotation, failure]
            .map((entry) => JSON.stringify({ ...entry, entry_index: undefined }))
            .join("\n") + "\n",
        );
      }
      const db = openDatabase(":memory:", { migrations: streamEntryIndexMigrations });
      const index = new StreamEntryIndexRepository({ db, dataDir: h.dataDir });
      index.backfillSession(h.sessionId);
      db.close();

      const error = vi.fn();
      const reader = new StreamReader({
        dataDir: h.dataDir,
        sessionId: h.sessionId,
        entryIndex: index,
        logger: { error },
      });
      const expected = h.reader.tail(10);
      expect(reader.tail(10)).toEqual(expected);
      expect(reader.scanReverse().entries).toEqual(expected);
      const forward = [];
      for await (const entry of reader.iterate()) forward.push(entry);
      expect(forward).toEqual(expected);
      expect(forward.map((entry) => entry.id)).toEqual([source.id, annotation.id, failure.id]);
      expect(forward[0]?.frame_anomalies).toEqual([
        expect.objectContaining({
          annotation_entry_id: annotation.id,
          label: "frame anomaly suspected: frame_assignment_claim (0.92)",
          rationale: "疑わしいフレーム。",
          disclosure_label: disclosure.disclosure_label,
        }),
      ]);
      expect(error).toHaveBeenCalledWith(
        `Stream index unavailable; reading JSONL metadata in ${h.path}`,
      );
    },
  );

  it("surfaces projection failures instead of dropping valid source entries", async () => {
    const h = harness();
    await h.writer.append({ kind: "internal_event", content: "valid event" });
    const failure = new Error("projection is corrupt");
    const reader = new StreamReader({
      dataDir: h.dataDir,
      sessionId: h.sessionId,
      entryIndex: {
        lookup: vi.fn(),
        annotateFrameAnomalies: () => {
          throw failure;
        },
      },
    });
    expect(() => reader.tail(10)).toThrow(failure);
    expect(() => reader.scanReverse()).toThrow(failure);
    await expect(reader.iterate().next()).rejects.toMatchObject({
      code: "BORG_STREAM_ERROR",
      cause: failure,
    });
  });

  it("retains unknown stamps opaquely in forward, reverse, offset and index reads while writers reject them", async () => {
    const h = harness();
    const entry = await h.writer.append({ kind: "agent_msg", content: "Future reply" });
    const stamp = { kind: "future_response", event: { id: "future", values: [1, null, "opaque"] } };
    writeFileSync(h.path, JSON.stringify({ ...entry, response_to: stamp }) + "\n");
    const expected = { ...entry, opaque_response_to: stamp };
    const forward = [];
    for await (const read of h.reader.iterate()) forward.push(read);
    expect(forward).toEqual([expected]);
    expect(h.reader.tail(10)).toEqual([expected]);
    expect(h.reader.scanReverse().entries).toEqual([expected]);
    expect(
      readStreamEntryAtOffset({ dataDir: h.dataDir, sessionId: h.sessionId, byteOffset: 0 }),
    ).toEqual(expected);
    const db = openDatabase(":memory:", { migrations: streamEntryIndexMigrations });
    try {
      const index = new StreamEntryIndexRepository({ db, dataDir: h.dataDir });
      index.backfillSession(h.sessionId);
      expect(index.lookup(entry.id)).toMatchObject({
        entry_id: entry.id,
        kind: "agent_msg",
        response_to_kind: "future_response",
      });
      expect(
        index.lookupSessionStreamBacklogResponseStamps({
          sessionId: h.sessionId,
          terminalKinds: ["agent_msg"],
        }),
      ).toEqual([]);
      expect(index.lookupSessionTaskEventResponseStamps(h.sessionId)).toEqual([]);
    } finally {
      db.close();
    }
    await expect(
      h.writer.append({
        kind: "agent_msg",
        content: "bad write",
        response_to: stamp,
      } as unknown as StreamEntryInput),
    ).rejects.toThrow("Invalid stream entry payload");
    expect(h.reader.tail(10)).toHaveLength(1);
  });

  it("continues rejecting malformed known stamps instead of treating them as future stamps", async () => {
    const h = harness();
    const entry = await h.writer.append({ kind: "agent_msg", content: "Invalid known reply" });
    for (const kind of ["stream_backlog", "task_event"]) {
      writeFileSync(h.path, JSON.stringify({ ...entry, response_to: { kind } }) + "\n");
      const error = vi.fn();
      expect(
        new StreamReader({ dataDir: h.dataDir, sessionId: h.sessionId, logger: { error } }).tail(1),
      ).toEqual([]);
      expect(error).toHaveBeenCalled();
    }
  });

  it("retains task terminals alongside ordinary replies across current reader entry points", async () => {
    const h = harness();
    const oldEntry = await h.writer.append({ kind: "agent_msg", content: "Earlier reply" });
    const terminal = await h.writer.append({
      kind: "agent_msg",
      content: "Task result",
      response_to: {
        kind: "task_event",
        event_id: "event",
        event_entry_id: createStreamEntryId(),
        task_id: "task",
        task_version: 1,
      },
    });
    const forward = [];
    for await (const entry of h.reader.iterate()) forward.push(entry);
    expect(forward).toEqual([oldEntry, terminal]);
    expect(h.reader.scanReverse().entries).toEqual([oldEntry, terminal]);
    expect(h.reader.tail(10)).toEqual([oldEntry, terminal]);
  });

  it("appends task stamps through the default writer", async () => {
    const h = harness();
    const input: StreamEntryInput = {
      kind: "agent_msg",
      content: "Task result",
      response_to: {
        kind: "task_event",
        event_id: "event",
        event_entry_id: createStreamEntryId(),
        task_id: "task",
        task_version: 1,
      },
    };
    const entry = await h.writer.append(input);
    expect(entry).toMatchObject({ response_to: input.response_to });
    expect(h.reader.tail(1)).toEqual([entry]);
  });
});
