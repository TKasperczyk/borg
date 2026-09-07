import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { openDatabase } from "../storage/sqlite/index.js";
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
