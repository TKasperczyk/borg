import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it, vi } from "vitest";
import { openDatabase } from "../../storage/sqlite/index.js";
import {
  StreamEntryIndexRepository,
  StreamReader,
  StreamWriter,
  streamEntryIndexMigrations,
  taskEventSchema,
} from "../../stream/index.js";
import { createSessionId } from "../../util/ids.js";
import { TaskEventService } from "./task-events.js";

it("repairs a committed task event before duplicate lookup after index and immediate repair failures", async () => {
  const dataDir = mkdtempSync(join(tmpdir(), "borg-task-event-poison-"));
  const db = openDatabase(join(dataDir, "borg.db"), { migrations: streamEntryIndexMigrations });
  const entryIndex = new StreamEntryIndexRepository({ db, dataDir });
  const sessionId = createSessionId();
  const reader = new StreamReader({ dataDir, sessionId });
  const backfill = entryIndex.backfillSession.bind(entryIndex);
  let repairAvailable = false;
  const repair = vi.fn(async () => {
    if (!repairAvailable) throw new Error("index repair unavailable");
    return backfill(sessionId);
  });
  const record = vi.spyOn(entryIndex, "recordEntry").mockImplementationOnce(() => {
    throw new Error("index update failed after append committed");
  });
  const createStreamWriter = vi.fn(
    () =>
      new StreamWriter({
        dataDir,
        sessionId,
        entryIndex,
        repairSession: repair,
        logger: { error: vi.fn() },
      }),
  );
  const service = new TaskEventService({
    dataDir,
    entryIndex,
    repairSessionStreamEntryIndex: repair,
    createStreamWriter,
  });
  const input = {
    sessionId,
    event: taskEventSchema.parse({
      schema_version: 1,
      event_id: "event",
      task_id: "task",
      task_version: 1,
      kind: "task_completed",
      occurred_at: "2026-09-06T12:00:00Z",
      outcome: { status: "succeeded", summary: "Done" },
      origin: { source_entry_ids: [] },
    }),
  };
  try {
    await expect(service.enqueue(input)).rejects.toMatchObject({ code: "STREAM_INDEX_POISONED" });
    expect(entryIndex.isPoisoned(sessionId)).toBe(true);
    const [committed] = reader.tail(10);
    expect(committed).toBeDefined();
    await expect(service.enqueue(input)).rejects.toMatchObject({ code: "STREAM_INDEX_POISONED" });
    expect(reader.tail(10)).toHaveLength(1);
    repairAvailable = true;
    await expect(service.enqueue(input)).resolves.toEqual({
      status: "duplicate",
      entry_id: committed!.id,
    });
    expect(entryIndex.isPoisoned(sessionId)).toBe(false);
    expect(service.list(sessionId)).toHaveLength(1);
    expect(reader.tail(10)).toHaveLength(1);
    expect(createStreamWriter).toHaveBeenCalledTimes(1);
    expect(record).toHaveBeenCalledTimes(1);
  } finally {
    record.mockRestore();
    db.close();
    rmSync(dataDir, { recursive: true, force: true });
  }
});
