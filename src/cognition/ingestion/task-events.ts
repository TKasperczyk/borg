import {
  readStreamEntryAtOffset,
  taskEventSchema,
  type StreamEntry,
  type StreamEntryIndexRepository,
  type StreamWriter,
  type TaskEvent,
} from "../../stream/index.js";
import type { SessionId, StreamEntryId } from "../../util/ids.js";
import { repairPoisonedSessionBeforeDedup } from "./repair-before-dedup.js";

export type StoredTaskEvent = { entry: StreamEntry; event: TaskEvent };

/** Durable event handles. All language interpretation belongs to the task-result runner. */
export class TaskEventService {
  constructor(
    private readonly options: {
      dataDir: string;
      entryIndex: Pick<
        StreamEntryIndexRepository,
        | "lookup"
        | "lookupSessionEntriesByKind"
        | "listSessionIdsWithTaskEvents"
        | "lookupSessionTaskEventResponseStamps"
        | "isPoisoned"
      >;
      repairSessionStreamEntryIndex: (sessionId: SessionId) => Promise<unknown>;
      createStreamWriter: (sessionId: SessionId) => Pick<StreamWriter, "append" | "close">;
    },
  ) {}

  readEntry(sessionId: SessionId, entryId: string): StreamEntry | null {
    const record = this.options.entryIndex.lookup(entryId);
    if (record === null || record.session_id !== sessionId) return null;
    return readStreamEntryAtOffset({
      dataDir: this.options.dataDir,
      sessionId,
      byteOffset: record.byte_offset,
    });
  }

  listSessionIds(): SessionId[] {
    return this.options.entryIndex.listSessionIdsWithTaskEvents();
  }

  list(sessionId: SessionId): StoredTaskEvent[] {
    const result: StoredTaskEvent[] = [];
    for (const record of this.options.entryIndex.lookupSessionEntriesByKind({
      sessionId,
      kind: "internal_event",
    })) {
      const entry = this.readEntry(sessionId, record.entry_id);
      const parsed = taskEventSchema.safeParse(entry?.metadata?.task_event);
      if (entry !== null && parsed.success) result.push({ entry, event: parsed.data });
    }
    return result.sort((a, b) => a.entry.entry_index! - b.entry.entry_index!);
  }

  listTerminals(sessionId: SessionId): StreamEntry[] {
    return this.options.entryIndex
      .lookupSessionTaskEventResponseStamps(sessionId)
      .flatMap((record) => {
        const entry = this.readEntry(sessionId, record.entry_id);
        return entry?.response_to?.kind === "task_event" ? [entry] : [];
      });
  }

  findTerminal(sessionId: SessionId, event: StoredTaskEvent): StreamEntry | null {
    return (
      this.listTerminals(sessionId).find((terminal) => {
        const stamp = terminal.response_to;
        return (
          stamp?.kind === "task_event" &&
          stamp.event_entry_id === event.entry.id &&
          stamp.event_id === event.event.event_id &&
          stamp.task_id === event.event.task_id &&
          stamp.task_version === event.event.task_version
        );
      }) ?? null
    );
  }

  listUnanswered(sessionId: SessionId): StoredTaskEvent[] {
    const terminals = this.listTerminals(sessionId);
    return this.list(sessionId).filter(
      ({ entry, event }) =>
        !terminals.some((terminal) => {
          const stamp = terminal.response_to;
          return (
            stamp?.kind === "task_event" &&
            stamp.event_entry_id === entry.id &&
            stamp.event_id === event.event_id &&
            stamp.task_id === event.task_id &&
            stamp.task_version === event.task_version
          );
        }),
    );
  }

  // The caller serializes enqueues under the tenant chain, just like MessageEnqueuer.
  async enqueue(input: { sessionId: SessionId; event: TaskEvent; audience?: string }): Promise<{
    status: "enqueued" | "duplicate";
    entry_id: StreamEntryId;
  }> {
    const event = taskEventSchema.parse(input.event);
    if (this.options.entryIndex.isPoisoned(input.sessionId)) {
      await repairPoisonedSessionBeforeDedup(
        input.sessionId,
        this.options.repairSessionStreamEntryIndex,
      );
    }
    const existing = this.list(input.sessionId).find(
      (item) => item.event.event_id === event.event_id,
    );
    if (existing !== undefined) return { status: "duplicate", entry_id: existing.entry.id };
    const writer = this.options.createStreamWriter(input.sessionId);
    try {
      const entry = await writer.append({
        kind: "internal_event",
        content: `Agent task event: ${event.kind}.`,
        metadata: { task_event: event },
        ...(input.audience === undefined ? {} : { audience: input.audience }),
      });
      return { status: "enqueued", entry_id: entry.id };
    } finally {
      writer.close();
    }
  }
}

export type TaskEventCatchUpRunner = {
  run(input: { sessionId: SessionId; taskEvent: StoredTaskEvent }): Promise<void>;
  reconcile(sessionId: SessionId): Promise<void>;
};
