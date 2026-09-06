import type { AgentObservedStreamContent } from "../generation/types.js";
import {
  hydrateStreamEntriesById,
  readStreamEntryAtOffset,
  type StreamCursor,
  type StreamEntry,
  type StreamEntryIndexRecord,
  type StreamEntryIndexRepository,
  type StreamReader,
  type StreamBacklogResponseTo,
  type TaskEventResponseTo,
  type StreamWriter,
} from "../../stream/index.js";
import { CognitionError } from "../../util/errors.js";
import type { SessionId, StreamEntryId } from "../../util/ids.js";
import { orderedPendingResponseUserRecords } from "./backlog-prefix.js";
import {
  CHAT_RESPONSE_TERMINAL_KINDS,
  type ChatResponseWatermarkCoordinator,
} from "./chat-response-watermark.js";
import type { StreamIngestionCoordinator } from "./coordinator.js";

type TerminalEntryIndex = Pick<
  StreamEntryIndexRepository,
  | "lookup"
  | "lookupMany"
  | "lookupSessionEntriesByKind"
  | "lookupSessionStreamBacklogResponseStamps"
>;

export type HydratedStreamBacklogBatch = {
  sourceEntries: readonly StreamEntry[];
  records: readonly StreamEntryIndexRecord[];
};

export type AppendBacklogTerminalInput = {
  sessionId: SessionId;
  sourceEntryIds: readonly StreamEntryId[];
  terminal: { kind: "agent_msg"; content: string } | { kind: "agent_observed"; reason: string };
  audience?: string;
  turnId?: string;
};

export type AppendBacklogTerminalResult = {
  terminalEntry: StreamEntry;
  responseTo: StreamBacklogResponseTo;
  sourceEntries: readonly StreamEntry[];
};

export type SealStaleBacklogInput = {
  sessionId: SessionId;
  staleBefore: number;
  reason?: string;
  audience?: string;
  turnId?: string;
};

export type SealPendingBacklogInput = Omit<SealStaleBacklogInput, "staleBefore">;

export type SealBacklogPrefixInput = SealPendingBacklogInput & {
  sourceEntryIds: readonly StreamEntryId[];
};

export type FindTerminalCoveringEntryResult =
  | { status: "unknown_entry" }
  | { status: "session_mismatch" }
  | { status: "pending" }
  | {
      status: "found";
      terminalEntry: StreamEntry;
      responseTo: StreamBacklogResponseTo;
    };

export type BacklogTerminalServiceOptions = {
  dataDir: string;
  entryIndex: TerminalEntryIndex;
  createStreamReader: (sessionId: SessionId) => StreamReader;
  createStreamWriter: (sessionId: SessionId) => Pick<StreamWriter, "append" | "close">;
  coordinator: ChatResponseWatermarkCoordinator;
  streamIngestionCoordinator?: Pick<StreamIngestionCoordinator, "ingest">;
  onTerminalCommitted?: (terminalEntry: StreamEntry) => void;
};

function requiredEntryIndex(record: StreamEntryIndexRecord, code: string): number {
  if (record.entry_index === null) {
    throw new CognitionError("Inbound batch source entry has no durable order", { code });
  }

  return record.entry_index;
}

function cursorForEntry(entry: StreamEntry): StreamCursor {
  return { ts: entry.timestamp, entryId: entry.id };
}

function watermarkEntryIndex(input: {
  entryIndex: Pick<StreamEntryIndexRepository, "lookup">;
  sessionId: SessionId;
  watermark: StreamCursor | null;
}): number {
  if (input.watermark === null) {
    return -1;
  }

  const record = input.entryIndex.lookup(input.watermark.entryId);
  if (record === null) {
    throw new CognitionError("Inbound batch watermark cursor is missing from the stream index", {
      code: "INBOUND_BATCH_WATERMARK_NOT_INDEXED",
    });
  }
  if (record.session_id !== input.sessionId || record.timestamp !== input.watermark.ts) {
    throw new CognitionError("Inbound batch watermark cursor mismatches the stream index", {
      code: "INBOUND_BATCH_WATERMARK_CURSOR_MISMATCH",
    });
  }

  return requiredEntryIndex(record, "INBOUND_BATCH_WATERMARK_ORDER_MISSING");
}

export async function hydrateStreamBacklogBatch(input: {
  dataDir: string;
  entryIndex: Pick<StreamEntryIndexRepository, "lookupMany">;
  createStreamReader: (sessionId: SessionId) => StreamReader;
  sessionId: SessionId;
  entryIds: readonly StreamEntryId[];
  throughCursorInclusive?: StreamCursor;
}): Promise<HydratedStreamBacklogBatch> {
  if (input.entryIds.length === 0) {
    throw new CognitionError("Inbound batch turns require at least one source entry", {
      code: "INBOUND_BATCH_EMPTY",
    });
  }
  if (new Set(input.entryIds).size !== input.entryIds.length) {
    throw new CognitionError("Inbound batch cannot contain duplicate source entries", {
      code: "INBOUND_BATCH_DUPLICATE_ENTRY",
    });
  }

  const recordsById = input.entryIndex.lookupMany(input.entryIds);
  const entriesById = await hydrateStreamEntriesById({
    dataDir: input.dataDir,
    sessionId: input.sessionId,
    streamEntryIds: input.entryIds,
    entryIndex: input.entryIndex,
    createStreamReader: input.createStreamReader,
  });
  const records: StreamEntryIndexRecord[] = [];
  const sourceEntries: StreamEntry[] = [];

  for (const entryId of input.entryIds) {
    const record = recordsById.get(entryId);
    if (record === undefined) {
      throw new CognitionError("Inbound batch source entry is missing from the stream index", {
        code: "INBOUND_BATCH_ENTRY_MISSING",
      });
    }
    if (record.session_id !== input.sessionId) {
      throw new CognitionError("Inbound batch source entry belongs to a different session", {
        code: "INBOUND_BATCH_SESSION_MISMATCH",
      });
    }
    if (record.kind !== "user_msg") {
      throw new CognitionError("Inbound batch entries must be user_msg stream entries", {
        code: "INBOUND_BATCH_ENTRY_KIND_INVALID",
      });
    }

    const entryIndex = requiredEntryIndex(record, "INBOUND_BATCH_ENTRY_ORDER_MISSING");
    const entry = entriesById.get(entryId);
    if (entry === undefined) {
      throw new CognitionError("Inbound batch source entry is missing from the stream", {
        code: "INBOUND_BATCH_ENTRY_MISSING",
      });
    }
    if (
      entry.session_id !== record.session_id ||
      entry.timestamp !== record.timestamp ||
      entry.kind !== record.kind ||
      (entry.sender_entity_id ?? null) !== record.sender_entity_id
    ) {
      throw new CognitionError("Inbound batch stream entry and index facts disagree", {
        code: "INBOUND_BATCH_INDEX_MISMATCH",
      });
    }
    if (typeof entry.content !== "string") {
      throw new CognitionError("Inbound batch entries must be text-only user messages", {
        code: "INBOUND_BATCH_CONTENT_INVALID",
      });
    }

    records.push(record);
    sourceEntries.push({ ...entry, entry_index: entryIndex });
  }

  for (let index = 1; index < records.length; index += 1) {
    const previous = requiredEntryIndex(records[index - 1]!, "INBOUND_BATCH_ENTRY_ORDER_MISSING");
    const current = requiredEntryIndex(records[index]!, "INBOUND_BATCH_ENTRY_ORDER_MISSING");
    if (current <= previous) {
      throw new CognitionError("Inbound batch source entries must be ordered oldest-first", {
        code: "INBOUND_BATCH_ORDER_INVALID",
      });
    }
  }

  const tail = sourceEntries[sourceEntries.length - 1]!;
  if (
    input.throughCursorInclusive !== undefined &&
    (input.throughCursorInclusive.entryId !== tail.id ||
      input.throughCursorInclusive.ts !== tail.timestamp)
  ) {
    throw new CognitionError(
      "Inbound batch through cursor does not match the hydrated tail entry",
      {
        code: "INBOUND_BATCH_CURSOR_MISMATCH",
      },
    );
  }

  return { sourceEntries, records };
}

export function buildStreamBacklogResponseTo(input: {
  coordinator: ChatResponseWatermarkCoordinator;
  entryIndex: Pick<StreamEntryIndexRepository, "lookup" | "lookupSessionEntriesByKind"> &
    Partial<Pick<StreamEntryIndexRepository, "lookupSessionStreamBacklogResponseStamps">>;
  sessionId: SessionId;
  sourceEntries: readonly StreamEntry[];
  records: readonly StreamEntryIndexRecord[];
  sourceEntryIds: readonly StreamEntryId[];
}): StreamBacklogResponseTo {
  const throughEntry = input.sourceEntries[input.sourceEntries.length - 1];
  const throughRecord = input.records[input.records.length - 1];
  if (throughEntry === undefined || throughRecord === undefined) {
    throw new CognitionError("Inbound batch turns require at least one source entry", {
      code: "INBOUND_BATCH_EMPTY",
    });
  }

  const throughCursorInclusive = cursorForEntry(throughEntry);
  const alreadyStamped = input.entryIndex
    .lookupSessionStreamBacklogResponseStamps?.({
      sessionId: input.sessionId,
      terminalKinds: CHAT_RESPONSE_TERMINAL_KINDS,
    })
    .some((record) => {
      if (record.response_to_source_entry_ids === null) {
        return false;
      }
      try {
        const ids = JSON.parse(record.response_to_source_entry_ids) as unknown;
        return (
          Array.isArray(ids) &&
          ids.length === input.sourceEntryIds.length &&
          ids.every((entryId, index) => entryId === input.sourceEntryIds[index])
        );
      } catch {
        return false;
      }
    });
  if (alreadyStamped === true) {
    throw new CognitionError("Inbound batch already has a terminal response stamp", {
      code: "INBOUND_BATCH_ALREADY_RESPONDED",
    });
  }
  const assertNoStamp = (fromCursorExclusive: StreamCursor | null) => {
    const existing = input.coordinator.findTerminalStampForBatch({
      sessionId: input.sessionId,
      fromCursorExclusive,
      throughCursorInclusive,
      sourceEntryIds: input.sourceEntryIds,
      count: input.sourceEntryIds.length,
    });
    if (existing !== null) {
      throw new CognitionError("Inbound batch already has a terminal response stamp", {
        code: "INBOUND_BATCH_ALREADY_RESPONDED",
      });
    }
  };

  assertNoStamp(input.coordinator.getWatermark(input.sessionId));
  const reconciled = input.coordinator.reconcile(input.sessionId);
  assertNoStamp(reconciled.watermark);
  const watermarkIndex = watermarkEntryIndex({
    entryIndex: input.entryIndex,
    sessionId: input.sessionId,
    watermark: reconciled.watermark,
  });
  const throughIndex = requiredEntryIndex(throughRecord, "INBOUND_BATCH_ENTRY_ORDER_MISSING");
  if (throughIndex <= watermarkIndex) {
    throw new CognitionError("Inbound batch through cursor is not after the response watermark", {
      code: "INBOUND_BATCH_STALE",
    });
  }

  const expectedIds = orderedPendingResponseUserRecords({
    records: input.entryIndex.lookupSessionEntriesByKind({
      sessionId: input.sessionId,
      kind: "user_msg",
    }),
    afterEntryIndex: watermarkIndex,
    throughEntryIndexInclusive: throughIndex,
    orderMissingCode: "INBOUND_BATCH_CONTIGUITY_ORDER_MISSING",
  }).map((record) => record.entry_id as StreamEntryId);
  if (
    expectedIds.length !== input.sourceEntryIds.length ||
    expectedIds.some((entryId, index) => entryId !== input.sourceEntryIds[index])
  ) {
    throw new CognitionError(
      "Inbound batch must be the contiguous oldest-first unresponded user_msg prefix",
      { code: "INBOUND_BATCH_NOT_CONTIGUOUS" },
    );
  }

  return {
    kind: "stream_backlog",
    from_cursor_exclusive: reconciled.watermark,
    through_cursor_inclusive: throughCursorInclusive,
    source_entry_ids: [...input.sourceEntryIds],
    count: input.sourceEntryIds.length,
  };
}

export class BacklogTerminalService {
  constructor(private readonly options: BacklogTerminalServiceOptions) {}

  async appendTaskEventTerminal(input: {
    sessionId: SessionId;
    responseTo: TaskEventResponseTo;
    content: string;
    audience?: string;
  }): Promise<StreamEntry> {
    const writer = this.options.createStreamWriter(input.sessionId);
    try {
      return await writer.append({
        kind: "agent_msg",
        content: input.content,
        response_to: input.responseTo,
        ...(input.audience === undefined ? {} : { audience: input.audience }),
      });
    } finally {
      writer.close();
    }
  }

  async ingestTaskEventTerminal(terminalEntry: StreamEntry): Promise<void> {
    if (terminalEntry.response_to?.kind !== "task_event") {
      throw new CognitionError("Task terminal requires a task_event response stamp", {
        code: "TASK_EVENT_TERMINAL_STAMP_INVALID",
      });
    }
    const result = await this.options.streamIngestionCoordinator?.ingest(terminalEntry.session_id, {
      answeredWindow: {
        responseTo: terminalEntry.response_to,
        terminalCursor: cursorForEntry(terminalEntry),
      },
    });
    if (result?.error !== undefined) throw result.error;
  }

  hydrateBacklogBatch(input: {
    sessionId: SessionId;
    entryIds: readonly StreamEntryId[];
    throughCursorInclusive?: StreamCursor;
  }): Promise<HydratedStreamBacklogBatch> {
    return hydrateStreamBacklogBatch({
      dataDir: this.options.dataDir,
      entryIndex: this.options.entryIndex,
      createStreamReader: this.options.createStreamReader,
      ...input,
    });
  }

  async appendBacklogTerminal(
    input: AppendBacklogTerminalInput,
  ): Promise<AppendBacklogTerminalResult> {
    const batch = await this.hydrateBacklogBatch({
      sessionId: input.sessionId,
      entryIds: input.sourceEntryIds,
    });
    return this.appendHydratedTerminal(input, batch, true);
  }

  async sealStaleBacklog(
    input: SealStaleBacklogInput,
  ): Promise<AppendBacklogTerminalResult | null> {
    return this.sealPendingPrefix(input, input.staleBefore);
  }

  async sealPendingBacklog(
    input: SealPendingBacklogInput,
  ): Promise<AppendBacklogTerminalResult | null> {
    return this.sealPendingPrefix(input);
  }

  async sealBacklogPrefix(input: SealBacklogPrefixInput): Promise<AppendBacklogTerminalResult> {
    const batch = await this.hydrateBacklogBatch({
      sessionId: input.sessionId,
      entryIds: input.sourceEntryIds,
    });
    return this.appendHydratedTerminal(
      {
        sessionId: input.sessionId,
        sourceEntryIds: input.sourceEntryIds,
        terminal: {
          kind: "agent_observed",
          reason: input.reason ?? "Legacy inbox backlog sealed without a response",
        },
        ...(input.audience === undefined ? {} : { audience: input.audience }),
        ...(input.turnId === undefined ? {} : { turnId: input.turnId }),
      },
      batch,
      false,
    );
  }

  private async sealPendingPrefix(
    input: SealPendingBacklogInput,
    staleBefore?: number,
  ): Promise<AppendBacklogTerminalResult | null> {
    const reconciled = this.options.coordinator.reconcile(input.sessionId);
    const afterEntryIndex = watermarkEntryIndex({
      entryIndex: this.options.entryIndex,
      sessionId: input.sessionId,
      watermark: reconciled.watermark,
    });
    const pendingRecords = orderedPendingResponseUserRecords({
      records: this.options.entryIndex.lookupSessionEntriesByKind({
        sessionId: input.sessionId,
        kind: "user_msg",
      }),
      afterEntryIndex,
      orderMissingCode: "INBOUND_BATCH_CONTIGUITY_ORDER_MISSING",
    });
    if (pendingRecords.length === 0) {
      return null;
    }

    const allEntryIds = pendingRecords.map((record) => record.entry_id as StreamEntryId);
    const all = await this.hydrateBacklogBatch({
      sessionId: input.sessionId,
      entryIds: allEntryIds,
    });
    const firstExcludedIndex =
      staleBefore === undefined
        ? -1
        : all.sourceEntries.findIndex(
            (entry) => (entry.observed_at ?? entry.timestamp) >= staleBefore,
          );
    const prefixLength = firstExcludedIndex === -1 ? all.sourceEntries.length : firstExcludedIndex;
    if (prefixLength === 0) {
      return null;
    }

    const batch = {
      sourceEntries: all.sourceEntries.slice(0, prefixLength),
      records: all.records.slice(0, prefixLength),
    };
    return this.appendHydratedTerminal(
      {
        sessionId: input.sessionId,
        sourceEntryIds: batch.sourceEntries.map((entry) => entry.id),
        terminal: {
          kind: "agent_observed",
          reason:
            input.reason ??
            (staleBefore === undefined
              ? "Pending inbox backlog sealed without a response"
              : "Stale inbox backlog sealed without a response"),
        },
        ...(input.audience === undefined ? {} : { audience: input.audience }),
        ...(input.turnId === undefined ? {} : { turnId: input.turnId }),
      },
      batch,
      false,
    );
  }

  findTerminalCoveringEntry(input: {
    sessionId: SessionId;
    entryId: StreamEntryId;
  }): FindTerminalCoveringEntryResult {
    const sourceRecord = this.options.entryIndex.lookup(input.entryId);
    if (sourceRecord === null) {
      return { status: "unknown_entry" };
    }
    if (sourceRecord.session_id !== input.sessionId) {
      return { status: "session_mismatch" };
    }
    if (sourceRecord.kind !== "user_msg") {
      return { status: "unknown_entry" };
    }

    const terminalRecord = this.options.entryIndex
      .lookupSessionStreamBacklogResponseStamps({
        sessionId: input.sessionId,
        terminalKinds: CHAT_RESPONSE_TERMINAL_KINDS,
      })
      .find((record) => {
        if (record.response_to_source_entry_ids === null) {
          return false;
        }
        try {
          const ids = JSON.parse(record.response_to_source_entry_ids) as unknown;
          return Array.isArray(ids) && ids.includes(input.entryId);
        } catch {
          return false;
        }
      });
    if (terminalRecord === undefined) {
      return { status: "pending" };
    }

    const terminalEntry = readStreamEntryAtOffset({
      dataDir: this.options.dataDir,
      sessionId: input.sessionId,
      byteOffset: terminalRecord.byte_offset,
    });
    if (terminalEntry?.response_to?.kind !== "stream_backlog") {
      return { status: "pending" };
    }
    return { status: "found", terminalEntry, responseTo: terminalEntry.response_to };
  }

  private async appendHydratedTerminal(
    input: AppendBacklogTerminalInput,
    batch: HydratedStreamBacklogBatch,
    exactIngestion: boolean,
  ): Promise<AppendBacklogTerminalResult> {
    const responseTo = buildStreamBacklogResponseTo({
      coordinator: this.options.coordinator,
      entryIndex: this.options.entryIndex,
      sessionId: input.sessionId,
      sourceEntries: batch.sourceEntries,
      records: batch.records,
      sourceEntryIds: input.sourceEntryIds,
    });
    const writer = this.options.createStreamWriter(input.sessionId);
    let terminalEntry: StreamEntry | undefined;
    try {
      try {
        const common = {
          ...(input.turnId === undefined ? {} : { turn_id: input.turnId }),
          ...(input.audience === undefined ? {} : { audience: input.audience }),
          response_to: responseTo,
        };
        if (input.terminal.kind === "agent_msg") {
          terminalEntry = await writer.append({
            ...common,
            kind: "agent_msg",
            content: input.terminal.content,
          });
        } else {
          terminalEntry = await writer.append({
            ...common,
            kind: "agent_observed",
            content: {
              reason: input.terminal.reason,
              ...(input.turnId === undefined ? {} : { turn_id: input.turnId }),
              user_entry_id: input.sourceEntryIds[0],
              user_entry_ids: [...input.sourceEntryIds],
            } satisfies AgentObservedStreamContent,
          });
        }
      } finally {
        writer.close();
      }

      this.options.coordinator.advanceThrough(input.sessionId, responseTo.through_cursor_inclusive);
      const ingestion = this.options.streamIngestionCoordinator;
      if (ingestion !== undefined) {
        const task = exactIngestion
          ? ingestion.ingest(input.sessionId, {
              answeredWindow: {
                responseTo,
                terminalCursor: cursorForEntry(terminalEntry),
              },
            })
          : ingestion.ingest(input.sessionId);
        void task.catch((error) => console.error("Live stream ingestion failed", error));
      }

      return { terminalEntry, responseTo, sourceEntries: batch.sourceEntries };
    } finally {
      if (terminalEntry !== undefined) {
        try {
          this.options.onTerminalCommitted?.(terminalEntry);
        } catch (error) {
          console.error("Backlog terminal commit observer failed", error);
        }
      }
    }
  }
}
