import {
  streamCursorFromWatermark,
  streamCursorsEqual,
  type StreamCursor,
  type StreamEntryIndexRecord,
  type StreamEntryIndexRepository,
  type StreamEntryKind,
  type StreamWatermarkRepository,
} from "../../stream/index.js";
import { CognitionError } from "../../util/errors.js";
import type { SessionId, StreamEntryId } from "../../util/ids.js";
import { orderedPendingResponseUserRecords } from "./backlog-prefix.js";

export const CHAT_RESPONSE_PROCESS_NAME = "chat-response";
export const CHAT_RESPONSE_TERMINAL_KINDS = [
  "agent_msg",
  "agent_suppressed",
  "agent_observed",
] as const satisfies readonly StreamEntryKind[];

export type ChatResponseTerminalKind = (typeof CHAT_RESPONSE_TERMINAL_KINDS)[number];

type LoggerLike = Pick<Console, "warn">;

export type ChatResponseWatermarkCoordinatorOptions = {
  watermarkRepository: Pick<StreamWatermarkRepository, "get" | "set">;
  entryIndex: Pick<
    StreamEntryIndexRepository,
    | "lookup"
    | "lookupSessionEntriesByKind"
    | "lookupSessionStreamBacklogResponseStamps"
    | "lookupExactStreamBacklogResponseStamp"
  >;
  logger?: LoggerLike;
};

export type ChatResponseReconcileResult = {
  watermark: StreamCursor | null;
  advancedThrough: StreamCursor | null;
  appliedStamps: number;
};

export type FindTerminalStampForBatchInput = {
  sessionId: SessionId;
  fromCursorExclusive: StreamCursor | null;
  throughCursorInclusive: StreamCursor;
  sourceEntryIds: readonly StreamEntryId[];
  count: number;
};

export type AdvanceChatResponseWatermarkResult = {
  advanced: boolean;
  watermark: StreamCursor;
};

function cursorFromFromColumns(record: StreamEntryIndexRecord): StreamCursor | null | undefined {
  const ts = record.response_to_from_cursor_ts;
  const entryId = record.response_to_from_cursor_entry_id;

  if (ts === null && entryId === null) {
    return null;
  }

  if (ts === null || entryId === null) {
    return undefined;
  }

  return { ts, entryId };
}

function cursorFromThroughColumns(record: StreamEntryIndexRecord): StreamCursor | null {
  const ts = record.response_to_through_cursor_ts;
  const entryId = record.response_to_through_cursor_entry_id;

  if (ts === null || entryId === null) {
    return null;
  }

  return { ts, entryId };
}

function parsedSourceEntryIds(record: StreamEntryIndexRecord): readonly StreamEntryId[] | null {
  if (record.response_to_source_entry_ids === null) {
    return null;
  }

  try {
    const parsed = JSON.parse(record.response_to_source_entry_ids) as unknown;

    return Array.isArray(parsed) && parsed.every((entryId) => typeof entryId === "string")
      ? (parsed as StreamEntryId[])
      : null;
  } catch {
    return null;
  }
}

function parsedSourceEntryIdsLength(record: StreamEntryIndexRecord): number | null {
  return parsedSourceEntryIds(record)?.length ?? null;
}

function stampStartsAt(record: StreamEntryIndexRecord, cursor: StreamCursor | null): boolean {
  const fromCursor = cursorFromFromColumns(record);

  return fromCursor !== undefined && streamCursorsEqual(fromCursor, cursor);
}

function stampIsStructurallyUsable(record: StreamEntryIndexRecord): boolean {
  const sourceEntryIdsLength = parsedSourceEntryIdsLength(record);

  return (
    record.response_to_count !== null &&
    Number.isInteger(record.response_to_count) &&
    record.response_to_count > 0 &&
    sourceEntryIdsLength !== null &&
    sourceEntryIdsLength === record.response_to_count &&
    cursorFromThroughColumns(record) !== null
  );
}

export class ChatResponseWatermarkCoordinator {
  private readonly logger: LoggerLike;

  constructor(private readonly options: ChatResponseWatermarkCoordinatorOptions) {
    this.logger = options.logger ?? console;
  }

  getWatermark(sessionId: SessionId): StreamCursor | null {
    return streamCursorFromWatermark(
      this.options.watermarkRepository.get(CHAT_RESPONSE_PROCESS_NAME, sessionId),
    );
  }

  reconcile(sessionId: SessionId): ChatResponseReconcileResult {
    const stamps = this.options.entryIndex.lookupSessionStreamBacklogResponseStamps({
      sessionId,
      terminalKinds: CHAT_RESPONSE_TERMINAL_KINDS,
    });
    const consumedStampIds = new Set<string>();
    let watermark = this.getWatermark(sessionId);
    let advancedThrough: StreamCursor | null = null;
    let appliedStamps = 0;

    while (true) {
      const stamp = stamps.find(
        (candidate) =>
          !consumedStampIds.has(candidate.entry_id) && stampStartsAt(candidate, watermark),
      );

      if (stamp === undefined) {
        break;
      }

      consumedStampIds.add(stamp.entry_id);

      if (!stampIsStructurallyUsable(stamp)) {
        this.logSkippedStamp(stamp, "malformed");
        continue;
      }

      const throughCursor = cursorFromThroughColumns(stamp);

      if (throughCursor === null) {
        this.logSkippedStamp(stamp, "missing_through_cursor");
        continue;
      }

      if (
        !this.stampMatchesContiguousQueuedPrefix({
          sessionId,
          stamp,
          fromCursorExclusive: watermark,
          throughCursorInclusive: throughCursor,
        })
      ) {
        this.logSkippedStamp(stamp, "non_contiguous_prefix");
        continue;
      }

      const advanceResult = this.advanceThrough(sessionId, throughCursor);
      watermark = advanceResult.watermark;

      if (advanceResult.advanced) {
        advancedThrough = throughCursor;
        appliedStamps += 1;
      }
    }

    return {
      watermark,
      advancedThrough,
      appliedStamps,
    };
  }

  findTerminalStampForBatch(input: FindTerminalStampForBatchInput): StreamEntryIndexRecord | null {
    return this.options.entryIndex.lookupExactStreamBacklogResponseStamp({
      ...input,
      terminalKinds: CHAT_RESPONSE_TERMINAL_KINDS,
    });
  }

  advanceThrough(
    sessionId: SessionId,
    throughCursor: StreamCursor,
  ): AdvanceChatResponseWatermarkResult {
    const current = this.getWatermark(sessionId);
    const targetEntryIndex = this.resolveCursorEntryIndex(sessionId, throughCursor, "target");

    if (current === null) {
      this.setWatermark(sessionId, throughCursor);
      return {
        advanced: true,
        watermark: throughCursor,
      };
    }

    const currentEntryIndex = this.resolveCursorEntryIndex(sessionId, current, "current");

    if (targetEntryIndex <= currentEntryIndex) {
      return {
        advanced: false,
        watermark: current,
      };
    }

    this.setWatermark(sessionId, throughCursor);
    return {
      advanced: true,
      watermark: throughCursor,
    };
  }

  cursorEntryIndex(sessionId: SessionId, cursor: StreamCursor, label = "cursor"): number {
    return this.resolveCursorEntryIndex(sessionId, cursor, label);
  }

  compareCursors(sessionId: SessionId, left: StreamCursor, right: StreamCursor): number {
    return (
      this.resolveCursorEntryIndex(sessionId, left, "left") -
      this.resolveCursorEntryIndex(sessionId, right, "right")
    );
  }

  private resolveCursorEntryIndex(
    sessionId: SessionId,
    cursor: StreamCursor,
    label: string,
  ): number {
    const record = this.options.entryIndex.lookup(cursor.entryId);

    if (record === null) {
      throw new CognitionError(`Chat response ${label} cursor is missing from the stream index`, {
        code: "CHAT_RESPONSE_WATERMARK_CURSOR_NOT_INDEXED",
      });
    }

    if (record.session_id !== sessionId) {
      throw new CognitionError(`Chat response ${label} cursor belongs to a different session`, {
        code: "CHAT_RESPONSE_WATERMARK_CURSOR_SESSION_MISMATCH",
      });
    }

    if (record.timestamp !== cursor.ts) {
      throw new CognitionError(`Chat response ${label} cursor timestamp mismatches stream index`, {
        code: "CHAT_RESPONSE_CURSOR_TS_MISMATCH",
      });
    }

    if (record.entry_index === null) {
      throw new CognitionError(`Chat response ${label} cursor has no entry_index`, {
        code: "CHAT_RESPONSE_WATERMARK_CURSOR_ORDER_MISSING",
      });
    }

    return record.entry_index;
  }

  private stampMatchesContiguousQueuedPrefix(input: {
    sessionId: SessionId;
    stamp: StreamEntryIndexRecord;
    fromCursorExclusive: StreamCursor | null;
    throughCursorInclusive: StreamCursor;
  }): boolean {
    const stampedSourceEntryIds = parsedSourceEntryIds(input.stamp);

    if (stampedSourceEntryIds === null || stampedSourceEntryIds.length === 0) {
      return false;
    }

    const watermarkIndex =
      input.fromCursorExclusive === null
        ? -1
        : this.resolveCursorEntryIndex(input.sessionId, input.fromCursorExclusive, "current");
    const throughIndex = this.resolveCursorEntryIndex(
      input.sessionId,
      input.throughCursorInclusive,
      "through",
    );
    const expectedSourceEntryIds = orderedPendingResponseUserRecords({
      records: this.options.entryIndex.lookupSessionEntriesByKind({
        sessionId: input.sessionId,
        kind: "user_msg",
      }),
      afterEntryIndex: watermarkIndex,
      throughEntryIndexInclusive: throughIndex,
      orderMissingCode: "CHAT_RESPONSE_WATERMARK_QUEUED_PREFIX_ORDER_MISSING",
    }).map((record) => record.entry_id as StreamEntryId);

    return (
      expectedSourceEntryIds.length === stampedSourceEntryIds.length &&
      expectedSourceEntryIds[expectedSourceEntryIds.length - 1] ===
        input.throughCursorInclusive.entryId &&
      expectedSourceEntryIds.every((entryId, index) => entryId === stampedSourceEntryIds[index])
    );
  }

  private setWatermark(sessionId: SessionId, cursor: StreamCursor): void {
    this.options.watermarkRepository.set(CHAT_RESPONSE_PROCESS_NAME, sessionId, {
      lastTs: cursor.ts,
      lastEntryId: cursor.entryId,
    });
  }

  private logSkippedStamp(record: StreamEntryIndexRecord, reason: string): void {
    this.logger.warn("Skipping unusable chat response terminal stamp", {
      reason,
      sessionId: record.session_id,
      entryId: record.entry_id,
    });
  }
}
