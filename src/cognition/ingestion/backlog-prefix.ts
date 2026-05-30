import {
  type StreamCursor,
  type StreamEntry,
  type StreamEntryIndexRecord,
  type StreamEntryIndexRepository,
  type StreamReader,
} from "../../stream/index.js";
import { CognitionError } from "../../util/errors.js";
import type { SessionId, StreamEntryId } from "../../util/ids.js";
import { estimatePromptTokens, stringifyPromptContent } from "../../util/token-estimate.js";

export type BacklogPrefixCaps = {
  maxMessages?: number;
  maxTokens?: number;
  maxChars?: number;
};

export type BacklogPrefixResult = {
  fromCursorExclusive: StreamCursor | null;
  entryIds: readonly StreamEntryId[];
  throughCursorInclusive: StreamCursor | null;
  includedCount: number;
  remainingCount: number;
  hasMore: boolean;
  estimatedTokens: number;
  estimatedChars: number;
};

export type ChatResponseBacklogPrefixBuilderOptions = {
  entryIndex: Pick<StreamEntryIndexRepository, "lookup" | "lookupSessionEntriesByKind">;
  createStreamReader: (sessionId: SessionId) => Pick<StreamReader, "iterate">;
  caps?: BacklogPrefixCaps;
};

export type BuildBacklogPrefixInput = {
  sessionId: SessionId;
  fromCursorExclusive: StreamCursor | null;
  caps?: BacklogPrefixCaps;
};

type ResolvedBacklogPrefixCaps = Required<BacklogPrefixCaps>;

const DEFAULT_BACKLOG_PREFIX_CAPS: ResolvedBacklogPrefixCaps = {
  maxMessages: 16,
  maxTokens: 6_000,
  maxChars: 24_000,
};

function resolveCaps(
  baseCaps: BacklogPrefixCaps | undefined,
  overrideCaps: BacklogPrefixCaps | undefined,
): ResolvedBacklogPrefixCaps {
  return {
    maxMessages:
      overrideCaps?.maxMessages ?? baseCaps?.maxMessages ?? DEFAULT_BACKLOG_PREFIX_CAPS.maxMessages,
    maxTokens:
      overrideCaps?.maxTokens ?? baseCaps?.maxTokens ?? DEFAULT_BACKLOG_PREFIX_CAPS.maxTokens,
    maxChars: overrideCaps?.maxChars ?? baseCaps?.maxChars ?? DEFAULT_BACKLOG_PREFIX_CAPS.maxChars,
  };
}

export function requiredQueuedUserRecordEntryIndex(
  record: StreamEntryIndexRecord,
  code = "CHAT_RESPONSE_BACKLOG_ENTRY_ORDER_MISSING",
): number {
  if (record.entry_index === null) {
    throw new CognitionError("Chat response backlog entry has no durable order", {
      code,
    });
  }

  return record.entry_index;
}

export function isPendingResponseUserRecord(record: StreamEntryIndexRecord): boolean {
  return record.kind === "user_msg" && record.turn_id === null;
}

export function orderedPendingResponseUserRecords(input: {
  records: readonly StreamEntryIndexRecord[];
  afterEntryIndex: number;
  throughEntryIndexInclusive?: number;
  orderMissingCode?: string;
}): StreamEntryIndexRecord[] {
  return input.records
    .filter(isPendingResponseUserRecord)
    .filter((record) => {
      const entryIndex = requiredQueuedUserRecordEntryIndex(record, input.orderMissingCode);

      return (
        entryIndex > input.afterEntryIndex &&
        (input.throughEntryIndexInclusive === undefined ||
          entryIndex <= input.throughEntryIndexInclusive)
      );
    })
    .sort(
      (left, right) =>
        requiredQueuedUserRecordEntryIndex(left, input.orderMissingCode) -
          requiredQueuedUserRecordEntryIndex(right, input.orderMissingCode) ||
        left.byte_offset - right.byte_offset,
    );
}

function cursorForRecord(record: StreamEntryIndexRecord): StreamCursor {
  return {
    ts: record.timestamp,
    entryId: record.entry_id as StreamEntryId,
  };
}

function estimateEntryPromptTokens(entry: StreamEntry): number {
  if (entry.token_estimate !== undefined) {
    return entry.token_estimate;
  }

  return estimatePromptTokens(stringifyPromptContent(entry.content));
}

function estimateEntryPromptChars(entry: StreamEntry): number {
  return stringifyPromptContent(entry.content).length;
}

function wouldExceedCaps(input: {
  caps: ResolvedBacklogPrefixCaps;
  nextCount: number;
  nextTokens: number;
  nextChars: number;
}): boolean {
  return (
    input.nextCount > input.caps.maxMessages ||
    input.nextTokens > input.caps.maxTokens ||
    input.nextChars > input.caps.maxChars
  );
}

class PendingUserEntryHydrator {
  private readonly iterator: AsyncGenerator<StreamEntry>;
  private readonly wantedIds: ReadonlySet<string>;
  private readonly entries = new Map<string, StreamEntry>();
  private done = false;

  constructor(
    reader: Pick<StreamReader, "iterate">,
    pendingRecords: readonly StreamEntryIndexRecord[],
  ) {
    this.iterator = reader.iterate({ kinds: ["user_msg"] });
    this.wantedIds = new Set(pendingRecords.map((record) => record.entry_id));
  }

  async read(record: StreamEntryIndexRecord): Promise<StreamEntry> {
    while (!this.entries.has(record.entry_id) && !this.done) {
      const next = await this.iterator.next();

      if (next.done === true) {
        this.done = true;
        break;
      }

      if (this.wantedIds.has(next.value.id)) {
        this.entries.set(next.value.id, next.value);
      }
    }

    const entry = this.entries.get(record.entry_id);

    if (entry === undefined) {
      throw new CognitionError("Chat response backlog entry is missing from the stream", {
        code: "CHAT_RESPONSE_BACKLOG_ENTRY_MISSING",
      });
    }

    if (
      entry.session_id !== record.session_id ||
      entry.timestamp !== record.timestamp ||
      entry.kind !== record.kind
    ) {
      throw new CognitionError("Chat response backlog stream entry and index facts disagree", {
        code: "CHAT_RESPONSE_BACKLOG_INDEX_MISMATCH",
      });
    }

    return entry;
  }

  async close(): Promise<void> {
    if (!this.done) {
      await this.iterator.return(undefined);
    }
  }
}

export class ChatResponseBacklogPrefixBuilder {
  constructor(private readonly options: ChatResponseBacklogPrefixBuilderOptions) {}

  async build(input: BuildBacklogPrefixInput): Promise<BacklogPrefixResult> {
    const caps = resolveCaps(this.options.caps, input.caps);
    const watermarkIndex = this.resolveWatermarkIndex(input.sessionId, input.fromCursorExclusive);
    const pendingRecords = this.pendingResponseUserRecords(input.sessionId, watermarkIndex);

    if (pendingRecords.length === 0) {
      return {
        fromCursorExclusive: input.fromCursorExclusive,
        entryIds: [],
        throughCursorInclusive: null,
        includedCount: 0,
        remainingCount: 0,
        hasMore: false,
        estimatedTokens: 0,
        estimatedChars: 0,
      };
    }

    const hydrator = new PendingUserEntryHydrator(
      this.options.createStreamReader(input.sessionId),
      pendingRecords,
    );
    const includedRecords: StreamEntryIndexRecord[] = [];
    let estimatedTokens = 0;
    let estimatedChars = 0;

    try {
      for (const record of pendingRecords) {
        if (includedRecords.length > 0 && includedRecords.length + 1 > caps.maxMessages) {
          break;
        }

        const entry = await hydrator.read(record);
        const entryTokens = estimateEntryPromptTokens(entry);
        const entryChars = estimateEntryPromptChars(entry);
        const nextCount = includedRecords.length + 1;
        const nextTokens = estimatedTokens + entryTokens;
        const nextChars = estimatedChars + entryChars;

        // Progress guarantee: the first pending message is included even if
        // it alone exceeds the configured caps, otherwise the drain can
        // deadlock forever on one oversized user message.
        if (
          includedRecords.length > 0 &&
          wouldExceedCaps({
            caps,
            nextCount,
            nextTokens,
            nextChars,
          })
        ) {
          break;
        }

        includedRecords.push(record);
        estimatedTokens = nextTokens;
        estimatedChars = nextChars;
      }
    } finally {
      await hydrator.close();
    }

    const throughRecord = includedRecords[includedRecords.length - 1] ?? null;
    const remainingCount = pendingRecords.length - includedRecords.length;

    return {
      fromCursorExclusive: input.fromCursorExclusive,
      entryIds: includedRecords.map((record) => record.entry_id as StreamEntryId),
      throughCursorInclusive: throughRecord === null ? null : cursorForRecord(throughRecord),
      includedCount: includedRecords.length,
      remainingCount,
      hasMore: remainingCount > 0,
      estimatedTokens,
      estimatedChars,
    };
  }

  private resolveWatermarkIndex(sessionId: SessionId, cursor: StreamCursor | null): number {
    if (cursor === null) {
      return -1;
    }

    const record = this.options.entryIndex.lookup(cursor.entryId);

    if (record === null) {
      throw new CognitionError("Chat response backlog watermark cursor is missing from the index", {
        code: "CHAT_RESPONSE_BACKLOG_WATERMARK_NOT_INDEXED",
      });
    }

    if (record.session_id !== sessionId) {
      throw new CognitionError(
        "Chat response backlog watermark cursor belongs to another session",
        {
          code: "CHAT_RESPONSE_BACKLOG_WATERMARK_SESSION_MISMATCH",
        },
      );
    }

    if (record.timestamp !== cursor.ts) {
      throw new CognitionError(
        "Chat response backlog watermark cursor timestamp mismatches the index",
        {
          code: "CHAT_RESPONSE_BACKLOG_WATERMARK_TS_MISMATCH",
        },
      );
    }

    const entryIndex = requiredQueuedUserRecordEntryIndex(record);

    if (record.kind !== "user_msg") {
      throw new CognitionError("Chat response backlog watermark cursor is not a user message", {
        code: "CHAT_RESPONSE_BACKLOG_WATERMARK_KIND_INVALID",
      });
    }

    return entryIndex;
  }

  private pendingResponseUserRecords(
    sessionId: SessionId,
    watermarkIndex: number,
  ): StreamEntryIndexRecord[] {
    return orderedPendingResponseUserRecords({
      records: this.options.entryIndex.lookupSessionEntriesByKind({
        sessionId,
        kind: "user_msg",
      }),
      afterEntryIndex: watermarkIndex,
    });
  }
}
