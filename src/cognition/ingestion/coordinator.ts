import type { EpisodicExtractor, ExtractFromStreamResult } from "../../memory/episodic/index.js";
import {
  StreamReader,
  type StreamCursor,
  type StreamEntry,
  type StreamResponseTo,
  type StreamWatermarkRepository,
} from "../../stream/index.js";
import { BorgError, CognitionError } from "../../util/errors.js";
import { type Clock, SystemClock } from "../../util/clock.js";
import type { SessionId } from "../../util/ids.js";
import type { ChatResponseWatermarkCoordinator } from "./chat-response-watermark.js";

const EPISODIC_PROCESS_NAME = "episodic-extractor";
type TimeoutHandle = ReturnType<typeof setTimeout>;
type SetTimeoutFn = (callback: () => void, delayMs: number) => TimeoutHandle;
type ClearTimeoutFn = (handle: TimeoutHandle) => void;

function isFileMissingError(error: unknown): boolean {
  if (error instanceof BorgError && error.cause !== undefined) {
    return isFileMissingError(error.cause);
  }

  if (
    error instanceof Error &&
    "code" in error &&
    typeof (error as { code: unknown }).code === "string"
  ) {
    return (error as { code: string }).code === "ENOENT";
  }

  return false;
}

export type StreamIngestionCoordinatorOptions = {
  extractor: EpisodicExtractor;
  watermarkRepository: StreamWatermarkRepository;
  chatResponseWatermarkCoordinator?: Pick<
    ChatResponseWatermarkCoordinator,
    "compareCursors" | "cursorEntryIndex" | "reconcile"
  >;
  dataDir: string;
  /**
   * Minimum number of new stream entries past the watermark required before
   * live extraction fires. Defaults to 2 (one user/agent pair). Below this,
   * the coordinator no-ops and waits for the next turn.
   */
  minEntriesThreshold?: number;
  /**
   * Optional debounce before a live ingestion pass starts. Defaults to 0,
   * which preserves immediate pass starts. Suggested busy-chat values are
   * settleMs ~= 3000 and maxSettleMs ~= 30000.
   */
  settleMs?: number;
  maxSettleMs?: number;
  clock?: Clock;
  setTimeoutFn?: SetTimeoutFn;
  clearTimeoutFn?: ClearTimeoutFn;
  /**
   * Called when extraction fails. Default is to swallow (live extraction
   * runs after the turn's response is returned -- failure should not
   * surface to the user). Pass a hook to log or rethrow.
   */
  onError?: (error: unknown, sessionId: SessionId) => void | Promise<void>;
};

export type IngestionResult = {
  ran: boolean;
  processedEntries: number;
  extractionResult?: ExtractFromStreamResult;
  error?: unknown;
};

export type AnsweredStreamWindow = {
  responseTo: StreamResponseTo;
  terminalCursor: StreamCursor;
};

export type IngestOptions = {
  /**
   * Override the default minEntriesThreshold for this call. Useful for
   * flush-on-close semantics where even a single new entry should be
   * ingested.
   */
  minEntriesThreshold?: number;
  /**
   * Process at most this many pending stream entries in this pass. The
   * watermark advances to the last entry in the processed prefix only after
   * extraction succeeds.
   */
  maxEntries?: number;
  /**
   * Catch-up-only guard: clamp the processed prefix to the chat-response
   * watermark so unanswered queued user_msg entries are not extracted before a
   * terminal response/marker exists.
   */
  clampToChatResponseWatermark?: boolean;
  /**
   * Exact answered window for a terminal response to a drained stream backlog
   * batch. This bypasses raw contiguous cursor ingestion so interleaved
   * unanswered user_msg entries stay out of the chunk.
   */
  answeredWindow?: AnsweredStreamWindow;
};

export type PreTurnCatchUpOptions = {
  maxEntries: number;
  clampToChatResponseWatermark?: boolean;
};

type ResumeOptions = {
  sinceCursor?: StreamCursor;
};

type ResolvedIngestOptions = {
  minEntriesThreshold: number;
  maxEntries?: number;
  clampToChatResponseWatermark?: boolean;
  answeredWindow?: AnsweredStreamWindow;
};

type InFlightIngestion = {
  promise: Promise<IngestionResult>;
  minEntriesThreshold: number;
  maxEntries?: number;
  clampToChatResponseWatermark?: boolean;
  answeredWindow?: AnsweredStreamWindow;
};

type PendingIngestionWaiter = {
  resolve: (result: IngestionResult) => void;
  reject: (error: unknown) => void;
};

type PendingIngestion = {
  minEntriesThreshold: number;
  maxEntries?: number;
  clampToChatResponseWatermark?: boolean;
  answeredWindow?: AnsweredStreamWindow;
  firstPendingAt: number;
  lastTriggerAt: number;
  waiters: PendingIngestionWaiter[];
};

type IngestionMode = "normal" | "clamped-catch-up" | "answered-window";

type SettlingIngestion = PendingIngestion & {
  timer: TimeoutHandle | null;
};

function mergeMaxEntries(left: number | undefined, right: number | undefined): number | undefined {
  if (left === undefined || right === undefined) {
    return undefined;
  }

  return Math.max(left, right);
}

function ingestionMode(options: ResolvedIngestOptions): IngestionMode {
  if (options.answeredWindow !== undefined) {
    return "answered-window";
  }

  if (options.clampToChatResponseWatermark === true) {
    return "clamped-catch-up";
  }

  return "normal";
}

function sameCursor(left: StreamCursor | null, right: StreamCursor | null): boolean {
  if (left === null || right === null) {
    return left === right;
  }

  return left.ts === right.ts && left.entryId === right.entryId;
}

function sameResponseTo(left: StreamResponseTo, right: StreamResponseTo): boolean {
  return (
    left.kind === right.kind &&
    sameCursor(left.from_cursor_exclusive, right.from_cursor_exclusive) &&
    sameCursor(left.through_cursor_inclusive, right.through_cursor_inclusive) &&
    left.count === right.count &&
    left.source_entry_ids.length === right.source_entry_ids.length &&
    left.source_entry_ids.every((entryId, index) => entryId === right.source_entry_ids[index])
  );
}

function sameAnsweredWindow(left: AnsweredStreamWindow, right: AnsweredStreamWindow): boolean {
  return (
    sameResponseTo(left.responseTo, right.responseTo) &&
    sameCursor(left.terminalCursor, right.terminalCursor)
  );
}

function mergeCompatibleIngestOptions(
  left: ResolvedIngestOptions,
  right: ResolvedIngestOptions,
): ResolvedIngestOptions | null {
  const mode = ingestionMode(left);

  if (mode !== ingestionMode(right)) {
    return null;
  }

  if (mode === "answered-window") {
    if (
      left.answeredWindow === undefined ||
      right.answeredWindow === undefined ||
      !sameAnsweredWindow(left.answeredWindow, right.answeredWindow)
    ) {
      return null;
    }

    return {
      minEntriesThreshold: Math.min(left.minEntriesThreshold, right.minEntriesThreshold),
      answeredWindow: left.answeredWindow,
    };
  }

  const maxEntries = mergeMaxEntries(left.maxEntries, right.maxEntries);

  return {
    minEntriesThreshold: Math.min(left.minEntriesThreshold, right.minEntriesThreshold),
    ...(maxEntries === undefined ? {} : { maxEntries }),
    ...(mode === "clamped-catch-up" ? { clampToChatResponseWatermark: true } : {}),
  };
}

/**
 * Fires episodic extraction after a turn completes, gated by a stream
 * watermark so each entry is processed at most once (the extractor keeps an
 * exact replay check on source stream ids, which makes late watermark
 * advancement safe without cross-turn merging).
 *
 * Callers should NOT await this in the critical path -- extraction calls
 * the LLM and adds latency. Instead: `void coordinator.ingest(sessionId)`
 * after the turn's response is sent. The turn orchestrator uses `catchUp`
 * for a bounded pre-turn retry before retrieval.
 */
export class StreamIngestionCoordinator {
  private readonly clock: Clock;
  private readonly setTimeoutFn: SetTimeoutFn;
  private readonly clearTimeoutFn: ClearTimeoutFn;
  private readonly minEntriesThreshold: number;
  private readonly settleMs: number;
  private readonly maxSettleMs: number;
  private readonly inFlight = new Map<SessionId, InFlightIngestion>();
  private readonly pending = new Map<SessionId, PendingIngestion[]>();
  private readonly settling = new Map<SessionId, SettlingIngestion>();
  private readonly trackedSessions = new Set<SessionId>();
  private readonly shutdownPendingDrain = new Set<SessionId>();
  private closePromise: Promise<void> | null = null;

  constructor(private readonly options: StreamIngestionCoordinatorOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.setTimeoutFn =
      options.setTimeoutFn ?? ((callback, delayMs) => setTimeout(callback, delayMs));
    this.clearTimeoutFn = options.clearTimeoutFn ?? ((handle) => clearTimeout(handle));
    this.minEntriesThreshold = options.minEntriesThreshold ?? 2;
    this.settleMs = Math.max(0, options.settleMs ?? 0);
    this.maxSettleMs = Math.max(0, options.maxSettleMs ?? 30_000);
  }

  /**
   * Trigger episodic extraction for a session if the backlog past the
   * watermark meets the threshold. Returns a promise that resolves to the
   * extraction result; the orchestrator usually doesn't await it.
   *
   * Concurrent calls for the same session are serialized: only one extraction
   * pass runs at a time per session, and callers arriving during an active
   * pass wait on a queued follow-up pass.
   */
  ingest(sessionId: SessionId, ingestOptions: IngestOptions = {}): Promise<IngestionResult> {
    const resolvedOptions = {
      minEntriesThreshold: ingestOptions.minEntriesThreshold ?? this.minEntriesThreshold,
      ...(ingestOptions.maxEntries === undefined ? {} : { maxEntries: ingestOptions.maxEntries }),
      ...(ingestOptions.clampToChatResponseWatermark === true
        ? { clampToChatResponseWatermark: true }
        : {}),
      ...(ingestOptions.answeredWindow === undefined
        ? {}
        : { answeredWindow: ingestOptions.answeredWindow }),
    };
    const existing = this.inFlight.get(sessionId);
    const settling = this.settling.get(sessionId);

    if (this.closePromise !== null && existing === undefined && settling === undefined) {
      return this.closePromise.then(() => ({
        ran: false,
        processedEntries: 0,
      }));
    }

    this.trackedSessions.add(sessionId);

    if (existing !== undefined) {
      return this.enqueueFollowUp(sessionId, resolvedOptions);
    }

    if (settling !== undefined) {
      return this.enqueueSettling(sessionId, resolvedOptions);
    }

    return this.startPassAfterSettle(sessionId, resolvedOptions);
  }

  async hasBacklog(sessionId: SessionId): Promise<boolean> {
    const resumeOptions = this.resolveResumeOptions(sessionId);
    const entries = await this.readEntriesPastWatermark(sessionId, resumeOptions, 1);

    return entries.length > 0;
  }

  async catchUp(sessionId: SessionId, options: PreTurnCatchUpOptions): Promise<IngestionResult> {
    if (this.closePromise !== null) {
      return this.closePromise.then(() => ({
        ran: false,
        processedEntries: 0,
      }));
    }

    if (this.inFlight.has(sessionId)) {
      return {
        ran: false,
        processedEntries: 0,
      };
    }

    const catchUpOptions = {
      minEntriesThreshold: 1,
      maxEntries: options.maxEntries,
      ...(options.clampToChatResponseWatermark === true
        ? { clampToChatResponseWatermark: true }
        : {}),
    };
    const settling = this.settling.get(sessionId);
    const settlingPass =
      settling === undefined
        ? undefined
        : this.flushSettlingPass(
            sessionId,
            settling.answeredWindow === undefined ? catchUpOptions : undefined,
          );

    if (settlingPass !== undefined) {
      return settlingPass;
    }

    if (!(await this.hasBacklog(sessionId))) {
      return {
        ran: false,
        processedEntries: 0,
      };
    }

    this.trackedSessions.add(sessionId);
    return this.startPass(sessionId, catchUpOptions);
  }

  private enqueueFollowUp(
    sessionId: SessionId,
    ingestOptions: ResolvedIngestOptions,
  ): Promise<IngestionResult> {
    return new Promise<IngestionResult>((resolve, reject) => {
      const pendingQueue = this.pending.get(sessionId) ?? [];
      const waiter = { resolve, reject };
      const now = this.clock.now();
      const lastPending = pendingQueue.at(-1);

      if (lastPending !== undefined) {
        const merged = mergeCompatibleIngestOptions(lastPending, ingestOptions);

        if (merged !== null) {
          pendingQueue[pendingQueue.length - 1] = {
            ...merged,
            firstPendingAt: lastPending.firstPendingAt,
            lastTriggerAt: now,
            waiters: [...lastPending.waiters, waiter],
          };
          this.pending.set(sessionId, pendingQueue);
          return;
        }
      }

      pendingQueue.push({
        ...ingestOptions,
        firstPendingAt: now,
        lastTriggerAt: now,
        waiters: [waiter],
      });
      this.pending.set(sessionId, pendingQueue);
    });
  }

  private enqueueSettling(
    sessionId: SessionId,
    ingestOptions: ResolvedIngestOptions,
  ): Promise<IngestionResult> {
    return new Promise<IngestionResult>((resolve, reject) => {
      const settling = this.settling.get(sessionId);

      if (settling === undefined) {
        this.startPassAfterSettle(sessionId, ingestOptions).then(resolve, reject);
        return;
      }

      const waiter = { resolve, reject };
      const now = this.clock.now();
      const merged = mergeCompatibleIngestOptions(settling, ingestOptions);

      if (merged === null) {
        this.enqueueFollowUpWithWaiter(sessionId, ingestOptions, waiter, now);
        return;
      }

      this.settling.set(sessionId, {
        ...settling,
        ...merged,
        firstPendingAt: settling.firstPendingAt,
        lastTriggerAt: now,
        waiters: [...settling.waiters, waiter],
      });
      this.scheduleSettleTimer(sessionId);
    });
  }

  private enqueueFollowUpWithWaiter(
    sessionId: SessionId,
    ingestOptions: ResolvedIngestOptions,
    waiter: PendingIngestionWaiter,
    pendingAt: number,
  ): void {
    const pendingQueue = this.pending.get(sessionId) ?? [];
    const lastPending = pendingQueue.at(-1);

    if (lastPending !== undefined) {
      const merged = mergeCompatibleIngestOptions(lastPending, ingestOptions);

      if (merged !== null) {
        pendingQueue[pendingQueue.length - 1] = {
          ...merged,
          firstPendingAt: lastPending.firstPendingAt,
          lastTriggerAt: pendingAt,
          waiters: [...lastPending.waiters, waiter],
        };
        this.pending.set(sessionId, pendingQueue);
        return;
      }
    }

    pendingQueue.push({
      ...ingestOptions,
      firstPendingAt: pendingAt,
      lastTriggerAt: pendingAt,
      waiters: [waiter],
    });
    this.pending.set(sessionId, pendingQueue);
  }

  private dequeueFollowUp(sessionId: SessionId): PendingIngestion | undefined {
    const pendingQueue = this.pending.get(sessionId);

    if (pendingQueue === undefined) {
      return undefined;
    }

    const pending = pendingQueue.shift();

    if (pendingQueue.length === 0) {
      this.pending.delete(sessionId);
    }

    return pending;
  }

  private startPendingPass(sessionId: SessionId, pending: PendingIngestion): void {
    const followUp = this.startPassAfterSettle(
      sessionId,
      this.resolvedOptionsFromPending(pending),
      {
        firstPendingAt: pending.firstPendingAt,
        lastTriggerAt: pending.lastTriggerAt,
      },
    );

    void followUp.then(
      (followUpResult) => {
        for (const waiter of pending.waiters) {
          waiter.resolve(followUpResult);
        }
      },
      (error) => {
        for (const waiter of pending.waiters) {
          waiter.reject(error);
        }
      },
    );
  }

  private resolvedOptionsFromPending(pending: PendingIngestion): ResolvedIngestOptions {
    return {
      minEntriesThreshold: pending.minEntriesThreshold,
      ...(pending.maxEntries === undefined ? {} : { maxEntries: pending.maxEntries }),
      ...(pending.clampToChatResponseWatermark === true
        ? { clampToChatResponseWatermark: true }
        : {}),
      ...(pending.answeredWindow === undefined ? {} : { answeredWindow: pending.answeredWindow }),
    };
  }

  private startPassAfterSettle(
    sessionId: SessionId,
    ingestOptions: ResolvedIngestOptions,
    timing?: { firstPendingAt: number; lastTriggerAt: number },
  ): Promise<IngestionResult> {
    if (this.settleMs === 0 || this.closePromise !== null) {
      return this.startPass(sessionId, ingestOptions);
    }

    const existing = this.settling.get(sessionId);

    if (existing !== undefined) {
      return this.enqueueSettling(sessionId, ingestOptions);
    }

    return new Promise<IngestionResult>((resolve, reject) => {
      const now = this.clock.now();
      this.settling.set(sessionId, {
        ...ingestOptions,
        firstPendingAt: timing?.firstPendingAt ?? now,
        lastTriggerAt: timing?.lastTriggerAt ?? now,
        timer: null,
        waiters: [{ resolve, reject }],
      });
      this.scheduleSettleTimer(sessionId);
    });
  }

  private scheduleSettleTimer(sessionId: SessionId): void {
    const settling = this.settling.get(sessionId);

    if (settling === undefined) {
      return;
    }

    if (settling.timer !== null) {
      this.clearTimeoutFn(settling.timer);
      settling.timer = null;
    }

    const now = this.clock.now();
    const quietDueAt = settling.lastTriggerAt + this.settleMs;
    const maxSettleDueAt = settling.firstPendingAt + this.maxSettleMs;
    const delayMs = Math.max(0, Math.min(quietDueAt, maxSettleDueAt) - now);

    settling.timer = this.setTimeoutFn(() => {
      const current = this.settling.get(sessionId);

      if (current !== undefined) {
        current.timer = null;
      }

      this.flushSettlingPass(sessionId);
    }, delayMs);
  }

  private flushSettlingPass(
    sessionId: SessionId,
    overrideOptions?: ResolvedIngestOptions,
  ): Promise<IngestionResult> | undefined {
    const settling = this.settling.get(sessionId);

    if (settling === undefined) {
      return undefined;
    }

    if (settling.timer !== null) {
      this.clearTimeoutFn(settling.timer);
    }

    this.settling.delete(sessionId);
    const pass = this.startPass(
      sessionId,
      overrideOptions ?? this.resolvedOptionsFromPending(settling),
    );

    void pass.then(
      (result) => {
        for (const waiter of settling.waiters) {
          waiter.resolve(result);
        }
      },
      (error) => {
        for (const waiter of settling.waiters) {
          waiter.reject(error);
        }
      },
    );

    return pass;
  }

  private startPass(
    sessionId: SessionId,
    ingestOptions: ResolvedIngestOptions,
  ): Promise<IngestionResult> {
    let settledResult: IngestionResult | undefined;
    let settledError: unknown;
    const promise = this.runPass(sessionId, ingestOptions)
      .then((result) => {
        settledResult = result;
        return result;
      })
      .catch((error) => {
        settledError = error;
        throw error;
      })
      .finally(() => {
        const needsShutdownDrain =
          this.closePromise !== null &&
          settledError === undefined &&
          settledResult !== undefined &&
          settledResult.error === undefined &&
          !settledResult.ran &&
          settledResult.processedEntries > 0;

        if (needsShutdownDrain) {
          this.shutdownPendingDrain.add(sessionId);
        } else {
          this.shutdownPendingDrain.delete(sessionId);
        }

        if (this.inFlight.get(sessionId)?.promise === promise) {
          this.inFlight.delete(sessionId);
        }

        const canStopTracking =
          (settledError === undefined && settledResult?.error === undefined) ||
          (this.closePromise !== null && settledResult?.error !== undefined);

        if (
          canStopTracking &&
          !this.inFlight.has(sessionId) &&
          !this.pending.has(sessionId) &&
          !this.settling.has(sessionId) &&
          !this.shutdownPendingDrain.has(sessionId)
        ) {
          this.trackedSessions.delete(sessionId);
        }
      });

    this.inFlight.set(sessionId, {
      promise,
      ...ingestOptions,
    });
    return promise;
  }

  private async runPass(
    sessionId: SessionId,
    ingestOptions: ResolvedIngestOptions,
  ): Promise<IngestionResult> {
    let result: IngestionResult | undefined;
    let failure: unknown;

    try {
      result = await this.ingestInternal(sessionId, {
        minEntriesThreshold: ingestOptions.minEntriesThreshold,
        ...(ingestOptions.maxEntries === undefined ? {} : { maxEntries: ingestOptions.maxEntries }),
        ...(ingestOptions.clampToChatResponseWatermark === true
          ? { clampToChatResponseWatermark: true }
          : {}),
        ...(ingestOptions.answeredWindow === undefined
          ? {}
          : { answeredWindow: ingestOptions.answeredWindow }),
      });
    } catch (error) {
      failure = error;
    }

    const pending = this.dequeueFollowUp(sessionId);

    if (pending !== undefined) {
      this.startPendingPass(sessionId, pending);
    }

    if (failure !== undefined) {
      throw failure;
    }

    return result as IngestionResult;
  }

  private resolveResumeOptions(sessionId: SessionId): ResumeOptions {
    const watermark = this.options.watermarkRepository.get(EPISODIC_PROCESS_NAME, sessionId);

    if (watermark === null) {
      return {};
    }

    return {
      sinceCursor: {
        ts: watermark.lastTs,
        entryId: watermark.lastEntryId as StreamCursor["entryId"],
      },
    };
  }

  private async readEntriesPastWatermark(
    sessionId: SessionId,
    resumeOptions: ResumeOptions,
    limit?: number,
  ): Promise<StreamEntry[]> {
    const reader = new StreamReader({
      dataDir: this.options.dataDir,
      sessionId,
    });
    const entries: StreamEntry[] = [];

    try {
      for await (const entry of reader.iterate({
        sinceCursor: resumeOptions.sinceCursor,
        ...(limit === undefined ? {} : { limit }),
      })) {
        entries.push(entry);
      }
    } catch (error) {
      // Tests may tear down the data dir between the turn's stream write and
      // this fire-and-forget ingestion running. A vanished stream file is
      // effectively "nothing to ingest"; production real runs won't see this.
      if (isFileMissingError(error)) {
        return [];
      }

      throw error;
    }

    return entries;
  }

  private requireChatResponseWatermarkCoordinator(): NonNullable<
    StreamIngestionCoordinatorOptions["chatResponseWatermarkCoordinator"]
  > {
    const coordinator = this.options.chatResponseWatermarkCoordinator;

    if (coordinator === undefined) {
      throw new CognitionError(
        "Clamped catch-up ingestion requires chat response watermark coordination",
        {
          code: "CHAT_RESPONSE_WATERMARK_COORDINATOR_REQUIRED",
        },
      );
    }

    return coordinator;
  }

  private cursorForEntry(entry: StreamEntry): StreamCursor {
    return {
      ts: entry.timestamp,
      entryId: entry.id,
    };
  }

  private streamEntryIndex(sessionId: SessionId, entry: StreamEntry): number {
    if (entry.entry_index !== undefined) {
      return entry.entry_index;
    }

    return this.requireChatResponseWatermarkCoordinator().cursorEntryIndex(
      sessionId,
      this.cursorForEntry(entry),
      "stream entry",
    );
  }

  private entriesThroughCursor(
    sessionId: SessionId,
    entries: readonly StreamEntry[],
    cursor: StreamCursor,
  ): StreamEntry[] {
    const cursorEntryIndex = this.requireChatResponseWatermarkCoordinator().cursorEntryIndex(
      sessionId,
      cursor,
      "effective until",
    );

    return entries.filter((entry) => this.streamEntryIndex(sessionId, entry) <= cursorEntryIndex);
  }

  private minCursorByEntryIndex(
    sessionId: SessionId,
    left: StreamCursor,
    right: StreamCursor,
  ): StreamCursor {
    return this.requireChatResponseWatermarkCoordinator().compareCursors(sessionId, left, right) <=
      0
      ? left
      : right;
  }

  private clampEntriesToChatResponseWatermark(
    sessionId: SessionId,
    entries: readonly StreamEntry[],
  ): StreamEntry[] | null {
    const lastEntry = entries.at(-1);

    if (lastEntry === undefined) {
      return [];
    }

    const respondedThrough =
      this.requireChatResponseWatermarkCoordinator().reconcile(sessionId).watermark;

    if (respondedThrough === null) {
      return null;
    }

    const candidateUntil = this.cursorForEntry(lastEntry);
    const effectiveUntil = this.minCursorByEntryIndex(sessionId, respondedThrough, candidateUntil);

    return this.entriesThroughCursor(sessionId, entries, effectiveUntil);
  }

  private async readEntriesById(
    sessionId: SessionId,
    entryIds: readonly StreamEntry["id"][],
  ): Promise<StreamEntry[]> {
    const wanted = new Set(entryIds);
    const entries: StreamEntry[] = [];

    if (wanted.size === 0) {
      return entries;
    }

    const reader = new StreamReader({
      dataDir: this.options.dataDir,
      sessionId,
    });

    try {
      for await (const entry of reader.iterate()) {
        if (!wanted.has(entry.id)) {
          continue;
        }

        entries.push(entry);

        if (entries.length >= wanted.size) {
          break;
        }
      }
    } catch (error) {
      if (isFileMissingError(error)) {
        return [];
      }

      throw error;
    }

    return entries;
  }

  private async ingestAnsweredWindow(
    sessionId: SessionId,
    answeredWindow: AnsweredStreamWindow,
  ): Promise<IngestionResult> {
    const entryIds = [
      ...answeredWindow.responseTo.source_entry_ids,
      answeredWindow.terminalCursor.entryId,
    ];
    const entries = await this.readEntriesById(sessionId, entryIds);
    const entriesById = new Map(entries.map((entry) => [entry.id, entry]));
    const missingEntryId = entryIds.find((entryId) => !entriesById.has(entryId));

    if (missingEntryId !== undefined) {
      throw new CognitionError("Answered ingestion window references a missing stream entry", {
        code: "ANSWERED_INGESTION_WINDOW_ENTRY_MISSING",
      });
    }

    const terminalEntry = entriesById.get(answeredWindow.terminalCursor.entryId);

    if (
      terminalEntry === undefined ||
      terminalEntry.timestamp !== answeredWindow.terminalCursor.ts ||
      terminalEntry.session_id !== sessionId
    ) {
      throw new CognitionError("Answered ingestion window terminal cursor mismatches the stream", {
        code: "ANSWERED_INGESTION_WINDOW_TERMINAL_MISMATCH",
      });
    }

    try {
      const extractionResult = await this.options.extractor.extractFromStream({
        session: sessionId,
        entryIds,
      });

      this.options.watermarkRepository.set(EPISODIC_PROCESS_NAME, sessionId, {
        lastTs: answeredWindow.terminalCursor.ts,
        lastEntryId: answeredWindow.terminalCursor.entryId,
      });

      return {
        ran: true,
        processedEntries: entries.length,
        extractionResult,
      };
    } catch (error) {
      try {
        await this.options.onError?.(error, sessionId);
      } catch {
        // Best-effort.
      }

      return {
        ran: false,
        processedEntries: entries.length,
        error,
      };
    }
  }

  private async ingestInternal(
    sessionId: SessionId,
    ingestOptions: IngestOptions,
  ): Promise<IngestionResult> {
    if (ingestOptions.answeredWindow !== undefined) {
      return this.ingestAnsweredWindow(sessionId, ingestOptions.answeredWindow);
    }

    const threshold = ingestOptions.minEntriesThreshold ?? this.minEntriesThreshold;
    const resumeOptions = this.resolveResumeOptions(sessionId);
    let newEntries = await this.readEntriesPastWatermark(
      sessionId,
      resumeOptions,
      ingestOptions.maxEntries,
    );

    if (ingestOptions.clampToChatResponseWatermark === true) {
      const clampedEntries = this.clampEntriesToChatResponseWatermark(sessionId, newEntries);

      if (clampedEntries === null) {
        return { ran: false, processedEntries: 0 };
      }

      newEntries = clampedEntries;
    }

    if (newEntries.length < threshold) {
      return { ran: false, processedEntries: newEntries.length };
    }

    try {
      const lastProcessedEntry = newEntries.at(-1);

      if (lastProcessedEntry === undefined) {
        return { ran: false, processedEntries: 0 };
      }

      const extractionResult = await this.options.extractor.extractFromStream({
        session: sessionId,
        sinceCursor: resumeOptions.sinceCursor,
        untilCursor: {
          ts: lastProcessedEntry.timestamp,
          entryId: lastProcessedEntry.id,
        },
      });

      this.options.watermarkRepository.set(EPISODIC_PROCESS_NAME, sessionId, {
        lastTs: lastProcessedEntry?.timestamp ?? 0,
        lastEntryId: lastProcessedEntry.id,
      });

      return {
        ran: true,
        processedEntries: newEntries.length,
        extractionResult,
      };
    } catch (error) {
      try {
        await this.options.onError?.(error, sessionId);
      } catch {
        // Best-effort.
      }

      return {
        ran: false,
        processedEntries: newEntries.length,
        error,
      };
    }
  }

  /**
   * Force a flush of any pending entries past the watermark regardless of
   * the threshold. Useful on session close or debug runs -- guarantees that
   * a committed turn's conversation lands in episodic memory before the
   * process exits.
   */
  flush(sessionId: SessionId): Promise<IngestionResult> {
    return this.ingest(sessionId, { minEntriesThreshold: 1 });
  }

  async close(): Promise<void> {
    if (this.closePromise !== null) {
      return this.closePromise;
    }

    this.closePromise = (async () => {
      while (true) {
        const sessionIds = new Set<SessionId>([
          ...this.trackedSessions,
          ...this.shutdownPendingDrain,
          ...this.inFlight.keys(),
          ...this.pending.keys(),
          ...this.settling.keys(),
        ]);

        if (sessionIds.size === 0) {
          return;
        }

        await Promise.all(
          [...sessionIds].map((sessionId) => {
            const active = this.inFlight.get(sessionId);
            const settlingPass =
              active === undefined ? this.flushSettlingPass(sessionId) : undefined;
            return (
              active?.promise ??
              settlingPass ??
              this.startPass(sessionId, {
                minEntriesThreshold: 1,
              })
            );
          }),
        );

        const hasOutstanding = [...sessionIds].some(
          (sessionId) =>
            this.shutdownPendingDrain.has(sessionId) ||
            this.inFlight.has(sessionId) ||
            this.pending.has(sessionId) ||
            this.settling.has(sessionId),
        );

        if (!hasOutstanding) {
          return;
        }
      }
    })();

    return this.closePromise;
  }

  now(): number {
    return this.clock.now();
  }
}
