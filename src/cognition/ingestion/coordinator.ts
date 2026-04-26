import type { EpisodicExtractor, ExtractFromStreamResult } from "../../memory/episodic/index.js";
import {
  StreamReader,
  type StreamCursor,
  type StreamEntry,
  type StreamWatermarkRepository,
} from "../../stream/index.js";
import { BorgError } from "../../util/errors.js";
import { type Clock, SystemClock } from "../../util/clock.js";
import type { SessionId } from "../../util/ids.js";

const EPISODIC_PROCESS_NAME = "episodic-extractor";

export type LegacyFallbackNotice = {
  processName: typeof EPISODIC_PROCESS_NAME;
  sessionId: SessionId;
  sinceTs: number;
  message: string;
};

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
  dataDir: string;
  /**
   * Minimum number of new stream entries past the watermark required before
   * live extraction fires. Defaults to 2 (one user/agent pair). Below this,
   * the coordinator no-ops and waits for the next turn.
   */
  minEntriesThreshold?: number;
  clock?: Clock;
  /**
   * Called when extraction fails. Default is to swallow (live extraction
   * runs after the turn's response is returned -- failure should not
   * surface to the user). Pass a hook to log or rethrow.
   */
  onError?: (error: unknown, sessionId: SessionId) => void | Promise<void>;
  /**
   * Called when resuming from a legacy watermark that has a timestamp but no
   * entry id. Defaults to console.warn so the fallback is observable without
   * polluting the stream.
   */
  onLegacyFallback?: (notice: LegacyFallbackNotice) => void | Promise<void>;
};

export type IngestionResult = {
  ran: boolean;
  processedEntries: number;
  extractionResult?: ExtractFromStreamResult;
  error?: unknown;
};

export type IngestOptions = {
  /**
   * Override the default minEntriesThreshold for this call. Useful for
   * flush-on-close semantics where even a single new entry should be
   * ingested.
   */
  minEntriesThreshold?: number;
};

type ResumeOptions = {
  sinceTs?: number;
  sinceCursor?: StreamCursor;
  usedLegacyFallback: boolean;
};

type InFlightIngestion = {
  promise: Promise<IngestionResult>;
  minEntriesThreshold: number;
};

type PendingIngestionWaiter = {
  resolve: (result: IngestionResult) => void;
  reject: (error: unknown) => void;
};

type PendingIngestion = {
  minEntriesThreshold: number;
  waiters: PendingIngestionWaiter[];
};

/**
 * Fires episodic extraction after a turn completes, gated by a stream
 * watermark so each entry is processed at most once (the extractor keeps an
 * exact replay check on source stream ids, which makes late watermark
 * advancement safe without cross-turn merging).
 *
 * Callers should NOT await this in the critical path -- extraction calls
 * the LLM and adds latency. Instead: `void coordinator.ingest(sessionId)`
 * after the turn's response is sent.
 */
export class StreamIngestionCoordinator {
  private readonly clock: Clock;
  private readonly minEntriesThreshold: number;
  private readonly inFlight = new Map<SessionId, InFlightIngestion>();
  private readonly pending = new Map<SessionId, PendingIngestion>();
  private readonly trackedSessions = new Set<SessionId>();
  private readonly shutdownPendingDrain = new Set<SessionId>();
  private closePromise: Promise<void> | null = null;

  constructor(private readonly options: StreamIngestionCoordinatorOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.minEntriesThreshold = options.minEntriesThreshold ?? 2;
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
    const minEntriesThreshold = ingestOptions.minEntriesThreshold ?? this.minEntriesThreshold;
    const existing = this.inFlight.get(sessionId);

    if (this.closePromise !== null && existing === undefined) {
      return this.closePromise.then(() => ({
        ran: false,
        processedEntries: 0,
      }));
    }

    this.trackedSessions.add(sessionId);

    if (existing !== undefined) {
      return this.enqueueFollowUp(
        sessionId,
        Math.min(existing.minEntriesThreshold, minEntriesThreshold),
      );
    }

    return this.startPass(sessionId, minEntriesThreshold);
  }

  private enqueueFollowUp(
    sessionId: SessionId,
    minEntriesThreshold: number,
  ): Promise<IngestionResult> {
    return new Promise<IngestionResult>((resolve, reject) => {
      const pending = this.pending.get(sessionId);

      if (pending === undefined) {
        this.pending.set(sessionId, {
          minEntriesThreshold,
          waiters: [{ resolve, reject }],
        });
        return;
      }

      pending.minEntriesThreshold = Math.min(pending.minEntriesThreshold, minEntriesThreshold);
      pending.waiters.push({ resolve, reject });
    });
  }

  private startPass(sessionId: SessionId, minEntriesThreshold: number): Promise<IngestionResult> {
    let settledResult: IngestionResult | undefined;
    let settledError: unknown;
    const promise = this.runPass(sessionId, minEntriesThreshold)
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
          !this.shutdownPendingDrain.has(sessionId)
        ) {
          this.trackedSessions.delete(sessionId);
        }
      });

    this.inFlight.set(sessionId, {
      promise,
      minEntriesThreshold,
    });
    return promise;
  }

  private async runPass(
    sessionId: SessionId,
    minEntriesThreshold: number,
  ): Promise<IngestionResult> {
    let result: IngestionResult | undefined;
    let failure: unknown;

    try {
      result = await this.ingestInternal(sessionId, {
        minEntriesThreshold,
      });
    } catch (error) {
      failure = error;
    }

    const pending = this.pending.get(sessionId);

    if (pending !== undefined) {
      this.pending.delete(sessionId);
      const followUp = this.startPass(sessionId, pending.minEntriesThreshold);

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

    if (failure !== undefined) {
      throw failure;
    }

    return result as IngestionResult;
  }

  private resolveResumeOptions(sessionId: SessionId): ResumeOptions {
    const watermark = this.options.watermarkRepository.get(EPISODIC_PROCESS_NAME, sessionId);

    if (watermark === null) {
      return {
        usedLegacyFallback: false,
      };
    }

    const lastEntryId = watermark.lastEntryId;

    if (lastEntryId === null) {
      return {
        sinceTs: watermark.lastTs + 1,
        usedLegacyFallback: true,
      };
    }

    return {
      sinceCursor: {
        ts: watermark.lastTs,
        entryId: lastEntryId as StreamCursor["entryId"],
      },
      usedLegacyFallback: false,
    };
  }

  private async reportLegacyFallback(sessionId: SessionId, sinceTs: number): Promise<void> {
    const notice: LegacyFallbackNotice = {
      processName: EPISODIC_PROCESS_NAME,
      sessionId,
      sinceTs,
      message: `legacy watermark fallback used for ${EPISODIC_PROCESS_NAME}; lastEntryId missing, resumed with sinceTs=${sinceTs}`,
    };
    const reporter =
      this.options.onLegacyFallback ??
      ((fallbackNotice: LegacyFallbackNotice) => {
        console.warn(fallbackNotice.message);
      });

    await reporter(notice);
  }

  private async ingestInternal(
    sessionId: SessionId,
    ingestOptions: IngestOptions,
  ): Promise<IngestionResult> {
    const threshold = ingestOptions.minEntriesThreshold ?? this.minEntriesThreshold;
    const resumeOptions = this.resolveResumeOptions(sessionId);

    const reader = new StreamReader({
      dataDir: this.options.dataDir,
      sessionId,
    });

    const newEntries: StreamEntry[] = [];

    try {
      for await (const entry of reader.iterate({
        sinceTs: resumeOptions.sinceTs,
        sinceCursor: resumeOptions.sinceCursor,
      })) {
        newEntries.push(entry);
      }
    } catch (error) {
      // Tests may tear down the data dir between the turn's stream write and
      // this fire-and-forget ingestion running. A vanished stream file is
      // effectively "nothing to ingest"; production real runs won't see this.
      if (isFileMissingError(error)) {
        return { ran: false, processedEntries: 0 };
      }

      throw error;
    }

    if (newEntries.length < threshold) {
      return { ran: false, processedEntries: newEntries.length };
    }

    try {
      if (resumeOptions.usedLegacyFallback && resumeOptions.sinceTs !== undefined) {
        try {
          await this.reportLegacyFallback(sessionId, resumeOptions.sinceTs);
        } catch {
          // Best-effort observability only.
        }
      }

      const extractionResult = await this.options.extractor.extractFromStream({
        session: sessionId,
        sinceTs: resumeOptions.sinceTs,
        sinceCursor: resumeOptions.sinceCursor,
      });
      const lastProcessedEntry = newEntries[newEntries.length - 1];

      this.options.watermarkRepository.set(EPISODIC_PROCESS_NAME, sessionId, {
        lastTs: lastProcessedEntry?.timestamp ?? 0,
        lastEntryId: lastProcessedEntry?.id ?? null,
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
        ]);

        if (sessionIds.size === 0) {
          return;
        }

        await Promise.all(
          [...sessionIds].map((sessionId) => {
            const active = this.inFlight.get(sessionId);
            return active?.promise ?? this.startPass(sessionId, 1);
          }),
        );

        const hasOutstanding = [...sessionIds].some(
          (sessionId) =>
            this.shutdownPendingDrain.has(sessionId) ||
            this.inFlight.has(sessionId) ||
            this.pending.has(sessionId),
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
