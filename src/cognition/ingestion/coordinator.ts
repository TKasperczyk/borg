import type { EpisodicExtractor, ExtractFromStreamResult } from "../../memory/episodic/index.js";
import { StreamReader, type StreamEntry, type StreamWatermarkRepository } from "../../stream/index.js";
import { BorgError } from "../../util/errors.js";
import { type Clock, SystemClock } from "../../util/clock.js";
import type { SessionId } from "../../util/ids.js";

const EPISODIC_PROCESS_NAME = "episodic-extractor";

function isFileMissingError(error: unknown): boolean {
  if (error instanceof BorgError && error.cause !== undefined) {
    return isFileMissingError(error.cause);
  }

  if (error instanceof Error && "code" in error && typeof (error as { code: unknown }).code === "string") {
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
  onError?: (error: unknown) => void | Promise<void>;
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

/**
 * Fires episodic extraction after a turn completes, gated by a stream
 * watermark so each entry is processed at most once (dedup in the extractor
 * itself makes it safe if the watermark is behind reality).
 *
 * Callers should NOT await this in the critical path -- extraction calls
 * the LLM and adds latency. Instead: `void coordinator.ingest(sessionId)`
 * after the turn's response is sent.
 */
export class StreamIngestionCoordinator {
  private readonly clock: Clock;
  private readonly minEntriesThreshold: number;
  private readonly inFlight = new Map<SessionId, Promise<IngestionResult>>();

  constructor(private readonly options: StreamIngestionCoordinatorOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.minEntriesThreshold = options.minEntriesThreshold ?? 2;
  }

  /**
   * Trigger episodic extraction for a session if the backlog past the
   * watermark meets the threshold. Returns a promise that resolves to the
   * extraction result; the orchestrator usually doesn't await it.
   *
   * Concurrent calls for the same session are coalesced: only one extraction
   * pass runs at a time per session.
   */
  ingest(sessionId: SessionId, ingestOptions: IngestOptions = {}): Promise<IngestionResult> {
    const existing = this.inFlight.get(sessionId);

    if (existing !== undefined) {
      return existing;
    }

    const promise = this.ingestInternal(sessionId, ingestOptions).finally(() => {
      this.inFlight.delete(sessionId);
    });

    this.inFlight.set(sessionId, promise);
    return promise;
  }

  private async ingestInternal(
    sessionId: SessionId,
    ingestOptions: IngestOptions,
  ): Promise<IngestionResult> {
    const threshold = ingestOptions.minEntriesThreshold ?? this.minEntriesThreshold;
    const watermark = this.options.watermarkRepository.get(EPISODIC_PROCESS_NAME, sessionId);
    // sinceTs is inclusive in the reader; bump by 1 ms past the watermark to
    // exclude the entry already marked as done. Brand-new sessions read from
    // the beginning of the stream.
    const sinceTs = watermark === null ? undefined : watermark.lastTs + 1;

    const reader = new StreamReader({
      dataDir: this.options.dataDir,
      sessionId,
    });

    const newEntries: StreamEntry[] = [];

    try {
      for await (const entry of reader.iterate({ sinceTs })) {
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
      const extractionResult = await this.options.extractor.extractFromStream({
        session: sessionId,
        sinceTs,
      });
      const newestTs = newEntries.reduce(
        (acc, entry) => Math.max(acc, entry.timestamp),
        watermark?.lastTs ?? 0,
      );
      const newestEntry = newEntries[newEntries.length - 1];

      this.options.watermarkRepository.set(EPISODIC_PROCESS_NAME, sessionId, {
        lastTs: newestTs,
        lastEntryId: newestEntry?.id ?? null,
      });

      return {
        ran: true,
        processedEntries: newEntries.length,
        extractionResult,
      };
    } catch (error) {
      try {
        await this.options.onError?.(error);
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

  now(): number {
    return this.clock.now();
  }
}
