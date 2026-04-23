import { closeSync, fsyncSync, fstatSync, mkdirSync, openSync, writeFileSync } from "node:fs";

import { SystemClock, type Clock } from "../util/clock.js";
import { StreamError } from "../util/errors.js";
import { createStreamEntryId } from "../util/ids.js";
import { serializeJsonValue } from "../util/json-value.js";

import { withFileLock } from "./file-lock.js";
import type { StreamEntryIndexRepository } from "./entry-index.js";
import { getSessionStreamPath, getStreamDirectory } from "./path.js";
import {
  DEFAULT_SESSION_ID,
  type SessionId,
  type StreamEntry,
  type StreamEntryInput,
  streamEntryInputSchema,
  streamEntrySchema,
} from "./types.js";

type LoggerLike = Pick<Console, "error">;

export type StreamWriterOptions = {
  dataDir: string;
  sessionId?: SessionId;
  clock?: Clock;
  logger?: LoggerLike;
  lockTimeoutMs?: number;
  lockRetryDelayMs?: number;
  entryIndex?: StreamEntryIndexRepository;
};

export class StreamWriter {
  private readonly dataDir: string;
  private readonly sessionId: SessionId;
  private readonly clock: Clock;
  private readonly logger: LoggerLike;
  private readonly lockTimeoutMs: number;
  private readonly lockRetryDelayMs: number;
  private readonly entryIndex?: StreamEntryIndexRepository;
  private closed = false;

  constructor(options: StreamWriterOptions) {
    this.dataDir = options.dataDir;
    this.sessionId = options.sessionId ?? DEFAULT_SESSION_ID;
    this.clock = options.clock ?? new SystemClock();
    this.logger = options.logger ?? console;
    this.lockTimeoutMs = options.lockTimeoutMs ?? 2_000;
    this.lockRetryDelayMs = options.lockRetryDelayMs ?? 20;
    this.entryIndex = options.entryIndex;
  }

  private ensureOpen(): void {
    if (this.closed) {
      throw new StreamError("StreamWriter is closed");
    }
  }

  private buildEntry(input: StreamEntryInput, timestamp: number): StreamEntry {
    const parsedInput = streamEntryInputSchema.safeParse(input);

    if (!parsedInput.success) {
      throw new StreamError("Invalid stream entry payload", {
        cause: parsedInput.error,
      });
    }

    const candidate = {
      ...parsedInput.data,
      id: createStreamEntryId(),
      timestamp,
      session_id: this.sessionId,
      compressed: parsedInput.data.compressed ?? false,
    };

    const parsedEntry = streamEntrySchema.safeParse(candidate);

    if (!parsedEntry.success) {
      throw new StreamError("Failed to construct a valid stream entry", {
        cause: parsedEntry.error,
      });
    }

    return parsedEntry.data;
  }

  private async appendEntries(inputs: readonly StreamEntryInput[]): Promise<StreamEntry[]> {
    const streamDir = getStreamDirectory(this.dataDir);
    const streamPath = getSessionStreamPath(this.dataDir, this.sessionId);
    const lockPath = `${streamPath}.lock`;
    let appendedEntries: StreamEntry[] = [];

    mkdirSync(streamDir, { recursive: true });

    await withFileLock(
      lockPath,
      async () => {
        let fileDescriptor: number | undefined;

        try {
          const entries = inputs.map((input) => this.buildEntry(input, this.clock.now()));
          const serializedEntries = entries.map((entry) => `${serializeJsonValue(entry)}\n`);
          const payload = serializedEntries.join("");

          // We intentionally open the stream file in append mode so the kernel uses
          // O_APPEND semantics for each write, while the lock file provides
          // best-effort cross-process serialization around multi-line appends.
          fileDescriptor = openSync(streamPath, "a");
          const fileSizeBeforeAppend = fstatSync(fileDescriptor).size;
          const byteOffsets: number[] = [];
          let nextByteOffset = fileSizeBeforeAppend;

          for (const serializedEntry of serializedEntries) {
            byteOffsets.push(nextByteOffset);
            nextByteOffset += Buffer.byteLength(serializedEntry);
          }

          writeFileSync(fileDescriptor, payload);
          fsyncSync(fileDescriptor);

          if (this.entryIndex !== undefined) {
            try {
              for (let index = 0; index < entries.length; index += 1) {
                const entry = entries[index];
                const byteOffset = byteOffsets[index];

                if (entry === undefined || byteOffset === undefined) {
                  continue;
                }

                this.entryIndex.record(entry.id, entry.session_id, byteOffset, entry.timestamp);
              }
            } catch (error) {
              this.logger.error(`Failed to update stream entry index for ${streamPath}`);
              this.logger.error(error instanceof Error ? error.message : String(error));
            }
          }

          appendedEntries = entries;
        } catch (error) {
          this.logger.error(`Failed to append to stream ${streamPath}`);

          if (error instanceof TypeError) {
            throw new StreamError(`Failed to serialize stream entries for ${streamPath}`, {
              cause: error,
              code: "STREAM_SERIALIZE_FAILED",
            });
          }

          throw new StreamError(`Failed to append to stream ${streamPath}`, {
            cause: error,
          });
        } finally {
          if (fileDescriptor !== undefined) {
            closeSync(fileDescriptor);
          }
        }
      },
      {
        timeoutMs: this.lockTimeoutMs,
        retryDelayMs: this.lockRetryDelayMs,
      },
    );

    return appendedEntries;
  }

  async append(input: StreamEntryInput): Promise<StreamEntry> {
    this.ensureOpen();
    const [entry] = await this.appendEntries([input]);

    if (entry === undefined) {
      throw new StreamError("Failed to append stream entry");
    }

    return entry;
  }

  async appendMany(inputs: readonly StreamEntryInput[]): Promise<StreamEntry[]> {
    this.ensureOpen();

    if (inputs.length === 0) {
      return [];
    }

    return this.appendEntries(inputs);
  }

  close(): void {
    this.closed = true;
  }
}
