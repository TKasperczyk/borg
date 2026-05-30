import {
  closeSync,
  fsyncSync,
  fstatSync,
  mkdirSync,
  openSync,
  readSync,
  writeFileSync,
} from "node:fs";

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
} from "./types.js";

type LoggerLike = Pick<Console, "error">;

const FORWARD_SCAN_CHUNK_SIZE_BYTES = 64 * 1024;
const NEWLINE_BYTE = 0x0a;

export type StreamWriterOptions = {
  dataDir: string;
  sessionId?: SessionId;
  clock?: Clock;
  logger?: LoggerLike;
  lockTimeoutMs?: number;
  lockRetryDelayMs?: number;
  entryIndex?: StreamEntryIndexRepository;
  onAppend?: (entries: readonly StreamEntry[]) => void;
};

export class StreamWriter {
  private readonly dataDir: string;
  private readonly sessionId: SessionId;
  private readonly clock: Clock;
  private readonly logger: LoggerLike;
  private readonly lockTimeoutMs: number;
  private readonly lockRetryDelayMs: number;
  private readonly entryIndex?: StreamEntryIndexRepository;
  private readonly onAppend?: (entries: readonly StreamEntry[]) => void;
  private closed = false;

  constructor(options: StreamWriterOptions) {
    this.dataDir = options.dataDir;
    this.sessionId = options.sessionId ?? DEFAULT_SESSION_ID;
    this.clock = options.clock ?? new SystemClock();
    this.logger = options.logger ?? console;
    this.lockTimeoutMs = options.lockTimeoutMs ?? 2_000;
    this.lockRetryDelayMs = options.lockRetryDelayMs ?? 20;
    this.entryIndex = options.entryIndex;
    this.onAppend = options.onAppend;
  }

  private ensureOpen(): void {
    if (this.closed) {
      throw new StreamError("StreamWriter is closed");
    }
  }

  private buildEntry(input: StreamEntryInput, timestamp: number, entryIndex: number): StreamEntry {
    const parsedInput = streamEntryInputSchema.safeParse(input);

    if (!parsedInput.success) {
      throw new StreamError("Invalid stream entry payload", {
        cause: parsedInput.error,
      });
    }

    const entry: StreamEntry = {
      ...parsedInput.data,
      id: createStreamEntryId(),
      timestamp,
      entry_index: entryIndex,
      session_id: this.sessionId,
      compressed: parsedInput.data.compressed ?? false,
    };

    return entry;
  }

  private countExistingEntries(fileDescriptor: number, fileSize: number): number {
    let position = 0;
    let count = 0;

    while (position < fileSize) {
      const chunkSize = Math.min(FORWARD_SCAN_CHUNK_SIZE_BYTES, fileSize - position);
      const chunk = Buffer.allocUnsafe(chunkSize);
      const bytesRead = readSync(fileDescriptor, chunk, 0, chunkSize, position);

      if (bytesRead <= 0) {
        break;
      }

      for (let index = 0; index < bytesRead; index += 1) {
        if (chunk[index] === NEWLINE_BYTE) {
          count += 1;
        }
      }

      position += bytesRead;
    }

    return count;
  }

  private poisonedIndexError(streamPath: string, cause: unknown): StreamError {
    return new StreamError(
      `Stream entry index is poisoned for committed session ${this.sessionId}`,
      {
        cause,
        code: "STREAM_INDEX_POISONED",
      },
    );
  }

  private async repairPoisonedSessionBeforeAppend(streamPath: string): Promise<void> {
    if (this.entryIndex === undefined) {
      return;
    }

    try {
      await this.entryIndex.backfillSession(this.sessionId);
    } catch (repairError) {
      this.logger.error("Failed to repair poisoned stream entry index before append", {
        streamPath,
        sessionId: this.sessionId,
        repairCause: repairError instanceof Error ? repairError.message : String(repairError),
      });

      throw this.poisonedIndexError(streamPath, repairError);
    }
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
          if (this.entryIndex?.isPoisoned(this.sessionId) === true) {
            await this.repairPoisonedSessionBeforeAppend(streamPath);
          }

          // We intentionally open the stream file in append mode so the kernel uses
          // O_APPEND semantics for each write, while the lock file provides
          // best-effort cross-process serialization around multi-line appends.
          fileDescriptor = openSync(streamPath, "a+");
          const fileSizeBeforeAppend = fstatSync(fileDescriptor).size;
          const firstEntryIndex =
            this.entryIndex?.nextEntryIndex(this.sessionId) ??
            this.countExistingEntries(fileDescriptor, fileSizeBeforeAppend);
          const entries: StreamEntry[] = [];
          const serializedEntries: string[] = [];
          const byteOffsets: number[] = [];
          let nextByteOffset = fileSizeBeforeAppend;

          for (let inputIndex = 0; inputIndex < inputs.length; inputIndex += 1) {
            const input = inputs[inputIndex];

            if (input === undefined) {
              continue;
            }

            const entry = this.buildEntry(input, this.clock.now(), firstEntryIndex + inputIndex);
            const serializedEntry = `${serializeJsonValue(entry)}\n`;

            entries.push(entry);
            serializedEntries.push(serializedEntry);
            byteOffsets.push(nextByteOffset);
            nextByteOffset += Buffer.byteLength(serializedEntry);
          }

          const payload = serializedEntries.join("");

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

                this.entryIndex.recordEntry(entry, byteOffset);
              }
            } catch (error) {
              this.logger.error("Failed to update stream entry index after committed append", {
                streamPath,
                sessionId: this.sessionId,
                entryIds: entries.map((entry) => entry.id),
                cause: error instanceof Error ? error.message : String(error),
              });

              try {
                await this.entryIndex.backfillSession(this.sessionId);
              } catch (repairError) {
                this.logger.error("Failed to repair stream entry index after committed append", {
                  streamPath,
                  sessionId: this.sessionId,
                  entryIds: entries.map((entry) => entry.id),
                  updateCause: error instanceof Error ? error.message : String(error),
                  repairCause:
                    repairError instanceof Error ? repairError.message : String(repairError),
                });

                this.entryIndex.markPoisoned(this.sessionId);
                throw this.poisonedIndexError(streamPath, repairError);
              }
            }
          }

          appendedEntries = entries;
        } catch (error) {
          if (error instanceof StreamError) {
            throw error;
          }

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

    if (this.onAppend !== undefined && appendedEntries.length > 0) {
      try {
        this.onAppend(appendedEntries);
      } catch (error) {
        this.logger.error("Stream append observer failed", {
          streamPath,
          sessionId: this.sessionId,
          entryIds: appendedEntries.map((entry) => entry.id),
          cause: error instanceof Error ? error.message : String(error),
        });
      }
    }

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
