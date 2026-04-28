import { closeSync, existsSync, fstatSync, openSync, readSync } from "node:fs";
import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";

import { StreamError } from "../util/errors.js";

import { getSessionStreamPath } from "./path.js";
import {
  DEFAULT_SESSION_ID,
  type SessionId,
  type StreamEntry,
  type StreamIterateOptions,
  streamEntrySchema,
} from "./types.js";

type LoggerLike = Pick<Console, "error">;

const REVERSE_TAIL_CHUNK_SIZE_BYTES = 64 * 1024;
const NEWLINE_BYTE = 0x0a;

function reverseLineToString(
  lineSegment: Buffer,
  carryChunks: readonly Buffer[],
  carryLength: number,
): string {
  if (carryChunks.length === 0) {
    return lineSegment.toString("utf8");
  }

  if (lineSegment.length === 0) {
    return Buffer.concat(carryChunks, carryLength).toString("utf8");
  }

  return Buffer.concat([lineSegment, ...carryChunks], lineSegment.length + carryLength).toString(
    "utf8",
  );
}

export type StreamReaderOptions = {
  dataDir: string;
  sessionId?: SessionId;
  logger?: LoggerLike;
};

export class StreamReader {
  private readonly dataDir: string;
  private readonly sessionId: SessionId;
  private readonly logger: LoggerLike;

  constructor(options: StreamReaderOptions) {
    this.dataDir = options.dataDir;
    this.sessionId = options.sessionId ?? DEFAULT_SESSION_ID;
    this.logger = options.logger ?? console;
  }

  private get streamPath(): string {
    return getSessionStreamPath(this.dataDir, this.sessionId);
  }

  private parseLine(line: string): StreamEntry | undefined {
    if (line.trim() === "") {
      return undefined;
    }

    try {
      const raw = JSON.parse(line) as unknown;
      const parsed = streamEntrySchema.safeParse(raw);

      if (!parsed.success) {
        this.logger.error(`Skipping invalid stream line in ${this.streamPath}`);
        return undefined;
      }

      return parsed.data;
    } catch (error) {
      this.logger.error(`Skipping unreadable stream line in ${this.streamPath}`);
      this.logger.error(error instanceof Error ? error.message : String(error));
      return undefined;
    }
  }

  private matchesFilters(
    entry: StreamEntry,
    options: StreamIterateOptions,
    allowedKinds: Set<StreamEntry["kind"]> | undefined,
  ): boolean {
    if (options.sinceTs !== undefined && entry.timestamp < options.sinceTs) {
      return false;
    }

    if (options.untilTs !== undefined && entry.timestamp > options.untilTs) {
      return false;
    }

    if (allowedKinds !== undefined && !allowedKinds.has(entry.kind)) {
      return false;
    }

    return true;
  }

  private resolveCursorBuffer(
    bufferedEntries: readonly StreamEntry[],
    cursorEntryId: StreamEntry["id"],
  ): StreamEntry[] {
    const cursorIndex = bufferedEntries.findIndex((entry) => entry.id === cursorEntryId);
    return cursorIndex === -1 ? [...bufferedEntries] : bufferedEntries.slice(cursorIndex + 1);
  }

  private matchesCursorLowerBound(entry: StreamEntry, cursorTs: number | undefined): boolean {
    return cursorTs === undefined || entry.timestamp >= cursorTs;
  }

  async *iterate(options: StreamIterateOptions = {}): AsyncGenerator<StreamEntry> {
    if (!existsSync(this.streamPath)) {
      return;
    }

    const allowedKinds =
      options.kinds === undefined ? undefined : new Set<StreamEntry["kind"]>(options.kinds);
    const filterOptions =
      options.sinceCursor === undefined ? options : { ...options, sinceTs: undefined };
    const cursor = options.sinceCursor;
    const cursorTs = cursor?.ts;
    const untilCursor = options.untilCursor;
    let reachedUntilCursor = false;

    const isPastUntilBound = (entry: StreamEntry): boolean => {
      if (untilCursor === undefined) {
        return false;
      }

      if (reachedUntilCursor) {
        return true;
      }

      return entry.timestamp > untilCursor.ts;
    };

    const markUntilBound = (entry: StreamEntry): void => {
      if (
        untilCursor !== undefined &&
        entry.timestamp === untilCursor.ts &&
        entry.id === untilCursor.entryId
      ) {
        reachedUntilCursor = true;
      }
    };

    const isUntilBound = (entry: StreamEntry): boolean =>
      untilCursor !== undefined &&
      entry.timestamp === untilCursor.ts &&
      entry.id === untilCursor.entryId;

    const input = createReadStream(this.streamPath, { encoding: "utf8" });
    const lines = createInterface({ input, crlfDelay: Infinity });
    let emitted = 0;
    let passedCursor = cursor === undefined;
    let bufferingCursorTs = false;
    let cursorBuffer: StreamEntry[] = [];

    try {
      outer: for await (const line of lines) {
        const entry = this.parseLine(line);

        if (entry === undefined) {
          continue;
        }

        if (!passedCursor && cursor !== undefined) {
          if (entry.timestamp < cursor.ts) {
            continue;
          }

          if (entry.timestamp === cursor.ts) {
            bufferingCursorTs = true;
            cursorBuffer.push(entry);
            continue;
          }

          passedCursor = true;
          for (const bufferedEntry of this.resolveCursorBuffer(cursorBuffer, cursor.entryId)) {
            if (isPastUntilBound(bufferedEntry)) {
              break outer;
            }

            const isUntilEntry = isUntilBound(bufferedEntry);

            if (!this.matchesCursorLowerBound(bufferedEntry, cursorTs)) {
              if (isUntilEntry) {
                break outer;
              }
              continue;
            }

            if (!this.matchesFilters(bufferedEntry, filterOptions, allowedKinds)) {
              if (isUntilEntry) {
                break outer;
              }
              continue;
            }

            yield bufferedEntry;
            markUntilBound(bufferedEntry);
            emitted += 1;

            if (
              reachedUntilCursor ||
              (options.limit !== undefined && emitted >= options.limit)
            ) {
              break outer;
            }
          }

          cursorBuffer = [];
          bufferingCursorTs = false;
        }

        if (!this.matchesCursorLowerBound(entry, cursorTs)) {
          continue;
        }

        if (isPastUntilBound(entry)) {
          break;
        }

        const isUntilEntry = isUntilBound(entry);

        if (!this.matchesFilters(entry, filterOptions, allowedKinds)) {
          if (isUntilEntry) {
            break;
          }
          continue;
        }

        yield entry;
        markUntilBound(entry);
        emitted += 1;

        if (reachedUntilCursor || (options.limit !== undefined && emitted >= options.limit)) {
          break;
        }
      }

      if (
        !reachedUntilCursor &&
        !passedCursor &&
        cursor !== undefined &&
        (bufferingCursorTs || cursorBuffer.length > 0)
      ) {
        for (const bufferedEntry of this.resolveCursorBuffer(cursorBuffer, cursor.entryId)) {
          if (isPastUntilBound(bufferedEntry)) {
            break;
          }

          const isUntilEntry = isUntilBound(bufferedEntry);

          if (!this.matchesCursorLowerBound(bufferedEntry, cursorTs)) {
            if (isUntilEntry) {
              break;
            }
            continue;
          }

          if (!this.matchesFilters(bufferedEntry, filterOptions, allowedKinds)) {
            if (isUntilEntry) {
              break;
            }
            continue;
          }

          yield bufferedEntry;
          markUntilBound(bufferedEntry);
          emitted += 1;

          if (reachedUntilCursor || (options.limit !== undefined && emitted >= options.limit)) {
            break;
          }
        }
      }
    } catch (error) {
      throw new StreamError(`Failed to read stream ${this.streamPath}`, {
        cause: error,
      });
    } finally {
      lines.close();
      input.close();
    }
  }

  tail(n: number): StreamEntry[] {
    if (n <= 0 || !existsSync(this.streamPath)) {
      return [];
    }

    const fileDescriptor = openSync(this.streamPath, "r");
    const entries: StreamEntry[] = [];

    try {
      const fileSize = fstatSync(fileDescriptor).size;

      if (fileSize === 0) {
        return [];
      }

      let position = fileSize;
      const carryChunks: Buffer[] = [];
      let carryLength = 0;

      while (position > 0 && entries.length < n) {
        const chunkSize = Math.min(REVERSE_TAIL_CHUNK_SIZE_BYTES, position);
        position -= chunkSize;

        const chunk = Buffer.allocUnsafe(chunkSize);
        const bytesRead = readSync(fileDescriptor, chunk, 0, chunkSize, position);
        if (bytesRead <= 0) {
          break;
        }

        const chunkBytes = bytesRead === chunkSize ? chunk : chunk.subarray(0, bytesRead);
        let lineEnd = chunkBytes.length;

        for (let index = chunkBytes.length - 1; index >= 0 && entries.length < n; index -= 1) {
          if (chunkBytes[index] !== NEWLINE_BYTE) {
            continue;
          }

          const entry = this.parseLine(
            reverseLineToString(chunkBytes.subarray(index + 1, lineEnd), carryChunks, carryLength),
          );

          if (entry !== undefined) {
            entries.push(entry);
          }

          carryChunks.length = 0;
          carryLength = 0;
          lineEnd = index;
        }

        if (lineEnd > 0) {
          const remainder = chunkBytes.subarray(0, lineEnd);
          carryChunks.unshift(remainder);
          carryLength += remainder.length;
        }
      }

      if (entries.length < n && carryLength > 0) {
        const entry = this.parseLine(Buffer.concat(carryChunks, carryLength).toString("utf8"));

        if (entry !== undefined) {
          entries.push(entry);
        }
      }
    } finally {
      closeSync(fileDescriptor);
    }

    return entries.reverse();
  }
}
