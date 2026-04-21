import { existsSync, readFileSync } from "node:fs";
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

  async *iterate(options: StreamIterateOptions = {}): AsyncGenerator<StreamEntry> {
    if (!existsSync(this.streamPath)) {
      return;
    }

    const allowedKinds =
      options.kinds === undefined ? undefined : new Set<StreamEntry["kind"]>(options.kinds);

    const input = createReadStream(this.streamPath, { encoding: "utf8" });
    const lines = createInterface({ input, crlfDelay: Infinity });
    let emitted = 0;

    try {
      for await (const line of lines) {
        const entry = this.parseLine(line);

        if (entry === undefined || !this.matchesFilters(entry, options, allowedKinds)) {
          continue;
        }

        yield entry;
        emitted += 1;

        if (options.limit !== undefined && emitted >= options.limit) {
          break;
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

    const raw = readFileSync(this.streamPath, "utf8");
    const lines = raw.split("\n");
    const entries: StreamEntry[] = [];

    for (let index = lines.length - 1; index >= 0 && entries.length < n; index -= 1) {
      const entry = this.parseLine(lines[index] ?? "");
      if (entry !== undefined) {
        entries.push(entry);
      }
    }

    return entries.reverse();
  }
}
