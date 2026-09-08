import type { LanceDbMemoryStats, LanceDbTableStats, TurnTracer } from "borg";
import { z } from "zod";

const configSchema = z.object({
  intervalMs: z.coerce
    .number()
    .int()
    .min(0)
    .max(2_147_483_647)
    .default(5 * 60_000),
  tableIntervalMs: z.coerce
    .number()
    .int()
    .min(0)
    .max(2_147_483_647)
    .default(30 * 60_000),
  historyLimit: z.coerce.number().int().min(1).max(2016).default(288),
});

export type MemorySamplerConfig = z.infer<typeof configSchema>;

export function readMemorySamplerConfig(env: NodeJS.ProcessEnv = process.env): MemorySamplerConfig {
  return configSchema.parse({
    intervalMs: env.DEMO_MEMORY_SAMPLE_INTERVAL_MS,
    tableIntervalMs: env.DEMO_MEMORY_TABLE_INTERVAL_MS,
    historyLimit: env.DEMO_MEMORY_HISTORY_LIMIT,
  });
}

type StorageProvider = {
  storageMemoryStats(): LanceDbMemoryStats;
  storageTableStats(): Promise<LanceDbTableStats[]>;
};

type TableSample = {
  sampledAt: number;
  durationMs: number;
  tables: LanceDbTableStats[];
  error?: string;
};

export type MemorySample = {
  event: "process.memory.sample";
  ts: number;
  pid: number;
  rss: number;
  heapUsed: number;
  heapTotal: number;
  external: number;
  arrayBuffers: number;
  uptime: number;
  activeTurns: number;
  ledgerCacheEntries: number;
  bufferedSessions: number;
  clients: number;
  lance: LanceDbMemoryStats | null;
  lanceTables: TableSample | null;
  tableStatsPending: boolean;
  storageError?: string;
};

export type MemorySamplerOptions = {
  config?: MemorySamplerConfig;
  getStorage(): StorageProvider | null;
  getActivity(): Pick<
    MemorySample,
    "activeTurns" | "ledgerCacheEntries" | "bufferedSessions" | "clients"
  >;
  tracer?: TurnTracer;
  log?: (line: string) => void;
  logger?: Pick<Console, "error">;
  memoryUsage?: () => NodeJS.MemoryUsage;
  uptime?: () => number;
  now?: () => number;
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export class MemorySampler {
  readonly config: MemorySamplerConfig;
  private readonly samples: MemorySample[] = [];
  private nextSample = 0;
  private timer: ReturnType<typeof setInterval> | undefined;
  private started = false;
  private source: StorageProvider | null = null;
  private tableSample: TableSample | null = null;
  private tableRead: Promise<void> | undefined;
  private lastTableAttempt: number | undefined;

  constructor(private readonly options: MemorySamplerOptions) {
    this.config = configSchema.parse(options.config ?? {});
  }

  private now(): number {
    return (this.options.now ?? Date.now)();
  }

  start(): void {
    if (this.started || this.config.intervalMs === 0) return;
    this.started = true;
    const tick = () => {
      try {
        this.sample();
      } catch (error) {
        (this.options.logger ?? console).error("Memory sampler failed", error);
      }
    };
    this.timer = setInterval(tick, this.config.intervalMs);
    this.timer.unref();
    tick();
  }

  async stop(): Promise<void> {
    if (this.timer !== undefined) clearInterval(this.timer);
    this.timer = undefined;
    this.started = false;
    // Finish the single metadata read before Borg closes its native tables.
    await this.tableRead;
  }

  snapshot(): {
    config: MemorySamplerConfig;
    latest: MemorySample | null;
    history: MemorySample[];
  } {
    const history =
      this.samples.length < this.config.historyLimit
        ? [...this.samples]
        : [...this.samples.slice(this.nextSample), ...this.samples.slice(0, this.nextSample)];
    return { config: { ...this.config }, latest: history.at(-1) ?? null, history };
  }

  sample(): MemorySample {
    const ts = this.now();
    const source = this.options.getStorage();
    if (source !== this.source) {
      this.source = source;
      this.tableSample = null;
      this.lastTableAttempt = undefined;
    }
    let lance: LanceDbMemoryStats | null = null;
    let storageError: string | undefined;
    try {
      lance = source?.storageMemoryStats() ?? null;
    } catch (error) {
      storageError = errorMessage(error);
    }

    if (
      source !== null &&
      this.config.tableIntervalMs > 0 &&
      this.tableRead === undefined &&
      (this.lastTableAttempt === undefined ||
        ts - this.lastTableAttempt >= this.config.tableIntervalMs)
    ) {
      this.lastTableAttempt = ts;
      this.tableRead = this.readTables(source, ts).finally(() => {
        this.tableRead = undefined;
      });
    }

    const sample: MemorySample = {
      event: "process.memory.sample",
      ts,
      pid: process.pid,
      ...(this.options.memoryUsage ?? process.memoryUsage)(),
      uptime: (this.options.uptime ?? process.uptime)(),
      ...this.options.getActivity(),
      lance,
      lanceTables: this.tableSample,
      tableStatsPending: this.tableRead !== undefined,
      ...(storageError === undefined ? {} : { storageError }),
    };
    if (this.samples.length < this.config.historyLimit) this.samples.push(sample);
    else this.samples[this.nextSample] = sample;
    this.nextSample = (this.nextSample + 1) % this.config.historyLimit;
    (this.options.log ?? console.log)(JSON.stringify(sample));
    try {
      this.options.tracer?.emit("process.memory.sample", {
        turnId: `process_memory_${sample.pid}_${ts}`,
        ...sample,
      });
    } catch (error) {
      (this.options.logger ?? console).error("Memory sample trace failed", error);
    }
    return sample;
  }

  private async readTables(source: StorageProvider, startedAt: number): Promise<void> {
    let tables: LanceDbTableStats[] = [];
    let error: string | undefined;
    try {
      tables = await source.storageTableStats();
    } catch (cause) {
      error = errorMessage(cause);
    }
    // Reset may swap the Borg handle while an old metadata read is finishing.
    if (this.options.getStorage() !== source) return;
    this.tableSample = {
      sampledAt: this.now(),
      durationMs: this.now() - startedAt,
      tables,
      ...(error === undefined ? {} : { error }),
    };
  }
}
