import {
  CallbackTracer,
  type CallbackTraceEntry,
  type TurnTraceData,
  type TurnTraceEventName,
  type TurnTracer,
} from "../tracing/tracer.js";

export const DEFAULT_MEMORY_TRACE_CAPACITY = 200;
export const DEFAULT_MEMORY_TRACE_MAX_TENANTS = 64;

export type MemoryTraceQueryResult = {
  events: CallbackTraceEntry[];
  nextSince: number;
  truncated: boolean;
};

export type MemoryTraceRegistryOptions = {
  capacity?: number;
  maxTenants?: number;
  includePayloads?: boolean;
  now?: () => number;
};

function normalizePositiveInteger(value: number | undefined, fallback: number): number {
  if (value === undefined || !Number.isFinite(value)) {
    return fallback;
  }

  return Math.max(1, Math.floor(value));
}

export function memoryTraceEnabledFromEnv(env: NodeJS.ProcessEnv = process.env): boolean {
  const raw = env.BORG_MEMORY_TRACE_ENABLED?.trim().toLowerCase();
  return raw === "1" || raw === "true";
}

export function memoryTraceCapacityFromEnv(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.BORG_MEMORY_TRACE_CAP;
  if (raw === undefined || raw.trim() === "") {
    return DEFAULT_MEMORY_TRACE_CAPACITY;
  }

  return normalizePositiveInteger(Number(raw), DEFAULT_MEMORY_TRACE_CAPACITY);
}

export function memoryTraceMaxTenantsFromEnv(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.BORG_MEMORY_TRACE_MAX_TENANTS;
  if (raw === undefined || raw.trim() === "") {
    return DEFAULT_MEMORY_TRACE_MAX_TENANTS;
  }

  return normalizePositiveInteger(Number(raw), DEFAULT_MEMORY_TRACE_MAX_TENANTS);
}

function shouldStoreMemoryTraceEvent(event: TurnTraceEventName, data: TurnTraceData): boolean {
  return (
    event === "recall_expansion.completed" ||
    event === "commitment_classification.downgraded" ||
    event.startsWith("retrieval.") ||
    event.startsWith("extraction.commitments.") ||
    event.startsWith("corrective_preference.") ||
    (event.startsWith("llm_call.") &&
      (data.label === "recall_expansion" || data.label === "corrective_preference_extractor"))
  );
}

export class MemoryTraceRegistry {
  private readonly capacity: number;
  private readonly maxTenants: number;
  private readonly includePayloads: boolean;
  private readonly now: () => number;
  private readonly buffers = new Map<string, CallbackTraceEntry[]>();
  private lastTs = 0;

  constructor(options: MemoryTraceRegistryOptions = {}) {
    this.capacity = normalizePositiveInteger(options.capacity, DEFAULT_MEMORY_TRACE_CAPACITY);
    this.maxTenants = normalizePositiveInteger(
      options.maxTenants,
      DEFAULT_MEMORY_TRACE_MAX_TENANTS,
    );
    this.includePayloads = options.includePayloads ?? true;
    this.now = options.now ?? Date.now;
  }

  tracerFor(tenantId: string): TurnTracer {
    return new CallbackTracer({
      includePayloads: this.includePayloads,
      timestamp: () => this.nextTimestamp(),
      sink: (entry) => {
        this.append(tenantId, entry);
      },
    });
  }

  query(tenantId: string, since = 0): MemoryTraceQueryResult {
    const buffer = this.buffers.get(tenantId) ?? [];
    const events = buffer.filter((entry) => entry.ts > since).map((entry) => ({ ...entry }));
    const lastEvent = events.at(-1);
    const oldest = buffer[0];

    return {
      events,
      nextSince: lastEvent?.ts ?? since,
      truncated: oldest !== undefined && since > 0 && since < oldest.ts,
    };
  }

  tenantBufferCount(): number {
    return this.buffers.size;
  }

  private nextTimestamp(): number {
    const next = Math.max(this.now(), this.lastTs + 1);
    this.lastTs = next;
    return next;
  }

  private append(tenantId: string, entry: CallbackTraceEntry): void {
    if (!shouldStoreMemoryTraceEvent(entry.event, entry)) {
      return;
    }

    const existing = this.buffers.get(tenantId);
    const buffer = existing ?? [];

    if (existing !== undefined) {
      this.buffers.delete(tenantId);
    }

    this.buffers.set(tenantId, buffer);

    while (this.buffers.size > this.maxTenants) {
      const oldestTenant = this.buffers.keys().next().value as string | undefined;
      if (oldestTenant === undefined) {
        break;
      }
      this.buffers.delete(oldestTenant);
    }

    buffer.push(entry);

    if (buffer.length > this.capacity) {
      buffer.splice(0, buffer.length - this.capacity);
    }
  }
}
