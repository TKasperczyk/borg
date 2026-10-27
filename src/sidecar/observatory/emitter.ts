import { hostname } from "node:os";

import { SystemClock, type Clock } from "../../util/clock.js";
import { describeError } from "../../util/errors.js";
import { createIdHelpers } from "../../util/ids.js";
import type { JsonValue } from "../../util/json-value.js";
import { utf16SafePrefix } from "../../util/utf16-boundary.js";

import type { ObservatoryConfig } from "./config.js";

const eventIds = createIdHelpers<"ObservatoryEventId">("evt");
const spanIds = createIdHelpers<"ObservatorySpanId">("sp");
const MAX_BODY_BYTES = 1024 * 1024;
const MAX_DATA_BYTES = 24 * 1024;
const MAX_LIST_ITEMS = 50;
const MAX_DEPTH = 6;
const MAX_VISITED_FIELDS = 200;

export type ObservatoryRefs = {
  thread_id?: string;
  sidecar_session_id?: string;
  entry_id?: string;
  entry_ids?: readonly string[];
  external_message_id?: string;
  context_id?: string;
  run_id?: string;
};
export type ObservatoryStatus = "ok" | "error" | "degraded" | "skipped";
export type ObservatoryEvent = {
  tenant: string;
  kind: string;
  phase: "start" | "end" | "point";
  ts?: number;
  span_id?: string;
  parent_span_id?: string;
  duration_ms?: number;
  status?: ObservatoryStatus;
  turn_id?: string;
  refs?: ObservatoryRefs;
  conversation?: { external_id?: string; type: string; name?: string };
  actor?: { external_id: string; display_name: string };
  data?: Record<string, unknown>;
};
export type ObservatoryStageInput = Omit<ObservatoryEvent, "phase" | "duration_ms" | "status">;
export type ObservatoryStageEnd = {
  status?: ObservatoryStatus;
  data?: Record<string, unknown>;
  refs?: ObservatoryRefs;
  turn_id?: string;
  error?: unknown;
};
export type ObservatoryStage = {
  spanId: string;
  end(input?: ObservatoryStageEnd): void;
};
export type ObservatoryObserver = {
  emit(event: ObservatoryEvent): void;
  stage(input: ObservatoryStageInput): ObservatoryStage;
};
export type ObservatoryEmission = {
  event: ObservatoryEvent;
  filterDataField?: (key: string, parent: Readonly<Record<string, unknown>>, depth: number) => boolean;
  traceTurnId?: string;
};
export type ObservatoryEmitterOptions = {
  config: Extract<ObservatoryConfig, { enabled: true }>;
  clock?: Clock;
  fetchFn?: typeof fetch;
  setTimeoutFn?: (callback: () => void, delayMs: number) => ReturnType<typeof setTimeout>;
  clearTimeoutFn?: (handle: ReturnType<typeof setTimeout>) => void;
};

// These are wire-format budgets, never interpretation of the text. Existing head/tail
// excerpts do not implement the contract's prefix plus removed-character suffix.
type ClampBudget = { remaining: number; truncated: boolean; ancestors: Set<object> };

function clampValue(
  value: unknown,
  key: string,
  path: string,
  lists: Set<string>,
  budget: ClampBudget,
  filterField?: ObservatoryEmission["filterDataField"],
  depth = 0,
): JsonValue {
  if (typeof value === "string") {
    // Slice before JSON serialization / UTF-8 byte scans; this helper inspects
    // only the two code units at the cut when preserving surrogate pairs.
    if (key.endsWith("_preview")) return utf16SafePrefix(value, 280);
    if (value.length <= 4096) return value;
    const prefix = utf16SafePrefix(value, 4096);
    return `${prefix}…[+${value.length - prefix.length} chars]`;
  }
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "object" && value !== null) {
    if (budget.ancestors.has(value)) throw new TypeError("Observatory data must not be cyclic");
    if (depth >= MAX_DEPTH) {
      budget.truncated = true;
      if (Array.isArray(value) && value.length > 0) lists.add(path);
      return "[truncated]";
    }
    budget.ancestors.add(value);
    try {
      if (Array.isArray(value)) {
        const result: JsonValue[] = [];
        const limit = Math.min(value.length, MAX_LIST_ITEMS);
        for (let index = 0; index < limit; index += 1) {
          if (budget.remaining === 0) {
            budget.truncated = true;
            break;
          }
          budget.remaining -= 1;
          result.push(clampValue(value[index], key, path, lists, budget, filterField, depth + 1));
        }
        if (result.length < value.length) lists.add(path);
        return result;
      }
      const record = value as Record<string, unknown>;
      const result: Record<string, JsonValue> = Object.create(null) as Record<string, JsonValue>;
      // Do not materialize Object.entries or copy the input before applying the
      // shared visit budget. Redaction runs before reading a field's value.
      for (const childKey in record) {
        if (budget.remaining === 0) {
          budget.truncated = true;
          break;
        }
        budget.remaining -= 1;
        if (!Object.hasOwn(record, childKey)) continue;
        const boundedKey = utf16SafePrefix(childKey, 4096);
        if (filterField?.(boundedKey, record, depth) === false) continue;
        const child = record[childKey];
        if (child === undefined) continue;
        if (boundedKey.length < childKey.length) budget.truncated = true;
        result[boundedKey] = clampValue(
          child, boundedKey, path ? `${path}.${boundedKey}` : boundedKey,
          lists, budget, filterField, depth + 1,
        );
      }
      return result;
    } finally {
      budget.ancestors.delete(value);
    }
  }
  throw new TypeError("Observatory data must be JSON");
}

export function reportObservatoryObserverError(hook: string, error: unknown): void {
  try {
    console.error("memory-sidecar: observatory observer failed", {
      hook,
      error_name: error instanceof Error ? error.name : typeof error,
    });
  } catch {
    // Even a failing log sink must not affect the observed pipeline.
  }
}

export function emitObservatoryEvent(
  observer: ObservatoryObserver | undefined,
  event: ObservatoryEvent,
): void {
  try {
    observer?.emit(event);
  } catch (error) {
    reportObservatoryObserverError(event.kind, error);
  }
}

export function startObservatoryStage(
  observer: ObservatoryObserver | undefined,
  input: ObservatoryStageInput,
): ObservatoryStage | undefined {
  try {
    return observer?.stage(input);
  } catch (error) {
    reportObservatoryObserverError(input.kind, error);
    return undefined;
  }
}

export function endObservatoryStage(
  stage: ObservatoryStage | undefined,
  input: ObservatoryStageEnd = {},
): void {
  try {
    stage?.end(input);
  } catch (error) {
    reportObservatoryObserverError("stage.end", error);
  }
}

export class ObservatoryEmitter implements ObservatoryObserver {
  private readonly clock: Clock;
  private readonly fetchFn: typeof fetch;
  private readonly setTimeoutFn: NonNullable<ObservatoryEmitterOptions["setTimeoutFn"]>;
  private readonly clearTimeoutFn: NonNullable<ObservatoryEmitterOptions["clearTimeoutFn"]>;
  private readonly instance = hostname();
  private readonly queue: Array<{ json: string; bytes: number }> = [];
  private timer: ReturnType<typeof setTimeout> | undefined;
  private timerImmediate = false;
  private inFlight: Promise<void> | undefined;
  private transportController: AbortController | undefined;
  private closing = false;
  private shutdownAborted = false;
  private shutdownDeadlineAt: number | undefined;
  private shutdownTask: Promise<void> | undefined;
  private dropped = 0;
  private lastDropLogAt = -Infinity;

  constructor(private readonly options: ObservatoryEmitterOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.fetchFn = options.fetchFn ?? fetch;
    this.setTimeoutFn = options.setTimeoutFn ?? setTimeout;
    this.clearTimeoutFn = options.clearTimeoutFn ?? clearTimeout;
  }

  emit(event: ObservatoryEvent): void {
    try {
      this.emitLazy(() => ({ event }));
    } catch {
      this.drop(1);
    }
  }

  emitLazy(create: () => ObservatoryEmission): void {
    try {
      if (this.closing || this.queue.length >= this.options.config.queueCap) {
        this.drop(1);
        return;
      }
      const { event, filterDataField, traceTurnId } = create();
      if (!utf16SafePrefix(event.tenant, 4096).trim()) {
        this.drop(1);
        return;
      }
      const lists = new Set<string>();
      // Reserve three fields for truncation/correlation metadata. The budget is
      // global across objects AND array elements, so nested arrays cannot fan out.
      const budget: ClampBudget = {
        remaining: MAX_VISITED_FIELDS - 3, truncated: false, ancestors: new Set<object>(),
      };
      const { data: inputData, ...envelope } = event;
      const fields = clampValue(envelope, "", "", lists, budget) as Record<string, JsonValue>;
      let data = clampValue(inputData ?? {}, "data", "", lists, budget, filterDataField) as Record<string, JsonValue>;
      if (traceTurnId !== undefined) {
        data.trace_turn_id = clampValue(traceTurnId, "trace_turn_id", "", lists, budget);
      }
      const truncatedLists = [...lists].slice(0, MAX_LIST_ITEMS).map((path) => utf16SafePrefix(path, 4096));
      if (truncatedLists.length > 0) data.truncated_lists = truncatedLists;
      if (budget.truncated) data.truncated = true;
      const bytes = Buffer.byteLength(JSON.stringify(data));
      if (bytes > MAX_DATA_BYTES) {
        const keys = Object.keys(data)
          .slice(0, MAX_LIST_ITEMS)
          .map((key) => utf16SafePrefix(key, 4096));
        data = {
          truncated: true, keys, bytes,
          ...(truncatedLists.length > 0 ? { truncated_lists: truncatedLists } : {}),
        };
        while (Buffer.byteLength(JSON.stringify(data)) > MAX_DATA_BYTES) {
          if (keys.length > 0) keys.pop();
          else if (truncatedLists.length > 0) truncatedLists.pop();
          else break;
        }
      }
      const json = JSON.stringify({
        ...fields,
        v: 1,
        id: eventIds.create(),
        ts: event.ts ?? this.clock.now(),
        instance: this.instance,
        source: "borg",
        ...(event.phase === "point" ? {} : { span_id: fields.span_id ?? spanIds.create() }),
        ...(event.phase === "end" ? { status: event.status ?? "ok" } : {}),
        data,
      });
      const eventBytes = Buffer.byteLength(json);
      if (eventBytes + 14 > MAX_BODY_BYTES) {
        this.drop(1);
        return;
      }
      this.queue.push({ json, bytes: eventBytes });
      this.schedule();
    } catch {
      this.drop(1);
    }
  }

  stage(input: ObservatoryStageInput): ObservatoryStage {
    try {
      const spanId = input.span_id ?? spanIds.create();
      const startedAt = input.ts ?? this.clock.now();
      this.emit({ ...input, ts: startedAt, span_id: spanId, phase: "start" });
      let ended = false;
      return {
        spanId,
        end: (end = {}) => {
          if (ended) return;
          ended = true;
          try {
            this.emit({
              ...input,
              ts: this.clock.now(),
              span_id: spanId,
              phase: "end",
              duration_ms: Math.max(0, this.clock.now() - startedAt),
              status: end.status ?? (end.error === undefined ? "ok" : "error"),
              turn_id: end.turn_id ?? input.turn_id,
              refs: { ...input.refs, ...end.refs },
              data: {
                ...input.data,
                ...end.data,
                ...(end.error === undefined
                  ? {}
                  : {
                      error_type: end.error instanceof Error ? end.error.name : typeof end.error,
                      error: describeError(end.error),
                    }),
              },
            });
          } catch {
            this.drop(1);
          }
        },
      };
    } catch {
      this.drop(1);
      return { spanId: "", end() {} };
    }
  }

  shutdown(options: { deadlineAt?: number; signal?: AbortSignal } = {}): Promise<void> {
    this.shutdownTask ??= this.finishShutdown(options);
    return this.shutdownTask;
  }

  private async finishShutdown(options: { deadlineAt?: number; signal?: AbortSignal }): Promise<void> {
    let deadlineTimer: ReturnType<typeof setTimeout> | undefined;
    const abort = (): void => this.abortShutdown();
    try {
      this.closing = true;
      this.shutdownDeadlineAt = options.deadlineAt;
      if (this.timer !== undefined) this.clearTimeoutFn(this.timer);
      this.timer = undefined;
      options.signal?.addEventListener("abort", abort, { once: true });
      if (
        options.signal?.aborted ||
        (options.deadlineAt !== undefined && options.deadlineAt <= this.clock.now())
      ) {
        this.abortShutdown();
        return;
      }
      if (options.deadlineAt !== undefined) {
        deadlineTimer = this.setTimeoutFn(abort, Math.max(0, options.deadlineAt - this.clock.now()));
        deadlineTimer.unref?.();
      }
      await this.inFlight;
      // One final bounded batch without retry, using only the remaining drain budget.
      if (!this.shutdownAborted) await this.flushBatch(false);
    } catch {
      this.abortShutdown();
    } finally {
      options.signal?.removeEventListener("abort", abort);
      if (this.queue.length > 0) this.drop(this.queue.length);
      this.queue.length = 0;
      try {
        if (deadlineTimer !== undefined) this.clearTimeoutFn(deadlineTimer);
      } catch (error) {
        reportObservatoryObserverError("shutdown.timer", error);
      }
    }
  }

  private abortShutdown(): void {
    this.shutdownAborted = true;
    if (this.queue.length > 0) this.drop(this.queue.length);
    this.queue.length = 0;
    this.transportController?.abort();
  }

  private schedule(): void {
    if (this.closing || this.inFlight !== undefined || this.queue.length === 0) return;
    const full = this.queue.length >= this.options.config.maxBatch;
    if (this.timer !== undefined) {
      if (!full || this.timerImmediate) return;
      this.clearTimeoutFn(this.timer);
    }
    this.timerImmediate = full;
    this.timer = this.setTimeoutFn(() => {
      this.timer = undefined;
      this.timerImmediate = false;
      this.inFlight = this.drainBatch();
    }, full ? 0 : this.options.config.flushMs);
    this.timer.unref?.();
  }

  private async drainBatch(): Promise<void> {
    try {
      await this.flushBatch(true);
    } catch {
      this.drop(1);
    } finally {
      this.inFlight = undefined;
      try {
        this.schedule();
      } catch {
        this.drop(this.queue.length);
        this.queue.length = 0;
      }
    }
  }

  private async flushBatch(retry: boolean): Promise<void> {
    const batch: string[] = [];
    let bytes = 13; // {"events":[]} and a conservative comma allowance
    while (batch.length < this.options.config.maxBatch && batch.length < 500) {
      const next = this.queue[0];
      if (next === undefined || bytes + next.bytes + 1 > MAX_BODY_BYTES) break;
      this.queue.shift();
      batch.push(next.json);
      bytes += next.bytes + 1;
    }
    if (batch.length === 0) return;
    const body = `{"events":[${batch.join(",")}]}`;
    for (let attempt = 0; attempt < (retry ? 2 : 1); attempt += 1) {
      if (this.shutdownDeadlineAt !== undefined && this.clock.now() >= this.shutdownDeadlineAt) {
        this.abortShutdown();
      }
      if (this.shutdownAborted) break;
      const controller = new AbortController();
      this.transportController = controller;
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        const response = await Promise.race([
          this.fetchFn(`${this.options.config.baseUrl.replace(/\/$/, "")}/ingest/events`, {
            method: "POST",
            redirect: "manual",
            headers: {
              authorization: `Bearer ${this.options.config.token}`,
              "content-type": "application/json",
            },
            signal: controller.signal,
            body,
          }),
          new Promise<never>((_, reject) => {
            // Race cancellation too: an injected or stalled fetch may ignore its signal.
            controller.signal.addEventListener("abort", () => {
              reject(new Error("Observatory transport aborted"));
            }, { once: true });
            timer = this.setTimeoutFn(() => controller.abort(), this.options.config.timeoutMs);
            timer.unref?.();
          }),
        ]);
        // No response content is needed or logged. Abort releases the response body too.
        if (response.ok) return;
        if (response.status < 500 || response.status >= 600) break;
      } catch {
        // Network errors get the same single retry as 5xx responses.
      } finally {
        controller.abort();
        this.transportController = undefined;
        if (timer !== undefined) this.clearTimeoutFn(timer);
      }
    }
    this.drop(batch.length);
  }

  private drop(count: number): void {
    try {
      this.dropped += count;
      const now = this.clock.now();
      if (now - this.lastDropLogAt < 60_000) return;
      this.lastDropLogAt = now;
      console.error("memory-sidecar: observatory events dropped", { dropped_total: this.dropped });
    } catch {
      // Telemetry, including its logging, is best effort.
    }
  }
}
