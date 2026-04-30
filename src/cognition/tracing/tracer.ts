import { closeSync, fsyncSync, mkdirSync, openSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { performance } from "node:perf_hooks";

import { SystemClock, type Clock } from "../../util/clock.js";
import { serializeJsonValue, type JsonValue } from "../../util/json-value.js";

export type TurnTraceEventName =
  | "perception_started"
  | "perception_completed"
  | "perception_classifier_degraded"
  | "working_memory_degraded"
  | "recency_compiled"
  | "retrieval_degraded"
  | "retrieval_started"
  | "retrieval_completed"
  | "path_selected"
  | "llm_call_started"
  | "llm_call_response"
  | "plan_extraction"
  | "s2_planner_exhausted"
  | "plan_persisted"
  | "plan_persistence_skipped"
  | "generation_suppressed"
  | "discourse_state_set"
  | "discourse_state_cleared"
  | "discourse_state_hard_cap"
  | "citation_unresolved"
  | "tool_call_dispatched"
  | "tool_call_completed"
  | "commitment_check"
  | "reflection_emitted";

export type TurnTraceData = {
  turnId: string;
  [key: string]: JsonValue | undefined;
};

export type TurnTracer = {
  readonly enabled: boolean;
  readonly includePayloads: boolean;
  emit(event: TurnTraceEventName, data: TurnTraceData): void;
};

export class NoopTracer implements TurnTracer {
  readonly enabled = false;
  readonly includePayloads = false;

  emit(): void {
    // Intentionally empty.
  }
}

export const NOOP_TRACER = new NoopTracer();

export function toTraceJsonValue(value: unknown): JsonValue {
  const serialized = JSON.stringify(value);

  if (serialized === undefined) {
    return null;
  }

  return JSON.parse(serialized) as JsonValue;
}

export type JsonlTracerOptions = {
  path: string;
  clock?: Clock;
  includePayloads?: boolean;
};

function appendJsonlLine(filePath: string, line: string): void {
  mkdirSync(dirname(filePath), { recursive: true });

  let fileDescriptor: number | undefined;

  try {
    fileDescriptor = openSync(filePath, "a");
    // One O_APPEND write per event, followed by fsync, keeps each JSONL record
    // crash-visible without rewriting the whole trace file.
    writeFileSync(fileDescriptor, line);
    fsyncSync(fileDescriptor);
  } finally {
    if (fileDescriptor !== undefined) {
      closeSync(fileDescriptor);
    }
  }
}

export class JsonlTracer implements TurnTracer {
  readonly enabled = true;
  readonly includePayloads: boolean;
  private readonly clock: Clock;
  private readonly path: string;

  constructor(options: JsonlTracerOptions) {
    this.path = options.path;
    this.clock = options.clock ?? new SystemClock();
    this.includePayloads = options.includePayloads ?? false;
  }

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    const { turnId, ...payload } = data;
    // ts uses the injected logical clock (ManualClock in tests, SystemClock
    // in prod) so trace event ordering follows Borg's logical time.
    // wallMs is high-resolution monotonic real time -- needed for intra-
    // turn latency measurement (e.g., metrics.capture's
    // retrieval/deliberation latency calculations) because under a
    // ManualClock all events within one turn share the same logical ts.
    const entry: Record<string, JsonValue> = {
      ts: this.clock.now(),
      wallMs: performance.now(),
      turnId,
      event,
    };

    for (const [key, value] of Object.entries(payload)) {
      if (value !== undefined) {
        entry[key] = value;
      }
    }

    appendJsonlLine(this.path, `${serializeJsonValue(entry)}\n`);
  }
}

export type CreateTurnTracerOptions = {
  tracerPath?: string;
  env?: NodeJS.ProcessEnv;
  clock?: Clock;
  includePayloads?: boolean;
};

export function createTurnTracer(options: CreateTurnTracerOptions = {}): TurnTracer {
  const tracePath = options.tracerPath?.trim() || options.env?.BORG_TRACE?.trim() || "";

  if (tracePath.length === 0) {
    return NOOP_TRACER;
  }

  return new JsonlTracer({
    path: tracePath,
    clock: options.clock,
    includePayloads: options.includePayloads ?? options.env?.BORG_TRACE_PROMPTS === "1",
  });
}
