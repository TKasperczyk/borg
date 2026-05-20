import { closeSync, fsyncSync, mkdirSync, openSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { performance } from "node:perf_hooks";

import { SystemClock, type Clock } from "../../util/clock.js";
import { serializeJsonValue, type JsonValue } from "../../util/json-value.js";

export type TurnTraceEventName =
  | "perception.started"
  | "perception.completed"
  | "perception.classifier.degraded"
  | "working_memory.degraded"
  | "recency.completed"
  | "participant_scan.skipped"
  | "retrieval.degraded"
  | "recall_expansion.completed"
  | "retrieval.started"
  | "retrieval.completed"
  | "evidence_ledger.completed"
  | "evidence_ledger.compaction.completed"
  | "shared_state.compile.skipped"
  | "shared_state.compile.transitioned"
  | "shared_state.compile.completed"
  | "shared_state.compile.degraded"
  | "shared_state.lifecycle.degraded"
  | "shared_state.reconcile.completed"
  | "shared_state.reconcile.skipped"
  | "shared_state.semantic_revision.degraded"
  | "semantic_revision.completed"
  | "semantic_revision.cache.completed"
  | "semantic_revision.degraded"
  | "deliberation.planner_ledger.completed"
  | "deliberation.contradiction_routing.completed"
  | "deliberation.contradiction_routing.transitioned"
  | "deliberation.path.completed"
  | "deliberation.path.transitioned"
  | "llm_call.started"
  | "llm_call.completed"
  | "deliberation.plan.completed"
  | "deliberation.planner.degraded"
  | "finalizer.completed"
  | "deliberation.plan_persistence.completed"
  | "deliberation.plan_persistence.skipped"
  | "post_generation.rejected"
  | "discourse_state.transitioned"
  | "discourse_state.rejected"
  | "citation_resolution.degraded"
  | "tool_call.started"
  | "tool_call.completed"
  | "extraction.commitments.degraded"
  | "extraction.commitments.rejected"
  | "extraction.commitments.transitioned"
  | "extraction.actions.rejected"
  | "action_persistence.dedup.skipped"
  | "action_persistence.dedup.degraded"
  | "action_state.transitioned"
  | "action_state.borg_self_performance.completed"
  | "action_state.archived"
  | "action_session_scope.expired"
  | "action_session_scope.rolled_over"
  | "action_inactivity_scan.completed"
  | "extraction.actions.completed"
  | "extraction.actions.degraded"
  | "action_duplicate_pressure.completed"
  | "shared_state.canonicalization.completed"
  | "extraction.goals.degraded"
  | "extraction.goals.completed"
  | "extraction.goals.rejected"
  | "extraction.goals.transitioned"
  | "extraction.goals.skipped"
  | "extraction.goals.dedup.degraded"
  | "frame_anomaly.degraded"
  | "frame_anomaly.degraded_fail_open"
  | "frame_anomaly.completed"
  | "frame_anomaly.transitioned"
  | "closure_loop.degraded"
  | "closure_loop.transitioned"
  | "open_question_resolution.degraded"
  | "open_question_resolution.started"
  | "open_question_resolution.transitioned"
  | "open_question_resolution.rejected"
  | "reflector.intent_update.rejected"
  | "reflector.intent_update.completed"
  | "offline_process.completed"
  | "review_resolver.started"
  | "review_resolver.decision.completed"
  | "review_resolver.completed"
  | "review_resolver.degraded"
  | "reflector.candidate.completed"
  | "review_queue.completed"
  | "semantic_extractor.started"
  | "semantic_extractor.degraded"
  | "semantic_insert.skipped"
  | "semantic_node.status.transitioned"
  | "maintenance_snapshot.completed"
  | "commitment_guard.shadow_observation"
  | "commitment_guard.enforce_suppression"
  | "commitment_guard.enforce_rewrite"
  | "commitment_check.completed"
  | "closure_response_guard.completed"
  | "internal_identifier_guard.completed"
  | "closure_pressure_audit.degraded"
  | "reflection.completed"
  | "session.completed"
  | "turn.rejected";

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

function traceValueType(value: unknown): string {
  if (value === null) {
    return "null";
  }

  if (Array.isArray(value)) {
    return "array";
  }

  return typeof value;
}

function uniqueTraceValueTypes(values: readonly unknown[]): string[] {
  return [...new Set(values.map((value) => traceValueType(value)))];
}

function summarizeTraceValueShapeInternal(value: unknown, depth: number): JsonValue {
  if (Array.isArray(value)) {
    return {
      type: "array",
      length: value.length,
      itemTypes: uniqueTraceValueTypes(value),
      sample:
        depth >= 3
          ? []
          : value.slice(0, 5).map((item) => summarizeTraceValueShapeInternal(item, depth + 1)),
    };
  }

  if (value !== null && typeof value === "object") {
    return {
      type: "object",
      fields: Object.entries(value as Record<string, unknown>).map(([name, fieldValue]) => ({
        name,
        type: traceValueType(fieldValue),
      })),
    };
  }

  return {
    type: traceValueType(value),
  };
}

export function summarizeTraceValueShape(value: unknown): JsonValue {
  return summarizeTraceValueShapeInternal(value, 0);
}

// Usage block builder that surfaces Anthropic prompt-cache token counts
// alongside fresh input/output tokens. Per Sprint 8d.6.4 cache fields
// (cache_creation_input_tokens / cache_read_input_tokens) live on the
// LLM result; callers should pass them through to the trace as separate
// keys, not summed into inputTokens. Cache reads cost ~0.1x fresh input
// and don't count against rate limits, so observability has to keep
// them in dedicated columns to be meaningful.
export type UsageTraceInput = {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
};

export function buildUsageTraceBlock(input: UsageTraceInput): {
  inputTokens: number;
  outputTokens: number;
  cacheCreationInputTokens?: number;
  cacheReadInputTokens?: number;
} {
  return {
    inputTokens: input.input_tokens,
    outputTokens: input.output_tokens,
    ...(input.cache_creation_input_tokens === undefined
      ? {}
      : { cacheCreationInputTokens: input.cache_creation_input_tokens }),
    ...(input.cache_read_input_tokens === undefined
      ? {}
      : { cacheReadInputTokens: input.cache_read_input_tokens }),
  };
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
