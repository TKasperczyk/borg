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
  | "recall_expansion_clipped"
  | "retrieval_started"
  | "retrieval_completed"
  | "evidence_ledger_built"
  | "evidence_ledger_compacted"
  | "decision_artifact_compile_skipped"
  | "decision_artifact_compile_unblocked"
  | "decision_artifact_compile_completed"
  | "decision_artifact_compile_over_budget"
  | "decision_artifact_lifecycle_unable_to_cap"
  | "decision_artifact_reconciliation_completed"
  | "planner_compact_ledger_built"
  | "contradiction_routing_classified"
  | "contradiction_routing_cooldown_demoted"
  | "path_selected"
  | "s2_routing_forced_by_contradiction"
  | "llm_call_started"
  | "llm_call_response"
  | "plan_extraction"
  | "s2_planner_exhausted"
  | "finalizer_emitted"
  | "plan_persisted"
  | "plan_persistence_skipped"
  | "generation_suppressed"
  | "discourse_state_set"
  | "discourse_state_cleared"
  | "discourse_state_hard_cap"
  | "citation_unresolved"
  | "tool_call_dispatched"
  | "tool_call_completed"
  | "commitment_extractor_degraded"
  | "action_state_extractor_degraded"
  | "goal_promotion_extractor_degraded"
  | "frame_anomaly_classifier_degraded"
  | "frame_anomaly_classified"
  | "frame_anomaly_quarantine_appended"
  | "frame_anomaly_degraded_fallback_match"
  | "frame_anomaly_degraded_fallback_normal"
  | "closure_loop_classifier_degraded"
  | "closure_loop_classifier_payload_normalized"
  | "open_question_resolution_degraded"
  | "open_question_resolution_attempt"
  | "open_question_merged"
  | "open_question_stale_dismissed"
  | "reflector_intent_update_suppressed"
  | "offline_process_completed"
  | "reflector_candidate_emitted"
  | "review_queue_decision"
  | "semantic_extractor_invoked"
  | "semantic_extractor_partial_failure"
  | "semantic_insert_skipped"
  | "maintenance_snapshot_finalized"
  | "commitment_check"
  | "closure_response_guard"
  | "internal_identifier_guard"
  | "closure_pressure_audit_inconsistent"
  | "reflection_emitted"
  | "turn_aborted";

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
