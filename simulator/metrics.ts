import { closeSync, fsyncSync, mkdirSync, openSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import type { Borg } from "../src/index.js";
import { DEFAULT_SESSION_ID } from "../src/index.js";
import { readTraceEvents } from "../assessor/trace-reader.js";
import type { TraceRecord } from "../assessor/types.js";

import type { MetricsRow } from "./types.js";

const LARGE_COUNT_LIMIT = 1_000_000;

type GoalTreeNodeLike = {
  children?: GoalTreeNodeLike[];
};

export type MetricsCaptureOptions = {
  tracePath?: string;
};

function appendJsonlLine(filePath: string, line: string): void {
  mkdirSync(dirname(filePath), { recursive: true });

  let fileDescriptor: number | undefined;

  try {
    fileDescriptor = openSync(filePath, "a");
    writeFileSync(fileDescriptor, line);
    fsyncSync(fileDescriptor);
  } finally {
    if (fileDescriptor !== undefined) {
      closeSync(fileDescriptor);
    }
  }
}

function flattenGoalCount(nodes: readonly GoalTreeNodeLike[]): number {
  let count = 0;
  const stack = [...nodes];

  while (stack.length > 0) {
    const next = stack.shift();

    if (next === undefined) {
      continue;
    }

    count += 1;
    stack.push(...(next.children ?? []));
  }

  return count;
}

function latencyBetween(
  records: readonly TraceRecord[],
  startEvent: string,
  endEvent: string,
): number | null {
  const start = records.find((record) => record.event === startEvent);
  const end = [...records].reverse().find((record) => record.event === endEvent);

  if (start === undefined || end === undefined) {
    return null;
  }

  // Tracer records both `ts` (logical clock, can be a ManualClock that
  // shares values across all events in a turn) and `wallMs`
  // (performance.now monotonic real time). Latency is meaningful only
  // off real time; fall back to ts only if wallMs is missing for some
  // reason (older records).
  const startWall = typeof start.wallMs === "number" ? start.wallMs : null;
  const endWall = typeof end.wallMs === "number" ? end.wallMs : null;

  if (startWall !== null && endWall !== null && endWall >= startWall) {
    return Math.round(endWall - startWall);
  }

  if (end.ts < start.ts) {
    return null;
  }

  return end.ts - start.ts;
}

function usageForTurn(records: readonly TraceRecord[]): {
  inputTokens: number;
  outputTokens: number;
} {
  let inputTokens = 0;
  let outputTokens = 0;

  for (const record of records) {
    if (record.event !== "llm_call_response") {
      continue;
    }

    const usage = record.usage;

    if (usage === null || typeof usage !== "object" || Array.isArray(usage)) {
      continue;
    }

    const input = (usage as { inputTokens?: unknown }).inputTokens;
    const output = (usage as { outputTokens?: unknown }).outputTokens;

    inputTokens += typeof input === "number" && Number.isFinite(input) ? input : 0;
    outputTokens += typeof output === "number" && Number.isFinite(output) ? output : 0;
  }

  return { inputTokens, outputTokens };
}

export class MetricsCapture {
  private readonly filepath: string;
  private readonly tracePath?: string;

  constructor(filepath: string, options: MetricsCaptureOptions = {}) {
    this.filepath = filepath;
    this.tracePath = options.tracePath;
  }

  async capture(borg: Borg, turnId: string, turnCounter: number): Promise<MetricsRow> {
    const traceRecords =
      this.tracePath === undefined
        ? []
        : readTraceEvents(this.tracePath).filter((record) => record.turnId === turnId);
    const usage = usageForTurn(traceRecords);
    const mood = borg.mood.current(DEFAULT_SESSION_ID);
    const episodeResult = await borg.episodic.list({ limit: LARGE_COUNT_LIMIT });
    const semanticNodes = await borg.semantic.nodes.list({ limit: LARGE_COUNT_LIMIT });
    const semanticEdges = borg.semantic.edges.list({ includeInvalid: true });
    const openQuestions = borg.self.openQuestions.list({
      status: "open",
      limit: LARGE_COUNT_LIMIT,
    });
    const activeGoals = borg.self.goals.list({ status: "active" });
    const row: MetricsRow = {
      ts: Date.now(),
      turn_counter: turnCounter,
      turnId,
      episode_count: episodeResult.items.length,
      semantic_node_count: semanticNodes.length,
      semantic_edge_count: semanticEdges.length,
      open_question_count: openQuestions.length,
      active_goal_count: flattenGoalCount(activeGoals),
      mood_valence: mood.valence,
      mood_arousal: mood.arousal,
      retrieval_latency_ms: latencyBetween(
        traceRecords,
        "retrieval_started",
        "retrieval_completed",
      ),
      deliberation_latency_ms: latencyBetween(
        traceRecords,
        "llm_call_started",
        "llm_call_response",
      ),
      borg_input_tokens: usage.inputTokens,
      borg_output_tokens: usage.outputTokens,
    };

    appendJsonlLine(this.filepath, `${JSON.stringify(row)}\n`);
    return row;
  }

  close(): void {
    // Metrics rows are fsynced on each append.
  }
}
