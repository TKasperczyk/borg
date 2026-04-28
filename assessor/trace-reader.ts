import { existsSync, readFileSync } from "node:fs";

import type { TracePhase, TraceRecord } from "./types.js";

export const TRACE_PHASES: readonly TracePhase[] = [
  "perception",
  "executive_focus",
  "retrieval",
  "deliberation",
  "action",
  "reflection",
  "ingestion",
  "other",
] as const;

const COLLAPSED_KEYS = new Set(["prompt", "response"]);
const MAX_VALUE_CHARS = 360;
const MAX_EVENTS_PER_PHASE = 12;

export type TraceReadOptions = {
  strict?: boolean;
};

export type TraceReadResult = {
  records: TraceRecord[];
  warnings: string[];
};

function isTraceRecord(value: unknown): value is TraceRecord {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof (value as TraceRecord).ts === "number" &&
    Number.isFinite((value as TraceRecord).ts) &&
    typeof (value as TraceRecord).turnId === "string" &&
    typeof (value as TraceRecord).event === "string"
  );
}

function compactJson(value: unknown): string {
  if (typeof value === "string") {
    return JSON.stringify(
      value.length > MAX_VALUE_CHARS ? `${value.slice(0, MAX_VALUE_CHARS)}...` : value,
    );
  }

  const serialized = JSON.stringify(value);

  if (serialized === undefined) {
    return "undefined";
  }

  return serialized.length > MAX_VALUE_CHARS
    ? `${serialized.slice(0, MAX_VALUE_CHARS)}...`
    : serialized;
}

export function readTraceFile(path: string, options: TraceReadOptions = {}): TraceReadResult {
  if (!existsSync(path)) {
    return {
      records: [],
      warnings: [],
    };
  }

  const raw = readFileSync(path, "utf8");
  const records: TraceRecord[] = [];
  const warnings: string[] = [];

  for (const [index, line] of raw.split(/\r?\n/).entries()) {
    if (line.trim().length === 0) {
      continue;
    }

    let parsed: unknown;

    try {
      parsed = JSON.parse(line);
    } catch (error) {
      const warning = `Invalid JSON on trace line ${index + 1}: ${
        error instanceof Error ? error.message : String(error)
      }`;

      if (options.strict === true) {
        throw new Error(warning);
      }

      warnings.push(warning);
      continue;
    }

    if (!isTraceRecord(parsed)) {
      const warning = `Invalid trace record on line ${index + 1}`;

      if (options.strict === true) {
        throw new Error(warning);
      }

      warnings.push(warning);
      continue;
    }

    records.push(parsed);
  }

  return {
    records,
    warnings,
  };
}

export function readTraceEvents(path: string, options: TraceReadOptions = {}): TraceRecord[] {
  const { records } = readTraceFile(path, options);

  return records;
}

export function phaseForTraceEvent(event: string): TracePhase {
  if (event === "recency_compiled" || event.startsWith("perception_")) {
    return "perception";
  }

  if (event.startsWith("executive_")) {
    return "executive_focus";
  }

  if (event.startsWith("retrieval_") || event === "citation_unresolved") {
    return "retrieval";
  }

  if (
    event.startsWith("llm_call_") ||
    event.startsWith("plan_") ||
    event === "s2_planner_exhausted" ||
    event === "path_selected"
  ) {
    return "deliberation";
  }

  if (event.startsWith("tool_call_") || event === "commitment_check") {
    return "action";
  }

  if (event === "reflection_emitted") {
    return "reflection";
  }

  if (event.startsWith("ingestion_") || event.startsWith("stream_ingestion_")) {
    return "ingestion";
  }

  return "other";
}

export function groupTraceByTurn(records: readonly TraceRecord[]): Map<string, TraceRecord[]> {
  const grouped = new Map<string, TraceRecord[]>();

  for (const record of records) {
    const current = grouped.get(record.turnId) ?? [];
    current.push(record);
    grouped.set(record.turnId, current);
  }

  return grouped;
}

export function groupTraceByPhase(records: readonly TraceRecord[]): Map<TracePhase, TraceRecord[]> {
  const grouped = new Map<TracePhase, TraceRecord[]>();

  for (const phase of TRACE_PHASES) {
    grouped.set(phase, []);
  }

  for (const record of records) {
    grouped.get(phaseForTraceEvent(record.event))?.push(record);
  }

  return grouped;
}

export function latestTurnId(records: readonly TraceRecord[]): string | null {
  return records[records.length - 1]?.turnId ?? null;
}

function formatEvent(record: TraceRecord): string {
  const parts: string[] = [];

  for (const [key, value] of Object.entries(record)) {
    if (key === "ts" || key === "turnId" || key === "event") {
      continue;
    }

    if (COLLAPSED_KEYS.has(key)) {
      parts.push(`${key}=[collapsed]`);
      continue;
    }

    parts.push(`${key}=${compactJson(value)}`);
  }

  return `${record.event}${parts.length === 0 ? "" : ` ${parts.join(" ")}`}`;
}

export function summarizeTraceTurn(
  records: readonly TraceRecord[],
  options: { phase?: TracePhase } = {},
): string {
  if (records.length === 0) {
    return "No trace events found for this turn.";
  }

  const grouped = groupTraceByPhase(records);
  const phases = options.phase === undefined ? TRACE_PHASES : [options.phase];
  const lines: string[] = [];

  for (const phase of phases) {
    const phaseRecords = grouped.get(phase) ?? [];

    if (phaseRecords.length === 0) {
      if (options.phase !== undefined) {
        lines.push(`${phase}: no events`);
      }
      continue;
    }

    lines.push(`${phase}: ${phaseRecords.length} event${phaseRecords.length === 1 ? "" : "s"}`);

    for (const record of phaseRecords.slice(0, MAX_EVENTS_PER_PHASE)) {
      lines.push(`- ${formatEvent(record)}`);
    }

    if (phaseRecords.length > MAX_EVENTS_PER_PHASE) {
      lines.push(`- ... ${phaseRecords.length - MAX_EVENTS_PER_PHASE} more event(s)`);
    }
  }

  return lines.join("\n");
}

export function summarizeTraceFile(
  path: string,
  turnId: string,
  options: { phase?: TracePhase } = {},
): string {
  const { records, warnings } = readTraceFile(path);
  const summary = summarizeTraceTurn(
    records.filter((record) => record.turnId === turnId),
    options,
  );

  if (warnings.length === 0) {
    return summary;
  }

  return [
    `trace warnings: ${warnings.length}`,
    ...warnings.map((warning) => `- ${warning}`),
    summary,
  ].join("\n");
}
