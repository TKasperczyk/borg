// Trace CLI commands for inspecting opt-in turn tracer JSONL files.
import { readFileSync } from "node:fs";

import type { CAC } from "cac";

import { createAnsi } from "../helpers/ansi.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

type TraceRecord = {
  ts: number;
  turnId: string;
  event: string;
  [key: string]: unknown;
};

const PHASES = [
  "perception",
  "retrieval",
  "deliberation",
  "tools",
  "commitments",
  "reflection",
  "other",
] as const;

type Phase = (typeof PHASES)[number];

const COLLAPSED_KEYS = new Set(["prompt", "response"]);

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

function readTrace(path: string): TraceRecord[] {
  const raw = readFileSync(path, "utf8");
  const records: TraceRecord[] = [];

  for (const [index, line] of raw.split(/\r?\n/).entries()) {
    if (line.trim().length === 0) {
      continue;
    }

    let parsed: unknown;

    try {
      parsed = JSON.parse(line);
    } catch (error) {
      throw new CliError(
        `Invalid JSON on trace line ${index + 1}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }

    if (!isTraceRecord(parsed)) {
      throw new CliError(`Invalid trace record on line ${index + 1}`);
    }

    records.push(parsed);
  }

  return records;
}

function phaseFor(event: string): Phase {
  if (event === "recency_compiled" || event.startsWith("perception_")) {
    return "perception";
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

  if (event.startsWith("tool_call_")) {
    return "tools";
  }

  if (event === "commitment_check") {
    return "commitments";
  }

  if (event === "reflection_emitted") {
    return "reflection";
  }

  return "other";
}

function stringifyCompact(value: unknown): string {
  if (typeof value === "string") {
    return JSON.stringify(value);
  }

  return JSON.stringify(value);
}

function formatEventDetails(record: TraceRecord, full: boolean): string {
  const parts: string[] = [];

  for (const [key, value] of Object.entries(record)) {
    if (key === "ts" || key === "turnId" || key === "event") {
      continue;
    }

    if (!full && COLLAPSED_KEYS.has(key)) {
      parts.push(`${key}="[collapsed; use --full]"`);
      continue;
    }

    parts.push(`${key}=${stringifyCompact(value)}`);
  }

  return parts.join(" ");
}

function groupByPhase(records: readonly TraceRecord[]): Map<Phase, TraceRecord[]> {
  const grouped = new Map<Phase, TraceRecord[]>();

  for (const phase of PHASES) {
    grouped.set(phase, []);
  }

  for (const record of records) {
    grouped.get(phaseFor(record.event))?.push(record);
  }

  return grouped;
}

function latestTurnId(records: readonly TraceRecord[]): string {
  const latest = records[records.length - 1];

  if (latest === undefined) {
    throw new CliError("Trace file has no events");
  }

  return latest.turnId;
}

export function registerTraceCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout } = deps;
  const ansi = createAnsi(stdout);

  cli
    .command("trace <action> [path]", "Inspect turn trace JSONL files")
    .option("--turn <id>", "Turn id to inspect")
    .option("--full", "Expand full prompt and response payloads", {
      default: false,
    })
    .action((action: string, path: string | undefined, commandOptions: CommandOptions) => {
      if (action !== "inspect") {
        throw new CliError(`Unknown trace action: ${action}`);
      }

      if (path === undefined || path.trim().length === 0) {
        throw new CliError("trace inspect requires a path");
      }

      const records = readTrace(path);
      const requestedTurn =
        typeof commandOptions.turn === "string" ? commandOptions.turn.trim() : "";
      const turnId = requestedTurn.length > 0 ? requestedTurn : latestTurnId(records);
      const full = commandOptions.full === true;
      const turnRecords = records.filter((record) => record.turnId === turnId);

      if (turnRecords.length === 0) {
        throw new CliError(`No events found for turn ${turnId}`);
      }

      writeLine(stdout, ansi.strong(`turn ${turnId}`));
      if (requestedTurn.length === 0) {
        const turnCount = new Set(records.map((record) => record.turnId)).size;
        if (turnCount > 1) {
          writeLine(
            stdout,
            ansi.dim(`showing latest turn from ${turnCount} turns; use --turn to filter`),
          );
        }
      }

      const grouped = groupByPhase(turnRecords);

      for (const phase of PHASES) {
        const phaseRecords = grouped.get(phase) ?? [];

        if (phaseRecords.length === 0) {
          continue;
        }

        writeLine(stdout, "");
        writeLine(stdout, ansi.accent(phase));

        for (const record of phaseRecords) {
          const time = new Date(record.ts).toISOString();
          const details = formatEventDetails(record, full);
          writeLine(
            stdout,
            `  ${ansi.dim(time)} ${ansi.green(record.event)}${details.length === 0 ? "" : ` ${details}`}`,
          );
        }
      }
    });
}
