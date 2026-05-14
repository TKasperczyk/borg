import { existsSync, lstatSync, readFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { readOverseerAuditTranscript } from "../assessor/borg-transport.ts";
import type { AuditTranscriptEntry } from "../assessor/borg-transport.ts";
import type { StreamEntry } from "../src/stream/index.ts";
import { validateOverseerVerdict, type OverseerAuditContext } from "../simulator/overseer.ts";
import type { OverseerVerdict, RawOverseerVerdict } from "../simulator/types.ts";

type LegacyCheckpoint = {
  turn: number;
  status: "healthy" | "concerning" | "failing";
  lines: string[];
};

type MetricsRowSubset = {
  turn_counter: number;
  turnId: string;
};

type CandidateVerdictLine = {
  checkpointTurn: number;
  line: string;
  assistantEntry: StreamEntry | null;
  quote: string | null;
  validation: "validated" | "rejected" | "skipped";
  reason: string;
};

type PersistedAuditRecord = {
  persisted_at?: number;
  turn_counter: number;
  audit_context: OverseerAuditContext;
  raw_verdict: RawOverseerVerdict;
  validated_verdict?: OverseerVerdict;
};

function usage(): never {
  console.error(
    [
      "Usage: pnpm exec tsx scripts/revalidate-overseer.ts <simulator-run-dir|report.md> [--data-dir <borg-data-dir>]",
      "",
      "Legacy report parsing is best-effort. v55+ exact replay should use the persisted overseer audit JSONL.",
    ].join("\n"),
  );
  process.exit(1);
}

function parseArgs(argv: readonly string[]): { inputPath: string; dataDir?: string } {
  const inputPath = argv[2];

  if (inputPath === undefined || inputPath.length === 0) {
    usage();
  }

  let dataDir: string | undefined;

  for (let index = 3; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--data-dir") {
      const value = argv[index + 1];

      if (value === undefined || value.length === 0) {
        usage();
      }

      dataDir = value;
      index += 1;
      continue;
    }

    usage();
  }

  return dataDir === undefined ? { inputPath } : { inputPath, dataDir };
}

function resolveReportPath(inputPath: string): string {
  const absolute = resolve(inputPath);

  if (!existsSync(absolute)) {
    throw new Error(`input path does not exist: ${absolute}`);
  }

  if (lstatSync(absolute).isDirectory()) {
    const directoryName = basename(absolute);
    const reportPath = join(absolute, `${directoryName}-report.md`);

    if (existsSync(reportPath)) {
      return reportPath;
    }

    const parentReportPath = `${absolute}-report.md`;

    if (existsSync(parentReportPath)) {
      return parentReportPath;
    }

    throw new Error(`could not infer report.md inside ${absolute}`);
  }

  return absolute;
}

function metricsPathForReport(reportPath: string): string {
  if (reportPath.endsWith("-report.md")) {
    return `${reportPath.slice(0, -"-report.md".length)}-metrics.jsonl`;
  }

  return join(dirname(reportPath), `${basename(reportPath, ".md")}-metrics.jsonl`);
}

function auditPathForReport(reportPath: string): string {
  if (reportPath.endsWith("-report.md")) {
    return `${reportPath.slice(0, -"-report.md".length)}-overseer-audit.jsonl`;
  }

  return join(dirname(reportPath), `${basename(reportPath, ".md")}-overseer-audit.jsonl`);
}

function readMetrics(path: string): MetricsRowSubset[] {
  if (!existsSync(path)) {
    return [];
  }

  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as MetricsRowSubset);
}

function parseRunId(reportText: string): string {
  const match = reportText.match(/^# Borg Simulator Run ([^\n]+)$/m);

  if (match?.[1] === undefined) {
    throw new Error("could not parse run id from report");
  }

  return match[1].trim();
}

function parsePersonas(reportText: string): string[] {
  const personasMatch = reportText.match(/^Personas: ([^\n]+)$/m);

  if (personasMatch?.[1] !== undefined) {
    return personasMatch[1].split(",").map((value) => value.trim());
  }

  const personaMatch = reportText.match(/^Persona: ([^\n]+)$/m);

  if (personaMatch?.[1] !== undefined) {
    return [personaMatch[1].trim()];
  }

  throw new Error("could not parse persona keys from report");
}

function inferDataDir(reportText: string): string {
  const runId = parseRunId(reportText);
  const personas = parsePersonas(reportText);

  return join("/tmp", `borg-assessor-${runId}-simulator-${personas.join("-")}`);
}

function parseCheckpoints(reportText: string): LegacyCheckpoint[] {
  const checkpoints: LegacyCheckpoint[] = [];
  let current: LegacyCheckpoint | null = null;

  for (const line of reportText.split(/\r?\n/)) {
    const checkpointMatch = line.match(/^- Turn (\d+): (healthy|concerning|failing)\b/);

    if (checkpointMatch?.[1] !== undefined && checkpointMatch[2] !== undefined) {
      current = {
        turn: Number(checkpointMatch[1]),
        status: checkpointMatch[2] as LegacyCheckpoint["status"],
        lines: [line],
      };
      checkpoints.push(current);
      continue;
    }

    if (current !== null && /^\s+- /.test(line)) {
      current.lines.push(line);
    }
  }

  return checkpoints;
}

function entryText(entry: StreamEntry): string {
  return typeof entry.content === "string" ? entry.content : JSON.stringify(entry.content);
}

function streamIdsInLine(line: string): string[] {
  return [...line.matchAll(/\bstrm_[a-z0-9]+\b/g)].map((match) => match[0]);
}

function bracketIndicesInLine(line: string): number[] {
  return [...line.matchAll(/\[(\d+)\]/g)].map((match) => Number(match[1]));
}

function quotedSpansInLine(line: string): string[] {
  return [
    ...[...line.matchAll(/(^|[\s([:;])'([^']{6,})'/g)].map((match) => match[2] ?? ""),
    ...[...line.matchAll(/"([^"]{6,})"/g)].map((match) => match[1] ?? ""),
  ].filter((value) => value.trim().length > 0);
}

function chooseQuote(line: string): string | null {
  const quoted = quotedSpansInLine(line)
    .map((value) => value.trim())
    .filter((value) => value.length > 0)
    .sort((left, right) => right.length - left.length);

  if (quoted.length > 0) {
    return quoted[0] ?? null;
  }

  const claimedMatch = line.match(/claimed\s+([^.;]+)[.;]/i);

  return claimedMatch?.[1]?.trim() ?? null;
}

function firstAssistantByStreamId(
  line: string,
  entriesById: ReadonlyMap<string, StreamEntry>,
): StreamEntry | null {
  for (const streamId of streamIdsInLine(line)) {
    const entry = entriesById.get(streamId);

    if (entry?.kind === "agent_msg") {
      return entry;
    }
  }

  return null;
}

function firstAssistantByTranscriptIndex(
  line: string,
  transcript: readonly AuditTranscriptEntry[],
): StreamEntry | null {
  for (const index of bracketIndicesInLine(line)) {
    const entry = transcript[index]?.entry;

    if (entry?.kind === "agent_msg") {
      return entry;
    }
  }

  return null;
}

function resolveAssistantEntry(input: {
  line: string;
  transcript: readonly AuditTranscriptEntry[];
  entriesById: ReadonlyMap<string, StreamEntry>;
}): StreamEntry | null {
  const lower = input.line.toLocaleLowerCase();
  const issueIndex = Math.min(
    ...[lower.indexOf("contradicted"), lower.indexOf("unsupported")].filter((index) => index >= 0),
  );

  if (Number.isFinite(issueIndex)) {
    const issueLine = input.line.slice(issueIndex);
    const issueEntry =
      firstAssistantByStreamId(issueLine, input.entriesById) ??
      firstAssistantByTranscriptIndex(issueLine, input.transcript);

    if (issueEntry !== null) {
      return issueEntry;
    }
  }

  return (
    firstAssistantByStreamId(input.line, input.entriesById) ??
    firstAssistantByTranscriptIndex(input.line, input.transcript)
  );
}

function temporalDirection(line: string): "before" | "after" | null {
  const lower = line.toLocaleLowerCase();

  if (lower.includes("before") || lower.includes("had not yet") || lower.includes("hadn't")) {
    return "before";
  }

  if (lower.includes("after")) {
    return "after";
  }

  return null;
}

function evidenceTimestamps(
  line: string,
  assistantEntry: StreamEntry,
  entriesById: ReadonlyMap<string, StreamEntry>,
): number[] {
  const evidence = streamIdsInLine(line)
    .filter((streamId) => streamId !== assistantEntry.id)
    .map((streamId) => entriesById.get(streamId)?.timestamp)
    .filter((timestamp): timestamp is number => timestamp !== undefined);

  return [...new Set(evidence)];
}

function isSuspiciousVerdictLine(line: string): boolean {
  const lower = line.toLocaleLowerCase();

  return (
    lower.includes("contradicted") ||
    lower.includes("unsupported") ||
    lower.includes("false-memory") ||
    lower.includes("false memory") ||
    lower.includes("before") ||
    lower.includes("after")
  );
}

function validateCandidate(input: {
  checkpointTurn: number;
  line: string;
  transcript: readonly AuditTranscriptEntry[];
  entriesById: ReadonlyMap<string, StreamEntry>;
}): CandidateVerdictLine | null {
  if (!isSuspiciousVerdictLine(input.line)) {
    return null;
  }

  const assistantEntry = resolveAssistantEntry(input);
  const quote = chooseQuote(input.line);
  const direction = temporalDirection(input.line);

  if (assistantEntry === null) {
    return {
      checkpointTurn: input.checkpointTurn,
      line: input.line,
      assistantEntry: null,
      quote,
      validation: "skipped",
      reason: "could not resolve an assistant emitted stream entry",
    };
  }

  if (quote !== null && !entryText(assistantEntry).includes(quote)) {
    return {
      checkpointTurn: input.checkpointTurn,
      line: input.line,
      assistantEntry,
      quote,
      validation: "rejected",
      reason: `quoted span is not in assistant emitted text ${assistantEntry.id}`,
    };
  }

  if (direction !== null) {
    const evidenceTs = evidenceTimestamps(input.line, assistantEntry, input.entriesById);

    if (evidenceTs.length === 0) {
      return {
        checkpointTurn: input.checkpointTurn,
        line: input.line,
        assistantEntry,
        quote,
        validation: "skipped",
        reason: "temporal claim has no resolvable evidence timestamps",
      };
    }

    if (direction === "before") {
      const earliestEvidenceTs = Math.min(...evidenceTs);

      if (assistantEntry.timestamp >= earliestEvidenceTs) {
        return {
          checkpointTurn: input.checkpointTurn,
          line: input.line,
          assistantEntry,
          quote,
          validation: "rejected",
          reason: `assistant ts ${assistantEntry.timestamp} is not before evidence ts ${earliestEvidenceTs}`,
        };
      }
    }

    if (direction === "after") {
      const latestEvidenceTs = Math.max(...evidenceTs);

      if (assistantEntry.timestamp <= latestEvidenceTs) {
        return {
          checkpointTurn: input.checkpointTurn,
          line: input.line,
          assistantEntry,
          quote,
          validation: "rejected",
          reason: `assistant ts ${assistantEntry.timestamp} is not after evidence ts ${latestEvidenceTs}`,
        };
      }
    }
  }

  if (quote === null) {
    return {
      checkpointTurn: input.checkpointTurn,
      line: input.line,
      assistantEntry,
      quote,
      validation: "skipped",
      reason: "legacy parser found no quoted emitted span or timestamp-checkable claim",
    };
  }

  return {
    checkpointTurn: input.checkpointTurn,
    line: input.line,
    assistantEntry,
    quote,
    validation: "validated",
    reason: "legacy report claim matches emitted text/timestamps available to the script",
  };
}

function legacyValidatedStatus(
  checkpoint: LegacyCheckpoint,
  candidates: readonly CandidateVerdictLine[],
): LegacyCheckpoint["status"] {
  if (checkpoint.status === "healthy" || candidates.length === 0) {
    return checkpoint.status;
  }

  const actionable = candidates.filter((candidate) => candidate.validation !== "skipped");

  if (
    actionable.length > 0 &&
    actionable.every((candidate) => candidate.validation === "rejected")
  ) {
    return "healthy";
  }

  if (
    checkpoint.status === "failing" &&
    actionable.some((candidate) => candidate.validation === "rejected")
  ) {
    return "concerning";
  }

  return checkpoint.status;
}

function renderCandidate(candidate: CandidateVerdictLine): string {
  const assistant = candidate.assistantEntry;
  const assistantSummary =
    assistant === null ? "assistant=n/a" : `assistant=${assistant.id} ts=${assistant.timestamp}`;
  const quoteSummary =
    candidate.quote === null ? "quote=n/a" : `quote=${JSON.stringify(candidate.quote)}`;

  return `  - ${candidate.validation}: ${candidate.reason}; ${assistantSummary}; ${quoteSummary}`;
}

function readPersistedAuditRecords(path: string): PersistedAuditRecord[] {
  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as PersistedAuditRecord);
}

function renderStatusTable(records: readonly PersistedAuditRecord[]): void {
  console.log("| Turn | Original status | Persisted validated | Fresh revalidation |");
  console.log("| ---: | --- | --- | --- |");

  for (const record of records) {
    const fresh = validateOverseerVerdict(record.raw_verdict, record.audit_context);
    const persistedStatus = record.validated_verdict?.status ?? "n/a";
    const drift = persistedStatus === fresh.status ? "" : " drift";

    console.log(
      `| ${record.turn_counter} | ${record.raw_verdict.status} | ${persistedStatus} | ${fresh.status}${drift} |`,
    );
  }
}

async function main(): Promise<void> {
  const { inputPath, dataDir: dataDirArg } = parseArgs(process.argv);
  const reportPath = resolveReportPath(inputPath);
  const reportText = readFileSync(reportPath, "utf8");
  const metricsPath = metricsPathForReport(reportPath);
  const auditPath = auditPathForReport(reportPath);

  console.log("# Overseer Revalidation");
  console.log(`Report: ${reportPath}`);

  if (existsSync(auditPath)) {
    const records = readPersistedAuditRecords(auditPath);

    console.log(`Audit JSONL: ${auditPath}`);
    console.log(`Audit records: ${records.length}`);
    console.log("");
    renderStatusTable(records);
    return;
  }

  const dataDir = dataDirArg ?? inferDataDir(reportText);

  if (!existsSync(dataDir)) {
    throw new Error(`data dir does not exist: ${dataDir}`);
  }

  const metricsRows = readMetrics(metricsPath);
  const transcript = await readOverseerAuditTranscript(dataDir);
  const entriesById = new Map(transcript.map(({ entry }) => [entry.id, entry] as const));
  const checkpoints = parseCheckpoints(reportText);
  const turnIdToCounter = new Map(metricsRows.map((row) => [row.turnId, row.turn_counter]));

  console.log("Audit JSONL: missing; using legacy markdown fallback.");
  console.log(`Metrics: ${existsSync(metricsPath) ? metricsPath : "missing"}`);
  console.log(`Data dir: ${dataDir}`);
  console.log(`Transcript entries: ${transcript.length}`);
  console.log(`Metrics rows: ${metricsRows.length}`);
  console.log("");

  for (const checkpoint of checkpoints) {
    const candidates = checkpoint.lines
      .map((line) =>
        validateCandidate({
          checkpointTurn: checkpoint.turn,
          line,
          transcript,
          entriesById,
        }),
      )
      .filter((candidate): candidate is CandidateVerdictLine => candidate !== null);
    const validated = legacyValidatedStatus(checkpoint, candidates);

    if (checkpoint.status === "healthy") {
      continue;
    }

    console.log(`- Turn ${checkpoint.turn}: original=${checkpoint.status} validated=${validated}`);

    if (candidates.length === 0) {
      console.log("  - no temporal/J-contradicted candidates found by legacy parser");
    } else {
      for (const candidate of candidates) {
        console.log(renderCandidate(candidate));
      }
    }

    for (const candidate of candidates) {
      const turnCounter =
        candidate.assistantEntry?.turn_id === undefined
          ? null
          : (turnIdToCounter.get(candidate.assistantEntry.turn_id) ?? null);

      if (turnCounter !== null) {
        console.log(`  - assistant_metrics_turn=${turnCounter}`);
      }
    }
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
