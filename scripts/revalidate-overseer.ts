import { existsSync, lstatSync, readFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  validateOverseerVerdict,
  type FindingCarryoverCache,
  type OverseerAuditContext,
} from "../simulator/overseer.ts";
import type { OverseerVerdict, RawOverseerVerdict } from "../simulator/types.ts";

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
      "Usage: pnpm exec tsx scripts/revalidate-overseer.ts <simulator-run-dir|report.md|overseer-audit.jsonl>",
      "",
      "Revalidation requires the persisted overseer audit JSONL.",
    ].join("\n"),
  );
  process.exit(1);
}

function parseArgs(argv: readonly string[]): { inputPath: string } {
  const inputPath = argv[2];

  if (inputPath === undefined || inputPath.length === 0 || argv.length > 3) {
    usage();
  }

  return { inputPath };
}

function auditPathForReport(reportPath: string): string {
  if (reportPath.endsWith("-report.md")) {
    return `${reportPath.slice(0, -"-report.md".length)}-overseer-audit.jsonl`;
  }

  return join(dirname(reportPath), `${basename(reportPath, ".md")}-overseer-audit.jsonl`);
}

function resolveAuditPath(inputPath: string): string {
  const absolute = resolve(inputPath);

  if (!existsSync(absolute)) {
    throw new Error(`input path does not exist: ${absolute}`);
  }

  if (lstatSync(absolute).isDirectory()) {
    const directoryName = basename(absolute);
    const candidates = [
      join(absolute, `${directoryName}-overseer-audit.jsonl`),
      `${absolute}-overseer-audit.jsonl`,
      join(absolute, "overseer-audit.jsonl"),
    ];
    const auditPath = candidates.find((candidate) => existsSync(candidate));

    if (auditPath !== undefined) {
      return auditPath;
    }

    throw new Error(`could not infer overseer audit JSONL for ${absolute}`);
  }

  if (absolute.endsWith(".jsonl")) {
    return absolute;
  }

  if (absolute.endsWith(".md")) {
    const auditPath = auditPathForReport(absolute);

    if (existsSync(auditPath)) {
      return auditPath;
    }

    throw new Error(`overseer audit JSONL does not exist: ${auditPath}`);
  }

  throw new Error(`unsupported input path: ${absolute}`);
}

function readPersistedAuditRecords(path: string): PersistedAuditRecord[] {
  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as PersistedAuditRecord);
}

function persistedValidationSubset(record: PersistedAuditRecord): {
  status: string;
  findings: unknown[];
  rejected_findings: unknown[];
} | null {
  if (record.validated_verdict === undefined) {
    return null;
  }

  return {
    status: record.validated_verdict.status,
    findings: record.validated_verdict.findings,
    rejected_findings: record.validated_verdict.rejected_findings,
  };
}

function validationMatches(
  record: PersistedAuditRecord,
  fresh: ReturnType<typeof validateOverseerVerdict>,
): boolean {
  const persisted = persistedValidationSubset(record);

  if (persisted === null) {
    return false;
  }

  return JSON.stringify(persisted) === JSON.stringify(fresh);
}

function renderStatusTable(records: readonly PersistedAuditRecord[]): void {
  const carryoverCache: FindingCarryoverCache = new Map();
  const orderedRecords = [...records].sort((left, right) => left.turn_counter - right.turn_counter);

  console.log("| Turn | Original status | Persisted validated | Fresh revalidation | Notes |");
  console.log("| ---: | --- | --- | --- | --- |");

  for (const record of orderedRecords) {
    const fresh = validateOverseerVerdict(record.raw_verdict, record.audit_context, carryoverCache);
    const persistedStatus = record.validated_verdict?.status ?? "n/a";
    const demotedCount = fresh.findings.filter(
      (finding) => finding.carryover_demoted === true,
    ).length;
    const notes = [
      validationMatches(record, fresh) ? "" : "drift",
      demotedCount > 0 ? `carryover_demoted=${demotedCount}` : "",
    ]
      .filter((note) => note.length > 0)
      .join(", ");

    console.log(
      `| ${record.turn_counter} | ${record.raw_verdict.status} | ${persistedStatus} | ${fresh.status} | ${notes} |`,
    );
  }
}

async function main(): Promise<void> {
  const { inputPath } = parseArgs(process.argv);
  const auditPath = resolveAuditPath(inputPath);
  const records = readPersistedAuditRecords(auditPath);

  console.log("# Overseer Revalidation");
  console.log(`Audit JSONL: ${auditPath}`);
  console.log(`Audit records: ${records.length}`);
  console.log("");
  renderStatusTable(records);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
