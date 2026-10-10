/**
 * Read-only legacy data-format inventory for one bank, or every immediate tenant
 * directory under a root. JSON on stdout (one document per tenant), diagnostics
 * on stderr. Never opens Borg, migrates, repairs, or touches bank lock files.
 *
 * Usage (Node >= 22.18, installed dependencies only):
 *   node --import tsx scripts/legacy-residue-report.ts --bank /data/banks/tenant-1
 *   node --import tsx scripts/legacy-residue-report.ts --all-tenants /data/banks
 *   ... --cache-dir /tmp/x/cache --repair-inputs /tmp/x/repair-inputs.json
 *
 * In a pod with a read-only app tree:
 *   mkdir -p /tmp/x/cache
 *   HOME=/tmp/x XDG_CACHE_HOME=/tmp/x/cache TMPDIR=/tmp/x TSX_DISABLE_CACHE=1 \
 *     /app/node_modules/.bin/tsx /app/scripts/legacy-residue-report.ts \
 *     --all-tenants /data/banks > /tmp/x/residue.jsonl
 * If /tmp is not writable, use a scratch directory under /data, outside banks.
 * SQLite main+WAL are copied to scratch and opened with DatabaseSync readOnly.
 * Source file signatures must stay stable during copying. LanceDB is checked
 * out at a fixed version. No cross-store snapshot is implied: use an idle bank
 * or filesystem snapshot for deletion evidence. Scratch copies are removed.
 * See docs/legacy-residue-report.md for null counts and optional repair inputs.
 */
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, realpathSync, statSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import {
  cacheDirectory,
  openEpisodeTable,
  openSqliteSnapshot,
  type EpisodeTable,
  type SqliteSnapshot,
} from "./legacy-residue/bank.js";
import { episodeChecks, fileChecks } from "./legacy-residue/file-checks.js";
import {
  outcomeChecks,
  repairChecks,
  repairInputsSchema,
  type TenantRepairInputs,
} from "./legacy-residue/repair-checks.js";
import {
  ReportChecks,
  ResidueReportError,
  safeFailure,
  type ResidueReport,
} from "./legacy-residue/report.js";
import { sqliteChecks } from "./legacy-residue/sqlite-checks.js";

const HELP = `Usage: legacy-residue-report.ts --bank <dir> | --all-tenants <root>
  --cache-dir <dir>       Scratch outside bank/app (else XDG_CACHE_HOME or OS temp)
  --repair-inputs <file>  JSON keyed by tenant, with audience and/or targetGoalIds
  --help                 Show this help
Counts only; no apply mode. One JSON document per tenant on stdout.
count:null is unavailable or explicitly skipped, never a clean zero.
Exit 1 for failed reads; detected residue and skipped input-dependent checks exit 0.
`;

export type LegacyReportOptions = {
  cacheDir?: string;
  repairInputs?: TenantRepairInputs;
  repairInputsSha256?: string;
  diagnostic?: (message: string) => void;
};

const SQLITE_IDS = [
  "L2",
  "L6.commitments",
  "L6.identity_events",
  "L7",
  "L7.unreverted",
  "L8",
  "L8.open",
  "L8.resolved",
  "L9",
  "L9.open",
  "L10",
  "L11",
  "L14",
  "S4.metadata",
  "S5",
  "S4.backlog_unknown_order",
  "S4.backlog_before_watermark",
  "S4.pending_pre_inbox_backlog",
  "R.migrate-audience-scoping",
  "R.repair-goal-target-at",
  "R.repair-goal-speaker-owner",
  "R.repair-goal-rollback-audit",
  "R.repair-rumination-scaffolding",
];

export async function generateLegacyResidueReport(
  bank: string,
  options: LegacyReportOptions = {},
): Promise<ResidueReport> {
  const directory = realpathSync(resolve(bank));
  const tenant = basename(directory);
  const diagnostic =
    options.diagnostic ??
    ((message: string) => process.stderr.write(`[legacy-residue] ${tenant}: ${message}\n`));
  const report = new ReportChecks(diagnostic);
  // Validate before any scratch write, even if borg.db is missing.
  const cacheDir = cacheDirectory(directory, options.cacheDir);
  let snapshot: SqliteSnapshot | undefined;
  let episodes: EpisodeTable | undefined;
  try {
    await report.run(
      "storage.sqlite",
      "Read-only SQLite snapshot opened",
      "Copy borg.db and existing borg.db-wal to external scratch, verify size/inode/device/mtime/ctime before and after, openReadOnlyDatabase(snapshot) using DatabaseSync({readOnly:true}); no migrations, bank SHM, journal, or locks.",
      () => {
        snapshot = openSqliteSnapshot(directory, cacheDir);
        return { count: 1 };
      },
    );
    if (snapshot !== undefined) {
      await sqliteChecks(snapshot.db, report);
      await repairChecks(snapshot, report, options.repairInputs);
    } else {
      for (const id of SQLITE_IDS)
        report.skip(
          id,
          `${id} requires readable SQLite`,
          "SQLite snapshot unavailable; see storage.sqlite.",
        );
    }
    await report.run(
      "storage.episodes",
      "Read-only episodes version opened",
      "Require existing lancedb/episodes.lance; connect, openTable('episodes'), table.version(), table.checkout(version). No create, schema evolution, indexes, or checkoutLatest.",
      async () => {
        episodes = await openEpisodeTable(directory);
        return { count: 1 };
      },
    );
    if (episodes !== undefined) await episodeChecks(episodes, report);
    else {
      for (const id of ["L5", "E2"])
        report.skip(
          id,
          `${id} requires readable episodes`,
          "Episodes table unavailable; see storage.episodes.",
        );
    }
    if (snapshot !== undefined && episodes !== undefined)
      await outcomeChecks(snapshot, episodes, report);
    else
      report.skip(
        "R.migrate-outcome-corpus",
        "Outcome-corpus candidate operations",
        "Both read-only SQLite and episodes handles are required; see storage checks.",
      );
    await fileChecks(directory, report);
    if (options.repairInputsSha256 !== undefined) {
      for (const check of report.checks) {
        if (check.id.startsWith("R."))
          check.query += `\nOperator input document SHA-256: ${options.repairInputsSha256}; use its entry keyed by tenant. IDs are not emitted.`;
      }
    }
    return { tenant, generated_at: new Date().toISOString(), checks: report.checks };
  } finally {
    try {
      episodes?.table.close();
    } finally {
      try {
        episodes?.connection.close();
      } finally {
        snapshot?.close();
      }
    }
  }
}

export async function main(
  args: string[] = process.argv.slice(2),
  output = {
    stdout: (text: string) => process.stdout.write(text),
    stderr: (text: string) => process.stderr.write(text),
  },
): Promise<0 | 1> {
  const { values } = parseArgs({
    args,
    strict: true,
    allowPositionals: false,
    options: {
      bank: { type: "string" },
      "all-tenants": { type: "string" },
      "cache-dir": { type: "string" },
      "repair-inputs": { type: "string" },
      help: { type: "boolean" },
    },
  });
  if (values.help) {
    output.stdout(HELP);
    return 0;
  }
  if ((values.bank === undefined) === (values["all-tenants"] === undefined))
    throw new ResidueReportError("Specify exactly one of --bank or --all-tenants");
  const root = realpathSync(resolve(values.bank ?? values["all-tenants"]!));
  let repairInputs: ReturnType<typeof repairInputsSchema.parse> = {};
  let repairInputsSha256: string | undefined;
  if (values["repair-inputs"] !== undefined) {
    const json = readFileSync(values["repair-inputs"], "utf8");
    const parsed = repairInputsSchema.safeParse(JSON.parse(json));
    if (!parsed.success)
      throw new ResidueReportError(
        "Invalid repair input document; expected tenant-keyed audience and/or targetGoalIds",
      );
    repairInputs = parsed.data;
    repairInputsSha256 = createHash("sha256").update(json).digest("hex");
  }
  const banks =
    values.bank !== undefined
      ? [root]
      : readdirSync(root, { withFileTypes: true })
          .filter((entry) => entry.isDirectory())
          .map((entry) => join(root, entry.name))
          .filter(
            (path) =>
              statSync(join(path, "borg.db"), { throwIfNoEntry: false })?.isFile() ||
              statSync(join(path, "lancedb"), { throwIfNoEntry: false })?.isDirectory(),
          )
          .sort();
  if (banks.length === 0) throw new ResidueReportError("No tenant bank directories found");
  // /data may be the only writable mount. A scratch sibling under the tenant
  // root is fine; validate against every actual bank before creating anything.
  let cacheDir = values["cache-dir"];
  for (const bank of banks) cacheDir = cacheDirectory(bank, cacheDir);
  let exitCode: 0 | 1 = 0;
  for (const bank of banks) {
    const tenant = basename(bank);
    output.stderr(`[legacy-residue] ${tenant}: reading bank\n`);
    try {
      const report = await generateLegacyResidueReport(bank, {
        cacheDir,
        repairInputs: repairInputs[tenant],
        repairInputsSha256,
        diagnostic: (message) => output.stderr(`[legacy-residue] ${tenant}: ${message}\n`),
      });
      output.stdout(`${JSON.stringify(report)}\n`);
      if (
        report.checks.some((check) => check.count === null && !check.query.startsWith("SKIPPED:"))
      )
        exitCode = 1;
    } catch (error) {
      exitCode = 1;
      const reason = safeFailure(error);
      output.stderr(`[legacy-residue] ${tenant}: ${reason}\n`);
      output.stdout(
        `${JSON.stringify({ tenant, generated_at: new Date().toISOString(), checks: [{ id: "bank", description: "Bank report unavailable", query: `UNAVAILABLE: ${reason}`, count: null }] })}\n`,
      );
    }
  }
  return exitCode;
}

if (
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  try {
    process.exitCode = await main();
  } catch (error) {
    process.stderr.write(`[legacy-residue] ${safeFailure(error)}\n`);
    process.exitCode = 1;
  }
}
