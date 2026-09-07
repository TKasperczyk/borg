import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { z } from "zod";
import {
  DEFAULT_TABLES,
  MeasurementError,
  openVectorBanks,
  outputDirectory,
  pairTables,
  readFamilies,
  readTable,
  resolveBankPaths,
  tableFingerprint,
  tableNameSchema,
} from "./similarity-distributions/bank.js";
import { measureRows } from "./similarity-distributions/statistics.js";
import { writeReports, type RunReport } from "./similarity-distributions/report.js";

const optionsSchema = z.object({
  bank: z.string().min(1),
  out: z.string().min(1),
  vectors: z.enum(["current", "prev"]),
  tables: z
    .array(tableNameSchema)
    .min(1)
    .refine((tables) => new Set(tables).size === tables.length, "Duplicate table names"),
  sample: z.coerce.number().int().positive().max(Number.MAX_SAFE_INTEGER),
  seed: z.string().min(1),
  cacheDir: z.string().min(1).optional(),
});
export type MeasurementOptions = z.infer<typeof optionsSchema>;

const HELP = `Usage: node --import tsx scripts/measure-similarity-distributions.ts --bank <tenant-copy> --out <dir> [options]

Read stored vectors only. Never calls an embedding gateway or opens Borg.
  --vectors current|prev   Model to measure (default current)
  --tables <csv>           Default: ${DEFAULT_TABLES.join(",")}
  --sample <n>             Unique random pairs/table and family stratum cap (default 10000)
  --seed <string>          Deterministic sample seed (default borg-similarity-v1)
  --bank <dir>             Quiescent copied tenant with lancedb and optional lancedb.prev-<N>
  --out <dir>              Report root outside the bank; one child directory per bank
  --cache-dir <dir>        Scratch/cache root (else XDG_CACHE_HOME, else OS temporary directory)

Run current and prev sequentially with the same bank, out, seed, tables and sample.
The second run adds percentile proposals to report.json and summary.md. Exact nearest
neighbors and duplicate counts require O(rows² * dimensions) time; --sample does not
limit that pass. Previous vectors are chosen by profile generation, else largest N.
`;

export function parseMeasurementArgs(args: string[]): MeasurementOptions | null {
  const { values } = parseArgs({
    args,
    allowPositionals: false,
    strict: true,
    options: {
      bank: { type: "string" },
      out: { type: "string" },
      vectors: { type: "string", default: "current" },
      tables: { type: "string", default: DEFAULT_TABLES.join(",") },
      sample: { type: "string", default: "10000" },
      seed: { type: "string", default: "borg-similarity-v1" },
      "cache-dir": { type: "string" },
      help: { type: "boolean", short: "h" },
    },
  });
  if (values.help) return null;
  return optionsSchema.parse({
    ...values,
    cacheDir: values["cache-dir"],
    tables: values.tables!.split(",").map((table) => table.trim()),
  });
}

export async function measureBank(
  options: MeasurementOptions,
  progress: (message: string) => void = (message) => process.stderr.write(`${message}\n`),
) {
  const paths = resolveBankPaths(options.bank, options.vectors);
  const directory = outputDirectory(paths.directory, options.out);
  const profile = options.vectors === "current" ? paths.profile : paths.previousProfile;
  const expectedDimensions = options.vectors === "current" ? 1024 : 4096;
  const assumedModel = options.vectors === "current" ? "scw/bge-m3" : "qwen3-embedding-8b";
  const run: RunReport = {
    version: 1,
    bank: paths.directory,
    vectors: options.vectors,
    vector_directory: options.vectors === "current" ? paths.current : paths.prev!,
    model: profile?.model ?? assumedModel,
    model_attribution:
      profile === null
        ? "Assumed migration model; verify against bank provenance (dimensions checked per table)"
        : "embedding-profile.json",
    expected_dimensions: profile?.dimensions ?? expectedDimensions,
    sample: options.sample,
    seed: options.seed,
    warnings: [...paths.warnings],
    tables: {},
  };
  progress(`Reading ${run.vectors}: ${run.vector_directory}`);
  for (const warning of run.warnings) progress(warning);
  const families = options.tables.includes("episodes")
    ? readFamilies(paths.directory, options.cacheDir)
    : undefined;
  const connections = await openVectorBanks(paths);
  try {
    for (const name of options.tables) {
      progress(`${name}: loading stored current/prev vectors and pairing by ID`);
      const current = await readTable(connections.current, name);
      const prev = connections.prev === null ? null : await readTable(connections.prev, name);
      const paired = pairTables(current, prev);
      const selected = options.vectors === "current" ? current : prev;
      if (selected === null) throw new MeasurementError("Previous vector bank disappeared");
      if (!current.present || (prev !== null && !prev.present))
        run.warnings.push(
          `${name}: table missing in ${!current.present ? "current" : "prev"}; paired corpus is empty.`,
        );
      if (selected.dimensions !== null && selected.dimensions !== run.expected_dimensions)
        run.warnings.push(
          `${name}: ${selected.dimensions} dimensions differs from model expectation ${run.expected_dimensions}.`,
        );
      progress(
        `${name}: ${paired.audit.common_count ?? paired.current.length} usable rows; excluded ${paired.audit.current_invalid.length}/${paired.audit.prev_invalid.length} invalid current/prev vectors`,
      );
      const tableFamilies = name === "episodes" ? families : undefined;
      const comparisonHash = createHash("sha256")
        .update(
          JSON.stringify({
            version: 1,
            current: tableFingerprint(current),
            prev: tableFingerprint(prev),
            paths: [paths.current, paths.prev],
            families:
              tableFamilies === undefined
                ? null
                : {
                    source: tableFamilies.source,
                    by_id: [...tableFamilies.byId],
                  },
          }),
        )
        .digest("hex");
      run.tables[name] = {
        present: selected.present,
        dimensions: selected.dimensions,
        comparison_sha256: comparisonHash,
        pairing: paired.audit,
        measurement: await measureRows({
          rows: paired[options.vectors],
          sample: options.sample,
          seed: `${options.seed}:${name}`,
          families: tableFamilies,
          progress: (message) => progress(`${options.vectors}/${name}: ${message}`),
        }),
      };
    }
  } finally {
    try {
      connections.current.close();
    } finally {
      connections.prev?.close();
    }
  }
  writeReports(directory, run);
  progress(`Wrote ${directory}/{${options.vectors}.json,report.json,summary.md}`);
  return { directory, run };
}

async function main(): Promise<void> {
  const options = parseMeasurementArgs(process.argv.slice(2));
  if (options === null) process.stdout.write(HELP);
  else await measureBank(options);
}

if (
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  try {
    await main();
  } catch (error) {
    process.stderr.write(
      `Similarity measurement failed: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    process.exitCode = 1;
  }
}
