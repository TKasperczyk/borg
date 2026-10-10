import { createReadStream, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { createGunzip } from "node:zlib";
import { fileSignature, type EpisodeTable } from "./bank.js";
import {
  counter,
  objectValue,
  parseStoredJson,
  ReportChecks,
  ResidueReportError,
} from "./report.js";

export async function episodeChecks(opened: EpisodeTable, report: ReportChecks): Promise<void> {
  const { table, version } = opened;
  const fields = new Set((await table.schema()).fields.map((field) => field.name));
  const projection = [
    "id",
    "audience_entity_id",
    "origin_audience_entity_ids",
    "episode_kind",
    "consolidation_version",
    "consolidation_embedding_input",
  ].filter((name) => fields.has(name));
  const query = `LanceDB episodes, checkout(${version}); query().select(${JSON.stringify(projection)}), iterate Arrow batches. All stored rows included, with no archived, visibility, current-version, or episode_stats filter.`;
  for (const id of ["L5", "E2"]) {
    const logic =
      id === "L5"
        ? "audience_entity_id is a nonempty string AND origin_audience_entity_ids is absent, null, empty string, JSON null, or an empty array (JSON text decoded first). Malformed/non-array origins make the count unavailable."
        : "episode_kind === 'consolidation_version' OR a non-null/nonempty consolidation_version column, if present; AND consolidation_embedding_input is absent, SQL null, or JSON text 'null'. Total is all episodes.";
    await report.run(
      id,
      id === "L5"
        ? "Audience-scoped episodes missing origin audiences, including archived"
        : "Consolidation episodes missing the recorded embedding input, including archived",
      `${query}\n${logic}`,
      async () => {
        if (!fields.has("id")) throw new ResidueReportError("Episodes table has no id column");
        const matches = counter();
        let total = 0;
        for await (const batch of table.query().select(projection)) {
          for (const raw of batch) {
            const row = raw as Record<string, unknown>;
            total += 1;
            if (id === "L5") {
              if (typeof row.audience_entity_id !== "string" || row.audience_entity_id.length === 0)
                continue;
              const stored = row.origin_audience_entity_ids;
              const origins =
                typeof stored === "string" && stored !== "" ? parseStoredJson(stored) : stored;
              if (
                origins == null ||
                origins === "" ||
                (Array.isArray(origins) && origins.length === 0)
              )
                matches.add(row.id);
              else if (
                !Array.isArray(origins) &&
                !(
                  typeof stored !== "string" &&
                  typeof origins === "object" &&
                  Symbol.iterator in origins &&
                  "length" in origins
                )
              )
                throw new ResidueReportError("Non-array episode origins; count unavailable");
              else if (!Array.isArray(origins) && (origins as { length: unknown }).length === 0)
                matches.add(row.id);
            } else {
              const consolidated =
                row.episode_kind === "consolidation_version" ||
                (row.consolidation_version != null && row.consolidation_version !== "");
              if (
                consolidated &&
                (row.consolidation_embedding_input == null ||
                  (typeof row.consolidation_embedding_input === "string" &&
                    row.consolidation_embedding_input.trim() === "null"))
              )
                matches.add(row.id);
            }
          }
        }
        return { ...matches.result, total };
      },
    );
  }
}

const CAPTURE_NAME = /^finalizer-contexts\.jsonl(?:\.rotated-\d{8}T\d{6}\.\d{3}Z|\.\d+)?(?:\.gz)?$/;
const MAX_CAPTURE_LINE_BYTES = 128 * 1024 * 1024;

function capturePaths(directory: string): string[] {
  if (!statSync(directory, { throwIfNoEntry: false })?.isDirectory()) return [];
  const names = readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && CAPTURE_NAME.test(entry.name))
    .map((entry) => entry.name);
  // Compression can temporarily leave both forms. Count the logical archive once.
  const available = new Set(names);
  return names
    .filter((name) => !name.endsWith(".gz") || !available.has(name.slice(0, -3)))
    .sort()
    .map((name) => join(directory, name));
}

async function* captureLines(path: string): AsyncGenerator<string> {
  const file = createReadStream(path);
  const input = path.endsWith(".gz") ? file.pipe(createGunzip()) : file;
  if (input !== file) file.on("error", (error) => input.destroy(error));
  let fragments: Buffer[] = [];
  let size = 0;
  try {
    for await (const chunk of input) {
      const bytes = chunk as Buffer;
      let start = 0;
      while (start < bytes.length) {
        const newline = bytes.indexOf(10, start);
        const end = newline === -1 ? bytes.length : newline;
        const fragment = bytes.subarray(start, end);
        size += fragment.length;
        if (size > MAX_CAPTURE_LINE_BYTES)
          throw new ResidueReportError("Capture line exceeds 128 MiB; schema counts unavailable");
        fragments.push(fragment);
        if (newline !== -1) {
          yield Buffer.concat(fragments, size).toString("utf8");
          fragments = [];
          size = 0;
        }
        start = end + 1;
      }
    }
    if (size > 0) yield Buffer.concat(fragments, size).toString("utf8");
  } finally {
    input.destroy();
    file.destroy();
  }
}

export async function fileChecks(bank: string, report: ReportChecks): Promise<void> {
  const journal = join(bank, ".embedding-migration.json");
  await report.run(
    "E3.exists",
    "Embedding migration journal exists",
    "stat(.embedding-migration.json); count = 1 for a regular file, else 0",
    () => ({ count: fileSignature(journal) === null ? 0 : 1 }),
  );
  await report.run(
    "E3",
    "Embedding migration journal has phase complete",
    "JSON.parse(readFile(.embedding-migration.json)).phase === 'complete'; count = 0 when absent; total = journal files present. Check file signature before/after reading.",
    () => {
      const before = fileSignature(journal);
      if (before === null) return { count: 0, total: 0 };
      const value = parseStoredJson(readFileSync(journal, "utf8"));
      if (!objectValue(value) || typeof value.phase !== "string")
        throw new ResidueReportError("Invalid embedding migration journal phase");
      if (fileSignature(journal) !== before)
        throw new ResidueReportError("Embedding migration journal changed during read");
      return { count: value.phase === "complete" ? 1 : 0, total: 1 };
    },
  );

  const query = `Read captures/ files matching ${CAPTURE_NAME.source}. Ignore symlinks and .gz.partial; prefer plain over gzip when both represent one archive. Decode gzip in memory, never extract. For each nonblank JSONL line, JSON.parse and group by positive integer root schema_version, 'missing', 'invalid_schema', or 'invalid_json'. Max line 128 MiB. Samples hash filename:line. Verify file signatures and inventory are unchanged after scanning.`;
  let inventory:
    | {
        files: string[];
        total: number;
        buckets: Map<string, ReturnType<typeof counter>>;
        filesByVersion: Map<string, Set<string>>;
      }
    | undefined;
  await report.run(
    "C4",
    "Finalizer capture records across current and rotated files",
    query,
    async () => {
      const directory = join(bank, "captures");
      const files = capturePaths(directory);
      const signatures = files.map(fileSignature);
      const buckets = new Map(
        ["1", "2", "missing", "invalid_schema", "invalid_json"].map((key) => [key, counter()]),
      );
      const filesByVersion = new Map<string, Set<string>>();
      let total = 0;
      for (const path of files) {
        let lineNumber = 0;
        for await (const line of captureLines(path)) {
          lineNumber += 1;
          if (line.trim().length === 0) continue;
          total += 1;
          let key: string;
          try {
            const value: unknown = JSON.parse(line);
            key = !objectValue(value)
              ? "invalid_schema"
              : value.schema_version === undefined
                ? "missing"
                : typeof value.schema_version === "number" &&
                    Number.isSafeInteger(value.schema_version) &&
                    value.schema_version > 0
                  ? String(value.schema_version)
                  : "invalid_schema";
          } catch {
            key = "invalid_json";
          }
          const bucket = buckets.get(key) ?? counter();
          bucket.add(`${path.slice(directory.length + 1)}:${lineNumber}`);
          buckets.set(key, bucket);
          const versionFiles = filesByVersion.get(key) ?? new Set<string>();
          versionFiles.add(path);
          filesByVersion.set(key, versionFiles);
        }
      }
      if (
        JSON.stringify(files) !== JSON.stringify(capturePaths(directory)) ||
        files.some((path, index) => fileSignature(path) !== signatures[index])
      )
        throw new ResidueReportError(
          "Capture inventory changed during read; retry on an idle bank or snapshot",
        );
      inventory = { files, total, buckets, filesByVersion };
      return { count: total };
    },
  );
  if (inventory !== undefined) {
    const { files, total, buckets, filesByVersion } = inventory;
    await report.run(
      "C4.files",
      "Finalizer capture files (one representation per archive)",
      query,
      () => ({ count: files.length }),
    );
    for (const [key, bucket] of [...buckets].sort(([a], [b]) => a.localeCompare(b))) {
      await report.run(
        `C4.schema_version.${key}`,
        `Finalizer capture records with schema_version ${key}`,
        query,
        () => ({ ...bucket.result, total }),
      );
      await report.run(
        `C4.files.schema_version.${key}`,
        `Finalizer capture files containing schema_version ${key}`,
        `${query} Count each file once per bucket; mixed-version files contribute to multiple buckets.`,
        () => ({ count: filesByVersion.get(key)?.size ?? 0, total: files.length }),
      );
    }
  }
}
