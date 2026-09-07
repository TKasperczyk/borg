import { createHash } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  realpathSync,
  rmSync,
  statSync,
} from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { connect, type Connection } from "@lancedb/lancedb";
import { z } from "zod";
import {
  embeddingDimensionsFromSchema,
  readBankEmbeddingProfile,
  VECTOR_TABLE_NAMES,
} from "../../src/embeddings/bank-profile.js";
import { toFloat32Array } from "../../src/storage/codecs.js";
import { openReadOnlyDatabase } from "../../src/storage/sqlite/index.js";
import { StorageError } from "../../src/util/errors.js";

export class MeasurementError extends StorageError {}
export type TableName = (typeof VECTOR_TABLE_NAMES)[number];
export const tableNameSchema = z.enum(VECTOR_TABLE_NAMES);
export const DEFAULT_TABLES: TableName[] = [
  "episodes",
  "semantic_nodes",
  "open_questions",
  "action_records",
  "skills",
  "observed_events",
];
export type VectorRow = { id: string; title: string; vector: Float32Array };
export type LoadedTable = {
  dimensions: number | null;
  present: boolean;
  rows: VectorRow[];
  invalid: { id: string; reason: string }[];
};
export type Families = { source: string | null; reason: string | null; byId: Map<string, string> };

export function resolveBankPaths(bank: string, vectors: "current" | "prev") {
  const directory = realpathSync(resolve(bank));
  const current = join(directory, "lancedb");
  if (!existsSync(current) || !statSync(current).isDirectory()) {
    throw new MeasurementError(`No existing lancedb directory in ${directory}`);
  }
  const profile = readBankEmbeddingProfile(directory);
  const candidates = readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^lancedb\.prev-[0-9]+$/.test(entry.name))
    .map((entry) => entry.name)
    .sort((left, right) => {
      const a = BigInt(left.slice("lancedb.prev-".length));
      const b = BigInt(right.slice("lancedb.prev-".length));
      return a > b ? -1 : a < b ? 1 : left.localeCompare(right);
    });
  const expected = profile === undefined ? undefined : `lancedb.prev-${profile.generation}`;
  // Migration names the old directory using the TARGET generation, not migrated_from.generation.
  const previousName =
    expected !== undefined && candidates.includes(expected) ? expected : candidates[0];
  const prev = previousName === undefined ? null : join(directory, previousName);
  if (vectors === "prev" && prev === null) {
    throw new MeasurementError(
      `No lancedb.prev-<N> directory in ${directory}; copy the previous vectors alongside lancedb first`,
    );
  }
  return {
    directory,
    current,
    prev,
    profile: profile ?? null,
    previousProfile: previousName === expected ? (profile?.migrated_from ?? null) : null,
    warnings: [
      ...(prev === null
        ? ["No previous vectors: current rows are unpaired and proposals remain unavailable."]
        : []),
      ...(previousName !== undefined && previousName !== expected
        ? [
            `Selected highest numeric previous directory ${previousName}; no matching generation metadata.`,
          ]
        : []),
    ],
  };
}

// Resolve existing ancestors too: an output symlink must not redirect writes into the bank.
function canonicalFuturePath(path: string): string {
  if (existsSync(path)) return realpathSync(path);
  return join(canonicalFuturePath(dirname(path)), basename(path));
}

export function outputDirectory(bank: string, out: string): string {
  const root = canonicalFuturePath(resolve(out));
  const child = relative(realpathSync(bank), root);
  if (child === "" || (!isAbsolute(child) && child !== ".." && !child.startsWith(`..${sep}`))) {
    throw new MeasurementError("--out must be outside the copied bank (including symlink targets)");
  }
  const key = createHash("sha256").update(realpathSync(bank)).digest("hex").slice(0, 12);
  const result = join(root, `${basename(bank)}-${key}`);
  if (existsSync(result) && realpathSync(result) !== result) {
    throw new MeasurementError(`Report directory must not be a symlink: ${result}`);
  }
  return result;
}

const TITLE_FIELDS: Record<TableName, string[]> = {
  episodes: ["title"],
  semantic_nodes: ["label", "description"],
  open_questions: ["question"],
  action_records: ["description"],
  skills: ["name", "applies_when"],
  observed_events: ["interaction_text"],
  image_perception_embeddings: ["embedding_text"],
};

export function decodeRows(
  rawRows: readonly Record<string, unknown>[],
  name: TableName,
  dimensions: number,
): LoadedTable {
  const rows: VectorRow[] = [];
  const invalid: LoadedTable["invalid"] = [];
  const ids = new Set<string>();
  for (const raw of rawRows) {
    const id = z
      .string()
      .min(1)
      .parse(raw[name === "image_perception_embeddings" ? "payload_id" : "id"]);
    if (ids.has(id)) throw new MeasurementError(`Duplicate id ${id} in ${name}; cannot pair rows`);
    ids.add(id);
    let vector: Float32Array;
    try {
      vector = toFloat32Array(raw.embedding, {
        arrayLikeErrorMessage: "Missing or non-array embedding",
        nonFiniteErrorMessage: "Non-finite embedding component",
        errorCode: "BORG_STORAGE_ERROR",
      });
    } catch (error) {
      if (!(error instanceof StorageError)) throw error;
      invalid.push({ id, reason: error.message });
      continue;
    }
    if (
      vector.length !== dimensions ||
      !vector.every(Number.isFinite) ||
      !vector.some((value) => value !== 0)
    ) {
      invalid.push({ id, reason: "Wrong dimension, non-finite or zero-norm embedding" });
      continue;
    }
    const title = TITLE_FIELDS[name]
      .map((field) => raw[field])
      .find((value) => typeof value === "string" && value.length > 0);
    rows.push({ id, title: typeof title === "string" ? title : id, vector });
  }
  rows.sort((left, right) => (left.id < right.id ? -1 : left.id > right.id ? 1 : 0));
  invalid.sort((left, right) => (left.id < right.id ? -1 : left.id > right.id ? 1 : 0));
  return { dimensions, present: true, rows, invalid };
}

export async function readTable(connection: Connection, name: TableName): Promise<LoadedTable> {
  if (!(await connection.tableNames()).includes(name)) {
    return { present: false, dimensions: null, rows: [], invalid: [] };
  }
  // Same direct-open pattern as eval/embedding-ab/bank.ts. Never use Borg.open,
  // LanceDbStore.openTable, repository backfills, migrations, or index construction.
  const table = await connection.openTable(name);
  try {
    const schema = await table.schema();
    const dimensions = embeddingDimensionsFromSchema(schema);
    const wanted = new Set(["id", "payload_id", "embedding", ...TITLE_FIELDS[name]]);
    const columns = schema.fields.map((field) => field.name).filter((name) => wanted.has(name));
    return decodeRows(await table.query().select(columns).toArray(), name, dimensions);
  } finally {
    table.close();
  }
}

export function pairTables(current: LoadedTable, prev: LoadedTable | null) {
  const currentIds = new Set(current.rows.map((row) => row.id));
  const prevIds = new Set(prev?.rows.map((row) => row.id) ?? []);
  const currentRows = current.rows.filter((row) => prev === null || prevIds.has(row.id));
  const previousRows = prev?.rows.filter((row) => currentIds.has(row.id)) ?? [];
  const sort = (rows: VectorRow[]) => rows.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return {
    current: sort(currentRows),
    prev: sort(previousRows),
    audit: {
      paired: prev !== null,
      common_count: prev === null ? null : currentRows.length,
      current_count: current.rows.length + current.invalid.length,
      prev_count: prev === null ? null : prev.rows.length + prev.invalid.length,
      current_only_ids: current.rows
        .filter((row) => !prevIds.has(row.id))
        .map((row) => row.id)
        .sort(),
      prev_only_ids:
        prev?.rows
          .filter((row) => !currentIds.has(row.id))
          .map((row) => row.id)
          .sort() ?? [],
      current_invalid: current.invalid,
      prev_invalid: prev?.invalid ?? [],
    },
  };
}

export function tableFingerprint(table: LoadedTable | null): string {
  const hash = createHash("sha256");
  hash.update(
    JSON.stringify(
      table === null
        ? null
        : { dimensions: table.dimensions, present: table.present, invalid: table.invalid },
    ),
  );
  for (const row of table?.rows ?? []) {
    hash.update(JSON.stringify([row.id, row.title]));
    hash.update(Buffer.from(row.vector.buffer, row.vector.byteOffset, row.vector.byteLength));
  }
  return hash.digest("hex");
}

export function readFamilies(bank: string): Families {
  const path = join(bank, "borg.db");
  const unavailable = (reason: string): Families => ({ source: null, reason, byId: new Map() });
  if (!existsSync(path)) return unavailable("No borg.db in bank copy");
  // SQLite readOnly can still create/update WAL shared-memory sidecars. Query
  // a disposable main+WAL copy under HOME so even those writes stay off-bank.
  // The input must be a quiescent copy, as with the embedding A/B evaluator.
  const scratchRoot = join(homedir(), ".cache", "borg-similarity-distributions");
  mkdirSync(scratchRoot, { recursive: true, mode: 0o700 });
  const scratch = mkdtempSync(join(scratchRoot, "sqlite-"));
  try {
    const snapshot = join(scratch, "borg.db");
    copyFileSync(path, snapshot);
    if (existsSync(`${path}-wal`)) copyFileSync(`${path}-wal`, `${snapshot}-wal`);
    const db = openReadOnlyDatabase(snapshot);
    try {
      const columns = db
        .prepare("PRAGMA table_info(episode_index)")
        .all()
        .map((column) => column.name);
      if (!columns.includes("episode_id") || !columns.includes("consolidation_family_id")) {
        return unavailable("SQLite episode_index lacks episode_id/consolidation_family_id");
      }
      const rows = db
        .prepare(
          "SELECT episode_id, consolidation_family_id FROM episode_index WHERE consolidation_family_id IS NOT NULL AND consolidation_family_id != '' ORDER BY episode_id",
        )
        .all();
      const byId = new Map(
        rows.map((row) => [
          z.string().parse(row.episode_id),
          z.string().parse(row.consolidation_family_id),
        ]),
      );
      return { source: "borg.db:episode_index.consolidation_family_id", reason: null, byId };
    } finally {
      db.close();
    }
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

export async function openVectorBanks(paths: ReturnType<typeof resolveBankPaths>) {
  const current = await connect(paths.current);
  try {
    const prev = paths.prev === null ? null : await connect(paths.prev);
    return { current, prev };
  } catch (error) {
    current.close();
    throw error;
  }
}
