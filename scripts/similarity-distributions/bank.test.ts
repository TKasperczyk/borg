import { createHash } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { spawnSync } from "node:child_process";
import { connect } from "@lancedb/lancedb";
import { Field, FixedSizeList, Float32, Schema, Utf8 } from "apache-arrow";
import { afterEach, describe, expect, it, vi } from "vitest";
import { measureBank, parseMeasurementArgs } from "../measure-similarity-distributions.js";
import {
  decodeRows,
  measurementCacheDirectory,
  outputDirectory,
  pairTables,
  readFamilies,
  readTable,
  resolveBankPaths,
} from "./bank.js";
import { buildProposals, type RunReport } from "./report.js";
import { THRESHOLDS } from "./inventory.js";

const cleanup: string[] = [];
const close: (() => void)[] = [];
afterEach(() => {
  vi.restoreAllMocks();
  for (const dispose of close.splice(0)) dispose();
  for (const directory of cleanup.splice(0)) rmSync(directory, { recursive: true, force: true });
});
function temporary() {
  const root = mkdtempSync(join(tmpdir(), "borg-similarity-test-"));
  cleanup.push(root);
  return root;
}
function vector(dimensions: number, x: number, y: number) {
  const values = new Array<number>(dimensions).fill(0);
  values[0] = x;
  values[1] = y;
  return values;
}
function raw(id: string, dimensions: number, x = 1, y = 0) {
  return { id, title: `Title ${id}`, embedding: vector(dimensions, x, y) };
}
async function fixture(previous = true) {
  const root = temporary();
  const bank = join(root, "tenant");
  mkdirSync(bank);
  for (const [name, dimensions] of previous
    ? ([
        ["lancedb", 1024],
        ["lancedb.prev-1", 4096],
      ] as const)
    : ([["lancedb", 1024]] as const)) {
    const connection = await connect(join(bank, name));
    try {
      const schema = new Schema([
        new Field("id", new Utf8(), false),
        new Field("title", new Utf8(), false),
        new Field(
          "embedding",
          new FixedSizeList(dimensions, new Field("item", new Float32(), true)),
          false,
        ),
      ]);
      const similarity = dimensions === 1024 ? 0.5 : 0.9;
      const rows = [
        raw("a", dimensions),
        raw("b", dimensions, similarity, Math.sqrt(1 - similarity ** 2)),
        raw("c", dimensions, -1, 0),
        raw(dimensions === 1024 ? "new-only" : "old-only", dimensions),
      ];
      if (dimensions === 4096) rows.reverse();
      const table = await connection.createTable("episodes", rows, { schema });
      table.close();
    } finally {
      connection.close();
    }
  }
  const db = new DatabaseSync(join(bank, "borg.db"));
  db.exec("CREATE TABLE episode_index (episode_id TEXT PRIMARY KEY, consolidation_family_id TEXT)");
  db.exec(
    "INSERT INTO episode_index VALUES ('a','family-1'), ('b','family-1'), ('c','family-2'), ('new-only',NULL)",
  );
  db.close();
  writeFileSync(
    join(bank, "embedding-profile.json"),
    JSON.stringify({
      version: 1,
      model: "scw/bge-m3",
      dimensions: 1024,
      generation: 1,
      created_at: 1,
      updated_at: 2,
      migrated_from: { model: "qwen3-embedding-8b", dimensions: 4096, generation: 0 },
    }),
  );
  return { bank, out: join(root, "out"), root };
}

function fingerprintTree(root: string): Record<string, { bytes: string; mtime: number }> {
  const files: Record<string, { bytes: string; mtime: number }> = {};
  const visit = (directory: string) => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) visit(path);
      else
        files[relative(root, path)] = {
          bytes: createHash("sha256").update(readFileSync(path)).digest("hex"),
          mtime: statSync(path).mtimeMs,
        };
    }
  };
  visit(root);
  return files;
}

describe("bank reading and model pairing", () => {
  it("honors an explicit cache and XDG_CACHE_HOME when HOME is read-only", async () => {
    const { bank, root } = await fixture();
    const cache = join(root, "cache");
    expect(measurementCacheDirectory(cache, { HOME: "/", XDG_CACHE_HOME: "/other" })).toBe(cache);
    expect(measurementCacheDirectory(undefined, { HOME: "/", XDG_CACHE_HOME: cache })).toBe(cache);
    expect(measurementCacheDirectory(undefined, { HOME: "/" })).not.toBe("/.cache");
    expect(readFamilies(bank, cache).byId.size).toBe(3);
    expect(existsSync(join(cache, "borg-similarity-distributions"))).toBe(true);
    expect(() => readFamilies(bank, bank)).toThrow("outside");
    expect(
      parseMeasurementArgs(["--bank", bank, "--out", root, "--cache-dir", cache])?.cacheDir,
    ).toBe(cache);
  });

  it.each([1024, 4096])(
    "reads all seven table shapes without a hidden row limit at %i dimensions",
    async (dimensions) => {
      const connection = await connect(join(temporary(), "lancedb"));
      const definitions = [
        ["episodes", "id", "title"],
        ["semantic_nodes", "id", "label"],
        ["open_questions", "id", "question"],
        ["action_records", "id", "description"],
        ["skills", "id", "name"],
        ["observed_events", "id", "interaction_text"],
        ["image_perception_embeddings", "payload_id", "embedding_text"],
      ] as const;
      try {
        for (const [name, key, title] of definitions) {
          const schema = new Schema([
            new Field(key, new Utf8(), false),
            new Field(title, new Utf8(), false),
            new Field(
              "embedding",
              new FixedSizeList(dimensions, new Field("item", new Float32(), true)),
              false,
            ),
          ]);
          const table = await connection.createTable(
            name,
            Array.from({ length: 15 }, (_, index) => ({
              [key]: `id-${String(index).padStart(2, "0")}`,
              [title]: `Text ${index}`,
              embedding: vector(dimensions, 1, index / 15),
            })),
            { schema },
          );
          table.close();
          const loaded = await readTable(connection, name);
          expect(loaded.rows).toHaveLength(15);
          expect(loaded.rows[14]).toMatchObject({ id: "id-14", title: "Text 14" });
          expect(loaded.dimensions).toBe(dimensions);
        }
      } finally {
        connection.close();
      }
    },
  );

  it("pairs by ID, not physical order, and excludes invalid vectors from both models", () => {
    const current = decodeRows(
      [raw("b", 1024), raw("a", 1024), raw("new", 1024), raw("bad", 1024, 0, 0)],
      "episodes",
      1024,
    );
    const prev = decodeRows(
      [raw("a", 4096), raw("bad", 4096), raw("b", 4096), raw("old", 4096)],
      "episodes",
      4096,
    );
    const paired = pairTables(current, prev);
    expect(paired.current.map((r) => r.id)).toEqual(["a", "b"]);
    expect(paired.prev.map((r) => r.id)).toEqual(["a", "b"]);
    expect(paired.current[0]?.vector.length).toBe(1024);
    expect(paired.prev[0]?.vector.length).toBe(4096);
    expect(paired.audit).toMatchObject({
      common_count: 2,
      current_only_ids: ["new"],
      prev_only_ids: ["bad", "old"],
      current_invalid: [{ id: "bad", reason: expect.stringContaining("zero-norm") }],
    });
    expect(() => decodeRows([raw("a", 1024), raw("a", 1024)], "episodes", 1024)).toThrow(
      "Duplicate id",
    );
    const invalid = decodeRows(
      [
        { id: "nan", embedding: new Float32Array([NaN, 1]) },
        { id: "missing" },
        { id: "dimension", embedding: [1, 0, 0] },
      ],
      "episodes",
      2,
    );
    expect(invalid.rows).toEqual([]);
    expect(invalid.invalid).toHaveLength(3);
  });

  it("runs both models on synthetic LanceDB tables without modifying the bank or calling a gateway", async () => {
    const { bank, out } = await fixture();
    const before = fingerprintTree(bank);
    const fetch = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("No gateway allowed"));
    const current = await measureBank(
      { bank, out, vectors: "current", tables: ["episodes"], sample: 2, seed: "integration" },
      () => {},
    );
    const prev = await measureBank(
      { bank, out, vectors: "prev", tables: ["episodes"], sample: 2, seed: "integration" },
      () => {},
    );
    expect(fetch).not.toHaveBeenCalled();
    expect(fingerprintTree(bank)).toEqual(before);
    expect(current.directory).toBe(prev.directory);
    const c = current.run.tables.episodes!;
    const p = prev.run.tables.episodes!;
    expect(c.pairing).toEqual(p.pairing);
    expect(c.pairing).toMatchObject({
      common_count: 3,
      current_only_ids: ["new-only"],
      prev_only_ids: ["old-only"],
    });
    expect(c.measurement.random_pair_ids_sha256).toBe(p.measurement.random_pair_ids_sha256);
    expect(c.comparison_sha256).toBe(p.comparison_sha256);
    expect(c.measurement.families?.within_family.percentiles.p50).toBeCloseTo(0.5);
    expect(p.measurement.families?.within_family.percentiles.p50).toBeCloseTo(0.9);
    expect(c.measurement.families?.across_family.population).toBe(2);
    const report = JSON.parse(readFileSync(join(current.directory, "report.json"), "utf8"));
    expect(report.proposals).toHaveLength(THRESHOLDS.length);
    expect(
      report.proposals.find(
        (proposal: { id: string }) => proposal.id === "consolidation_similarity",
      ),
    ).toMatchObject({ status: "proxy_proposal", value: 0.82 });
    expect(
      report.proposals.find(
        (proposal: { id: string }) => proposal.id === "BORG_RECALL_ABSTAIN_THRESHOLD",
      ),
    ).toMatchObject({ status: "not_a_cosine_threshold", proposed_value: null });
    expect(readdirSync(current.directory).sort()).toEqual([
      "current.json",
      "prev.json",
      "report.json",
      "summary.md",
    ]);
    expect(readFileSync(join(current.directory, "summary.md"), "utf8")).toContain(
      "consolidation_family_id",
    );
    const repeated = await measureBank(
      { bank, out, vectors: "current", tables: ["episodes"], sample: 2, seed: "integration" },
      () => {},
    );
    expect(repeated.run).toEqual(current.run);
  });

  it("fails clearly for a missing previous directory but can report current-only data", async () => {
    const { bank, out } = await fixture(false);
    await expect(
      measureBank(
        { bank, out, vectors: "prev", tables: ["episodes"], sample: 10, seed: "test" },
        () => {},
      ),
    ).rejects.toThrow("No lancedb.prev-<N>");
    expect(existsSync(out)).toBe(false);
    const current = await measureBank(
      { bank, out, vectors: "current", tables: ["episodes", "skills"], sample: 10, seed: "test" },
      () => {},
    );
    expect(current.run.tables.episodes?.measurement.row_count).toBe(4);
    expect(current.run.tables.episodes?.pairing.paired).toBe(false);
    expect(current.run.tables.skills?.present).toBe(false);
    expect(
      buildProposals(current.run, undefined).find((p) => p.id === "consolidation_similarity")
        ?.status,
    ).toBe("run_both_current_and_prev");
  });

  it("chooses the profile's target generation, otherwise the largest numeric suffix", () => {
    const bank = temporary();
    for (const name of ["lancedb", "lancedb.prev-2", "lancedb.prev-10", "lancedb.prev-bogus"])
      mkdirSync(join(bank, name));
    expect(resolveBankPaths(bank, "prev").prev).toBe(join(bank, "lancedb.prev-10"));
    writeFileSync(
      join(bank, "embedding-profile.json"),
      JSON.stringify({
        version: 1,
        model: "scw/bge-m3",
        dimensions: 1024,
        generation: 2,
        created_at: 1,
        updated_at: 2,
        migrated_from: { model: "qwen3-embedding-8b", dimensions: 4096, generation: 1 },
      }),
    );
    expect(resolveBankPaths(bank, "prev").prev).toBe(join(bank, "lancedb.prev-2"));
  });

  it("reads uncheckpointed family metadata from a WAL copy without sidecar writes to the bank", () => {
    const source = temporary();
    const db = new DatabaseSync(join(source, "borg.db"));
    close.push(() => db.close());
    db.exec("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0");
    db.exec(
      "CREATE TABLE episode_index (episode_id TEXT, consolidation_family_id TEXT); INSERT INTO episode_index VALUES ('a','one')",
    );
    const bank = temporary();
    copyFileSync(join(source, "borg.db"), join(bank, "borg.db"));
    copyFileSync(join(source, "borg.db-wal"), join(bank, "borg.db-wal"));
    const before = fingerprintTree(bank);
    expect(readFamilies(bank).byId.get("a")).toBe("one");
    expect(fingerprintTree(bank)).toEqual(before);
    expect(existsSync(join(bank, "borg.db-shm"))).toBe(false);
    const legacy = temporary();
    const old = new DatabaseSync(join(legacy, "borg.db"));
    old.exec("CREATE TABLE episode_index (episode_id TEXT)");
    old.close();
    expect(readFamilies(legacy).reason).toContain("lacks");
    expect(readFamilies(temporary()).reason).toBe("No borg.db in bank copy");
  });

  it("rejects output paths in the bank, including paths through symlinks", () => {
    const root = temporary();
    const bank = join(root, "tenant");
    mkdirSync(bank);
    expect(() => outputDirectory(bank, bank)).toThrow("outside");
    expect(() => outputDirectory(bank, join(bank, "reports"))).toThrow("outside");
    symlinkSync(bank, join(root, "alias"));
    expect(() => outputDirectory(bank, join(root, "alias", "reports"))).toThrow("outside");
  });
});

describe("proposal compatibility and CLI", () => {
  it("does not map stale snapshots, seeds, samples, or the wrong model/dimensions", async () => {
    const { bank, out } = await fixture();
    const current = (
      await measureBank(
        { bank, out, vectors: "current", tables: ["episodes"], sample: 10, seed: "same" },
        () => {},
      )
    ).run;
    const prev = (
      await measureBank(
        { bank, out, vectors: "prev", tables: ["episodes"], sample: 10, seed: "same" },
        () => {},
      )
    ).run;
    const status = (c: RunReport, p: RunReport) =>
      buildProposals(c, p).find((item) => item.id === "consolidation_similarity")!;
    expect(status(current, prev).proposed_value).toBeTypeOf("number");
    for (const patch of [{ seed: "different" }, { sample: 11 }, { bank: "other" }]) {
      expect(status({ ...current, ...patch }, prev).proposed_value).toBeNull();
    }
    const stale = structuredClone(current);
    stale.tables.episodes!.comparison_sha256 = "changed";
    expect(status(stale, prev).status).toContain("incompatible_runs");
    const wrongDimension = structuredClone(current);
    wrongDimension.tables.episodes!.dimensions = 4096;
    expect(status(wrongDimension, prev).status).toContain("expected_bge");
    expect(status({ ...current, model: "unrelated-1024-model" }, prev).proposed_value).toBeNull();
  });

  it("validates options and runs with node --import tsx without pnpm", async () => {
    expect(parseMeasurementArgs(["--bank", "/bank", "--out", "/out"])?.tables).toHaveLength(6);
    expect(parseMeasurementArgs(["--help"])).toBeNull();
    for (const extra of [
      ["--sample", "0"],
      ["--sample", "NaN"],
      ["--vectors", "both"],
      ["--tables", "episodes,episodes"],
      ["--tables", "not_a_table"],
    ]) {
      expect(() => parseMeasurementArgs(["--bank", "/bank", "--out", "/out", ...extra])).toThrow();
    }
    const { bank, out } = await fixture();
    const result = spawnSync(
      process.execPath,
      [
        "--import",
        "tsx",
        "scripts/measure-similarity-distributions.ts",
        "--bank",
        bank,
        "--out",
        out,
        "--vectors",
        "prev",
        "--tables",
        "episodes",
        "--sample",
        "2",
      ],
      { encoding: "utf8" },
    );
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout).toBe("");
    expect(result.stderr).toContain("Wrote");
    expect(existsSync(join(outputDirectory(bank, out), "prev.json"))).toBe(true);
  });
});
