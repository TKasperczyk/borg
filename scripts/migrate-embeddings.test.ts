import {
  appendFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { connect } from "@lancedb/lancedb";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LanceDbTable } from "../src/storage/lancedb/index.js";
import { FakeEmbeddingClient } from "../src/embeddings/index.js";
import {
  acquireEmbeddingBankAccess,
  EMBEDDING_FENCE_FILE,
  readBankEmbeddingProfile,
} from "../src/embeddings/bank-profile.js";
import { readJsonFile } from "../src/util/atomic-write.js";
import { parseJsonLines } from "../src/util/durable-jsonl.js";
import { inventoryBank, migrationRowText, VECTOR_TABLES } from "./embedding-migration/inventory.js";
import {
  migrateTenant,
  MIGRATION_JOURNAL,
  type MigrationDependencies,
} from "./embedding-migration/migrate.js";
import { parseEmbeddingMigrationArgs } from "./migrate-embeddings.js";
import { migrationHeadroom } from "./embedding-migration/backup.js";

const cleanup: string[] = [];
afterEach(() => {
  for (const directory of cleanup.splice(0)) rmSync(directory, { recursive: true, force: true });
});

async function fixture(count = 2, sourceDimensions = 4, targetDimensions = 2) {
  const root = mkdtempSync(join(tmpdir(), "borg-embedding-migration-"));
  cleanup.push(root);
  const tenantDir = join(root, "team-agent-ai");
  mkdirSync(tenantDir);
  const db = new DatabaseSync(join(tenantDir, "borg.db"));
  db.exec("PRAGMA journal_mode=WAL");
  const connection = await connect(join(tenantDir, "lancedb"));
  const original: Record<string, Record<string, unknown>[]> = {};
  try {
    for (const definition of VECTOR_TABLES) {
      const schema = definition.schema(sourceDimensions);
      const raw = await connection.createEmptyTable(definition.name, schema);
      const table = new LanceDbTable(raw);
      try {
        const rows = Array.from({ length: count }, (_, index) => {
          const fields = Object.fromEntries(
            schema.fields
              .filter((field) => field.name !== "embedding")
              .map((field) => [
                field.name,
                field.nullable
                  ? null
                  : field.type.toString() === "Float64"
                    ? 1_000 + index
                    : field.type.toString() === "Bool"
                      ? true
                      : "stored",
              ]),
          );
          return {
            ...fields,
            [definition.key]: `${definition.name}-${index}`,
            embedding: Array.from({ length: sourceDimensions }, (_, index) => index + 1),
            ...(definition.name === "episodes"
              ? {
                  title: `Episode ${index}`,
                  narrative: `Preserved narrative ${index}`,
                  tags: '["tag","étiquette"]',
                  participants: '["participant"]',
                  source_stream_ids: '["stream-source"]',
                  lineage_derived_from: "[]",
                  lineage_supersedes: "[]",
                  episode_kind: index === 1 ? "consolidation_version" : "raw",
                }
              : {}),
            ...(definition.name === "semantic_nodes"
              ? {
                  label: `Node ${index}`,
                  description: `Node description ${index}`,
                  aliases: '["alias"]',
                  source_episode_ids: '["episode-source"]',
                  archived: true,
                  superseded_by: "successor",
                }
              : {}),
          };
        });
        original[definition.name] = rows;
        if (rows.length) {
          await table.upsert(
            rows.map((row) => (definition.name === "episodes" ? { ...row, shared: false } : row)),
            { on: definition.key },
          );
          if (definition.name === "episodes")
            await raw.update({ valuesSql: { shared: "CAST(NULL AS BOOLEAN)" } });
        }
        const sqlFields =
          definition.name === "episodes"
            ? ["episode_id", "archived"]
            : schema.fields
                .filter((field) => field.name !== "embedding")
                .map((field) => field.name);
        db.exec(
          `CREATE TABLE ${definition.sql} (${sqlFields.map((field) => `"${field}"`).join(", ")})`,
        );
        for (const row of rows) {
          const record: Record<string, unknown> = row;
          const values =
            definition.name === "episodes"
              ? [record[definition.key], 1]
              : sqlFields.map((field) =>
                  typeof record[field] === "boolean" ? Number(record[field]) : record[field],
                );
          db.prepare(
            `INSERT INTO ${definition.sql} VALUES (${values.map(() => "?").join(", ")})`,
          ).run(...(values as (string | number | null)[]));
        }
      } finally {
        raw.close();
      }
    }
    db.exec(
      "CREATE TABLE review_queue (id INTEGER PRIMARY KEY, refs TEXT); CREATE TABLE maintenance_audit (id INTEGER PRIMARY KEY, reversal TEXT, targets TEXT)",
    );
    db.prepare("INSERT INTO review_queue VALUES (1, ?)").run(
      JSON.stringify({
        node: { label: "Queued", description: "text", aliases: [], embedding: [1, 2, 3, 4] },
      }),
    );
    db.prepare("INSERT INTO maintenance_audit VALUES (1, ?, '{}')").run(
      JSON.stringify({ previous: [{ embedding: [1, 2, 3, 4] }] }),
    );
    writeFileSync(
      join(tenantDir, "saved-plan.json"),
      JSON.stringify({ kind: "borg_maintenance_plan", items: [{ embedding: [1, 2, 3, 4] }] }),
    );
    writeFileSync(join(tenantDir, "stream.jsonl"), '{"unchanged":"preserve me"}\n');
  } finally {
    db.close();
    connection.close();
  }
  const delegate = new FakeEmbeddingClient(targetDimensions);
  const embedBatch = vi.fn(delegate.embedBatch.bind(delegate));
  const client = {
    profile: { model: "scw/bge-m3", dimensions: targetDimensions },
    embed: delegate.embed.bind(delegate),
    embedBatch,
  };
  const options = {
    tenantDir,
    backupDir: join(root, "backups"),
    target: client.profile,
    sourceModel: "generative-apis/qwen3-embedding-8b",
    batchSize: 1,
    concurrency: 1,
  };
  return { root, tenantDir, original, client, options };
}

describe("storage-only embedding migration", () => {
  it("migrates the production 4096-to-1024 geometry across all seven tables", async () => {
    const bank = await fixture(1, 4096, 1024);
    expect(await migrateTenant(bank.options, { client: bank.client })).toMatchObject({
      complete: true,
      profile: { dimensions: 1024, migrated_from: { dimensions: 4096 } },
    });
    expect(
      (await inventoryBank(bank.tenantDir, 1024)).tables.map((table) => table.dimensions),
    ).toEqual(Array(7).fill(1024));
  });
  it("inventories every table, lifecycle row and serialized vector without changing a dry-run bank", async () => {
    const bank = await fixture();
    const before = readdirSync(bank.tenantDir).sort();
    const databaseBefore = readFileSync(join(bank.tenantDir, "borg.db"));
    const report = await migrateTenant({ ...bank.options, dryRun: true });
    expect(report).toMatchObject({
      dry_run: true,
      complete: true,
      inventory: {
        tables: expect.arrayContaining(
          VECTOR_TABLES.map((table) =>
            expect.objectContaining({ name: table.name, rows: expect.any(Array) }),
          ),
        ),
        serialized_vectors: expect.any(Array),
      },
    });
    expect(
      (report.inventory as Awaited<ReturnType<typeof inventoryBank>>).serialized_vectors,
    ).toHaveLength(3);
    // SQLite read-only opens may materialize WAL/SHM coordination files.
    expect(
      readdirSync(bank.tenantDir)
        .filter((name) => !["borg.db-wal", "borg.db-shm"].includes(name))
        .sort(),
    ).toEqual(before);
    expect(readFileSync(join(bank.tenantDir, "borg.db"))).toEqual(databaseBefore);
    expect(existsSync(bank.options.backupDir)).toBe(false);
    expect(bank.client.embedBatch).not.toHaveBeenCalled();
  });

  it("backs up and preserves every non-vector field across all seven tables", async () => {
    const bank = await fixture();
    const report = await migrateTenant(bank.options, { client: bank.client });
    expect(report).toMatchObject({ complete: true, verified: { sanity_queries: 14 } });
    expect(readBankEmbeddingProfile(bank.tenantDir)).toMatchObject({
      model: "scw/bge-m3",
      dimensions: 2,
      generation: 1,
      migrated_from: { model: bank.options.sourceModel, dimensions: 4, generation: 0 },
    });
    expect(existsSync(join(bank.tenantDir, EMBEDDING_FENCE_FILE))).toBe(false);
    expect(existsSync(join(bank.tenantDir, "lancedb.prev-1"))).toBe(true);
    expect(readFileSync(join(bank.tenantDir, "stream.jsonl"), "utf8")).toBe(
      '{"unchanged":"preserve me"}\n',
    );
    const connection = await connect(join(bank.tenantDir, "lancedb"));
    try {
      for (const definition of VECTOR_TABLES) {
        const table = await connection.openTable(definition.name);
        try {
          const rows = await table.query().toArray();
          const fields = rows.map(({ embedding, ...row }) => ({
            ...row,
            embedding: Array.from(embedding as ArrayLike<number>),
          }));
          expect(fields).toEqual(
            expect.arrayContaining(
              bank.original[definition.name]!.map((row) => ({
                ...row,
                embedding: expect.arrayContaining([expect.any(Number), expect.any(Number)]),
              })),
            ),
          );
          for (const row of rows) expect((row.embedding as ArrayLike<number>).length).toBe(2);
        } finally {
          table.close();
        }
      }
    } finally {
      connection.close();
    }
    const texts = bank.client.embedBatch.mock.calls.flatMap(([texts]) => [...texts]);
    for (const definition of VECTOR_TABLES)
      for (const row of bank.original[definition.name]!)
        expect(texts).toContain(migrationRowText(definition.name, row));
    const before = bank.client.embedBatch.mock.calls.length;
    expect(await migrateTenant({ ...bank.options, verifyOnly: true })).toMatchObject({
      complete: true,
    });
    expect(bank.client.embedBatch).toHaveBeenCalledTimes(before);
  });

  it("resumes an interrupted committed batch and ignores a torn checkpoint tail", async () => {
    const bank = await fixture();
    await expect(
      migrateTenant(bank.options, {
        client: bank.client,
        afterBatch: () => {
          throw new Error("crash after batch");
        },
      }),
    ).rejects.toThrow("crash after batch");
    expect(existsSync(join(bank.tenantDir, EMBEDDING_FENCE_FILE))).toBe(true);
    const checkpoint = join(bank.tenantDir, ".embedding-migration-g1.jsonl");
    expect(parseJsonLines(checkpoint)).toHaveLength(1);
    appendFileSync(checkpoint, '{"interrupted":');
    await migrateTenant({ ...bank.options, resume: true }, { client: bank.client });
    expect(bank.client.embedBatch).toHaveBeenCalledTimes(14);
    expect(parseJsonLines(checkpoint)).toHaveLength(14);
  });

  it("replays a table commit interrupted before its checkpoint without duplicating ids", async () => {
    const bank = await fixture(1);
    await expect(
      migrateTenant(bank.options, {
        client: bank.client,
        afterTableWrite: () => {
          throw new Error("uncheckpointed commit");
        },
      }),
    ).rejects.toThrow("uncheckpointed commit");
    expect(
      await migrateTenant({ ...bank.options, resume: true }, { client: bank.client }),
    ).toMatchObject({ complete: true });
    expect(bank.client.embedBatch).toHaveBeenCalledTimes(8);
    expect(
      (await inventoryBank(bank.tenantDir, 2)).tables.map((table) => table.rows.length),
    ).toEqual(Array(7).fill(1));
  });

  it("bounds gateway retries and keeps the failed tenant fenced", async () => {
    const bank = await fixture(1);
    bank.client.embedBatch.mockRejectedValue(new Error("gateway unavailable"));
    await expect(
      migrateTenant(bank.options, { client: bank.client, retryDelaysMs: [0, 0] }),
    ).rejects.toThrow("gateway unavailable");
    expect(bank.client.embedBatch).toHaveBeenCalledTimes(3);
    expect(existsSync(join(bank.tenantDir, EMBEDDING_FENCE_FILE))).toBe(true);
    expect(readBankEmbeddingProfile(bank.tenantDir)).toBeUndefined();
  });

  it("refuses insufficient disk headroom before creating a backup", async () => {
    const bank = await fixture(0);
    const inventory = await inventoryBank(bank.tenantDir, 2);
    // Represent a corpus whose target vectors cannot fit on this filesystem.
    const oversized = {
      ...inventory,
      tables: inventory.tables.map((table) => ({
        ...table,
        rows: { length: Number.MAX_SAFE_INTEGER } as unknown as typeof table.rows,
      })),
    };
    expect(() =>
      migrationHeadroom(bank.tenantDir, bank.options.backupDir, oversized, 1024),
    ).toThrow("Insufficient disk headroom");
    expect(existsSync(bank.options.backupDir)).toBe(false);
  });

  it("embeds with bounded concurrent batches", async () => {
    const bank = await fixture(3);
    let active = 0;
    let peak = 0;
    const delegate = new FakeEmbeddingClient(2);
    bank.client.embedBatch.mockImplementation(async (texts) => {
      active += 1;
      peak = Math.max(peak, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active -= 1;
      return delegate.embedBatch(texts);
    });
    expect(
      await migrateTenant({ ...bank.options, concurrency: 2 }, { client: bank.client }),
    ).toMatchObject({ complete: true });
    expect(peak).toBe(2);
  });

  it.each(["previous", "live"] as const)(
    "recovers a crash immediately after the %s rename",
    async (step) => {
      const bank = await fixture(1);
      const deps: MigrationDependencies = {
        client: bank.client,
        afterRename: (renamed) => {
          if (renamed === step) throw new Error("rename crash");
        },
      };
      await expect(migrateTenant(bank.options, deps)).rejects.toThrow("rename crash");
      expect(existsSync(join(bank.tenantDir, EMBEDDING_FENCE_FILE))).toBe(true);
      expect(readJsonFile(join(bank.tenantDir, MIGRATION_JOURNAL))).toMatchObject({
        phase: "cutover",
      });
      expect(existsSync(join(bank.tenantDir, "lancedb"))).toBe(step === "live");
      const count = bank.client.embedBatch.mock.calls.length;
      expect(
        await migrateTenant({ ...bank.options, resume: true }, { client: bank.client }),
      ).toMatchObject({ complete: true });
      expect(bank.client.embedBatch).toHaveBeenCalledTimes(count);
    },
  );

  it("recreates all empty tables at the target dimension without embedding calls", async () => {
    const bank = await fixture(0);
    expect(await migrateTenant(bank.options, { client: bank.client })).toMatchObject({
      complete: true,
      verified: { sanity_queries: 0 },
    });
    expect(bank.client.embedBatch).not.toHaveBeenCalled();
    expect(
      (await inventoryBank(bank.tenantDir, 2)).tables.map((table) => table.dimensions),
    ).toEqual(Array(7).fill(2));
  });

  it("records a second generation for a same-dimension model migration", async () => {
    const bank = await fixture(1);
    await migrateTenant(bank.options, { client: bank.client });
    const profile = { model: "subsequent-model", dimensions: 2 };
    await migrateTenant(
      { ...bank.options, sourceModel: bank.options.target.model, target: profile },
      { client: { ...bank.client, profile } },
    );
    expect(readBankEmbeddingProfile(bank.tenantDir)).toMatchObject({
      ...profile,
      generation: 2,
      migrated_from: { model: "scw/bge-m3", dimensions: 2, generation: 1 },
    });
    expect(existsSync(join(bank.tenantDir, "lancedb.prev-1"))).toBe(true);
    expect(existsSync(join(bank.tenantDir, "lancedb.prev-2"))).toBe(true);
  });

  it("refuses verification when the backup completion manifest is lost", async () => {
    const bank = await fixture(0);
    const report = await migrateTenant(bank.options, { client: bank.client });
    rmSync(join(String(report.backup), ".embedding-backup.json"));
    await expect(migrateTenant({ ...bank.options, verifyOnly: true })).rejects.toMatchObject({
      code: "EMBEDDING_MIGRATION_BACKUP_INVALID",
    });
  });

  it("keeps the fence when a live tenant must first be evicted", async () => {
    const bank = await fixture(0);
    const release = await acquireEmbeddingBankAccess(bank.tenantDir);
    try {
      await expect(migrateTenant(bank.options, { client: bank.client })).rejects.toMatchObject({
        code: "EMBEDDING_BANK_BUSY",
      });
    } finally {
      await release();
    }
    expect(existsSync(join(bank.tenantDir, EMBEDDING_FENCE_FILE))).toBe(true);
    expect(
      await migrateTenant({ ...bank.options, resume: true }, { client: bank.client }),
    ).toMatchObject({ complete: true });
  });

  it("reports SQL-only rows and refuses to silently repair or discard them", async () => {
    const bank = await fixture();
    const db = new DatabaseSync(join(bank.tenantDir, "borg.db"));
    db.exec("INSERT INTO episode_stats VALUES ('orphan', 1)");
    db.close();
    const report = await migrateTenant({ ...bank.options, dryRun: true });
    expect(report.complete).toBe(false);
    await expect(migrateTenant(bank.options, { client: bank.client })).rejects.toMatchObject({
      code: "EMBEDDING_MIGRATION_INCOMPLETE",
    });
    expect(bank.client.embedBatch).not.toHaveBeenCalled();
    expect(existsSync(join(bank.tenantDir, "lancedb"))).toBe(true);
  });

  it("parses repeatable tenants and keeps backups out of tenant discovery", async () => {
    const bank = await fixture(0);
    const args = [
      "--data-root",
      bank.root,
      "--target-model",
      "scw/bge-m3",
      "--target-dims",
      "1024",
    ];
    expect(
      parseEmbeddingMigrationArgs([
        ...args,
        "--tenant",
        "team-agent-ai",
        "--tenant",
        "team-agent-esb",
      ]),
    ).toMatchObject({
      tenants: ["team-agent-ai", "team-agent-esb"],
      batchSize: 32,
      concurrency: 2,
    });
    expect(() =>
      parseEmbeddingMigrationArgs([...args, "--tenant", "team-agent-ai", "--all-tenants"]),
    ).toThrow("Choose");
    expect(() =>
      parseEmbeddingMigrationArgs([...args, "--all-tenants", "--concurrency", "0"]),
    ).toThrow();
  });
});
