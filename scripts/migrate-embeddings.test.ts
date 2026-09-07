import {
  appendFileSync,
  cpSync,
  chmodSync,
  truncateSync,
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
import { spawnSync } from "node:child_process";
import { connect } from "@lancedb/lancedb";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LanceDbTable } from "../src/storage/lancedb/index.js";
import { FakeEmbeddingClient } from "../src/embeddings/index.js";
import {
  acquireEmbeddingBankAccess,
  EMBEDDING_ACCESS_FILE,
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
import { migrationHeadroom, verifyTenantBackup } from "./embedding-migration/backup.js";
import {
  buildConsolidationEpisodeEmbeddingText,
  buildEpisodeEmbeddingText,
  collectProtectedEpisodeTokenLines,
  preserveProtectedEpisodeTokenLines,
} from "../src/memory/episodic/protected-lines.js";
import { Borg } from "../src/borg.js";
import { FakeLLMClient } from "../src/llm/test-support/fake-client.js";
import { EpisodicRepository } from "../src/memory/episodic/repository.js";
import { openDatabase } from "../src/storage/sqlite/index.js";
import { acquireFileLockLease, FILE_LOCK_STALE_MS } from "../src/stream/file-lock.js";
import { FILE_LOCK_GUARD_SUFFIX } from "../src/stream/file-lock-guard.js";
import { createEpisodeFixture } from "../src/offline/test-support.js";

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

async function mixedLegacyConsolidations(missingInputColumn = false) {
  const bank = await fixture(5);
  const connection = await connect(join(bank.tenantDir, "lancedb"));
  const table = await connection.openTable("episodes");
  const inlineSources =
    "First source prose OUTCOME fp=one decision=send\nSecond source prose decision=hold";
  const standaloneSource = "OUTCOME fp=two";
  try {
    if (missingInputColumn) await table.dropColumns(["consolidation_embedding_input"]);
    await table.update({ where: "id = 'episodes-0'", values: { narrative: inlineSources } });
    await table.update({ where: "id = 'episodes-2'", values: { narrative: standaloneSource } });
    for (const index of [1, 3, 4]) {
      await table.update({
        where: `id = 'episodes-${index}'`,
        values: {
          episode_kind: "consolidation_version",
          narrative: preserveProtectedEpisodeTokenLines(`Synthesis ${index}`, [
            index === 3 ? standaloneSource : inlineSources,
          ]),
          lineage_derived_from: JSON.stringify([index === 3 ? "episodes-2" : "episodes-0"]),
        },
      });
    }
  } finally {
    table.close();
    connection.close();
  }
  return bank;
}

describe("storage-only embedding migration", () => {
  it.each([false, true])(
    "labels only ambiguous legacy inputs with their vectors and verifies them (missing column=%s)",
    async (missingColumn) => {
      const bank = await mixedLegacyConsolidations(missingColumn);
      const source = await connect(join(bank.tenantDir, "lancedb"));
      const sourceTable = await source.openTable("episodes");
      const before = await sourceTable.query().toArray();
      sourceTable.close();
      source.close();
      const options = { ...bank.options, legacyConsolidationInput: "longest-prefix" as const };
      expect(await migrateTenant({ ...options, dryRun: true })).toMatchObject({
        complete: true,
        embedding_text_unrecoverable: { count: 2 },
        legacy_consolidation_resolutions: { count: 2, ids: ["episodes-1", "episodes-4"] },
      });
      expect(bank.client.embedBatch).not.toHaveBeenCalled();
      expect(existsSync(join(bank.tenantDir, MIGRATION_JOURNAL))).toBe(false);
      await expect(migrateTenant(bank.options, { client: bank.client })).rejects.toMatchObject({
        code: "EMBEDDING_TEXT_UNRECOVERABLE",
        report: { embedding_text_unrecoverable: { ids: ["episodes-1", "episodes-4"] } },
      });
      const report = await migrateTenant({ ...options, resume: true }, { client: bank.client });
      expect(report).toMatchObject({
        complete: true,
        embedding_text_unrecoverable: { count: 0 },
        legacy_consolidation_resolutions: { count: 2, ids: ["episodes-1", "episodes-4"] },
      });
      const target = await connect(join(bank.tenantDir, "lancedb"));
      const table = await target.openTable("episodes");
      try {
        const after = await table.query().toArray();
        for (const row of after) {
          const original = before.find((original) => original.id === row.id)!;
          const {
            embedding: _oldVector,
            consolidation_embedding_input: _oldInput,
            ...oldFields
          } = original;
          const { embedding: vector, consolidation_embedding_input: recorded, ...newFields } = row;
          expect(newFields).toEqual(oldFields);
          expect(vector.length).toBe(2);
          if (row.id === "episodes-1" || row.id === "episodes-4") {
            const input = JSON.parse(recorded);
            expect(input).toEqual({
              synthesized_narrative: original.narrative,
              protected_source_lines: collectProtectedEpisodeTokenLines([
                String(before.find((row) => row.id === "episodes-0")!.narrative),
              ]),
            });
            expect(
              preserveProtectedEpisodeTokenLines(
                input.synthesized_narrative,
                input.protected_source_lines,
              ),
            ).toBe(row.narrative);
            const text = buildConsolidationEpisodeEmbeddingText({
              title: String(row.title),
              synthesizedNarrative: input.synthesized_narrative,
              protectedSourceTexts: input.protected_source_lines,
              tags: JSON.parse(row.tags),
              participants: JSON.parse(row.participants),
            });
            expect(migrationRowText("episodes", row)).toBe(text);
            expect(bank.client.embedBatch.mock.calls.flatMap(([texts]) => texts)).toContain(text);
            expect(Array.from(vector)).toEqual(Array.from(await bank.client.embed(text)));
          } else expect(recorded).toBeNull();
        }
        // Unambiguous rows retain exact reconstruction without labelling their input.
        expect(bank.client.embedBatch.mock.calls.flatMap(([texts]) => texts)).toContain(
          "Episode 3\nSynthesis 3\nOUTCOME fp=two\ntag étiquette\nparticipant",
        );
      } finally {
        table.close();
        target.close();
      }
      const verification = spawnSync(
        process.execPath,
        [
          "--import",
          "tsx",
          "scripts/migrate-embeddings.ts",
          "--data-root",
          bank.root,
          "--tenant",
          "team-agent-ai",
          "--target-model",
          bank.options.target.model,
          "--target-dims",
          "2",
          "--verify-only",
        ],
        { encoding: "utf8" },
      );
      expect(verification.status, verification.stderr).toBe(0);
      expect(JSON.parse(verification.stdout.trim()).report).toMatchObject({
        complete: true,
        legacy_consolidation_resolutions: { count: 2 },
      });
      expect(
        await migrateTenant({
          ...bank.options,
          sourceModel: bank.options.target.model,
          dryRun: true,
        }),
      ).toMatchObject({
        complete: true,
        embedding_text_unrecoverable: { count: 0, ids: [] },
        legacy_consolidation_resolutions: { count: 0 },
      });

      const tampered = await connect(join(bank.tenantDir, "lancedb"));
      const episodes = await tampered.openTable("episodes");
      await episodes.update({
        where: "id = 'episodes-1'",
        values: {
          consolidation_embedding_input: JSON.stringify({
            synthesized_narrative: "Wrong prose",
            protected_source_lines: [],
          }),
        },
      });
      episodes.close();
      tampered.close();
      await expect(migrateTenant({ ...bank.options, verifyOnly: true })).rejects.toMatchObject({
        code: "EMBEDDING_TEXT_UNRECOVERABLE",
      });
    },
    60_000,
  );

  it.each(["checkpointed", "uncheckpointed"])(
    "resumes labelled vectors after a %s interruption without changing the recorded policy",
    async (boundary) => {
      const bank = await mixedLegacyConsolidations(true);
      let batches = 0;
      const stop = () => {
        batches += 1;
        if (
          bank.client.embedBatch.mock.calls
            .at(-1)![0]
            .some((text) => text.startsWith("Episode 1\n"))
        )
          throw new Error("interrupted after labelled vector");
      };
      await expect(
        migrateTenant(
          { ...bank.options, legacyConsolidationInput: "longest-prefix" },
          {
            client: bank.client,
            ...(boundary === "checkpointed" ? { afterBatch: stop } : { afterTableWrite: stop }),
          },
        ),
      ).rejects.toThrow("interrupted after labelled vector");
      const journal = readJsonFile<{
        consolidation_resolutions: { episode_id: string }[];
        legacy_consolidation_input: string;
      }>(join(bank.tenantDir, MIGRATION_JOURNAL))!;
      expect(journal).toMatchObject({
        legacy_consolidation_input: "longest-prefix",
        consolidation_resolutions: [
          expect.objectContaining({ episode_id: "episodes-1" }),
          expect.objectContaining({ episode_id: "episodes-4" }),
        ],
      });
      const staging = await connect(join(bank.tenantDir, "lancedb.staging-1"));
      const table = await staging.openTable("episodes");
      const row = (await table.query().where("id = 'episodes-1'").toArray())[0]!;
      expect(row.embedding.length).toBe(2);
      expect(JSON.parse(row.consolidation_embedding_input).synthesized_narrative).toBe(
        row.narrative,
      );
      table.close();
      staging.close();
      const committed = parseJsonLines(join(bank.tenantDir, ".embedding-migration-g1.jsonl"))
        .flatMap((batch) => (batch as { rows: { id: string }[] }).rows)
        .map((row) => row.id);
      expect(committed.includes("episodes-1")).toBe(boundary === "checkpointed");
      const beforeResume = bank.client.embedBatch.mock.calls.length;
      // Omitted flags on resume retain the explicit policy and choices in the journal.
      expect(
        await migrateTenant({ ...bank.options, resume: true }, { client: bank.client }),
      ).toMatchObject({ complete: true, legacy_consolidation_resolutions: { count: 2 } });
      expect(bank.client.embedBatch.mock.calls.length - beforeResume).toBe(
        35 - (boundary === "checkpointed" ? batches : batches - 1),
      );
      expect(await migrateTenant({ ...bank.options, verifyOnly: true })).toMatchObject({
        complete: true,
      });
    },
    60_000,
  );

  it.each(["previous", "live"] as const)(
    "recovers labelled rows after the %s cutover rename",
    async (boundary) => {
      const bank = await mixedLegacyConsolidations(true);
      await expect(
        migrateTenant(
          { ...bank.options, batchSize: 5, legacyConsolidationInput: "longest-prefix" },
          {
            client: bank.client,
            afterRename: (step) => {
              if (step === boundary) throw new Error("cutover interrupted");
            },
          },
        ),
      ).rejects.toThrow("cutover interrupted");
      const calls = bank.client.embedBatch.mock.calls.length;
      expect(
        await migrateTenant({ ...bank.options, resume: true }, { client: bank.client }),
      ).toMatchObject({
        complete: true,
        legacy_consolidation_resolutions: { ids: ["episodes-1", "episodes-4"] },
      });
      expect(bank.client.embedBatch).toHaveBeenCalledTimes(calls);
      expect(await migrateTenant({ ...bank.options, verifyOnly: true })).toMatchObject({
        complete: true,
      });
    },
    60_000,
  );

  it("skips completed strict migrations when an all-tenant resume opts in for remaining banks", async () => {
    const bank = await fixture(1);
    expect(await migrateTenant(bank.options, { client: bank.client })).toMatchObject({
      complete: true,
    });
    const calls = bank.client.embedBatch.mock.calls.length;
    expect(
      await migrateTenant(
        { ...bank.options, resume: true, legacyConsolidationInput: "longest-prefix" },
        { client: bank.client },
      ),
    ).toMatchObject({
      complete: true,
      already_complete: true,
      legacy_consolidation_resolutions: { count: 0 },
    });
    expect(bank.client.embedBatch).toHaveBeenCalledTimes(calls);
  });

  it("rejects a changed source after recording fallback choices", async () => {
    const bank = await mixedLegacyConsolidations(true);
    await expect(
      migrateTenant(
        { ...bank.options, legacyConsolidationInput: "longest-prefix" },
        {
          client: bank.client,
          afterBatch: () => {
            throw new Error("interrupted");
          },
        },
      ),
    ).rejects.toThrow("interrupted");
    const source = await connect(join(bank.tenantDir, "lancedb"));
    const table = await source.openTable("episodes");
    await table.update({
      where: "id = 'episodes-1'",
      values: {
        narrative:
          "Changed synthesis\nFirst source prose OUTCOME fp=one decision=send\nSecond source prose decision=hold",
      },
    });
    table.close();
    source.close();
    const calls = bank.client.embedBatch.mock.calls.length;
    await expect(
      migrateTenant({ ...bank.options, resume: true }, { client: bank.client }),
    ).rejects.toMatchObject({ code: "EMBEDDING_MIGRATION_SOURCE_CHANGED" });
    expect(bank.client.embedBatch).toHaveBeenCalledTimes(calls);
  });

  it("keeps missing or inconsistent inputs blocked even with the longest-prefix opt-in", async () => {
    const bank = await mixedLegacyConsolidations();
    const connection = await connect(join(bank.tenantDir, "lancedb"));
    const table = await connection.openTable("episodes");
    await table.update({
      where: "id = 'episodes-3'",
      values: {
        consolidation_embedding_input:
          '{"synthesized_narrative":"Incorrect","protected_source_lines":[]}',
      },
    });
    table.close();
    connection.close();
    await expect(
      migrateTenant(
        { ...bank.options, legacyConsolidationInput: "longest-prefix" },
        { client: bank.client },
      ),
    ).rejects.toMatchObject({
      code: "EMBEDDING_TEXT_UNRECOVERABLE",
      report: {
        embedding_text_unrecoverable: { ids: ["episodes-1", "episodes-3", "episodes-4"] },
        blocked_embedding_inputs: { ids: ["episodes-3"] },
        legacy_consolidation_resolutions: { ids: ["episodes-1", "episodes-4"] },
      },
    });
    expect(bank.client.embedBatch).not.toHaveBeenCalled();
    expect(existsSync(bank.options.backupDir)).toBe(false);
  });

  it("opens a migrated bank through Borg and renders the persisted fallback with the runtime recipe", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-consolidation-runtime-"));
    cleanup.push(root);
    const tenantDir = join(root, "team-agent-ai");
    const sourceModel = "generative-apis/qwen3-embedding-8b";
    const sourceClient = new FakeEmbeddingClient(4, sourceModel);
    await (
      await Borg.open({
        dataDir: tenantDir,
        embeddingClient: sourceClient,
        llmClient: new FakeLLMClient(),
      })
    ).close();
    const db = openDatabase(join(tenantDir, "borg.db"));
    const connection = await connect(join(tenantDir, "lancedb"));
    const table = await connection.openTable("episodes");
    const repository = new EpisodicRepository({ table: new LanceDbTable(table), db });
    const raw = createEpisodeFixture({ narrative: "Source prose OUTCOME fp=legacy decision=send" });
    const consolidated = {
      ...createEpisodeFixture({
        narrative: preserveProtectedEpisodeTokenLines("Synthesized prose", [raw.narrative]),
        lineage: { derived_from: [raw.id], supersedes: [] },
      }),
      episode_kind: "consolidation_version" as const,
    };
    try {
      await repository.createEpisode(raw);
      await repository.createEpisode(consolidated);
    } finally {
      table.close();
      connection.close();
      db.close();
    }
    const client = new FakeEmbeddingClient(2, "scw/bge-m3");
    const options = {
      tenantDir,
      backupDir: join(root, "backups"),
      sourceModel,
      target: client.profile,
      legacyConsolidationInput: "longest-prefix" as const,
    };
    expect(await migrateTenant(options, { client })).toMatchObject({
      complete: true,
      legacy_consolidation_resolutions: { ids: [consolidated.id] },
    });
    expect(
      await migrateTenant({ ...options, legacyConsolidationInput: undefined, verifyOnly: true }),
    ).toMatchObject({ complete: true });
    const reopened = await Borg.open({
      dataDir: tenantDir,
      embeddingClient: client,
      llmClient: new FakeLLMClient(),
    });
    try {
      const stored = await reopened.episodic.inspect(consolidated.id);
      expect(stored?.consolidation_embedding_input).toEqual({
        synthesized_narrative: consolidated.narrative,
        protected_source_lines: [raw.narrative],
      });
      expect(buildEpisodeEmbeddingText(stored!)).toBe(
        buildConsolidationEpisodeEmbeddingText({
          title: consolidated.title,
          synthesizedNarrative: consolidated.narrative,
          protectedSourceTexts: [raw.narrative],
          tags: consolidated.tags,
          participants: consolidated.participants,
        }),
      );
      expect(Array.from(stored!.embedding)).toEqual(
        Array.from(await client.embed(buildEpisodeEmbeddingText(stored!))),
      );
    } finally {
      await reopened.close();
    }
  }, 60_000);

  it("reports every ambiguous legacy input in a mixed dry-run inventory and exits zero across tenants", async () => {
    const bank = await mixedLegacyConsolidations();
    const report = await migrateTenant({ ...bank.options, dryRun: true });
    expect(report).toMatchObject({
      dry_run: true,
      complete: false,
      embedding_text_unrecoverable: {
        count: 2,
        ids: ["episodes-1", "episodes-4"],
        rows: [1, 4].map((index) => ({
          episode_id: `episodes-${index}`,
          reason: "Legacy consolidation embedding input is ambiguous",
          candidate_count: 3,
        })),
      },
    });
    const inventory = report.inventory as Awaited<ReturnType<typeof inventoryBank>>;
    expect(inventory.tables.map((table) => table.rows.length)).toEqual(Array(7).fill(5));
    expect(
      inventory.tables[0]!.rows.filter((row) => row.text_hash === null).map((row) => row.id),
    ).toEqual(["episodes-1", "episodes-4"]);
    cpSync(bank.tenantDir, join(bank.root, "team-agent-esb"), { recursive: true });
    const result = spawnSync(
      process.execPath,
      [
        "--import",
        "tsx",
        "scripts/migrate-embeddings.ts",
        "--data-root",
        bank.root,
        "--all-tenants",
        "--source-model",
        bank.options.sourceModel,
        "--target-model",
        bank.options.target.model,
        "--target-dims",
        "2",
        "--dry-run",
      ],
      { encoding: "utf8" },
    );
    expect(result.status, result.stderr).toBe(0);
    const reports = result.stdout
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line).report);
    expect(reports).toHaveLength(2);
    expect(reports.map((entry) => entry.embedding_text_unrecoverable.ids)).toEqual(
      Array(2).fill(["episodes-1", "episodes-4"]),
    );
    expect(existsSync(join(bank.tenantDir, MIGRATION_JOURNAL))).toBe(false);
    expect(existsSync(join(bank.tenantDir, EMBEDDING_FENCE_FILE))).toBe(false);
  });

  it("fails a real run only after reporting the full unrecoverable list, including missing raw sources", async () => {
    const bank = await mixedLegacyConsolidations();
    const connection = await connect(join(bank.tenantDir, "lancedb"));
    const table = await connection.openTable("episodes");
    await table.update({
      where: "id = 'episodes-3'",
      values: { lineage_derived_from: '["missing-source"]' },
    });
    table.close();
    connection.close();
    const result = spawnSync(
      process.execPath,
      [
        "--import",
        "tsx",
        "scripts/migrate-embeddings.ts",
        "--data-root",
        bank.root,
        "--tenant",
        "team-agent-ai",
        "--source-model",
        bank.options.sourceModel,
        "--target-model",
        bank.options.target.model,
        "--target-dims",
        "2",
      ],
      { encoding: "utf8", env: { ...process.env, LLM_API_KEY: "test-key" } },
    );
    expect(result.status).toBe(1);
    const failure = JSON.parse(result.stderr.trim().split("\n").at(-1)!);
    expect(failure).toMatchObject({
      code: "EMBEDDING_TEXT_UNRECOVERABLE",
      report: {
        complete: false,
        embedding_text_unrecoverable: {
          count: 3,
          ids: ["episodes-1", "episodes-3", "episodes-4"],
          rows: expect.arrayContaining([
            expect.objectContaining({ episode_id: "episodes-3", candidate_count: 0 }),
          ]),
        },
      },
    });
    const events = result.stdout
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    expect(
      events
        .find((event) => event.phase === "inventory")
        .inventory.tables.map((entry: { rows: unknown[] }) => entry.rows.length),
    ).toEqual(Array(7).fill(5));
    expect(existsSync(bank.options.backupDir)).toBe(false);
    expect(existsSync(join(bank.tenantDir, MIGRATION_JOURNAL))).toBe(false);
    expect(existsSync(join(bank.tenantDir, "lancedb.staging-1"))).toBe(false);
    expect(existsSync(join(bank.tenantDir, EMBEDDING_FENCE_FILE))).toBe(true);
  });
  it("validates an injected migration client before fencing or backing up", async () => {
    const bank = await fixture(0);
    const delegate = new FakeEmbeddingClient(2);
    await expect(
      migrateTenant(bank.options, {
        client: {
          embed: delegate.embed.bind(delegate),
          embedBatch: delegate.embedBatch.bind(delegate),
        },
      }),
    ).rejects.toMatchObject({ code: "EMBEDDING_CLIENT_PROFILE_REQUIRED" });
    await expect(
      migrateTenant(bank.options, { client: new FakeEmbeddingClient(2, "wrong-model") }),
    ).rejects.toMatchObject({ code: "EMBEDDING_PROFILE_MISMATCH" });
    expect(existsSync(join(bank.tenantDir, EMBEDDING_FENCE_FILE))).toBe(false);
    expect(existsSync(bank.options.backupDir)).toBe(false);
  });
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

  it("keeps capacity for a large final row instead of assuming average row sizes", async () => {
    const bank = await fixture(0);
    const inventory = await inventoryBank(bank.tenantDir, 2);
    const large = join(bank.tenantDir, "lancedb", "episodes.lance", "large-row-size-fixture");
    writeFileSync(large, "");
    truncateSync(large, 128 * 1024 * 1024);
    expect(() =>
      migrationHeadroom(
        bank.tenantDir,
        bank.options.backupDir,
        inventory,
        2,
        { backup: false, remainingRows: 1, remainingTables: ["episodes"] },
        () => ({ bsize: 1, bavail: 200 * 1024 * 1024 }),
      ),
    ).toThrow("Insufficient disk headroom");
  });

  it("checks backup capacity without a progress callback and uses the journal destination on resume", async () => {
    const bank = await fixture(0);
    const full = vi.fn(() => ({ bavail: 0, bsize: 4096 }));
    await expect(
      migrateTenant(bank.options, { client: bank.client, diskSpace: full }),
    ).rejects.toMatchObject({ code: "EMBEDDING_MIGRATION_DISK_FULL" });
    expect(existsSync(bank.options.backupDir)).toBe(false);
    expect(bank.client.embedBatch).not.toHaveBeenCalled();
    const state = readJsonFile<{ backup: string }>(join(bank.tenantDir, MIGRATION_JOURNAL))!;
    mkdirSync(bank.options.backupDir, { recursive: true });
    const changedDestination = join(bank.root, "unused-backups");
    full.mockClear();
    await expect(
      migrateTenant(
        { ...bank.options, backupDir: changedDestination, resume: true },
        { client: bank.client, diskSpace: full },
      ),
    ).rejects.toMatchObject({ code: "EMBEDDING_MIGRATION_DISK_FULL" });
    expect(full.mock.calls).toContainEqual([bank.options.backupDir]);
    expect(existsSync(state.backup)).toBe(false);
    expect(existsSync(changedDestination)).toBe(false);
  });

  it("rechecks remaining staging capacity after interruption and retains its verified backup", async () => {
    const bank = await fixture(2);
    await expect(
      migrateTenant(bank.options, {
        client: bank.client,
        afterBatch: () => {
          throw new Error("interrupt");
        },
      }),
    ).rejects.toThrow("interrupt");
    const state = readJsonFile<{ backup: string }>(join(bank.tenantDir, MIGRATION_JOURNAL))!;
    const marker = readFileSync(join(state.backup, ".embedding-backup.json"), "utf8");
    bank.client.embedBatch.mockClear();
    await expect(
      migrateTenant(
        { ...bank.options, resume: true },
        { client: bank.client, diskSpace: () => ({ bavail: 0, bsize: 4096 }) },
      ),
    ).rejects.toMatchObject({ code: "EMBEDDING_MIGRATION_DISK_FULL" });
    expect(bank.client.embedBatch).not.toHaveBeenCalled();
    expect(readFileSync(join(state.backup, ".embedding-backup.json"), "utf8")).toBe(marker);
    const progress = vi.fn();
    await migrateTenant({ ...bank.options, resume: true }, { client: bank.client, progress });
    expect(
      progress.mock.calls.some(
        ([event]) =>
          event.phase === "headroom" && event.remaining_rows === 13 && event.backup_bytes === 0,
      ),
    ).toBe(true);
    expect(bank.client.embedBatch).toHaveBeenCalledTimes(13);
  });

  it("backs up a restored snapshot and completes a new migration after rollback", async () => {
    const bank = await fixture(1);
    const before = await inventoryBank(bank.tenantDir, 2);
    const first = await migrateTenant(bank.options, { client: bank.client });
    rmSync(bank.tenantDir, { recursive: true });
    cpSync(String(first.backup), bank.tenantDir, { recursive: true });
    expect(existsSync(join(bank.tenantDir, ".embedding-backup.json"))).toBe(true);
    expect(existsSync(join(bank.tenantDir, MIGRATION_JOURNAL))).toBe(false);
    const second = await migrateTenant(bank.options, { client: bank.client });
    expect(second.complete).toBe(true);
    await verifyTenantBackup(String(second.backup), before, 2);
    const manifest = readJsonFile<{ files: Record<string, string> }>(
      join(String(second.backup), ".embedding-backup.json"),
    )!;
    expect(manifest.files).not.toHaveProperty(".embedding-backup.json");
    expect(manifest.files).not.toHaveProperty(EMBEDDING_FENCE_FILE);
  });

  it.skipIf(process.getuid?.() === 0)(
    "all-tenants CLI exits nonzero before migrating when a candidate is unreadable",
    async () => {
      const bank = await fixture(0);
      const blocked = join(bank.root, "unreadable");
      mkdirSync(blocked);
      writeFileSync(join(blocked, "borg.db"), "fixture");
      chmodSync(blocked, 0);
      try {
        const child = spawnSync(
          process.execPath,
          [
            "--import",
            "tsx",
            "scripts/migrate-embeddings.ts",
            "--data-root",
            bank.root,
            "--all-tenants",
            "--source-model",
            "old",
            "--target-model",
            "new",
            "--target-dims",
            "2",
            "--dry-run",
          ],
          { encoding: "utf8", timeout: 15_000, env: process.env },
        );
        expect(child.status, child.stderr).toBe(1);
        expect(child.stderr).toContain("EMBEDDING_TENANT_DISCOVERY_FAILED");
        expect(child.stderr).toContain("unreadable");
        expect(child.stdout).not.toContain('"complete":true');
        expect(existsSync(join(bank.tenantDir, MIGRATION_JOURNAL))).toBe(false);
      } finally {
        chmodSync(blocked, 0o700);
      }
    },
  );

  it("refuses resume when source text changes after a committed batch", async () => {
    const bank = await fixture(1);
    await expect(
      migrateTenant(bank.options, {
        client: bank.client,
        afterBatch: () => {
          throw new Error("interrupt");
        },
      }),
    ).rejects.toThrow("interrupt");
    const connection = await connect(join(bank.tenantDir, "lancedb"));
    const table = await connection.openTable("episodes");
    await table.update({ values: { narrative: "Source text changed after checkpoint" } });
    table.close();
    connection.close();
    bank.client.embedBatch.mockClear();
    await expect(
      migrateTenant({ ...bank.options, resume: true }, { client: bank.client }),
    ).rejects.toMatchObject({ code: "EMBEDDING_MIGRATION_SOURCE_CHANGED" });
    expect(bank.client.embedBatch).not.toHaveBeenCalled();
    expect(existsSync(join(bank.tenantDir, EMBEDDING_FENCE_FILE))).toBe(true);
  });

  it("recovers legacy consolidation input from archived raw lineage without adding schema columns", async () => {
    const bank = await fixture(2);
    const connection = await connect(join(bank.tenantDir, "lancedb"));
    const table = await connection.openTable("episodes");
    await table.dropColumns(["consolidation_embedding_input"]);
    await table.update({
      where: "id = 'episodes-0'",
      values: { narrative: "Raw evidence\nOUTCOME fp=z\nOUTCOME fp=a" },
    });
    await table.update({
      where: "id = 'episodes-1'",
      values: {
        narrative: "Synthesis\nOUTCOME fp=a\nOUTCOME fp=z",
        lineage_derived_from: '["episodes-0"]',
      },
    });
    table.close();
    connection.close();
    const db = new DatabaseSync(join(bank.tenantDir, "borg.db"));
    db.exec("UPDATE episode_stats SET archived = 1 WHERE episode_id = 'episodes-0'");
    db.close();
    const before = await inventoryBank(bank.tenantDir, 2);
    const result = await migrateTenant(bank.options, { client: bank.client });
    expect(result.complete).toBe(true);
    expect(bank.client.embedBatch.mock.calls.flatMap(([texts]) => texts)).toContain(
      "Episode 1\nSynthesis\nOUTCOME fp=z\nOUTCOME fp=a\ntag étiquette\nparticipant",
    );
    const after = await inventoryBank(bank.tenantDir, 2);
    expect(after.tables.map((item) => item.rows)).toEqual(before.tables.map((item) => item.rows));
    expect(after.tables.map((item) => item.schema_hash)).toEqual(
      before.tables.map((item) => item.schema_hash),
    );
  });

  it.each(["backed_up", "previous", "live", "complete"])(
    "recovers durable state after abrupt process exit at %s",
    async (phase) => {
      const bank = await fixture(1);
      const child = spawnSync(
        process.execPath,
        [
          "--import",
          "tsx",
          "--input-type=module",
          "--eval",
          `
      import { migrateTenant } from './scripts/embedding-migration/migrate.ts';
      import { FakeEmbeddingClient } from './src/embeddings/index.ts';
      const options = JSON.parse(process.env.BORG_TEST_MIGRATION_OPTIONS);
      const stop = process.env.BORG_TEST_MIGRATION_STOP;
      await migrateTenant(options, {
        client: new FakeEmbeddingClient(options.target.dimensions, options.target.model),
        progress: event => { if (event.phase === stop) process.exit(99); },
        afterRename: step => { if (step === stop) process.exit(99); },
      });
    `,
        ],
        {
          encoding: "utf8",
          timeout: 15_000,
          env: {
            ...process.env,
            BORG_TEST_MIGRATION_OPTIONS: JSON.stringify(bank.options),
            BORG_TEST_MIGRATION_STOP: phase,
          },
        },
      );
      expect(child.status, child.stderr).toBe(99);
      expect(existsSync(join(bank.tenantDir, EMBEDDING_FENCE_FILE))).toBe(true);
      const result = await migrateTenant(
        { ...bank.options, resume: true },
        { client: bank.client },
      );
      expect(result.complete).toBe(true);
      expect(existsSync(join(bank.tenantDir, EMBEDDING_FENCE_FILE))).toBe(false);
      expect(await migrateTenant({ ...bank.options, verifyOnly: true })).toMatchObject({
        complete: true,
      });
    },
  );

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

  it("recovers foreign migration leases, renews during embedding, and excludes guards from backups", async () => {
    const bank = await fixture(1);
    const paths = [EMBEDDING_ACCESS_FILE, ".embedding-migration-owner.lock"].map((name) =>
      join(bank.tenantDir, name),
    );
    for (const path of paths) {
      writeFileSync(
        path,
        JSON.stringify({
          pid: 999_999,
          host: "previous-pod",
          timestamp: Date.now() - FILE_LOCK_STALE_MS - 1,
        }),
      );
    }
    vi.useFakeTimers({ toFake: ["Date", "setInterval", "clearInterval"] });
    const embed = bank.client.embedBatch.getMockImplementation()!;
    bank.client.embedBatch.mockImplementationOnce(async (texts) => {
      await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS * 3);
      for (const path of paths) {
        expect(readJsonFile<{ heartbeat: number }>(path)?.heartbeat).toBe(Date.now());
        expect(existsSync(`${path}${FILE_LOCK_GUARD_SUFFIX}-journal`)).toBe(true);
        await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
        // A different process must still see the kernel lock after backup. A
        // same-process SQLite connection alone cannot detect accidental fd-close
        // loss of a POSIX lock caused by copying a guard with ordinary fs APIs.
        const probe = spawnSync(
          process.execPath,
          [
            "--input-type=module",
            "-e",
            `
          import { DatabaseSync } from 'node:sqlite';
          const db = new DatabaseSync(process.argv[1]);
          try { db.exec('BEGIN IMMEDIATE'); process.exitCode = 1; }
          catch (error) { if (error.errcode !== 5) throw error; }
          finally { db.close(); }
        `,
            `${path}${FILE_LOCK_GUARD_SUFFIX}`,
          ],
          { encoding: "utf8", timeout: 5_000 },
        );
        expect(probe.status, probe.stderr).toBe(0);
      }
      return await embed(texts);
    });
    try {
      const report = await migrateTenant(bank.options, { client: bank.client });
      expect(report.complete).toBe(true);
      const manifest = readJsonFile<{ files: Record<string, string> }>(
        join(String(report.backup), ".embedding-backup.json"),
      )!;
      expect(
        Object.keys(manifest.files).some((name) => name.includes(FILE_LOCK_GUARD_SUFFIX)),
      ).toBe(false);
      for (const path of paths) expect(existsSync(path)).toBe(false);
    } finally {
      vi.useRealTimers();
    }
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
    expect(
      parseEmbeddingMigrationArgs([
        ...args,
        "--all-tenants",
        "--legacy-consolidation-input",
        "longest-prefix",
      ]),
    ).toMatchObject({ legacyConsolidationInput: "longest-prefix" });
    expect(() =>
      parseEmbeddingMigrationArgs([
        ...args,
        "--all-tenants",
        "--legacy-consolidation-input",
        "shortest-prefix",
      ]),
    ).toThrow();
  });
});
