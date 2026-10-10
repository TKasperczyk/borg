import { existsSync, mkdirSync, renameSync, unlinkSync } from "node:fs";
import { hostname } from "node:os";
import { basename, join } from "node:path";
import { connect } from "@lancedb/lancedb";
import { z } from "zod";
import type { EmbeddingClient } from "../../src/embeddings/index.js";
import {
  acquireEmbeddingBankAccess,
  assertEmbeddingProfilesMatch,
  bankEmbeddingProfileSchema,
  EMBEDDING_FENCE_FILE,
  EMBEDDING_PROFILE_FILE,
  embeddingFenceSchema,
  embeddingProfileSchema,
  EmbeddingBankError,
  readBankEmbeddingProfile,
  requireEmbeddingClientProfile,
  type BankEmbeddingProfile,
  type EmbeddingProfile,
} from "../../src/embeddings/bank-profile.js";
import { assertUsableEmbedding } from "../../src/embeddings/serialized.js";
import {
  consolidationEmbeddingInputSchema,
  preserveProtectedEpisodeTokenLines,
} from "../../src/memory/episodic/protected-lines.js";
import { readJsonFile, syncDirectory, writeJsonFileAtomic } from "../../src/util/atomic-write.js";
import { appendDurableJsonl, parseJsonLines } from "../../src/util/durable-jsonl.js";
import { withFileLock } from "../../src/stream/file-lock.js";
import { sleep } from "../../src/util/clock.js";
import { LanceDbTable } from "../../src/storage/lancedb/index.js";
import { quoteSqlString } from "../../src/storage/codecs.js";
import { fingerprintCanonicalValue } from "../../src/cognition/deliberation/request-fingerprint.js";
import {
  backupTenant,
  migrationHeadroom,
  verifyTenantBackup,
  type DiskSpaceReader,
} from "./backup.js";
import {
  inventoryBank,
  migrationRecords,
  migrationRows,
  migrationSchema,
  migrationSchemaHash,
  unrecoverableInputSchema,
  consolidationResolutionSchema,
  legacyConsolidationPolicySchema,
  planConsolidationInputs,
  VECTOR_TABLES,
  type BankInventory,
  type ConsolidationResolution,
} from "./inventory.js";

const rowIdentitySchema = z.object({
  id: z.string(),
  text_hash: z.string().nullable(),
  fields_hash: z.string(),
});
const inventorySchema = z.object({
  tables: z.array(
    z.object({
      name: z.enum(VECTOR_TABLES.map((table) => table.name)),
      present: z.boolean(),
      dimensions: z.number().nullable(),
      schema_hash: z.string(),
      rows: z.array(rowIdentitySchema),
      sql_only: z.array(z.string()),
      vector_only: z.array(z.string()),
      text_disagreements: z.array(z.string()),
    }),
  ),
  sqlite_counts: z.record(z.string(), z.number()),
  serialized_vectors: z.array(z.object({ location: z.string(), paths: z.array(z.string()) })),
  problems: z.array(z.string()),
  embedding_text_unrecoverable: z.array(unrecoverableInputSchema).optional(),
});
const journalSchema = z.object({
  version: z.literal(1),
  source: bankEmbeddingProfileSchema,
  target: bankEmbeddingProfileSchema,
  backup: z.string(),
  inventory: inventorySchema,
  target_inventory: inventorySchema.optional(),
  legacy_consolidation_input: legacyConsolidationPolicySchema.optional(),
  consolidation_resolutions: z.array(consolidationResolutionSchema).optional(),
  phase: z.enum(["inventoried", "backed_up", "embedding", "verified", "cutover", "complete"]),
  started_at: z.number(),
  updated_at: z.number(),
});
type Journal = z.infer<typeof journalSchema>;
const expectedInventory = (journal: Journal): BankInventory =>
  journal.target_inventory ?? journal.inventory;
const checkpointSchema = z.object({
  version: z.literal(1),
  generation: z.number(),
  target: embeddingProfileSchema,
  table: z.enum(VECTOR_TABLES.map((table) => table.name)),
  rows: z.array(rowIdentitySchema),
});
export const MIGRATION_JOURNAL = ".embedding-migration.json";

function readCheckpoints(tenantDir: string, journal: Journal) {
  const done = new Map<string, z.infer<typeof rowIdentitySchema>>();
  for (const record of parseJsonLines(
    join(tenantDir, `.embedding-migration-g${journal.target.generation}.jsonl`),
  )) {
    const batch = checkpointSchema.parse(record);
    assertEmbeddingProfilesMatch(batch.target, journal.target);
    if (batch.generation !== journal.target.generation)
      throw new EmbeddingBankError("Checkpoint generation mismatch", {
        code: "EMBEDDING_MIGRATION_CHECKPOINT_INVALID",
      });
    for (const row of batch.rows) done.set(`${batch.table}/${row.id}`, row);
  }
  return done;
}

async function remainingStagingRows(
  tenantDir: string,
  stagingDir: string,
  journal: Journal,
): Promise<{ remainingRows: number; remainingTables: string[] }> {
  const remaining = new Map(
    expectedInventory(journal).tables.map((table) => [table.name, table.rows.length]),
  );
  const result = () => ({
    remainingRows: [...remaining.values()].reduce((sum, count) => sum + count, 0),
    remainingTables: [...remaining.entries()]
      .filter(([, count]) => count > 0)
      .map(([name]) => name),
  });
  if (!existsSync(stagingDir)) return result();
  const checkpoints = readCheckpoints(tenantDir, journal);
  const connection = await connect(stagingDir);
  try {
    const names = await connection.tableNames();
    for (const expected of expectedInventory(journal).tables) {
      if (!names.includes(expected.name)) continue;
      const table = await connection.openTable(expected.name);
      try {
        const source = new Map(expected.rows.map((row) => [row.id, row]));
        const counted = new Set<string>();
        for await (const rows of migrationRows(table, 128, true)) {
          for (const row of rows) {
            const { embedding, ...fields } = row;
            const id = String(
              row[expected.name === "image_perception_embeddings" ? "payload_id" : "id"],
            );
            const checkpoint = checkpoints.get(`${expected.name}/${id}`);
            const original = source.get(id);
            assertUsableEmbedding(embedding as ArrayLike<number>, journal.target.dimensions);
            if (
              !counted.has(id) &&
              checkpoint &&
              original &&
              checkpoint.text_hash === original.text_hash &&
              checkpoint.fields_hash === original.fields_hash &&
              fingerprintCanonicalValue(fields).canonicalSha256 === original.fields_hash
            ) {
              counted.add(id);
              remaining.set(expected.name, remaining.get(expected.name)! - 1);
            }
          }
        }
      } finally {
        table.close();
      }
    }
  } finally {
    connection.close();
  }
  return result();
}
export type MigrationOptions = {
  tenantDir: string;
  backupDir: string;
  target: EmbeddingProfile;
  sourceModel?: string;
  dryRun?: boolean;
  resume?: boolean;
  verifyOnly?: boolean;
  batchSize?: number;
  concurrency?: number;
  /** Allow a CLI invocation to observe remote leases before it times out. */
  lockTimeoutMs?: number;
  legacyConsolidationInput?: z.infer<typeof legacyConsolidationPolicySchema>;
};
export type MigrationDependencies = {
  client?: EmbeddingClient;
  progress?: (event: Record<string, unknown>) => void;
  /** Fault injection for crash recovery tests; no CLI switches expose these. */
  afterBatch?: () => void | Promise<void>;
  afterTableWrite?: () => void | Promise<void>;
  afterRename?: (step: "previous" | "live") => void | Promise<void>;
  retryDelaysMs?: readonly number[];
  diskSpace?: DiskSpaceReader;
};

function inputReport(
  inventory: BankInventory,
  resolutions: readonly ConsolidationResolution[] = [],
) {
  const rows = inventory.embedding_text_unrecoverable ?? [];
  return {
    legacy_consolidation_resolutions: {
      count: resolutions.length,
      ids: resolutions.map((row) => row.episode_id),
      rows: resolutions.map(({ episode_id, candidate_count }) => ({
        episode_id,
        candidate_count,
        policy: "longest-prefix",
      })),
    },
    embedding_text_unrecoverable: {
      count: rows.length,
      ids: rows.map((row) => row.episode_id),
      rows,
    },
  };
}

export class MigrationInputBlockedError extends EmbeddingBankError {
  readonly report;
  constructor(
    inventory: BankInventory,
    source = inventory,
    resolutions: readonly ConsolidationResolution[] = [],
  ) {
    const report = inputReport(inventory);
    super(
      `Unrecoverable embedding input for ${report.embedding_text_unrecoverable.count} episodes: ${report.embedding_text_unrecoverable.ids.join(", ")}`,
      { code: "EMBEDDING_TEXT_UNRECOVERABLE" },
    );
    this.report = {
      complete: false,
      inventory: source,
      ...inputReport(source, resolutions),
      blocked_embedding_inputs: report.embedding_text_unrecoverable,
    };
  }
}

function requireComplete(
  inventory: BankInventory,
  source = inventory,
  resolutions: readonly ConsolidationResolution[] = [],
): void {
  if (inventory.embedding_text_unrecoverable?.length)
    throw new MigrationInputBlockedError(inventory, source, resolutions);
  if (inventory.problems.length)
    throw new EmbeddingBankError(inventory.problems.join("; "), {
      code: "EMBEDDING_MIGRATION_INCOMPLETE",
    });
}

function sourceProfile(
  tenantDir: string,
  inventory: BankInventory,
  sourceModel: string | undefined,
): BankEmbeddingProfile {
  const stored = readBankEmbeddingProfile(tenantDir);
  if (stored === undefined)
    throw new EmbeddingBankError(
      "Existing bank is missing embedding-profile.json; label a restored pre-profile backup with the label-source-profile subcommand before migrating",
      { code: "EMBEDDING_PROFILE_REQUIRED" },
    );
  const dimensions = new Set(
    inventory.tables.flatMap((table) => (table.dimensions === null ? [] : [table.dimensions])),
  );
  if (dimensions.size !== 1)
    throw new EmbeddingBankError("Source tables have mixed or unknown dimensions", {
      code: "EMBEDDING_MIGRATION_INCOMPLETE",
    });
  const dims = [...dimensions][0]!;
  assertEmbeddingProfilesMatch(stored, { model: sourceModel ?? stored.model, dimensions: dims });
  return stored;
}

export async function verifyMigratedBank(
  tenantDir: string,
  lanceDir: string,
  inventory: BankInventory,
  target: EmbeddingProfile,
  resolutions: readonly ConsolidationResolution[] = [],
): Promise<{ tables: Record<string, number>; sanity_queries: number }> {
  const actual = await inventoryBank(tenantDir, target.dimensions, 256, lanceDir);
  requireComplete(actual);
  if (
    fingerprintCanonicalValue(actual.sqlite_counts).canonicalSha256 !==
    fingerprintCanonicalValue(inventory.sqlite_counts).canonicalSha256
  )
    throw new EmbeddingBankError("SQLite row counts changed during migration", {
      code: "EMBEDDING_MIGRATION_VERIFY_FAILED",
    });
  const connection = await connect(lanceDir);
  const labelledEpisodes = new Set(resolutions.map((resolution) => resolution.episode_id));
  const report = { tables: {} as Record<string, number>, sanity_queries: 0 };
  try {
    for (const expected of inventory.tables) {
      const observed = actual.tables.find((table) => table.name === expected.name)!;
      if (
        !observed.present ||
        observed.dimensions !== target.dimensions ||
        observed.schema_hash !== expected.schema_hash ||
        fingerprintCanonicalValue(observed.rows).canonicalSha256 !==
          fingerprintCanonicalValue(expected.rows).canonicalSha256
      )
        throw new EmbeddingBankError(
          `Exact id, field or schema verification failed: ${expected.name}`,
          { code: "EMBEDDING_MIGRATION_VERIFY_FAILED" },
        );
      const table = await connection.openTable(expected.name);
      try {
        let probes = 0;
        for await (const rows of migrationRows(table, 128, true))
          for (const row of rows) {
            if (expected.name === "episodes" && labelledEpisodes.has(String(row.id))) {
              const input = consolidationEmbeddingInputSchema.parse(
                JSON.parse(String(row.consolidation_embedding_input)),
              );
              if (
                preserveProtectedEpisodeTokenLines(
                  input.synthesized_narrative,
                  input.protected_source_lines,
                ) !== row.narrative
              )
                throw new EmbeddingBankError(`Fallback input does not preserve episode ${row.id}`, {
                  code: "EMBEDDING_MIGRATION_VERIFY_FAILED",
                });
            }
            const vector = row.embedding as ArrayLike<number>;
            assertUsableEmbedding(vector, target.dimensions);
            if (probes < 3) {
              const neighbors = await table
                .vectorSearch(Array.from(vector))
                .distanceType("cosine")
                .limit(3)
                .toArray();
              if (
                neighbors.length === 0 ||
                !Number.isFinite(neighbors[0]?._distance) ||
                Number(neighbors[0]?._distance) > 0.001
              )
                throw new EmbeddingBankError(
                  `Nearest-neighbour sanity query failed: ${expected.name}`,
                  { code: "EMBEDDING_MIGRATION_VERIFY_FAILED" },
                );
              probes += 1;
              report.sanity_queries += 1;
            }
          }
        report.tables[expected.name] = observed.rows.length;
      } finally {
        table.close();
      }
    }
    return report;
  } finally {
    connection.close();
  }
}

async function embedStaging(
  options: MigrationOptions,
  deps: MigrationDependencies,
  journal: Journal,
  sourceDir: string,
  stagingDir: string,
): Promise<void> {
  const client = deps.client;
  if (!client)
    throw new EmbeddingBankError("An embedding client is required to migrate", {
      code: "EMBEDDING_MIGRATION_CLIENT_REQUIRED",
    });
  assertEmbeddingProfilesMatch(journal.target, requireEmbeddingClientProfile(client));
  const checkpointPath = join(
    options.tenantDir,
    `.embedding-migration-g${journal.target.generation}.jsonl`,
  );
  const done = readCheckpoints(options.tenantDir, journal);
  const source = await connect(sourceDir);
  const staging = await connect(stagingDir);
  try {
    const existing = await staging.tableNames();
    const sourceNames = await source.tableNames();
    for (const definition of VECTOR_TABLES) {
      const original = sourceNames.includes(definition.name)
        ? await source.openTable(definition.name)
        : undefined;
      const expected = expectedInventory(journal).tables.find(
        (table) => table.name === definition.name,
      )!;
      const targetSchema = migrationSchema(
        original ? await original.schema() : definition.schema(journal.target.dimensions),
        journal.target.dimensions,
        definition.name === "episodes" && (journal.consolidation_resolutions?.length ?? 0) > 0,
      );
      const destination = existing.includes(definition.name)
        ? await staging.openTable(definition.name)
        : await staging.createEmptyTable(definition.name, targetSchema);
      try {
        if (migrationSchemaHash(await destination.schema()) !== expected.schema_hash)
          throw new EmbeddingBankError(`Staging schema mismatch: ${definition.name}`, {
            code: "EMBEDDING_MIGRATION_CHECKPOINT_INVALID",
          });
        // A checkpoint is only trusted if its committed staging row still exists
        // with the recorded text, fields and a valid target vector.
        const staged = new Map<string, string | null>();
        for await (const records of migrationRecords(
          destination,
          definition.name,
          128,
          true,
          original,
        ))
          for (const { row, identity, unrecoverable } of records) {
            if (unrecoverable)
              throw new MigrationInputBlockedError({
                ...journal.inventory,
                embedding_text_unrecoverable: [unrecoverable],
              });
            assertUsableEmbedding(row.embedding as ArrayLike<number>, journal.target.dimensions);
            const committed = done.get(`${definition.name}/${identity.id}`);
            if (
              committed &&
              fingerprintCanonicalValue(identity).canonicalSha256 !==
                fingerprintCanonicalValue(committed).canonicalSha256
            )
              throw new EmbeddingBankError(
                `Staging checkpoint content mismatch: ${definition.name}/${identity.id}`,
                { code: "EMBEDDING_MIGRATION_CHECKPOINT_INVALID" },
              );
            staged.set(identity.id, identity.text_hash);
          }
        let completed = 0;
        const wrapper = new LanceDbTable(destination);
        const iterator = original
          ? migrationRecords(
              original,
              definition.name,
              options.batchSize ?? 32,
              false,
              original,
              journal.consolidation_resolutions,
            )[Symbol.asyncIterator]()
          : undefined;
        let stopped = false;
        // Bounded worker pool; Lance commits and checkpoint appends are serialized.
        let commitQueue = Promise.resolve();
        async function worker(): Promise<void> {
          while (!stopped && iterator) {
            const next = await iterator.next();
            if (next.done) return;
            for (const record of next.value)
              if (record.unrecoverable)
                throw new MigrationInputBlockedError({
                  ...journal.inventory,
                  embedding_text_unrecoverable: [record.unrecoverable],
                });
            const pending = next.value.filter(({ identity }) => {
              const committed = done.get(`${definition.name}/${identity.id}`);
              if (
                committed &&
                committed.text_hash === identity.text_hash &&
                committed.fields_hash === identity.fields_hash &&
                staged.get(identity.id) === identity.text_hash
              ) {
                completed += 1;
                return false;
              }
              return true;
            });
            if (pending.length === 0) continue;
            const delays = deps.retryDelaysMs ?? [1000, 4000, 10_000];
            let vectors: Float32Array[] | undefined;
            for (let attempt = 0; ; attempt += 1) {
              try {
                vectors = await client!.embedBatch(pending.map((record) => record.text!));
                if (vectors.length !== pending.length)
                  throw new Error("Embedding batch response count mismatch");
                for (const vector of vectors)
                  assertUsableEmbedding(vector, journal.target.dimensions);
                break;
              } catch (error) {
                if (attempt >= delays.length) throw error;
                deps.progress?.({ phase: "retry", table: definition.name, attempt: attempt + 1 });
                await sleep(delays[attempt]!);
              }
            }
            const embeddings = vectors;
            const commit = async () => {
              // Arrow 18 serializes an all-null Boolean with no values bitmap;
              // Lance rejects that IPC. Only in the unserved staging table,
              // write a Boolean value then restore NULL with Lance SQL before
              // checkpointing. A crash here simply replays the uncommitted batch.
              const nullBooleans = targetSchema.fields.filter(
                (field) =>
                  field.type.toString() === "Bool" &&
                  pending.every(({ row }) => row[field.name] === null),
              );
              await wrapper.upsert(
                pending.map(({ row }, index) => ({
                  ...row,
                  ...Object.fromEntries(nullBooleans.map((field) => [field.name, false])),
                  embedding: Array.from(embeddings![index]!),
                })),
                { on: definition.key },
              );
              if (nullBooleans.length)
                await destination.update({
                  where: `${definition.key} IN (${pending.map(({ row }) => quoteSqlString(String(row[definition.key]))).join(", ")})`,
                  valuesSql: Object.fromEntries(
                    nullBooleans.map((field) => [field.name, "CAST(NULL AS BOOLEAN)"]),
                  ),
                });
              await deps.afterTableWrite?.();
              await appendDurableJsonl(checkpointPath, {
                version: 1,
                generation: journal.target.generation,
                target: { model: journal.target.model, dimensions: journal.target.dimensions },
                table: definition.name,
                rows: pending.map((record) => record.identity),
              });
              completed += pending.length;
              deps.progress?.({
                phase: "embedding",
                table: definition.name,
                completed,
                total: expected.rows.length,
              });
              await deps.afterBatch?.();
            };
            const priorCommit = commitQueue;
            let releaseCommit!: () => void;
            commitQueue = new Promise<void>((resolve) => {
              releaseCommit = resolve;
            });
            await priorCommit;
            try {
              await commit();
            } finally {
              releaseCommit();
            }
          }
        }
        const results = await Promise.allSettled(
          Array.from({ length: options.concurrency ?? 2 }, async () => {
            try {
              await worker();
            } catch (error) {
              stopped = true;
              throw error;
            }
          }),
        );
        const failed = results.find((result) => result.status === "rejected");
        if (failed?.status === "rejected") throw failed.reason;
        deps.progress?.({
          phase: "table_complete",
          table: definition.name,
          completed,
          total: expected.rows.length,
        });
      } finally {
        destination.close();
        original?.close();
      }
    }
  } finally {
    source.close();
    staging.close();
  }
}

export async function migrateTenant(
  options: MigrationOptions,
  deps: MigrationDependencies = {},
): Promise<Record<string, unknown>> {
  const target = embeddingProfileSchema.parse(options.target);
  const policy = legacyConsolidationPolicySchema.optional().parse(options.legacyConsolidationInput);
  if (!options.dryRun && !options.verifyOnly && deps.client)
    assertEmbeddingProfilesMatch(target, requireEmbeddingClientProfile(deps.client));
  const tenantDir = options.tenantDir;
  const live = join(tenantDir, "lancedb");
  const journalPath = join(tenantDir, MIGRATION_JOURNAL);
  const fencePath = join(tenantDir, EMBEDDING_FENCE_FILE);
  if (options.dryRun) {
    const inventory = await inventoryBank(tenantDir, target.dimensions);
    const { targetInventory, resolutions } = await planConsolidationInputs(
      tenantDir,
      target.dimensions,
      inventory,
      policy,
    );
    const source = sourceProfile(tenantDir, inventory, options.sourceModel);
    const headroom = migrationHeadroom(
      tenantDir,
      options.backupDir,
      inventory,
      target.dimensions,
      {},
      deps.diskSpace,
    );
    return {
      tenant: tenantDir,
      dry_run: true,
      complete:
        targetInventory.problems.length === 0 &&
        !targetInventory.embedding_text_unrecoverable?.length,
      provisional: true,
      source,
      target,
      inventory,
      ...inputReport(inventory, resolutions),
      headroom,
    };
  }
  // The primitive renews this owner lock throughout backup, embedding,
  // verification and cutover, including migrations lasting many minutes.
  const lockOptions = {
    timeoutMs: options.lockTimeoutMs ?? 1000,
    // A long CLI wait does not need to open a guard every 20 milliseconds.
    retryDelayMs: 250,
  };
  return withFileLock(
    join(tenantDir, ".embedding-migration-owner.lock"),
    async () => {
      let journal =
        readJsonFile<unknown>(journalPath) === undefined
          ? undefined
          : journalSchema.parse(readJsonFile(journalPath));
      if (
        journal?.phase === "complete" &&
        !options.verifyOnly &&
        (journal.target.model !== target.model || journal.target.dimensions !== target.dimensions)
      ) {
        if (existsSync(fencePath))
          assertEmbeddingProfilesMatch(
            embeddingFenceSchema.parse(readJsonFile(fencePath)).target,
            target,
          );
        assertEmbeddingProfilesMatch(
          readBankEmbeddingProfile(tenantDir) ?? journal.source,
          journal.target,
        );
        writeJsonFileAtomic(
          join(tenantDir, `.embedding-migration-history-${journal.target.generation}.json`),
          journal,
          { mode: 0o600 },
        );
        journal = undefined;
      }
      if (journal) {
        assertEmbeddingProfilesMatch(journal.target, target);
        if (
          policy &&
          policy !== journal.legacy_consolidation_input &&
          journal.phase !== "complete" &&
          !options.verifyOnly
        )
          throw new EmbeddingBankError(
            "Cannot change the consolidation input policy of an existing migration",
            { code: "EMBEDDING_MIGRATION_CHECKPOINT_INVALID" },
          );
        if (options.sourceModel)
          assertEmbeddingProfilesMatch(journal.source, {
            model: options.sourceModel,
            dimensions: journal.source.dimensions,
          });
      }
      if (options.verifyOnly) {
        if (!journal)
          throw new EmbeddingBankError(
            "Full verification requires the migration journal and backup",
            { code: "EMBEDDING_MIGRATION_VERIFY_FAILED" },
          );
        const release = await acquireEmbeddingBankAccess(tenantDir, lockOptions);
        try {
          await verifyTenantBackup(journal.backup, journal.inventory, target.dimensions);
          const staging = join(tenantDir, `lancedb.staging-${journal.target.generation}`);
          const directory = existsSync(staging) ? staging : live;
          const verified = await verifyMigratedBank(
            tenantDir,
            directory,
            expectedInventory(journal),
            target,
            journal.consolidation_resolutions,
          );
          if (directory === live)
            assertEmbeddingProfilesMatch(
              readBankEmbeddingProfile(tenantDir) ?? journal.source,
              target,
            );
          return {
            tenant: tenantDir,
            verified,
            complete: directory === live && !existsSync(fencePath),
            phase: journal.phase,
            backup: journal.backup,
            serialized_vectors: journal.inventory.serialized_vectors,
            ...inputReport(expectedInventory(journal), journal.consolidation_resolutions),
          };
        } finally {
          await release();
        }
      }
      if (journal && !options.resume)
        throw new EmbeddingBankError(
          "Migration journal already exists; use --resume or --verify-only",
          { code: "EMBEDDING_MIGRATION_RESUME_REQUIRED" },
        );
      if (journal?.phase === "complete" && !existsSync(fencePath)) {
        assertEmbeddingProfilesMatch(readBankEmbeddingProfile(tenantDir) ?? journal.source, target);
        return {
          tenant: tenantDir,
          complete: true,
          already_complete: true,
          profile: journal.target,
          backup: journal.backup,
          ...inputReport(expectedInventory(journal), journal.consolidation_resolutions),
        };
      }
      const labelled = readBankEmbeddingProfile(tenantDir);
      if (!journal && labelled === undefined)
        throw new EmbeddingBankError(
          "Existing bank is missing embedding-profile.json; run the label-source-profile subcommand before migrating",
          { code: "EMBEDDING_PROFILE_REQUIRED" },
        );
      if (
        !journal &&
        labelled?.model === target.model &&
        labelled.dimensions === target.dimensions
      ) {
        throw new EmbeddingBankError("Bank already uses the target profile; no migration needed", {
          code: "EMBEDDING_MIGRATION_ALREADY_CURRENT",
        });
      }
      if (existsSync(fencePath)) {
        if (!options.resume)
          throw new EmbeddingBankError("Tenant is fenced; use --resume after draining it", {
            code: "EMBEDDING_MIGRATION_RESUME_REQUIRED",
          });
        assertEmbeddingProfilesMatch(
          embeddingFenceSchema.parse(readJsonFile(fencePath)).target,
          target,
        );
      } else {
        writeJsonFileAtomic(
          fencePath,
          { host: hostname(), pid: process.pid, started_at: Date.now(), target },
          { mode: 0o600 },
        );
        deps.progress?.({ phase: "fenced", tenant: tenantDir, drain_route: "/memory/admin/evict" });
      }
      const release = await acquireEmbeddingBankAccess(tenantDir, lockOptions);
      try {
        if (!journal) {
          const inventory = await inventoryBank(tenantDir, target.dimensions);
          const { targetInventory, resolutions } = await planConsolidationInputs(
            tenantDir,
            target.dimensions,
            inventory,
            policy,
          );
          deps.progress?.({
            phase: "inventory",
            inventory,
            ...inputReport(inventory, resolutions),
          });
          requireComplete(targetInventory, inventory, resolutions);
          const source = sourceProfile(tenantDir, inventory, options.sourceModel);
          if (source.model === target.model && source.dimensions === target.dimensions)
            throw new EmbeddingBankError("Bank already uses the target profile", {
              code: "EMBEDDING_MIGRATION_ALREADY_CURRENT",
            });
          const now = Date.now();
          const migrated: BankEmbeddingProfile = {
            ...target,
            version: 1,
            generation: source.generation + 1,
            created_at: source.created_at,
            updated_at: now,
            migrated_from: {
              model: source.model,
              dimensions: source.dimensions,
              generation: source.generation,
            },
          };
          journal = {
            version: 1,
            source,
            target: migrated,
            inventory,
            ...(policy ? { legacy_consolidation_input: policy } : {}),
            ...(resolutions.length
              ? { target_inventory: targetInventory, consolidation_resolutions: resolutions }
              : {}),
            backup: join(
              options.backupDir,
              basename(tenantDir),
              `generation-${migrated.generation}-${now}`,
            ),
            phase: "inventoried",
            started_at: now,
            updated_at: now,
          };
          writeJsonFileAtomic(journalPath, journal, { mode: 0o600 });
        }
        const state = journal;
        const staging = join(tenantDir, `lancedb.staging-${state.target.generation}`);
        const previous = join(tenantDir, `lancedb.prev-${state.target.generation}`);
        function advance(phase: Journal["phase"]): void {
          state.phase = phase;
          state.updated_at = Date.now();
          writeJsonFileAtomic(journalPath, state, { mode: 0o600 });
          deps.progress?.({ phase, tenant: tenantDir });
        }
        const original = existsSync(previous) ? previous : live;
        const currentSource = await inventoryBank(tenantDir, target.dimensions, 256, original);
        if (
          fingerprintCanonicalValue(currentSource).canonicalSha256 !==
          fingerprintCanonicalValue(state.inventory).canonicalSha256
        )
          throw new EmbeddingBankError(
            "Source bank changed since inventory; keep fence and investigate",
            { code: "EMBEDDING_MIGRATION_SOURCE_CHANGED" },
          );
        if (state.phase === "inventoried") {
          await backupTenant(
            tenantDir,
            state.backup,
            state.inventory,
            target.dimensions,
            deps.diskSpace,
          );
          advance("backed_up");
        } else await verifyTenantBackup(state.backup, state.inventory, target.dimensions);
        if (state.phase === "backed_up" || state.phase === "embedding") {
          if (existsSync(previous))
            throw new EmbeddingBankError("Unexpected previous directory before verification", {
              code: "EMBEDDING_MIGRATION_RECOVERY_REQUIRED",
            });
          const remaining = await remainingStagingRows(tenantDir, staging, state);
          const headroom = migrationHeadroom(
            tenantDir,
            state.backup,
            state.inventory,
            target.dimensions,
            { backup: false, ...remaining },
            deps.diskSpace,
          );
          deps.progress?.({
            phase: "headroom",
            remaining_rows: remaining.remainingRows,
            ...headroom,
          });
          mkdirSync(staging, { recursive: true, mode: 0o700 });
          advance("embedding");
          await embedStaging(options, deps, state, live, staging);
          await verifyMigratedBank(
            tenantDir,
            staging,
            expectedInventory(state),
            target,
            state.consolidation_resolutions,
          );
          advance("verified");
        }
        // Fence + durable intent make the two filesystem renames one service-level
        // cutover. Inspect paths on resume to cover a crash before journal updates.
        if (state.phase === "verified") advance("cutover");
        if (state.phase === "cutover") {
          if (!existsSync(previous)) {
            if (!existsSync(live) || !existsSync(staging))
              throw new EmbeddingBankError("Cutover paths are incomplete", {
                code: "EMBEDDING_MIGRATION_RECOVERY_REQUIRED",
              });
            await verifyMigratedBank(
              tenantDir,
              staging,
              expectedInventory(state),
              target,
              state.consolidation_resolutions,
            );
            renameSync(live, previous);
            syncDirectory(tenantDir);
            await deps.afterRename?.("previous");
          }
          if (existsSync(staging)) {
            if (existsSync(live))
              throw new EmbeddingBankError(
                "Both live and staging exist after the previous rename",
                { code: "EMBEDDING_MIGRATION_RECOVERY_REQUIRED" },
              );
            await verifyMigratedBank(
              tenantDir,
              staging,
              expectedInventory(state),
              target,
              state.consolidation_resolutions,
            );
            renameSync(staging, live);
            syncDirectory(tenantDir);
            await deps.afterRename?.("live");
          }
          await verifyMigratedBank(
            tenantDir,
            live,
            expectedInventory(state),
            target,
            state.consolidation_resolutions,
          );
          state.target.updated_at = Date.now();
          writeJsonFileAtomic(join(tenantDir, EMBEDDING_PROFILE_FILE), state.target, {
            mode: 0o600,
          });
          advance("complete");
        }
        assertEmbeddingProfilesMatch(readBankEmbeddingProfile(tenantDir) ?? state.source, target);
        const verified = await verifyMigratedBank(
          tenantDir,
          live,
          expectedInventory(state),
          target,
          state.consolidation_resolutions,
        );
        const report = {
          tenant: tenantDir,
          complete: true,
          profile: state.target,
          backup: state.backup,
          previous,
          verified,
          serialized_vectors: state.inventory.serialized_vectors,
          ...inputReport(expectedInventory(state), state.consolidation_resolutions),
          started_at: state.started_at,
          completed_at: Date.now(),
        };
        writeJsonFileAtomic(
          join(tenantDir, `.embedding-migration-report-${state.target.generation}.json`),
          report,
          { mode: 0o600 },
        );
        unlinkSync(fencePath);
        syncDirectory(tenantDir);
        return report;
      } finally {
        await release();
      }
    },
    lockOptions,
  );
}
