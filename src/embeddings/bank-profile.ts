import { existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { connect } from "@lancedb/lancedb";
import { DataType, Precision } from "apache-arrow";
import { z } from "zod";

import { readJsonFile, writeJsonFileAtomic } from "../util/atomic-write.js";
import { StorageError } from "../util/errors.js";
import { acquireFileLockLease } from "../stream/file-lock.js";

export const EMBEDDING_PROFILE_FILE = "embedding-profile.json";
export const EMBEDDING_FENCE_FILE = "embedding-migration.lock";
export const EMBEDDING_ACCESS_FILE = "embedding-bank-access.lock";
export const VECTOR_TABLE_NAMES = [
  "action_records",
  "episodes",
  "image_perception_embeddings",
  "observed_events",
  "open_questions",
  "semantic_nodes",
  "skills",
] as const;

export const embeddingProfileSchema = z.object({
  model: z.string().trim().min(1),
  dimensions: z.number().int().positive(),
});
export type EmbeddingProfile = z.infer<typeof embeddingProfileSchema>;
const sourceProfileSchema = embeddingProfileSchema.extend({
  generation: z.number().int().nonnegative(),
});
export const bankEmbeddingProfileSchema = sourceProfileSchema.extend({
  version: z.literal(1),
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
  migrated_from: sourceProfileSchema.nullable(),
});
export type BankEmbeddingProfile = z.infer<typeof bankEmbeddingProfileSchema>;
export const embeddingFenceSchema = z.object({
  host: z.string().min(1),
  pid: z.number().int().positive(),
  started_at: z.number().finite(),
  target: embeddingProfileSchema,
});

export class EmbeddingBankError extends StorageError {}

export function requireEmbeddingClientProfile(client: {
  readonly profile?: unknown;
}): EmbeddingProfile {
  const parsed = embeddingProfileSchema.safeParse(client.profile);
  if (!parsed.success) {
    throw new EmbeddingBankError("Embedding client must declare its model and dimensions", {
      code: "EMBEDDING_CLIENT_PROFILE_REQUIRED",
      cause: parsed.error,
    });
  }
  return parsed.data;
}

export function readBankEmbeddingProfile(dataDir: string): BankEmbeddingProfile | undefined {
  try {
    const value = readJsonFile<unknown>(join(dataDir, EMBEDDING_PROFILE_FILE));
    return value === undefined ? undefined : bankEmbeddingProfileSchema.parse(value);
  } catch (cause) {
    throw new EmbeddingBankError("Bank embedding profile is invalid", {
      code: "EMBEDDING_PROFILE_INVALID",
      cause,
    });
  }
}

export function assertBankNotFenced(dataDir: string): void {
  // A migration fence survives its owner. Only successful cutover removes it.
  if (existsSync(join(dataDir, EMBEDDING_FENCE_FILE))) {
    throw new EmbeddingBankError(
      "Tenant unavailable: embedding migration in progress; drain the tenant and resume the migration",
      {
        code: "EMBEDDING_MIGRATION_FENCED",
      },
    );
  }
}

export function assertEmbeddingProfilesMatch(
  stored: EmbeddingProfile,
  effective: EmbeddingProfile,
): void {
  if (stored.model !== effective.model || stored.dimensions !== effective.dimensions) {
    throw new EmbeddingBankError(
      `Bank embedding profile mismatch: stored ${stored.model}/${stored.dimensions}, configured ${effective.model}/${effective.dimensions}`,
      { code: "EMBEDDING_PROFILE_MISMATCH" },
    );
  }
}

export function embeddingDimensionsFromSchema(tableSchema: {
  fields: readonly { name: string; type: unknown }[];
}): number {
  const type = tableSchema.fields.find((field) => field.name === "embedding")?.type as
    | DataType
    | undefined;
  if (
    type !== undefined &&
    DataType.isFixedSizeList(type) &&
    Number.isInteger(type.listSize) &&
    type.listSize > 0 &&
    DataType.isFloat(type.valueType) &&
    type.valueType.precision === Precision.SINGLE
  ) {
    return type.listSize;
  }
  throw new EmbeddingBankError(
    "Vector table requires embedding: FixedSizeList(dimensions, float32)",
    {
      code: "EMBEDDING_SCHEMA_INVALID",
    },
  );
}

/** Check every stored vector schema without evolving tables or opening SQLite. */
export async function validateBankEmbeddingSchemas(
  dataDir: string,
  dimensions: number,
): Promise<number> {
  let vectorTables = 0;
  try {
    if (existsSync(join(dataDir, "lancedb"))) {
      const connection = await connect(join(dataDir, "lancedb"));
      try {
        for (const name of await connection.tableNames()) {
          const table = await connection.openTable(name);
          try {
            const tableSchema = await table.schema();
            if (
              !(VECTOR_TABLE_NAMES as readonly string[]).includes(name) &&
              !tableSchema.fields.some((field) => field.name === "embedding")
            )
              continue;
            if (embeddingDimensionsFromSchema(tableSchema) !== dimensions) {
              throw new EmbeddingBankError(
                `Bank embedding dimension mismatch in ${name}; migration required`,
                { code: "EMBEDDING_PROFILE_MISMATCH" },
              );
            }
            vectorTables += 1;
          } finally {
            table.close();
          }
        }
      } finally {
        connection.close();
      }
    }
  } catch (cause) {
    if (cause instanceof EmbeddingBankError) throw cause;
    throw new EmbeddingBankError("Cannot validate bank embedding schemas; refusing to open bank", {
      code: "EMBEDDING_PROFILE_UNVERIFIABLE",
      cause,
    });
  }
  return vectorTables;
}

export async function guardBankEmbeddingProfile(
  dataDir: string,
  effective: EmbeddingProfile,
): Promise<BankEmbeddingProfile> {
  assertBankNotFenced(dataDir);
  const profile = embeddingProfileSchema.parse(effective);
  const stored = readBankEmbeddingProfile(dataDir);
  const existingBank = existsSync(join(dataDir, "borg.db")) || existsSync(join(dataDir, "lancedb"));
  if (stored === undefined && existingBank) {
    throw new EmbeddingBankError(
      "Existing bank is missing embedding-profile.json; for a restored pre-profile backup, run scripts/migrate-embeddings.ts label-source-profile with an explicit --model and --dims before opening it",
      { code: "EMBEDDING_PROFILE_REQUIRED" },
    );
  }
  if (stored !== undefined) assertEmbeddingProfilesMatch(stored, profile);
  await validateBankEmbeddingSchemas(dataDir, profile.dimensions);
  if (stored !== undefined) return stored;
  const now = Date.now();
  const initialized: BankEmbeddingProfile = {
    ...profile,
    version: 1,
    generation: 0,
    created_at: now,
    updated_at: now,
    migrated_from: null,
  };
  writeJsonFileAtomic(join(dataDir, EMBEDDING_PROFILE_FILE), initialized, { mode: 0o600 });
  return initialized;
}

// Bank-lifetime lease: the primitive heartbeats even while an idle sidecar pool
// keeps this bank open for hours. Migration takes the same renewable lease after
// installing its persistent fence; only close/release stops the heartbeat.
export async function acquireEmbeddingBankAccess(
  dataDir: string,
  options: { timeoutMs?: number; retryDelayMs?: number } = {},
): Promise<() => Promise<void>> {
  mkdirSync(dataDir, { recursive: true });
  try {
    const lease = await acquireFileLockLease(join(dataDir, EMBEDDING_ACCESS_FILE), {
      ...options,
      timeoutMs: options.timeoutMs ?? 1000,
    });
    return lease.release;
  } catch (cause) {
    throw new EmbeddingBankError(
      "Tenant bank is in use; drain it through /memory/admin/evict before resuming the embedding migration",
      { code: "EMBEDDING_BANK_BUSY", cause },
    );
  }
}
