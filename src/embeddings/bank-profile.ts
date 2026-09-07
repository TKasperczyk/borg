import { existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { connect } from "@lancedb/lancedb";
import { z } from "zod";

import { readJsonFile, writeJsonFileAtomic } from "../util/atomic-write.js";
import { StorageError } from "../util/errors.js";
import { withFileLock } from "../stream/file-lock.js";

export const EMBEDDING_PROFILE_FILE = "embedding-profile.json";
export const EMBEDDING_FENCE_FILE = "embedding-migration.lock";
export const EMBEDDING_ACCESS_FILE = "embedding-bank-access.lock";

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
    | { listSize?: unknown }
    | undefined;
  return z.number().int().positive().parse(type?.listSize);
}

export async function guardBankEmbeddingProfile(
  dataDir: string,
  effective: EmbeddingProfile,
): Promise<BankEmbeddingProfile> {
  assertBankNotFenced(dataDir);
  const profile = embeddingProfileSchema.parse(effective);
  const stored = readBankEmbeddingProfile(dataDir);
  if (stored !== undefined) assertEmbeddingProfilesMatch(stored, profile);
  try {
    // Direct opens only: no schema evolution, SQL migrations or reconciliation.
    if (existsSync(join(dataDir, "lancedb"))) {
      const connection = await connect(join(dataDir, "lancedb"));
      try {
        for (const name of await connection.tableNames()) {
          const table = await connection.openTable(name);
          try {
            const tableSchema = await table.schema();
            if (!tableSchema.fields.some((field) => field.name === "embedding")) continue;
            if (embeddingDimensionsFromSchema(tableSchema) !== profile.dimensions) {
              throw new EmbeddingBankError(
                `Bank embedding dimension mismatch in ${name}; migration required`,
                { code: "EMBEDDING_PROFILE_MISMATCH" },
              );
            }
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
  if (stored !== undefined) return stored;
  const now = Date.now();
  const adopted: BankEmbeddingProfile = {
    ...profile,
    version: 1,
    generation: 0,
    created_at: now,
    updated_at: now,
    migrated_from: null,
  };
  writeJsonFileAtomic(join(dataDir, EMBEDDING_PROFILE_FILE), adopted, { mode: 0o600 });
  return adopted;
}

// Bank-lifetime lease. Uses the same lock ownership/stale-owner rules as stream
// writes. Migration also takes this lock, after installing its persistent fence.
export async function acquireEmbeddingBankAccess(dataDir: string): Promise<() => Promise<void>> {
  mkdirSync(dataDir, { recursive: true });
  let release!: () => void;
  let acquired!: () => void;
  let failed!: (error: unknown) => void;
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  const ready = new Promise<void>((resolve, reject) => {
    acquired = resolve;
    failed = reject;
  });
  const holding = (async () => {
    try {
      await withFileLock(
        join(dataDir, EMBEDDING_ACCESS_FILE),
        async () => {
          acquired();
          await released;
        },
        { timeoutMs: 1000 },
      );
    } catch (cause) {
      const error = new EmbeddingBankError(
        "Tenant bank is in use; drain it through /memory/admin/evict before resuming the embedding migration",
        { code: "EMBEDDING_BANK_BUSY", cause },
      );
      failed(error);
      throw error;
    }
  })();
  void holding.catch(() => undefined);
  await ready;
  return async () => {
    release();
    await holding;
  };
}
