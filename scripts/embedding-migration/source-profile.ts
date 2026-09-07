import { lstatSync } from "node:fs";
import { join } from "node:path";

import {
  acquireEmbeddingBankAccess,
  assertBankNotFenced,
  bankEmbeddingProfileSchema,
  EMBEDDING_PROFILE_FILE,
  EmbeddingBankError,
  embeddingProfileSchema,
  readBankEmbeddingProfile,
  validateBankEmbeddingSchemas,
  type BankEmbeddingProfile,
  type EmbeddingProfile,
} from "../../src/embeddings/bank-profile.js";
import { writeJsonFileAtomic } from "../../src/util/atomic-write.js";

/** Operator assertion for restored backups whose vector model is known externally. */
export async function labelRestoredEmbeddingBank(
  dataDir: string,
  source: EmbeddingProfile,
  options: { timeoutMs?: number } = {},
): Promise<BankEmbeddingProfile> {
  const profile = embeddingProfileSchema.parse(source);
  if (!lstatSync(dataDir).isDirectory()) {
    throw new EmbeddingBankError("Source bank must be an existing directory, not a symlink", {
      code: "EMBEDDING_PROFILE_UNVERIFIABLE",
    });
  }
  const release = await acquireEmbeddingBankAccess(dataDir, options);
  try {
    assertBankNotFenced(dataDir);
    if (readBankEmbeddingProfile(dataDir) !== undefined) {
      throw new EmbeddingBankError(
        "Bank already has embedding-profile.json; refusing to replace it",
        {
          code: "EMBEDDING_PROFILE_EXISTS",
        },
      );
    }
    if ((await validateBankEmbeddingSchemas(dataDir, profile.dimensions)) === 0) {
      throw new EmbeddingBankError("No stored vector schema can verify the asserted dimensions", {
        code: "EMBEDDING_PROFILE_UNVERIFIABLE",
      });
    }
    const now = Date.now();
    const labelled = bankEmbeddingProfileSchema.parse({
      ...profile,
      version: 1,
      generation: 0,
      created_at: now,
      updated_at: now,
      migrated_from: null,
    });
    writeJsonFileAtomic(join(dataDir, EMBEDDING_PROFILE_FILE), labelled, { mode: 0o600 });
    return labelled;
  } finally {
    await release();
  }
}
