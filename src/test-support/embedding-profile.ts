import { join } from "node:path";

import { EMBEDDING_PROFILE_FILE, type EmbeddingProfile } from "../embeddings/bank-profile.js";
import { writeJsonFileAtomic } from "../util/atomic-write.js";

/** Label synthetic vectors in storage fixtures that are built without Borg.open. */
export function seedTestEmbeddingProfile(
  dataDir: string,
  profile: EmbeddingProfile = { model: "fake-embed", dimensions: 4 },
): void {
  writeJsonFileAtomic(join(dataDir, EMBEDDING_PROFILE_FILE), {
    ...profile,
    version: 1,
    generation: 0,
    created_at: 1_000,
    updated_at: 1_000,
    migrated_from: null,
  });
}
