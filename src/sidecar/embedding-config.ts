import type { EmbeddingProfile } from "../embeddings/bank-profile.js";
import { ConfigError } from "../util/errors.js";
import { parsePositiveIntegerValue } from "../util/parse.js";

export function sidecarEmbeddingProfileFromEnv(env: NodeJS.ProcessEnv): EmbeddingProfile {
  const model = env.EMBEDDING_MODEL?.trim();
  if (!model) {
    throw new ConfigError("memory-sidecar: EMBEDDING_MODEL is required");
  }
  if (!env.EMBEDDING_DIMS?.trim()) {
    throw new ConfigError("memory-sidecar: EMBEDDING_DIMS is required");
  }
  const dimensions = parsePositiveIntegerValue(env.EMBEDDING_DIMS);
  if (dimensions === null) {
    throw new ConfigError("memory-sidecar: EMBEDDING_DIMS must be a positive integer");
  }
  return { model, dimensions };
}
