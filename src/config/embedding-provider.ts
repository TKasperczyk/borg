import { z } from "zod";

import { ConfigError } from "../util/errors.js";
import { isPlainRecord } from "../util/guards.js";
import { jsonValueSchema, type JsonValue } from "../util/json-value.js";

// Routing options belong to the embedding service; preserve all JSON keys and values.
export const embeddingProviderSchema = z.custom<Record<string, JsonValue>>(
  (value) => isPlainRecord(value) && jsonValueSchema.safeParse(value).success,
  { message: "Expected a plain JSON object" },
);
export type EmbeddingProvider = z.infer<typeof embeddingProviderSchema>;

export function readEmbeddingProviderEnv(env: NodeJS.ProcessEnv): EmbeddingProvider | undefined {
  const raw = env.BORG_EMBEDDING_PROVIDER;
  if (raw === undefined) return undefined;

  try {
    return embeddingProviderSchema.parse(JSON.parse(raw) as unknown);
  } catch (cause) {
    throw new ConfigError("BORG_EMBEDDING_PROVIDER must be a JSON object", { cause });
  }
}
