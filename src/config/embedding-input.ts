import { z } from "zod";

import { ConfigError } from "../util/errors.js";

export const embeddingMaxInputBytesSchema = z.number().int().positive();

export function readEmbeddingMaxInputBytesEnv(env: NodeJS.ProcessEnv): number | undefined {
  const raw = env.BORG_EMBEDDING_MAX_INPUT_BYTES;
  if (raw === undefined) return undefined;

  // Reject numeric coercions such as hex, exponents, fractions and empty strings.
  const parsed = embeddingMaxInputBytesSchema.safeParse(
    /^\d+$/.test(raw.trim()) ? Number(raw) : undefined,
  );
  if (!parsed.success) {
    throw new ConfigError("BORG_EMBEDDING_MAX_INPUT_BYTES must be a positive integer", {
      cause: parsed.error,
    });
  }
  return parsed.data;
}
