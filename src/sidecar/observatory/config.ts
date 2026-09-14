import { z } from "zod";

import { ConfigError } from "../../util/errors.js";

export type ObservatoryConfig =
  | { enabled: false }
  | {
      enabled: true;
      baseUrl: string;
      token: string;
      flushMs: number;
      maxBatch: number;
      queueCap: number;
      timeoutMs: number;
    };

const timerMs = z.coerce.number().int().positive().max(2_147_483_647);
const enabledObservatoryEnvSchema = z.object({
  OBSERVATORY_BASE_URL: z.url().refine((value) => {
    const url = new URL(value);
    return (url.protocol === "http:" || url.protocol === "https:") &&
      !url.username && !url.password && !url.search && !url.hash;
  }),
  OBSERVATORY_TOKEN: z.string().trim().min(1),
  OBSERVATORY_FLUSH_MS: timerMs.default(250),
  OBSERVATORY_MAX_BATCH: z.coerce.number().int().positive().max(500).default(100),
  OBSERVATORY_QUEUE_CAP: z.coerce.number().int().positive().default(5_000),
  OBSERVATORY_TIMEOUT_MS: timerMs.default(2_000),
});

export function observatoryConfigFromEnv(env: NodeJS.ProcessEnv): ObservatoryConfig {
  if (env.OBSERVATORY_BASE_URL === undefined || env.OBSERVATORY_BASE_URL.trim() === "") {
    return { enabled: false };
  }
  const parsed = enabledObservatoryEnvSchema.safeParse(env);
  if (!parsed.success) {
    throw new ConfigError("Invalid observatory configuration", { cause: parsed.error });
  }
  return {
    enabled: true,
    baseUrl: parsed.data.OBSERVATORY_BASE_URL,
    token: parsed.data.OBSERVATORY_TOKEN,
    flushMs: parsed.data.OBSERVATORY_FLUSH_MS,
    maxBatch: parsed.data.OBSERVATORY_MAX_BATCH,
    queueCap: parsed.data.OBSERVATORY_QUEUE_CAP,
    timeoutMs: parsed.data.OBSERVATORY_TIMEOUT_MS,
  };
}
