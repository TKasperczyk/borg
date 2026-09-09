import { z } from "zod";

import { MODEL_SLOT_ENV_NAMES, optionAProbeEnabledFromEnv } from "../config/index.js";
import { ConfigError } from "../util/errors.js";

export function defaultSidecarModelSlots(env: NodeJS.ProcessEnv, model: string): void {
  if (!optionAProbeEnabledFromEnv(env)) {
    // Preserve HEAD's seven slots, order and nullish-only assignment. Blank
    // values intentionally fall through to each tenant's config/defaults.
    for (const slot of [
      "BORG_MODEL_EXTRACTION",
      "BORG_MODEL_RECALL_EXPANSION",
      "BORG_MODEL_COGNITION",
      "BORG_MODEL_BACKGROUND",
      "BORG_MODEL_CREATOR_DIRECTIVE",
      "BORG_MODEL_CORRECTIVE_PREFERENCE",
      "BORG_MODEL_SHARED_STATE_COMPILER",
    ])
      env[slot] ??= model;
    return;
  }
  for (const envName of Object.values(MODEL_SLOT_ENV_NAMES)) {
    if (env[envName] === undefined || env[envName]?.trim() === "") env[envName] = model;
  }
}

const gatewayOptionsSchema = z.object({
  requestTimeoutMs: z.coerce.number().int().positive(),
  maxOutputTokens: z.coerce.number().int().positive().max(16_384).optional(),
  reasoningEffort: z.enum(["none", "minimal", "low", "medium", "high", "xhigh"]).optional(),
});

export function sidecarLlmGatewayOptionsFromEnv(
  env: NodeJS.ProcessEnv,
  probeEnabled: boolean,
): z.infer<typeof gatewayOptionsSchema> {
  if (!probeEnabled) {
    // Deliberately retain Number(), including 0/blank/fractional values and
    // HEAD's downstream SDK validation. Probe-only knobs have no effect here.
    return { requestTimeoutMs: Number(env.BORG_MEMORY_LLM_TIMEOUT_MS ?? 120_000) };
  }
  const parsed = gatewayOptionsSchema.safeParse({
    requestTimeoutMs: env.BORG_MEMORY_LLM_TIMEOUT_MS ?? 180_000,
    maxOutputTokens: env.BORG_MEMORY_LLM_MAX_TOKENS ?? 16_384,
    reasoningEffort: env.BORG_MEMORY_LLM_REASONING_EFFORT ?? "none",
  });
  if (!parsed.success) {
    throw new ConfigError("Invalid sidecar LLM gateway configuration", { cause: parsed.error });
  }
  return parsed.data;
}
