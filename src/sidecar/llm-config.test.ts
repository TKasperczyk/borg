import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DEFAULT_CONFIG, loadConfig, MODEL_SLOT_ENV_NAMES } from "../config/index.js";
import { defaultSidecarModelSlots, sidecarLlmGatewayOptionsFromEnv } from "./llm-config.js";
let dataDir: string;
beforeEach(() => {
  dataDir = mkdtempSync(join(tmpdir(), "borg-sidecar-models-"));
});
afterEach(() => rmSync(dataDir, { recursive: true, force: true }));

describe("sidecar LLM configuration", () => {
  it("defaults every model slot and makes every slot independently configurable", () => {
    const env: NodeJS.ProcessEnv = { BORG_OPTION_A_PROBE: "1" };
    defaultSidecarModelSlots(env, "p4-default");
    expect(Object.values(loadConfig({ env, dataDir }).anthropic.models)).toEqual(
      Object.keys(MODEL_SLOT_ENV_NAMES).map(() => "p4-default"),
    );
    for (const [slot, key] of Object.entries(MODEL_SLOT_ENV_NAMES)) {
      env[key] = `model-${slot}`;
    }
    defaultSidecarModelSlots(env, "ignored");
    expect(loadConfig({ env, dataDir }).anthropic.models).toEqual(
      Object.fromEntries(Object.keys(MODEL_SLOT_ENV_NAMES).map((slot) => [slot, `model-${slot}`])),
    );
  });
  it("treats blank overrides as unset, matching the config loader", () => {
    const env = { BORG_OPTION_A_PROBE: "true", BORG_MODEL_COGNITION: " " };
    defaultSidecarModelSlots(env, "fallback");
    expect(env.BORG_MODEL_COGNITION).toBe("fallback");
  });
  it.each([undefined, "", "  ", "explicit-model"])(
    "preserves HEAD model selection with probe off and overrides=%j",
    (value) => {
      for (const tenantConfigured of [false, true]) {
        const tenantDir = join(dataDir, tenantConfigured ? "configured-tenant" : "default-tenant");
        mkdirSync(tenantDir);
        const models = Object.fromEntries(
          Object.keys(MODEL_SLOT_ENV_NAMES).map((slot) => [slot, `tenant-${slot}`]),
        );
        if (tenantConfigured)
          writeFileSync(join(tenantDir, "config.json"), JSON.stringify({ anthropic: { models } }));
        const env: NodeJS.ProcessEnv = { BORG_OPTION_A_PROBE: "0" };
        for (const key of Object.values(MODEL_SLOT_ENV_NAMES))
          if (value !== undefined) env[key] = value;
        defaultSidecarModelSlots(env, "sidecar-model");
        const effective = loadConfig({ env, dataDir: tenantDir }).anthropic.models;
        for (const [slot, key] of Object.entries(MODEL_SLOT_ENV_NAMES)) {
          const fallback = tenantConfigured
            ? models[slot]
            : DEFAULT_CONFIG.anthropic.models[slot as keyof typeof MODEL_SLOT_ENV_NAMES];
          expect(effective[slot as keyof typeof effective]).toBe(
            slot === "imagePerception"
              ? fallback
              : value === undefined
                ? "sidecar-model"
                : value.trim() || fallback,
          );
          expect(env[key]).toBe(slot === "imagePerception" ? value : (value ?? "sidecar-model"));
        }
      }
    },
  );
  it.each([undefined, "0", "", " ", "12.5", "-1", "invalid"])(
    "preserves HEAD Number timeout parsing (%j) and ignores probe gateway env",
    (timeout) => {
      expect(
        sidecarLlmGatewayOptionsFromEnv(
          {
            BORG_MEMORY_LLM_TIMEOUT_MS: timeout,
            BORG_MEMORY_LLM_MAX_TOKENS: "invalid-even-for-probe",
            BORG_MEMORY_LLM_REASONING_EFFORT: "invalid-even-for-probe",
          },
          false,
        ),
      ).toEqual({ requestTimeoutMs: Number(timeout ?? 120_000) });
    },
  );
  it("retains the default transport with the probe off and opts into kratos limits with it on", () => {
    expect(sidecarLlmGatewayOptionsFromEnv({}, false)).toEqual({ requestTimeoutMs: 120_000 });
    expect(sidecarLlmGatewayOptionsFromEnv({}, true)).toEqual({
      requestTimeoutMs: 180_000,
      maxOutputTokens: 16_384,
      reasoningEffort: "none",
    });
    expect(
      sidecarLlmGatewayOptionsFromEnv(
        {
          BORG_MEMORY_LLM_TIMEOUT_MS: "200000",
          BORG_MEMORY_LLM_MAX_TOKENS: "4096",
          BORG_MEMORY_LLM_REASONING_EFFORT: "low",
        },
        true,
      ),
    ).toEqual({ requestTimeoutMs: 200_000, maxOutputTokens: 4096, reasoningEffort: "low" });
    expect(() =>
      sidecarLlmGatewayOptionsFromEnv({ BORG_MEMORY_LLM_MAX_TOKENS: "16385" }, true),
    ).toThrow("gateway configuration");
  });
});
