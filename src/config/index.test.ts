import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { writeJsonFileAtomic } from "../util/atomic-write.js";
import { ConfigError } from "../util/errors.js";
import { DEFAULT_CONFIG, loadConfig, redactConfig } from "./index.js";

describe("config", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("loads defaults without requiring API keys", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {},
    });

    expect(config.dataDir).toBe(tempDir);
    expect(config.embedding.baseUrl).toBe("http://localhost:1234/v1");
    expect(config.embedding.model).toBe("text-embedding-qwen3-embedding-8b");
    expect(config.anthropic.auth).toBe("auto");
    expect(config.anthropic.apiKey).toBeUndefined();
    expect(config.anthropic.models).toEqual({
      cognition: "claude-opus-4-7",
      background: "claude-haiku-4-5-20251001",
      extraction: "claude-sonnet-4-6",
    });
    expect(config.self.autoBootstrapPeriod).toBe(true);
    expect(config.perception.useLlmFallback).toBe(true);
    expect(config.offline.curator.traitHalfLifeDays).toBe(30);
  });

  it("merges config file values with environment overrides", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    writeJsonFileAtomic(join(tempDir, "config.json"), {
      embedding: {
        model: "file-model",
        dims: 2048,
      },
      anthropic: {
        models: {
          cognition: "file-cognition",
        },
      },
    });

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_EMBEDDING_MODEL: "env-model",
        BORG_EMBEDDING_DIMS: "1024",
        BORG_PERCEPTION_USE_LLM_FALLBACK: "false",
        ANTHROPIC_API_KEY: "secret",
      },
    });

    expect(config.embedding.model).toBe("env-model");
    expect(config.embedding.dims).toBe(1024);
    expect(config.perception.useLlmFallback).toBe(false);
    expect(config.anthropic.auth).toBe("auto");
    expect(config.anthropic.apiKey).toBe("secret");
    expect(config.anthropic.models.cognition).toBe("file-cognition");
  });

  it("exposes sprint 4 anthropic model defaults", () => {
    expect(DEFAULT_CONFIG.anthropic.models).toEqual({
      cognition: "claude-opus-4-7",
      background: "claude-haiku-4-5-20251001",
      extraction: "claude-sonnet-4-6",
    });
  });

  it("requires an api key when anthropic auth mode is api-key", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    expect(() =>
      loadConfig({
        dataDir: tempDir,
        env: {
          BORG_ANTHROPIC_AUTH: "api-key",
        },
      }),
    ).toThrow(ConfigError);
  });

  it("allows disabling self period bootstrap via environment", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_SELF_AUTO_BOOTSTRAP_PERIOD: "false",
      },
    });

    expect(config.self.autoBootstrapPeriod).toBe(false);
  });

  it("throws config errors for invalid numeric environment values", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    expect(() =>
      loadConfig({
        dataDir: tempDir,
        env: {
          BORG_EMBEDDING_DIMS: "nope",
        },
      }),
    ).toThrow(ConfigError);
  });

  it("accepts negative and zero env numbers when the schema allows them", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_AUTONOMY_CONDITION_MOOD_VALENCE_DROP_THRESHOLD: "-0.5",
        BORG_OFFLINE_CURATOR_ARCHIVE_MIN_HEAT: "0",
        BORG_AUTONOMY_CONDITION_OPEN_QUESTION_URGENCY_BUMP_THRESHOLD: "0",
      },
    });

    expect(config.autonomy.conditions.moodValenceDrop.threshold).toBe(-0.5);
    expect(config.offline.curator.archiveMinHeat).toBe(0);
    expect(config.autonomy.conditions.openQuestionUrgencyBump.threshold).toBe(0);
  });

  it("rejects non-finite env numbers", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    expect(() =>
      loadConfig({
        dataDir: tempDir,
        env: {
          BORG_OFFLINE_CURATOR_ARCHIVE_MIN_HEAT: "NaN",
        },
      }),
    ).toThrow(ConfigError);
  });

  it("rejects reflector confidence ceilings above the hard cap", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    expect(() =>
      loadConfig({
        dataDir: tempDir,
        env: {
          BORG_OFFLINE_REFLECTOR_CEILING_CONFIDENCE: "0.9",
        },
      }),
    ).toThrow(ConfigError);
  });

  it("wraps invalid config file JSON in a typed config error", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const configPath = join(tempDir, "config.json");
    writeFileSync(configPath, '{"broken"', "utf8");

    try {
      loadConfig({
        dataDir: tempDir,
        env: {},
      });
      expect.unreachable("loadConfig should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(ConfigError);
      expect((error as ConfigError).code).toBe("CONFIG_FILE_INVALID");
      expect((error as ConfigError).message).toContain(configPath);
    }
  });

  it("redacts secrets for display", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        ANTHROPIC_API_KEY: "secret",
      },
    });

    expect(redactConfig(config)).toMatchObject({
      embedding: {
        apiKey: "[REDACTED]",
      },
      anthropic: {
        apiKey: "[REDACTED]",
      },
    });
  });
});
