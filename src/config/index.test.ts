import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { writeJsonFileAtomic } from "../util/atomic-write.js";
import { ConfigError } from "../util/errors.js";
import { loadConfig, redactConfig } from "./index.js";

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
    expect(config.anthropic.apiKey).toBeUndefined();
    expect(config.perception.useLlmFallback).toBe(true);
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
    expect(config.anthropic.apiKey).toBe("secret");
    expect(config.anthropic.models.cognition).toBe("file-cognition");
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
