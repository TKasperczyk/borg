import { mkdtempSync, rmSync, existsSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { connect } from "@lancedb/lancedb";
import { Field, FixedSizeList, Float64, Int32, List, Schema } from "apache-arrow";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Borg } from "../borg.js";
import { FakeEmbeddingClient } from "./index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import { schema, utf8Field, vectorField } from "../storage/lancedb/index.js";
import { loadConfig } from "../config/index.js";
import {
  BGE_SIMILARITY_MODEL,
  QWEN_SIMILARITY_MODEL,
  similarityThresholds,
} from "../config/similarity.js";
import { createCachingEmbeddingClient } from "./cache.js";
import {
  guardBankEmbeddingProfile,
  EMBEDDING_FENCE_FILE,
  EMBEDDING_PROFILE_FILE,
  readBankEmbeddingProfile,
  VECTOR_TABLE_NAMES,
} from "./bank-profile.js";

const directories: string[] = [];
afterEach(() => {
  vi.restoreAllMocks();
  for (const path of directories.splice(0)) rmSync(path, { recursive: true, force: true });
});
async function bank(dimensions = 4) {
  const dir = mkdtempSync(join(tmpdir(), "embedding-profile-"));
  directories.push(dir);
  const connection = await connect(join(dir, "lancedb"));
  const table = await connection.createEmptyTable(
    "episodes",
    schema([utf8Field("id"), vectorField("embedding", dimensions)]),
  );
  table.close();
  connection.close();
  return dir;
}

describe("bank embedding profile", () => {
  it("adopts a legacy bank only after checking its actual dimension", async () => {
    const dir = await bank();
    await guardBankEmbeddingProfile(dir, { model: "old", dimensions: 4 }, "old");
    expect(readBankEmbeddingProfile(dir)).toMatchObject({
      model: "old",
      dimensions: 4,
      generation: 0,
      migrated_from: null,
    });
    const original = readBankEmbeddingProfile(dir);
    await guardBankEmbeddingProfile(dir, { model: "old", dimensions: 4 });
    expect(readBankEmbeddingProfile(dir)).toEqual(original);
  });
  it("rejects a same-dimension model mismatch before SQLite is opened", async () => {
    const dir = await bank();
    await guardBankEmbeddingProfile(dir, { model: "old", dimensions: 4 }, "old");
    await expect(
      Borg.open({
        dataDir: dir,
        embeddingClient: new FakeEmbeddingClient(4, "new"),
        embeddingProfile: { model: "new", dimensions: 4 },
      }),
    ).rejects.toMatchObject({ code: "EMBEDDING_PROFILE_MISMATCH" });
    expect(existsSync(join(dir, "borg.db"))).toBe(false);
  });
  it("rejects legacy dimension mismatch without adopting or migrating SQL", async () => {
    const dir = await bank();
    await expect(
      guardBankEmbeddingProfile(dir, { model: "new", dimensions: 2 }),
    ).rejects.toMatchObject({ code: "EMBEDDING_PROFILE_MISMATCH" });
    expect(existsSync(join(dir, EMBEDDING_PROFILE_FILE))).toBe(false);
  });
  it("refuses even stale or malformed migration fences", async () => {
    const dir = await bank();
    writeFileSync(join(dir, EMBEDDING_FENCE_FILE), "interrupted");
    await expect(
      guardBankEmbeddingProfile(dir, { model: "old", dimensions: 4 }),
    ).rejects.toMatchObject({ code: "EMBEDDING_MIGRATION_FENCED" });
  });
  it("uses an injected client profile rather than config.json", async () => {
    const dir = mkdtempSync(join(tmpdir(), "embedding-effective-"));
    directories.push(dir);
    const delegate = new FakeEmbeddingClient(4);
    const borg = await Borg.open({
      dataDir: dir,
      embeddingClient: {
        profile: { model: "injected", dimensions: 4 },
        embed: delegate.embed.bind(delegate),
        embedBatch: delegate.embedBatch.bind(delegate),
      },
      llmClient: new FakeLLMClient(),
    });
    try {
      expect(readBankEmbeddingProfile(dir)).toMatchObject({ model: "injected", dimensions: 4 });
    } finally {
      await borg.close();
    }
  });

  it("selects and logs thresholds using the effective bank-guard identity", async () => {
    const dir = mkdtempSync(join(tmpdir(), "embedding-effective-similarity-"));
    directories.push(dir);
    writeFileSync(
      join(dir, "config.json"),
      JSON.stringify({
        embedding: { model: QWEN_SIMILARITY_MODEL, dims: 4096 },
        similarity: { overrides: { actionThread: 0.81 } },
      }),
    );
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const config = loadConfig({ dataDir: dir, env: {} });
    expect(similarityThresholds(config).consolidationSimilarity).toBe(0.82);
    const borg = await Borg.open({
      dataDir: dir,
      config,
      embeddingClient: new FakeEmbeddingClient(1024, BGE_SIMILARITY_MODEL),
      llmClient: new FakeLLMClient(),
    });
    try {
      expect(readBankEmbeddingProfile(dir)).toMatchObject({
        model: BGE_SIMILARITY_MODEL,
        dimensions: 1024,
      });
      expect(borg.similarityThresholds).toMatchObject({
        consolidationSimilarity: 0.76,
        semanticExtractionDuplicate: 0.84,
        actionThread: 0.81,
      });
      expect(Object.isFrozen(borg.similarityThresholds)).toBe(true);
      expect(
        stderr.mock.calls.filter(([line]) => String(line).startsWith("borg open: similarity")),
      ).toEqual([
        [
          `borg open: similarity model="${BGE_SIMILARITY_MODEL}" profile="${BGE_SIMILARITY_MODEL}" fallback=false overrides={"actionThread":0.81}\n`,
        ],
      ]);
    } finally {
      await borg.close();
    }
  });

  it.each([false, true])(
    "refuses an unidentified injected client before creating storage (cached=%s)",
    async (cached) => {
      const dir = mkdtempSync(join(tmpdir(), "embedding-unidentified-"));
      directories.push(dir);
      const delegate = new FakeEmbeddingClient(2);
      const unidentified = {
        embed: delegate.embed.bind(delegate),
        embedBatch: delegate.embedBatch.bind(delegate),
      };
      await expect(
        Borg.open({
          dataDir: dir,
          embeddingClient: cached
            ? createCachingEmbeddingClient(unidentified, { model: "claimed", dims: 4 })
            : unidentified,
          embeddingProfile: { model: "claimed", dimensions: 4 },
          embeddingDimensions: 4,
        }),
      ).rejects.toMatchObject({ code: "EMBEDDING_CLIENT_PROFILE_REQUIRED" });
      expect(existsSync(join(dir, "borg.db"))).toBe(false);
      expect(existsSync(join(dir, "lancedb"))).toBe(false);
      expect(readBankEmbeddingProfile(dir)).toBeUndefined();
    },
  );

  it.each([undefined, "old"])(
    "refuses unlabelled same-dimension banks without a matching assertion (%s)",
    async (legacySourceModel) => {
      const dir = await bank();
      writeFileSync(
        join(dir, "config.json"),
        JSON.stringify({ embedding: { model: "old", dims: 4 } }),
      );
      await expect(
        Borg.open({
          dataDir: dir,
          embeddingClient: new FakeEmbeddingClient(4, "new"),
          embeddingLegacySourceModel: legacySourceModel,
        }),
      ).rejects.toMatchObject({ code: "EMBEDDING_LEGACY_SOURCE_REQUIRED" });
      expect(existsSync(join(dir, "borg.db"))).toBe(false);
      expect(readBankEmbeddingProfile(dir)).toBeUndefined();
    },
  );

  it("adopts through the library config source assertion", async () => {
    const dir = mkdtempSync(join(tmpdir(), "embedding-adoption-config-"));
    directories.push(dir);
    const options = {
      dataDir: dir,
      embeddingClient: new FakeEmbeddingClient(4),
      llmClient: new FakeLLMClient(),
    };
    await (await Borg.open(options)).close();
    rmSync(join(dir, EMBEDDING_PROFILE_FILE));
    const config = loadConfig({
      dataDir: dir,
      env: { BORG_EMBEDDING_LEGACY_SOURCE_MODEL: "fake-embed" },
    });
    expect(config.embedding.legacySourceModel).toBe("fake-embed");
    await (await Borg.open({ ...options, config })).close();
    expect(readBankEmbeddingProfile(dir)).toMatchObject({
      model: "fake-embed",
      dimensions: 4,
      generation: 0,
    });
  });

  it.each(VECTOR_TABLE_NAMES)(
    "rejects a missing embedding column in %s before SQLite/profile writes",
    async (name) => {
      const dir = mkdtempSync(join(tmpdir(), "embedding-missing-column-"));
      directories.push(dir);
      const connection = await connect(join(dir, "lancedb"));
      (await connection.createEmptyTable(name, schema([utf8Field("id")]))).close();
      connection.close();
      await expect(
        Borg.open({
          dataDir: dir,
          embeddingClient: new FakeEmbeddingClient(4),
          embeddingLegacySourceModel: "fake-embed",
        }),
      ).rejects.toMatchObject({ code: "EMBEDDING_SCHEMA_INVALID" });
      expect(existsSync(join(dir, "borg.db"))).toBe(false);
      expect(readBankEmbeddingProfile(dir)).toBeUndefined();
    },
  );

  it.each([
    new FixedSizeList(4, new Field("item", new Float64(), true)),
    new FixedSizeList(4, new Field("item", new Int32(), true)),
    new List(new Field("item", new Float64(), true)),
  ])("rejects invalid vector element/list types (%s)", async (type) => {
    const dir = mkdtempSync(join(tmpdir(), "embedding-invalid-type-"));
    directories.push(dir);
    const connection = await connect(join(dir, "lancedb"));
    (
      await connection.createEmptyTable(
        "episodes",
        new Schema([utf8Field("id"), new Field("embedding", type, true)]),
      )
    ).close();
    connection.close();
    await expect(
      guardBankEmbeddingProfile(dir, { model: "old", dimensions: 4 }, "old"),
    ).rejects.toMatchObject({ code: "EMBEDDING_SCHEMA_INVALID" });
    expect(readBankEmbeddingProfile(dir)).toBeUndefined();
  });
});
