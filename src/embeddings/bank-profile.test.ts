import { mkdtempSync, rmSync, existsSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { connect } from "@lancedb/lancedb";
import { afterEach, describe, expect, it } from "vitest";
import { Borg } from "../borg.js";
import { FakeEmbeddingClient } from "./index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import { schema, utf8Field, vectorField } from "../storage/lancedb/index.js";
import {
  guardBankEmbeddingProfile,
  EMBEDDING_FENCE_FILE,
  EMBEDDING_PROFILE_FILE,
  readBankEmbeddingProfile,
} from "./bank-profile.js";

const directories: string[] = [];
afterEach(() => {
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
    await guardBankEmbeddingProfile(dir, { model: "old", dimensions: 4 });
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
    await guardBankEmbeddingProfile(dir, { model: "old", dimensions: 4 });
    await expect(
      Borg.open({
        dataDir: dir,
        embeddingClient: new FakeEmbeddingClient(4),
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
});
