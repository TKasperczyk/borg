import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { connect } from "@lancedb/lancedb";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  acquireEmbeddingBankAccess,
  EMBEDDING_FENCE_FILE,
  EMBEDDING_PROFILE_FILE,
  guardBankEmbeddingProfile,
  readBankEmbeddingProfile,
} from "../../src/embeddings/bank-profile.js";
import { schema, utf8Field, vectorField } from "../../src/storage/lancedb/index.js";
import { seedTestEmbeddingProfile } from "../../src/test-support/embedding-profile.js";
import { embeddingMigrationMain } from "../migrate-embeddings.js";
import { labelRestoredEmbeddingBank } from "./source-profile.js";

const directories: string[] = [];
afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of directories.splice(0)) rmSync(dir, { recursive: true, force: true });
});

async function restoredBank(dimensions: readonly number[] = [4]): Promise<string> {
  const dir = mkdtempSync(join(tmpdir(), "borg-restored-profile-"));
  directories.push(dir);
  if (dimensions.length > 0) {
    const connection = await connect(join(dir, "lancedb"));
    try {
      for (const [index, dims] of dimensions.entries()) {
        const table = await connection.createEmptyTable(
          index === 0 ? "episodes" : "additional_vectors",
          schema([utf8Field("id"), vectorField("embedding", dims)]),
        );
        table.close();
      }
    } finally {
      connection.close();
    }
  }
  return dir;
}

describe("label-source-profile", () => {
  it("writes a generation-zero profile through the CLI without embedding requests", async () => {
    const dir = await restoredBank();
    vi.spyOn(process.stdout, "write").mockReturnValue(true);
    await expect(
      embeddingMigrationMain([
        "label-source-profile",
        "--data-dir",
        dir,
        "--model",
        "restored-model",
        "--dims",
        "4",
      ]),
    ).resolves.toBe(0);
    expect(readBankEmbeddingProfile(dir)).toMatchObject({
      model: "restored-model",
      dimensions: 4,
      generation: 0,
      migrated_from: null,
    });
    await expect(
      guardBankEmbeddingProfile(dir, { model: "restored-model", dimensions: 4 }),
    ).resolves.toEqual(readBankEmbeddingProfile(dir));
    expect(existsSync(join(dir, "borg.db"))).toBe(false);
  });

  it.each([[], [2], [4, 2]])(
    "refuses unverifiable or mismatched vector dimensions %j",
    async (...dims) => {
      const dir = await restoredBank(dims);
      await expect(
        labelRestoredEmbeddingBank(dir, { model: "restored-model", dimensions: 4 }),
      ).rejects.toThrow(/dimensions|dimension mismatch/);
      expect(readBankEmbeddingProfile(dir)).toBeUndefined();
    },
  );

  it("refuses to replace an existing profile", async () => {
    const dir = await restoredBank();
    seedTestEmbeddingProfile(dir);
    const original = readFileSync(join(dir, EMBEDDING_PROFILE_FILE), "utf8");
    await expect(
      labelRestoredEmbeddingBank(dir, { model: "other", dimensions: 4 }),
    ).rejects.toMatchObject({ code: "EMBEDDING_PROFILE_EXISTS" });
    expect(readFileSync(join(dir, EMBEDDING_PROFILE_FILE), "utf8")).toBe(original);
  });

  it("honors the persistent migration fence", async () => {
    const dir = await restoredBank();
    writeFileSync(join(dir, EMBEDDING_FENCE_FILE), "interrupted");
    await expect(
      labelRestoredEmbeddingBank(dir, { model: "restored-model", dimensions: 4 }),
    ).rejects.toMatchObject({ code: "EMBEDDING_MIGRATION_FENCED" });
    expect(readBankEmbeddingProfile(dir)).toBeUndefined();
  });

  it("requires exclusive access to the bank", async () => {
    const dir = await restoredBank();
    const release = await acquireEmbeddingBankAccess(dir);
    try {
      await expect(
        labelRestoredEmbeddingBank(
          dir,
          { model: "restored-model", dimensions: 4 },
          { timeoutMs: 20 },
        ),
      ).rejects.toMatchObject({ code: "EMBEDDING_BANK_BUSY" });
      expect(readBankEmbeddingProfile(dir)).toBeUndefined();
    } finally {
      await release();
    }
  });

  it.each(["--data-dir", "--model", "--dims"])("requires %s explicitly", async (missing) => {
    const dir = await restoredBank();
    const args = [
      ["--data-dir", dir],
      ["--model", "restored-model"],
      ["--dims", "4"],
    ]
      .filter(([flag]) => flag !== missing)
      .flat();
    await expect(embeddingMigrationMain(["label-source-profile", ...args])).rejects.toThrow(
      "label-source-profile requires --data-dir, --model and --dims",
    );
    expect(readBankEmbeddingProfile(dir)).toBeUndefined();
  });
});
