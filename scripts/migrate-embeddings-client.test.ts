import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ConfigError } from "../src/util/errors.js";
import { migrateTenant } from "./embedding-migration/migrate.js";
import { embeddingMigrationMain } from "./migrate-embeddings.js";

vi.mock("./embedding-migration/migrate.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./embedding-migration/migrate.js")>()),
  migrateTenant: vi.fn(),
}));

describe("embedding migration CLI client routing", () => {
  let root: string;
  let args: string[];

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "borg-embedding-client-"));
    const tenant = join(root, "test-tenant");
    mkdirSync(tenant);
    // Discovery checks for this file; migration itself is mocked and never opens it.
    writeFileSync(join(tenant, "borg.db"), "");
    args = [
      "--data-root",
      root,
      "--tenant",
      "test-tenant",
      "--target-model",
      "qwen/qwen3-embedding-8b",
      "--target-dims",
      "2",
      "--batch-size",
      "2",
    ];
    vi.mocked(migrateTenant).mockReset();
    vi.stubEnv("LLM_API_KEY", undefined);
    vi.stubEnv("KRATOS_BASE_URL", undefined);
    vi.stubEnv("BORG_EMBEDDING_API_KEY", "test-key");
    vi.stubEnv("BORG_EMBEDDING_BASE_URL", "https://embeddings.invalid/v1");
    vi.stubEnv("BORG_EMBEDDING_PROVIDER", undefined);
    vi.spyOn(process.stdout, "write").mockReturnValue(true);
    // An unexpected request must fail inside the mock, never reach the network.
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Unexpected fetch")));
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    rmSync(root, { recursive: true, force: true });
  });

  it.each([
    undefined,
    {},
    { data_collection: "deny", ignore: ["SiliconFlow"], nested: { values: [null, false, 2] } },
  ])("sends provider %j through the fresh CLI SDK client", async (provider) => {
    if (provider !== undefined) vi.stubEnv("BORG_EMBEDDING_PROVIDER", JSON.stringify(provider));
    const bodies: string[] = [];
    const fetch = vi.fn(async (url: unknown, init?: RequestInit) => {
      expect(String(url)).toBe("https://embeddings.invalid/v1/embeddings");
      bodies.push(String(init?.body));
      const body = JSON.parse(String(init?.body)) as { input: string | string[] };
      const texts = Array.isArray(body.input) ? body.input : [body.input];
      return new Response(
        JSON.stringify({ data: texts.map((_, index) => ({ index, embedding: [1, 2] })) }),
        { headers: { "content-type": "application/json" } },
      );
    });
    vi.stubGlobal("fetch", fetch);
    vi.mocked(migrateTenant).mockImplementation(async (_options, dependencies) => {
      expect(dependencies?.client?.profile).toEqual({
        model: "qwen/qwen3-embedding-8b",
        dimensions: 2,
      });
      await dependencies!.client!.embedBatch(["first", "second", "third"]);
      return { complete: true };
    });

    await expect(embeddingMigrationMain(args)).resolves.toBe(0);

    expect(migrateTenant).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(bodies).toEqual(
      [["first", "second"], "third"].map((input) =>
        JSON.stringify({
          input,
          model: "qwen/qwen3-embedding-8b",
          encoding_format: "float",
          ...(provider === undefined ? {} : { provider }),
        }),
      ),
    );
    if (provider === undefined) {
      for (const body of bodies) expect(JSON.parse(body)).not.toHaveProperty("provider");
    }
  });

  it.each(["{", "[]", "", "  ", "null"])(
    "rejects malformed provider %j before migration or requests",
    async (provider) => {
      vi.stubEnv("BORG_EMBEDDING_PROVIDER", provider);

      await expect(embeddingMigrationMain(args)).rejects.toThrow(ConfigError);
      expect(migrateTenant).not.toHaveBeenCalled();
      expect(fetch).not.toHaveBeenCalled();
    },
  );

  it.each(["--dry-run", "--verify-only"])("validates provider JSON before %s", async (mode) => {
    vi.stubEnv("BORG_EMBEDDING_PROVIDER", "false");

    await expect(embeddingMigrationMain([...args, mode])).rejects.toThrow(
      "BORG_EMBEDDING_PROVIDER must be a JSON object",
    );
    expect(migrateTenant).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });
});
