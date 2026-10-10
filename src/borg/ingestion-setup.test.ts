import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Borg } from "../borg.js";
import { EMBEDDING_ACCESS_FILE } from "../embeddings/bank-profile.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import { EpisodicExtractor } from "../memory/episodic/extractor.js";
import { createTestConfig, TestEmbeddingClient } from "../offline/test-support.js";
import { StreamWriter } from "../stream/index.js";
import type { BorgDependencies } from "./types.js";

describe("ingestion setup failure reporting", () => {
  afterEach(() => vi.restoreAllMocks());
  it.each(["synchronous", "asynchronous"])(
    "retains a %s production ingestion report failure for shutdown",
    async (mode) => {
      const dataDir = mkdtempSync(join(tmpdir(), "borg-ingestion-report-"));
      const borg = await Borg.open({
        config: createTestConfig({ dataDir }),
        embeddingDimensions: 4,
        embeddingClient: new TestEmbeddingClient(),
        llmClient: new FakeLLMClient(),
      });
      const { deps } = borg as unknown as { deps: BorgDependencies };
      const original = new Error("extraction failed");
      const failure = new Error("ingestion report append failed");
      const entry = await borg.stream.append({
        kind: "user_msg",
        content: "Source for extraction",
      });
      vi.spyOn(EpisodicExtractor.prototype, "extractFromStream").mockRejectedValue(original);
      vi.spyOn(StreamWriter.prototype, "append").mockImplementation(() => {
        if (mode === "synchronous") throw failure;
        return Promise.reject(failure);
      });
      const log = vi.spyOn(console, "error").mockImplementation(() => {});
      const sqliteClose = vi.spyOn(deps.sqlite, "close");
      try {
        await expect(
          deps.streamIngestionCoordinator!.ingest(entry.session_id, { minEntriesThreshold: 1 }),
        ).resolves.toMatchObject({ error: original });
        expect(log).toHaveBeenCalled();
        await expect(borg.close()).rejects.toThrow(AggregateError);
        await expect(deps.failureReports!.flush()).rejects.toMatchObject({
          errors: expect.arrayContaining([failure]),
        });
        expect(sqliteClose).toHaveBeenCalledOnce();
        expect(existsSync(join(dataDir, EMBEDDING_ACCESS_FILE))).toBe(true);
      } finally {
        if (sqliteClose.mock.calls.length === 0) await borg.close().catch(() => {});
        await deps.releaseEmbeddingBankAccess?.();
        rmSync(dataDir, { recursive: true, force: true });
      }
    },
  );
});
