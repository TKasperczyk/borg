import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { createSemanticNodeId, type EpisodeId, type SemanticNodeId } from "../../util/ids.js";
import { semanticMigrations } from "./migrations.js";
import { SemanticNodeRepository, createSemanticNodesTableSchema } from "./repository.js";
import { SemanticReviewService } from "./review-service.js";

const CONTRADICTION_TOOL_NAME = "EmitContradictionJudgment";

async function createSemanticFixture() {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
  const store = new LanceDbStore({
    uri: join(tempDir, "lancedb"),
  });
  const db = openDatabase(join(tempDir, "borg.db"), {
    migrations: semanticMigrations,
  });
  const table = await store.openTable({
    name: "semantic_nodes",
    schema: createSemanticNodesTableSchema(4),
  });
  const clock = new FixedClock(1_000);
  const nodeRepository = new SemanticNodeRepository({
    table,
    db,
    clock,
  });

  return {
    tempDir,
    store,
    db,
    table,
    clock,
    nodeRepository,
  };
}

function buildProposition(id: SemanticNodeId, label: string) {
  return {
    id,
    kind: "proposition" as const,
    label,
    description: `${label} description`,
    aliases: [],
    confidence: 0.7,
    source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
    created_at: 1_000,
    updated_at: 1_000,
    last_verified_at: 1_000,
    embedding: Float32Array.from([1, 0, 0, 0]),
    archived: false,
    superseded_by: null,
  };
}

describe("semantic review service", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("keeps repository inserts inert until duplicate review is explicitly requested", async () => {
    const fixture = await createSemanticFixture();
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: CONTRADICTION_TOOL_NAME,
              input: { contradicts: true, confidence: 0.9 },
            },
          ],
        },
      ],
    });
    const enqueueReview = vi.fn();
    const reviewService = new SemanticReviewService({
      nodeRepository: fixture.nodeRepository,
      llmClient: llm,
      contradictionJudgeModel: "haiku",
      enqueueReview,
    });

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    await fixture.nodeRepository.insert(
      buildProposition(createSemanticNodeId(), "Atlas is stable"),
    );
    const inserted = await fixture.nodeRepository.insert(
      buildProposition(createSemanticNodeId(), "Atlas is unstable"),
    );

    expect(llm.requests).toHaveLength(0);
    expect(enqueueReview).not.toHaveBeenCalled();

    await reviewService.reviewDuplicateCandidate(inserted);

    expect(llm.requests).toHaveLength(1);
    expect(enqueueReview).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "duplicate",
        refs: expect.objectContaining({
          node_ids: expect.arrayContaining([inserted.id]),
        }),
      }),
    );
  });

  it("logs duplicate-review background failures without an unhandled rejection", async () => {
    const fixture = await createSemanticFixture();
    const writer = new StreamWriter({
      dataDir: fixture.tempDir,
      clock: fixture.clock,
    });
    const logged: Promise<void>[] = [];
    const reviewService = new SemanticReviewService({
      nodeRepository: fixture.nodeRepository,
      enqueueReview: vi.fn(),
      llmClient: new FakeLLMClient(),
      contradictionJudgeModel: "haiku",
      onDuplicateReviewError: (error) => {
        const promise = writer
          .append({
            kind: "internal_event",
            content: {
              hook: "semantic_duplicate_review",
              error: error instanceof Error ? error.message : String(error),
            },
          })
          .then(() => undefined);
        logged.push(promise);
        return promise;
      },
    });

    cleanup.push(async () => {
      writer.close();
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    vi.spyOn(fixture.nodeRepository, "searchByVector").mockRejectedValue(
      new Error("vector exploded"),
    );

    const inserted = await fixture.nodeRepository.insert(
      buildProposition(createSemanticNodeId(), "Atlas is unstable"),
    );

    reviewService.queueDuplicateReview(inserted);
    await new Promise((resolve) => {
      setImmediate(resolve);
    });
    await Promise.all(logged);

    const [entry] = new StreamReader({
      dataDir: fixture.tempDir,
    }).tail(1);

    expect(entry?.kind).toBe("internal_event");
    expect(entry?.content).toMatchObject({
      hook: "semantic_duplicate_review",
      error: "vector exploded",
    });
  });
});
