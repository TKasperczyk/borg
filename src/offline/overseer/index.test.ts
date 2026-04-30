import { afterEach, describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "../../config/index.js";
import { FakeLLMClient } from "../../llm/index.js";
import type { ReviewOpenQuestionExtractorLike } from "../../memory/self/index.js";
import { StreamReader } from "../../stream/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { createMaintenanceRunId } from "../../util/ids.js";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../test-support.js";
import { OverseerProcess } from "./index.js";

const OVERSEER_TOOL_NAME = "EmitOverseerFlags";

function createOverseerResponse(flags: unknown[], inputTokens = 12, outputTokens = 8) {
  return {
    text: "",
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: OVERSEER_TOOL_NAME,
        input: { flags },
      },
    ],
  };
}

describe("overseer process", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("flags misattribution-like issues and can revert the audit item", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient({
      responses: [
        createOverseerResponse([
          {
            kind: "misattribution",
            reason: "The narrative mentions Alex, but Alex is missing from participants.",
            confidence: 0.8,
          },
        ]),
      ],
    });
    const reviewOpenQuestionExtractor: ReviewOpenQuestionExtractorLike = {
      async extract(_item, context) {
        return {
          question: "¿Qué atribución debería corregirse?",
          urgency: 0.61,
          related_episode_ids: [...context.allowed_episode_ids],
          related_semantic_node_ids: [],
        };
      },
    };
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      reviewOpenQuestionExtractor,
      configOverrides: {
        anthropic: {
          models: {
            cognition: "cog-model",
            background: "bg-model",
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Misattributed meeting",
          narrative: "Alex led the meeting, but the participants only mention the team.",
          participants: ["team"],
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
        },
        [0, 1, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });
    await harness.flushHookLogs();

    expect(result.errors).toEqual([]);
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: OVERSEER_TOOL_NAME,
    });
    expect(llm.requests[0]?.model).toBe("bg-model");
    expect(result.changes[0]).toMatchObject({
      action: "flag",
      targets: {
        kind: "misattribution",
      },
    });
    expect(harness.reviewQueueRepository.getOpen()[0]).toMatchObject({
      kind: "misattribution",
    });
    expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([
      expect.objectContaining({
        source: "overseer",
        question: "¿Qué atribución debería corregirse?",
        urgency: 0.61,
      }),
    ]);

    const auditRow = harness.auditLog.list({ process: "overseer" })[0];
    await harness.auditLog.revert(auditRow!.id, "test");
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
  });

  it("stays quiet on clean fixtures", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient({
      responses: [createOverseerResponse([], 8, 4)],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Clean semantic memory",
          description: "This proposition is aligned with the supporting evidence.",
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
          last_verified_at: nowMs - 1_000,
          source_episode_ids: [createEpisodeFixture().id],
        },
        [0, 0, 1, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });

    expect(result.changes).toEqual([]);
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
  });

  it("queues temporal drift reviews for semantic edges without mutating them", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const suggestedValidTo = nowMs - 500;
    const llm = new FakeLLMClient({
      responses: [
        createOverseerResponse([
          {
            kind: "temporal_drift",
            reason: "The edge only held before the later rollback evidence.",
            confidence: 0.82,
            suggested_valid_to: suggestedValidTo,
          },
        ]),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      configOverrides: {
        offline: {
          ...DEFAULT_CONFIG.offline,
          overseer: {
            ...DEFAULT_CONFIG.offline.overseer,
            maxChecksPerRun: 1,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    const episodeId = createEpisodeFixture().id;
    const first = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Atlas edge source",
          description: "Atlas was stable.",
          source_episode_ids: [episodeId],
          created_at: nowMs - 2_000,
          updated_at: nowMs - 2_000,
        },
        [1, 0, 0, 0],
      ),
    );
    const second = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Atlas edge target",
          description: "Rollback had completed.",
          source_episode_ids: [episodeId],
          created_at: nowMs - 1_900,
          updated_at: nowMs - 1_900,
        },
        [0, 1, 0, 0],
      ),
    );
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "supports",
      confidence: 0.8,
      evidence_episode_ids: [episodeId],
      created_at: nowMs - 100,
      last_verified_at: nowMs - 100,
      valid_from: nowMs - 100,
    });

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });

    expect(result.errors).toEqual([]);
    expect(result.changes[0]).toMatchObject({
      targets: {
        kind: "temporal_drift",
        target_type: "semantic_edge",
        target_id: edge.id,
      },
      preview: {
        suggested_valid_to: suggestedValidTo,
      },
    });
    expect(harness.reviewQueueRepository.getOpen()[0]).toMatchObject({
      kind: "temporal_drift",
      refs: {
        target_type: "semantic_edge",
        target_kind: "semantic_edge",
        target_id: edge.id,
        suggested_valid_to: suggestedValidTo,
      },
    });
    expect(harness.semanticEdgeRepository.getEdge(edge.id)?.valid_to).toBeNull();
  });

  it("halts further llm work after budget exhaustion", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createOverseerResponse(
          [
            {
              kind: "misattribution",
              reason: "First target issue.",
              confidence: 0.8,
            },
          ],
          35,
          25,
        ),
        createOverseerResponse(
          [
            {
              kind: "temporal_drift",
              reason: "Second target issue.",
              confidence: 0.8,
            },
          ],
          35,
          25,
        ),
        createOverseerResponse(
          [
            {
              kind: "identity_inconsistency",
              reason: "Third target issue.",
              confidence: 0.8,
            },
          ],
          35,
          25,
        ),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      configOverrides: {
        offline: {
          ...DEFAULT_CONFIG.offline,
          overseer: {
            ...DEFAULT_CONFIG.offline.overseer,
            maxChecksPerRun: 3,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Target one",
          created_at: 1_000,
          updated_at: 1_000,
        },
        [1, 0, 0, 0],
      ),
    );
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Target two",
          created_at: 2_000,
          updated_at: 2_000,
        },
        [1, 0, 0, 0],
      ),
    );
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Target three",
          created_at: 3_000,
          updated_at: 3_000,
        },
        [1, 0, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    const result = await process.run(harness.createContext(), {
      dryRun: false,
      budget: 100,
    });

    expect(result.budget_exhausted).toBe(true);
    expect(llm.requests).toHaveLength(2);
    expect(result.changes).toHaveLength(1);
  });

  it("respects the lookback window when prior audit history is stale", async () => {
    const nowMs = 20 * 24 * 60 * 60 * 1_000;
    const clock = new ManualClock(nowMs - 10 * 24 * 60 * 60 * 1_000);
    const recentEpisode = createEpisodeFixture(
      {
        title: "Recent target",
        created_at: nowMs - 60 * 60 * 1_000,
        updated_at: nowMs - 60 * 60 * 1_000,
      },
      [1, 0, 0, 0],
    );
    const oldEpisode = createEpisodeFixture(
      {
        title: "Old target",
        created_at: nowMs - 5 * 24 * 60 * 60 * 1_000,
        updated_at: nowMs - 5 * 24 * 60 * 60 * 1_000,
      },
      [1, 0, 0, 0],
    );
    const llm = new FakeLLMClient({
      responses: [
        createOverseerResponse([
          {
            kind: "misattribution",
            reason: "Recent target issue.",
            confidence: 0.8,
          },
        ]),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock,
      llmClient: llm,
      configOverrides: {
        offline: {
          ...DEFAULT_CONFIG.offline,
          overseer: {
            ...DEFAULT_CONFIG.offline.overseer,
            lookbackHours: 24,
            maxChecksPerRun: 8,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    harness.auditLog.record({
      run_id: createMaintenanceRunId(),
      process: "overseer",
      action: "flag",
      targets: {
        seed: true,
      },
      reversal: {},
    });
    clock.set(nowMs);

    await harness.episodicRepository.insert(oldEpisode);
    await harness.episodicRepository.insert(recentEpisode);

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });

    expect(result.errors).toEqual([]);
    expect(llm.requests).toHaveLength(1);
    expect(harness.reviewQueueRepository.getOpen()[0]).toMatchObject({
      refs: {
        target_id: recentEpisode.id,
      },
    });
  });

  it("continues and logs when the review-to-open-question hook fails", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient({
      responses: [
        createOverseerResponse([
          {
            kind: "misattribution",
            reason: "The narrative mentions Alex, but Alex is missing from participants.",
            confidence: 0.8,
          },
        ]),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    const reviewQueueRepository = harness.reviewQueueRepository as unknown as {
      options: {
        onEnqueue?: (item: unknown, input: unknown) => void;
      };
    };
    const originalHook = reviewQueueRepository.options.onEnqueue;
    reviewQueueRepository.options.onEnqueue = () => {
      throw new Error("hook exploded");
    };

    try {
      await harness.episodicRepository.insert(
        createEpisodeFixture(
          {
            title: "Misattributed meeting",
            narrative: "Alex led the meeting, but the participants only mention the team.",
            participants: ["team"],
            created_at: nowMs - 1_000,
            updated_at: nowMs - 1_000,
          },
          [0, 1, 0, 0],
        ),
      );

      const process = new OverseerProcess({
        reviewQueueRepository: harness.reviewQueueRepository,
        registry: harness.registry,
      });

      const result = await process.run(harness.createContext(), {
        dryRun: false,
      });

      await harness.flushHookLogs();

      const entries = new StreamReader({
        dataDir: harness.tempDir,
      }).tail(1);

      expect(result.errors).toEqual([]);
      expect(harness.reviewQueueRepository.getOpen()).toHaveLength(1);
      expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([]);
      expect(entries[0]).toMatchObject({
        kind: "internal_event",
        content: {
          hook: "review_queue_open_question",
        },
      });
    } finally {
      reviewQueueRepository.options.onEnqueue = originalHook;
    }
  });
});
