import { afterEach, describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "../../config/index.js";
import { FakeLLMClient } from "../../llm/index.js";
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
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
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

    expect(result.errors).toEqual([]);
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: OVERSEER_TOOL_NAME,
    });
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
