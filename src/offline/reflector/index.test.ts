import { afterEach, describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "../../config/index.js";
import { FakeLLMClient } from "../../llm/index.js";

import { createEpisodeFixture, createOfflineTestHarness } from "../test-support.js";
import { ReflectorProcess } from "./index.js";

const REFLECTOR_TOOL_NAME = "EmitReflectorInsights";

function createReflectorResponse(input: {
  label: string;
  description: string;
  confidence: number;
  source_episode_ids: string[];
}) {
  return {
    text: "",
    input_tokens: 18,
    output_tokens: 12,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: REFLECTOR_TOOL_NAME,
        input,
      },
    ],
  };
}

describe("reflector process", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("creates low-confidence insights with review items and supports reversal", async () => {
    const episodes = [
      createEpisodeFixture(
        {
          title: "Rollback drill",
          narrative: "The team practiced a rollback plan before a deploy.",
          tags: ["deploy-pattern"],
          created_at: 10_000,
          updated_at: 10_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Staging deploy",
          narrative: "A documented rollback plan reduced confusion during staging deploys.",
          tags: ["deploy-pattern"],
          created_at: 20_000,
          updated_at: 20_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Production deploy",
          narrative: "Rollback planning helped the production deploy stay calm.",
          tags: ["deploy-pattern"],
          created_at: 30_000,
          updated_at: 30_000,
        },
        [1, 0, 0, 0],
      ),
    ];
    const llm = new FakeLLMClient({
      responses: [
        createReflectorResponse({
          label: "Deploys stabilize when rollback plans are documented",
          description:
            "Across the supporting episodes, explicit rollback plans correlate with steadier deploys.",
          confidence: 0.8,
          source_episode_ids: episodes.map((episode) => episode.id),
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      configOverrides: {
        offline: {
          ...DEFAULT_CONFIG.offline,
          reflector: {
            ...DEFAULT_CONFIG.offline.reflector,
            ceilingConfidence: 0.9,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    for (const episode of episodes) {
      await harness.episodicRepository.insert(episode);
    }

    const process = new ReflectorProcess({
      semanticNodeRepository: harness.semanticNodeRepository,
      semanticEdgeRepository: harness.semanticEdgeRepository,
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });

    expect(result.errors).toEqual([]);
    expect(result.changes).toHaveLength(1);
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: REFLECTOR_TOOL_NAME,
    });

    const nodes = await harness.semanticNodeRepository.list({
      includeArchived: true,
      limit: 10,
    });
    const insightNode = nodes.find((node) =>
      node.label.includes("Deploys stabilize when rollback plans are documented"),
    );

    expect(insightNode?.confidence).toBe(0.5);
    expect(harness.reviewQueueRepository.getOpen()).toEqual([
      expect.objectContaining({
        kind: "new_insight",
      }),
    ]);
    expect(
      harness.semanticEdgeRepository
        .listEdges({ relation: "supports" })
        .some((edge) => edge.to_node_id === insightNode?.id),
    ).toBe(true);

    const auditRow = harness.auditLog.list({ process: "reflector" })[0];
    await harness.auditLog.revert(auditRow!.id, "test");

    const remainingNodes = await harness.semanticNodeRepository.list({
      includeArchived: true,
      limit: 10,
    });
    expect(
      remainingNodes.some((node) =>
        node.label.includes("Deploys stabilize when rollback plans are documented"),
      ),
    ).toBe(false);
    expect(harness.semanticEdgeRepository.listEdges({ relation: "supports" })).toEqual([]);
  });

  it("skips insights when support is insufficient or provenance is hallucinated", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          ...createReflectorResponse({
            label: "Bad insight",
            description: "This cites a missing episode.",
            confidence: 0.6,
            source_episode_ids: ["ep_missing"],
          }),
          input_tokens: 5,
          output_tokens: 5,
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Only one note",
          tags: ["solo"],
          created_at: 10_000,
          updated_at: 10_000,
        },
        [0, 0, 1, 0],
      ),
    );

    const process = new ReflectorProcess({
      semanticNodeRepository: harness.semanticNodeRepository,
      semanticEdgeRepository: harness.semanticEdgeRepository,
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });

    const lowSupport = await process.run(harness.createContext(), {
      dryRun: false,
    });
    expect(lowSupport.changes).toEqual([]);

    const supportedEpisodes = [
      createEpisodeFixture(
        {
          title: "Pattern one",
          tags: ["pattern"],
          created_at: 20_000,
          updated_at: 20_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          title: "Pattern two",
          tags: ["pattern"],
          created_at: 30_000,
          updated_at: 30_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          title: "Pattern three",
          tags: ["pattern"],
          created_at: 40_000,
          updated_at: 40_000,
        },
        [0, 0, 1, 0],
      ),
    ];

    for (const episode of supportedEpisodes) {
      await harness.episodicRepository.insert(episode);
    }

    const hallucinated = await process.run(harness.createContext(), {
      dryRun: false,
    });
    expect(hallucinated.errors[0]?.message).toContain("outside the support set");
    expect(
      await harness.semanticNodeRepository.list({
        includeArchived: true,
        limit: 10,
      }),
    ).toEqual([]);
  });

  it("halts further llm work after budget exhaustion", async () => {
    const llm = new FakeLLMClient({
      responses: [
        ({ messages }) => {
          const ids = [...messages[0]!.content.matchAll(/"id":"(ep_[a-z0-9]{16})"/g)].map(
            (match) => match[1]!,
          );

          return {
            text: "",
            input_tokens: 35,
            output_tokens: 25,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "toolu_1",
                name: REFLECTOR_TOOL_NAME,
                input: {
                  label: `Insight from ${ids[0]}`,
                  description: "Pattern insight.",
                  confidence: 0.6,
                  source_episode_ids: ids,
                },
              },
            ],
          };
        },
        ({ messages }) => {
          const ids = [...messages[0]!.content.matchAll(/"id":"(ep_[a-z0-9]{16})"/g)].map(
            (match) => match[1]!,
          );

          return {
            text: "",
            input_tokens: 35,
            output_tokens: 25,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "toolu_2",
                name: REFLECTOR_TOOL_NAME,
                input: {
                  label: `Insight from ${ids[0]}`,
                  description: "Pattern insight.",
                  confidence: 0.6,
                  source_episode_ids: ids,
                },
              },
            ],
          };
        },
        ({ messages }) => {
          const ids = [...messages[0]!.content.matchAll(/"id":"(ep_[a-z0-9]{16})"/g)].map(
            (match) => match[1]!,
          );

          return {
            text: "",
            input_tokens: 35,
            output_tokens: 25,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "toolu_3",
                name: REFLECTOR_TOOL_NAME,
                input: {
                  label: `Insight from ${ids[0]}`,
                  description: "Pattern insight.",
                  confidence: 0.6,
                  source_episode_ids: ids,
                },
              },
            ],
          };
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      configOverrides: {
        offline: {
          ...DEFAULT_CONFIG.offline,
          reflector: {
            ...DEFAULT_CONFIG.offline.reflector,
            maxInsightsPerRun: 3,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    const episodes = [
      createEpisodeFixture(
        {
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Alpha one",
          tags: ["alpha"],
          created_at: 1_000,
          updated_at: 1_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          title: "Alpha two",
          tags: ["alpha"],
          created_at: 2_000,
          updated_at: 2_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          id: "ep_cccccccccccccccc" as never,
          title: "Alpha three",
          tags: ["alpha"],
          created_at: 3_000,
          updated_at: 3_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          id: "ep_dddddddddddddddd" as never,
          title: "Beta one",
          tags: ["beta"],
          created_at: 4_000,
          updated_at: 4_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          id: "ep_eeeeeeeeeeeeeeee" as never,
          title: "Beta two",
          tags: ["beta"],
          created_at: 5_000,
          updated_at: 5_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          id: "ep_ffffffffffffffff" as never,
          title: "Beta three",
          tags: ["beta"],
          created_at: 6_000,
          updated_at: 6_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          id: "ep_gggggggggggggggg" as never,
          title: "Gamma one",
          tags: ["gamma"],
          created_at: 7_000,
          updated_at: 7_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          id: "ep_hhhhhhhhhhhhhhhh" as never,
          title: "Gamma two",
          tags: ["gamma"],
          created_at: 8_000,
          updated_at: 8_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          id: "ep_iiiiiiiiiiiiiiii" as never,
          title: "Gamma three",
          tags: ["gamma"],
          created_at: 9_000,
          updated_at: 9_000,
        },
        [0, 0, 1, 0],
      ),
    ];

    for (const episode of episodes) {
      await harness.episodicRepository.insert(episode);
    }

    const process = new ReflectorProcess({
      semanticNodeRepository: harness.semanticNodeRepository,
      semanticEdgeRepository: harness.semanticEdgeRepository,
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
});
