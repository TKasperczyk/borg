import { afterEach, describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "../../config/index.js";
import { FakeLLMClient, type LLMCompleteOptions } from "../../llm/index.js";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
  TestEmbeddingClient,
} from "../test-support.js";
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

  it("gates low-confidence insights behind review acceptance and supports reversal", async () => {
    const insightLabel = "Deploys stabilize when rollback plans are documented";
    const insightDescription =
      "Across the supporting episodes, explicit rollback plans correlate with steadier deploys.";
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
          label: insightLabel,
          description: insightDescription,
          confidence: 0.8,
          source_episode_ids: episodes.map((episode) => episode.id),
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [insightLabel, [1, 0, 0, 0]],
          [`${insightLabel}\n${insightDescription}`, [1, 0, 0, 0]],
          ["Rollback plan", [1, 0, 0, 0]],
        ]),
      ),
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

    const evidenceAnchor = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          kind: "entity",
          label: "Rollback plan",
          description: "A rollback plan extracted from the source episode.",
          source_episode_ids: [episodes[0]!.id],
          confidence: 0.8,
        },
        [1, 0, 0, 0],
      ),
    );
    const process = new ReflectorProcess({
      semanticNodeRepository: harness.semanticNodeRepository,
      semanticEdgeRepository: harness.semanticEdgeRepository,
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
      clock: harness.clock,
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

    const nodesBeforeReview = await harness.semanticNodeRepository.list({
      includeArchived: true,
      limit: 10,
    });
    expect(
      nodesBeforeReview.some(
        (node) => node.label === "Deploys stabilize when rollback plans are documented",
      ),
    ).toBe(false);
    expect(harness.semanticEdgeRepository.listEdges({ relation: "supports" })).toEqual([]);

    const beforeRetrieval = await harness.retrievalPipeline.searchWithContext(
      "Deploys stabilize when rollback plans are documented",
      { limit: 1 },
    );
    expect(
      beforeRetrieval.semantic.matched_nodes.some(
        (node) => node.label === "Deploys stabilize when rollback plans are documented",
      ),
    ).toBe(false);

    const openReview = harness.reviewQueueRepository.getOpen()[0];
    expect(openReview).toEqual(
      expect.objectContaining({
        kind: "new_insight",
        refs: expect.objectContaining({
          evidence_cluster_key: "public:shared|tag:deploy-pattern",
          evidence_cluster_size: episodes.length,
          reflector_pending_insight: expect.objectContaining({
            candidate_support_edges: [
              expect.objectContaining({
                target_node_id: evidenceAnchor.id,
                source_episode_ids: [episodes[0]!.id],
              }),
            ],
            evidence_cluster: expect.objectContaining({
              key: "public:shared|tag:deploy-pattern",
              size: episodes.length,
            }),
          }),
        }),
      }),
    );

    await harness.reviewQueueRepository.resolve(openReview!.id, "accept");

    const nodes = await harness.semanticNodeRepository.list({
      includeArchived: true,
      limit: 10,
    });
    const insightNode = nodes.find(
      (node) => node.label === "Deploys stabilize when rollback plans are documented",
    );

    expect(insightNode?.confidence).toBe(0.5);
    expect(
      nodes.filter((node) => node.kind === "proposition" && /^Evidence cluster /.test(node.label)),
    ).toEqual([]);
    const supportEdges = harness.semanticEdgeRepository.listEdges({ relation: "supports" });
    expect(supportEdges).toEqual([
      expect.objectContaining({
        from_node_id: evidenceAnchor.id,
        to_node_id: insightNode?.id,
        evidence_episode_ids: [episodes[0]!.id],
      }),
    ]);
    expect(
      harness.semanticBeliefDependencyRepository.listBySourceEdge(supportEdges[0]!.id),
    ).toEqual([
      expect.objectContaining({
        target_type: "semantic_node",
        target_id: insightNode?.id,
        source_edge_id: supportEdges[0]!.id,
        dependency_kind: "supports",
      }),
    ]);

    const afterRetrieval = await harness.retrievalPipeline.searchWithContext(
      "Deploys stabilize when rollback plans are documented",
      { limit: 1 },
    );
    expect(afterRetrieval.semantic.matched_nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: insightNode?.id,
        }),
      ]),
    );

    // Sprint 52 regression: querying by the evidence anchor concept must
    // surface the insight via the supports-out walk. Pre-fix the supports
    // edge ran insight->target, so walking supports OUT from the matched
    // anchor found nothing and the insight stayed invisible.
    const anchorRetrieval = await harness.retrievalPipeline.searchWithContext("Rollback plan", {
      limit: 1,
    });
    expect(anchorRetrieval.semantic.support_hits.map((hit) => hit.node.id)).toContain(
      insightNode?.id,
    );

    const auditRow = harness.auditLog.list({ process: "reflector" })[0];
    await harness.auditLog.revert(auditRow!.id, "test");

    const reversedInsightNode =
      insightNode === undefined ? null : await harness.semanticNodeRepository.get(insightNode.id);
    expect(reversedInsightNode).toMatchObject({
      archived: true,
      confidence: 0.5,
      source_episode_ids: episodes.map((episode) => episode.id),
    });
    expect(harness.semanticEdgeRepository.listEdges({ relation: "supports" })).toEqual([]);
    expect(
      harness.semanticEdgeRepository.listEdges({ relation: "supports", includeInvalid: true }),
    ).toEqual([
      expect.objectContaining({
        id: supportEdges[0]?.id,
        valid_to: expect.any(Number),
        invalidated_by_process: "maintenance",
      }),
    ]);
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
  });

  it("keeps existing insight updates pending until review acceptance and restores snapshots", async () => {
    const previousEpisodes = [
      createEpisodeFixture(
        {
          title: "Previous rollback signal one",
          tags: ["existing-reflect"],
          created_at: 1_000,
          updated_at: 1_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Previous rollback signal two",
          tags: ["existing-reflect"],
          created_at: 2_000,
          updated_at: 2_000,
        },
        [1, 0, 0, 0],
      ),
    ];
    const updateEpisodes = [
      createEpisodeFixture(
        {
          title: "Updated rollback signal one",
          tags: ["update-reflect"],
          created_at: 3_000,
          updated_at: 3_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Updated rollback signal two",
          tags: ["update-reflect"],
          created_at: 4_000,
          updated_at: 4_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Updated rollback signal three",
          tags: ["update-reflect"],
          created_at: 5_000,
          updated_at: 5_000,
        },
        [1, 0, 0, 0],
      ),
    ];
    const llm = new FakeLLMClient({
      responses: [
        createReflectorResponse({
          label: "Rollback planning lowers deploy stress",
          description: "Updated evidence says rollback planning lowers deploy stress.",
          confidence: 0.8,
          source_episode_ids: updateEpisodes.map((episode) => episode.id),
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

    for (const episode of [...previousEpisodes, ...updateEpisodes]) {
      await harness.episodicRepository.insert(episode);
    }

    const existingNode = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Rollback planning lowers deploy stress",
          description: "Previous evidence says rollback planning lowers deploy stress.",
          confidence: 0.4,
          source_episode_ids: previousEpisodes.map((episode) => episode.id),
        },
        [1, 0, 0, 0],
      ),
    );
    const process = new ReflectorProcess({
      semanticNodeRepository: harness.semanticNodeRepository,
      semanticEdgeRepository: harness.semanticEdgeRepository,
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
      clock: harness.clock,
    });

    await process.run(harness.createContext(), {
      dryRun: false,
    });

    expect(await harness.semanticNodeRepository.get(existingNode.id)).toEqual(
      expect.objectContaining({
        description: "Previous evidence says rollback planning lowers deploy stress.",
        source_episode_ids: previousEpisodes.map((episode) => episode.id),
      }),
    );

    const pendingReview = harness.reviewQueueRepository.getOpen()[0];
    await harness.reviewQueueRepository.resolve(pendingReview!.id, "accept");

    expect(await harness.semanticNodeRepository.get(existingNode.id)).toEqual(
      expect.objectContaining({
        description: "Updated evidence says rollback planning lowers deploy stress.",
        source_episode_ids: [
          ...previousEpisodes.map((episode) => episode.id),
          ...updateEpisodes.map((episode) => episode.id),
        ],
      }),
    );

    const auditRow = harness.auditLog.list({ process: "reflector" })[0];
    await harness.auditLog.revert(auditRow!.id, "test");

    expect(await harness.semanticNodeRepository.get(existingNode.id)).toEqual(
      expect.objectContaining({
        description: "Previous evidence says rollback planning lowers deploy stress.",
        source_episode_ids: previousEpisodes.map((episode) => episode.id),
      }),
    );
  });

  it("rejects malformed reflector reversal edge ids before repository calls", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    let invalidateCalled = false;
    const originalInvalidate = harness.semanticEdgeRepository.invalidateEdge.bind(
      harness.semanticEdgeRepository,
    );
    harness.semanticEdgeRepository.invalidateEdge = ((
      ...args: Parameters<typeof harness.semanticEdgeRepository.invalidateEdge>
    ) => {
      invalidateCalled = true;
      return originalInvalidate(...args);
    }) as typeof harness.semanticEdgeRepository.invalidateEdge;

    new ReflectorProcess({
      semanticNodeRepository: harness.semanticNodeRepository,
      semanticEdgeRepository: harness.semanticEdgeRepository,
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
      clock: harness.clock,
    });
    const audit = harness.auditLog.record({
      run_id: harness.createContext().runId,
      process: "reflector",
      action: "insight",
      targets: {},
      reversal: {
        nodeId: createSemanticNodeFixture().id,
        nodeCreated: true,
        edgeIds: ["not-an-edge-id"],
      },
    });

    await expect(harness.auditLog.revert(audit.id, "test")).rejects.toThrow(
      "Invalid semantic edge id",
    );
    expect(invalidateCalled).toBe(false);
  });

  it.each([
    {
      label: "nodeId",
      reversal: {
        nodeId: "not-a-node-id",
        nodeCreated: true,
        edgeIds: [],
      },
    },
    {
      label: "anchorNodeId",
      reversal: {
        nodeId: createSemanticNodeFixture().id,
        nodeCreated: true,
        anchorNodeId: "not-a-node-id",
        edgeIds: [],
      },
    },
  ])("rejects malformed reflector reversal $label before node updates", async ({ reversal }) => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    let updateCalled = false;
    const originalUpdate = harness.semanticNodeRepository.update.bind(
      harness.semanticNodeRepository,
    );
    harness.semanticNodeRepository.update = (async (
      ...args: Parameters<typeof harness.semanticNodeRepository.update>
    ) => {
      updateCalled = true;
      return originalUpdate(...args);
    }) as typeof harness.semanticNodeRepository.update;

    new ReflectorProcess({
      semanticNodeRepository: harness.semanticNodeRepository,
      semanticEdgeRepository: harness.semanticEdgeRepository,
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
      clock: harness.clock,
    });
    const audit = harness.auditLog.record({
      run_id: harness.createContext().runId,
      process: "reflector",
      action: "insight",
      targets: {},
      reversal,
    });

    await expect(harness.auditLog.revert(audit.id, "test")).rejects.toThrow(
      "Invalid semantic node id",
    );
    expect(updateCalled).toBe(false);
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

  it("partitions reflection clusters by audience scope", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    const sam = harness.entityRepository.resolve("Sam");

    const publicEpisodes = [
      createEpisodeFixture(
        {
          title: "Public deploy note one",
          tags: ["scope-reflect"],
          audience_entity_id: null,
          shared: true,
          created_at: 10_000,
          updated_at: 10_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Public deploy note two",
          tags: ["scope-reflect"],
          audience_entity_id: null,
          shared: true,
          created_at: 20_000,
          updated_at: 20_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Public deploy note three",
          tags: ["scope-reflect"],
          audience_entity_id: null,
          shared: true,
          created_at: 30_000,
          updated_at: 30_000,
        },
        [1, 0, 0, 0],
      ),
    ];
    const scopedEpisodes = [
      createEpisodeFixture(
        {
          title: "Sam deploy note one",
          tags: ["scope-reflect"],
          audience_entity_id: sam,
          shared: false,
          created_at: 40_000,
          updated_at: 40_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Sam deploy note two",
          tags: ["scope-reflect"],
          audience_entity_id: sam,
          shared: false,
          created_at: 50_000,
          updated_at: 50_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Sam deploy note three",
          tags: ["scope-reflect"],
          audience_entity_id: sam,
          shared: false,
          created_at: 60_000,
          updated_at: 60_000,
        },
        [1, 0, 0, 0],
      ),
    ];

    llm.pushResponse(
      createReflectorResponse({
        label: "Sam deploy insight",
        description: "Sam-only deploy episodes imply a private deploy habit.",
        confidence: 0.6,
        source_episode_ids: scopedEpisodes.map((episode) => episode.id),
      }),
    );
    llm.pushResponse(
      createReflectorResponse({
        label: "Public deploy insight",
        description: "Public deploy episodes imply a reusable deploy habit.",
        confidence: 0.6,
        source_episode_ids: publicEpisodes.map((episode) => episode.id),
      }),
    );

    for (const episode of [...publicEpisodes, ...scopedEpisodes]) {
      await harness.episodicRepository.insert(episode);
    }

    const process = new ReflectorProcess({
      semanticNodeRepository: harness.semanticNodeRepository,
      semanticEdgeRepository: harness.semanticEdgeRepository,
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    const result = await process.run(harness.createContext(), {
      dryRun: true,
    });

    expect(result.changes).toHaveLength(2);
    expect(result.changes.map((change) => change.targets.cluster)).toEqual(
      expect.arrayContaining([
        "public:shared|tag:scope-reflect",
        `${sam}:private|tag:scope-reflect`,
      ]),
    );
  });

  it("does not update an existing semantic node from an incompatible audience scope", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    const sam = harness.entityRepository.resolve("Sam");

    const publicEpisodes = [
      createEpisodeFixture(
        {
          title: "Public pattern one",
          tags: ["scope-update"],
          audience_entity_id: null,
          shared: true,
          created_at: 10_000,
          updated_at: 10_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Public pattern two",
          tags: ["scope-update"],
          audience_entity_id: null,
          shared: true,
          created_at: 20_000,
          updated_at: 20_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Public pattern three",
          tags: ["scope-update"],
          audience_entity_id: null,
          shared: true,
          created_at: 30_000,
          updated_at: 30_000,
        },
        [1, 0, 0, 0],
      ),
    ];
    const scopedEpisodes = [
      createEpisodeFixture(
        {
          title: "Sam pattern one",
          tags: ["scope-update"],
          audience_entity_id: sam,
          shared: false,
          created_at: 40_000,
          updated_at: 40_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Sam pattern two",
          tags: ["scope-update"],
          audience_entity_id: sam,
          shared: false,
          created_at: 50_000,
          updated_at: 50_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Sam pattern three",
          tags: ["scope-update"],
          audience_entity_id: sam,
          shared: false,
          created_at: 60_000,
          updated_at: 60_000,
        },
        [1, 0, 0, 0],
      ),
    ];

    for (const episode of [...publicEpisodes, ...scopedEpisodes]) {
      await harness.episodicRepository.insert(episode);
    }

    const existingNode = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Shared label insight",
        description: "Public insight description.",
        source_episode_ids: publicEpisodes.map((episode) => episode.id),
      }),
    );

    llm.pushResponse(
      createReflectorResponse({
        label: "Shared label insight",
        description: "Sam insight description.",
        confidence: 0.6,
        source_episode_ids: scopedEpisodes.map((episode) => episode.id),
      }),
    );
    llm.pushResponse(
      createReflectorResponse({
        label: "Shared label insight",
        description: "Public insight description.",
        confidence: 0.6,
        source_episode_ids: publicEpisodes.map((episode) => episode.id),
      }),
    );

    const process = new ReflectorProcess({
      semanticNodeRepository: harness.semanticNodeRepository,
      semanticEdgeRepository: harness.semanticEdgeRepository,
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    await process.run(harness.createContext(), {
      dryRun: false,
    });
    for (const item of harness.reviewQueueRepository.getOpen()) {
      await harness.reviewQueueRepository.resolve(item.id, "accept");
    }

    const nodes = await harness.semanticNodeRepository.list({
      includeArchived: true,
      limit: 20,
    });
    const sharedLabelNodes = nodes.filter((node) => node.label === "Shared label insight");

    expect(sharedLabelNodes).toHaveLength(2);
    expect(sharedLabelNodes.find((node) => node.id === existingNode.id)?.description).toBe(
      "Public insight description.",
    );
    expect(
      sharedLabelNodes.find((node) => node.id !== existingNode.id)?.source_episode_ids,
    ).toEqual(scopedEpisodes.map((episode) => episode.id));
  });

  it("halts further llm work after budget exhaustion", async () => {
    const llm = new FakeLLMClient({
      responses: [
        ({ messages }: LLMCompleteOptions) => {
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
        ({ messages }: LLMCompleteOptions) => {
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
        ({ messages }: LLMCompleteOptions) => {
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

  it("can resurrect an archived matching node by label", async () => {
    const episodes = [
      createEpisodeFixture(
        {
          title: "Pattern one",
          tags: ["deploy-pattern"],
          created_at: 10_000,
          updated_at: 10_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Pattern two",
          tags: ["deploy-pattern"],
          created_at: 20_000,
          updated_at: 20_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Pattern three",
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
          label: "Rollback plans reduce deploy stress",
          description: "Fresh evidence says documented rollback plans reduce deploy stress.",
          confidence: 0.7,
          source_episode_ids: episodes.map((episode) => episode.id),
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    for (const episode of episodes) {
      await harness.episodicRepository.insert(episode);
    }

    const archived = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        kind: "proposition",
        label: "Rollback plans reduce deploy stress",
        description: "Archived stale insight",
        archived: true,
        confidence: 0.2,
        source_episode_ids: [episodes[0]!.id],
      }),
    );
    const process = new ReflectorProcess({
      semanticNodeRepository: harness.semanticNodeRepository,
      semanticEdgeRepository: harness.semanticEdgeRepository,
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });
    const pendingReview = harness.reviewQueueRepository.getOpen()[0];
    await harness.reviewQueueRepository.resolve(pendingReview!.id, "accept");

    const nodes = await harness.semanticNodeRepository.list({
      includeArchived: true,
      limit: 20,
    });
    const matchingNodes = nodes.filter((node) => node.label === archived.label);

    expect(result.errors).toEqual([]);
    expect(matchingNodes).toHaveLength(1);
    expect(matchingNodes[0]).toMatchObject({
      id: archived.id,
      archived: false,
    });
  });
});
