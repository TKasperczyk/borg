import { afterEach, describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";

import { createEpisodeFixture, createOfflineTestHarness } from "../test-support.js";
import { ConsolidatorProcess } from "./index.js";

const CONSOLIDATION_TOOL_NAME = "EmitConsolidation";

function createConsolidationResponse(title: string, narrative: string) {
  return {
    text: "",
    input_tokens: 20,
    output_tokens: 15,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: CONSOLIDATION_TOOL_NAME,
        input: { title, narrative },
      },
    ],
  };
}

async function snapshotEpisodes(harness: Awaited<ReturnType<typeof createOfflineTestHarness>>) {
  const episodes = await harness.episodicRepository.listAll();
  const stats = harness.episodicRepository.listStats();
  const mergedIds = new Set(
    episodes
      .filter((episode) => episode.lineage.derived_from.length > 0)
      .map((episode) => episode.id),
  );

  return {
    episodes: episodes
      .map((episode) => ({
        ...episode,
        id: mergedIds.has(episode.id) ? "<merged>" : episode.id,
        embedding: Array.from(episode.embedding),
        lineage: {
          derived_from: episode.lineage.derived_from,
          supersedes: episode.lineage.supersedes.map((id) => (mergedIds.has(id) ? "<merged>" : id)),
        },
      }))
      .sort((left, right) => left.id.localeCompare(right.id)),
    stats: stats
      .map((stat) => ({
        ...stat,
        episode_id: mergedIds.has(stat.episode_id) ? "<merged>" : stat.episode_id,
      }))
      .sort((left, right) => left.episode_id.localeCompare(right.episode_id)),
  };
}

describe("consolidator process", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("detects redundant clusters, consolidates them, and supports reversal", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createConsolidationResponse(
          "Merged planning incident",
          "The team merged two overlapping planning notes into one grounded summary. The merged episode preserves both source anchors.",
        ),
        createConsolidationResponse(
          "Merged planning incident",
          "The team merged two overlapping planning notes into one grounded summary. The merged episode preserves both source anchors.",
        ),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    const first = createEpisodeFixture(
      {
        title: "Sprint planning note",
        narrative: "The team planned the sprint and listed the deploy checklist.",
        tags: ["planning", "deploy"],
        created_at: 10_000,
        updated_at: 10_000,
      },
      [0, 1, 0, 0],
    );
    const second = createEpisodeFixture(
      {
        title: "Sprint planning follow-up",
        narrative: "The same planning session captured the deploy checklist again.",
        tags: ["planning", "deploy"],
        created_at: 20_000,
        updated_at: 20_000,
      },
      [0, 0.99, 0, 0],
    );

    await harness.episodicRepository.insert(first);
    await harness.episodicRepository.insert(second);
    harness.episodicRepository.updateStats(first.id, {
      tier: "T2",
    });

    const process = new ConsolidatorProcess({
      episodicRepository: harness.episodicRepository,
      registry: harness.registry,
    });

    const dryRun = await process.run(harness.createContext(), {
      dryRun: true,
    });
    expect(dryRun.changes).toHaveLength(1);
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: CONSOLIDATION_TOOL_NAME,
    });
    expect((await harness.episodicRepository.listAll()).map((episode) => episode.id)).toEqual([
      second.id,
      first.id,
    ]);

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });
    expect(result.errors).toEqual([]);
    expect(result.tokens_used).toBe(35);

    const episodes = await harness.episodicRepository.listAll();
    const merged = episodes.find((episode) => episode.title === "Merged planning incident");

    expect(merged).toBeDefined();
    expect(merged?.lineage.derived_from).toEqual([first.id, second.id]);
    // Sprint 53: lineage now points the right way -- the merged episode
    // supersedes its sources, sources stay untouched in lineage.
    expect(merged?.lineage.supersedes).toEqual([first.id, second.id]);
    expect(harness.episodicRepository.getStats(merged!.id)?.tier).toBe("T2");
    expect(harness.episodicRepository.getStats(first.id)?.archived).toBe(true);
    expect(harness.episodicRepository.getStats(second.id)?.archived).toBe(true);
    expect(
      (await harness.episodicRepository.get(first.id, { includeArchived: true }))?.lineage
        .supersedes,
    ).toEqual([]);

    const auditEntry = harness.auditLog.list({ process: "consolidator" })[0];
    expect(auditEntry?.action).toBe("consolidate");

    await harness.auditLog.revert(auditEntry!.id, "test");

    expect(await harness.episodicRepository.get(merged!.id)).toBeNull();
    expect(harness.episodicRepository.getStats(first.id)?.archived).toBe(false);
    expect(harness.episodicRepository.getStats(second.id)?.archived).toBe(false);
  });

  it("applies a saved plan without additional llm calls and matches a direct run", async () => {
    const response = createConsolidationResponse(
      "Merged deploy prep",
      "Two overlapping deploy-prep notes were merged into one grounded episode.",
    );
    const planHarness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [response],
      }),
    });
    cleanup.push(planHarness.cleanup);

    const first = createEpisodeFixture(
      {
        title: "Deploy prep one",
        narrative: "The team reviewed rollback steps before the Atlas deploy.",
        tags: ["deploy", "atlas"],
        created_at: 10_000,
        updated_at: 10_000,
      },
      [1, 0, 0, 0],
    );
    const second = createEpisodeFixture(
      {
        title: "Deploy prep two",
        narrative: "The Atlas deploy checklist repeated the rollback prep discussion.",
        tags: ["deploy", "atlas"],
        created_at: 20_000,
        updated_at: 20_000,
      },
      [0.99, 0, 0, 0],
    );

    await planHarness.episodicRepository.insert(first);
    await planHarness.episodicRepository.insert(second);

    const plannedProcess = new ConsolidatorProcess({
      episodicRepository: planHarness.episodicRepository,
      registry: planHarness.registry,
    });
    const planLlm = planHarness.llmClient as FakeLLMClient;
    const plan = await plannedProcess.plan(planHarness.createContext());

    expect(planLlm.requests).toHaveLength(1);
    expect(plan.items).toHaveLength(1);

    await plannedProcess.apply(planHarness.createContext(), plan);
    expect(planLlm.requests).toHaveLength(1);

    const directHarness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [response],
      }),
    });
    cleanup.push(directHarness.cleanup);

    await directHarness.episodicRepository.insert(first);
    await directHarness.episodicRepository.insert(second);

    const directProcess = new ConsolidatorProcess({
      episodicRepository: directHarness.episodicRepository,
      registry: directHarness.registry,
    });
    await directProcess.run(directHarness.createContext(), {
      dryRun: false,
    });

    expect(await snapshotEpisodes(planHarness)).toEqual(await snapshotEpisodes(directHarness));
    expect((directHarness.llmClient as FakeLLMClient).requests).toHaveLength(1);
  });

  it("clusters only episodes that share the same audience scope", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createConsolidationResponse(
          "Merged public cluster",
          "Two public notes were merged while scoped notes stayed isolated.",
        ),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    const sam = harness.entityRepository.resolve("Sam");
    const alex = harness.entityRepository.resolve("Alex");

    const publicEpisodes = [
      createEpisodeFixture(
        {
          title: "Public architecture note one",
          tags: ["scope-public"],
          audience_entity_id: null,
          shared: true,
          created_at: 10_000,
          updated_at: 10_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Public architecture note two",
          tags: ["scope-public"],
          audience_entity_id: null,
          shared: true,
          created_at: 20_000,
          updated_at: 20_000,
        },
        [0.99, 0, 0, 0],
      ),
    ];
    const differentAudienceEpisodes = [
      createEpisodeFixture(
        {
          title: "Sam-only architecture note",
          tags: ["scope-private"],
          audience_entity_id: sam,
          shared: false,
          created_at: 30_000,
          updated_at: 30_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Alex-only architecture note",
          tags: ["scope-private"],
          audience_entity_id: alex,
          shared: false,
          created_at: 40_000,
          updated_at: 40_000,
        },
        [0.99, 0, 0, 0],
      ),
    ];
    const mixedScopeEpisodes = [
      createEpisodeFixture(
        {
          title: "Public deploy note",
          tags: ["scope-mixed"],
          audience_entity_id: null,
          shared: true,
          created_at: 50_000,
          updated_at: 50_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Sam deploy note",
          tags: ["scope-mixed"],
          audience_entity_id: sam,
          shared: false,
          created_at: 60_000,
          updated_at: 60_000,
        },
        [0, 1, 0, 0],
      ),
    ];

    for (const episode of [
      ...publicEpisodes,
      ...differentAudienceEpisodes,
      ...mixedScopeEpisodes,
    ]) {
      await harness.episodicRepository.insert(episode);
    }

    const process = new ConsolidatorProcess({
      episodicRepository: harness.episodicRepository,
      registry: harness.registry,
    });
    const dryRun = await process.run(harness.createContext(), {
      dryRun: true,
    });

    expect(dryRun.changes).toHaveLength(1);
    expect(dryRun.changes[0]?.targets.source_ids).toEqual(
      [...publicEpisodes, mixedScopeEpisodes[0]!].map((episode) => episode.id),
    );
    expect(llm.requests).toHaveLength(1);
  });

  it("halts further llm work after budget exhaustion", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          ...createConsolidationResponse("Merged cluster A", "Merged cluster A."),
          input_tokens: 40,
          output_tokens: 30,
        },
        {
          ...createConsolidationResponse("Merged cluster B", "Merged cluster B."),
          input_tokens: 40,
          output_tokens: 30,
        },
        {
          ...createConsolidationResponse("Merged cluster C", "Merged cluster C."),
          input_tokens: 40,
          output_tokens: 30,
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    const episodes = [
      createEpisodeFixture(
        {
          title: "Alpha one",
          tags: ["alpha"],
          created_at: 1_000,
          updated_at: 1_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Alpha two",
          tags: ["alpha"],
          created_at: 2_000,
          updated_at: 2_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Beta one",
          tags: ["beta"],
          created_at: 3_000,
          updated_at: 3_000,
        },
        [0, 1, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Beta two",
          tags: ["beta"],
          created_at: 4_000,
          updated_at: 4_000,
        },
        [0, 1, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Gamma one",
          tags: ["gamma"],
          created_at: 5_000,
          updated_at: 5_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          title: "Gamma two",
          tags: ["gamma"],
          created_at: 6_000,
          updated_at: 6_000,
        },
        [0, 0, 1, 0],
      ),
    ];

    for (const episode of episodes) {
      await harness.episodicRepository.insert(episode);
    }

    const process = new ConsolidatorProcess({
      episodicRepository: harness.episodicRepository,
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
