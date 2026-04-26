import { afterEach, describe, expect, it } from "vitest";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../offline/test-support.js";
import { SemanticGraph } from "../memory/semantic/index.js";
import { ManualClock } from "../util/clock.js";
import { resolveSemanticContext, toRetrievedSemantic } from "./semantic-retrieval.js";

describe("resolveSemanticContext temporal validity", () => {
  let harness: Awaited<ReturnType<typeof createOfflineTestHarness>> | undefined;

  afterEach(async () => {
    await harness?.cleanup();
    harness = undefined;
  });

  it("windows contradiction walks at the requested semantic as-of", async () => {
    const clock = new ManualClock(1_000_000);
    harness = await createOfflineTestHarness({ clock });
    const episode = createEpisodeFixture({
      title: "Atlas deployment note",
      tags: ["atlas"],
    });
    await harness.episodicRepository.insert(episode);
    const atlas = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        kind: "entity",
        label: "Atlas",
        description: "Atlas deployment service",
        source_episode_ids: [episode.id],
      }),
    );
    const contradiction = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        kind: "proposition",
        label: "Atlas needs no deploy work",
        description: "A stale claim that Atlas deployment needs no action.",
        source_episode_ids: [episode.id],
      }),
    );
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: contradiction.id,
      relation: "contradicts",
      confidence: 0.7,
      evidence_episode_ids: [episode.id],
      created_at: 1_000_000,
      last_verified_at: 1_000_000,
    });
    harness.semanticEdgeRepository.invalidateEdge(edge.id, {
      at: 1_000_500,
      by_process: "manual",
    });
    clock.set(1_001_000);
    const semanticGraph = new SemanticGraph({
      nodeRepository: harness.semanticNodeRepository,
      edgeRepository: harness.semanticEdgeRepository,
    });

    const current = await resolveSemanticContext(
      "Atlas",
      {
        graphWalkDepth: 1,
        maxGraphNodes: 4,
      },
      {
        embeddingClient: harness.embeddingClient,
        episodicRepository: harness.episodicRepository,
        semanticNodeRepository: harness.semanticNodeRepository,
        semanticGraph,
      },
    );
    const historical = await resolveSemanticContext(
      "Atlas",
      {
        graphWalkDepth: 1,
        maxGraphNodes: 4,
        asOf: 1_000_250,
      },
      {
        embeddingClient: harness.embeddingClient,
        episodicRepository: harness.episodicRepository,
        semanticNodeRepository: harness.semanticNodeRepository,
        semanticGraph,
      },
    );

    expect(current.context.contradicts).toEqual([]);
    expect(current.contradictionPresent).toBe(false);
    expect(current.contradictionHits).toEqual([]);
    expect(historical.context.contradicts.map((node) => node.id)).toContain(contradiction.id);
    expect(historical.contradictionPresent).toBe(true);
    expect(historical.contradictionHits.some((hit) => hit.edgePath[0]?.id === edge.id)).toBe(true);
  });

  it("tags directly matched propositions whose support edges are all closed", async () => {
    const clock = new ManualClock(1_000_000);
    harness = await createOfflineTestHarness({ clock });
    const episode = createEpisodeFixture({
      title: "Atlas install note",
      tags: ["atlas"],
    });
    await harness.episodicRepository.insert(episode);
    const proposition = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        kind: "proposition",
        label: "Atlas requires pnpm install",
        description: "Atlas deployment currently requires rerunning pnpm install.",
        source_episode_ids: [episode.id],
      }),
    );
    const support = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        kind: "proposition",
        label: "pnpm install fixed Atlas",
        description: "Rerunning pnpm install fixed a previous Atlas deployment failure.",
        source_episode_ids: [episode.id],
      }),
    );
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: support.id,
      to_node_id: proposition.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: [episode.id],
      created_at: 1_000_000,
      last_verified_at: 1_000_000,
    });
    harness.semanticEdgeRepository.invalidateEdge(edge.id, {
      at: 1_000_500,
      by_process: "manual",
    });
    clock.set(1_001_000);
    const semanticGraph = new SemanticGraph({
      nodeRepository: harness.semanticNodeRepository,
      edgeRepository: harness.semanticEdgeRepository,
    });

    const current = await resolveSemanticContext(
      "Atlas requires pnpm install",
      {
        graphWalkDepth: 1,
        maxGraphNodes: 4,
      },
      {
        embeddingClient: harness.embeddingClient,
        episodicRepository: harness.episodicRepository,
        semanticNodeRepository: harness.semanticNodeRepository,
        semanticGraph,
      },
    );
    const beforeClosure = await resolveSemanticContext(
      "Atlas requires pnpm install",
      {
        graphWalkDepth: 1,
        maxGraphNodes: 4,
        asOf: 1_000_250,
      },
      {
        embeddingClient: harness.embeddingClient,
        episodicRepository: harness.episodicRepository,
        semanticNodeRepository: harness.semanticNodeRepository,
        semanticGraph,
      },
    );

    expect(current.matchedNodes.find((node) => node.id === proposition.id)).toMatchObject({
      historical: true,
    });
    expect(beforeClosure.matchedNodes.find((node) => node.id === proposition.id)).toMatchObject({
      id: proposition.id,
    });
    expect(
      beforeClosure.matchedNodes.find((node) => node.id === proposition.id)?.historical,
    ).toBeUndefined();
  });

  it("walks causal edges outward into causal hits", async () => {
    const clock = new ManualClock(1_000_000);
    harness = await createOfflineTestHarness({ clock });
    const episode = createEpisodeFixture({
      title: "Atlas causal note",
      tags: ["atlas"],
    });
    await harness.episodicRepository.insert(episode);
    const cause = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        kind: "proposition",
        label: "Atlas failed deploys",
        description: "Atlas failed deploys create extra rollback pressure.",
        source_episode_ids: [episode.id],
      }),
    );
    const effect = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        kind: "proposition",
        label: "Rollback pressure rises",
        description: "Rollback pressure rises when Atlas deploys fail.",
        source_episode_ids: [episode.id],
      }),
    );
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: cause.id,
      to_node_id: effect.id,
      relation: "causes",
      confidence: 0.7,
      evidence_episode_ids: [episode.id],
      created_at: 1_000_000,
      last_verified_at: 1_000_000,
    });
    const semanticGraph = new SemanticGraph({
      nodeRepository: harness.semanticNodeRepository,
      edgeRepository: harness.semanticEdgeRepository,
    });

    const fromCause = toRetrievedSemantic(
      await resolveSemanticContext(
        "Atlas failed deploys",
        {
          graphWalkDepth: 1,
          maxGraphNodes: 4,
        },
        {
          embeddingClient: harness.embeddingClient,
          episodicRepository: harness.episodicRepository,
          semanticNodeRepository: harness.semanticNodeRepository,
          semanticGraph,
        },
      ),
    );
    const fromEffect = toRetrievedSemantic(
      await resolveSemanticContext(
        "Rollback pressure rises",
        {
          graphWalkDepth: 1,
          maxGraphNodes: 4,
        },
        {
          embeddingClient: harness.embeddingClient,
          episodicRepository: harness.episodicRepository,
          semanticNodeRepository: harness.semanticNodeRepository,
          semanticGraph,
        },
      ),
    );

    expect(fromCause.causal_hits).toEqual([
      expect.objectContaining({
        root_node_id: cause.id,
        node: expect.objectContaining({ id: effect.id }),
        edgePath: [expect.objectContaining({ id: edge.id, relation: "causes" })],
      }),
    ]);
    expect(fromEffect.causal_hits).toEqual([]);
  });
});
