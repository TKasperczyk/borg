import { z } from "zod";

import { FixedClock } from "../../src/index.js";
import { createEpisodeFixture, createOfflineTestHarness } from "../../src/offline/test-support.js";

import { DeterministicEmbeddingClient } from "../support/embedding.js";
import { loadMetricFixtures } from "../support/fixtures.js";
import type { EvalCaseResult, EvalMetricModule, EvalMetricResult } from "../support/scorecard.js";

const METRIC_NAME = "contradiction_detection";
const METRIC_DESCRIPTION =
  "Tests graph-based contradiction surfacing with an explicit contradicts edge, not claim-level contradiction derivation.";
const EMBEDDING_DIMS = 64;

const contradictionFixtureSchema = z.object({
  name: z.string().min(1),
  query: z.string().min(1),
  episodes: z.array(
    z.object({
      id: z.string().min(1),
      title: z.string().min(1),
      narrative: z.string().min(1),
      tags: z.array(z.string().min(1)),
      participants: z.array(z.string().min(1)),
      start_time: z.number().finite(),
      end_time: z.number().finite(),
    }),
  ),
  nodes: z.array(
    z.object({
      id: z.string().min(1),
      label: z.string().min(1),
      description: z.string().min(1),
      source_episode_ids: z.array(z.string().min(1)).min(1),
    }),
  ),
  contradiction_edge: z.object({
    from_node_id: z.string().min(1),
    to_node_id: z.string().min(1),
    confidence: z.number().min(0).max(1),
  }),
  expected_contradiction_node_id: z.string().min(1),
});

type ContradictionFixture = z.infer<typeof contradictionFixtureSchema>;

function buildEpisodeEmbeddingText(episode: ContradictionFixture["episodes"][number]): string {
  return [
    episode.title,
    episode.narrative,
    episode.tags.join(" "),
    episode.participants.join(" "),
  ].join("\n");
}

export const contradictionDetectionMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const fixtures = loadMetricFixtures(METRIC_NAME, contradictionFixtureSchema);
    const cases: EvalCaseResult[] = [];
    let passed = true;
    let surfacedCount = 0;

    for (const fixture of fixtures) {
      const embeddingClient = new DeterministicEmbeddingClient(EMBEDDING_DIMS);
      const harness = await createOfflineTestHarness({
        clock: new FixedClock(40_000),
        embeddingClient,
        embeddingDimensions: EMBEDDING_DIMS,
      });

      try {
        for (const episode of fixture.data.episodes) {
          await harness.episodicRepository.insert(
            createEpisodeFixture({
              id: episode.id as never,
              title: episode.title,
              narrative: episode.narrative,
              tags: episode.tags,
              participants: episode.participants,
              start_time: episode.start_time,
              end_time: episode.end_time,
              created_at: episode.start_time,
              updated_at: episode.end_time,
              embedding: await embeddingClient.embed(buildEpisodeEmbeddingText(episode)),
            }),
          );
        }

        for (const node of fixture.data.nodes) {
          await harness.semanticNodeRepository.insert({
            id: node.id as never,
            kind: "proposition",
            label: node.label,
            description: node.description,
            aliases: [],
            confidence: 0.8,
            source_episode_ids: node.source_episode_ids as never,
            created_at: 40_000,
            updated_at: 40_000,
            last_verified_at: 40_000,
            embedding: await embeddingClient.embed(`${node.label}\n${node.description}`),
            archived: false,
            superseded_by: null,
          });
        }

        harness.semanticEdgeRepository.addEdge({
          from_node_id: fixture.data.contradiction_edge.from_node_id as never,
          to_node_id: fixture.data.contradiction_edge.to_node_id as never,
          relation: "contradicts",
          confidence: fixture.data.contradiction_edge.confidence,
          evidence_episode_ids: fixture.data.nodes[0]?.source_episode_ids as never,
          created_at: 40_000,
          last_verified_at: 40_000,
        });

        const result = await harness.retrievalPipeline.searchWithContext(fixture.data.query, {
          limit: 5,
          graphWalkDepth: 1,
          maxGraphNodes: 8,
        });
        const actualContradictionNodeIds = result.semantic.contradiction_hits.map((hit) => hit.node.id);
        const actualContradictionHitCount = actualContradictionNodeIds.length;
        const casePassed =
          result.contradiction_present &&
          actualContradictionHitCount >= 1 &&
          actualContradictionNodeIds.includes(fixture.data.expected_contradiction_node_id as never);

        surfacedCount += casePassed ? 1 : 0;
        passed &&= casePassed;
        cases.push({
          name: fixture.name,
          passed: casePassed,
          actual: {
            contradiction_present: result.contradiction_present,
            contradiction_hit_count: actualContradictionHitCount,
            contradiction_hit_ids: actualContradictionNodeIds,
          },
          expected: {
            contradiction_present: true,
            minimum_contradiction_hit_count: 1,
            expected_hit_id: fixture.data.expected_contradiction_node_id,
            extra_hits_allowed: true,
          },
          note:
            "Extra contradiction hits are allowed because retrieval may surface both sides of an explicit contradicts edge.",
        });
      } finally {
        await harness.cleanup();
      }
    }

    return {
      name: METRIC_NAME,
      description: METRIC_DESCRIPTION,
      passed,
      actual: {
        contradiction_cases: `${surfacedCount}/${fixtures.length}`,
      },
      expected: {
        contradiction_cases: `${fixtures.length}/${fixtures.length}`,
      },
      duration_ms: Date.now() - startedAt,
      cases,
    };
  },
} satisfies EvalMetricModule;

export default contradictionDetectionMetric;
