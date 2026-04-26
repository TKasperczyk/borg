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
const NOW_MS = 40_000;
const HISTORICAL_VALID_FROM_MS = 20_000;
const HISTORICAL_AS_OF_MS = 25_000;
const HISTORICAL_CLOSED_AT_MS = 30_000;

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

function isEdgeValidAt(
  edge: { valid_from: number; valid_to: number | null },
  asOf: number,
): boolean {
  return edge.valid_from <= asOf && (edge.valid_to === null || edge.valid_to > asOf);
}

function validContradictionNodeIdsAt(
  result: {
    semantic: {
      contradiction_hits: Array<{
        node: { id: string };
        edgePath: Array<{ valid_from: number; valid_to: number | null }>;
      }>;
    };
  },
  asOf: number,
): string[] {
  return result.semantic.contradiction_hits
    .filter((hit) => hit.edgePath.some((edge) => isEdgeValidAt(edge, asOf)))
    .map((hit) => hit.node.id);
}

async function seedFixture(
  fixture: ContradictionFixture,
  embeddingClient: DeterministicEmbeddingClient,
  harness: Awaited<ReturnType<typeof createOfflineTestHarness>>,
): Promise<void> {
  for (const episode of fixture.episodes) {
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

  for (const node of fixture.nodes) {
    await harness.semanticNodeRepository.insert({
      id: node.id as never,
      kind: "proposition",
      label: node.label,
      description: node.description,
      aliases: [],
      confidence: 0.8,
      source_episode_ids: node.source_episode_ids as never,
      created_at: NOW_MS,
      updated_at: NOW_MS,
      last_verified_at: NOW_MS,
      embedding: await embeddingClient.embed(`${node.label}\n${node.description}`),
      archived: false,
      superseded_by: null,
    });
  }
}

function addContradictionEdge(
  fixture: ContradictionFixture,
  harness: Awaited<ReturnType<typeof createOfflineTestHarness>>,
  validity: { valid_from?: number; close_at?: number } = {},
): void {
  const edge = harness.semanticEdgeRepository.addEdge({
    from_node_id: fixture.contradiction_edge.from_node_id as never,
    to_node_id: fixture.contradiction_edge.to_node_id as never,
    relation: "contradicts",
    confidence: fixture.contradiction_edge.confidence,
    evidence_episode_ids: fixture.nodes[0]?.source_episode_ids as never,
    created_at: validity.valid_from ?? NOW_MS,
    last_verified_at: validity.valid_from ?? NOW_MS,
    ...(validity.valid_from === undefined ? {} : { valid_from: validity.valid_from }),
  });

  if (validity.close_at !== undefined) {
    harness.semanticEdgeRepository.invalidateEdge(edge.id, {
      at: validity.close_at,
      by_process: "maintenance",
      reason: "eval_historical_contradiction",
    });
  }
}

export const contradictionDetectionMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const fixtures = loadMetricFixtures(METRIC_NAME, contradictionFixtureSchema);
    const cases: EvalCaseResult[] = [];
    let passed = true;
    let passedCases = 0;
    let totalCases = 0;

    for (const fixture of fixtures) {
      {
        const embeddingClient = new DeterministicEmbeddingClient(EMBEDDING_DIMS);
        const harness = await createOfflineTestHarness({
          clock: new FixedClock(NOW_MS),
          embeddingClient,
          embeddingDimensions: EMBEDDING_DIMS,
        });

        try {
          await seedFixture(fixture.data, embeddingClient, harness);
          addContradictionEdge(fixture.data, harness);

          const result = await harness.retrievalPipeline.searchWithContext(fixture.data.query, {
            limit: 5,
            graphWalkDepth: 1,
            maxGraphNodes: 8,
          });
          const actualContradictionNodeIds = validContradictionNodeIdsAt(result, NOW_MS);
          const actualContradictionHitCount = actualContradictionNodeIds.length;
          const casePassed =
            result.contradiction_present &&
            actualContradictionHitCount >= 1 &&
            actualContradictionNodeIds.includes(
              fixture.data.expected_contradiction_node_id as never,
            );

          totalCases += 1;
          passedCases += casePassed ? 1 : 0;
          passed &&= casePassed;
          cases.push({
            name: `${fixture.name}:current`,
            passed: casePassed,
            actual: {
              contradiction_present: result.contradiction_present,
              current_valid_contradiction_hit_count: actualContradictionHitCount,
              current_valid_contradiction_hit_ids: actualContradictionNodeIds,
            },
            expected: {
              contradiction_present: true,
              minimum_current_valid_contradiction_hit_count: 1,
              expected_hit_id: fixture.data.expected_contradiction_node_id,
              extra_hits_allowed: true,
            },
            note: "Extra contradiction hits are allowed because retrieval may surface both sides of an explicit contradicts edge.",
          });
        } finally {
          await harness.cleanup();
        }
      }

      {
        const embeddingClient = new DeterministicEmbeddingClient(EMBEDDING_DIMS);
        const harness = await createOfflineTestHarness({
          clock: new FixedClock(NOW_MS),
          embeddingClient,
          embeddingDimensions: EMBEDDING_DIMS,
        });

        try {
          await seedFixture(fixture.data, embeddingClient, harness);
          addContradictionEdge(fixture.data, harness, {
            valid_from: HISTORICAL_VALID_FROM_MS,
            close_at: HISTORICAL_CLOSED_AT_MS,
          });

          const currentResult = await harness.retrievalPipeline.searchWithContext(
            fixture.data.query,
            {
              limit: 5,
              graphWalkDepth: 1,
              maxGraphNodes: 8,
            },
          );
          const historicalResult = await harness.retrievalPipeline.searchWithContext(
            fixture.data.query,
            {
              limit: 5,
              graphWalkDepth: 1,
              maxGraphNodes: 8,
              asOf: HISTORICAL_AS_OF_MS,
            },
          );
          const currentIds = validContradictionNodeIdsAt(currentResult, NOW_MS);
          const historicalIds = validContradictionNodeIdsAt(historicalResult, HISTORICAL_AS_OF_MS);
          const currentCasePassed = !currentResult.contradiction_present && currentIds.length === 0;
          const historicalCasePassed =
            historicalResult.contradiction_present &&
            historicalIds.includes(fixture.data.expected_contradiction_node_id as never);

          totalCases += 2;
          passedCases += currentCasePassed ? 1 : 0;
          passedCases += historicalCasePassed ? 1 : 0;
          passed &&= currentCasePassed && historicalCasePassed;
          cases.push({
            name: `${fixture.name}:closed_contradiction_current`,
            passed: currentCasePassed,
            actual: {
              contradiction_present: currentResult.contradiction_present,
              current_valid_contradiction_hit_ids: currentIds,
            },
            expected: {
              contradiction_present: false,
              current_valid_contradiction_hit_ids: [],
            },
            note: "A contradiction edge closed before now must not be flagged in current-mode retrieval.",
          });
          cases.push({
            name: `${fixture.name}:closed_contradiction_historical`,
            passed: historicalCasePassed,
            actual: {
              contradiction_present: historicalResult.contradiction_present,
              historical_valid_contradiction_hit_ids: historicalIds,
            },
            expected: {
              contradiction_present: true,
              expected_hit_id: fixture.data.expected_contradiction_node_id,
              as_of: HISTORICAL_AS_OF_MS,
            },
          });
        } finally {
          await harness.cleanup();
        }
      }
    }

    return {
      name: METRIC_NAME,
      description: METRIC_DESCRIPTION,
      passed,
      actual: {
        contradiction_cases: `${passedCases}/${totalCases}`,
      },
      expected: {
        contradiction_cases: `${totalCases}/${totalCases}`,
      },
      duration_ms: Date.now() - startedAt,
      cases,
    };
  },
} satisfies EvalMetricModule;

export default contradictionDetectionMetric;
