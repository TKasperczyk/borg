import { FixedClock } from "../../src/index.js";
import { createEpisodeFixture, createOfflineTestHarness } from "../../src/offline/test-support.js";

import { DeterministicEmbeddingClient } from "../support/embedding.js";
import type { EvalCaseResult, EvalMetricModule, EvalMetricResult } from "../support/scorecard.js";

const METRIC_NAME = "semantic_temporal_recall";
const METRIC_DESCRIPTION =
  "Measures semantic graph precision/recall for open versus closed edges under asOf queries.";
const EMBEDDING_DIMS = 64;
const NOW_MS = 40_000;

const ROOT_NODE_ID = "semn_0000000000001010";
const HISTORICAL_NODE_ID = "semn_0000000000001011";
const CURRENT_NODE_ID = "semn_0000000000001012";
const QUERY = "Atlas deployment status";

type SemanticTemporalCase = {
  name: string;
  asOf: number;
  expected_support_node_ids: string[];
};

const CASES: readonly SemanticTemporalCase[] = [
  {
    name: "current_open_edge",
    asOf: NOW_MS,
    expected_support_node_ids: [CURRENT_NODE_ID],
  },
  {
    name: "historical_closed_edge",
    asOf: 15_000,
    expected_support_node_ids: [HISTORICAL_NODE_ID],
  },
];

export const semanticTemporalRecallMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const embeddingClient = new DeterministicEmbeddingClient(EMBEDDING_DIMS);
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient,
      embeddingDimensions: EMBEDDING_DIMS,
    });
    const cases: EvalCaseResult[] = [];
    let passed = true;
    let totalRelevantHits = 0;
    let totalRetrieved = 0;
    let totalRelevantExpected = 0;

    try {
      const historicalEpisode = createEpisodeFixture({
        id: "ep_0000000000001011" as never,
        title: "Atlas deployment auth issue",
        narrative: "Atlas deployment status was blocked by an auth issue.",
        tags: ["atlas", "deployment", "status"],
        participants: ["team"],
        start_time: 10_000,
        end_time: 11_000,
        created_at: 10_000,
        updated_at: 11_000,
        embedding: await embeddingClient.embed("Atlas deployment status auth issue"),
      });
      const currentEpisode = createEpisodeFixture({
        id: "ep_0000000000001012" as never,
        title: "Atlas deployment stable",
        narrative: "Atlas deployment status is stable after the rollback.",
        tags: ["atlas", "deployment", "status"],
        participants: ["team"],
        start_time: 30_000,
        end_time: 31_000,
        created_at: 30_000,
        updated_at: 31_000,
        embedding: await embeddingClient.embed("Atlas deployment status stable"),
      });

      await harness.episodicRepository.insert(historicalEpisode);
      await harness.episodicRepository.insert(currentEpisode);

      await harness.semanticNodeRepository.insert({
        id: ROOT_NODE_ID as never,
        kind: "proposition",
        label: QUERY,
        description: "Root node for the Atlas deployment status query.",
        aliases: [],
        confidence: 0.8,
        source_episode_ids: [currentEpisode.id],
        created_at: NOW_MS,
        updated_at: NOW_MS,
        last_verified_at: NOW_MS,
        embedding: await embeddingClient.embed(`${QUERY}\nAtlas deployment status query root`),
        archived: false,
        superseded_by: null,
      });
      await harness.semanticNodeRepository.insert({
        id: HISTORICAL_NODE_ID as never,
        kind: "proposition",
        label: "Atlas deployment status was blocked",
        description: "Atlas deployment was historically blocked by an auth issue.",
        aliases: [],
        confidence: 0.8,
        source_episode_ids: [historicalEpisode.id],
        created_at: 10_000,
        updated_at: 10_000,
        last_verified_at: 10_000,
        embedding: await embeddingClient.embed("Atlas deployment status blocked"),
        archived: false,
        superseded_by: null,
      });
      await harness.semanticNodeRepository.insert({
        id: CURRENT_NODE_ID as never,
        kind: "proposition",
        label: "Atlas deployment status is stable",
        description: "Atlas deployment is currently stable.",
        aliases: [],
        confidence: 0.8,
        source_episode_ids: [currentEpisode.id],
        created_at: 30_000,
        updated_at: 30_000,
        last_verified_at: 30_000,
        embedding: await embeddingClient.embed("Atlas deployment status stable"),
        archived: false,
        superseded_by: null,
      });

      const historicalEdge = harness.semanticEdgeRepository.addEdge({
        from_node_id: ROOT_NODE_ID as never,
        to_node_id: HISTORICAL_NODE_ID as never,
        relation: "supports",
        confidence: 0.8,
        evidence_episode_ids: [historicalEpisode.id],
        created_at: 10_000,
        last_verified_at: 10_000,
        valid_from: 10_000,
      });
      harness.semanticEdgeRepository.invalidateEdge(historicalEdge.id, {
        at: 20_000,
        by_process: "maintenance",
        reason: "eval_semantic_temporal_recall",
      });
      harness.semanticEdgeRepository.addEdge({
        from_node_id: ROOT_NODE_ID as never,
        to_node_id: CURRENT_NODE_ID as never,
        relation: "supports",
        confidence: 0.8,
        evidence_episode_ids: [currentEpisode.id],
        created_at: 30_000,
        last_verified_at: 30_000,
        valid_from: 30_000,
      });

      for (const testCase of CASES) {
        const result = await harness.retrievalPipeline.searchWithContext(QUERY, {
          limit: 3,
          graphWalkDepth: 1,
          maxGraphNodes: 8,
          asOf: testCase.asOf,
        });
        const actualSupportNodeIds = result.semantic.support_hits
          .filter((hit) => hit.root_node_id === ROOT_NODE_ID)
          .map((hit) => hit.node.id)
          .sort();
        const expectedSupportNodeIds = [...testCase.expected_support_node_ids].sort();
        const expectedSet = new Set(expectedSupportNodeIds);
        const relevantHits = actualSupportNodeIds.filter((id) => expectedSet.has(id)).length;
        const precision =
          actualSupportNodeIds.length === 0 ? 0 : relevantHits / actualSupportNodeIds.length;
        const recall =
          expectedSupportNodeIds.length === 0 ? 1 : relevantHits / expectedSupportNodeIds.length;
        const casePassed =
          precision === 1 &&
          recall === 1 &&
          JSON.stringify(actualSupportNodeIds) === JSON.stringify(expectedSupportNodeIds);

        totalRelevantHits += relevantHits;
        totalRetrieved += actualSupportNodeIds.length;
        totalRelevantExpected += expectedSupportNodeIds.length;
        passed &&= casePassed;
        cases.push({
          name: testCase.name,
          passed: casePassed,
          actual: {
            support_node_ids: actualSupportNodeIds,
            precision: `${relevantHits}/${actualSupportNodeIds.length}`,
            recall: `${relevantHits}/${expectedSupportNodeIds.length}`,
            as_of: testCase.asOf,
          },
          expected: {
            support_node_ids: expectedSupportNodeIds,
            precision_min: 1,
            recall_min: 1,
            as_of: testCase.asOf,
          },
        });
      }
    } finally {
      await harness.cleanup();
    }

    return {
      name: METRIC_NAME,
      description: METRIC_DESCRIPTION,
      passed,
      actual: {
        precision: `${totalRelevantHits}/${totalRetrieved}`,
        recall: `${totalRelevantHits}/${totalRelevantExpected}`,
      },
      expected: {
        precision_min: 1,
        recall_min: 1,
      },
      duration_ms: Date.now() - startedAt,
      cases,
    };
  },
} satisfies EvalMetricModule;

export default semanticTemporalRecallMetric;
