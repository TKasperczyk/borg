import { z } from "zod";

import { FixedClock } from "../../src/index.js";
import { createEpisodeFixture, createOfflineTestHarness } from "../../src/offline/test-support.js";

import { DeterministicEmbeddingClient } from "../support/embedding.js";
import { loadMetricFixtures } from "../support/fixtures.js";
import type { EvalCaseResult, EvalMetricModule, EvalMetricResult } from "../support/scorecard.js";

const METRIC_NAME = "cross_audience_leakage";
const METRIC_DESCRIPTION =
  "Ensures audience-scoped episodes and mixed-visibility semantic nodes never leak across audiences.";
const EMBEDDING_DIMS = 64;

const leakageFixtureSchema = z.object({
  name: z.string().min(1),
  query: z.string().min(1),
  episodes: z.array(
    z.object({
      id: z.string().min(1),
      title: z.string().min(1),
      narrative: z.string().min(1),
      participants: z.array(z.string().min(1)),
      tags: z.array(z.string().min(1)),
      visibility: z.enum(["alice_private", "bob_private", "public"]),
      start_time: z.number().finite(),
      end_time: z.number().finite(),
    }),
  ),
  semantic_nodes: z.array(
    z.object({
      id: z.string().min(1),
      label: z.string().min(1),
      description: z.string().min(1),
      source_episode_ids: z.array(z.string().min(1)).min(1),
    }),
  ),
  scenarios: z.array(
    z.object({
      name: z.string().min(1),
      audience: z.string().min(1).nullable(),
      expected_episode_ids: z.array(z.string().min(1)),
      expected_semantic_node_ids: z.array(z.string().min(1)),
    }),
  ),
});

type LeakageFixture = z.infer<typeof leakageFixtureSchema>;

function buildEpisodeEmbeddingText(episode: LeakageFixture["episodes"][number]): string {
  return [
    episode.title,
    episode.narrative,
    episode.tags.join(" "),
    episode.participants.join(" "),
  ].join("\n");
}

export const crossAudienceLeakageMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const fixtures = loadMetricFixtures(METRIC_NAME, leakageFixtureSchema);
    const cases: EvalCaseResult[] = [];
    let passed = true;
    let scenariosPassed = 0;
    let totalScenarios = 0;

    for (const fixture of fixtures) {
      const embeddingClient = new DeterministicEmbeddingClient(EMBEDDING_DIMS);
      const harness = await createOfflineTestHarness({
        clock: new FixedClock(20_000),
        embeddingClient,
        embeddingDimensions: EMBEDDING_DIMS,
      });

      try {
        const alice = harness.entityRepository.resolve("Alice");
        const bob = harness.entityRepository.resolve("Bob");

        for (const episode of fixture.data.episodes) {
          const visibility =
            episode.visibility === "alice_private"
              ? { audience_entity_id: alice, shared: false }
              : episode.visibility === "bob_private"
                ? { audience_entity_id: bob, shared: false }
                : { audience_entity_id: null, shared: true };
          await harness.episodicRepository.insert(
            createEpisodeFixture({
              id: episode.id as never,
              title: episode.title,
              narrative: episode.narrative,
              participants: episode.participants,
              tags: episode.tags,
              start_time: episode.start_time,
              end_time: episode.end_time,
              created_at: episode.start_time,
              updated_at: episode.end_time,
              embedding: await embeddingClient.embed(buildEpisodeEmbeddingText(episode)),
              ...visibility,
            }),
          );
        }

        for (const node of fixture.data.semantic_nodes) {
          await harness.semanticNodeRepository.insert({
            id: node.id as never,
            kind: "proposition",
            label: node.label,
            description: node.description,
            aliases: [],
            confidence: 0.8,
            source_episode_ids: node.source_episode_ids as never,
            created_at: 20_000,
            updated_at: 20_000,
            last_verified_at: 20_000,
            embedding: await embeddingClient.embed(`${node.label}\n${node.description}`),
            archived: false,
            superseded_by: null,
          });
        }

        const publicNode = fixture.data.semantic_nodes.find((node) =>
          node.source_episode_ids.every((episodeId) =>
            fixture.data.episodes.some(
              (episode) => episode.id === episodeId && episode.visibility === "public",
            ),
          ),
        );
        const privateNode = fixture.data.semantic_nodes.find(
          (node) => publicNode !== undefined && node.id !== publicNode.id,
        );
        const alicePrivateEpisode = fixture.data.episodes.find(
          (episode) => episode.visibility === "alice_private",
        );

        if (
          publicNode !== undefined &&
          privateNode !== undefined &&
          alicePrivateEpisode !== undefined
        ) {
          const closedEdge = harness.semanticEdgeRepository.addEdge({
            from_node_id: publicNode.id as never,
            to_node_id: privateNode.id as never,
            relation: "supports",
            confidence: 0.8,
            evidence_episode_ids: [alicePrivateEpisode.id] as never,
            created_at: 10_000,
            last_verified_at: 10_000,
            valid_from: 10_000,
          });
          harness.semanticEdgeRepository.invalidateEdge(closedEdge.id, {
            at: 15_000,
            by_process: "maintenance",
            reason: "eval_closed_edge_audience_scope",
          });
        }

        for (const scenario of fixture.data.scenarios) {
          let audienceEntityId: ReturnType<typeof harness.entityRepository.resolve> | null = null;
          let audienceProfile:
            | ReturnType<typeof harness.socialRepository.upsertProfile>
            | undefined;
          if (scenario.audience !== null) {
            audienceEntityId = harness.entityRepository.resolve(scenario.audience);
            audienceProfile = harness.socialRepository.upsertProfile(audienceEntityId);
          }
          const result = await harness.retrievalPipeline.searchWithContext(fixture.data.query, {
            limit: 5,
            audienceEntityId,
            audienceTerms: scenario.audience === null ? undefined : [scenario.audience],
            audienceProfile,
            graphWalkDepth: 1,
            maxGraphNodes: 8,
          });
          const actualEpisodeIds = result.episodes.map((item) => item.episode.id).sort();
          const actualSemanticNodeIds = [...result.semantic.matched_node_ids].sort();
          const expectedEpisodeIds = [...scenario.expected_episode_ids].sort();
          const expectedSemanticNodeIds = [...scenario.expected_semantic_node_ids].sort();
          const casePassed =
            JSON.stringify(actualEpisodeIds) === JSON.stringify(expectedEpisodeIds) &&
            JSON.stringify(actualSemanticNodeIds) === JSON.stringify(expectedSemanticNodeIds);

          totalScenarios += 1;
          scenariosPassed += casePassed ? 1 : 0;
          passed &&= casePassed;
          cases.push({
            name: `${fixture.name}:${scenario.name}`,
            passed: casePassed,
            actual: {
              episode_ids: actualEpisodeIds,
              semantic_node_ids: actualSemanticNodeIds,
            },
            expected: {
              episode_ids: expectedEpisodeIds,
              semantic_node_ids: expectedSemanticNodeIds,
            },
          });
        }

        if (
          publicNode !== undefined &&
          privateNode !== undefined &&
          alicePrivateEpisode !== undefined
        ) {
          for (const scenario of [
            {
              name: "closed_edge_history_alice",
              audience: "Alice",
              expected_support_hit_ids: [privateNode.id],
            },
            {
              name: "closed_edge_history_bob",
              audience: "Bob",
              expected_support_hit_ids: [],
            },
            {
              name: "closed_edge_history_no_audience",
              audience: null,
              expected_support_hit_ids: [],
            },
          ] as const) {
            let audienceEntityId: ReturnType<typeof harness.entityRepository.resolve> | null = null;
            let audienceProfile:
              | ReturnType<typeof harness.socialRepository.upsertProfile>
              | undefined;
            if (scenario.audience !== null) {
              audienceEntityId = harness.entityRepository.resolve(scenario.audience);
              audienceProfile = harness.socialRepository.upsertProfile(audienceEntityId);
            }

            const result = await harness.retrievalPipeline.searchWithContext(publicNode.label, {
              limit: 5,
              audienceEntityId,
              audienceTerms: scenario.audience === null ? undefined : [scenario.audience],
              audienceProfile,
              graphWalkDepth: 1,
              maxGraphNodes: 8,
              asOf: 12_000,
            });
            const actualSupportHitIds = result.semantic.support_hits
              .map((hit) => hit.node.id)
              .sort();
            const expectedSupportHitIds = [...scenario.expected_support_hit_ids].sort();
            const casePassed =
              JSON.stringify(actualSupportHitIds) === JSON.stringify(expectedSupportHitIds);

            totalScenarios += 1;
            scenariosPassed += casePassed ? 1 : 0;
            passed &&= casePassed;
            cases.push({
              name: `${fixture.name}:${scenario.name}`,
              passed: casePassed,
              actual: {
                support_hit_ids: actualSupportHitIds,
                as_of: 12_000,
              },
              expected: {
                support_hit_ids: expectedSupportHitIds,
                as_of: 12_000,
              },
              note: "Historical asOf graph walks must still enforce audience visibility on closed edge evidence.",
            });
          }
        }
      } finally {
        await harness.cleanup();
      }
    }

    return {
      name: METRIC_NAME,
      description: METRIC_DESCRIPTION,
      passed,
      actual: {
        scenarios: `${scenariosPassed}/${totalScenarios}`,
      },
      expected: {
        scenarios: `${totalScenarios}/${totalScenarios}`,
      },
      duration_ms: Date.now() - startedAt,
      cases,
    };
  },
} satisfies EvalMetricModule;

export default crossAudienceLeakageMetric;
