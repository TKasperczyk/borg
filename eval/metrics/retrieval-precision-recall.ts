import { z } from "zod";

import { FixedClock } from "../../src/index.js";
import { createEpisodeFixture, createOfflineTestHarness } from "../../src/offline/test-support.js";

import { DeterministicEmbeddingClient } from "../support/embedding.js";
import { loadMetricFixtures } from "../support/fixtures.js";
import type { EvalCaseResult, EvalMetricModule, EvalMetricResult } from "../support/scorecard.js";

const METRIC_NAME = "retrieval_precision_recall";
const METRIC_DESCRIPTION =
  "Measures precision@5 and recall@5 on a deterministic seeded episodic retrieval corpus.";
const EMBEDDING_DIMS = 64;
const RETRIEVAL_WEIGHTS = {
  semantic: 1,
  goal_relevance: 0,
  mood: 0,
  time: 0,
  social: 0,
  entity: 0,
  heat: 0,
  suppression_penalty: 0.5,
} as const;

const retrievalFixtureSchema = z.object({
  name: z.string().min(1),
  k: z.number().int().positive(),
  episodes: z.array(
    z.object({
      id: z.string().min(1),
      title: z.string().min(1),
      narrative: z.string().min(1),
      participants: z.array(z.string().min(1)).default([]),
      tags: z.array(z.string().min(1)).default([]),
      start_time: z.number().finite(),
      end_time: z.number().finite(),
    }),
  ),
  queries: z.array(
    z.object({
      name: z.string().min(1),
      query: z.string().min(1),
      relevant_episode_ids: z.array(z.string().min(1)).min(1),
      precision_at_5_min: z.number().min(0).max(1),
      recall_at_5_min: z.number().min(0).max(1),
    }),
  ),
});

type RetrievalFixture = z.infer<typeof retrievalFixtureSchema>;

function buildEpisodeEmbeddingText(episode: RetrievalFixture["episodes"][number]): string {
  return [
    episode.title,
    episode.narrative,
    episode.tags.join(" "),
    episode.participants.join(" "),
  ].join("\n");
}

export const retrievalPrecisionRecallMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const fixtures = loadMetricFixtures(METRIC_NAME, retrievalFixtureSchema);
    const cases: EvalCaseResult[] = [];
    let totalRelevantHits = 0;
    let totalTopKSlots = 0;
    let totalRelevantExpected = 0;
    let passed = true;

    for (const fixture of fixtures) {
      const embeddingClient = new DeterministicEmbeddingClient(EMBEDDING_DIMS);
      const harness = await createOfflineTestHarness({
        clock: new FixedClock(10_000),
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
              participants: episode.participants,
              tags: episode.tags,
              start_time: episode.start_time,
              end_time: episode.end_time,
              created_at: episode.start_time,
              updated_at: episode.end_time,
              embedding: await embeddingClient.embed(buildEpisodeEmbeddingText(episode)),
            }),
          );
        }

        for (const query of fixture.data.queries) {
          const results = await harness.retrievalPipeline.search(query.query, {
            limit: fixture.data.k,
            attentionWeights: RETRIEVAL_WEIGHTS,
          });
          const retrievedEpisodeIds = results.map((result) => result.episode.id);
          const relevantIds = new Set(query.relevant_episode_ids);
          const relevantHits = retrievedEpisodeIds.filter((id) => relevantIds.has(id)).length;
          const precision = relevantHits / fixture.data.k;
          const recall = relevantHits / query.relevant_episode_ids.length;
          const casePassed =
            precision >= query.precision_at_5_min && recall >= query.recall_at_5_min;

          totalRelevantHits += relevantHits;
          totalTopKSlots += fixture.data.k;
          totalRelevantExpected += query.relevant_episode_ids.length;
          passed &&= casePassed;

          cases.push({
            name: `${fixture.name}:${query.name}`,
            passed: casePassed,
            actual: {
              retrieved_episode_ids: retrievedEpisodeIds,
              precision_at_5: `${relevantHits}/${fixture.data.k}`,
              recall_at_5: `${relevantHits}/${query.relevant_episode_ids.length}`,
            },
            expected: {
              relevant_episode_ids: query.relevant_episode_ids,
              precision_at_5_min: query.precision_at_5_min,
              recall_at_5_min: query.recall_at_5_min,
            },
          });
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
        precision_at_5: `${totalRelevantHits}/${totalTopKSlots}`,
        recall_at_5: `${totalRelevantHits}/${totalRelevantExpected}`,
      },
      expected: {
        precision_at_5_min: 0.6,
        recall_at_5_min: 1,
      },
      duration_ms: Date.now() - startedAt,
      cases,
    };
  },
} satisfies EvalMetricModule;

export default retrievalPrecisionRecallMetric;
