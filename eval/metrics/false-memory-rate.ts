import { z } from "zod";

import { FixedClock } from "../../src/index.js";
import { createEpisodeFixture, createOfflineTestHarness } from "../../src/offline/test-support.js";

import { DeterministicEmbeddingClient } from "../support/embedding.js";
import { loadMetricFixtures } from "../support/fixtures.js";
import type { EvalCaseResult, EvalMetricModule, EvalMetricResult } from "../support/scorecard.js";

const METRIC_NAME = "false_memory_rate";
const METRIC_DESCRIPTION =
  "Measures how often absent-topic queries still produce medium-confidence retrieval results.";
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

const falseMemoryFixtureSchema = z.object({
  name: z.string().min(1),
  max_results: z.number().int().nonnegative(),
  max_score: z.number().min(0).max(1),
  episodes: z.array(
    z.object({
      id: z.string().min(1),
      title: z.string().min(1),
      narrative: z.string().min(1),
      participants: z.array(z.string().min(1)),
      tags: z.array(z.string().min(1)),
      start_time: z.number().finite(),
      end_time: z.number().finite(),
    }),
  ),
  queries: z.array(
    z.object({
      name: z.string().min(1),
      query: z.string().min(1),
    }),
  ),
});

type FalseMemoryFixture = z.infer<typeof falseMemoryFixtureSchema>;

function buildEpisodeEmbeddingText(episode: FalseMemoryFixture["episodes"][number]): string {
  return [
    episode.title,
    episode.narrative,
    episode.tags.join(" "),
    episode.participants.join(" "),
  ].join("\n");
}

export const falseMemoryRateMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const fixtures = loadMetricFixtures(METRIC_NAME, falseMemoryFixtureSchema);
    const cases: EvalCaseResult[] = [];
    let passed = true;
    let safeQueries = 0;
    let totalQueries = 0;

    for (const fixture of fixtures) {
      const embeddingClient = new DeterministicEmbeddingClient(EMBEDDING_DIMS);
      const harness = await createOfflineTestHarness({
        clock: new FixedClock(50_000),
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
            limit: fixture.data.max_results,
            attentionWeights: RETRIEVAL_WEIGHTS,
          });
          const maxScore = results.reduce((best, result) => Math.max(best, result.score), 0);
          const casePassed =
            results.length <= fixture.data.max_results && maxScore < fixture.data.max_score;

          totalQueries += 1;
          safeQueries += casePassed ? 1 : 0;
          passed &&= casePassed;
          cases.push({
            name: `${fixture.name}:${query.name}`,
            passed: casePassed,
            actual: {
              result_count: results.length,
              max_score: Number(maxScore.toFixed(3)),
            },
            expected: {
              result_count_lte: fixture.data.max_results,
              max_score_lt: fixture.data.max_score,
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
        safe_queries: `${safeQueries}/${totalQueries}`,
      },
      expected: {
        safe_queries: `${totalQueries}/${totalQueries}`,
      },
      duration_ms: Date.now() - startedAt,
      cases,
    };
  },
} satisfies EvalMetricModule;

export default falseMemoryRateMetric;
