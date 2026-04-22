import { FixedClock } from "../../src/index.js";
import { createEpisodeFixture, createOfflineTestHarness } from "../../src/offline/test-support.js";

import type { EvalCaseResult, EvalMetricModule, EvalMetricResult } from "../support/scorecard.js";

const METRIC_NAME = "preference_formation";
const METRIC_DESCRIPTION =
  "Checks that an established held value can rescue a value-aligned episode from a vector-heavy decoy pool.";
const NOW_MS = 90_000;
const RETRIEVAL_WEIGHTS = {
  semantic: 0.2,
  goal_relevance: 0,
  value_alignment: 0.35,
  mood: 0,
  time: 0,
  social: 0,
  entity: 0,
  heat: 0.1,
  suppression_penalty: 0.5,
} as const;

export const preferenceFormationMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
    });

    try {
      const supportEpisodes = [
        createEpisodeFixture(
          {
            id: "ep_aaaaaaaaaaaaaaaa" as never,
            title: "Clarity-first review",
            narrative: "Explicit state and direct handoffs kept the release calm.",
            tags: ["clarity", "handoff"],
            created_at: NOW_MS - 30,
            updated_at: NOW_MS - 30,
          },
          [0, 0, 0, 1],
        ),
        createEpisodeFixture(
          {
            id: "ep_bbbbbbbbbbbbbbbb" as never,
            title: "Precise status note",
            narrative: "A precise status note removed ambiguity for the team.",
            tags: ["clarity", "status"],
            created_at: NOW_MS - 20,
            updated_at: NOW_MS - 20,
          },
          [0, 0, 0, 1],
        ),
        createEpisodeFixture(
          {
            id: "ep_cccccccccccccccc" as never,
            title: "Explicit handoff debrief",
            narrative: "Explicit state and careful handoffs prevented a messy rollback.",
            tags: ["clarity", "handoff"],
            created_at: NOW_MS - 10,
            updated_at: NOW_MS - 10,
          },
          [0, 0, 0, 1],
        ),
      ];

      for (const episode of supportEpisodes) {
        await harness.episodicRepository.insert(episode);
      }

      for (let index = 0; index < 4; index += 1) {
        await harness.episodicRepository.insert(
          createEpisodeFixture(
            {
              title: `Architecture decoy ${index}`,
              narrative: `A strong architecture retrieval match ${index}.`,
              tags: ["architecture"],
              created_at: NOW_MS - 100 - index,
              updated_at: NOW_MS - 100 - index,
            },
            [1, 0, 0, 0],
          ),
        );
      }

      const seededValue = harness.valuesRepository.add({
        label: "clarity",
        description: "Prefer explicit state and careful handoffs.",
        priority: 10,
        provenance: {
          kind: "episodes",
          episode_ids: [supportEpisodes[0]!.id],
        },
        createdAt: NOW_MS - 30,
      });
      harness.valuesRepository.reinforce(
        seededValue.id,
        {
          kind: "episodes",
          episode_ids: [supportEpisodes[1]!.id],
        },
        NOW_MS - 20,
      );
      const establishedValue = harness.valuesRepository.reinforce(
        seededValue.id,
        {
          kind: "episodes",
          episode_ids: [supportEpisodes[2]!.id],
        },
        NOW_MS - 10,
      );

      const vectorOnly = await harness.retrievalPipeline.search("architecture", {
        limit: 3,
        attentionWeights: RETRIEVAL_WEIGHTS,
      });
      const withPreference = await harness.retrievalPipeline.search("architecture", {
        limit: 3,
        attentionWeights: RETRIEVAL_WEIGHTS,
        activeValues: [establishedValue],
      });

      const rescuedId = supportEpisodes[2]!.id;
      const casePassed =
        vectorOnly[0]?.episode.id !== rescuedId &&
        withPreference[0]?.episode.id === rescuedId &&
        (withPreference[0]?.scoreBreakdown.valueAlignment ?? 0) > 0;
      const cases: EvalCaseResult[] = [
        {
          name: "established_value_rescues_aligned_episode",
          passed: casePassed,
          actual: {
            vector_only_top_id: vectorOnly[0]?.episode.id ?? null,
            value_aligned_top_id: withPreference[0]?.episode.id ?? null,
            value_alignment: Number((withPreference[0]?.scoreBreakdown.valueAlignment ?? 0).toFixed(3)),
          },
          expected: {
            vector_only_top_id_not: rescuedId,
            value_aligned_top_id: rescuedId,
            value_alignment_gt: 0,
          },
        },
      ];

      return {
        name: METRIC_NAME,
        description: METRIC_DESCRIPTION,
        passed: casePassed,
        actual: {
          rescued_cases: `${casePassed ? 1 : 0}/1`,
        },
        expected: {
          rescued_cases: "1/1",
        },
        duration_ms: Date.now() - startedAt,
        cases,
      };
    } finally {
      await harness.cleanup();
    }
  },
} satisfies EvalMetricModule;

export default preferenceFormationMetric;
