import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { z } from "zod";

import { FakeLLMClient, ManualClock } from "../../src/index.js";
import type { LLMCompleteResult, StreamEntry } from "../../src/index.js";

import { createEvalBorg } from "../support/create-eval-borg.js";
import { loadMetricFixtures } from "../support/fixtures.js";
import type { EvalCaseResult, EvalMetricModule, EvalMetricResult } from "../support/scorecard.js";

const METRIC_NAME = "dedup_correctness";
const METRIC_DESCRIPTION =
  "Checks that the consolidator does not falsely merge repeated but distinct events on different days.";
const EPISODE_TOOL_NAME = "EmitEpisodeCandidates";
const CONSOLIDATION_TOOL_NAME = "EmitConsolidation";
const DAY_MS = 24 * 60 * 60 * 1_000;

const dedupFixtureSchema = z.object({
  name: z.string().min(1),
  events: z.array(
    z.object({
      entry: z.object({
        kind: z.enum(["user_msg", "agent_msg"]),
        content: z.string().min(1),
      }),
      scripted_episode: z.object({
        title: z.string().min(1),
        narrative: z.string().min(1),
        participants: z.array(z.string().min(1)),
        tags: z.array(z.string().min(1)),
        confidence: z.number().min(0).max(1),
        significance: z.number().min(0).max(1),
      }),
      advance_days_after: z.number().int().nonnegative(),
    }),
  ),
});

type DedupFixture = z.infer<typeof dedupFixtureSchema>;

function createEpisodeExtractionResponse(entry: StreamEntry, event: DedupFixture["events"][number]): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 18,
    output_tokens: 9,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_eval_dedup_episode",
        name: EPISODE_TOOL_NAME,
        input: {
          episodes: [
            {
              title: event.scripted_episode.title,
              narrative: event.scripted_episode.narrative,
              source_stream_ids: [entry.id],
              participants: event.scripted_episode.participants,
              location: null,
              tags: event.scripted_episode.tags,
              confidence: event.scripted_episode.confidence,
              significance: event.scripted_episode.significance,
            },
          ],
        },
      },
    ],
  };
}

function createConsolidationResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 10,
    output_tokens: 6,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_eval_consolidate",
        name: CONSOLIDATION_TOOL_NAME,
        input: {
          title: "Merged architecture review",
          narrative: "Two similar architecture review notes were merged into one grounded episode.",
        },
      },
    ],
  };
}

export const dedupCorrectnessMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const fixtures = loadMetricFixtures(METRIC_NAME, dedupFixtureSchema);
    const cases: EvalCaseResult[] = [];
    let passed = true;
    let totalPlannedMerges = 0;

    for (const fixture of fixtures) {
      const tempDir = mkdtempSync(join(tmpdir(), "borg-eval-"));
      const clock = new ManualClock(1_000_000);
      const llm = new FakeLLMClient();
      const borg = await createEvalBorg({
        tempDir,
        llm,
        clock,
      });

      try {
        for (const event of fixture.data.events) {
          const entry = await borg.stream.append(event.entry);
          llm.pushResponse(createEpisodeExtractionResponse(entry, event));
          await borg.episodic.extract({
            sinceTs: entry.timestamp,
          });
          clock.advance(event.advance_days_after * DAY_MS);
        }

        llm.pushResponse(createConsolidationResponse());
        llm.pushResponse(createConsolidationResponse());
        const plan = await borg.dream.plan({
          processes: ["consolidator"],
        });
        const consolidatorPlan = plan.processes.find(
          (process) => process.process === "consolidator",
        );
        const mergedItems =
          consolidatorPlan !== undefined && consolidatorPlan.process === "consolidator"
            ? consolidatorPlan.items.length
            : 0;
        const episodes = (await borg.episodic.list({ limit: 10 })).items;
        const distinctStartTimes = new Set(episodes.map((episode) => episode.start_time)).size;
        const casePassed = mergedItems === 0 && episodes.length === 2 && distinctStartTimes === 2;

        totalPlannedMerges += mergedItems;
        passed &&= casePassed;
        cases.push({
          name: fixture.name,
          passed: casePassed,
          actual: {
            planned_merges: mergedItems,
            episode_count: episodes.length,
            distinct_start_times: distinctStartTimes,
          },
          expected: {
            planned_merges: 0,
            episode_count: 2,
            distinct_start_times: 2,
          },
          note: casePassed
            ? undefined
            : "False-merge bug exposed: consolidator clustering still lacks temporal distance awareness.",
        });
      } finally {
        await borg.close();
        rmSync(tempDir, { recursive: true, force: true });
      }
    }

    return {
      name: METRIC_NAME,
      description: METRIC_DESCRIPTION,
      passed,
      actual: {
        planned_merges: totalPlannedMerges,
      },
      expected: {
        planned_merges: 0,
      },
      duration_ms: Date.now() - startedAt,
      cases,
    };
  },
} satisfies EvalMetricModule;

export default dedupCorrectnessMetric;
