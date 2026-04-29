import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { z } from "zod";

import { FakeLLMClient, ManualClock } from "../../src/index.js";
import type { LLMCompleteResult, StreamEntry } from "../../src/index.js";

import { createEvalBorg } from "../support/create-eval-borg.js";
import { loadMetricFixtures } from "../support/fixtures.js";
import type { EvalCaseResult, EvalMetricModule, EvalMetricResult } from "../support/scorecard.js";

const METRIC_NAME = "episodic_extraction_quality";
const METRIC_DESCRIPTION =
  "Validates episodic extraction inserts the expected grounded episodes from a scripted stream chunk.";
const EPISODE_TOOL_NAME = "EmitEpisodeCandidates";

const extractionFixtureSchema = z.object({
  name: z.string().min(1),
  stream_entries: z.array(
    z.object({
      kind: z.enum(["user_msg", "agent_msg"]),
      content: z.string().min(1),
      audience: z.string().min(1).optional(),
    }),
  ),
  scripted_episodes: z.array(
    z.object({
      title: z.string().min(1),
      narrative: z.string().min(1),
      participants: z.array(z.string().min(1)),
      tags: z.array(z.string().min(1)),
      source_entry_indexes: z.array(z.number().int().nonnegative()).min(1),
      confidence: z.number().min(0).max(1),
      significance: z.number().min(0).max(1),
    }),
  ),
  expected: z.array(
    z.object({
      name: z.string().min(1),
      title_keywords: z.array(z.string().min(1)).min(1),
      narrative_keywords: z.array(z.string().min(1)).min(1),
      participants: z.array(z.string().min(1)).min(1),
    }),
  ),
});

type ExtractionFixture = z.infer<typeof extractionFixtureSchema>;

function createEpisodeToolResponse(
  entries: readonly StreamEntry[],
  scriptedEpisodes: ExtractionFixture["scripted_episodes"],
): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 24,
    output_tokens: 12,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_eval_extract",
        name: EPISODE_TOOL_NAME,
        input: {
          episodes: scriptedEpisodes.map((episode) => ({
            title: episode.title,
            narrative: episode.narrative,
            source_stream_ids: episode.source_entry_indexes
              .map((index) => entries[index]?.id)
              .filter((id): id is StreamEntry["id"] => id !== undefined),
            participants: episode.participants,
            location: null,
            tags: episode.tags,
            confidence: episode.confidence,
            significance: episode.significance,
          })),
        },
      },
    ],
  };
}

function overlapsParticipants(expected: readonly string[], actual: readonly string[]): boolean {
  const actualSet = new Set(actual.map((value) => value.toLowerCase()));
  return expected.some((value) => actualSet.has(value.toLowerCase()));
}

function tokenizeEvalText(text: string): Set<string> {
  return new Set(
    text
      .normalize("NFKC")
      .toLowerCase()
      .split(/[^\p{L}\p{N}_-]+/u)
      .map((token) => token.trim())
      .filter((token) => token.length >= 2),
  );
}

function containsKeywords(text: string, keywords: readonly string[]): boolean {
  const tokens = tokenizeEvalText(text);
  return keywords.every((keyword) => tokens.has(keyword.toLowerCase()));
}

export const episodicExtractionQualityMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const fixtures = loadMetricFixtures(METRIC_NAME, extractionFixtureSchema);
    const cases: EvalCaseResult[] = [];
    let totalInserted = 0;
    let totalExpectedEpisodes = 0;
    let totalMatches = 0;
    let passed = true;

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
        const entries: StreamEntry[] = [];

        for (const entry of fixture.data.stream_entries) {
          entries.push(
            await borg.stream.append({
              kind: entry.kind,
              content: entry.content,
              ...(entry.audience === undefined ? {} : { audience: entry.audience }),
            }),
          );
          clock.advance(60_000);
        }

        llm.pushResponse(createEpisodeToolResponse(entries, fixture.data.scripted_episodes));
        const extraction = await borg.episodic.extract();
        const listed = (await borg.episodic.list({ limit: 20 })).items;

        totalInserted += extraction.inserted;
        totalExpectedEpisodes += fixture.data.expected.length;

        for (const expected of fixture.data.expected) {
          const match = listed.find(
            (episode) =>
              containsKeywords(episode.title, expected.title_keywords) &&
              containsKeywords(episode.narrative, expected.narrative_keywords) &&
              overlapsParticipants(expected.participants, episode.participants),
          );
          const casePassed = match !== undefined;
          totalMatches += casePassed ? 1 : 0;
          passed &&= casePassed;

          cases.push({
            name: `${fixture.name}:${expected.name}`,
            passed: casePassed,
            actual:
              match === undefined
                ? { matched_episode_title: null }
                : {
                    matched_episode_title: match.title,
                    matched_participants: match.participants,
                  },
            expected: {
              title_keywords: expected.title_keywords,
              narrative_keywords: expected.narrative_keywords,
              participants: expected.participants,
            },
          });
        }

        const countPassed =
          extraction.inserted === fixture.data.expected.length &&
          fixture.data.expected.length === fixture.data.scripted_episodes.length;
        passed &&= countPassed;
        cases.push({
          name: `${fixture.name}:insert-count`,
          passed: countPassed,
          actual: `${extraction.inserted}/${fixture.data.expected.length}`,
          expected: `${fixture.data.expected.length}/${fixture.data.expected.length}`,
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
        episodes_inserted: `${totalInserted}/${totalExpectedEpisodes}`,
        matched_expectations: `${totalMatches}/${totalExpectedEpisodes}`,
      },
      expected: {
        episodes_inserted: "2/2",
        matched_expectations: "2/2",
      },
      duration_ms: Date.now() - startedAt,
      cases,
    };
  },
} satisfies EvalMetricModule;

export default episodicExtractionQualityMetric;
