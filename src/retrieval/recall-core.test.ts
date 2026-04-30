import { readFileSync } from "node:fs";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../llm/index.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  TestEmbeddingClient,
  type OfflineTestHarness,
} from "../offline/test-support.js";
import { FixedClock } from "../util/clock.js";

const NOW_MS = 10_000_000_000;
const MAYA_TURN = "my partner's not Maya. Also, Thursday's design review is next week.";

function recallExpansion(input: {
  facets?: Array<{
    kind: "topic" | "relationship" | "commitment" | "open_question";
    query: string;
    priority: number;
  }>;
  named_terms?: string[];
}): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_recall_expansion",
        name: "EmitRecallExpansion",
        input: {
          facets: input.facets ?? [],
          named_terms: input.named_terms ?? [],
        },
      },
    ],
  };
}

function throwingRecallExpansion() {
  return new FakeLLMClient({
    responses: [
      () => {
        throw new Error("recall expansion unavailable");
      },
    ],
  });
}

function createEmbeddingClient() {
  return new TestEmbeddingClient(
    new Map([
      [MAYA_TURN, [1, 0, 0, 0]],
      ["Maya", [0, 1, 0, 0]],
      ["recent memory", [0, 0, 1, 0]],
      ["unrelated turn", [1, 0, 0, 0]],
    ]),
  );
}

async function insertMayaAndDesignReview(harness: OfflineTestHarness) {
  const nextWeekStart = NOW_MS + 5 * 24 * 60 * 60 * 1_000;
  const nextWeekEnd = nextWeekStart + 7 * 24 * 60 * 60 * 1_000;
  const mayaEpisode = createEpisodeFixture(
    {
      title: "Prior relationship correction",
      narrative: "Earlier turns associated the user's partner context with Maya.",
      participants: ["Maya"],
      tags: ["Maya", "relationship"],
      significance: 1,
      created_at: 1_000,
      updated_at: 1_000,
      start_time: 1_000,
      end_time: 2_000,
    },
    [0, 1, 0, 0],
  );
  const designReviewEpisode = createEpisodeFixture(
    {
      title: "Thursday design review",
      narrative: "The design review is scheduled for Thursday next week.",
      participants: ["design"],
      tags: ["review"],
      significance: 0.8,
      created_at: NOW_MS,
      updated_at: NOW_MS,
      start_time: nextWeekStart + 3 * 24 * 60 * 60 * 1_000,
      end_time: nextWeekStart + 3 * 24 * 60 * 60 * 1_000 + 60 * 60 * 1_000,
    },
    [1, 0, 0, 0],
  );

  await harness.episodicRepository.insert(mayaEpisode);
  await harness.episodicRepository.insert(designReviewEpisode);

  return {
    mayaEpisode,
    designReviewEpisode,
    nextWeekStart,
    nextWeekEnd,
  };
}

describe("Recall Core", () => {
  let harness: OfflineTestHarness | undefined;

  afterEach(async () => {
    await harness?.cleanup();
    harness = undefined;
  });

  it("uses LLM named_terms for known-term recall when perception omitted the name", async () => {
    const llmClient = new FakeLLMClient({
      responses: [recallExpansion({ named_terms: ["Maya"] })],
    });
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createEmbeddingClient(),
      llmClient,
    });
    const { mayaEpisode, nextWeekStart, nextWeekEnd } = await insertMayaAndDesignReview(harness);

    const result = await harness.retrievalPipeline.searchWithContext(MAYA_TURN, {
      limit: 5,
      entityTerms: ["Otto"],
      temporalCue: {
        label: "next week",
        sinceTs: nextWeekStart,
        untilTs: nextWeekEnd,
      },
      strictTimeRange: true,
    });

    expect(result.episodes.map((item) => item.episode.id)).toContain(mayaEpisode.id);
    expect(
      result.recall_intents.find(
        (intent) => intent.kind === "known_term" && intent.terms[0] === "Maya",
      )?.source,
    ).toBe("llm-expansion");
    expect(result.evidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          source: "episode",
          provenance: expect.objectContaining({ episodeId: mayaEpisode.id }),
          matchedTerms: ["Maya"],
        }),
      ]),
    );
  });

  it("falls back to perception entities when recall expansion fails", async () => {
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createEmbeddingClient(),
      llmClient: throwingRecallExpansion(),
    });
    const { mayaEpisode } = await insertMayaAndDesignReview(harness);

    const result = await harness.retrievalPipeline.searchWithContext(MAYA_TURN, {
      limit: 5,
      entityTerms: ["Maya"],
    });

    expect(result.episodes.map((item) => item.episode.id)).toContain(mayaEpisode.id);
    expect(
      result.recall_intents.find(
        (intent) => intent.kind === "known_term" && intent.terms[0] === "Maya",
      )?.source,
    ).toBe("perception-entities");
  });

  it("degrades to raw-text and recent intents when expansion and perception terms are unavailable", async () => {
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createEmbeddingClient(),
      llmClient: throwingRecallExpansion(),
    });
    const recentEpisode = createEpisodeFixture({
      title: "Recent fallback memory",
      narrative: "A recent memory remains available when semantic expansion fails.",
      created_at: NOW_MS,
      updated_at: NOW_MS,
    });
    await harness.episodicRepository.insert(recentEpisode);

    const result = await harness.retrievalPipeline.searchWithContext("unrelated turn", {
      limit: 3,
    });

    expect(result.recall_intents.map((intent) => intent.kind)).toEqual(
      expect.arrayContaining(["raw_text", "recent"]),
    );
    expect(result.evidence.length).toBeGreaterThan(0);
    expect(result.episodes.map((item) => item.episode.id)).toContain(recentEpisode.id);
  });

  it("keeps strict temporal filters local to the time intent", async () => {
    const llmClient = new FakeLLMClient({
      responses: [recallExpansion({ named_terms: ["Maya"] })],
    });
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createEmbeddingClient(),
      llmClient,
    });
    const { mayaEpisode, nextWeekStart, nextWeekEnd } = await insertMayaAndDesignReview(harness);

    const result = await harness.retrievalPipeline.searchWithContext(MAYA_TURN, {
      limit: 5,
      temporalCue: {
        label: "next week",
        sinceTs: nextWeekStart,
        untilTs: nextWeekEnd,
      },
      strictTimeRange: true,
    });

    const knownTermIntent = result.recall_intents.find((intent) => intent.kind === "known_term");
    const timeIntent = result.recall_intents.find((intent) => intent.kind === "time");

    expect(timeIntent).toEqual(expect.objectContaining({ strictTime: true }));
    expect(knownTermIntent?.timeRange).toBeUndefined();
    expect(result.episodes.map((item) => item.episode.id)).toContain(mayaEpisode.id);
  });

  it("hydrates episode provenance into raw stream evidence when source entries exist", async () => {
    const llmClient = new FakeLLMClient({
      responses: [recallExpansion({ named_terms: ["Maya"] })],
    });
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createEmbeddingClient(),
      llmClient,
    });
    const entry = await harness.streamWriter.append({
      kind: "user_msg",
      content: "Maya source snippet",
    });
    const episode = createEpisodeFixture(
      {
        title: "Maya source-backed episode",
        narrative: "The source stream has the raw Maya wording.",
        participants: ["Maya"],
        tags: ["Maya"],
        source_stream_ids: [entry.id],
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.insert(episode);

    const result = await harness.retrievalPipeline.searchWithContext("Maya", {
      limit: 1,
      entityTerms: ["Maya"],
    });

    expect(result.episodes[0]?.citationChain[0]?.content).toBe("Maya source snippet");
    expect(result.evidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          source: "raw_stream",
          text: "Maya source snippet",
          provenance: expect.objectContaining({ streamIds: [entry.id] }),
        }),
      ]),
    );
  });

  it("does not add a bolt-on factual-challenge, Maya-specific, or correction-only lane", () => {
    const retrievalSource = readFileSync(
      join(process.cwd(), "src", "retrieval", "pipeline.ts"),
      "utf8",
    );

    expect(retrievalSource).not.toContain("factual-challenge");
    expect(retrievalSource).not.toContain("Maya");
    expect(retrievalSource).not.toContain("correction-only");
  });

  it("keeps candidate-term identification in the recall expansion tool output", () => {
    const expansionSource = readFileSync(
      join(process.cwd(), "src", "retrieval", "recall-expansion.ts"),
      "utf8",
    );
    const pipelineSource = readFileSync(
      join(process.cwd(), "src", "retrieval", "pipeline.ts"),
      "utf8",
    );

    expect(expansionSource).toContain("named_terms");
    expect(`${expansionSource}\n${pipelineSource}`).not.toContain(`tokenize${"Text"}`);
    expect(`${expansionSource}\n${pipelineSource}`).not.toContain("capitalized");
    expect(`${expansionSource}\n${pipelineSource}`).not.toContain("n-gram");
    expect(`${expansionSource}\n${pipelineSource}`).not.toContain("ngram");
  });
});
