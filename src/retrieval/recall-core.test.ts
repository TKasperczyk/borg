import { readFileSync } from "node:fs";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { TurnTracer } from "../cognition/tracing/tracer.js";
import { FakeLLMClient, type LLMCompleteResult } from "../llm/index.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  TestEmbeddingClient,
  type OfflineTestHarness,
} from "../offline/test-support.js";
import { StreamWriter } from "../stream/index.js";
import { FixedClock, ManualClock } from "../util/clock.js";
import { createSessionId } from "../util/ids.js";
import { RetrievalPipeline } from "./pipeline.js";
import { expandRecall } from "./recall-expansion.js";

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

function createProjectionEmbeddingClient() {
  return new TestEmbeddingClient(
    new Map([
      ["Atlas projection", [1, 0, 0, 0]],
      ["Atlas", [1, 0, 0, 0]],
      ["recent memory", [0, 0, 1, 0]],
    ]),
  );
}

function createStructuralEmbeddingClient() {
  return new TestEmbeddingClient(
    new Map([
      ["Atlas", [1, 0, 0, 0]],
      ["Atlas dedupe", [1, 0, 0, 0]],
      ["Atlas semantic shape", [1, 0, 0, 0]],
      ["Atlas open questions", [1, 0, 0, 0]],
      ["Atlas MMR drop", [1, 0, 0, 0]],
      ["recent memory", [0, 0, 1, 0]],
    ]),
  );
}

function createCommitmentEmbeddingClient(vectors: ReadonlyMap<string, readonly number[]>) {
  return new TestEmbeddingClient(vectors);
}

function createTracer() {
  const emit = vi.fn<TurnTracer["emit"]>();

  return {
    enabled: true,
    includePayloads: false,
    emit,
  } satisfies TurnTracer & { emit: typeof emit };
}

function createTracedRetrievalPipeline(harness: OfflineTestHarness, tracer: TurnTracer) {
  return new RetrievalPipeline({
    embeddingClient: harness.embeddingClient,
    llmClient: harness.llmClient,
    recallExpansionModel: harness.config.anthropic.models.recallExpansion,
    episodicRepository: harness.episodicRepository,
    semanticNodeRepository: harness.semanticNodeRepository,
    semanticGraph: harness.semanticGraph,
    reviewQueueRepository: harness.reviewQueueRepository,
    openQuestionsRepository: harness.openQuestionsRepository,
    entityRepository: harness.entityRepository,
    commitmentRepository: harness.commitmentRepository,
    dataDir: harness.tempDir,
    clock: harness.clock,
    tracer,
    semanticUnderReviewMultiplier: harness.config.retrieval.semantic.underReviewMultiplier,
  });
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

  it("accepts sixteen recall expansion named terms and rejects more than sixteen", async () => {
    const namedTerms = Array.from({ length: 16 }, (_, index) => `Term ${index + 1}`);
    const acceptedClient = new FakeLLMClient({
      responses: [recallExpansion({ named_terms: namedTerms })],
    });

    await expect(
      expandRecall({
        llmClient: acceptedClient,
        model: "test-recall-expansion",
        userMessage: "Remember these entity-rich project references.",
      }),
    ).resolves.toEqual({
      facets: [],
      named_terms: namedTerms,
    });

    const rejectedClient = new FakeLLMClient({
      responses: [recallExpansion({ named_terms: [...namedTerms, "Term 17"] })],
    });

    await expect(
      expandRecall({
        llmClient: rejectedClient,
        model: "test-recall-expansion",
        userMessage: "Remember these entity-rich project references.",
      }),
    ).rejects.toThrow();
  });

  it("traces recall expansion LLM calls on success", async () => {
    const tracer = createTracer();
    const llmClient = new FakeLLMClient({
      responses: [recallExpansion({ named_terms: ["Maya"] })],
    });
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createEmbeddingClient(),
      llmClient,
    });
    const pipeline = createTracedRetrievalPipeline(harness, tracer);

    await pipeline.searchWithContext(MAYA_TURN, {
      limit: 3,
      traceTurnId: "turn-recall-expansion",
    });

    expect(tracer.emit).toHaveBeenCalledWith("llm_call_started", {
      turnId: "turn-recall-expansion",
      label: "recall_expansion",
      model: harness.config.anthropic.models.recallExpansion,
      promptCharCount: expect.any(Number),
      toolSchemas: expect.any(Array),
    });
    expect(tracer.emit).toHaveBeenCalledWith("llm_call_response", {
      turnId: "turn-recall-expansion",
      label: "recall_expansion",
      responseShape: {
        textLength: 0,
        toolUseBlocks: [
          {
            id: "toolu_recall_expansion",
            name: "EmitRecallExpansion",
          },
        ],
      },
      stopReason: "tool_use",
      usage: {
        inputTokens: 0,
        outputTokens: 0,
      },
    });
  });

  it("traces recall expansion LLM responses before schema parse failures degrade retrieval", async () => {
    const tracer = createTracer();
    const llmClient = new FakeLLMClient({
      responses: [
        recallExpansion({
          named_terms: Array.from({ length: 17 }, (_, index) => `Term ${index + 1}`),
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createEmbeddingClient(),
      llmClient,
    });
    const pipeline = createTracedRetrievalPipeline(harness, tracer);

    await pipeline.searchWithContext(MAYA_TURN, {
      limit: 3,
      entityTerms: ["Maya"],
      traceTurnId: "turn-recall-expansion-parse-failure",
    });

    expect(tracer.emit).toHaveBeenCalledWith(
      "llm_call_response",
      expect.objectContaining({
        turnId: "turn-recall-expansion-parse-failure",
        label: "recall_expansion",
        stopReason: "tool_use",
        usage: {
          inputTokens: 0,
          outputTokens: 0,
        },
      }),
    );
    expect(tracer.emit).toHaveBeenCalledWith(
      "retrieval_degraded",
      expect.objectContaining({
        turnId: "turn-recall-expansion-parse-failure",
        subsystem: "recall_expansion",
      }),
    );
  });

  it("traces recall expansion transport failures as LLM responses", async () => {
    const tracer = createTracer();
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createEmbeddingClient(),
      llmClient: throwingRecallExpansion(),
    });
    const pipeline = createTracedRetrievalPipeline(harness, tracer);

    await pipeline.searchWithContext(MAYA_TURN, {
      limit: 3,
      entityTerms: ["Maya"],
      traceTurnId: "turn-recall-expansion-transport-failure",
    });

    expect(tracer.emit).toHaveBeenCalledWith("llm_call_response", {
      turnId: "turn-recall-expansion-transport-failure",
      label: "recall_expansion",
      responseShape: {
        error: "recall expansion unavailable",
      },
      stopReason: null,
      usage: null,
    });
  });

  it("unions perception entities when recall expansion succeeds with no named terms", async () => {
    const llmClient = new FakeLLMClient({
      responses: [recallExpansion({ named_terms: [] })],
    });
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createEmbeddingClient(),
      llmClient,
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

  it("dedupes known terms with LLM expansion source precedence", async () => {
    const llmClient = new FakeLLMClient({
      responses: [recallExpansion({ named_terms: ["Maya"] })],
    });
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createEmbeddingClient(),
      llmClient,
    });
    await insertMayaAndDesignReview(harness);

    const result = await harness.retrievalPipeline.searchWithContext(MAYA_TURN, {
      limit: 5,
      entityTerms: ["Maya"],
      audienceTerms: ["Maya"],
    });
    const mayaIntents = result.recall_intents.filter(
      (intent) => intent.kind === "known_term" && intent.terms[0] === "Maya",
    );

    expect(mayaIntents).toHaveLength(1);
    expect(mayaIntents[0]?.source).toBe("llm-expansion");
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

  it("emits commitment evidence only for embedding-matched commitments", async () => {
    const commitmentQuery = "Atlas confidentiality boundary";
    const matchingDirective = "Do not discuss Atlas private deployment details with Sam.";
    const unrelatedDirective = "Send Alice the weekly deployment summary.";
    const llmClient = new FakeLLMClient({
      responses: [
        recallExpansion({
          facets: [{ kind: "commitment", query: commitmentQuery, priority: 1 }],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createCommitmentEmbeddingClient(
        new Map([
          [commitmentQuery, [1, 0, 0, 0]],
          [matchingDirective, [1, 0, 0, 0]],
          [unrelatedDirective, [0, 1, 0, 0]],
        ]),
      ),
      llmClient,
    });
    const matching = harness.commitmentRepository.add({
      type: "boundary",
      directiveFamily: "atlas_confidentiality",
      directive: matchingDirective,
      priority: 8,
      provenance: { kind: "manual" },
    });
    const unrelated = harness.commitmentRepository.add({
      type: "promise",
      directiveFamily: "public_launch_date",
      directive: unrelatedDirective,
      priority: 9,
      provenance: { kind: "manual" },
    });

    const result = await harness.retrievalPipeline.searchWithContext(
      "Can we talk about Atlas confidentiality?",
      { limit: 3 },
    );
    const commitmentIds = result.evidence
      .filter((item) => item.source === "commitment")
      .map((item) => item.provenance?.commitmentId);

    expect(commitmentIds).toEqual([matching.id]);
    expect(commitmentIds).not.toContain(unrelated.id);
  });

  it("emits no commitment evidence when a commitment intent has no embedding match", async () => {
    const commitmentQuery = "public launch-date promise";
    const firstDirective = "Do not discuss Atlas private deployment details with Sam.";
    const secondDirective = "Keep Sam planning details scoped to Sam.";
    const llmClient = new FakeLLMClient({
      responses: [
        recallExpansion({
          facets: [{ kind: "commitment", query: commitmentQuery, priority: 1 }],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createCommitmentEmbeddingClient(
        new Map([
          [commitmentQuery, [1, 0, 0, 0]],
          [firstDirective, [0, 1, 0, 0]],
          [secondDirective, [0, 0, 1, 0]],
        ]),
      ),
      llmClient,
    });
    harness.commitmentRepository.add({
      type: "boundary",
      directiveFamily: "first_commitment_directive",
      directive: firstDirective,
      priority: 8,
      provenance: { kind: "manual" },
    });
    harness.commitmentRepository.add({
      type: "rule",
      directiveFamily: "second_commitment_directive",
      directive: secondDirective,
      priority: 7,
      provenance: { kind: "manual" },
    });

    const result = await harness.retrievalPipeline.searchWithContext("What can we promise?", {
      limit: 3,
    });

    expect(result.evidence.filter((item) => item.source === "commitment")).toEqual([]);
  });

  it("does not use substring matching for commitment evidence", () => {
    const retrievalSource = readFileSync(
      join(process.cwd(), "src", "retrieval", "pipeline.ts"),
      "utf8",
    );

    expect(retrievalSource).not.toContain("matchedCommitmentTerms");
    expect(retrievalSource).not.toMatch(/directive[\s\S]{0,200}\.indexOf\s*\(/);
  });

  it("ranks matched episode evidence above recent raw stream tail context", async () => {
    const clock = new ManualClock(NOW_MS - 10_000);
    harness = await createOfflineTestHarness({
      clock,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          ["Atlas", [1, 0, 0, 0]],
          ["recent memory", [0, 1, 0, 0]],
        ]),
      ),
      llmClient: throwingRecallExpansion(),
    });
    const episode = createEpisodeFixture(
      {
        title: "Atlas source-backed memory",
        narrative: "Atlas has a known-term episode that should outrank raw recency tail.",
        participants: ["Atlas"],
        tags: ["Atlas"],
        significance: 1,
        created_at: NOW_MS - 1_000_000,
        updated_at: NOW_MS - 1_000_000,
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(episode);
    clock.set(NOW_MS);
    const recent = await harness.streamWriter.append({
      kind: "user_msg",
      content: "Unrelated recent chatter",
    });

    const result = await harness.retrievalPipeline.searchWithContext("Atlas", {
      limit: 3,
      entityTerms: ["Atlas"],
    });
    const episodeEvidenceIndex = result.evidence.findIndex(
      (item) => item.source === "episode" && item.provenance?.episodeId === episode.id,
    );
    const recentTailIndex = result.evidence.findIndex(
      (item) =>
        item.source === "recent_raw_stream" && item.provenance?.streamIds?.includes(recent.id),
    );

    expect(episodeEvidenceIndex).toBeGreaterThanOrEqual(0);
    expect(recentTailIndex).toBeGreaterThanOrEqual(0);
    expect(episodeEvidenceIndex).toBeLessThan(recentTailIndex);
  });

  it("does not include prior-session raw stream tail evidence for a fresh session", async () => {
    const clock = new ManualClock(NOW_MS - 1_000);
    harness = await createOfflineTestHarness({
      clock,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          ["nothing relevant", [0, 1, 0, 0]],
          ["recent memory", [0, 1, 0, 0]],
        ]),
      ),
      llmClient: throwingRecallExpansion(),
    });
    const priorSession = createSessionId();
    const freshSession = createSessionId();
    const priorWriter = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: priorSession,
      clock,
    });
    const freshWriter = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: freshSession,
      clock,
    });

    const priorEntry = await priorWriter.append({
      kind: "user_msg",
      content: "Prior session stream tail",
    });
    clock.set(NOW_MS);
    const freshEntry = await freshWriter.append({
      kind: "user_msg",
      content: "Fresh session stream tail",
    });
    priorWriter.close();
    freshWriter.close();

    const result = await harness.retrievalPipeline.searchWithContext("nothing relevant", {
      sessionId: freshSession,
      limit: 3,
    });
    const recentStreamIds = result.evidence
      .filter((item) => item.source === "recent_raw_stream")
      .flatMap((item) => item.provenance?.streamIds ?? []);

    expect(recentStreamIds).toContain(freshEntry.id);
    expect(recentStreamIds).not.toContain(priorEntry.id);
  });

  it("projects legacy fields from the ranked evidence pool", async () => {
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createProjectionEmbeddingClient(),
      llmClient: throwingRecallExpansion(),
    });
    const entry = await harness.streamWriter.append({
      kind: "user_msg",
      content: "Atlas projection source",
    });
    const episode = createEpisodeFixture(
      {
        title: "Atlas projection episode",
        narrative: "The Atlas projection needs evidence-backed retrieval.",
        participants: ["Atlas"],
        tags: ["Atlas"],
        source_stream_ids: [entry.id],
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(episode);
    const atlas = await harness.semanticNodeRepository.insert({
      id: "semn_aaaaaaaaaaaaaaaa" as never,
      kind: "entity",
      label: "Atlas",
      description: "Atlas projection root",
      aliases: [],
      confidence: 0.9,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const support = await harness.semanticNodeRepository.insert({
      id: "semn_bbbbbbbbbbbbbbbb" as never,
      kind: "proposition",
      label: "Projection is evidence-backed",
      description: "Projection should hydrate compatibility fields from evidence.",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const supportEdge = harness.semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: support.id,
      relation: "supports",
      confidence: 0.8,
      evidence_episode_ids: [episode.id],
      created_at: 1,
      last_verified_at: 1,
    });
    const question = harness.openQuestionsRepository.add({
      question: "Why does Atlas projection need evidence pool invariants?",
      urgency: 0.9,
      related_semantic_node_ids: [atlas.id],
      source: "reflection",
    });

    const result = await harness.retrievalPipeline.searchWithContext("Atlas projection", {
      limit: 3,
      entityTerms: ["Atlas"],
      includeOpenQuestions: true,
      graphWalkDepth: 1,
      maxGraphNodes: 4,
    });
    const episodeEvidenceIds = new Set(
      result.evidence
        .filter((item) => item.source === "episode")
        .map((item) => item.provenance?.episodeId),
    );
    const semanticNodeEvidenceIds = new Set(
      result.evidence
        .filter((item) => item.source === "semantic_node")
        .map((item) => item.provenance?.nodeId),
    );
    const semanticEdgeEvidenceIds = new Set(
      result.evidence
        .filter((item) => item.source === "semantic_edge")
        .map((item) => item.provenance?.edgeId),
    );
    const openQuestionEvidenceIds = new Set(
      result.evidence
        .filter((item) => item.source === "open_question")
        .map((item) => item.provenance?.openQuestionId),
    );

    expect(result.episodes.length).toBeGreaterThan(0);
    expect(result.semantic.matched_nodes.length).toBeGreaterThan(0);
    expect(result.semantic.support_hits.length).toBeGreaterThan(0);
    expect(result.open_questions).toEqual([expect.objectContaining({ id: question.id })]);
    for (const item of result.episodes) {
      expect(episodeEvidenceIds.has(item.episode.id)).toBe(true);
    }
    for (const node of result.semantic.matched_nodes) {
      expect(semanticNodeEvidenceIds.has(node.id)).toBe(true);
    }
    for (const hit of result.semantic.support_hits) {
      expect(semanticEdgeEvidenceIds.has(hit.edgePath.at(-1)?.id)).toBe(true);
    }
    for (const openQuestion of result.open_questions) {
      expect(openQuestionEvidenceIds.has(openQuestion.id)).toBe(true);
    }
    expect(semanticEdgeEvidenceIds.has(supportEdge.id)).toBe(true);
  });

  it("projects a multi-intent deduped episode once in episodes and evidence", async () => {
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createStructuralEmbeddingClient(),
      llmClient: throwingRecallExpansion(),
    });
    const episode = createEpisodeFixture(
      {
        title: "Atlas dedupe episode",
        narrative: "Atlas should be found by both vector and known-term recall.",
        participants: ["Atlas"],
        tags: ["Atlas"],
        significance: 1,
        created_at: NOW_MS,
        updated_at: NOW_MS,
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(episode);

    const result = await harness.retrievalPipeline.searchWithContext("Atlas dedupe", {
      limit: 5,
      entityTerms: ["Atlas"],
    });
    const projected = result.episodes.filter((item) => item.episode.id === episode.id);
    const evidence = result.evidence.filter(
      (item) => item.source === "episode" && item.provenance?.episodeId === episode.id,
    );

    expect(projected).toHaveLength(1);
    expect(evidence).toHaveLength(1);
  });

  it("projects semantic node and edge evidence with matching provenance", async () => {
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createStructuralEmbeddingClient(),
      llmClient: throwingRecallExpansion(),
    });
    const episode = createEpisodeFixture(
      {
        title: "Atlas semantic source",
        narrative: "Atlas semantic retrieval has a support edge.",
        participants: ["Atlas"],
        tags: ["Atlas"],
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(episode);
    const atlas = await harness.semanticNodeRepository.insert({
      id: "semn_cccccccccccccccc" as never,
      kind: "entity",
      label: "Atlas",
      description: "Atlas semantic shape root",
      aliases: [],
      confidence: 0.9,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const support = await harness.semanticNodeRepository.insert({
      id: "semn_dddddddddddddddd" as never,
      kind: "proposition",
      label: "Atlas has edge support",
      description: "A support node reached through the graph should project from evidence.",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: support.id,
      relation: "supports",
      confidence: 0.85,
      evidence_episode_ids: [episode.id],
      created_at: 1,
      last_verified_at: 1,
    });

    const result = await harness.retrievalPipeline.searchWithContext("Atlas semantic shape", {
      limit: 3,
      entityTerms: ["Atlas"],
      graphWalkDepth: 1,
      maxGraphNodes: 4,
    });
    const evidenceNodeIds = result.evidence
      .filter((item) => item.source === "semantic_node")
      .map((item) => item.provenance?.nodeId);
    const evidenceEdgeIds = result.evidence
      .filter((item) => item.source === "semantic_edge")
      .map((item) => item.provenance?.edgeId);
    const projectedNodeIds = new Set(result.semantic.matched_nodes.map((node) => node.id));
    const projectedEdgeIds = new Set(
      result.semantic.support_hits.map((hit) => hit.edgePath.at(-1)?.id),
    );

    expect(evidenceNodeIds).toContain(atlas.id);
    expect(evidenceEdgeIds).toContain(edge.id);
    expect(projectedNodeIds.has(atlas.id)).toBe(true);
    expect(projectedEdgeIds.has(edge.id)).toBe(true);
  });

  it("projects multiple matched open questions from the evidence pool", async () => {
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createStructuralEmbeddingClient(),
      llmClient: throwingRecallExpansion(),
    });
    const episode = createEpisodeFixture(
      {
        title: "Atlas question source",
        narrative: "Atlas has unresolved reflective questions.",
        participants: ["Atlas"],
        tags: ["Atlas"],
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(episode);
    const atlas = await harness.semanticNodeRepository.insert({
      id: "semn_eeeeeeeeeeeeeeee" as never,
      kind: "entity",
      label: "Atlas",
      description: "Atlas open-question root",
      aliases: [],
      confidence: 0.9,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const first = harness.openQuestionsRepository.add({
      question: "What Atlas invariant needs monitoring?",
      urgency: 0.8,
      related_semantic_node_ids: [atlas.id],
      source: "reflection",
    });
    const second = harness.openQuestionsRepository.add({
      question: "Which Atlas projection could drift next?",
      urgency: 0.7,
      related_semantic_node_ids: [atlas.id],
      source: "reflection",
    });

    const result = await harness.retrievalPipeline.searchWithContext("Atlas open questions", {
      limit: 3,
      entityTerms: ["Atlas"],
      includeOpenQuestions: true,
      openQuestionsLimit: 3,
    });
    const projectedIds = result.open_questions.map((question) => question.id);
    const evidenceIds = result.evidence
      .filter((item) => item.source === "open_question")
      .map((item) => item.provenance?.openQuestionId);

    expect(projectedIds).toEqual(expect.arrayContaining([first.id, second.id]));
    expect(evidenceIds).toEqual(expect.arrayContaining([first.id, second.id]));
  });

  it("keeps MMR-dropped episode evidence in the evidence pool", async () => {
    harness = await createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: createStructuralEmbeddingClient(),
      llmClient: throwingRecallExpansion(),
    });
    const primary = createEpisodeFixture(
      {
        title: "Atlas primary MMR episode",
        narrative: "The higher-scoring Atlas episode should be projected.",
        participants: ["Atlas"],
        tags: ["Atlas"],
        significance: 1,
        created_at: NOW_MS,
        updated_at: NOW_MS,
      },
      [1, 0, 0, 0],
    );
    const secondary = createEpisodeFixture(
      {
        title: "Atlas secondary MMR episode",
        narrative: "The lower-scoring Atlas episode should remain evidence even if unprojected.",
        participants: ["Atlas"],
        tags: ["Atlas"],
        significance: 0.2,
        created_at: NOW_MS - 100_000,
        updated_at: NOW_MS - 100_000,
      },
      [0.8, 0.2, 0, 0],
    );
    await harness.episodicRepository.insert(primary);
    await harness.episodicRepository.insert(secondary);

    const result = await harness.retrievalPipeline.searchWithContext("Atlas MMR drop", {
      limit: 1,
      entityTerms: ["Atlas"],
    });
    const candidateIds = [primary.id, secondary.id];
    const projectedIds = new Set(result.episodes.map((item) => item.episode.id));
    const evidenceIds = result.evidence
      .filter((item) => item.source === "episode")
      .map((item) => item.provenance?.episodeId);
    const droppedIds = candidateIds.filter((id) => !projectedIds.has(id));

    expect(result.episodes).toHaveLength(1);
    expect(evidenceIds).toEqual(expect.arrayContaining(candidateIds));
    expect(droppedIds).toHaveLength(1);
    expect(evidenceIds).toContain(droppedIds[0]);
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
