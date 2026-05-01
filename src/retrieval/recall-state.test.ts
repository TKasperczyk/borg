import { afterEach, describe, expect, it, vi } from "vitest";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  TestEmbeddingClient,
  testSessionId,
  type OfflineTestHarness,
} from "../offline/test-support.js";
import { createWorkingMemory, workingMemorySchema } from "../memory/working/index.js";
import { FixedClock } from "../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createCommitmentId,
  createEntityId,
  createEpisodeId,
  createOpenQuestionId,
  createSemanticEdgeId,
  createSemanticNodeId,
  createStreamEntryId,
  type SessionId,
} from "../util/ids.js";

import type { RecallStateHandle } from "./recall-state.js";
import { RetrievalPipeline } from "./pipeline.js";

const NOW_MS = 10_000;
const DISTRACTOR_COUNT = 16;

function createEmbeddingClient() {
  return new TestEmbeddingClient(
    new Map([
      ["Atlas", [1, 0, 0, 0]],
      ["Cache", [1, 0, 0, 0]],
      ["Atlas current", [1, 0, 0, 0]],
      ["unrelated recall", [0, 1, 0, 0]],
      ["quiet turn", [0, 1, 0, 0]],
      ["nothing relevant", [0, 1, 0, 0]],
      ["recent memory", [0, 1, 0, 0]],
      ["Maya is my partner. She's making elaborate ramen tonight.", [1, 0, 0, 0]],
      ["My partner isn't Maya. I never said that.", [0, 1, 0, 0]],
    ]),
  );
}

async function createHarness(): Promise<OfflineTestHarness> {
  return createOfflineTestHarness({
    clock: new FixedClock(NOW_MS),
    embeddingClient: createEmbeddingClient(),
  });
}

async function insertDistractors(harness: OfflineTestHarness, options: { prefix?: string } = {}) {
  const prefix = options.prefix ?? "Noise";

  for (let index = 0; index < DISTRACTOR_COUNT; index += 1) {
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: `${prefix} distractor ${index}`,
          narrative: `${prefix} distractor ${index} should occupy fresh retrieval lanes.`,
          participants: [prefix],
          tags: [prefix],
          significance: 1,
          created_at: 20_000 + index,
          updated_at: 20_000 + index,
        },
        [0, 1, 0, 0],
      ),
    );
  }
}

function loadSerializedState(harness: OfflineTestHarness, scopeKey: string): unknown {
  const row = harness.db
    .prepare("SELECT state_json FROM recall_state WHERE scope_key = ?")
    .get(scopeKey) as { state_json: string } | undefined;

  expect(row).toBeDefined();
  return JSON.parse(row!.state_json) as unknown;
}

function collectKeys(value: unknown): string[] {
  if (value === null || typeof value !== "object") {
    return [];
  }

  if (Array.isArray(value)) {
    return value.flatMap((item) => collectKeys(item));
  }

  return Object.entries(value).flatMap(([key, nested]) => [key, ...collectKeys(nested)]);
}

function collectStringValues(value: unknown): string[] {
  if (typeof value === "string") {
    return [value];
  }

  if (value === null || typeof value !== "object") {
    return [];
  }

  if (Array.isArray(value)) {
    return value.flatMap((item) => collectStringValues(item));
  }

  return Object.values(value).flatMap((nested) => collectStringValues(nested));
}

function seedRecallHandles(input: {
  harness: OfflineTestHarness;
  scopeKey: string;
  activeHandles: RecallStateHandle[];
  lastRefreshTurn?: number;
  ttlTurns?: number;
  suppressedHandles?: Record<string, number>;
}) {
  input.harness.recallStateRepository.save({
    scopeKey: input.scopeKey,
    activeHandles: input.activeHandles,
    suppressedHandles: input.suppressedHandles ?? {},
    lastRefreshTurn: input.lastRefreshTurn ?? 1,
    updatedAt: NOW_MS,
    ttlTurns: input.ttlTurns ?? 6,
  });
}

function seedEpisodeHandle(harness: OfflineTestHarness, scopeKey: string, episodeId: string) {
  seedRecallHandles({
    harness,
    scopeKey,
    activeHandles: [
      {
        handle: {
          source: "episode",
          episodeId: episodeId as never,
        },
        firstSeenTurn: 1,
        lastSeenTurn: 1,
        lastRenderedTurn: 1,
        expiresAfterTurn: 7,
        reinforcementCount: 1,
      },
    ],
  });
}

function seedRawStreamHandle(input: {
  harness: OfflineTestHarness;
  scopeKey: string;
  streamId: string;
  parentEpisodeId: string;
}) {
  input.harness.recallStateRepository.save({
    scopeKey: input.scopeKey,
    activeHandles: [
      {
        handle: {
          source: "raw_stream",
          streamIds: [input.streamId as never],
          parentEpisodeId: input.parentEpisodeId as never,
        },
        firstSeenTurn: 1,
        lastSeenTurn: 1,
        lastRenderedTurn: 1,
        expiresAfterTurn: 7,
        reinforcementCount: 1,
      },
    ],
    suppressedHandles: {},
    lastRefreshTurn: 1,
    updatedAt: NOW_MS,
    ttlTurns: 6,
  });
}

function createStateHandle(
  handle: RecallStateHandle["handle"],
  overrides: Partial<Omit<RecallStateHandle, "handle">> = {},
): RecallStateHandle {
  return {
    handle,
    firstSeenTurn: overrides.firstSeenTurn ?? 1,
    lastSeenTurn: overrides.lastSeenTurn ?? 1,
    lastRenderedTurn: overrides.lastRenderedTurn ?? null,
    expiresAfterTurn: overrides.expiresAfterTurn ?? 100,
    reinforcementCount: overrides.reinforcementCount ?? 1,
  };
}

async function insertMatchingEpisodes(input: {
  harness: OfflineTestHarness;
  count: number;
  prefix: string;
  ids?: readonly ReturnType<typeof createEpisodeId>[];
}) {
  const episodes = [];

  for (let index = 0; index < input.count; index += 1) {
    const episode = createEpisodeFixture(
      {
        ...(input.ids?.[index] === undefined ? {} : { id: input.ids[index] }),
        title: `${input.prefix} ${index}`,
        narrative: `${input.prefix} ${index} should be eligible as fresh recall evidence.`,
        participants: [input.prefix],
        tags: [input.prefix],
        significance: 1,
        created_at: 30_000 + index,
        updated_at: 30_000 + index,
      },
      [1, 0, 0, 0],
    );

    await input.harness.episodicRepository.insert(episode);
    episodes.push(episode);
  }

  return episodes;
}

function episodeHandleIds(
  handles: readonly RecallStateHandle[],
): Set<ReturnType<typeof createEpisodeId>> {
  const ids = new Set<ReturnType<typeof createEpisodeId>>();

  for (const item of handles) {
    if (item.handle.source === "episode") {
      ids.add(item.handle.episodeId);
    }
  }

  return ids;
}

describe("retrieval recall_state", () => {
  let harness: OfflineTestHarness | undefined;

  afterEach(async () => {
    vi.restoreAllMocks();
    await harness?.cleanup();
    harness = undefined;
  });

  it("rehydrates stable episode handles instead of per-intent evidence item ids", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const episode = createEpisodeFixture(
      {
        title: "Atlas durable handle",
        narrative: "Atlas should be carried by source handle across turns.",
        participants: ["Atlas"],
        tags: ["Atlas"],
        significance: 0.1,
        created_at: 1_000,
        updated_at: 1_000,
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(episode);
    await insertDistractors(harness);

    const staleEvidenceId = `evidence_episode_${episode.id}_recall_known_term_0`;
    seedEpisodeHandle(harness, sessionId, episode.id);

    const second = await harness.retrievalPipeline.searchWithContext("unrelated recall", {
      sessionId,
      turnCounter: 2,
      limit: 1,
    });
    const warm = second.evidence.find(
      (item) => item.source === "warm_recall" && item.provenance?.episodeId === episode.id,
    );
    const serialized = JSON.stringify(loadSerializedState(harness, sessionId));

    expect(warm).toBeDefined();
    expect(warm?.id).not.toBe(staleEvidenceId);
    expect(serialized).toContain(episode.id);
    expect(serialized).not.toContain(staleEvidenceId);
    expect(serialized).not.toContain("evidence_episode_");
  });

  it("serializes only handles and counters, not text, descriptions, summaries, or beliefs", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const episode = createEpisodeFixture({
      title: "Atlas serialized title",
      narrative: "Atlas serialized narrative must not enter recall_state.",
      participants: ["Atlas"],
      tags: ["Atlas"],
      significance: 0.1,
      created_at: 1_000,
      updated_at: 1_000,
    });
    await harness.episodicRepository.insert(episode);
    seedEpisodeHandle(harness, sessionId, episode.id);

    const state = loadSerializedState(harness, sessionId);
    const keys = collectKeys(state);
    const serialized = JSON.stringify(state);

    expect(keys).not.toEqual(
      expect.arrayContaining([
        "id",
        "text",
        "summary",
        "description",
        "content",
        "belief",
        "claim",
      ]),
    );
    expect(serialized).not.toContain("Atlas serialized title");
    expect(serialized).not.toContain("Atlas serialized narrative");
  });

  it("rehydrates episode text from the current source repository", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const episode = createEpisodeFixture(
      {
        title: "Atlas mutable source",
        narrative: "Original source narrative.",
        participants: ["Atlas"],
        tags: ["Atlas"],
        significance: 0.1,
        created_at: 1_000,
        updated_at: 1_000,
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(episode);
    seedEpisodeHandle(harness, sessionId, episode.id);

    await harness.episodicRepository.update(episode.id, {
      narrative: "Updated source narrative from storage.",
    });
    await insertDistractors(harness);
    const getSpy = vi.spyOn(harness.episodicRepository, "get");

    const second = await harness.retrievalPipeline.searchWithContext("unrelated recall", {
      sessionId,
      turnCounter: 2,
      limit: 1,
    });
    const warm = second.evidence.find(
      (item) => item.source === "warm_recall" && item.provenance?.episodeId === episode.id,
    );

    expect(getSpy).toHaveBeenCalledWith(episode.id);
    expect(warm?.text).toContain("Updated source narrative from storage.");
    expect(warm?.text).not.toContain("Original source narrative.");
  });

  it("ranks fresh matched episode evidence before warm raw-stream evidence", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const cachedEntry = await harness.streamWriter.append({
      kind: "user_msg",
      content: "Cached raw stream receipt",
    });
    for (let index = 0; index < 3; index += 1) {
      await harness.streamWriter.append({
        kind: "user_msg",
        content: `Later stream tail ${index}`,
      });
    }
    const cachedEpisode = createEpisodeFixture(
      {
        title: "Cache source-backed memory",
        narrative: "Cache has a raw stream receipt.",
        participants: ["Cache"],
        tags: ["Cache"],
        source_stream_ids: [cachedEntry.id],
        significance: 0.1,
        created_at: 1_000,
        updated_at: 1_000,
      },
      [0, 1, 0, 0],
    );
    const freshEpisode = createEpisodeFixture(
      {
        title: "Atlas fresh match",
        narrative: "Atlas should beat carried raw stream context.",
        participants: ["Atlas"],
        tags: ["Atlas"],
        significance: 1,
        created_at: 1_500,
        updated_at: 1_500,
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(cachedEpisode);
    await harness.episodicRepository.insert(freshEpisode);
    await insertDistractors(harness);
    seedRawStreamHandle({
      harness,
      scopeKey: sessionId,
      streamId: cachedEntry.id,
      parentEpisodeId: cachedEpisode.id,
    });

    const second = await harness.retrievalPipeline.searchWithContext("Atlas current", {
      sessionId,
      turnCounter: 2,
      limit: 1,
      minSimilarity: 0.1,
      entityTerms: ["Atlas"],
    });
    const freshIndex = second.evidence.findIndex(
      (item) => item.source === "episode" && item.provenance?.episodeId === freshEpisode.id,
    );
    const warmRawIndex = second.evidence.findIndex(
      (item) =>
        item.source === "warm_recall" && item.provenance?.streamIds?.includes(cachedEntry.id),
    );

    expect(freshIndex).toBeGreaterThanOrEqual(0);
    expect(warmRawIndex).toBeGreaterThanOrEqual(0);
    expect(freshIndex).toBeLessThan(warmRawIndex);
  });

  it("keeps the fresh duplicate evidence item and reinforces the handle", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const episode = createEpisodeFixture({
      title: "Atlas duplicate source",
      narrative: "Atlas can be both warm and freshly retrieved.",
      participants: ["Atlas"],
      tags: ["Atlas"],
      significance: 1,
      created_at: 1_000,
      updated_at: 1_000,
    });
    await harness.episodicRepository.insert(episode);
    await insertDistractors(harness);

    await harness.retrievalPipeline.searchWithContext("Atlas", {
      sessionId,
      turnCounter: 1,
      limit: 1,
      entityTerms: ["Atlas"],
    });

    const second = await harness.retrievalPipeline.searchWithContext("Atlas", {
      sessionId,
      turnCounter: 2,
      limit: 1,
      entityTerms: ["Atlas"],
    });
    const duplicateEvidence = second.evidence.filter(
      (item) => item.provenance?.episodeId === episode.id,
    );
    const state = harness.recallStateRepository.load(sessionId);
    const handle = state?.activeHandles.find((item) => item.handle.source === "episode");
    const key = `episode:${episode.id}`;

    expect(duplicateEvidence).toHaveLength(1);
    expect(duplicateEvidence[0]?.source).toBe("episode");
    expect(handle?.reinforcementCount).toBe(2);
    expect(state?.suppressedHandles[key]).toBeUndefined();
  });

  it("does not renew TTL or reinforcement from rendered warm-only handles", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const episode = createEpisodeFixture(
      {
        title: "Atlas warm-only source",
        narrative: "Atlas warm-only evidence must not refresh itself.",
        participants: ["Atlas"],
        tags: ["Atlas"],
        significance: 0.1,
        created_at: 1_000,
        updated_at: 1_000,
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(episode);
    await insertDistractors(harness);
    seedRecallHandles({
      harness,
      scopeKey: sessionId,
      activeHandles: [
        {
          handle: {
            source: "episode",
            episodeId: episode.id,
          },
          firstSeenTurn: 1,
          lastSeenTurn: 1,
          lastRenderedTurn: 1,
          expiresAfterTurn: 7,
          reinforcementCount: 10,
        },
      ],
    });

    const second = await harness.retrievalPipeline.searchWithContext("unrelated recall", {
      sessionId,
      turnCounter: 2,
      limit: 1,
    });
    const warmEvidence = second.evidence.find(
      (item) => item.source === "warm_recall" && item.provenance?.episodeId === episode.id,
    );
    const key = `episode:${episode.id}`;
    const stateAfterWarmRender = harness.recallStateRepository.load(sessionId);
    const handleAfterWarmRender = stateAfterWarmRender?.activeHandles.find(
      (item) => item.handle.source === "episode" && item.handle.episodeId === episode.id,
    );

    expect(warmEvidence).toBeDefined();
    expect(handleAfterWarmRender?.lastSeenTurn).toBe(1);
    expect(handleAfterWarmRender?.lastRenderedTurn).toBe(2);
    expect(handleAfterWarmRender?.expiresAfterTurn).toBe(7);
    expect(handleAfterWarmRender?.reinforcementCount).toBe(10);
    expect(stateAfterWarmRender?.suppressedHandles[key]).toBe(4);

    const third = await harness.retrievalPipeline.searchWithContext("unrelated recall", {
      sessionId,
      turnCounter: 3,
      limit: 1,
    });

    expect(
      third.evidence.some(
        (item) => item.source === "warm_recall" && item.provenance?.episodeId === episode.id,
      ),
    ).toBe(false);

    const fifth = await harness.retrievalPipeline.searchWithContext("unrelated recall", {
      sessionId,
      turnCounter: 5,
      limit: 1,
    });

    expect(
      fifth.evidence.some(
        (item) => item.source === "warm_recall" && item.provenance?.episodeId === episode.id,
      ),
    ).toBe(true);
  });

  it("preserves reinforced handles under cap pressure from broad fresh evidence", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const reinforcedEpisodeId = createEpisodeId();
    const activeHandles = [
      createStateHandle(
        {
          source: "episode",
          episodeId: reinforcedEpisodeId,
        },
        {
          reinforcementCount: 10,
        },
      ),
      ...Array.from({ length: 23 }, () =>
        createStateHandle({
          source: "episode",
          episodeId: createEpisodeId(),
        }),
      ),
    ];
    seedRecallHandles({
      harness,
      scopeKey: sessionId,
      activeHandles,
      lastRefreshTurn: 1,
      ttlTurns: 100,
    });
    await insertMatchingEpisodes({
      harness,
      count: 30,
      prefix: "Fresh cap pressure",
    });

    await harness.retrievalPipeline.searchWithContext("Atlas current", {
      sessionId,
      turnCounter: 2,
      limit: 15,
      minSimilarity: 0.99,
    });

    const state = harness.recallStateRepository.load(sessionId);

    expect(state?.activeHandles).toHaveLength(24);
    expect(
      state?.activeHandles.some(
        (item) => item.handle.source === "episode" && item.handle.episodeId === reinforcedEpisodeId,
      ),
    ).toBe(true);
  });

  it("bounds brand-new fresh handle admission per turn", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    await insertMatchingEpisodes({
      harness,
      count: 30,
      prefix: "Fresh admission",
    });

    await harness.retrievalPipeline.searchWithContext("Atlas current", {
      sessionId,
      turnCounter: 1,
      limit: 15,
      minSimilarity: 0.99,
    });

    const state = harness.recallStateRepository.load(sessionId);

    expect(state?.activeHandles).toHaveLength(6);
  });

  it("reinforces a fresh duplicate in a full pool without consuming a new admission slot", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const duplicateEpisodeId = createEpisodeId();
    const activeHandles = [
      createStateHandle({
        source: "episode",
        episodeId: duplicateEpisodeId,
      }),
      ...Array.from({ length: 23 }, () =>
        createStateHandle({
          source: "episode",
          episodeId: createEpisodeId(),
        }),
      ),
    ];
    seedRecallHandles({
      harness,
      scopeKey: sessionId,
      activeHandles,
      lastRefreshTurn: 1,
      ttlTurns: 6,
    });
    await insertMatchingEpisodes({
      harness,
      count: 1,
      prefix: "Fresh duplicate",
      ids: [duplicateEpisodeId],
    });
    const newEpisodes = await insertMatchingEpisodes({
      harness,
      count: 10,
      prefix: "Fresh duplicate admission",
    });
    const newEpisodeIds = new Set(newEpisodes.map((episode) => episode.id));

    await harness.retrievalPipeline.searchWithContext("Atlas current", {
      sessionId,
      turnCounter: 2,
      limit: 6,
      minSimilarity: 0.99,
    });

    const state = harness.recallStateRepository.load(sessionId);
    const duplicate = state?.activeHandles.find(
      (item) => item.handle.source === "episode" && item.handle.episodeId === duplicateEpisodeId,
    );
    const admittedNewCount = [...episodeHandleIds(state?.activeHandles ?? [])].filter((episodeId) =>
      newEpisodeIds.has(episodeId),
    ).length;

    expect(state?.activeHandles).toHaveLength(24);
    expect(duplicate?.reinforcementCount).toBe(2);
    expect(duplicate?.expiresAfterTurn).toBe(8);
    expect(admittedNewCount).toBe(6);
  });

  it("prunes expired handles before cap and bounded admission", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const expiredEpisodeIds = Array.from({ length: 10 }, () => createEpisodeId());
    const activeHandles = [
      ...expiredEpisodeIds.map((episodeId) =>
        createStateHandle(
          {
            source: "episode",
            episodeId,
          },
          {
            expiresAfterTurn: 2,
          },
        ),
      ),
      ...Array.from({ length: 14 }, () =>
        createStateHandle({
          source: "episode",
          episodeId: createEpisodeId(),
        }),
      ),
    ];
    seedRecallHandles({
      harness,
      scopeKey: sessionId,
      activeHandles,
      lastRefreshTurn: 2,
      ttlTurns: 6,
    });
    const freshEpisodes = await insertMatchingEpisodes({
      harness,
      count: 8,
      prefix: "Fresh after expiry",
    });
    const freshEpisodeIds = new Set(freshEpisodes.map((episode) => episode.id));

    await harness.retrievalPipeline.searchWithContext("Atlas current", {
      sessionId,
      turnCounter: 3,
      limit: 4,
      minSimilarity: 0.99,
    });

    const state = harness.recallStateRepository.load(sessionId);
    const activeEpisodeIds = episodeHandleIds(state?.activeHandles ?? []);

    expect(state?.activeHandles.length).toBeLessThanOrEqual(24);
    expect(state?.activeHandles).toHaveLength(20);
    for (const episodeId of expiredEpisodeIds) {
      expect(activeEpisodeIds.has(episodeId)).toBe(false);
    }
    expect(
      [...activeEpisodeIds].filter((episodeId) => freshEpisodeIds.has(episodeId)),
    ).toHaveLength(6);
  });

  it("uses source retention rank when otherwise equal handles exceed the cap", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const parentEpisodeId = createEpisodeId();
    seedRecallHandles({
      harness,
      scopeKey: sessionId,
      activeHandles: [
        createStateHandle({
          source: "raw_stream",
          streamIds: [createStreamEntryId()],
        }),
        createStateHandle({
          source: "semantic_node",
          nodeId: createSemanticNodeId(),
        }),
        createStateHandle({
          source: "semantic_edge",
          edgeId: createSemanticEdgeId(),
        }),
        createStateHandle({
          source: "open_question",
          openQuestionId: createOpenQuestionId(),
        }),
        createStateHandle({
          source: "commitment",
          commitmentId: createCommitmentId(),
        }),
        createStateHandle({
          source: "raw_stream",
          streamIds: [createStreamEntryId()],
          parentEpisodeId,
        }),
        createStateHandle({
          source: "episode",
          episodeId: createEpisodeId(),
        }),
      ],
      lastRefreshTurn: 1,
      ttlTurns: 100,
    });
    const pipeline = new RetrievalPipeline({
      embeddingClient: harness.embeddingClient,
      episodicRepository: harness.episodicRepository,
      recallStateRepository: harness.recallStateRepository,
      dataDir: harness.tempDir,
      clock: harness.clock,
      recallStateMaxActiveHandles: 6,
      recallStateMaxNewHandlesPerTurn: 0,
    });

    await pipeline.searchWithContext("quiet turn", {
      sessionId,
      turnCounter: 2,
      limit: 1,
    });

    const state = harness.recallStateRepository.load(sessionId);
    const sources = (state?.activeHandles ?? []).map((item) => {
      if (item.handle.source !== "raw_stream") {
        return item.handle.source;
      }

      return item.handle.parentEpisodeId === undefined ? "raw_stream" : "raw_stream:parent";
    });

    expect(sources).toEqual([
      "episode",
      "raw_stream:parent",
      "commitment",
      "open_question",
      "semantic_edge",
      "semantic_node",
    ]);
    expect(
      state?.activeHandles.some(
        (item) => item.handle.source === "raw_stream" && item.handle.parentEpisodeId === undefined,
      ),
    ).toBe(false);
  });

  it("limits warm evidence rendered per turn by reinforcement then oldest render", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const episodes = await Promise.all(
      Array.from({ length: 5 }, async (_, index) => {
        const episode = createEpisodeFixture(
          {
            title: `Atlas warm candidate ${index}`,
            narrative: `Atlas warm candidate ${index} should be ranked by handle state.`,
            participants: ["Atlas"],
            tags: ["Atlas"],
            significance: 0.1,
            created_at: 1_000 + index,
            updated_at: 1_000 + index,
          },
          [1, 0, 0, 0],
        );
        await harness!.episodicRepository.insert(episode);
        return episode;
      }),
    );
    await insertDistractors(harness);
    seedRecallHandles({
      harness,
      scopeKey: sessionId,
      activeHandles: [
        {
          handle: { source: "episode", episodeId: episodes[0]!.id },
          firstSeenTurn: 1,
          lastSeenTurn: 1,
          lastRenderedTurn: 9,
          expiresAfterTurn: 7,
          reinforcementCount: 5,
        },
        {
          handle: { source: "episode", episodeId: episodes[1]!.id },
          firstSeenTurn: 1,
          lastSeenTurn: 1,
          lastRenderedTurn: 1,
          expiresAfterTurn: 7,
          reinforcementCount: 3,
        },
        {
          handle: { source: "episode", episodeId: episodes[2]!.id },
          firstSeenTurn: 1,
          lastSeenTurn: 1,
          lastRenderedTurn: 2,
          expiresAfterTurn: 7,
          reinforcementCount: 3,
        },
        {
          handle: { source: "episode", episodeId: episodes[3]!.id },
          firstSeenTurn: 1,
          lastSeenTurn: 1,
          lastRenderedTurn: 3,
          expiresAfterTurn: 7,
          reinforcementCount: 3,
        },
        {
          handle: { source: "episode", episodeId: episodes[4]!.id },
          firstSeenTurn: 1,
          lastSeenTurn: 1,
          lastRenderedTurn: 4,
          expiresAfterTurn: 7,
          reinforcementCount: 3,
        },
      ],
    });

    const result = await harness.retrievalPipeline.searchWithContext("unrelated recall", {
      sessionId,
      turnCounter: 2,
      limit: 1,
    });
    const warmEpisodeIds = new Set(
      result.evidence
        .filter((item) => item.source === "warm_recall")
        .map((item) => item.provenance?.episodeId)
        .filter(
          (episodeId): episodeId is (typeof episodes)[number]["id"] => episodeId !== undefined,
        ),
    );

    expect(warmEpisodeIds).toEqual(
      new Set([episodes[0]!.id, episodes[1]!.id, episodes[2]!.id, episodes[3]!.id]),
    );
  });

  it("does not persist recent_raw_stream tail evidence", async () => {
    harness = await createHarness();
    const sessionId = testSessionId();
    const recent = await harness.streamWriter.append({
      kind: "user_msg",
      content: "Recent stream tail only",
    });

    const result = await harness.retrievalPipeline.searchWithContext("nothing relevant", {
      sessionId,
      turnCounter: 1,
      limit: 1,
    });
    const state = harness.recallStateRepository.load(sessionId);

    expect(
      result.evidence.some(
        (item) =>
          item.source === "recent_raw_stream" && item.provenance?.streamIds?.includes(recent.id),
      ),
    ).toBe(true);
    expect(state?.activeHandles.some((item) => item.handle.source === "raw_stream")).toBe(false);
  });

  it("isolates audience recall state and drops rehydration when visibility changes", async () => {
    harness = await createHarness();
    const audienceA = createEntityId();
    const audienceB = createEntityId();
    const sessionId = testSessionId();
    const privateEpisode = createEpisodeFixture({
      title: "Atlas private to A",
      narrative: "Audience A private recall should not warm audience B.",
      participants: ["Atlas"],
      tags: ["Atlas"],
      audience_entity_id: audienceA,
      shared: false,
      significance: 0.1,
      created_at: 1_000,
      updated_at: 1_000,
    });
    await harness.episodicRepository.insert(privateEpisode);
    await insertDistractors(harness);
    seedEpisodeHandle(harness, audienceA, privateEpisode.id);

    const audienceBResult = await harness.retrievalPipeline.searchWithContext("unrelated recall", {
      audienceEntityId: audienceB,
      sessionId,
      turnCounter: 1,
      limit: 1,
    });

    expect(harness.recallStateRepository.load(audienceA)?.activeHandles.length).toBeGreaterThan(0);
    expect(
      audienceBResult.evidence.some((item) => item.provenance?.episodeId === privateEpisode.id),
    ).toBe(false);
    expect(
      harness.recallStateRepository
        .load(audienceB)
        ?.activeHandles.some(
          (item) => item.handle.source === "episode" && item.handle.episodeId === privateEpisode.id,
        ) ?? false,
    ).toBe(false);

    const publicEpisode = createEpisodeFixture(
      {
        title: "Atlas visibility changes",
        narrative: "This public handle will become private before rehydration.",
        participants: ["Atlas"],
        tags: ["Atlas"],
        significance: 0.1,
        created_at: 900,
        updated_at: 900,
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(publicEpisode);
    seedEpisodeHandle(harness, sessionId, publicEpisode.id);
    await harness.episodicRepository.update(publicEpisode.id, {
      audience_entity_id: audienceA,
      shared: false,
    });

    const publicResult = await harness.retrievalPipeline.searchWithContext("quiet turn", {
      sessionId,
      turnCounter: 2,
      limit: 1,
    });

    expect(
      publicResult.evidence.some((item) => item.provenance?.episodeId === publicEpisode.id),
    ).toBe(false);
  });

  it("leaves the working memory schema untouched", () => {
    const workingMemory = createWorkingMemory(DEFAULT_SESSION_ID, NOW_MS);

    expect(Object.keys(workingMemory)).not.toEqual(
      expect.arrayContaining(["recall_state", "activeEvidenceIds", "warmEvidence"]),
    );
    expect(
      workingMemorySchema.safeParse({
        ...workingMemory,
        recall_state: {},
      }).success,
    ).toBe(false);
  });

  it("survives session rotation when an audience entity id is present", async () => {
    harness = await createHarness();
    const audienceA = createEntityId();
    const firstSession = testSessionId("sess_aaaaaaaaaaaaaaaa" as SessionId);
    const secondSession = testSessionId("sess_bbbbbbbbbbbbbbbb" as SessionId);
    const episode = createEpisodeFixture(
      {
        title: "Atlas audience continuity",
        narrative: "Audience-keyed recall should survive session rotation.",
        participants: ["Atlas"],
        tags: ["Atlas"],
        audience_entity_id: audienceA,
        shared: false,
        significance: 0.1,
        created_at: 1_000,
        updated_at: 1_000,
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(episode);
    await insertDistractors(harness);
    seedEpisodeHandle(harness, audienceA, episode.id);

    expect(harness.recallStateRepository.load(firstSession)).toBeNull();

    const second = await harness.retrievalPipeline.searchWithContext("unrelated recall", {
      audienceEntityId: audienceA,
      sessionId: secondSession,
      turnCounter: 1,
      limit: 1,
    });

    expect(
      second.evidence.some(
        (item) => item.source === "warm_recall" && item.provenance?.episodeId === episode.id,
      ),
    ).toBe(true);
    expect(harness.recallStateRepository.load(audienceA)?.lastRefreshTurn).toBe(2);
  });

  it("carries Maya denial evidence across session rotation without storing text", async () => {
    harness = await createHarness();
    const tomAudience = createEntityId();
    const firstSession = testSessionId("sess_aaaaaaaaaaaaaaaa" as SessionId);
    const secondSession = testSessionId("sess_bbbbbbbbbbbbbbbb" as SessionId);
    const mayaTurn = "Maya is my partner. She's making elaborate ramen tonight.";
    const denialTurn = "My partner isn't Maya. I never said that.";
    const episode = createEpisodeFixture(
      {
        title: "Maya ramen source episode",
        narrative: "Tom said Maya is his partner and that she is making elaborate ramen tonight.",
        participants: ["Maya"],
        tags: ["Maya"],
        audience_entity_id: tomAudience,
        shared: false,
        significance: 0.1,
        created_at: 1_000,
        updated_at: 1_000,
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(episode);

    const first = await harness.retrievalPipeline.searchWithContext(mayaTurn, {
      audienceEntityId: tomAudience,
      sessionId: firstSession,
      turnCounter: 1,
      limit: 1,
      minSimilarity: 0.1,
    });

    expect(
      first.evidence.some(
        (item) => item.source === "episode" && item.provenance?.episodeId === episode.id,
      ),
    ).toBe(true);
    expect(harness.recallStateRepository.load(firstSession)).toBeNull();

    await insertDistractors(harness, { prefix: "Denial" });

    const denial = await harness.retrievalPipeline.searchWithContext(denialTurn, {
      audienceEntityId: tomAudience,
      sessionId: secondSession,
      turnCounter: 2,
      limit: 1,
      minSimilarity: 0.1,
    });
    const mayaEvidence = denial.evidence.find(
      (item) =>
        item.provenance?.episodeId === episode.id &&
        (item.source === "episode" ||
          item.source === "raw_stream" ||
          item.source === "warm_recall"),
    );
    const state = harness.recallStateRepository.load(tomAudience);
    const serializedState = loadSerializedState(harness, tomAudience);
    const stringValues = collectStringValues(serializedState);

    expect(mayaEvidence?.provenance?.episodeId).toBe(episode.id);
    expect(
      state?.activeHandles.some(
        (item) => item.handle.source === "episode" && item.handle.episodeId === episode.id,
      ),
    ).toBe(true);
    expect(collectKeys(serializedState)).not.toEqual(
      expect.arrayContaining(["text", "summary", "description", "content", "belief", "claim"]),
    );
    expect(stringValues).not.toEqual(
      expect.arrayContaining([mayaTurn, denialTurn, episode.title, episode.narrative]),
    );
  });
});
