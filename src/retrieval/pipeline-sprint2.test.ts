import { afterEach, describe, expect, it, vi } from "vitest";

import type { EpisodeSearchCandidate } from "../memory/episodic/types.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  type OfflineTestHarness,
} from "../offline/test-support.js";
import { FixedClock } from "../util/clock.js";

const QUERY = "architecture";
const NOW_MS = 10_000_000_000;

function defaultWeights() {
  return {
    semantic: 0.2,
    goal_relevance: 0,
    mood: 0,
    time: 0,
    social: 0,
    entity: 0.2,
    heat: 0.7,
    suppression_penalty: 0.5,
  };
}

function searchWeights(overrides: Partial<Record<keyof ReturnType<typeof defaultWeights>, number>> = {}) {
  return {
    ...defaultWeights(),
    ...overrides,
  };
}

async function createHarness(): Promise<OfflineTestHarness> {
  return createOfflineTestHarness({
    clock: new FixedClock(NOW_MS),
  });
}

function markHot(harness: OfflineTestHarness, episodeId: string): void {
  harness.episodicRepository.updateStats(episodeId as never, {
    retrieval_count: 12,
    win_rate: 0.9,
    last_retrieved: harness.clock.now() - 1_000,
  });
}

async function insertHotVectorDecoys(
  harness: OfflineTestHarness,
  count: number,
  overrides: Partial<Parameters<typeof createEpisodeFixture>[0]> = {},
): Promise<void> {
  for (let index = 0; index < count; index += 1) {
    const updatedAt = overrides.updated_at ?? NOW_MS - index;
    const createdAt = overrides.created_at ?? updatedAt;
    const episode = createEpisodeFixture(
      {
        title: `Vector decoy ${index}`,
        narrative: `A strong semantic architecture match ${index}.`,
        tags: ["decoy"],
        participants: ["team"],
        significance: 0.3,
        created_at: createdAt,
        updated_at: updatedAt,
        start_time: overrides.start_time ?? 900_000 + index,
        end_time: overrides.end_time ?? 901_000 + index,
        ...overrides,
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(episode);
    markHot(harness, episode.id);
  }
}

async function insertColdVectorDecoys(
  harness: OfflineTestHarness,
  count: number,
  overrides: Partial<Parameters<typeof createEpisodeFixture>[0]> = {},
): Promise<void> {
  for (let index = 0; index < count; index += 1) {
    const updatedAt = overrides.updated_at ?? 1_000 + index;
    const createdAt = overrides.created_at ?? updatedAt;
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: `Cold vector decoy ${index}`,
          narrative: `A strong semantic architecture match ${index}.`,
          tags: ["decoy"],
          participants: ["team"],
          significance: 0.2,
          created_at: createdAt,
          updated_at: updatedAt,
          start_time: overrides.start_time ?? 900_000 + index,
          end_time: overrides.end_time ?? 901_000 + index,
          ...overrides,
        },
        [1, 0, 0, 0],
      ),
    );
  }
}

describe("RetrievalPipeline Sprint 2 multi-candidate retrieval", () => {
  let harness: OfflineTestHarness | undefined;

  afterEach(async () => {
    await harness?.cleanup();
    harness = undefined;
  });

  it("rescues old cold episodes from an explicit time window and scores time relevance from timeRange", async () => {
    harness = await createHarness();
    await insertHotVectorDecoys(harness, 80, {
      start_time: 900_000,
      end_time: 901_000,
    });

    const rescued = createEpisodeFixture(
      {
        title: "Yesterday's architecture review",
        tags: ["temporal"],
        significance: 1,
        created_at: 10,
        updated_at: 10,
        start_time: 150_000,
        end_time: 160_000,
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.insert(rescued);

    const queryVector = await harness.embeddingClient.embed(QUERY);
    const timeRange = {
      start: 140_000,
      end: 170_000,
    };
    const vectorOnly = await harness.episodicRepository.searchByVector(queryVector, {
      limit: 12,
      timeRange,
    });
    const results = await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      timeRange,
      attentionWeights: searchWeights({
        semantic: 0.05,
        time: 0.85,
        heat: 0.02,
        entity: 0,
      }),
    });

    expect(vectorOnly.map((item) => item.episode.id)).not.toContain(rescued.id);
    expect(results[0]?.episode.id).toBe(rescued.id);
    expect(results[0]?.scoreBreakdown.timeRelevance).toBe(1);
  });

  it("rescues old cold audience-scoped episodes against hot public decoys", async () => {
    harness = await createHarness();
    const sam = harness.entityRepository.resolve("Sam");
    harness.socialRepository.upsertProfile(sam);
    harness.socialRepository.adjustTrust(sam, 0.3, { kind: "manual" });
    await insertHotVectorDecoys(harness, 80);

    const rescued = createEpisodeFixture(
      {
        title: "Sam-only architecture decision",
        participants: ["Sam"],
        audience_entity_id: sam,
        shared: false,
        significance: 1,
        created_at: 10,
        updated_at: 10,
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.insert(rescued);

    const queryVector = await harness.embeddingClient.embed(QUERY);
    const vectorOnly = await harness.episodicRepository.searchByVector(queryVector, {
      limit: 12,
      audienceEntityId: sam,
    });
    const results = await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      audienceEntityId: sam,
      audienceProfile: harness.socialRepository.getProfile(sam),
      audienceTerms: ["Sam"],
      attentionWeights: searchWeights({
        semantic: 0.05,
        social: 2.2,
        heat: 0.02,
        entity: 0,
      }),
    });

    expect(vectorOnly.map((item) => item.episode.id)).not.toContain(rescued.id);
    expect(results[0]?.episode.id).toBe(rescued.id);
    expect(results[0]?.scoreBreakdown.socialRelevance).toBeGreaterThan(0);
  });

  it("rescues old cold entity matches over hot recent semantic decoys", async () => {
    harness = await createHarness();
    await insertHotVectorDecoys(harness, 80, {
      tags: ["decoy"],
      participants: ["team"],
    });

    const rescued = createEpisodeFixture(
      {
        title: "Atlas architecture note",
        tags: ["Atlas"],
        significance: 1,
        created_at: 10,
        updated_at: 10,
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.insert(rescued);

    const queryVector = await harness.embeddingClient.embed(QUERY);
    const vectorOnly = await harness.episodicRepository.searchByVector(queryVector, {
      limit: 12,
    });
    const results = await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      entityTerms: ["atlas"],
      attentionWeights: searchWeights({
        semantic: 0.05,
        entity: 1.5,
        heat: 0.02,
      }),
    });

    expect(vectorOnly.map((item) => item.episode.id)).not.toContain(rescued.id);
    expect(results[0]?.episode.id).toBe(rescued.id);
    expect(results[0]?.scoreBreakdown.entityRelevance).toBe(1);
  });

  it("rescues hot or recent episodes even without temporal or entity cues", async () => {
    harness = await createHarness();
    await insertColdVectorDecoys(harness, 80, {
      updated_at: 1_000,
      significance: 0.2,
    });

    const rescued = createEpisodeFixture(
      {
        title: "Critical architecture escalation",
        tags: ["urgent"],
        significance: 1,
        created_at: NOW_MS,
        updated_at: NOW_MS,
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.insert(rescued);
    markHot(harness, rescued.id);

    const queryVector = await harness.embeddingClient.embed(QUERY);
    const vectorOnly = await harness.episodicRepository.searchByVector(queryVector, {
      limit: 12,
    });
    const results = await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      attentionWeights: searchWeights({
        semantic: 0.15,
        heat: 0.9,
        entity: 0,
      }),
    });

    expect(vectorOnly.map((item) => item.episode.id)).not.toContain(rescued.id);
    expect(results[0]?.episode.id).toBe(rescued.id);
  });

  it("merges vector and temporal hits for the same episode while preserving vector similarity", async () => {
    harness = await createHarness();
    const shared = createEpisodeFixture(
      {
        title: "Atlas architecture context",
        tags: ["Atlas"],
        significance: 1,
        created_at: 10,
        updated_at: 10,
        start_time: 150_000,
        end_time: 160_000,
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.insert(shared);

    const queryVector = await harness.embeddingClient.embed(QUERY);
    const vectorCandidate = (await harness.episodicRepository.searchByVector(queryVector, {
      limit: 5,
    })).find((item) => item.episode.id === shared.id);
    const temporalCandidate = (
      await harness.episodicRepository.searchByTimeRange(
        {
          start: 140_000,
          end: 170_000,
        },
        {
          limit: 5,
        },
      )
    ).find((item) => item.episode.id === shared.id);

    expect(vectorCandidate).toBeDefined();
    expect(temporalCandidate).toBeDefined();

    const merged = (
      harness.retrievalPipeline as unknown as {
        mergeCandidates(
          candidateSets: ReadonlyArray<
            Array<{
              candidate: EpisodeSearchCandidate;
              sources: Set<"vector" | "temporal">;
            }>
          >,
        ): Array<{
          candidate: EpisodeSearchCandidate;
          sources: Set<"vector" | "temporal">;
        }>;
      }
    ).mergeCandidates([
      [
        {
          candidate: vectorCandidate!,
          sources: new Set(["vector"]),
        },
      ],
      [
        {
          candidate: temporalCandidate!,
          sources: new Set(["temporal"]),
        },
      ],
    ]);

    expect(merged).toHaveLength(1);
    expect(merged[0]?.sources).toEqual(new Set(["vector", "temporal"]));
    expect(merged[0]?.candidate.similarity ?? 0).toBeGreaterThan(0);

    const results = await harness.retrievalPipeline.search(QUERY, {
      limit: 5,
      timeRange: { start: 140_000, end: 170_000 },
      attentionWeights: searchWeights({ semantic: 0.7, time: 0.2, heat: 0 }),
    });

    expect(results.filter((item) => item.episode.id === shared.id)).toHaveLength(1);
  });

  it("hard-excludes out-of-range episodes when strictTimeRange is enabled", async () => {
    harness = await createHarness();
    await insertHotVectorDecoys(harness, 40, {
      start_time: 900_000,
      end_time: 901_000,
    });

    const inRange = createEpisodeFixture(
      {
        title: "Windowed architecture incident",
        significance: 1,
        created_at: 10,
        updated_at: 10,
        start_time: 150_000,
        end_time: 160_000,
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.insert(inRange);

    const results = await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      timeRange: {
        start: 140_000,
        end: 170_000,
      },
      strictTimeRange: true,
      attentionWeights: searchWeights({
        semantic: 0.05,
        time: 0.85,
        heat: 0.02,
        entity: 0,
      }),
    });

    expect(results).toHaveLength(1);
    expect(results[0]?.episode.id).toBe(inRange.id);
    expect(
      results.every(
        (item) => item.episode.start_time <= 170_000 && item.episode.end_time >= 140_000,
      ),
    ).toBe(true);
  });

  it("hard-filters cue-only searches when strictTimeRange is enabled", async () => {
    harness = await createHarness();
    await insertHotVectorDecoys(harness, 12, {
      start_time: 900_000,
      end_time: 901_000,
    });

    const inRange = createEpisodeFixture(
      {
        title: "Yesterday architecture review",
        significance: 1,
        created_at: 10,
        updated_at: 10,
        start_time: 150_000,
        end_time: 160_000,
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.insert(inRange);

    const results = await harness.retrievalPipeline.search(QUERY, {
      limit: 10,
      temporalCue: {
        phrase: "yesterday",
        granularity: "day",
        sinceTs: 140_000,
        untilTs: 170_000,
      },
      strictTimeRange: true,
      attentionWeights: searchWeights({
        semantic: 0.05,
        time: 0.85,
        heat: 0.02,
        entity: 0,
      }),
    });

    expect(results).toHaveLength(1);
    expect(results[0]?.episode.id).toBe(inRange.id);
  });

  it("keeps cue-only searches as scoring boosts when strictTimeRange is disabled", async () => {
    harness = await createHarness();
    await insertHotVectorDecoys(harness, 12, {
      start_time: 900_000,
      end_time: 901_000,
    });

    const inRange = createEpisodeFixture(
      {
        title: "Yesterday architecture review",
        significance: 1,
        created_at: 10,
        updated_at: 10,
        start_time: 150_000,
        end_time: 160_000,
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.insert(inRange);

    const results = await harness.retrievalPipeline.search(QUERY, {
      limit: 10,
      temporalCue: {
        phrase: "yesterday",
        granularity: "day",
        sinceTs: 140_000,
        untilTs: 170_000,
      },
      strictTimeRange: false,
      attentionWeights: searchWeights({
        semantic: 0.4,
        time: 0.15,
        heat: 0.7,
        entity: 0,
      }),
    });

    expect(results).toHaveLength(10);
    expect(results.some((item) => item.episode.start_time > 170_000)).toBe(true);
    expect(
      results.every(
        (item) => item.episode.start_time <= 170_000 && item.episode.end_time >= 140_000,
      ),
    ).toBe(false);
  });

  it("prefers explicit timeRange over a temporal cue when strict filtering", async () => {
    harness = await createHarness();
    const cueOnly = createEpisodeFixture(
      {
        title: "Cue window architecture note",
        significance: 1,
        created_at: 10,
        updated_at: 10,
        start_time: 150_000,
        end_time: 160_000,
      },
      [0, 1, 0, 0],
    );
    const explicit = createEpisodeFixture(
      {
        title: "Explicit window architecture note",
        significance: 1,
        created_at: 10,
        updated_at: 10,
        start_time: 320_000,
        end_time: 330_000,
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.insert(cueOnly);
    await harness.episodicRepository.insert(explicit);

    const results = await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      timeRange: {
        start: 300_000,
        end: 340_000,
      },
      temporalCue: {
        phrase: "yesterday",
        granularity: "day",
        sinceTs: 140_000,
        untilTs: 170_000,
      },
      strictTimeRange: true,
      attentionWeights: searchWeights({
        semantic: 0.05,
        time: 0.85,
        heat: 0.02,
        entity: 0,
      }),
    });

    expect(results).toHaveLength(1);
    expect(results[0]?.episode.id).toBe(explicit.id);
  });

  it("returns empty results instead of widening when a strict cue range matches nothing", async () => {
    harness = await createHarness();
    await insertHotVectorDecoys(harness, 12, {
      start_time: 900_000,
      end_time: 901_000,
    });
    const results = await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      temporalCue: {
        phrase: "yesterday",
        granularity: "day",
        sinceTs: 140_000,
        untilTs: 170_000,
      },
      strictTimeRange: true,
      attentionWeights: searchWeights({
        semantic: 0.05,
        time: 0.85,
        heat: 0.02,
        entity: 0,
      }),
    });

    expect(results).toEqual([]);
  });

  it("reuses a single visible-corpus scan across scan-based generators", async () => {
    harness = await createHarness();
    const sam = harness.entityRepository.resolve("Sam");
    const scanned = createEpisodeFixture({
      title: "Sam Atlas note",
      participants: ["Sam"],
      tags: ["Atlas"],
      audience_entity_id: sam,
      shared: false,
      created_at: NOW_MS,
      updated_at: NOW_MS,
    });
    await harness.episodicRepository.insert(scanned);
    markHot(harness, scanned.id);

    const visibleSpy = vi.spyOn(harness.episodicRepository, "listVisibleEpisodes");

    await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      audienceEntityId: sam,
      audienceTerms: ["Sam"],
      audienceProfile: harness.socialRepository.upsertProfile(sam),
      entityTerms: ["atlas"],
      attentionWeights: searchWeights({
        semantic: 0.1,
        social: 0.4,
        entity: 0.4,
        heat: 0.2,
      }),
    });

    expect(visibleSpy).toHaveBeenCalledTimes(1);
  });

  it("keeps every non-vector generator audience-safe", async () => {
    harness = await createHarness();
    const sam = harness.entityRepository.resolve("Sam");
    const alex = harness.entityRepository.resolve("Alex");
    const hidden = createEpisodeFixture(
      {
        title: "Sam-only Atlas incident",
        participants: ["Sam"],
        tags: ["Atlas"],
        audience_entity_id: sam,
        shared: false,
        significance: 1,
        created_at: NOW_MS,
        updated_at: NOW_MS,
        start_time: 150_000,
        end_time: 160_000,
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.insert(hidden);
    markHot(harness, hidden.id);

    const temporal = await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      audienceEntityId: alex,
      timeRange: { start: 140_000, end: 170_000 },
      attentionWeights: searchWeights({ time: 0.7, heat: 0.05 }),
    });
    const audience = await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      audienceEntityId: alex,
      audienceTerms: ["Alex"],
      attentionWeights: searchWeights({ social: 1.5, heat: 0.05 }),
    });
    const entity = await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      audienceEntityId: alex,
      entityTerms: ["atlas"],
      attentionWeights: searchWeights({ entity: 1.2, heat: 0.05 }),
    });
    const recentHeat = await harness.retrievalPipeline.search(QUERY, {
      limit: 3,
      audienceEntityId: alex,
      attentionWeights: searchWeights({ heat: 0.9 }),
    });

    expect(temporal.map((item) => item.episode.id)).not.toContain(hidden.id);
    expect(audience.map((item) => item.episode.id)).not.toContain(hidden.id);
    expect(entity.map((item) => item.episode.id)).not.toContain(hidden.id);
    expect(recentHeat.map((item) => item.episode.id)).not.toContain(hidden.id);
  });

  it("lets MMR select a zero-similarity rescued candidate with a valid embedding", async () => {
    harness = await createHarness();
    const primary = createEpisodeFixture(
      {
        title: "Primary architecture note",
        significance: 0.8,
      },
      [1, 0, 0, 0],
    );
    const duplicate = createEpisodeFixture(
      {
        title: "Near duplicate architecture note",
        significance: 0.75,
      },
      [0.99, 0.01, 0, 0],
    );
    const rescued = createEpisodeFixture(
      {
        title: "Hot orthogonal incident",
        significance: 1,
        created_at: NOW_MS,
        updated_at: NOW_MS,
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.insert(primary);
    await harness.episodicRepository.insert(duplicate);
    await harness.episodicRepository.insert(rescued);
    markHot(harness, rescued.id);

    const results = await harness.retrievalPipeline.search(QUERY, {
      limit: 2,
      mmrLambda: 0.4,
      attentionWeights: searchWeights({ semantic: 0.35, heat: 0.55, entity: 0 }),
    });

    expect(results.map((item) => item.episode.id)).toContain(rescued.id);
    expect(results.find((item) => item.episode.id === rescued.id)?.scoreBreakdown.similarity).toBe(0);
  });
});
