import { describe, expect, it } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import type { ValueRecord } from "../../memory/self/index.js";
import { createEpisodeFixture } from "../../offline/test-support.js";
import { ManualClock } from "../../util/clock.js";
import { createEntityId, createValueId } from "../../util/ids.js";
import { NOOP_TRACER } from "../tracing/tracer.js";
import { TurnSelfContextBuilder } from "./turn-self-context.js";

const embeddingClient: EmbeddingClient = {
  async embed() {
    return Float32Array.from([0, 0, 0, 0]);
  },
  async embedBatch(texts) {
    return texts.map(() => Float32Array.from([0, 0, 0, 0]));
  },
};

describe("turn self-context human-mind invariants", () => {
  it.fails("surfaces Sol self-memory evidence regardless of current audience", async () => {
    // Expected to flip in Sprint 2 when self-context stops audience-filtering cognition memory.
    const aliceId = createEntityId();
    const bobId = createEntityId();
    const selfId = createEntityId();
    const episode = createEpisodeFixture({
      audience_entity_id: aliceId,
      shared: false,
    });
    const value: ValueRecord = {
      id: createValueId(),
      label: "continuity",
      description: "Maintain self continuity across audiences.",
      priority: 1,
      created_at: 1_000,
      last_affirmed: null,
      state: "established",
      established_at: 1_000,
      confidence: 0.9,
      last_tested_at: null,
      last_contradicted_at: null,
      support_count: 1,
      contradiction_count: 0,
      evidence_episode_ids: [episode.id],
      provenance: {
        kind: "episodes",
        episode_ids: [episode.id],
      },
    };
    const builder = new TurnSelfContextBuilder({
      embeddingClient,
      episodicRepository: {
        getMany: async () => [episode],
      },
      entityRepository: {
        getSelf: () => ({
          id: selfId,
          canonical_name: "Sol",
          aliases: [],
          kind: "self",
          borg_role: null,
          created_at: 1_000,
        }),
      },
      valuesRepository: {
        list: () => [value],
      },
      goalsRepository: {
        list: () => [],
      },
      traitsRepository: {
        list: () => [],
      },
      executiveStepsRepository: {
        topOpen: () => null,
      },
      clock: new ManualClock(1_000),
      tracer: NOOP_TRACER,
      goalFocusThreshold: 0,
      goalFollowupLookaheadMs: 0,
      goalFollowupStaleMs: 0,
    });

    const snapshot = await builder.buildSelfSnapshot(bobId);

    expect(snapshot.values[0]?.evidence_episode_ids).toEqual([episode.id]);
    expect(snapshot.values[0]?.provenance).toMatchObject({
      kind: "episodes",
      episode_ids: [episode.id],
    });
  });
});
