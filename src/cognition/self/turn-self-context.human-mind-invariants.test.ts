import { describe, expect, it } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import type { ValueRecord } from "../../memory/self/index.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import { createEpisodeFixture } from "../../offline/test-support.js";
import { ManualClock } from "../../util/clock.js";
import { createEntityId, createValueId, DEFAULT_SESSION_ID } from "../../util/ids.js";
import { buildBaseSystemPrompt } from "../deliberation/prompt/system-prompt.js";
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
  it("surfaces Sol self-memory evidence regardless of current audience with disclosure labels", async () => {
    const aliceId = createEntityId();
    const bobId = createEntityId();
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

    const prompt = buildBaseSystemPrompt(
      {
        sessionId: DEFAULT_SESSION_ID,
        userMessage: "What do you remember about yourself?",
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
          temporalCue: null,
        },
        retrievalResult: [],
        workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
        selfSnapshot: snapshot,
      },
      {
        retrievalContextBudget: 1_000,
        semanticContextBudget: 1_000,
      },
    );

    expect(prompt).toContain("continuity");
    expect(prompt).toContain("disclosure_class=self_private");
    expect(prompt).toContain(
      "usable internally; do not disclose to current audience unless authorized",
    );
  });
});
