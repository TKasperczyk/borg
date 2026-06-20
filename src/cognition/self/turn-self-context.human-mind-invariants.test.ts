import { describe, expect, it } from "vitest";

import type { ExecutiveStep } from "../../executive/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import type { GoalRecord, ValueRecord } from "../../memory/self/index.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import { createEpisodeFixture } from "../../offline/test-support.js";
import { relationshipPrivateMemoryDisclosureLabel } from "../../retrieval/index.js";
import { ManualClock } from "../../util/clock.js";
import {
  createEntityId,
  createExecutiveStepId,
  createGoalId,
  createStreamEntryId,
  createValueId,
  DEFAULT_SESSION_ID,
} from "../../util/ids.js";
import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import { buildBaseSystemPrompt } from "../deliberation/prompt/system-prompt.js";
import { NOOP_TRACER } from "../../tracing/tracer.js";
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
  it("surfaces self-memory evidence regardless of current audience with disclosure labels", async () => {
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
      "I can use this internally; I do not disclose it to the current audience unless authorized",
    );
  });

  it("threads executive-focus source disclosure into selected self goals and next steps", async () => {
    const aliceId = createEntityId();
    const clock = new ManualClock(2_000_000);
    const goal: GoalRecord = {
      id: createGoalId(),
      description: "Follow up on the Alice-private source",
      terminal_condition: null,
      priority: 10,
      parent_goal_id: null,
      status: "active",
      progress_notes: null,
      last_progress_ts: null,
      created_at: clock.now() - 90_000,
      target_at: null,
      audience_entity_id: null,
      owner_entity_id: null,
      source_stream_entry_ids: [createStreamEntryId()],
      provenance: { kind: "manual" },
    };
    const nextStep: ExecutiveStep = {
      id: createExecutiveStepId(),
      goal_id: goal.id,
      description: "Use the Alice-private source carefully",
      status: "queued",
      kind: "think",
      due_at: null,
      last_attempt_ts: null,
      created_at: clock.now(),
      updated_at: clock.now(),
      provenance: { kind: "manual" },
    };
    const disclosureFields = memoryDisclosurePayloadFields(
      relationshipPrivateMemoryDisclosureLabel([aliceId]),
    );
    const builder = new TurnSelfContextBuilder({
      embeddingClient,
      valuesRepository: {
        list: () => [],
      },
      goalsRepository: {
        list: () => [{ ...goal, children: [] }],
      },
      traitsRepository: {
        list: () => [],
      },
      executiveStepsRepository: {
        topOpen: () => nextStep,
      },
      clock,
      tracer: NOOP_TRACER,
      goalFocusThreshold: 0,
      goalFollowupLookaheadMs: 0,
      goalFollowupStaleMs: 86_400_000,
    });

    const context = await builder.build({
      turnId: "turn_self_context_disclosure",
      cognitionInput: "autonomous wake",
      perception: {
        entities: [],
        mode: "reflective",
        affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
        temporalCue: null,
      },
      autonomyTrigger: {
        source_name: "executive_focus_due",
        source_type: "trigger",
        event_id: "event_private_source",
        sort_ts: clock.now(),
        payload: {
          reason: "goal_stale",
          selected_goal_id: goal.id,
          selected_goal: {
            goal_id: goal.id,
            description: goal.description,
            ...disclosureFields,
          },
        },
      },
      audienceEntityId: null,
    });

    expect(context.selfSnapshot.goals[0]).toMatchObject({
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [aliceId],
      },
    });
    expect(context.executiveFocus.selected_goal).toMatchObject({
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [aliceId],
      },
    });
    expect(context.executiveFocus.next_step).toMatchObject({
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [aliceId],
      },
    });
  });
});
