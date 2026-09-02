import { describe, expect, it, vi } from "vitest";

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
        topOpenForGoals: () => [],
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
        topOpenForGoals: () => [
          {
            goal_id: goal.id,
            step: nextStep,
            open_step_count: 1,
          },
        ],
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
    expect(context.executiveFocus.candidate_steps?.top_open_steps).toEqual([
      context.executiveFocus.next_step,
    ]);
  });

  it("forces a validated followup goal below threshold over a higher-scoring competitor", async () => {
    const clock = new ManualClock(3_000_000);
    const selectedGoal: GoalRecord = {
      id: createGoalId(),
      description: "Low-priority followup target",
      terminal_condition: "The followup is complete",
      priority: 1,
      parent_goal_id: null,
      status: "active",
      progress_notes: null,
      last_progress_ts: null,
      created_at: clock.now(),
      target_at: null,
      audience_entity_id: null,
      owner_entity_id: null,
      source_stream_entry_ids: [],
      provenance: { kind: "manual" },
    };
    const competitor: GoalRecord = {
      ...selectedGoal,
      id: createGoalId(),
      description: "Higher-scoring competing goal",
      terminal_condition: null,
      priority: 10,
    };
    const builder = new TurnSelfContextBuilder({
      embeddingClient,
      valuesRepository: { list: () => [] },
      goalsRepository: {
        list: () => [
          { ...selectedGoal, children: [] },
          { ...competitor, children: [] },
        ],
      },
      traitsRepository: { list: () => [] },
      executiveStepsRepository: { topOpenForGoals: () => [] },
      clock,
      tracer: NOOP_TRACER,
      goalFocusThreshold: 0.45,
      goalFollowupLookaheadMs: 20_000,
      goalFollowupStaleMs: 100_000,
    });

    const context = await builder.build({
      turnId: "turn_followup_forced_focus",
      cognitionInput: "",
      perception: {
        entities: [],
        mode: "reflective",
        affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
        temporalCue: null,
      },
      autonomyTrigger: {
        source_name: "goal_followup_due",
        source_type: "trigger",
        event_id: "followup-forced-focus",
        sort_ts: clock.now(),
        payload: {
          selected_goal_id: selectedGoal.id,
          selected_goal: selectedGoal,
        },
      },
      audienceEntityId: null,
    });

    expect(context.executiveFocus.candidates[0]?.goal_id).toBe(competitor.id);
    expect(
      context.executiveFocus.candidates.find((candidate) => candidate.goal_id === selectedGoal.id)
        ?.score,
    ).toBeLessThan(context.executiveFocus.threshold);
    expect(context.executiveFocus.selected_goal?.id).toBe(selectedGoal.id);
  });

  it("loads top open steps and exact extra-step counts for expanded candidates in one batch", async () => {
    const clock = new ManualClock(4_000_000);
    const goals = Array.from(
      { length: 5 },
      (_, index): GoalRecord => ({
        id: createGoalId(),
        description: `Candidate goal ${index}`,
        terminal_condition: null,
        priority: 10 - index,
        parent_goal_id: null,
        status: "active",
        progress_notes: null,
        last_progress_ts: null,
        created_at: clock.now() - index,
        target_at: null,
        audience_entity_id: null,
        owner_entity_id: null,
        source_stream_entry_ids: [],
        provenance: { kind: "manual" },
      }),
    );
    const steps = goals.slice(0, 2).map(
      (goal, index): ExecutiveStep => ({
        id: createExecutiveStepId(),
        goal_id: goal.id,
        description: `Top open step ${index}`,
        status: index === 0 ? "doing" : "queued",
        kind: "think",
        due_at: null,
        last_attempt_ts: null,
        created_at: clock.now(),
        updated_at: clock.now(),
        provenance: { kind: "manual" },
      }),
    );
    const topOpenForGoals = vi.fn((goalIds: readonly GoalRecord["id"][]) =>
      steps
        .filter((step) => goalIds.includes(step.goal_id))
        .map((step, index) => ({
          goal_id: step.goal_id,
          step,
          open_step_count: index === 0 ? 3 : 2,
        })),
    );
    const builder = new TurnSelfContextBuilder({
      embeddingClient,
      valuesRepository: { list: () => [] },
      goalsRepository: { list: () => goals.map((goal) => ({ ...goal, children: [] })) },
      traitsRepository: { list: () => [] },
      executiveStepsRepository: { topOpenForGoals },
      clock,
      tracer: NOOP_TRACER,
      goalFocusThreshold: 0,
      goalFollowupLookaheadMs: 0,
      goalFollowupStaleMs: 86_400_000,
    });

    const context = await builder.build({
      turnId: "turn_candidate_step_batch",
      cognitionInput: "advance the leading work",
      perception: {
        entities: [],
        mode: "problem_solving",
        affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
        temporalCue: null,
      },
      audienceEntityId: null,
    });

    expect(topOpenForGoals).toHaveBeenCalledTimes(1);
    expect(topOpenForGoals.mock.calls[0]?.[0]).toEqual(goals.slice(0, 4).map((goal) => goal.id));
    expect(context.executiveFocus.candidate_steps?.top_open_steps.map((step) => step.id)).toEqual(
      steps.map((step) => step.id),
    );
    expect(context.executiveFocus.candidate_steps?.omitted_open_step_count).toBe(3);
    expect(context.executiveFocus.next_step?.id).toBe(steps[0]?.id);
  });
});
