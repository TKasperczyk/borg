import { describe, expect, it } from "vitest";

import { createWorkingMemory, workingMemorySchema } from "../../memory/working/index.js";
import { DEFAULT_SESSION_ID, createStreamEntryId } from "../../util/ids.js";
import { NOOP_TRACER } from "../../tracing/tracer.js";
import {
  clearClosureLoop,
  clearStopUntilSubstantiveContent,
  appendClosurePressureHistory,
  appendRecentRegeneration,
  appendRecentSuppression,
  markClosureLoopNamed,
  reviewStopHardCap,
  setClosureLoopDetected,
  setStopUntilSubstantiveContent,
} from "./discourse-state.js";
import { TurnDiscourseStateService } from "./turn-discourse-state.js";

describe("discourse state", () => {
  it("sets and clears stop-until-substantive-content with provenance", () => {
    const sourceStreamEntryId = createStreamEntryId();
    const workingMemory = createWorkingMemory(DEFAULT_SESSION_ID, 100);
    const stopped = setStopUntilSubstantiveContent(workingMemory, {
      provenance: "finalizer_emission_metadata",
      sourceStreamEntryId,
      reason: "Agent committed to stop responding to minimal inputs.",
      sinceTurn: 12,
    });

    expect(stopped.discourse_state?.stop_until_substantive_content).toEqual({
      provenance: "finalizer_emission_metadata",
      source_stream_entry_id: sourceStreamEntryId,
      reason: "Agent committed to stop responding to minimal inputs.",
      since_turn: 12,
    });
    expect(
      clearStopUntilSubstantiveContent(stopped).discourse_state?.stop_until_substantive_content,
    ).toBeNull();
  });

  it("marks hard-cap review due without clearing the state", () => {
    const workingMemory = setStopUntilSubstantiveContent(
      createWorkingMemory(DEFAULT_SESSION_ID, 100),
      {
        provenance: "generation_gate",
        reason: "Repeated minimal prompts.",
        sinceTurn: 3,
      },
    );

    expect(reviewStopHardCap(workingMemory, 52, 50)).toEqual({
      due: false,
      activeTurns: 49,
    });
    expect(reviewStopHardCap(workingMemory, 53, 50)).toEqual({
      due: true,
      activeTurns: 50,
    });
    expect(workingMemory.discourse_state?.stop_until_substantive_content).not.toBeNull();
  });

  it("tracks closure-loop detection, naming, and clearing", () => {
    const sourceStreamEntryId = createStreamEntryId();
    const detected = setClosureLoopDetected(createWorkingMemory(DEFAULT_SESSION_ID, 100), {
      sourceStreamEntryIds: [sourceStreamEntryId],
      reason: "Two mutual closure cycles detected.",
      sinceTurn: 12,
    });

    expect(detected.discourse_state?.closure_loop).toEqual({
      status: "detected",
      source_stream_entry_ids: [sourceStreamEntryId],
      reason: "Two mutual closure cycles detected.",
      since_turn: 12,
      named_at_turn: null,
    });

    const named = markClosureLoopNamed(detected, {
      sourceStreamEntryId,
      reason: "Named once.",
      turn: 13,
    });

    expect(named.discourse_state?.closure_loop).toMatchObject({
      status: "named",
      reason: "Named once.",
      since_turn: 12,
      named_at_turn: 13,
    });
    expect(clearClosureLoop(named).discourse_state?.closure_loop).toBeNull();
  });

  it("preserves closure pressure history when closure-loop state is cleared", () => {
    const workingMemory = appendClosurePressureHistory(
      setClosureLoopDetected(createWorkingMemory(DEFAULT_SESSION_ID, 100), {
        sourceStreamEntryIds: [createStreamEntryId()],
        reason: "Two mutual closure cycles detected.",
        sinceTurn: 12,
      }),
      {
        turnId: "turn-history",
        reason: "span_removed",
        ts: 1_000,
      },
    );
    const cleared = clearClosureLoop(workingMemory);

    expect(cleared.discourse_state?.closure_loop).toBeNull();
    expect(cleared.discourse_state?.closure_pressure_history).toEqual([
      {
        turn_id: "turn-history",
        turn: 0,
        reason: "span_removed",
        ts: 1_000,
      },
    ]);
  });

  it("caps closure pressure history at the five most recent entries", () => {
    let workingMemory = createWorkingMemory(DEFAULT_SESSION_ID, 100);

    for (let index = 0; index < 7; index += 1) {
      workingMemory = appendClosurePressureHistory(workingMemory, {
        turnId: `turn-${index}`,
        reason: "span_removed",
        ts: index,
      });
    }

    expect(
      workingMemory.discourse_state?.closure_pressure_history?.map((entry) => entry.turn_id),
    ).toEqual(["turn-2", "turn-3", "turn-4", "turn-5", "turn-6"]);
  });

  it("caps recent suppression visibility at the ten most recent entries", () => {
    let workingMemory = createWorkingMemory(DEFAULT_SESSION_ID, 100);

    for (let index = 0; index < 12; index += 1) {
      workingMemory = appendRecentSuppression(workingMemory, {
        turnId: `turn-${index}`,
        reason: "commitment_violation",
        ts: index,
      });
    }

    expect(
      workingMemory.discourse_state?.recent_suppressions?.map((entry) => entry.turn_id),
    ).toEqual([
      "turn-2",
      "turn-3",
      "turn-4",
      "turn-5",
      "turn-6",
      "turn-7",
      "turn-8",
      "turn-9",
      "turn-10",
      "turn-11",
    ]);
  });

  it("appends and caps recent regeneration breadcrumbs without draft content", () => {
    const sourceStreamEntryId = createStreamEntryId();
    let workingMemory = createWorkingMemory(DEFAULT_SESSION_ID, 100);

    for (let index = 0; index < 12; index += 1) {
      workingMemory = appendRecentRegeneration(workingMemory, {
        turnId: `turn-${index}`,
        ts: index,
        sourceStreamEntryId,
      });
    }

    expect(
      workingMemory.discourse_state?.recent_regenerations?.map((entry) => entry.turn_id),
    ).toEqual([
      "turn-2",
      "turn-3",
      "turn-4",
      "turn-5",
      "turn-6",
      "turn-7",
      "turn-8",
      "turn-9",
      "turn-10",
      "turn-11",
    ]);
    expect(workingMemory.discourse_state?.recent_regenerations?.[0]).toEqual({
      turn_id: "turn-2",
      mechanism: "commitment_guard_regeneration",
      ts: 2,
      source_stream_entry_id: sourceStreamEntryId,
    });
    expect(JSON.stringify(workingMemory.discourse_state?.recent_regenerations)).not.toContain(
      "violating",
    );
  });

  it("records the commitments a regeneration was gated on", () => {
    const sourceStreamEntryId = createStreamEntryId();
    const workingMemory = appendRecentRegeneration(createWorkingMemory(DEFAULT_SESSION_ID, 100), {
      turnId: "turn-1",
      ts: 1,
      sourceStreamEntryId,
      commitments: [
        {
          id: "cmt_aaaaaaaaaaaaaaaa",
          kind: "participant_preference",
          critical_domain: "explicit_no_disclosure",
          directive_family: "rollout_privacy",
        },
      ],
    });

    expect(workingMemory.discourse_state?.recent_regenerations?.[0]).toEqual({
      turn_id: "turn-1",
      mechanism: "commitment_guard_regeneration",
      ts: 1,
      source_stream_entry_id: sourceStreamEntryId,
      commitments: [
        {
          id: "cmt_aaaaaaaaaaaaaaaa",
          kind: "participant_preference",
          critical_domain: "explicit_no_disclosure",
          directive_family: "rollout_privacy",
        },
      ],
    });
    expect(workingMemorySchema.parse(workingMemory)).toBeDefined();
  });

  it("keeps a regeneration that named no commitment distinct from one that recorded none", () => {
    const namedNone = appendRecentRegeneration(createWorkingMemory(DEFAULT_SESSION_ID, 100), {
      turnId: "turn-named-none",
      ts: 1,
      commitments: [],
    });
    const unrecorded = appendRecentRegeneration(createWorkingMemory(DEFAULT_SESSION_ID, 100), {
      turnId: "turn-unrecorded",
      ts: 1,
    });

    expect(namedNone.discourse_state?.recent_regenerations?.[0]).toEqual({
      turn_id: "turn-named-none",
      mechanism: "commitment_guard_regeneration",
      ts: 1,
      commitments: [],
    });
    expect(unrecorded.discourse_state?.recent_regenerations?.[0]).not.toHaveProperty("commitments");
    expect(workingMemorySchema.parse(namedNone)).toBeDefined();
    expect(workingMemorySchema.parse(unrecorded)).toBeDefined();
  });

  it("marks a detected closure loop named after S2 planner no-output", () => {
    const sourceStreamEntryId = createStreamEntryId();
    const suppressionStreamEntryId = createStreamEntryId();
    const workingMemory = {
      ...createWorkingMemory(DEFAULT_SESSION_ID, 100),
      turn_counter: 14,
    };
    const detected = setClosureLoopDetected(workingMemory, {
      sourceStreamEntryIds: [sourceStreamEntryId],
      reason: "Two mutual closure cycles detected.",
      sinceTurn: 13,
    });
    const service = new TurnDiscourseStateService({
      tracer: NOOP_TRACER,
      clock: { now: () => 2_000 },
    });

    const named = service.applySuppressedEmissionState({
      workingMemory: detected,
      reason: "s2_planner_no_output",
      sourceStreamEntryId: suppressionStreamEntryId,
      turnId: "turn-s2-no-output",
    });

    expect(named.discourse_state?.closure_loop).toMatchObject({
      status: "named",
      source_stream_entry_ids: [sourceStreamEntryId, suppressionStreamEntryId],
      named_at_turn: 14,
    });
    expect(named.discourse_state?.recent_suppressions).toEqual([
      {
        turn_id: "turn-s2-no-output",
        reason: "s2_planner_no_output",
        source_stream_entry_id: suppressionStreamEntryId,
        ts: 2_000,
      },
    ]);
  });
});
