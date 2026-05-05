import { describe, expect, it } from "vitest";

import { createWorkingMemory } from "../../memory/working/index.js";
import { DEFAULT_SESSION_ID, createStreamEntryId } from "../../util/ids.js";
import { NOOP_TRACER } from "../tracing/tracer.js";
import {
  clearClosureLoop,
  clearStopUntilSubstantiveContent,
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
      provenance: "self_commitment_extractor",
      sourceStreamEntryId,
      reason: "Agent committed to stop responding to minimal inputs.",
      sinceTurn: 12,
    });

    expect(stopped.discourse_state?.stop_until_substantive_content).toEqual({
      provenance: "self_commitment_extractor",
      source_stream_entry_id: sourceStreamEntryId,
      reason: "Agent committed to stop responding to minimal inputs.",
      since_turn: 12,
    });
    expect(clearStopUntilSubstantiveContent(stopped).discourse_state).toEqual({
      stop_until_substantive_content: null,
    });
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
  });
});
