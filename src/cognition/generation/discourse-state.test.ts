import { describe, expect, it } from "vitest";

import { createWorkingMemory } from "../../memory/working/index.js";
import { DEFAULT_SESSION_ID, createStreamEntryId } from "../../util/ids.js";
import {
  clearStopUntilSubstantiveContent,
  reviewStopHardCap,
  setStopUntilSubstantiveContent,
} from "./discourse-state.js";

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
});
