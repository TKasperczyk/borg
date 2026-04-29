import type {
  DiscourseStopProvenance,
  StopUntilSubstantiveContent,
  WorkingMemory,
} from "../../memory/working/index.js";
import type { StreamEntryId } from "../../util/ids.js";

export type SetStopUntilSubstantiveContentInput = {
  provenance: DiscourseStopProvenance;
  sourceStreamEntryId?: StreamEntryId;
  reason: string;
  sinceTurn: number;
};

export type StopHardCapReview = {
  due: boolean;
  activeTurns: number;
};

export function setStopUntilSubstantiveContent(
  workingMemory: WorkingMemory,
  input: SetStopUntilSubstantiveContentInput,
): WorkingMemory {
  const state: StopUntilSubstantiveContent = {
    provenance: input.provenance,
    reason: input.reason.trim(),
    since_turn: input.sinceTurn,
    ...(input.sourceStreamEntryId === undefined
      ? {}
      : { source_stream_entry_id: input.sourceStreamEntryId }),
  };

  return {
    ...workingMemory,
    discourse_state: {
      ...(workingMemory.discourse_state ?? { stop_until_substantive_content: null }),
      stop_until_substantive_content: state,
    },
  };
}

export function clearStopUntilSubstantiveContent(workingMemory: WorkingMemory): WorkingMemory {
  if ((workingMemory.discourse_state?.stop_until_substantive_content ?? null) === null) {
    return workingMemory;
  }

  return {
    ...workingMemory,
    discourse_state: {
      ...(workingMemory.discourse_state ?? { stop_until_substantive_content: null }),
      stop_until_substantive_content: null,
    },
  };
}

export function reviewStopHardCap(
  workingMemory: WorkingMemory,
  currentTurn: number,
  hardCapTurns: number,
): StopHardCapReview {
  const active = workingMemory.discourse_state?.stop_until_substantive_content ?? null;

  if (active === null) {
    return {
      due: false,
      activeTurns: 0,
    };
  }

  const activeTurns = Math.max(0, currentTurn - active.since_turn);

  return {
    due: activeTurns >= hardCapTurns,
    activeTurns,
  };
}
