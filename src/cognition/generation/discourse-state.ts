import type {
  ClosureLoopState,
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

export type SetClosureLoopDetectedInput = {
  sourceStreamEntryIds: readonly StreamEntryId[];
  reason: string;
  sinceTurn: number;
};

export type MarkClosureLoopNamedInput = {
  sourceStreamEntryId?: StreamEntryId;
  reason: string;
  turn: number;
};

function baseDiscourseState(workingMemory: WorkingMemory): WorkingMemory["discourse_state"] {
  return workingMemory.discourse_state ?? { stop_until_substantive_content: null };
}

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
      ...baseDiscourseState(workingMemory),
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
      ...baseDiscourseState(workingMemory),
      stop_until_substantive_content: null,
    },
  };
}

export function setClosureLoopDetected(
  workingMemory: WorkingMemory,
  input: SetClosureLoopDetectedInput,
): WorkingMemory {
  const active = workingMemory.discourse_state?.closure_loop ?? null;

  if (active?.status === "named") {
    return workingMemory;
  }

  const state: ClosureLoopState = {
    status: "detected",
    source_stream_entry_ids: [...input.sourceStreamEntryIds],
    reason: input.reason.trim(),
    since_turn: active?.since_turn ?? input.sinceTurn,
    named_at_turn: null,
  };

  return {
    ...workingMemory,
    discourse_state: {
      ...baseDiscourseState(workingMemory),
      closure_loop: state,
    },
  };
}

export function markClosureLoopNamed(
  workingMemory: WorkingMemory,
  input: MarkClosureLoopNamedInput,
): WorkingMemory {
  const active = workingMemory.discourse_state?.closure_loop ?? null;

  if (active === null) {
    return workingMemory;
  }

  const sourceStreamEntryIds =
    input.sourceStreamEntryId === undefined
      ? active.source_stream_entry_ids
      : [...active.source_stream_entry_ids, input.sourceStreamEntryId];
  const state: ClosureLoopState = {
    status: "named",
    source_stream_entry_ids: sourceStreamEntryIds,
    reason: input.reason.trim(),
    since_turn: active.since_turn,
    named_at_turn: input.turn,
  };

  return {
    ...workingMemory,
    discourse_state: {
      ...baseDiscourseState(workingMemory),
      closure_loop: state,
    },
  };
}

export function clearClosureLoop(workingMemory: WorkingMemory): WorkingMemory {
  if ((workingMemory.discourse_state?.closure_loop ?? null) === null) {
    return workingMemory;
  }

  return {
    ...workingMemory,
    discourse_state: {
      ...baseDiscourseState(workingMemory),
      closure_loop: null,
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
