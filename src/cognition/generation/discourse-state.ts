import type {
  ClosurePressureHistoryReason,
  ClosureLoopState,
  DiscourseStopProvenance,
  RecentRegenerationEntry,
  RecentSuppressionEntry,
  StopUntilSubstantiveContent,
  WorkingMemory,
} from "../../memory/working/index.js";
import type { StreamEntryId } from "../../util/ids.js";

export const CLOSURE_PRESSURE_HISTORY_LIMIT = 5;
export const RECENT_SUPPRESSIONS_LIMIT = 10;
export const RECENT_REGENERATIONS_LIMIT = 10;

export type SetStopUntilSubstantiveContentInput = {
  provenance: DiscourseStopProvenance;
  sourceStreamEntryId?: StreamEntryId;
  sourceStreamEntryIds?: readonly StreamEntryId[];
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
  sourceStreamEntryIds?: readonly StreamEntryId[];
  reason: string;
  turn: number;
};

export type AppendClosurePressureHistoryInput = {
  turnId: string;
  reason: ClosurePressureHistoryReason;
  ts: number;
};

export type AppendRecentSuppressionInput = {
  turnId: string;
  reason: string;
  ts: number;
  sourceStreamEntryId?: StreamEntryId;
  sourceStreamEntryIds?: readonly StreamEntryId[];
};

export type AppendRecentRegenerationInput = {
  turnId: string;
  ts: number;
  sourceStreamEntryId?: StreamEntryId;
};

function baseDiscourseState(workingMemory: WorkingMemory): WorkingMemory["discourse_state"] {
  return workingMemory.discourse_state ?? { stop_until_substantive_content: null };
}

function capNewest<T>(values: readonly T[], limit: number): T[] {
  return values.slice(Math.max(0, values.length - limit));
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
    ...(input.sourceStreamEntryIds === undefined || input.sourceStreamEntryIds.length === 0
      ? {}
      : { source_stream_entry_ids: [...input.sourceStreamEntryIds] }),
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

export function appendClosurePressureHistory(
  workingMemory: WorkingMemory,
  input: AppendClosurePressureHistoryInput,
): WorkingMemory {
  const state = baseDiscourseState(workingMemory);
  const next = capNewest(
    [
      ...(state.closure_pressure_history ?? []),
      {
        turn_id: input.turnId,
        turn: workingMemory.turn_counter,
        reason: input.reason,
        ts: input.ts,
      },
    ],
    CLOSURE_PRESSURE_HISTORY_LIMIT,
  );

  return {
    ...workingMemory,
    discourse_state: {
      ...state,
      closure_pressure_history: next,
    },
  };
}

export function appendRecentSuppression(
  workingMemory: WorkingMemory,
  input: AppendRecentSuppressionInput,
): WorkingMemory {
  const state = baseDiscourseState(workingMemory);
  const entry: RecentSuppressionEntry = {
    turn_id: input.turnId,
    reason: input.reason,
    ts: input.ts,
    ...(input.sourceStreamEntryId === undefined
      ? {}
      : { source_stream_entry_id: input.sourceStreamEntryId }),
    ...(input.sourceStreamEntryIds === undefined || input.sourceStreamEntryIds.length === 0
      ? {}
      : { source_stream_entry_ids: [...input.sourceStreamEntryIds] }),
  };
  const next = capNewest([...(state.recent_suppressions ?? []), entry], RECENT_SUPPRESSIONS_LIMIT);

  return {
    ...workingMemory,
    discourse_state: {
      ...state,
      recent_suppressions: next,
    },
  };
}

export function appendRecentRegeneration(
  workingMemory: WorkingMemory,
  input: AppendRecentRegenerationInput,
): WorkingMemory {
  const state = baseDiscourseState(workingMemory);
  const entry: RecentRegenerationEntry = {
    turn_id: input.turnId,
    mechanism: "commitment_guard_regeneration",
    ts: input.ts,
    ...(input.sourceStreamEntryId === undefined
      ? {}
      : { source_stream_entry_id: input.sourceStreamEntryId }),
  };
  const next = capNewest(
    [...(state.recent_regenerations ?? []), entry],
    RECENT_REGENERATIONS_LIMIT,
  );

  return {
    ...workingMemory,
    discourse_state: {
      ...state,
      recent_regenerations: next,
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
    input.sourceStreamEntryIds !== undefined && input.sourceStreamEntryIds.length > 0
      ? [...active.source_stream_entry_ids, ...input.sourceStreamEntryIds]
      : input.sourceStreamEntryId === undefined
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
