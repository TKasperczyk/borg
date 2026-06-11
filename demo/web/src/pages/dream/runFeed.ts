import type {
  DreamProcessCompletedFrame,
  DreamProcessStartedFrame,
  LiveFrame,
  MaintenanceTickFrame,
  OfflineProcessName,
} from "../../api/types";

export type DreamRunFeedEntry =
  | {
      id: string;
      kind: "request";
      ts: number;
      action: "plan" | "apply";
      processes: OfflineProcessName[];
      plan_id?: string;
    }
  | {
      id: string;
      kind: "started";
      ts: number;
      phase: "plan" | "apply";
      process: string;
      run_id: string | null;
    }
  | {
      id: string;
      kind: "completed";
      ts: number;
      phase: "plan" | "apply";
      process: string;
      run_id: string | null;
      duration_ms?: number;
      errors: number;
      candidates_accepted: number;
    }
  | {
      id: string;
      kind: "tick";
      ts: number;
      cadence: MaintenanceTickFrame["cadence"];
      status: MaintenanceTickFrame["status"];
      run_id?: string | null;
      changes: number;
      errors: number;
      duration_ms?: number;
    };

export type DreamRunSummary = {
  key: string;
  phase: "plan" | "apply";
  run_id: string | null;
  changes: number;
  errors: number;
  duration_ms: number | null;
};

type ActiveDreamRun = {
  key: string;
  phase: "plan" | "apply";
  run_id: string | null;
  pending: Record<string, number>;
};

export type DreamRunFeedState = {
  entries: DreamRunFeedEntry[];
  inFlight: boolean;
  summary: DreamRunSummary | null;
  activeRuns: Record<string, ActiveDreamRun>;
};

const MAX_ENTRIES = 80;

export const EMPTY_DREAM_RUN_FEED: DreamRunFeedState = {
  entries: [],
  inFlight: false,
  summary: null,
  activeRuns: {},
};

function withEntry(state: DreamRunFeedState, entry: DreamRunFeedEntry): DreamRunFeedState {
  return {
    ...state,
    entries: [...state.entries, entry].slice(-MAX_ENTRIES),
  };
}

function runKey(phase: "plan" | "apply", runId: string | null): string {
  return `${phase}:${runId ?? "none"}`;
}

function pendingTotal(run: ActiveDreamRun): number {
  return Object.values(run.pending).reduce((total, count) => total + count, 0);
}

function anyRunInFlight(activeRuns: Record<string, ActiveDreamRun>): boolean {
  return Object.values(activeRuns).some((run) => pendingTotal(run) > 0);
}

function freshSummary(input: {
  key: string;
  phase: "plan" | "apply";
  run_id: string | null;
}): DreamRunSummary {
  return {
    key: input.key,
    phase: input.phase,
    run_id: input.run_id,
    changes: 0,
    errors: 0,
    duration_ms: null,
  };
}

function summaryForRun(
  state: DreamRunFeedState,
  input: {
    key: string;
    phase: "plan" | "apply";
    run_id: string | null;
  },
): DreamRunSummary {
  return state.summary?.key === input.key ? state.summary : freshSummary(input);
}

export function appendDreamRequest(
  state: DreamRunFeedState,
  input: {
    action: "plan" | "apply";
    processes: OfflineProcessName[];
    plan_id?: string;
    ts?: number;
  },
): DreamRunFeedState {
  const ts = input.ts ?? Date.now();
  return withEntry(
    {
      ...state,
      summary: input.action === "apply" ? null : state.summary,
    },
    {
      id: `request:${input.action}:${ts}`,
      kind: "request",
      ts,
      action: input.action,
      processes: input.processes,
      ...(input.plan_id === undefined ? {} : { plan_id: input.plan_id }),
    },
  );
}

function reduceStarted(
  state: DreamRunFeedState,
  frame: DreamProcessStartedFrame,
): DreamRunFeedState {
  const key = runKey(frame.phase, frame.run_id);
  const activeRuns = { ...state.activeRuns };
  const existing = activeRuns[key] ?? {
    key,
    phase: frame.phase,
    run_id: frame.run_id,
    pending: {},
  };
  activeRuns[key] = {
    ...existing,
    pending: {
      ...existing.pending,
      [frame.process]: (existing.pending[frame.process] ?? 0) + 1,
    },
  };

  return withEntry(
    {
      ...state,
      activeRuns,
      inFlight: true,
      summary: state.summary?.key === key ? state.summary : null,
    },
    {
      id: `started:${frame.phase}:${frame.process}:${frame.ts}`,
      kind: "started",
      ts: frame.ts,
      phase: frame.phase,
      process: frame.process,
      run_id: frame.run_id,
    },
  );
}

function reduceCompleted(
  state: DreamRunFeedState,
  frame: DreamProcessCompletedFrame,
): DreamRunFeedState {
  const key = runKey(frame.phase, frame.run_id);
  const activeRuns = { ...state.activeRuns };
  const active = activeRuns[key];
  if (active !== undefined) {
    const nextPending = { ...active.pending };
    const current = nextPending[frame.process] ?? 0;
    if (current <= 1) {
      delete nextPending[frame.process];
    } else {
      nextPending[frame.process] = current - 1;
    }

    const nextActive = { ...active, pending: nextPending };
    if (pendingTotal(nextActive) === 0) {
      delete activeRuns[key];
    } else {
      activeRuns[key] = nextActive;
    }
  }

  const previous = summaryForRun(state, {
    key,
    phase: frame.phase,
    run_id: frame.run_id,
  });
  const duration =
    frame.duration_ms === undefined
      ? previous.duration_ms
      : (previous.duration_ms ?? 0) + frame.duration_ms;
  return withEntry(
    {
      ...state,
      activeRuns,
      inFlight: anyRunInFlight(activeRuns),
      summary: {
        key,
        phase: frame.phase,
        run_id: frame.run_id,
        changes: previous.changes + frame.candidates_accepted,
        errors: previous.errors + frame.errors,
        duration_ms: duration,
      },
    },
    {
      id: `completed:${frame.phase}:${frame.process}:${frame.ts}`,
      kind: "completed",
      ts: frame.ts,
      phase: frame.phase,
      process: frame.process,
      run_id: frame.run_id,
      duration_ms: frame.duration_ms,
      errors: frame.errors,
      candidates_accepted: frame.candidates_accepted,
    },
  );
}

function reduceTick(state: DreamRunFeedState, frame: MaintenanceTickFrame): DreamRunFeedState {
  const key = runKey("apply", frame.run_id ?? null);
  const activeRuns = { ...state.activeRuns };
  if (frame.cadence === "manual") {
    delete activeRuns[key];
  }

  const next = withEntry(
    {
      ...state,
      activeRuns,
      inFlight: anyRunInFlight(activeRuns),
      summary:
        frame.cadence === "manual"
          ? {
              key,
              phase: "apply",
              run_id: frame.run_id ?? null,
              changes: frame.changes,
              errors: frame.errors,
              duration_ms: frame.duration_ms ?? (state.summary?.key === key ? state.summary.duration_ms : null),
            }
          : state.summary,
    },
    {
      id: `tick:${frame.cadence}:${frame.ts}`,
      kind: "tick",
      ts: frame.ts,
      cadence: frame.cadence,
      status: frame.status,
      run_id: frame.run_id,
      changes: frame.changes,
      errors: frame.errors,
      duration_ms: frame.duration_ms,
    },
  );

  return next;
}

export function reduceDreamRunFeed(
  state: DreamRunFeedState,
  frame: LiveFrame,
): DreamRunFeedState {
  if (frame.type === "dream:process:started") {
    return reduceStarted(state, frame);
  }
  if (frame.type === "dream:process:completed") {
    return reduceCompleted(state, frame);
  }
  if (frame.type === "maintenance:tick") {
    return reduceTick(state, frame);
  }

  return state;
}
