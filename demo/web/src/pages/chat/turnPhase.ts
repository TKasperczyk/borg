import type { InflightTurn, LiveFrame, TurnPhaseFrame, TurnPhaseName } from "../../api/types";
import { TURN_PHASES } from "../../api/types";

export type PhaseDotState = "idle" | "active" | "done" | "failed";

export type PhaseCellState = {
  phase: TurnPhaseName;
  label: string;
  state: PhaseDotState;
  durationMs: number | null;
};

export type TurnPhaseGridState = {
  turnId: string | null;
  sessionId: string | null;
  phases: Record<TurnPhaseName, PhaseCellState>;
};

export const PHASE_LABELS: Record<TurnPhaseName, string> = {
  ingest: "INGEST",
  audience: "AUDIENCE",
  perception: "PERCEPT",
  frame: "FRAME",
  extract: "EXTRACT",
  closure_loop: "CLOSURE",
  generation_gate: "GEN GATE",
  retrieval: "RETRIEVE",
  ledger: "LEDGER",
  shared: "SHARED",
  delib: "DELIB",
  final: "FINAL",
  guards: "GUARDS",
  persist: "PERSIST",
  reflect: "REFLECT",
};

function makePhases(): Record<TurnPhaseName, PhaseCellState> {
  return Object.fromEntries(
    TURN_PHASES.map((phase) => [
      phase,
      {
        phase,
        label: PHASE_LABELS[phase],
        state: "idle" as const,
        durationMs: null,
      },
    ]),
  ) as Record<TurnPhaseName, PhaseCellState>;
}

export function initialPhaseGridState(): TurnPhaseGridState {
  return {
    turnId: null,
    sessionId: null,
    phases: makePhases(),
  };
}

function isTurnPhaseName(phase: string): phase is TurnPhaseName {
  return (TURN_PHASES as readonly string[]).includes(phase);
}

// Seeds the grid from the server's in-flight snapshot so a page mounted
// mid-turn shows the turn's real progress instead of an idle core. Live
// frames continue advancing the grid from here.
export function seedPhaseGridFromInflight(inflight: InflightTurn): TurnPhaseGridState {
  const phases = makePhases();

  for (const entry of inflight.phases) {
    if (!isTurnPhaseName(entry.phase)) {
      continue;
    }

    phases[entry.phase] = {
      ...phases[entry.phase],
      state:
        entry.status === "active" ? "active" : entry.status === "completed" ? "done" : "failed",
      durationMs: entry.duration_ms,
    };
  }

  // Server-side phase spans overlap (e.g. retrieval/ledger/shared run as
  // nested spans), so a snapshot can carry several actives at once. The grid
  // renders only the most recently started one -- the same single-active rule
  // applyPhaseFrame enforces via clearActivePhases. Snapshot order is started
  // order, so the last active entry wins.
  const activeNames = inflight.phases
    .filter((entry) => entry.status === "active" && isTurnPhaseName(entry.phase))
    .map((entry) => entry.phase as TurnPhaseName);
  const lastActive = activeNames.at(-1);
  for (const name of activeNames) {
    if (name !== lastActive) {
      phases[name] = { ...phases[name], state: "idle", durationMs: null };
    }
  }

  return {
    turnId: inflight.turn_id,
    sessionId: inflight.session_id,
    phases,
  };
}

// Live frames can race ahead of the snapshot fetch: a frame for the same turn
// may have already advanced the grid. The snapshot still carries the phase
// history those frames missed, so merge -- snapshot as the base, non-idle live
// cells winning (they are newer). When live state has its own active phase,
// snapshot-sourced actives are stale and demote to idle (same honest-fallback
// rule as clearActivePhases).
export function mergeInflightIntoPhaseGrid(
  current: TurnPhaseGridState,
  inflight: InflightTurn,
): TurnPhaseGridState {
  const seeded = seedPhaseGridFromInflight(inflight);

  if (current.turnId !== inflight.turn_id) {
    return seeded;
  }

  const liveHasActive = TURN_PHASES.some((phase) => current.phases[phase].state === "active");
  const phases = { ...seeded.phases };

  for (const phase of TURN_PHASES) {
    const live = current.phases[phase];

    if (live.state !== "idle") {
      phases[phase] = live;
      continue;
    }

    if (liveHasActive && phases[phase].state === "active") {
      phases[phase] = { ...phases[phase], state: "idle", durationMs: null };
    }
  }

  return { ...seeded, sessionId: current.sessionId ?? seeded.sessionId, phases };
}

function isTurnPhaseFrame(frame: LiveFrame): frame is TurnPhaseFrame {
  return (
    frame.type === "turn:phase:started" ||
    frame.type === "turn:phase:completed" ||
    frame.type === "turn:phase:failed"
  );
}

function clearActivePhases(
  phases: Record<TurnPhaseName, PhaseCellState>,
  nextActivePhase: TurnPhaseName,
): Record<TurnPhaseName, PhaseCellState> {
  return Object.fromEntries(
    TURN_PHASES.map((phase) => {
      const cell = phases[phase];
      if (phase === nextActivePhase || cell.state !== "active") {
        return [phase, cell];
      }

      // Honest fallback for missing completion frames: do not mark done without duration.
      return [phase, { ...cell, state: "idle" as const, durationMs: null }];
    }),
  ) as Record<TurnPhaseName, PhaseCellState>;
}

export function applyPhaseFrame(
  state: TurnPhaseGridState,
  frame: LiveFrame,
): TurnPhaseGridState {
  if (!isTurnPhaseFrame(frame)) {
    return state;
  }

  const turnId = frame.data.turn_id;
  const isNewTurn = frame.type === "turn:phase:started" && turnId !== state.turnId;
  const base = isNewTurn
    ? {
        turnId,
        sessionId: frame.data.session_id ?? null,
        phases: makePhases(),
      }
    : state;

  if (base.turnId !== turnId) {
    return state;
  }

  const phase = frame.data.phase;
  const nextState =
    frame.type === "turn:phase:started"
      ? "active"
      : frame.type === "turn:phase:completed"
        ? "done"
        : "failed";
  const phaseBase =
    frame.type === "turn:phase:started" ? clearActivePhases(base.phases, phase) : base.phases;
  const current = phaseBase[phase];

  return {
    ...base,
    sessionId: frame.data.session_id ?? base.sessionId,
    phases: {
      ...phaseBase,
      [phase]: {
        ...current,
        state: nextState,
        durationMs:
          frame.type === "turn:phase:started" ? current.durationMs : frame.data.duration_ms ?? null,
      },
    },
  };
}
