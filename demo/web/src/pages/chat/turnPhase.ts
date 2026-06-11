import type { LiveFrame, TurnPhaseFrame, TurnPhaseName } from "../../api/types";
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
