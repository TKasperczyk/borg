import type { LiveFrame } from "../../api/types";

const SUPPRESSED_TERMINAL_OUTCOMES = new Set([
  "suppressed_closure",
  "suppressed_generation_gate",
  "suppressed_action",
]);

export function isSuppressedTerminalOutcome(outcome: string): boolean {
  return SUPPRESSED_TERMINAL_OUTCOMES.has(outcome);
}

export type DraftState = {
  current: {
    turnId: string;
    sessionId: string | null;
    text: string;
  } | null;
  withheldByTurn: Record<string, string>;
};

export const EMPTY_DRAFT_STATE: DraftState = {
  current: null,
  withheldByTurn: {},
};

export function applyDraftFrame(state: DraftState, frame: LiveFrame): DraftState {
  if (frame.type === "turn:token" && frame.phase === "final") {
    const existing = state.current?.turnId === frame.turn_id ? state.current.text : "";
    return {
      ...state,
      current: {
        turnId: frame.turn_id,
        sessionId: frame.session_id ?? null,
        text: `${existing}${frame.chunk_text}`,
      },
    };
  }

  if (frame.type === "turn:token:flush" && frame.phase === "final") {
    return {
      ...state,
      current: {
        turnId: frame.turn_id,
        sessionId: frame.session_id ?? null,
        text: frame.full_text,
      },
    };
  }

  if (frame.type !== "turn:terminal") {
    return state;
  }

  const turnId = frame.data.turn_id;
  if (state.current?.turnId !== turnId) {
    return state;
  }

  const suppressed = isSuppressedTerminalOutcome(frame.data.outcome);
  if (!suppressed) {
    return {
      ...state,
      current: null,
    };
  }

  return {
    current: null,
    withheldByTurn: {
      ...state.withheldByTurn,
      [turnId]: state.current.text,
    },
  };
}
