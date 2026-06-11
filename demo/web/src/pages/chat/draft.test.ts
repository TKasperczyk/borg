import type { LiveFrame, TurnTerminalOutcome } from "../../api/types";
import { applyDraftFrame, EMPTY_DRAFT_STATE, isSuppressedTerminalOutcome } from "./draft";

function token(chunk: string, sequence: number): LiveFrame {
  return {
    type: "turn:token",
    ts: sequence,
    turn_id: "t1",
    session_id: "s1",
    phase: "final",
    chunk_text: chunk,
    sequence,
  };
}

function flush(text: string): LiveFrame {
  return {
    type: "turn:token:flush",
    ts: 3,
    turn_id: "t1",
    session_id: "s1",
    phase: "final",
    full_text: text,
  };
}

function terminal(outcome: TurnTerminalOutcome): LiveFrame {
  return {
    type: "turn:terminal",
    ts: 4,
    event: "turn.terminal",
    data: {
      turnId: "t1",
      turn_id: "t1",
      session_id: "s1",
      outcome,
      ts: 4,
      duration_ms: 10,
    },
  };
}

describe("draft accumulation", () => {
  it("uses the exact suppressed terminal outcome set", () => {
    expect(isSuppressedTerminalOutcome("suppressed_closure")).toBe(true);
    expect(isSuppressedTerminalOutcome("suppressed_generation_gate")).toBe(true);
    expect(isSuppressedTerminalOutcome("suppressed_action")).toBe(true);
    expect(isSuppressedTerminalOutcome("suppressed_other")).toBe(false);
    expect(isSuppressedTerminalOutcome("reflected")).toBe(false);
  });

  it("accumulates final tokens and flush replaces accumulated text", () => {
    let state = applyDraftFrame(EMPTY_DRAFT_STATE, token("hel", 1));
    state = applyDraftFrame(state, token("lo", 2));
    expect(state.current?.text).toBe("hello");

    state = applyDraftFrame(state, flush("hello final"));
    expect(state.current?.text).toBe("hello final");
  });

  it("hands suppressed terminal text to withheld previews and clears emitted drafts", () => {
    let state = applyDraftFrame(EMPTY_DRAFT_STATE, flush("withheld"));
    state = applyDraftFrame(state, terminal("suppressed_action"));
    expect(state.current).toBeNull();
    expect(state.withheldByTurn.t1).toBe("withheld");

    state = applyDraftFrame(EMPTY_DRAFT_STATE, flush("emitted"));
    state = applyDraftFrame(state, terminal("reflected"));
    expect(state.current).toBeNull();
    expect(state.withheldByTurn.t1).toBeUndefined();
  });
});
