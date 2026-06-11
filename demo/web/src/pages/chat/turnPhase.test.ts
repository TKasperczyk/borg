import type { TurnPhaseFrame } from "../../api/types";
import { applyPhaseFrame, initialPhaseGridState } from "./turnPhase";

function frame(
  type: TurnPhaseFrame["type"],
  turnId: string,
  phase: TurnPhaseFrame["data"]["phase"],
  duration_ms?: number,
): TurnPhaseFrame {
  return {
    type,
    ts: 1,
    event:
      type === "turn:phase:started"
        ? "turn_phase.started"
        : type === "turn:phase:completed"
          ? "turn_phase.completed"
          : "turn_phase.failed",
    data: {
      turnId,
      turn_id: turnId,
      session_id: "s1",
      phase,
      ts: 1,
      ...(duration_ms === undefined ? {} : { duration_ms }),
    },
  };
}

describe("turn phase reducer", () => {
  it("updates active, done, failed, and resets on a new turn", () => {
    let state = initialPhaseGridState();
    state = applyPhaseFrame(state, frame("turn:phase:started", "t1", "retrieval"));
    expect(state.turnId).toBe("t1");
    expect(state.phases.retrieval.state).toBe("active");

    state = applyPhaseFrame(state, frame("turn:phase:completed", "t1", "retrieval", 712));
    expect(state.phases.retrieval).toMatchObject({ state: "done", durationMs: 712 });

    state = applyPhaseFrame(state, frame("turn:phase:failed", "t1", "ledger", 33));
    expect(state.phases.ledger).toMatchObject({ state: "failed", durationMs: 33 });

    state = applyPhaseFrame(state, frame("turn:phase:started", "t2", "ingest"));
    expect(state.turnId).toBe("t2");
    expect(state.phases.retrieval.state).toBe("idle");
    expect(state.phases.ledger.state).toBe("idle");
    expect(state.phases.ingest.state).toBe("active");
  });

  it("keeps only one active phase and handles out-of-order completions", () => {
    let state = initialPhaseGridState();
    state = applyPhaseFrame(state, frame("turn:phase:started", "t1", "retrieval"));
    state = applyPhaseFrame(state, frame("turn:phase:started", "t1", "ledger"));

    expect(state.phases.retrieval).toMatchObject({ state: "idle", durationMs: null });
    expect(state.phases.ledger.state).toBe("active");
    expect(Object.values(state.phases).filter((cell) => cell.state === "active")).toHaveLength(1);

    state = applyPhaseFrame(state, frame("turn:phase:completed", "t1", "retrieval", 44));

    expect(state.phases.retrieval).toMatchObject({ state: "done", durationMs: 44 });
    expect(state.phases.ledger.state).toBe("active");
    expect(Object.values(state.phases).filter((cell) => cell.state === "active")).toHaveLength(1);
  });
});
