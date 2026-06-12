import type { InflightTurn, TurnPhaseFrame } from "../../api/types";
import {
  applyPhaseFrame,
  initialPhaseGridState,
  mergeInflightIntoPhaseGrid,
} from "./turnPhase";

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

  it("merges an in-flight snapshot under live frames that raced ahead of it", () => {
    const inflight: InflightTurn = {
      turn_id: "t1",
      session_id: "s1",
      started_at: 1,
      last_event_at: 2,
      phases: [
        { phase: "ingest", status: "completed", duration_ms: 42 },
        { phase: "retrieval", status: "completed", duration_ms: 311 },
        { phase: "delib", status: "active", duration_ms: null },
      ],
    };

    // Fresh mount, no live frames yet: snapshot is the whole truth.
    const seededOnly = mergeInflightIntoPhaseGrid(initialPhaseGridState(), inflight);
    expect(seededOnly.turnId).toBe("t1");
    expect(seededOnly.phases.ingest).toMatchObject({ state: "done", durationMs: 42 });
    expect(seededOnly.phases.retrieval).toMatchObject({ state: "done", durationMs: 311 });
    expect(seededOnly.phases.delib.state).toBe("active");

    // A live frame for the SAME turn advanced past the snapshot before the
    // fetch resolved: live cells win, the snapshot's history backfills, and
    // the snapshot's stale active demotes instead of leaving two actives.
    let raced = initialPhaseGridState();
    raced = applyPhaseFrame(raced, frame("turn:phase:started", "t1", "final"));
    const merged = mergeInflightIntoPhaseGrid(raced, inflight);
    expect(merged.turnId).toBe("t1");
    expect(merged.phases.ingest).toMatchObject({ state: "done", durationMs: 42 });
    expect(merged.phases.retrieval).toMatchObject({ state: "done", durationMs: 311 });
    expect(merged.phases.final.state).toBe("active");
    expect(merged.phases.delib.state).toBe("idle");
    expect(Object.values(merged.phases).filter((cell) => cell.state === "active")).toHaveLength(1);

    // A different turn's grid is replaced wholesale.
    let otherTurn = initialPhaseGridState();
    otherTurn = applyPhaseFrame(otherTurn, frame("turn:phase:started", "t0", "final"));
    const replaced = mergeInflightIntoPhaseGrid(otherTurn, inflight);
    expect(replaced.turnId).toBe("t1");
    expect(replaced.phases.final.state).toBe("idle");
    expect(replaced.phases.delib.state).toBe("active");
  });
});
