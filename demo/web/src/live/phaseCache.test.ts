import type { TurnPhaseFrame, TurnPhaseName } from "../api/types";
import { PhaseDurationCache } from "./phaseCache";

function frame(
  type: TurnPhaseFrame["type"],
  turnId: string,
  phase: TurnPhaseName,
  durationMs?: number,
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
      ...(durationMs === undefined ? {} : { duration_ms: durationMs }),
    },
  };
}

describe("PhaseDurationCache", () => {
  it("records completed and failed phase durations by turn", () => {
    const cache = new PhaseDurationCache();

    expect(cache.get("turn_1")).toBeNull();
    expect(cache.apply(frame("turn:phase:started", "turn_1", "retrieval"))).toBe(false);
    expect(cache.get("turn_1")).toBeNull();
    expect(cache.apply(frame("turn:phase:completed", "turn_1", "retrieval", 25))).toBe(true);
    expect(cache.apply(frame("turn:phase:failed", "turn_1", "guards", 5))).toBe(true);

    expect(cache.get("turn_1")).toMatchObject({
      turnId: "turn_1",
      sessionId: "s1",
      totalMs: 30,
      phases: [
        { phase: "retrieval", durationMs: 25, state: "done" },
        { phase: "guards", durationMs: 5, state: "failed" },
      ],
    });
  });

  it("bounds entries with an LRU policy", () => {
    const cache = new PhaseDurationCache(2);

    cache.apply(frame("turn:phase:completed", "turn_1", "retrieval", 1));
    cache.apply(frame("turn:phase:completed", "turn_2", "retrieval", 2));
    expect(cache.get("turn_1")).not.toBeNull();
    cache.apply(frame("turn:phase:completed", "turn_3", "retrieval", 3));

    expect(cache.size()).toBe(2);
    expect(cache.get("turn_2")).toBeNull();
    expect(cache.get("turn_1")).not.toBeNull();
    expect(cache.get("turn_3")).not.toBeNull();
  });
});
