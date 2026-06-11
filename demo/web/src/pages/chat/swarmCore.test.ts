import { swarmParamsForPhase } from "./swarmCore";

describe("swarm phase parameters", () => {
  it("maps real phases to distinct motion regimes", () => {
    expect(swarmParamsForPhase({ phase: "perception", delibPath: null, outcome: "idle", arousal: 0 })).toMatchObject({
      speed: 2.4,
      jitter: 4,
    });
    expect(swarmParamsForPhase({ phase: "frame", delibPath: null, outcome: "idle", arousal: 0 }).jitter).toBe(4);
    expect(swarmParamsForPhase({ phase: "extract", delibPath: null, outcome: "idle", arousal: 0 })).toMatchObject({
      targetR: 78,
      speed: 1.6,
    });
    expect(swarmParamsForPhase({ phase: "ledger", delibPath: null, outcome: "idle", arousal: 0 })).toMatchObject({
      targetR: 70,
      evidencePull: true,
    });
    expect(swarmParamsForPhase({ phase: "final", delibPath: null, outcome: "idle", arousal: 0 }).tokenEject).toBe(true);
  });

  it("distinguishes system-1 and system-2 deliberation", () => {
    const sys1 = swarmParamsForPhase({
      phase: "delib",
      delibPath: "system_1",
      outcome: "idle",
      arousal: 0,
    });
    const sys2 = swarmParamsForPhase({
      phase: "delib",
      delibPath: "system_2",
      outcome: "idle",
      arousal: 0,
    });

    expect(sys2.targetR).toBeLessThan(sys1.targetR);
    expect(sys2.speed).toBeGreaterThan(sys1.speed);
    expect(sys2.counterRotate).toBe(true);
  });

  it("stills the swarm after deliberate silence", () => {
    expect(
      swarmParamsForPhase({ phase: "delib", delibPath: "system_2", outcome: "silence", arousal: 1 }),
    ).toMatchObject({
      targetR: 96,
      speed: 0.12,
      alpha: 0.3,
    });
  });
});
