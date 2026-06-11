import type { TurnPhaseName } from "../../api/types";

export type DelibPath = "system_1" | "system_2" | null;
export type SwarmOutcome = "idle" | "emitted" | "silence" | "observed" | "suppressed" | "error";

export type SwarmPhaseInput = {
  phase: TurnPhaseName | null;
  delibPath: DelibPath;
  outcome: SwarmOutcome;
  arousal: number;
};

export type SwarmParams = {
  targetR: number;
  speed: number;
  alpha: number;
  jitter: number;
  counterRotate: boolean;
  evidencePull: boolean;
  tokenEject: boolean;
  barrier: boolean;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function breathForArousal(now: number, arousal: number, multiplier = 1): number {
  const a = clamp(arousal, 0, 1);
  return Math.sin(now * (0.0012 + a * 0.0035)) * (2.5 + a * 6) * multiplier;
}

export function swarmParamsForPhase(input: SwarmPhaseInput): SwarmParams {
  const arousalBoost = 1 + clamp(input.arousal, 0, 1) * 0.18;
  const neutral: SwarmParams = {
    targetR: 90,
    speed: 1 * arousalBoost,
    alpha: 0.55,
    jitter: 0,
    counterRotate: false,
    evidencePull: false,
    tokenEject: false,
    barrier: false,
  };

  if (input.outcome === "silence") {
    return {
      ...neutral,
      targetR: 96,
      speed: 0.12,
      alpha: 0.3,
      jitter: 0,
    };
  }

  if (input.phase === "perception" || input.phase === "frame") {
    return {
      ...neutral,
      speed: 2.4 * arousalBoost,
      jitter: 4,
    };
  }

  if (input.phase === "extract") {
    return {
      ...neutral,
      targetR: 78,
      speed: 1.6 * arousalBoost,
      jitter: 1,
    };
  }

  if (input.phase === "retrieval" || input.phase === "ledger" || input.phase === "shared") {
    return {
      ...neutral,
      targetR: 70,
      speed: 1.35 * arousalBoost,
      alpha: 0.65,
      evidencePull: true,
    };
  }

  if (input.phase === "delib") {
    const system2 = input.delibPath === "system_2";
    return {
      ...neutral,
      targetR: system2 ? 46 : 58,
      speed: (system2 ? 3.2 : 2) * arousalBoost,
      alpha: 0.85,
      jitter: system2 ? 0.7 : 0.35,
      counterRotate: system2,
    };
  }

  if (input.phase === "final") {
    return {
      ...neutral,
      targetR: 60,
      speed: 1.8 * arousalBoost,
      alpha: 0.75,
      tokenEject: true,
    };
  }

  if (input.phase === "guards") {
    return {
      ...neutral,
      targetR: 60,
      speed: 1.8 * arousalBoost,
      alpha: 0.72,
      barrier: input.outcome === "suppressed",
    };
  }

  if (
    input.phase === "ingest" ||
    input.phase === "audience" ||
    input.phase === "closure_loop" ||
    input.phase === "generation_gate" ||
    input.phase === "persist" ||
    input.phase === "reflect"
  ) {
    return {
      ...neutral,
      targetR: 86,
      speed: 1.1 * arousalBoost,
      alpha: 0.5,
      jitter: 0.25,
    };
  }

  return neutral;
}
