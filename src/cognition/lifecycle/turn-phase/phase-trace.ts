import { performance } from "node:perf_hooks";

import type { Clock } from "../../../util/clock.js";
import type { SessionId } from "../../../util/ids.js";
import type { TurnTraceData, TurnTracer } from "../../tracing/tracer.js";

export type TurnPhaseName =
  | "ingest"
  | "audience"
  | "perception"
  | "frame"
  | "extract"
  | "closure_loop"
  | "generation_gate"
  | "retrieval"
  | "ledger"
  | "shared"
  | "delib"
  | "final"
  | "guards"
  | "persist"
  | "reflect";

export type TraceTurnPhaseOptions<T> = {
  tracer: TurnTracer;
  clock: Clock;
  turnId: string;
  sessionId?: SessionId;
  phase: TurnPhaseName;
  sub?: string;
  completedSub?: (result: T) => string | undefined;
  run: () => Promise<T>;
};

function phaseTraceData(input: {
  turnId: string;
  sessionId?: SessionId;
  phase: TurnPhaseName;
  clock: Clock;
  durationMs?: number;
  sub?: string;
}): {
  turnId: string;
  turn_id: string;
  session_id?: SessionId;
  phase: TurnPhaseName;
  ts: number;
  duration_ms?: number;
  sub?: string;
} & TurnTraceData {
  return {
    turnId: input.turnId,
    turn_id: input.turnId,
    ...(input.sessionId === undefined ? {} : { session_id: input.sessionId }),
    phase: input.phase,
    ts: input.clock.now(),
    ...(input.durationMs === undefined ? {} : { duration_ms: input.durationMs }),
    ...(input.sub === undefined ? {} : { sub: input.sub }),
  };
}

function errorSub(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }

  return String(error);
}

export async function traceTurnPhase<T>(options: TraceTurnPhaseOptions<T>): Promise<T> {
  if (!options.tracer.enabled) {
    return options.run();
  }

  const startWallMs = performance.now();
  options.tracer.emit(
    "turn_phase.started",
    phaseTraceData({
      turnId: options.turnId,
      sessionId: options.sessionId,
      phase: options.phase,
      clock: options.clock,
      sub: options.sub,
    }),
  );

  try {
    const result = await options.run();
    const sub = options.completedSub?.(result) ?? options.sub;
    options.tracer.emit(
      "turn_phase.completed",
      phaseTraceData({
        turnId: options.turnId,
        sessionId: options.sessionId,
        phase: options.phase,
        clock: options.clock,
        durationMs: Math.max(0, performance.now() - startWallMs),
        sub,
      }),
    );
    return result;
  } catch (error) {
    options.tracer.emit(
      "turn_phase.failed",
      phaseTraceData({
        turnId: options.turnId,
        sessionId: options.sessionId,
        phase: options.phase,
        clock: options.clock,
        durationMs: Math.max(0, performance.now() - startWallMs),
        sub: errorSub(error),
      }),
    );
    throw error;
  }
}
