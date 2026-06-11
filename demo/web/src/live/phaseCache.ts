import type { LiveFrame, TurnPhaseFrame, TurnPhaseName } from "../api/types";
import { TURN_PHASES } from "../api/types";

export type CachedPhaseDuration = {
  phase: TurnPhaseName;
  durationMs: number;
  state: "done" | "failed";
};

export type CachedTurnPhaseDurations = {
  turnId: string;
  sessionId: string | null;
  phases: CachedPhaseDuration[];
  totalMs: number;
};

function isPhaseFrame(frame: LiveFrame): frame is TurnPhaseFrame {
  return (
    frame.type === "turn:phase:started" ||
    frame.type === "turn:phase:completed" ||
    frame.type === "turn:phase:failed"
  );
}

export class PhaseDurationCache {
  private readonly entries = new Map<string, CachedTurnPhaseDurations>();

  constructor(private readonly limit = 100) {}

  apply(frame: LiveFrame): boolean {
    if (!isPhaseFrame(frame)) {
      return false;
    }

    const turnId = frame.data.turn_id;
    const current =
      this.entries.get(turnId) ??
      ({
        turnId,
        sessionId: frame.data.session_id ?? null,
        phases: [],
        totalMs: 0,
      } satisfies CachedTurnPhaseDurations);

    if (frame.type === "turn:phase:started") {
      this.touch(current);
      return false;
    }

    const durationMs = frame.data.duration_ms;
    if (durationMs === null || durationMs === undefined || !Number.isFinite(durationMs)) {
      this.touch(current);
      return false;
    }

    const nextPhases = current.phases.filter((phase) => phase.phase !== frame.data.phase);
    nextPhases.push({
      phase: frame.data.phase,
      durationMs,
      state: frame.type === "turn:phase:completed" ? "done" : "failed",
    });
    nextPhases.sort(
      (left, right) => TURN_PHASES.indexOf(left.phase) - TURN_PHASES.indexOf(right.phase),
    );

    this.touch({
      turnId,
      sessionId: frame.data.session_id ?? current.sessionId,
      phases: nextPhases,
      totalMs: nextPhases.reduce((sum, phase) => sum + phase.durationMs, 0),
    });
    return true;
  }

  get(turnId: string): CachedTurnPhaseDurations | null {
    const entry = this.entries.get(turnId);
    if (entry === undefined || entry.phases.length === 0) {
      return null;
    }

    this.entries.delete(turnId);
    this.entries.set(turnId, entry);
    return entry;
  }

  size(): number {
    return this.entries.size;
  }

  private touch(entry: CachedTurnPhaseDurations): void {
    this.entries.delete(entry.turnId);
    this.entries.set(entry.turnId, entry);

    while (this.entries.size > this.limit) {
      const oldest = this.entries.keys().next().value as string | undefined;
      if (oldest === undefined) {
        return;
      }

      this.entries.delete(oldest);
    }
  }
}
