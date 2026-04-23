/* Time-signal helpers for retrieval scoring and strict range filtering. */
import type { TemporalCue } from "../cognition/types.js";
import type { Episode } from "../memory/episodic/types.js";

export type ResolvedTimeRange = {
  start: number;
  end: number;
};

export type ResolvedTimeSignals = {
  scoringRange: ResolvedTimeRange | null;
  strictFilterRange: ResolvedTimeRange | null;
};

export type TimeSignalOptions = {
  timeRange?: {
    start: number;
    end: number;
  };
  temporalCue?: TemporalCue | null;
  strictTimeRange?: boolean;
};

function resolveTemporalCueTimeRange(
  temporalCue: TemporalCue | null | undefined,
): ResolvedTimeRange | null {
  if (temporalCue === null || temporalCue === undefined) {
    return null;
  }

  return {
    start: temporalCue.sinceTs ?? Number.NEGATIVE_INFINITY,
    end: temporalCue.untilTs ?? Number.POSITIVE_INFINITY,
  };
}

export function resolveTimeSignals(options: TimeSignalOptions): ResolvedTimeSignals {
  const temporalCueRange = resolveTemporalCueTimeRange(options.temporalCue);
  const explicitRange = options.timeRange ?? null;
  const effectiveRange = explicitRange ?? temporalCueRange;

  return {
    scoringRange: effectiveRange,
    strictFilterRange: options.strictTimeRange === true ? effectiveRange : null,
  };
}

export function overlapsTimeRange(
  episode: Pick<Episode, "start_time" | "end_time">,
  range: ResolvedTimeRange,
): boolean {
  return episode.start_time <= range.end && episode.end_time >= range.start;
}

export function computeTimeRelevance(
  episode: Pick<Episode, "start_time" | "end_time">,
  timeRange: ResolvedTimeRange | null,
): number {
  if (timeRange === null) {
    return 0;
  }

  return overlapsTimeRange(episode, timeRange) ? 1 : 0;
}
