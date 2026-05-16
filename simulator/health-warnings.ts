import type { MetricsRow, SimulatorHealthWarning } from "./types.js";

export const ACTIVE_GOAL_HIGH_THRESHOLD = 25;
export const ACTIVE_GOAL_GROWTH_START_TURN = 20;
export const ACTIVE_GOAL_GROWTH_WINDOW_ROWS = 10;
export const ACTIVE_GOAL_GROWTH_THRESHOLD_PER_TURN = 0.5;

export function simulatorHealthWarningsForRows(
  rows: readonly MetricsRow[],
): SimulatorHealthWarning[] {
  const latest = rows.at(-1);

  if (latest === undefined) {
    return [];
  }

  const warnings: SimulatorHealthWarning[] = [];

  if (latest.active_goal_count > ACTIVE_GOAL_HIGH_THRESHOLD) {
    warnings.push({
      kind: "active_goals_high",
      turn_counter: latest.turn_counter,
      turnId: latest.turnId,
      threshold: ACTIVE_GOAL_HIGH_THRESHOLD,
      observed_value: latest.active_goal_count,
    });
  }

  if (latest.turn_counter > ACTIVE_GOAL_GROWTH_START_TURN) {
    const postStartRows = rows.filter(
      (row) => row.turn_counter > ACTIVE_GOAL_GROWTH_START_TURN,
    );
    const windowRows = postStartRows.slice(-ACTIVE_GOAL_GROWTH_WINDOW_ROWS);
    const first = windowRows[0];
    const last = windowRows.at(-1);

    if (
      windowRows.length === ACTIVE_GOAL_GROWTH_WINDOW_ROWS &&
      first !== undefined &&
      last !== undefined &&
      last.turn_counter > first.turn_counter
    ) {
      const slope =
        (last.active_goal_count - first.active_goal_count) /
        (last.turn_counter - first.turn_counter);

      if (slope > ACTIVE_GOAL_GROWTH_THRESHOLD_PER_TURN) {
        warnings.push({
          kind: "active_goals_growth_high",
          turn_counter: latest.turn_counter,
          turnId: latest.turnId,
          threshold: ACTIVE_GOAL_GROWTH_THRESHOLD_PER_TURN,
          observed_value: slope,
          window_start_turn: first.turn_counter,
          window_turns: last.turn_counter - first.turn_counter,
        });
      }
    }
  }

  return warnings;
}
