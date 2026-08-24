import type { TurnInput } from "../cognition/index.js";
import type { ExecutiveGoalScore } from "../executive/index.js";

export const AUTONOMY_TRIGGER_NAMES = [
  "commitment_expiring",
  "open_question_dormant",
  "scheduled_reflection",
  "scheduled_wake",
  "goal_followup_due",
  "executive_focus_due",
] as const;

export type AutonomyTriggerName = (typeof AUTONOMY_TRIGGER_NAMES)[number];

export const AUTONOMY_CONDITION_NAMES = [
  "commitment_revoked",
  "mood_valence_drop",
  "open_question_urgency_bump",
] as const;

export type AutonomyConditionName = (typeof AUTONOMY_CONDITION_NAMES)[number];

export const AUTONOMY_WAKE_SOURCE_NAMES = [
  ...AUTONOMY_TRIGGER_NAMES,
  ...AUTONOMY_CONDITION_NAMES,
] as const;

export type AutonomyWakeSourceName = (typeof AUTONOMY_WAKE_SOURCE_NAMES)[number];
export type AutonomyWakeSourceType = "trigger" | "condition";
export type AutonomyWakeSourceCategory = "contemplative" | "operational";

export const AUTONOMY_WAKE_OUTCOMES = ["headway", "silent", "error", "busy"] as const;
export type AutonomyWakeOutcome = (typeof AUTONOMY_WAKE_OUTCOMES)[number];

export const AUTONOMY_WAKE_SOURCE_METADATA = {
  commitment_expiring: {
    type: "trigger",
    category: "operational",
  },
  open_question_dormant: {
    type: "trigger",
    category: "operational",
  },
  scheduled_reflection: {
    type: "trigger",
    category: "contemplative",
  },
  scheduled_wake: {
    type: "trigger",
    category: "contemplative",
  },
  goal_followup_due: {
    type: "trigger",
    category: "operational",
  },
  executive_focus_due: {
    type: "trigger",
    category: "operational",
  },
  commitment_revoked: {
    type: "condition",
    category: "operational",
  },
  mood_valence_drop: {
    type: "condition",
    category: "operational",
  },
  open_question_urgency_bump: {
    type: "condition",
    category: "operational",
  },
} as const satisfies Record<
  AutonomyWakeSourceName,
  {
    type: AutonomyWakeSourceType;
    category: AutonomyWakeSourceCategory;
  }
>;

export type DueEvent<Payload extends Record<string, unknown> = Record<string, unknown>> = {
  id: string;
  sourceName: AutonomyWakeSourceName;
  sourceType: AutonomyWakeSourceType;
  watermarkProcessName: string;
  sortTs: number;
  stateTs?: number;
  goalStaleBackoffActionAvailabilityKey?: string;
  // Internal wake-selection metadata. It is deliberately outside payload so
  // the single-goal model-facing shape remains byte-for-byte unchanged.
  executiveGoalScore?: ExecutiveGoalScore;
  executiveGoalRank?: number;
  payload: Payload;
};

export type AutonomyWakeSource<Payload extends Record<string, unknown> = Record<string, unknown>> =
  {
    name: AutonomyWakeSourceName;
    type: AutonomyWakeSourceType;
    sourceCategory: AutonomyWakeSourceCategory;
    scan(): Promise<DueEvent<Payload>[]>;
    buildTurn(event: DueEvent<Payload>): TurnInput;
    nextDueAt?(): Promise<number | null>;
    // Optional lifecycle hook invoked by the scheduler immediately after a wake
    // fires successfully (watermark committed). Lets a source make its own
    // persisted state authoritative at fire-time instead of waiting for the next
    // scan to reconcile. Best-effort: the watermark remains the idempotency
    // source of truth, so a throwing onFired never causes a re-fire.
    onFired?(event: DueEvent<Payload>): void | Promise<void>;
  };

export type AutonomyTrigger<Payload extends Record<string, unknown> = Record<string, unknown>> =
  AutonomyWakeSource<Payload>;

export type AutonomyCondition<Payload extends Record<string, unknown> = Record<string, unknown>> =
  AutonomyWakeSource<Payload>;

export type AutonomySchedulerWakeGroupDescription = {
  trigger_name: AutonomyWakeSourceName;
  wake_count: number;
  in_flight: number;
  outcome_counts: Record<AutonomyWakeOutcome, number>;
};

export type AutonomySchedulerBudgetDescription = {
  max_wakes_per_window: number;
  window_ms: number;
  /**
   * Lower edge of the rolling window the counts below are taken over, inclusive.
   * The window is anchored at the describe call's now, so two descriptions taken
   * minutes apart cover different intervals; without this the counts are not
   * comparable across reads.
   */
  window_started_at: number;
  used_in_current_window: number;
  reserved_contemplative_wakes_per_window: number;
  contemplative_used_in_current_window: number;
  wakes_in_current_window_by_trigger: AutonomySchedulerWakeGroupDescription[];
  next_budget_slot_frees_at: number | null;
};

export type AutonomySchedulerTriggerSourceDescription = {
  name: AutonomyTriggerName;
  type: "trigger";
  category: AutonomyWakeSourceCategory;
  enabled: boolean;
  next_due_at: number | null;
};

export type AutonomySchedulerConditionSourceDescription = {
  name: AutonomyConditionName;
  type: "condition";
  category: AutonomyWakeSourceCategory;
  enabled: boolean;
};

export type AutonomySchedulerSourceDescription =
  AutonomySchedulerTriggerSourceDescription | AutonomySchedulerConditionSourceDescription;

export type AutonomySchedulerFleetBrakeDescription = {
  enabled: boolean;
  /**
   * Consecutive completed operational wakes recorded `silent`. Errored and
   * busy-skipped wakes are transparent to it -- they neither increment nor
   * reset -- so the streak is consecutive within the *completed operational*
   * subsequence, not within the wake sequence, and can span any number of
   * intervening wakes and any amount of wall-clock.
   */
  empty_streak: number;
  empty_streak_threshold: number;
  streak_anchor_ts: number | null;
  cooldown_until: number | null;
  error_streak: number;
  error_streak_threshold: number;
  error_paused_until: number | null;
  bypass_count: number;
  /**
   * Outcome tally over the *budget* window, across both source categories.
   * A different population from `empty_streak` above: it is time-bounded where
   * the streak is not, counts contemplative wakes where the streak ignores
   * them, and counts errors where the streak passes over them. It does not
   * feed the streak and cannot be differenced into one.
   */
  window_outcomes: Record<AutonomyWakeOutcome, number>;
};

export type AutonomySchedulerDescription = {
  /**
   * The clock read every other field here is as of. `describe()` takes it once
   * and derives the budget cutoff, the window counts and `next_tick_at` from
   * it, so any surface that wants to say "these numbers are as of X" must use
   * this stamp and not its own. A caller's own `now` is necessarily earlier --
   * it was taken before it awaited `describe()` -- and quoting that instead
   * ages every count below it by however long the caller spent in between.
   */
  observed_at: number;
  enabled: boolean;
  interval_ms: number;
  next_tick_at: number | null;
  budget: AutonomySchedulerBudgetDescription;
  fleet_brake: AutonomySchedulerFleetBrakeDescription;
  sources: AutonomySchedulerSourceDescription[];
};

export type AutonomyTickEventResult = {
  id: string;
  sourceName: AutonomyWakeSourceName;
  sourceType: AutonomyWakeSourceType;
  sourceCategory: AutonomyWakeSourceCategory;
  status:
    | "fired"
    | "budget_skipped"
    | "fleet_cooldown_skipped"
    | "error_circuit_skipped"
    | "busy_skipped"
    | "bookkeeping_error"
    | "error";
  payload: Record<string, unknown>;
  outcomeSummary?: string;
  turnResultId?: string | null;
  error?: string;
};

export type TickResult = {
  status: "disabled" | "ok";
  ts: number;
  scannedSources: AutonomyWakeSourceName[];
  dueEvents: number;
  firedEvents: number;
  budgetSkipped: number;
  fleetCooldownSkipped: number;
  errorCircuitSkipped: number;
  busySkipped: number;
  errorCount: number;
  sourceErrorCount: number;
  bookkeepingErrorCount: number;
  events: AutonomyTickEventResult[];
};
