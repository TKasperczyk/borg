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

export const AUTONOMY_WAKE_OUTCOMES = [
  "headway",
  "silent",
  "error",
  "busy",
  "interrupted",
] as const;
export type AutonomyWakeOutcome = (typeof AUTONOMY_WAKE_OUTCOMES)[number];

/**
 * The terminal emission kinds that make a wake `headway` on their own. Named
 * here rather than inlined at the one comparison because the outcome's
 * predicate is also printed to the entity, and a list transcribed onto that page
 * would go quietly false the next time this one changes -- the same
 * copied-instead-of-derived failure that has already cost this repo a stale
 * fixture and a stale test array.
 */
export const HEADWAY_EMISSION_KINDS = ["message", "continue_thought"] as const;

/**
 * Distinct-detail tally for one outcome bucket over a window. `total` is the
 * bucket's own count, so `reasons` summing short of it is not a discrepancy --
 * `without_detail` is the named difference, and the three always reconcile.
 */
export type AutonomyWakeOutcomeDetailTally = {
  total: number;
  without_detail: number;
  reasons: Array<{ detail: string; count: number }>;
};

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
  /**
   * Fire stamps of the rows counted by `in_flight`, oldest first.
   *
   * `in_flight` alone is identity-free: a healthy transient and a wake whose
   * terminal write has not yet landed both render as the same integer. The
   * bookkeeping catch records the latter as `interrupted`, and startup
   * reconciliation closes any NULL row left by a prior process. Carrying the
   * stamps still makes open rows separable across reads -- a stamp that repeats
   * is one row not moving; a stamp that changes is a new wake.
   *
   * This count is taken over the rolling budget window, so an open row can leave
   * it by ageing past the window's lower edge before its terminal outcome lands.
   * A disappearing stamp therefore means either closure or window expiry.
   */
  in_flight_started_at: number[];
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
  | AutonomySchedulerTriggerSourceDescription
  | AutonomySchedulerConditionSourceDescription;

export type AutonomySchedulerFleetBrakeDescription = {
  enabled: boolean;
  /**
   * Consecutive counted-empty operational wakes. Chosen silence and failed
   * emission advance it; a post-generation guard block, an error, or a busy
   * skip is transparent -- it neither increments nor resets -- so the streak is
   * consecutive within that filtered operational subsequence, not within the
   * wake sequence, and can span intervening wakes and wall-clock time.
   */
  empty_streak: number;
  empty_streak_threshold: number;
  streak_anchor_ts: number | null;
  cooldown_until: number | null;
  error_streak: number;
  error_streak_threshold: number;
  error_paused_until: number | null;
  /**
   * Freshness bypasses spent -- neither a streak nor a window count. A bypass is
   * only ever offered while the empty-streak cooldown is actively holding, so a
   * clear cooldown freezes this rather than resetting it; a deadline bypass does
   * not spend one. It returns to zero only on an operational wake that came back
   * `headway` or a contemplative wake that delivered an outbound post -- not on
   * cooldown expiry and not on the budget window rolling -- so a non-zero value
   * can outlive the cooldown that produced it and is not a count over
   * `window_outcomes`.
   */
  bypass_count: number;
  /**
   * The bound `bypass_count` is spent against: at the cap a fresh concern stops
   * earning a bypass and is refused with everything else the cooldown is
   * holding. Carried here because the counter is otherwise a bare number whose
   * distance to its own refusal is unreadable from the value alone.
   */
  freshness_bypass_cap: number;
  /**
   * Outcome tally over the *budget* window, across both source categories.
   * A different population from `empty_streak` above: it is time-bounded where
   * the streak is not, counts contemplative wakes where the streak ignores
   * them, and counts errors and interruptions where the streak passes over them.
   * It does not feed the streak and cannot be differenced into one.
   */
  window_outcomes: Record<AutonomyWakeOutcome, number>;
  /**
   * The `headway` entry of `window_outcomes`, split by the structural basis the
   * scheduler recorded for the same rows. A row may name more than one basis;
   * the joined detail remains one tally key rather than being reinterpreted.
   */
  window_headway_reasons: AutonomyWakeOutcomeDetailTally;
  /**
   * The `error` entry of `window_outcomes` above, split by the failure the
   * scheduler recorded -- same rows, same window, same categories, one level of
   * detail further down. `total` repeats that count so the split can be checked
   * against it, and `without_detail` names the rows whose failure was not
   * recorded (every row written before the detail column existed), so the
   * reasons never have to be read as covering the bucket.
   */
  window_error_reasons: AutonomyWakeOutcomeDetailTally;
  /**
   * The `silent` entry of `window_outcomes` above, split by what actually ended
   * the wake -- same rows, same window, same categories, one level of detail
   * further down, and read the same way as `window_error_reasons`.
   *
   * `silent` is a union, not a behaviour: a wake the entity deliberately closed
   * with no output, a wake whose emission failed on the way out, and a wake a
   * post-generation guard blocked all land in it. The first two advance
   * `empty_streak`; the guard block leaves it unchanged. The scheduler uses the
   * same structural classification (`classifySuppressionReason`) for this row
   * and the streak disposition, so the diagnostic says which policy was applied
   * without recomputing the ending.
   */
  window_silent_reasons: AutonomyWakeOutcomeDetailTally;
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
  /**
   * The configuration flag the scheduler was constructed with, and nothing
   * else: it says the loop was asked to run, never that it is alive. Every
   * liveness question -- is a tick moving, is the interval still firing --
   * belongs to `tick_in_flight` and `scheduled_tick_at` below.
   */
  enabled: boolean;
  /**
   * Whether a tick was already running at the read. Load-bearing because the
   * two ways the loop falls behind are indistinguishable from the stamps
   * alone: the tick anchor is written on tick *entry*, and the interval
   * callback early-returns on every fire while a tick is in flight, so a long
   * tick holds `scheduled_tick_at` still while the overdue amount grows --
   * exactly the page an interval merely running behind produces. Reading them
   * apart used to need two reads far enough apart to see whether the stamp
   * moved. This is the same discriminator on a single read.
   *
   * True by construction on an autonomous turn: that turn is running inside
   * the tick that is being reported, so it only carries information off a
   * live turn, and any surface rendering it has to say so.
   */
  tick_in_flight: boolean;
  /**
   * The interval the handle was armed with. Load-bearing to any consumer
   * comparing `scheduled_tick_at` across reads: that stamp is the last tick
   * entry plus this, so while one handle stays armed consecutive values differ
   * by a multiple of it (plus timer drift), and a delta that is not a multiple
   * means the handle was re-armed. Without this term a series of stamps is
   * just a series of stamps -- the phase is only readable against the period.
   */
  interval_ms: number;
  /**
   * Interval fires refused because a tick was already running. The callback
   * early-returns on those, and the early return is the only event in the
   * scheduler that writes nothing at all: no wake row, no outcome, no error.
   * So a stretch in which every fire was refused and a stretch in which
   * nothing was due leave the same trace in `budget` and `window_outcomes`,
   * which is none. This is that difference, counted where it is refused.
   *
   * `current_tick` is null exactly when no tick is in flight; when one is, it
   * is that tick's own tally and resets on the next tick entry. It carries
   * information on an autonomous turn even though `tick_in_flight` cannot --
   * that turn is inside the tick, so the flag is true by construction, but how
   * many fires the tick has already refused is a fact about the freeze.
   */
  dropped_interval_fires: {
    since_interval_armed: number;
    current_tick: number | null;
  };
  /**
   * When the current interval handle was armed, or null when none is. Null under
   * the same condition as `next_tick_at`/`scheduled_tick_at`, so the three are
   * never half-present.
   *
   * `since_interval_armed` above names an epoch it does not identify: the count
   * resets to zero on every arm, so a low value is a long-quiet loop and a
   * freshly restarted one alike, and a value that repeats across two reads is
   * "unmoved" and "reset then re-earned" alike. That ambiguity is not removable
   * from the count at any number of reads -- it is the same identity-free-count
   * shape as `in_flight`, and the same remedy applies. This stamp is the epoch:
   * one that repeats across reads is one handle still armed, one that changes is
   * a re-arm, which needs the loop stopped and started.
   *
   * The scheduler has held this value since the first version of `start()` (it
   * is one of the two terms in the tick anchor); it simply never left the class.
   * Re-arm was inferable only off `scheduled_tick_at` grid arithmetic, which
   * needs two reads, an unbroken series, and a delta large enough for the
   * off-grid residue to exceed timer drift.
   */
  interval_armed_at: number | null;
  /**
   * `max(scheduled_tick_at, observed_at)` -- a tick already due at the read is
   * reported as the read clock rather than as a past instant, so a consumer
   * that renders it as "next evaluation" never shows a time that has been and
   * gone. The floor is lossy: it discards how overdue the tick was, and the
   * relative age anything hangs on the floored stamp is time since the read,
   * not tick lateness. `scheduled_tick_at` below is the unfloored value, so
   * the discarded quantity is recoverable rather than destroyed here.
   */
  next_tick_at: number | null;
  /**
   * `tickAnchor + interval_ms` with no floor: when the loop is behind, this is
   * in the past of `observed_at` and the difference is exactly how overdue the
   * tick was at the read. Null under the same condition as `next_tick_at` (no
   * interval handle), so the pair is never half-present.
   */
  scheduled_tick_at: number | null;
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
