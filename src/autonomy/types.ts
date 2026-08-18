import type { TurnInput } from "../cognition/index.js";

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

export type AutonomySchedulerBudgetDescription = {
  max_wakes_per_window: number;
  window_ms: number;
  used_in_current_window: number;
  reserved_contemplative_wakes_per_window: number;
  contemplative_used_in_current_window: number;
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
  empty_streak: number;
  streak_anchor_ts: number | null;
  cooldown_until: number | null;
  error_streak: number;
  error_paused_until: number | null;
  bypass_count: number;
  window_outcomes: Record<AutonomyWakeOutcome, number>;
};

export type AutonomySchedulerDescription = {
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
