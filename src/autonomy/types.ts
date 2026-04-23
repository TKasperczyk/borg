import type { TurnInput } from "../cognition/index.js";

export const AUTONOMY_TRIGGER_NAMES = [
  "commitment_expiring",
  "open_question_dormant",
  "scheduled_reflection",
  "goal_followup_due",
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

export type DueEvent<Payload extends Record<string, unknown> = Record<string, unknown>> = {
  id: string;
  sourceName: AutonomyWakeSourceName;
  sourceType: AutonomyWakeSourceType;
  watermarkProcessName: string;
  sortTs: number;
  payload: Payload;
};

export type AutonomyWakeSource<Payload extends Record<string, unknown> = Record<string, unknown>> = {
  name: AutonomyWakeSourceName;
  type: AutonomyWakeSourceType;
  scan(): Promise<DueEvent<Payload>[]>;
  buildTurn(event: DueEvent<Payload>): TurnInput;
};

export type AutonomyTrigger<Payload extends Record<string, unknown> = Record<string, unknown>> =
  AutonomyWakeSource<Payload>;

export type AutonomyCondition<Payload extends Record<string, unknown> = Record<string, unknown>> =
  AutonomyWakeSource<Payload>;

export type AutonomyTickEventResult = {
  id: string;
  sourceName: AutonomyWakeSourceName;
  sourceType: AutonomyWakeSourceType;
  status: "fired" | "budget_skipped" | "busy_skipped" | "error";
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
  busySkipped: number;
  errorCount: number;
  events: AutonomyTickEventResult[];
};
