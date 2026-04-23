import type { TurnInput } from "../cognition/index.js";

export const AUTONOMY_TRIGGER_NAMES = [
  "commitment_expiring",
  "open_question_dormant",
  "scheduled_reflection",
] as const;

export type AutonomyTriggerName = (typeof AUTONOMY_TRIGGER_NAMES)[number];

export type DueEvent<Payload extends Record<string, unknown> = Record<string, unknown>> = {
  id: string;
  trigger: AutonomyTriggerName;
  watermarkProcessName: string;
  sortTs: number;
  payload: Payload;
};

export type AutonomyTrigger<Payload extends Record<string, unknown> = Record<string, unknown>> = {
  name: AutonomyTriggerName;
  scan(): Promise<DueEvent<Payload>[]>;
  buildTurn(event: DueEvent<Payload>): TurnInput;
};

export type AutonomyTickEventResult = {
  id: string;
  trigger: AutonomyTriggerName;
  status: "fired" | "budget_skipped" | "busy_skipped" | "error";
  payload: Record<string, unknown>;
  outcomeSummary?: string;
  turnResultId?: string | null;
  error?: string;
};

export type TickResult = {
  status: "disabled" | "ok";
  ts: number;
  scannedTriggers: AutonomyTriggerName[];
  dueEvents: number;
  firedEvents: number;
  budgetSkipped: number;
  busySkipped: number;
  errorCount: number;
  events: AutonomyTickEventResult[];
};
