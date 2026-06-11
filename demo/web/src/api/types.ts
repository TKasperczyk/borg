export type MoodState = {
  session_id: string;
  valence: number;
  arousal: number;
  updated_at: number;
  half_life_hours: number;
  recent_triggers: string[];
};

export type ApiState = {
  active_session: string;
  audiences: string[];
  counts: {
    turns: number;
    commitments: number;
    open_qs: number;
    open_reviews: number;
    dream_audit_rows: number;
  };
  current_mood: MoodState | null;
  runtime?: {
    model: string;
    embedding: {
      model: string;
      dims: number;
    };
  };
  version: string;
};

// Source: src/cognition/lifecycle/turn-phase/phase-trace.ts
export const TURN_PHASES = [
  "ingest",
  "audience",
  "perception",
  "frame",
  "extract",
  "closure_loop",
  "generation_gate",
  "retrieval",
  "ledger",
  "shared",
  "delib",
  "final",
  "guards",
  "persist",
  "reflect",
] as const;

export type TurnPhaseName = (typeof TURN_PHASES)[number];
export type TurnTokenPhase = "delib" | "final";
export type TurnTerminalOutcome =
  | "reflected"
  | "suppressed_closure"
  | "suppressed_generation_gate"
  | "suppressed_action"
  | "aborted"
  | "error";

export type TurnPhaseTraceData = {
  turnId: string;
  turn_id: string;
  session_id?: string;
  phase: TurnPhaseName;
  ts: number;
  duration_ms?: number;
  sub?: string;
  [key: string]: unknown;
};

export type BaseLiveFrame<TType extends string> = {
  type: TType;
  ts: number;
};

export type TurnPhaseFrame = BaseLiveFrame<
  "turn:phase:started" | "turn:phase:completed" | "turn:phase:failed"
> & {
  event: "turn_phase.started" | "turn_phase.completed" | "turn_phase.failed";
  data: TurnPhaseTraceData;
};

export type TurnPhaseDetailFrame = BaseLiveFrame<"turn:phase:detail"> & {
  turn_id: string;
  session_id?: string;
  phase?: TurnPhaseName | string;
  event: string;
  summary: string;
};

export type TurnTokenFrame = BaseLiveFrame<"turn:token"> & {
  turn_id: string;
  session_id?: string;
  phase: TurnTokenPhase;
  chunk_text: string;
  sequence: number;
};

export type TurnTokenFlushFrame = BaseLiveFrame<"turn:token:flush"> & {
  turn_id: string;
  session_id?: string;
  phase: TurnTokenPhase;
  full_text: string;
};

export type TurnDeliberationPathFrame = BaseLiveFrame<"turn:delib_path"> & {
  turn_id: string;
  session_id?: string;
  path: "system_1" | "system_2";
};

export type TurnFinalAttemptFrame = BaseLiveFrame<"turn:final_attempt"> & {
  turn_id: string;
  session_id?: string;
  attempt: number;
};

export type EvidenceLedgerBuiltFrame = BaseLiveFrame<"evidence_ledger:built"> & {
  turn_id: string;
  session_id?: string;
  ledger: unknown;
};

export type TurnTerminalFrame = BaseLiveFrame<"turn:terminal"> & {
  event: "turn.terminal";
  data: {
    turnId: string;
    turn_id: string;
    session_id: string;
    outcome: TurnTerminalOutcome;
    ts: number;
    duration_ms: number;
    [key: string]: unknown;
  };
};

export type StreamAppendFrame = BaseLiveFrame<"stream:append"> & {
  session_id?: string;
  entries: Record<string, unknown>[];
};

export type MaintenanceTickFrame = BaseLiveFrame<"maintenance:tick"> & {
  cadence: "light" | "heavy" | "manual";
  status: "ok" | "skipped_busy" | "skipped_empty" | "disabled" | "error";
  processes: string[];
  changed: boolean;
  changes: number;
  errors: number;
  pending_extraction_episodes?: number;
  run_id?: string | null;
  duration_ms?: number;
  reason?: string;
};

export type DreamProcessStartedFrame = BaseLiveFrame<"dream:process:started"> & {
  process: string;
  run_id: string | null;
  phase: "plan" | "apply";
};

export type DreamProcessCompletedFrame = BaseLiveFrame<"dream:process:completed"> & {
  process: string;
  run_id: string | null;
  phase: "plan" | "apply";
  duration_ms?: number;
  errors: number;
  candidates_accepted: number;
};

export type BorgResetFrame = BaseLiveFrame<"borg:reset">;

export type LiveFrame =
  | TurnPhaseFrame
  | TurnPhaseDetailFrame
  | TurnTokenFrame
  | TurnTokenFlushFrame
  | TurnDeliberationPathFrame
  | TurnFinalAttemptFrame
  | EvidenceLedgerBuiltFrame
  | TurnTerminalFrame
  | StreamAppendFrame
  | MaintenanceTickFrame
  | DreamProcessStartedFrame
  | DreamProcessCompletedFrame
  | BorgResetFrame;

export type LiveFrameType = LiveFrame["type"];
