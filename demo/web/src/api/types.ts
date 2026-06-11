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

export type SessionRecord = {
  session_id: string;
  source_type: string;
  source_external_id: string | null;
  source_url: string | null;
  label: string;
  audience_label: string;
  audience_entity_id: string | null;
  conversation_kind: "dm" | "channel" | "thread" | "demo";
  created_at: number;
  last_activity_at: number;
  last_turn_id: string | null;
  message_count: number;
  status: "active" | "idle" | "archived";
  privacy_level: "payload_off" | "payload_on";
  participation_policy: "active" | "paused" | "observing" | "muted";
  audience_role: "participant" | "operator";
};

export type SessionsResponse = {
  sessions: SessionRecord[];
};

export type StreamEntryKind =
  | "user_msg"
  | "user_image_attachment"
  | "agent_msg"
  | "agent_suppressed"
  | "agent_observed"
  | "thought"
  | "tool_call"
  | "tool_result"
  | "perception"
  | "internal_event"
  | "dream_report";

export type StreamEntry = {
  id: string;
  timestamp: number;
  entry_index?: number;
  kind: StreamEntryKind;
  content: unknown;
  display_content?: unknown;
  turn_id?: string;
  turn_status?: "active" | "aborted";
  token_estimate?: number;
  tool_calls?: unknown[];
  audience?: string;
  sender_entity_id: string | null;
  reply_target_entity_id: string | null;
  source_message_key?: {
    source_type: string;
    source_external_id: string;
    external_message_id: string;
  };
  response_to?: {
    kind: "stream_backlog";
    from_cursor_exclusive: { ts: number; entryId: string } | null;
    through_cursor_inclusive: { ts: number; entryId: string };
    source_entry_ids: string[];
    count: number;
  };
  persistence_class?: "assistant_self_report";
  receipt_pending?: boolean;
  session_id: string;
  compressed?: boolean;
  sender_label: string | null;
  session_label: string | null;
  audience_label: string | null;
};

export type StreamResponse = {
  entries: StreamEntry[];
  next_cursor: string | null;
};

export type SuppressionOutcomeClass =
  | "deliberate-silence"
  | "emission-failed"
  | "guard-blocked"
  | "observed"
  | "unknown";

export type TurnHistoryOutcome = "emitted" | "failed" | SuppressionOutcomeClass;

export type TurnHistoryRow = {
  turn_id: string;
  started_at: number;
  audience: string | null;
  outcome: TurnHistoryOutcome;
  suppression_reason: string | null;
};

export type TurnsResponse = {
  rows: TurnHistoryRow[];
  next_cursor: string | null;
};

export type EvidenceLedgerEntry = {
  id: string;
  source_type?: string;
  session_scope?: "current_session" | "prior_session" | "global";
  actor?: "user" | "assistant" | "system" | "memory";
  trust_rank?: number;
  text?: string;
  value?: string;
  state?: string;
  salience_class?: string;
  state_metadata?: Record<string, unknown>;
  taint?: string;
  persistence_class?: string;
  via_retrieval?: boolean;
  stream_index?: number;
  citations?: string[];
  citation_type?: string;
};

export type EvidenceLedgerSection = {
  id: string;
  label: string;
  framing?: {
    text: string;
    counts?: Record<string, number>;
  };
  entries: EvidenceLedgerEntry[];
};

export type EvidenceLedger = {
  sections: EvidenceLedgerSection[];
  audienceStanding?: Record<string, EvidenceLedgerEntry[]>;
  sharedState?: unknown;
  transcriptIncluded?: boolean;
  transcriptCompacted?: boolean;
  transcriptOmittedReason?: string;
  originalTranscriptTokenEstimate?: number;
  compactedTranscriptEntryCount?: number;
  rawPreservedUserTranscriptEntryCount?: number;
  estimatedTokens?: number;
  imageAttachments?: Array<{
    label: string;
    attachment_id: string;
    byte_size?: number;
    citation_type: "original_image";
  }>;
};

export type LedgerResponse = {
  turn_id: string;
  ledger: EvidenceLedger | null;
};

export type TurnPostResponse = {
  ok: boolean;
  status: string;
  stream_entry_id: string;
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
  ledger: EvidenceLedger | null;
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
