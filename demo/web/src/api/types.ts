export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

// Narrow copies of the public Borg DTOs used by demo/server. Keeping them local
// avoids pulling Node-oriented Borg runtime imports into the browser bundle.
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

export type StreamChatKind = "user_msg" | "agent_msg" | "user_image_attachment";

export type StreamEntry = {
  id: string;
  timestamp: number;
  kind: StreamEntryKind;
  content: unknown;
  turn_id?: string;
  turn_status?: "active" | "aborted";
  token_estimate?: number;
  tool_calls?: unknown[];
  audience?: string;
  sender_entity_id: string | null;
  reply_target_entity_id: string | null;
  persistence_class?: "assistant_self_report";
  session_id: string;
  compressed: boolean;
};

export type StreamResponse = {
  entries: StreamEntry[];
  next_cursor: string | null;
};

export type SharedStateEntryKind =
  | "locked"
  | "live"
  | "low_salience_live"
  | "dormant_live"
  | "tentative"
  | "invalidated"
  | "pending";

export type SharedStateEntry = {
  id: string;
  audience_entity_id: string;
  state_key: string | null;
  kind: SharedStateEntryKind;
  text: string;
  owner_entity_id: string | null;
  provenance_stream_entry_ids: string[];
  last_updated_stream_entry_ids: string[];
  created_at: number;
  last_updated_at: number;
  last_updated_turn_global: number | null;
  superseded_by_id: string | null;
  rank: number;
  canonicalizes: {
    goal_ids: string[];
    commitment_ids: string[];
    action_ids: string[];
    open_question_ids: string[];
  };
};

export type SharedStateResponse = {
  audience: string;
  entries: SharedStateEntry[];
};

export type EvidenceLedgerSourceType =
  | "current_user_message"
  | "current_session_stream"
  | "prior_session_stream"
  | "episode"
  | "semantic_node"
  | "semantic_edge"
  | "action_record"
  | "relational_slot"
  | "commitment"
  | "image_attachment"
  | "assistant_stream"
  | "system_metadata";

export type EvidenceLedgerEntry = {
  id: string;
  source_type: EvidenceLedgerSourceType;
  session_scope: "current_session" | "prior_session" | "global";
  actor: "user" | "assistant" | "system" | "memory";
  trust_rank: number;
  text?: string;
  value?: string;
  state?: string;
  salience_class?: string;
  state_metadata?: Record<string, unknown>;
  taint?: "none" | "assistant_seeded" | "quarantined" | "contested";
  persistence_class?: "assistant_self_report";
  via_retrieval?: boolean;
  stream_index?: number;
  citations?: string[];
  citation_type?: "original_image" | "generated_perception_text" | "parent_user_message";
};

export type EvidenceLedgerSection = {
  id: string;
  label: string;
  entries: EvidenceLedgerEntry[];
};

export type EvidenceLedgerImageAttachment = {
  label: string;
  attachment_id: string;
  byte_size?: number;
  citation_type: "original_image";
};

export type EvidenceLedger = {
  sections: EvidenceLedgerSection[];
  sharedState?: {
    entries: SharedStateEntry[];
    [key: string]: unknown;
  } | null;
  transcriptIncluded: boolean;
  transcriptCompacted: boolean;
  transcriptOmittedReason?: "over_budget";
  originalTranscriptTokenEstimate: number;
  compactedTranscriptEntryCount: number;
  rawPreservedUserTranscriptEntryCount: number;
  estimatedTokens: number;
  imageAttachments?: EvidenceLedgerImageAttachment[];
};

export type LedgerResponse = {
  turn_id: string;
  ledger: EvidenceLedger;
};

export type MoodSnapshot = {
  session_id: string;
  valence: number;
  arousal: number;
  updated_at: number;
  half_life_hours: number;
  recent_triggers: string[];
};

export type StateSnapshot = {
  active_session: string;
  audiences: string[];
  counts: {
    turns: number;
    commitments: number;
    open_qs: number;
    dream_audit_rows: number;
  };
  current_mood: MoodSnapshot;
  version: string;
};

export type TurnStakes = "low" | "medium" | "high";

export type TurnRequest = {
  message: string;
  audience: string;
  stakes?: TurnStakes;
};

export type TurnResponse = {
  turn_id: string;
  ok: true;
};

export type TurnPhaseName =
  | "perception"
  | "frame"
  | "extract"
  | "retrieval"
  | "ledger"
  | "shared"
  | "delib"
  | "reflect";

export type PhaseEventData = {
  turnId: string;
  turn_id?: string;
  phase?: TurnPhaseName;
  ts?: number;
  duration_ms?: number;
  sub?: string;
  [key: string]: JsonValue | undefined;
};

export type LiveFrameBase = {
  type: string;
  ts: number;
};

export type StreamAppendFrame = LiveFrameBase & {
  type: "stream:append";
  entries: StreamEntry[];
};

export type TurnPhaseFrame = LiveFrameBase & {
  type: "turn:phase:started" | "turn:phase:completed" | "turn:phase:failed";
  event: "turn_phase.started" | "turn_phase.completed" | "turn_phase.failed";
  data: PhaseEventData;
};

export type EvidenceLedgerBuiltFrame = LiveFrameBase & {
  type: "evidence_ledger:built";
  turn_id: string;
  ledger: EvidenceLedger | null;
};

export type LiveFrame = StreamAppendFrame | TurnPhaseFrame | EvidenceLedgerBuiltFrame;

export type WsState = "live" | "reconnecting" | "down";
