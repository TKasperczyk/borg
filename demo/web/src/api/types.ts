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
  entry_index?: number;
  kind: StreamEntryKind;
  content: unknown;
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
    source_entry_ids?: string[];
    [key: string]: unknown;
  };
  persistence_class?: "assistant_self_report";
  session_id: string;
  compressed: boolean;
};

export type StreamResponse = {
  entries: StreamEntry[];
  next_cursor: string | null;
};

export type AttachmentRecord = {
  attachment_id: string;
  sha256: string;
  media_type: string;
  byte_size: number;
  width: number;
  height: number;
  storage_ref: string;
  thumbnail_ref: string | null;
  perception_id: string | null;
  text_embedding_ref: string | null;
  visual_embedding_ref: string | null;
  active: boolean;
  audience: string | null;
  created_turn_global: number | null;
  parent_entry_id: string;
  stream_entry_id: string | null;
  parent_turn_id: string;
  created_at: number;
};

export type ImagePerceptionRecord = {
  perception_id: string;
  payload_id: string;
  attachment_id: string;
  caption: string;
  image_kind: string;
  active: boolean;
  audience: string | null;
  visible_text: string[];
  objects: string[];
  people_or_roles: string[];
  scene: string;
  colors_and_visual_attributes: string[];
  spatial_relationships: string[];
  possible_user_relevant_details: string[];
  search_terms: string[];
  uncertainties: string[];
  embedding_status: "pending" | "complete" | "failed";
};

export type AttachmentMetadataResponse = {
  attachment: AttachmentRecord;
  perception: ImagePerceptionRecord | null;
  status: {
    active: boolean;
    quarantined: boolean;
    stream_active?: boolean;
    parent_active?: boolean;
  };
};

export type AttachmentStatusItem = {
  id: string;
  status: AttachmentMetadataResponse["status"];
};

export type MemoryBandId =
  | "episodic"
  | "semantic"
  | "procedural"
  | "affective"
  | "self"
  | "commitments"
  | "social"
  | "relational";

export type MemoryBandSummary = {
  id: MemoryBandId;
  n?: string;
  name: string;
  desc?: string;
  count: number;
  growth?: number[];
  stats: Array<{ k: string; v: number | string }>;
};

export type MemoryBandsResponse = {
  bands: MemoryBandSummary[];
};

export type EpisodeMemoryItem = {
  id: string;
  title: string;
  narrative: string;
  participants: string[];
  location: string | null;
  start_time: number;
  end_time: number;
  audience: string | null;
  significance: number;
  confidence: number;
  tags: string[];
  source_stream_ids: string[];
  source_count: number;
  lineage: { derived_from: string[]; supersedes: string[] };
  emotional_arc: unknown | null;
  vector_dims: number;
  created_at: number;
  updated_at: number;
};

export type SemanticMemoryNode = {
  id: string;
  kind: "concept" | "entity" | "proposition";
  label: string;
  description: string;
  domain: string | null;
  aliases: string[];
  confidence: number;
  status: "active" | "superseded" | "contradicted" | "quarantined";
  source_episode_ids: string[];
  source_count: number;
  created_at: number;
  updated_at: number;
};

export type SemanticMemoryEdge = {
  id: string;
  from_node_id: string;
  to_node_id: string;
  relation: string;
  confidence: number;
  evidence_episode_ids: string[];
  source_count: number;
  valid_from: number;
  valid_to: number | null;
  invalidated_at: number | null;
  invalidated_by_edge_id: string | null;
  invalidated_by_review_id: number | null;
  invalidated_by_process: string | null;
  invalidated_reason: string | null;
};

export type SemanticGraphNodeStatus = "active" | "contested" | "contradicted" | "quarantined";

export type SemanticGraphNode = {
  id: string;
  label: string;
  status: SemanticGraphNodeStatus;
  kind?: string;
  salience?: number;
  edge_count: number;
};

export type SemanticGraphEdge = {
  id: string;
  source: string;
  target: string;
  type: string;
  weight?: number;
};

export type SemanticGraphResponse = {
  nodes: SemanticGraphNode[];
  edges: SemanticGraphEdge[];
  total_nodes: number;
  total_edges: number;
  rendered: {
    nodes: number;
    edges: number;
  };
};

export type ProceduralMemoryItem = {
  id: string;
  applies_when: string;
  approach: string;
  status: "active" | "superseded";
  alpha: number;
  beta: number;
  attempts: number;
  successes: number;
  failures: number;
  sample_count: number;
  source_episode_ids: string[];
  last_used: number | null;
  last_successful: number | null;
  requires_manual_review: boolean;
  created_at: number;
  updated_at: number;
};

export type MoodHistoryEntry = {
  id: number;
  session_id: string;
  ts: number;
  valence: number;
  arousal: number;
  trigger_reason: string | null;
  provenance: Record<string, unknown>;
};

export type IdentityValue = {
  id: string;
  label: string;
  description: string;
  priority: number;
  created_at: number;
  last_affirmed: number | null;
  state: "candidate" | "established";
  confidence: number;
  support_count: number;
  contradiction_count: number;
  evidence_episode_ids: string[];
};

export type IdentityGoal = {
  id: string;
  description: string;
  priority: number;
  status: "active" | "done" | "abandoned" | "blocked";
  progress_notes: string | null;
  created_at: number;
  target_at: number | null;
};

export type IdentityTrait = {
  id: string;
  label: string;
  strength: number;
  state: "candidate" | "established";
  confidence: number;
  support_count: number;
  contradiction_count: number;
  evidence_episode_ids: string[];
};

export type OpenQuestion = {
  id: string;
  question: string;
  urgency: number;
  status: "open" | "resolved" | "abandoned";
  goal_id: string | null;
  source: string;
  created_at: number;
  last_touched: number;
  resolved_at: number | null;
  abandoned_at: number | null;
  abandoned_reason: string | null;
  resolution_note: string | null;
  unresolved_rumination_ticks: number;
  last_ruminated_at: number | null;
};

export type GrowthMarker = {
  id: string;
  ts: number;
  category: string;
  what_changed: string;
  before_description: string | null;
  after_description: string | null;
  evidence_episode_ids: string[];
  confidence: number;
  source_process: string;
  created_at: number;
};

export type AutobiographicalPeriod = {
  id: string;
  label: string;
  start_ts: number;
  end_ts: number | null;
  narrative: string;
  key_episode_ids: string[];
  themes: string[];
  created_at: number;
  last_updated: number;
};

export type IdentityEvent = {
  id: number;
  record_type: string;
  record_id: string;
  action: string;
  old_value: unknown | null;
  new_value: unknown | null;
  reason: string | null;
  review_item_id: number | null;
  overwrite_without_review: boolean;
  ts: number;
};

export type IdentityResponse = {
  values: IdentityValue[];
  goals: IdentityGoal[];
  traits: IdentityTrait[];
  open_questions: OpenQuestion[];
  growth_markers: GrowthMarker[];
  periods: AutobiographicalPeriod[];
  open_question_events: IdentityEvent[];
};

export type CommitmentState = "active" | "revoked" | "expired";
export const COMMITMENT_CREATE_TYPES = ["rule", "preference", "boundary"] as const;
export const COMMITMENT_KINDS = [
  "assistant_commitment",
  "audience_rule",
  "participant_preference",
  "boundary",
  "process_norm",
] as const;
export type CommitmentCreateType = (typeof COMMITMENT_CREATE_TYPES)[number];
export type CommitmentKind = (typeof COMMITMENT_KINDS)[number];
export type CommitmentEnforcement = "critical" | "advisory";

export type CommitmentItem = {
  id: string;
  text: string;
  type: string;
  kind: string;
  enforcement_class: CommitmentEnforcement;
  critical_domain: string | null;
  state: CommitmentState;
  priority: number;
  directive_family: string;
  audience: string | null;
  made_to: string | null;
  about: string | null;
  committed_by: string | null;
  source: string;
  source_stream_entry_ids: string[];
  created_at: number;
  expires_at: number | null;
  expired_at: number | null;
  revoked_at: number | null;
  revoked_reason: string | null;
  superseded_by_id: string | null;
  canonicalized_by_artifact_entry_id: string | null;
  last_reinforced_at: number;
};

export type CommitmentsResponse = {
  commitments: CommitmentItem[];
};

export type CreatorDirectiveKind =
  | "self_identity"
  | "subject_fact"
  | "disclosure_boundary"
  | "response_policy"
  | "routing_instruction";

export type CreatorDirectiveStatus = "active" | "superseded" | "revoked";

export type CreatorDirectiveSubjectKind = "borg_self" | "entity" | "system" | "unknown";

export type CreatorDirectiveActivationScope =
  | "same_as_disclosure"
  | "operator_only"
  | "public"
  | "allow_list"
  | "subject_only"
  | "all_except";

export type CreatorDirectiveContentScope =
  | "operator_only"
  | "public"
  | "allow_list"
  | "subject_only"
  | "all_except";

export type CreatorDirectiveMentionPolicy =
  | "proactive"
  | "answer_if_asked"
  | "only_if_topic_raised"
  | "never_mention";

export type CreatorDirectiveItem = {
  id: string;
  kind: CreatorDirectiveKind;
  text: string | null;
  canonical_fact: string | null;
  operational_directive: string | null;
  activation_scope: CreatorDirectiveActivationScope;
  activation_allowed_entity_ids: string[];
  activation_excluded_entity_ids: string[];
  content_scope: CreatorDirectiveContentScope;
  mention_policy: CreatorDirectiveMentionPolicy;
  status: CreatorDirectiveStatus;
  subject_kind: CreatorDirectiveSubjectKind;
  subject_entity_id: string | null;
  subject_entity_name: string | null;
  priority: number;
  created_at: number;
};

export type CreatorDirectivesResponse = {
  directives: CreatorDirectiveItem[];
};

export type SocialMemoryItem = {
  entity_id: string;
  name: string | null;
  trust: number;
  attachment: number;
  interaction_count: number;
  history_count: number;
  commitment_count: number;
  last_interaction_at: number | null;
  updated_at: number;
};

export type RelationalMemoryItem = {
  id: string;
  slot: string;
  subject_entity_id: string;
  subject: string | null;
  slot_key: string;
  value: string;
  state: "established" | "contested" | "quarantined" | "revoked";
  sources_count: number;
  contradicted_count: number;
  alternate_count: number;
  name_provenance: string;
  created_at: number;
  updated_at: number;
};

export type MemoryBandDetail =
  | { band: "episodic"; items: EpisodeMemoryItem[]; nextCursor: string | null }
  | { band: "semantic"; nodes: SemanticMemoryNode[]; edges: SemanticMemoryEdge[] }
  | { band: "procedural"; items: ProceduralMemoryItem[] }
  | { band: "affective"; current: MoodSnapshot; history: MoodHistoryEntry[] }
  | ({ band: "self" } & IdentityResponse)
  | { band: "commitments"; items: CommitmentItem[] }
  | { band: "social"; items: SocialMemoryItem[] }
  | {
      band: "relational";
      counts: Record<string, number>;
      items: RelationalMemoryItem[];
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

export type MaintenanceAuditRow = {
  id: number;
  run_id: string;
  process: string;
  action: string;
  targets: Record<string, unknown>;
  reversal: Record<string, unknown>;
  applied_at: number;
  reverted_at: number | null;
  reverted_by: string | null;
};

export type ReviewRow = {
  id: number;
  kind: string;
  refs: Record<string, unknown>;
  reason: string;
  created_at: number;
  resolved_at: number | null;
  resolution: string | null;
};

export type WhyResponse = Record<string, unknown>;

export type CorrectionForgetResponse = {
  id: string;
  target_type: string;
  archived: true;
  provenance: { kind: "manual" };
};

export type CorrectionReviewsResponse = {
  rows: ReviewRow[];
};

export type CorrectMemoryRequest = {
  patch: Record<string, unknown>;
  reason?: string;
};

export type InvalidateSemanticEdgeRequest = {
  at?: number;
  reason?: string;
};

export type PatchCorrectionReviewRequest = {
  action: "accept" | "reject";
  note?: string;
};

export type DreamProcessName =
  | "consolidator"
  | "reflector"
  | "semantic-extractor"
  | "curator"
  | "overseer"
  | "review-resolver"
  | "ruminator"
  | "self-narrator"
  | "procedural-synthesizer"
  | "belief-reviser";

export type DreamProcessSummary = {
  name: DreamProcessName;
  description: string;
  last_run_at: number | null;
  last_status: "ok" | "error" | null;
  last_audit_id: number | null;
  budget: number | null;
  enabled: boolean;
};

export type DreamScheduleItem = {
  process: DreamProcessName;
  scheduled_at: number;
  source: "audit" | "scheduler" | "stream";
  audit_id?: number;
  stream_entry_id?: string;
};

export type DreamStateResponse = {
  processes: DreamProcessSummary[];
  schedule: DreamScheduleItem[];
  audit_rows: MaintenanceAuditRow[];
  belief_revision_rows: ReviewRow[];
  scheduler: {
    enabled: boolean;
    light_interval_ms: number;
    heavy_interval_ms: number;
    light_processes: DreamProcessName[];
    heavy_processes: DreamProcessName[];
    process_budgets: Partial<Record<DreamProcessName, number | null>>;
  };
};

export type DreamAuditResponse = {
  rows: MaintenanceAuditRow[];
};

export type DreamPlanRequest = {
  processes?: DreamProcessName[];
  budget?: number;
};

export type DreamPlanProcess = {
  name: DreamProcessName;
  would_change: boolean;
  summary: string;
  budget_used: number;
  changes: Array<{
    process: DreamProcessName;
    action: string;
    targets: Record<string, unknown>;
    preview?: Record<string, unknown>;
  }>;
  errors: Array<{
    process: DreamProcessName;
    message: string;
    code?: string;
    target_type?: string;
    target_id?: string;
  }>;
  budget_exhausted: boolean;
};

export type DreamPlanResponse = {
  plan_id: string;
  processes: DreamPlanProcess[];
  total_budget_used: number;
  changes: number;
};

export type DreamApplyRequest = DreamPlanRequest & {
  plan_id?: string;
};

export type DreamApplyResponse = {
  run_id: string;
  applied: Array<{
    name: DreamProcessName;
    audit_id: number | null;
    audit_ids: number[];
    changes: number;
  }>;
  failed: Array<{
    name: DreamProcessName;
    message: string;
    code?: string;
  }>;
  duration_ms: number;
  total_budget_used: number;
};

export type CreateValueRequest = {
  name: string;
  description?: string;
};

export type CreateGoalRequest = {
  description: string;
  priority?: number;
};

export type CreateCommitmentRequest = {
  type: CommitmentCreateType;
  kind: CommitmentKind;
  directive: string;
  priority: number;
  audience?: string;
  made_to?: string;
  about?: string;
  directive_family?: string;
  expires_at?: number;
};

export type RevokeCommitmentRequest = {
  reason?: string;
};

export type PatchGoalRequest =
  | { action: "complete"; note?: string }
  | { action: "block"; note?: string }
  | { action: "progress"; note?: string; progress?: number };

export type CreateGrowthMarkerRequest = {
  description: string;
  source?: string;
};

export type PatchOpenQuestionRequest =
  | { action: "resolve"; resolution: string }
  | { action: "abandon"; reason: string }
  | { action: "bump"; delta?: number };

export type PatchReviewItemRequest = {
  action: "dismiss";
  note?: string;
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

export type SessionSourceType = "demo" | "slack" | "discord" | "imessage" | "autonomy";

export type ConversationKind = "dm" | "channel" | "thread" | "demo";

export type SessionStatus = "active" | "idle" | "archived";

export type SessionPrivacyLevel = "payload_off" | "payload_on";

export type SessionParticipationPolicy = "active" | "paused" | "observing" | "muted";
export type SessionAudienceRole = "participant" | "operator";

export type EntityBorgRole = "creator" | null;

export type EntityRecord = {
  id: string;
  canonical_name: string;
  aliases: string[];
  kind: "person" | "group" | "self" | "abstract" | null;
  borg_role: EntityBorgRole;
  name_provenance?: string;
  created_at: number;
};

export type SessionRecord = {
  session_id: string;
  source_type: SessionSourceType;
  source_external_id: string | null;
  source_url: string | null;
  label: string;
  audience_label: string;
  audience_entity_id: string | null;
  conversation_kind: ConversationKind;
  created_at: number;
  last_activity_at: number;
  last_turn_id: string | null;
  message_count: number;
  status: SessionStatus;
  privacy_level: SessionPrivacyLevel;
  participation_policy: SessionParticipationPolicy;
  audience_role: SessionAudienceRole;
};

export type SessionsResponse = {
  sessions: SessionRecord[];
};

export type TurnRequest = {
  message: string;
  external_message_id: string;
  audience: string;
  audience_entity_id?: string | null;
  session?: string;
  attachments?: readonly File[];
};

export type TurnResponse = {
  ok: boolean;
  status: "enqueued" | "duplicate";
  stream_entry_id: string;
};

export type TurnPhaseName =
  | "ingest"
  | "audience"
  | "perception"
  | "frame"
  | "extract"
  | "closure_loop"
  | "generation_gate"
  | "retrieval"
  | "ledger"
  | "shared"
  | "delib"
  | "final"
  | "guards"
  | "persist"
  | "reflect";

export type PhaseEventData = {
  turnId: string;
  turn_id?: string;
  session_id?: string;
  phase?: TurnPhaseName;
  ts?: number;
  duration_ms?: number;
  sub?: string;
  [key: string]: JsonValue | undefined;
};

export type TurnTerminalOutcome =
  | "reflected"
  | "suppressed_closure"
  | "suppressed_generation_gate"
  | "suppressed_action"
  | "aborted"
  | "error";

export type TurnTerminalData = {
  turnId: string;
  turn_id?: string;
  session_id?: string;
  outcome: TurnTerminalOutcome;
  ts?: number;
  duration_ms?: number;
  [key: string]: JsonValue | undefined;
};

export type LiveFrameBase = {
  type: string;
  ts: number;
  session_id?: string;
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

export type TurnTerminalFrame = LiveFrameBase & {
  type: "turn:terminal";
  event: "turn.terminal";
  data: TurnTerminalData;
};

export type LiveTokenFrame = LiveFrameBase & {
  type: "turn:token";
  turn_id: string;
  phase: TurnPhaseName;
  chunk_text: string;
  sequence: number;
};

export type LiveTokenFlushFrame = LiveFrameBase & {
  type: "turn:token:flush";
  turn_id: string;
  phase: TurnPhaseName;
  full_text: string;
};

export type EvidenceLedgerBuiltFrame = LiveFrameBase & {
  type: "evidence_ledger:built";
  turn_id: string;
  ledger: EvidenceLedger | null;
};

export type TurnDelibPathFrame = LiveFrameBase & {
  type: "turn:delib_path";
  turn_id: string;
  path: "system_1" | "system_2";
};

export type TurnFinalAttemptFrame = LiveFrameBase & {
  type: "turn:final_attempt";
  turn_id: string;
  attempt: number;
};

export type TurnPhaseDetailFrame = LiveFrameBase & {
  type: "turn:phase:detail";
  turn_id: string;
  phase?: string;
  event: string;
  summary: string;
};

export type DreamProcessStartedFrame = LiveFrameBase & {
  type: "dream:process:started";
  process: string;
  run_id: string | null;
  phase: "plan" | "apply";
};

export type DreamProcessCompletedFrame = LiveFrameBase & {
  type: "dream:process:completed";
  process: string;
  run_id: string | null;
  phase: "plan" | "apply";
  duration_ms?: number;
  errors: number;
  candidates_accepted: number;
};

export type BorgResetFrame = LiveFrameBase & {
  type: "borg:reset";
};

export type LiveFrame =
  | StreamAppendFrame
  | TurnPhaseFrame
  | LiveTokenFrame
  | LiveTokenFlushFrame
  | TurnTerminalFrame
  | EvidenceLedgerBuiltFrame
  | TurnDelibPathFrame
  | TurnFinalAttemptFrame
  | TurnPhaseDetailFrame
  | DreamProcessStartedFrame
  | DreamProcessCompletedFrame
  | BorgResetFrame;

export type WsState = "live" | "reconnecting" | "down";

export type PromptKey =
  | "base_identity_preamble"
  | "voice_and_posture"
  | "epistemic_posture"
  | "identity_posture"
  | "host_capabilities";

export type PromptBlockView = {
  key: PromptKey;
  label: string;
  description: string;
  default_text: string;
  current_text: string;
  overridden: boolean;
  updated_at: number | null;
};

export type PromptBlocksResponse = {
  blocks: PromptBlockView[];
};
