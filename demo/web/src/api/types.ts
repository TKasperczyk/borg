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

export type SessionParticipationPolicy = SessionRecord["participation_policy"];

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

export type InflightPhase = {
  phase: string;
  status: "active" | "completed" | "failed";
  duration_ms: number | null;
};

export type InflightTurn = {
  turn_id: string;
  session_id: string;
  started_at: number;
  last_event_at: number;
  phases: InflightPhase[];
};

export type InflightResponse = {
  inflight: InflightTurn | null;
};

export type ActivityOrigin = "user" | "autonomous" | "dream";

export type ActivityDigest = {
  turns: number;
  autonomous_wakes: number;
  emissions: number;
  silences: number;
  observations: number;
  suppressions: number;
  dream_changes: number;
  journal_notes: number;
};

export type ActivityTurnRow = {
  id: string;
  kind: "turn";
  started_at: number;
  session_id: string;
  session_label: string | null;
  origin: "user" | "autonomous";
  trigger: string | null;
  outcome: TurnHistoryOutcome;
  suppression_reason: string | null;
  duration_ms: number | null;
  excerpt: string | null;
  turn_id: string;
};

export type ActivityDreamRow = {
  id: string;
  kind: "dream";
  started_at: number;
  session_id: string;
  session_label: string | null;
  origin: "dream";
  trigger: string | null;
  outcome: "dream";
  suppression_reason: null;
  duration_ms: number | null;
  excerpt: string | null;
  turn_id: null;
  dream: {
    run_id: string;
    process_count: number;
    changes: number;
    errors: number;
  };
};

export type ActivityRow = ActivityTurnRow | ActivityDreamRow;

export type ActivityResponse = {
  day: string;
  days: string[];
  rows: ActivityRow[];
  truncated: boolean;
  digest: ActivityDigest;
};

type AutonomyWakeSourceBase = {
  name: string;
  enabled: boolean;
  source_category: "contemplative" | "operational";
  last_fired: number | null;
  wake_count: number;
};

export type AutonomyWakeSource =
  | (AutonomyWakeSourceBase & {
      wake_source_type: "trigger";
      next_due_at: number | null;
    })
  | (AutonomyWakeSourceBase & {
      wake_source_type: "condition";
      next_due_at?: never;
    });

export type AutonomyWakeRecord = {
  id: string;
  ts: number;
  trigger_name: string;
  condition_name: string | null;
  session_id: string | null;
  session_label: string | null;
  wake_source_type: "trigger" | "condition";
  source_category: "contemplative" | "operational";
  selected_goal_id: string | null;
  outcome: "headway" | "silent" | "error" | "busy" | null;
  outcome_detail: string | null;
  headway_bases: string[] | null;
  finalizer_rounds: number | null;
  stall_retries: number | null;
};

export type AutonomyStateResponse = {
  scheduler: {
    enabled: boolean;
    interval_ms: number;
    next_tick_at: number | null;
  };
  wake_sources: AutonomyWakeSource[];
  wake_budget: {
    used: number;
    limit: number;
    window_ms: number;
  } | null;
  self_scheduled_wakes: Array<{
    id: string;
    due_at: number;
    note: string;
    created_at: number;
    status: string;
  }>;
  can_cancel_wakes: boolean;
  recent_wakes: AutonomyWakeRecord[];
};

export type JournalEntry = {
  id: number;
  self_entity_id: string;
  self_label: string | null;
  text: string;
  disclosure_class: "self_private";
  created_at: number;
  updated_at: number;
  source_turn_id: string | null;
  marker_stream_entry_id: string | null;
};

export type JournalResponse = {
  entries: JournalEntry[];
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

export type IdentityStateName = "candidate" | "established";
export type GoalStatus = "active" | "done" | "abandoned" | "blocked";
export type OpenQuestionStatus = "open" | "resolved" | "abandoned";

export type IdentityValue = {
  id: string;
  label: string;
  description: string;
  priority: number;
  state: IdentityStateName;
  confidence: number;
  created_at: number;
  last_affirmed: number | null;
  established_at: number | null;
  support_count: number;
  contradiction_count: number;
};

export type GoalBlocker =
  | { kind: "goal"; goal_id: string }
  | { kind: "entity"; entity_id: string }
  | { kind: "until"; until: number };
export type GoalBlockRecord = {
  blocker: GoalBlocker | { kind: "legacy_unknown" };
  attempt_status: "attempted_unavailable" | "not_recorded";
  reason: string;
  blocked_at: number | null;
  disclosure_label?: DisclosureLabel;
  unblocked_at: number | null;
  unblock_reason: string | null;
  attempt_evidence?: { kind: string; id: string | number };
};
export type IdentityGoal = {
  block_history?: GoalBlockRecord[];
  id: string;
  description: string;
  priority: number;
  status: GoalStatus;
  progress_notes: string | null;
  last_progress_ts: number | null;
  created_at: number;
  target_at: number | null;
  counterparty_entity_id: string | null;
  disclosure?: string;
  disclosure_label?: DisclosureLabel;
  children?: IdentityGoal[];
};

export type IdentityTrait = {
  id: string;
  label: string;
  strength: number;
  state: IdentityStateName;
  confidence: number;
  established_at: number | null;
  last_reinforced: number;
};

export type IdentityOpenQuestion = {
  id: string;
  question: string;
  urgency: number;
  status: OpenQuestionStatus;
  source: string;
  goal_id: string | null;
  created_at: number;
  last_touched: number;
  related_episode_ids: string[];
  related_semantic_node_ids: string[];
  resolution_note: string | null;
  resolved_at: number | null;
  abandoned_reason: string | null;
  abandoned_at: number | null;
};

export type IdentityGrowthMarker = {
  id: string;
  ts: number;
  category: string;
  what_changed: string;
  confidence: number;
  source_process: string;
};

export type IdentityPeriod = {
  id: string;
  record_version?: number;
  label: string;
  start_ts: number;
  end_ts: number | null;
  narrative: string;
  key_episode_ids?: string[];
  disclosure_label?: DisclosureLabel;
  themes: string[];
  provenance?: Record<string, unknown>;
  created_at: number;
  last_updated: number;
};

export type IdentityEvent = {
  id?: string | number;
  ts?: number;
  action?: string;
  record_id?: string;
  [key: string]: unknown;
};

export type IdentityResponse = {
  values: IdentityValue[];
  goals: IdentityGoal[];
  traits: IdentityTrait[];
  open_questions: IdentityOpenQuestion[];
  growth_markers: IdentityGrowthMarker[];
  periods: IdentityPeriod[];
  open_question_events: IdentityEvent[];
};

export type GoalPatchBody =
  | { action: "complete"; note?: string }
  | { action: "block"; note: string; blocker: GoalBlocker; attempt_status: "attempted_unavailable" }
  | { action: "progress"; note?: string; progress?: number };

export type IdentityValueCreateBody = {
  name: string;
  description?: string;
};

export type OpenQuestionPatchBody =
  | { action: "resolve"; resolution: string }
  | { action: "abandon"; reason: string }
  | { action: "bump"; delta?: number };

export type CreatorDirectiveStatus = "active" | "superseded" | "revoked";
export type CreatorDirectiveKind =
  | "self_identity"
  | "subject_fact"
  | "disclosure_boundary"
  | "response_policy"
  | "routing_instruction";

export type CreatorDirective = {
  id: string;
  kind: CreatorDirectiveKind;
  text: string | null;
  canonical_fact: string | null;
  operational_directive: string | null;
  activation_scope: string;
  content_scope: string;
  mention_policy: string;
  status: CreatorDirectiveStatus;
  subject_kind: string;
  subject_entity_id: string | null;
  subject_entity_name: string | null;
  priority: number;
  superseded_by_id: string | null;
  revoked_reason: string | null;
  created_at: number;
  updated_at: number;
};

export type CreatorDirectivesResponse = {
  directives: CreatorDirective[];
};

export type CommitmentState = "active" | "revoked" | "expired";
export type CommitmentEnforcement = "critical" | "advisory";

export type Commitment = {
  id: string;
  text: string;
  type: "promise" | "boundary" | "rule" | "preference";
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
  created_at: number;
  expires_at: number | null;
  expired_at: number | null;
  revoked_at: number | null;
  revoked_reason: string | null;
  superseded_by_id: string | null;
  last_reinforced_at: number;
  disclosure?: string;
  disclosure_label?: DisclosureLabel;
};

export type CommitmentsResponse = {
  commitments: Commitment[];
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
  n: string;
  name: string;
  desc: string;
  count: number;
  count_is_lower_bound?: boolean;
  stats: Array<{ k: string; v: string | number }>;
};

export type MemoryBandsResponse = {
  bands: MemoryBandSummary[];
};

export type SemanticGraphNodeStatus = "active" | "contested" | "contradicted" | "quarantined";
export type SemanticGraphEdgeType =
  | "is_a"
  | "part_of"
  | "causes"
  | "prevents"
  | "supports"
  | "contradicts"
  | "related_to"
  | "instance_of";

export type SemanticGraphNode = {
  id: string;
  label: string;
  display_label: string | null;
  status: SemanticGraphNodeStatus;
  kind: string;
  edge_count: number;
};

export type SemanticGraphEdge = {
  id: string;
  source: string;
  target: string;
  type: SemanticGraphEdgeType;
  weight: number;
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

export type LabelRef = {
  value: string;
  id: string | null;
  label: string | null;
};

export type DisclosureLabel = {
  disclosure_class?: string;
  origin_audience_entity_ids?: string[];
  private_to_entity_ids?: string[];
  public_to_entity_ids?: string[];
};

export type SemanticNodeDetail = {
  id: string;
  kind: string;
  label: string;
  display_label: string | null;
  description: string;
  domain: string | null;
  aliases: string[];
  confidence: number;
  status: SemanticGraphNodeStatus | "superseded";
  source_episode_ids: string[];
  source_count: number;
  origin_audience_refs?: LabelRef[];
  disclosure_class?: string;
  disclosure_label?: DisclosureLabel;
  created_at: number;
  updated_at: number;
};

export type SemanticNodeDetailResponse = {
  node: SemanticNodeDetail;
};

export type SemanticEdgeDetail = {
  id: string;
  from_node_id: string;
  to_node_id: string;
  relation: SemanticGraphEdgeType;
  confidence: number;
  evidence_episode_ids: string[];
  source_count: number;
  origin_audience_refs?: LabelRef[];
  disclosure_class?: string;
  disclosure_label?: DisclosureLabel;
  valid_from: number;
  valid_to: number | null;
  invalidated_at: number | null;
  invalidated_reason: string | null;
};

export type SemanticEdgeDetailResponse = {
  edge: SemanticEdgeDetail;
};

export type EpisodeDetail = {
  id: string;
  title: string;
  narrative: string;
  participants?: string[];
  participant_refs?: LabelRef[];
  location?: string | null;
  start_time: number;
  end_time: number;
  audience?: string | null;
  origin_audience_refs?: LabelRef[];
  shared?: boolean;
  disclosure_class?: string;
  disclosure_label?: DisclosureLabel;
  significance?: number;
  confidence?: number;
  tags?: string[];
  source_count?: number;
};

export type EpisodeDetailResponse = {
  episode: EpisodeDetail;
};

export type ReviewKind =
  | "contradiction"
  | "duplicate"
  | "new_insight"
  | "misattribution"
  | "temporal_drift"
  | "identity_inconsistency"
  | "correction"
  | "belief_revision"
  | "skill_split"
  | "creator_directive_reconciliation"
  | "commitment_reconciliation"
  | "relationship_claim_ungrounded";

export type ReviewResolution =
  | "keep_both"
  | "supersede"
  | "invalidate"
  | "dismiss"
  | "accept"
  | "reject"
  | "keep"
  | "weaken"
  | "archive_node"
  | "invalidate_edge";

export type ReviewRow = {
  id: number;
  kind: ReviewKind;
  refs: Record<string, unknown>;
  reason: string;
  created_at: number;
  resolved_at: number | null;
  resolution: ReviewResolution | null;
};

export type ReviewsResponse = {
  rows: ReviewRow[];
};

export type ReviewGenericPatchBody = {
  action: ReviewResolution;
  note?: string;
  winner_node_id?: string;
};

export type CreatorDirectiveReconciliationBody =
  | { action: "supersede"; survivor_id: string; reason?: string }
  | { action: "keep"; reason?: string };

export type CorrectionReviewPatchBody = {
  action: "accept" | "reject";
  note?: string;
};

export type CorrectionWhyResponse = Record<string, unknown>;

export type BandDetailResponse = {
  band: MemoryBandId;
  mode: "browse" | "search";
  query?: string;
  next_cursor?: string | null;
  items?: Array<Record<string, unknown>>;
  nodes?: SemanticNodeDetail[];
  edges?: SemanticEdgeDetail[];
  current?: MoodState | null;
  history?: MoodState[];
  counts?: Record<string, number>;
  values?: IdentityValue[];
  goals?: IdentityGoal[];
  traits?: IdentityTrait[];
  open_questions?: IdentityOpenQuestion[];
  growth_markers?: IdentityGrowthMarker[];
  periods?: IdentityPeriod[];
  open_question_events?: IdentityEvent[];
};

export type OfflineProcessName =
  | "consolidator"
  | "reflector"
  | "semantic-extractor"
  | "curator"
  | "overseer"
  | "associator"
  | "review-resolver"
  | "ruminator"
  | "self-narrator"
  | "procedural-synthesizer"
  | "belief-reviser"
  | "creator-directive-reconciler"
  | "commitment-reconciler";

export type DreamProcessRow = {
  name: OfflineProcessName;
  description: string;
  last_run_at: number | null;
  last_status: "ok" | "error" | null;
  last_audit_id: number | null;
  budget: number | null;
  enabled: boolean;
};

export type DreamScheduleRow = {
  process: OfflineProcessName;
  scheduled_at: number;
  source: "stream" | "audit";
  stream_entry_id?: string;
  audit_id?: number;
};

export type DreamReportError = {
  process?: string;
  message?: string;
  code?: string;
  target_type?: string;
  target_id?: string;
};

export type DreamReport = {
  run_id: string;
  processes: OfflineProcessName[];
  dry_run: boolean;
  planned_at: number | null;
  changes: number;
  tokens_used: number;
  errors: DreamReportError[];
  budget_exhausted_processes: OfflineProcessName[];
  notes: string[];
};

export type MaintenanceAuditRow = {
  id: number;
  run_id: string;
  process: string;
  action: string;
  targets: string[];
  reversal: Record<string, unknown>;
  applied_at: number;
  reverted_at: number | null;
  reverted_by: string | null;
};

export type DreamSchedulerConfig = {
  enabled: boolean;
  light_interval_ms: number;
  heavy_interval_ms: number;
  optimize_storage: boolean;
  light_processes: OfflineProcessName[];
  heavy_processes: OfflineProcessName[];
  process_budgets: Partial<Record<OfflineProcessName, number>>;
};

export type DreamStateResponse = {
  processes: DreamProcessRow[];
  pending_extraction_episodes: number;
  schedule: DreamScheduleRow[];
  dream_reports: DreamReport[];
  audit_rows: MaintenanceAuditRow[];
  belief_revision_rows: ReviewRow[];
  scheduler: DreamSchedulerConfig;
};

export type DreamAuditResponse = {
  rows: MaintenanceAuditRow[];
};

export type DreamPlanProcessPreview = {
  name: OfflineProcessName;
  would_change: boolean;
  summary: string;
  budget_used: number;
  changes: unknown[];
  errors: unknown[];
  budget_exhausted: boolean;
};

export type DreamPlanResponse = {
  plan_id: string;
  processes: DreamPlanProcessPreview[];
  total_budget_used: number;
  changes: number;
};

export type DreamApplyResponse = {
  run_id: string;
  applied: Array<{
    name: OfflineProcessName;
    audit_id: number | null;
    audit_ids: number[];
    changes: number;
  }>;
  failed: Array<{
    name: OfflineProcessName;
    message: string;
    code?: string;
  }>;
  duration_ms: number;
  total_budget_used: number;
};

export type PromptBlock = {
  key: string;
  label: string;
  description: string;
  default_text: string;
  current_text: string;
  current_text_kind: "static_default" | "runtime_composed" | "stored_override";
  overridden: boolean;
  updated_at: number | null;
};

export type PromptsResponse = {
  blocks: PromptBlock[];
};

export type AssembledPromptSegment = {
  id: string;
  label: string;
  editable_key: string | null;
  start: number;
  end: number;
};

export type AssembledPromptResponse = {
  text: string;
  sections: string[];
  segments: AssembledPromptSegment[];
};

export type EntityRecord = {
  id: string;
  canonical_name: string;
  aliases: string[];
  kind: "person" | "group" | "self" | "abstract" | null;
  borg_role: "creator" | null;
  name_provenance?: string;
  created_at: number;
  [key: string]: unknown;
};

export type AdminResetResponse = {
  ok: boolean;
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
