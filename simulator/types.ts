import type {
  ActionCandidateClassification,
  ActionRecordCreationSource,
  ActionState,
  GenerationSuppressionReason,
  GoalPromotionClassification,
  RelationalSlotState,
  ReviewKind,
  SemanticNodeStatus,
  SessionId,
  CommitmentKind,
} from "../src/index.js";

export type GoalPromotionClassificationMetricKey =
  | GoalPromotionClassification
  | "invalid_classification";
export type ActionCandidateClassificationMetricKey =
  | ActionCandidateClassification
  | "invalid_classification";

export type Persona = {
  key: string;
  displayName: string;
  systemPrompt: string;
  seedFacts?: string[];
};

export type SimulatorScenarioDefinition = {
  key: string;
  description: string;
  channelName: string;
  personas: readonly Persona[];
};

export type MetricsRow = {
  event: "turn_metrics" | "aborted_turn" | "aborted_attempt";
  ts: number;
  turn_counter: number;
  turnId: string;
  transport_chat_attempts: number;
  failure_reason?: string;
  episode_count: number;
  semantic_node_count: number;
  semantic_node_count_by_status: Record<SemanticNodeStatus, number>;
  semantic_edge_count: number;
  semantic_nodes_added_since_last_check: number;
  semantic_edges_added_since_last_check: number;
  open_question_count: number;
  active_goal_count: number;
  generation_suppression_count: number;
  mood_valence: number;
  mood_arousal: number;
  retrieval_latency_ms: number | null;
  deliberation_latency_ms: number | null;
  borg_input_tokens: number;
  borg_output_tokens: number;
  // JSONL key order is part of the simulator metrics contract; new 6d-9 fields append here.
  open_question_resolved_count: number;
  action_record_count_total: number;
  action_record_count_by_state: Record<ActionState, number>;
  action_record_count_committed_to_do: number;
  action_record_count_canonicalized: number;
  action_record_count_active: number;
  borg_owned_active_actions: number;
  participant_owned_active_actions: number;
  group_owned_active_actions: number;
  action_record_creation_source_per_turn: Record<ActionRecordCreationSource, number>;
  action_record_creation_count_this_turn: number;
  action_candidate_classifications_per_turn: Record<ActionCandidateClassificationMetricKey, number>;
  action_candidate_rejected_classification: number;
  action_persistence_dedup_skipped_embedding: number;
  action_persistence_dedup_degraded: number;
  actions_closed_by_terminal_emission: number;
  actions_rejected_capability: number;
  actions_canonicalized: number;
  actions_completed_via_canonicalization: number;
  recent_completed_action_count: number;
  commitment_count_active: number;
  commitment_count_active_by_kind: Record<CommitmentKind, number>;
  commitment_count_superseded: number;
  // These lifecycle counts are not mutually exclusive: commitments canonicalized
  // through the shared state are revoked by design.
  commitment_count_revoked: number;
  commitment_count_expired: number;
  commitment_count_canonicalized: number;
  pending_action_count: number;
  pending_action_merge_count: number;
  relational_slot_count_by_state: Record<RelationalSlotState, number>;
  review_queue_open_count_by_type: Record<ReviewKind, number>;
  review_resolver_attempted?: number;
  review_resolver_accepted?: number;
  review_resolver_dismissed?: number;
  review_resolver_rejected?: number;
  review_resolver_needs_manual?: number;
  review_queue_enqueued_this_turn?: number;
  review_queue_resolved_this_turn?: number;
  review_queue_drain_rate?: number | null;
  frame_anomaly_classifier_calls: number;
  frame_anomaly_classified_normal_count: number;
  frame_anomaly_actual_anomaly_count: number;
  frame_anomaly_degraded_count: number;
  frame_anomaly_degraded_fallback_match_count: number;
  quarantined_user_entry_count: number;
  early_extractors_skipped_frame_anomaly_count: number;
  goal_promotion_salvaged_promotions: number;
  goal_promotion_skipped_promotions: number;
  goal_promotion_initial_step_downgraded: number;
  goal_promotion_dedup_skipped_extractor_signal: number;
  goal_promotion_dedup_skipped_embedding: number;
  goal_promotion_dedup_degraded: number;
  goal_promotion_classifications_per_turn: Record<GoalPromotionClassificationMetricKey, number>;
  goal_promotion_rejected_classification: number;
  goal_promotion_cap_rejections: number;
  decision_artifact_semantic_revisions_attempted: number;
  decision_artifact_semantic_revisions_completed_succeeded: number;
  decision_artifact_semantic_nodes_marked_superseded: number;
  decision_artifact_semantic_nodes_marked_contradicted: number;
  decision_artifact_semantic_revision_cache_hits: number;
  decision_artifact_semantic_revision_cache_size: number;
  // True when a checkpoint was scheduled on a suppressed turn; an actual run
  // is represented by that turn's overseer verdict.
  overseer_due_on_suppressed_turn: boolean;
};

export type SimulatorHealthWarningKind =
  | "active_goals_high"
  | "active_goals_growth_high"
  | "active_actions_final_high"
  | "committed_to_do_actions_final_high"
  | "actions_per_turn_high"
  | "action_canonicalization_rate_low"
  | "retrieval_latency_max_high"
  | "deliberation_latency_max_high"
  | "semantic_revision_llm_calls_high"
  | "semantic_revision_transition_yield_low"
  | "classifier_degraded_rate_high"
  | "capability_overclaim_count_high"
  | "review_queue_backlog_high";

export type SimulatorHealthWarning = {
  kind: SimulatorHealthWarningKind;
  turn_counter: number;
  turnId: string;
  threshold: number;
  observed_value: number;
  window_start_turn?: number;
  window_turns?: number;
};

export type OverseerVerdict = {
  ts: number;
  turn_counter: number;
  status: "healthy" | "concerning" | "failing";
  observations: string[];
  recommendation: string;
  findings: OverseerFinding[];
  rejected_findings: RejectedOverseerFinding[];
  raw_verdict: RawOverseerVerdict;
};

export type OverseerFindingCategory =
  | "A"
  | "B"
  | "C"
  | "D"
  | "E"
  | "F"
  | "G"
  | "H"
  | "I"
  | "J"
  | "K";

export type OverseerClaimStatus = "grounded" | "unsupported" | "contradicted" | "unclear";

export type OverseerFindingSourceKind =
  | "emitted_output"
  | "prompt_visible_memory"
  | "snapshot_memory";

export type OverseerFindingStatusImpact = "none" | "concerning" | "failing";

export type OverseerTemporalDirection =
  | "claim_before_evidence"
  | "claim_after_evidence"
  | "claim_simultaneous";

export type OverseerFinding = {
  category: OverseerFindingCategory;
  claim_status: OverseerClaimStatus;
  source_kind: OverseerFindingSourceKind;
  status_impact?: OverseerFindingStatusImpact;
  assistant_stream_entry_id?: string;
  assistant_ts?: number;
  metrics_turn_counter?: number;
  quoted_emitted_span?: string;
  cited_evidence_stream_ids?: string[];
  cited_evidence_ts?: number[];
  temporal_direction?: OverseerTemporalDirection;
  evidence_summary: string;
  carryover_demoted?: boolean;
  carryover_original_status_impact?: OverseerFindingStatusImpact;
  carryover_cached_status_impact?: OverseerFindingStatusImpact;
  carryover_cached_stream_entry_id?: string;
  carryover_cached_at_turn?: number;
};

export type RejectedOverseerFinding = OverseerFinding & {
  validation_warning: string;
};

export type RawOverseerVerdict = {
  status: "healthy" | "concerning" | "failing";
  observations: string[];
  recommendation: string;
  findings: OverseerFinding[];
};

export type SimulatorResultState = "completed" | "max_sessions_reached";

export type SimulatorSessionRecord = {
  sessionIndex: number;
  sessionId: SessionId;
  startedAtTurn: number;
  endedAtTurn: number;
  endReason: "suppression" | "run_complete";
  suppressionReason?: GenerationSuppressionReason;
};

export type SimulatorSuppressionRecord = {
  sessionIndex: number;
  sessionId: SessionId;
  turn: number;
  reason: GenerationSuppressionReason;
};

export type SimulatorRunReport = {
  runId: string;
  persona: string;
  personas: string[];
  audience: string;
  totalTurns: number;
  resultState: SimulatorResultState;
  sessions: SimulatorSessionRecord[];
  suppressionEvents: SimulatorSuppressionRecord[];
  overseerCheckpoints: OverseerVerdict[];
  healthWarnings?: SimulatorHealthWarning[];
  turnFailures: Array<{ turn: number; error: string; attempts: number }>;
  finalMetrics: MetricsRow;
  durationMs: number;
};
