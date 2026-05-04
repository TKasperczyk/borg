import type {
  ActionState,
  GenerationSuppressionReason,
  RelationalSlotState,
  ReviewKind,
  SessionId,
} from "../src/index.js";

export type Persona = {
  key: string;
  displayName: string;
  systemPrompt: string;
  seedFacts?: string[];
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
  recent_completed_action_count: number;
  commitment_count_active: number;
  commitment_count_superseded: number;
  pending_action_count: number;
  pending_action_merge_count: number;
  relational_slot_count_by_state: Record<RelationalSlotState, number>;
  review_queue_open_count_by_type: Record<ReviewKind, number>;
};

export type OverseerVerdict = {
  ts: number;
  turn_counter: number;
  status: "healthy" | "concerning" | "failing";
  observations: string[];
  recommendation: string;
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
  totalTurns: number;
  resultState: SimulatorResultState;
  sessions: SimulatorSessionRecord[];
  suppressionEvents: SimulatorSuppressionRecord[];
  overseerCheckpoints: OverseerVerdict[];
  turnFailures: Array<{ turn: number; error: string; attempts: number }>;
  finalMetrics: MetricsRow;
  durationMs: number;
};
