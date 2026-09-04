import type { RecallQueryPlan } from "../../src/retrieval/recall-expansion.js";

import type {
  EmbeddingCallRecord,
  ErrorSummary,
  JudgeRating,
  RankMetrics,
} from "../embedding-ab/types.js";

export type RecallPlannerConversation = {
  type: "personal" | "groupChat" | "channel";
  name: string;
};

export type RecallPlannerCase = {
  id: string;
  focus: string;
  context_turns: Array<{
    role: "user" | "assistant";
    content: string;
  }>;
  identity: {
    memory_owner_name: string;
    current_sender_name?: string;
    current_audience_name?: string;
    current_venue?: RecallPlannerConversation;
    entity_terms?: string[];
  };
  owner_recent_activity: Array<{
    excerpt: string;
    occurred_at: number;
    venue: RecallPlannerConversation;
    counterparty_name?: string;
  }>;
  expected_episode_ids: string[];
  notes?: string;
};

export type RecallPlannerConfiguration = {
  id: string;
  label: string;
  mode: "baseline_raw_focus_only" | "planner";
  semantic_variant_count: number | null;
};

export type PlannerInvocationRecord = {
  sequence: number;
  cache_key: string;
  cache_hit: boolean;
  started_at: string;
  latency_ms: number | null;
  outcome: "pending" | "success" | "error";
  error?: ErrorSummary;
};

export type EmbeddingLogicalCallRecord = {
  sequence: number;
  text_sha256: string;
  source: "disk_cache" | "pending_cache" | "gateway";
  started_at: string;
  latency_ms: number | null;
  outcome: "pending" | "success" | "error";
  error?: ErrorSummary;
};

export type RecallPlannerLaneRank = {
  intent_id: string;
  intent_kind: string;
  intent_source: string;
  intent_priority: number;
  intent_query: string | null;
  candidate_count: number;
  candidates: Array<{
    rank: number;
    episode_id: string;
    score: number | null;
    vector_score: number | null;
  }>;
  expected_ranks: Record<string, number | null>;
};

export type RecallPlannerTopResult = {
  rank: number;
  episode_id: string;
  title: string;
  score: number;
  raw_score: number;
  vector_similarity: number;
};

export type RecallPlannerCompletedCaseRun = {
  status: "completed";
  case_id: string;
  configuration_id: string;
  planner_output: RecallQueryPlan | null;
  planner_latency_ms: number | null;
  planner_cache_hit: boolean | null;
  lane_ranks: RecallPlannerLaneRank[];
  expected_final_ranks: Record<string, number | null>;
  best_expected_rank: number | null;
  top_10: RecallPlannerTopResult[];
  embedding: {
    logical_call_count: number;
    disk_cache_hit_count: number;
    pending_cache_hit_count: number;
    gateway_logical_call_count: number;
    gateway_attempt_count: number;
    gateway_error_count: number;
    gateway_timeout_count: number;
    logical_calls: EmbeddingLogicalCallRecord[];
    gateway_attempts: EmbeddingCallRecord[];
  };
  degradations: Array<{
    subsystem: string;
    reason: string;
  }>;
  visibility: {
    audience_entity_id: string | null;
    visible_audience_entity_ids: string[];
    unresolved_names: string[];
  };
};

export type RecallPlannerFailedCaseRun = {
  status: "failed";
  case_id: string;
  configuration_id: string;
  error: ErrorSummary;
  planner_output: RecallQueryPlan | null;
  planner_latency_ms: number | null;
  planner_cache_hit: boolean | null;
  lane_ranks: RecallPlannerLaneRank[];
  expected_final_ranks: Record<string, null>;
  best_expected_rank: null;
  top_10: [];
  embedding: RecallPlannerCompletedCaseRun["embedding"];
  degradations: RecallPlannerCompletedCaseRun["degradations"];
  visibility: RecallPlannerCompletedCaseRun["visibility"];
};

export type RecallPlannerCaseRun = RecallPlannerCompletedCaseRun | RecallPlannerFailedCaseRun;

export type RecallPlannerConfigurationSummary = {
  configuration_id: string;
  metrics: RankMetrics;
  completed_case_count: number;
  failed_case_count: number;
  degraded_case_count: number;
  planner_latency: {
    measured_count: number;
    cache_hit_count: number;
    cache_miss_count: number;
    p50_ms: number | null;
    p95_ms: number | null;
    max_ms: number | null;
  };
  embedding: {
    logical_calls_per_query: number | null;
    gateway_attempts_per_query: number | null;
    gateway_latency: {
      measured_count: number;
      p50_ms: number | null;
      p95_ms: number | null;
      max_ms: number | null;
    };
    disk_cache_hit_count: number;
    pending_cache_hit_count: number;
    gateway_error_count: number;
    gateway_timeout_count: number;
  };
};

export type GeneratedCaseRecord = {
  source_episode_id: string;
  cache_hit: boolean;
  case: RecallPlannerCase | null;
  error?: ErrorSummary;
};

export type RecallPlannerJudgeCase = {
  case_id: string;
  cache_hit: boolean;
  ratings: JudgeRating[];
  error?: ErrorSummary;
};

export type RecallPlannerAbResults = {
  schema_version: 1;
  generated_at: string;
  inputs: {
    data_dir: string;
    cases_path: string;
    out_dir: string;
    variant_counts: number[];
    baseline_requested: boolean;
    embedding_model: string;
    planner_model: string;
    judge_requested: boolean;
    requested_judge_model: string | null;
    generate_cases: number;
  };
  bank: {
    all_episode_count: number;
    active_episode_count: number;
    inactive_episode_count: number;
    embedding_dimensions: number;
    active_corpus_sha256: string;
    configured_embedding_model: string;
    memory_owner_name: string;
    missing_expected_episode_ids: string[];
  };
  gateway: {
    base_url: string;
    available_models: string[];
    embedding_initialization_attempts: EmbeddingCallRecord[];
    case_generation_model: string | null;
  };
  generation: {
    requested: number;
    generated: number;
    prompt_version: string;
    records: GeneratedCaseRecord[];
  };
  configurations: RecallPlannerConfiguration[];
  cases: RecallPlannerCase[];
  runs: RecallPlannerCaseRun[];
  summaries: RecallPlannerConfigurationSummary[];
  judging: null | {
    model: string;
    prompt_version: string;
    cases: RecallPlannerJudgeCase[];
    per_configuration: Array<{
      configuration_id: string;
      rated_result_count: number;
      mean_relevance: number | null;
    }>;
  };
};
