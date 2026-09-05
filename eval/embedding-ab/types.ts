export type ErrorSummary = {
  name: string;
  message: string;
  status?: number;
  code?: string;
};

export type EpisodeDocument = {
  id: string;
  title: string;
  narrative: string;
  tags: string[];
  embedding_text: string;
  embedding_text_sha256: string;
};

export type EmbeddingPurpose = "dimension_probe" | "episode" | "gold_question" | "real_query";

export type EmbeddingCallRecord = {
  sequence: number;
  purpose: EmbeddingPurpose;
  batch_size: number;
  attempt: number;
  started_at: string;
  latency_ms: number;
  outcome: "success" | "error" | "timeout";
  error?: ErrorSummary;
};

export type LatencySummary = {
  call_count: number;
  successful_call_count: number;
  error_count: number;
  timeout_count: number;
  retry_attempt_count: number;
  p50_ms: number | null;
  p95_ms: number | null;
  max_ms: number | null;
  calls: EmbeddingCallRecord[];
};

export type EmbeddingFailure = {
  key: string;
  error: ErrorSummary;
  timeout: boolean;
};

export type EmbeddingCoverage = {
  requested: number;
  available: number;
  cache_hits: number;
  cache_misses: number;
  embedded_this_run: number;
  failed: number;
  failures: EmbeddingFailure[];
};

export type RankMetrics = {
  question_count: number;
  ranked_source_count: number;
  recall_at_1: number | null;
  recall_at_3: number | null;
  recall_at_10: number | null;
  mrr: number | null;
};

export type RankedEpisode = {
  rank: number;
  episode_id: string;
  title: string;
  cosine_similarity: number;
};

export type GoldQuestionRecord = {
  index: number;
  source_episode_id: string;
  question: string | null;
  cache_hit: boolean;
  error?: ErrorSummary;
};

export type GoldQuestionRank = {
  index: number;
  source_episode_id: string;
  rank: number | null;
  source_cosine_similarity: number | null;
  error?: ErrorSummary;
};

export type RealQueryModelResult = {
  query_index: number;
  error?: ErrorSummary;
  top_10: RankedEpisode[];
};

export type CompletedModelEvaluationResult = {
  model: string;
  status: "completed";
  initialization_error: null;
  dimensions: number;
  episode_vectors: EmbeddingCoverage;
  gold_question_vectors: EmbeddingCoverage;
  real_query_vectors: EmbeddingCoverage;
  gold: {
    metrics: RankMetrics;
    per_question: GoldQuestionRank[];
  };
  real_queries: RealQueryModelResult[];
  latency: LatencySummary;
};

export type FailedModelEvaluationResult = {
  model: string;
  status: "initialization_failed";
  initialization_error: ErrorSummary;
  dimensions: null;
  episode_vectors: null;
  gold_question_vectors: null;
  real_query_vectors: null;
  gold: null;
  real_queries: null;
  latency: LatencySummary;
};

export type ModelEvaluationResult = CompletedModelEvaluationResult | FailedModelEvaluationResult;

export type CrossModelRank = {
  episode_id: string;
  source_rank: number;
  target_rank: number | null;
};

export type ReplayComparison = {
  query_index: number;
  left_model: string;
  right_model: string;
  top_10_overlap_count: number;
  top_10_overlap_denominator: number;
  top_10_overlap_ratio: number | null;
  left_ranks_of_right_top_3: CrossModelRank[];
  right_ranks_of_left_top_3: CrossModelRank[];
};

export type JudgeRating = {
  episode_id: string;
  relevance: 0 | 1 | 2 | 3;
};

export type JudgeQueryResult = {
  query_index: number;
  cache_hit: boolean;
  ratings: JudgeRating[];
  error?: ErrorSummary;
};

export type JudgeModelSummary = {
  model: string;
  rated_result_count: number;
  mean_relevance: number | null;
};

export type EmbeddingAbResults = {
  schema_version: 1;
  generated_at: string;
  inputs: {
    data_dir: string;
    out_dir: string;
    models: string[];
    queries_source: string | null;
    queries: string[];
    requested_gold_size: number;
    batch_size: number;
    embedding_timeout_ms: number;
    judge_requested: boolean;
    requested_judge_model: string | null;
  };
  bank: {
    all_episode_count: number;
    active_episode_count: number;
    inactive_episode_count: number;
    source_embedding_dimensions: number;
    active_corpus_sha256: string;
  };
  comparison_corpus: {
    participating_models: string[];
    active_bank_episode_count: number;
    common_episode_count: number;
    excluded_episode_count: number;
    episode_ids: string[];
    excluded_episode_ids: string[];
    coverage_complete: boolean;
    comparative_metrics_comparable_to_full_bank_recall: boolean;
  };
  gateway: {
    base_url: string;
    available_models: string[];
    gold_generation_model: string | null;
  };
  gold_set: {
    seed: string;
    prompt_version: string;
    requested_size: number;
    sampled_size: number;
    generated_question_count: number;
    questions: GoldQuestionRecord[];
  };
  models: ModelEvaluationResult[];
  replay_comparisons: ReplayComparison[];
  judging: null | {
    model: string;
    prompt_version: string;
    queries: JudgeQueryResult[];
    per_model: JudgeModelSummary[];
  };
};
