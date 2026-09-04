import { mkdirSync } from "node:fs";
import { join, resolve } from "node:path";

import { loadActiveEpisodeBank } from "./bank.js";
import { JsonlValueCache, VectorCache } from "./cache.js";
import { embedItems, type EmbeddedItems } from "./embed-items.js";
import {
  createGatewayLlmClient,
  createModelEmbeddingRuntime,
  createOpenAIClient,
  discoverGatewayModels,
  EMBEDDING_TIMEOUT_MS,
  ModelEmbeddingInitializationError,
  normalizeGatewayBaseUrl,
  selectStrongModel,
  summarizeError,
  summarizeLatency,
  type ModelEmbeddingRuntime,
} from "./gateway.js";
import {
  generateGoldQuestions,
  GOLD_PROMPT_VERSION,
  GOLD_SEED,
  judgeRealQuery,
  JUDGE_PROMPT_VERSION,
  meanRelevance,
  parseCachedJudgment,
  parseCachedQuestion,
  seededEpisodeSample,
  uniqueTopFiveCandidateIds,
  type CachedJudgment,
  type CachedQuestion,
} from "./llm-tasks.js";
import {
  commonEpisodeIds,
  rankEpisodes,
  ranksOfSourceTopKInTarget,
  summarizeRanks,
  topKOverlap,
  type EpisodeVector,
} from "./ranking.js";
import type {
  CompletedModelEvaluationResult,
  EmbeddingAbResults,
  ErrorSummary,
  GoldQuestionRank,
  JudgeModelSummary,
  JudgeQueryResult,
  JudgeRating,
  FailedModelEvaluationResult,
  RankedEpisode,
  ReplayComparison,
} from "./types.js";

export type EmbeddingAbOptions = {
  dataDir: string;
  models: string[];
  outDir: string;
  queries: string[];
  queriesSource: string | null;
  goldSize: number;
  judgeRequested: boolean;
  judgeModel?: string;
  batchSize: number;
  baseUrl: string;
  apiKey: string;
  log?: (message: string) => void;
};

type InternalModelResult = {
  result: CompletedModelEvaluationResult;
  fullRealQueryRankings: RankedEpisode[][];
};

type PreparedModel = {
  model: string;
  runtime: ModelEmbeddingRuntime;
  episodeEmbeddings: EmbeddedItems;
  goldEmbeddings: EmbeddedItems;
  realQueryEmbeddings: EmbeddedItems;
};

type ModelRun =
  | { status: "prepared"; prepared: PreparedModel }
  | { status: "initialization_failed"; result: FailedModelEvaluationResult };

function missingVectorError(kind: "query" | "source episode"): ErrorSummary {
  return {
    name: "EmbeddingUnavailable",
    message: `${kind} vector is unavailable because its embedding batch failed`,
  };
}

function commonCorpusExclusionError(): ErrorSummary {
  return {
    name: "CommonCorpusExclusion",
    message:
      "source episode is outside the common corpus because at least one participating model lacks its embedding",
  };
}

function failureError(
  failures: readonly { key: string; error: ErrorSummary }[],
  key: string,
): ErrorSummary | undefined {
  return failures.find((failure) => failure.key === key)?.error;
}

function buildReplayComparisons(
  models: readonly InternalModelResult[],
  queryCount: number,
): ReplayComparison[] {
  const comparisons: ReplayComparison[] = [];

  for (let queryIndex = 0; queryIndex < queryCount; queryIndex += 1) {
    for (let leftIndex = 0; leftIndex < models.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < models.length; rightIndex += 1) {
        const left = models[leftIndex];
        const right = models[rightIndex];
        if (left === undefined || right === undefined) {
          continue;
        }
        const leftRanking = left.fullRealQueryRankings[queryIndex] ?? [];
        const rightRanking = right.fullRealQueryRankings[queryIndex] ?? [];
        const overlap = topKOverlap(leftRanking, rightRanking, 10);
        comparisons.push({
          query_index: queryIndex + 1,
          left_model: left.result.model,
          right_model: right.result.model,
          top_10_overlap_count: overlap.count,
          top_10_overlap_denominator: overlap.denominator,
          top_10_overlap_ratio: overlap.ratio,
          left_ranks_of_right_top_3: ranksOfSourceTopKInTarget(rightRanking, leftRanking, 3),
          right_ranks_of_left_top_3: ranksOfSourceTopKInTarget(leftRanking, rightRanking, 3),
        });
      }
    }
  }

  return comparisons;
}

function progressLogger(
  log: ((message: string) => void) | undefined,
  label: string,
): (progress: { completed: number; total: number }) => void {
  let lastReported = -1;
  return ({ completed, total }) => {
    if (log === undefined || total === 0) {
      return;
    }
    const percent = Math.floor((completed / total) * 100);
    if (completed === total || percent >= lastReported + 10) {
      lastReported = percent;
      log(`${label}: ${completed}/${total} (${percent}%)`);
    }
  };
}

export async function runEmbeddingAbEvaluation(
  options: EmbeddingAbOptions,
): Promise<EmbeddingAbResults> {
  const outDir = resolve(options.outDir);
  mkdirSync(outDir, { recursive: true, mode: 0o700 });
  const baseUrl = normalizeGatewayBaseUrl(options.baseUrl);
  const log = options.log;

  log?.("Opening source bank with read-only SQLite and LanceDB episode reads...");
  const bank = await loadActiveEpisodeBank(options.dataDir);
  log?.(
    `Loaded ${bank.episodes.length} active episode(s) from ${bank.allEpisodeCount} total row(s).`,
  );

  const openai = createOpenAIClient(baseUrl, options.apiKey);
  log?.("Discovering gateway models with GET /v1/models...");
  const availableModels = await discoverGatewayModels(openai);
  const availableIds = new Set(availableModels.map((model) => model.id));

  for (const model of options.models) {
    if (!availableIds.has(model)) {
      log?.(
        `Warning: embedding model ${model} was not advertised by /v1/models; trying it anyway.`,
      );
    }
  }

  const needsAutomaticStrongModel =
    options.goldSize > 0 || (options.judgeRequested && options.judgeModel === undefined);
  const strongModel = needsAutomaticStrongModel ? selectStrongModel(availableModels) : null;
  const goldModel = options.goldSize > 0 ? strongModel : null;
  const judgeModel = options.judgeRequested
    ? (options.judgeModel ?? strongModel ?? undefined)
    : undefined;

  if (
    judgeModel !== undefined &&
    options.judgeModel !== undefined &&
    !availableIds.has(judgeModel)
  ) {
    throw new Error(`Requested judge model ${judgeModel} was not advertised by GET /v1/models`);
  }

  const llmClient =
    goldModel !== null || judgeModel !== undefined
      ? createGatewayLlmClient(baseUrl, options.apiKey)
      : undefined;
  const sampledEpisodes = seededEpisodeSample(bank.episodes, options.goldSize);
  const questionCache = new JsonlValueCache<CachedQuestion>(
    join(outDir, "cache", "gold-questions.jsonl"),
    parseCachedQuestion,
    join(outDir, "cache"),
  );
  const goldQuestions =
    goldModel === null || llmClient === undefined
      ? []
      : await generateGoldQuestions({
          episodes: sampledEpisodes,
          llmClient,
          model: goldModel,
          cache: questionCache,
          onProgress: progressLogger(log, "Gold questions"),
        });
  const usableGoldQuestions = goldQuestions.filter(
    (question): question is typeof question & { question: string } => question.question !== null,
  );

  const modelRuns: ModelRun[] = [];
  for (const model of options.models) {
    log?.(`Evaluating embedding model ${model}...`);
    let runtime: ModelEmbeddingRuntime;
    try {
      runtime = await createModelEmbeddingRuntime({
        model,
        models: availableModels,
        openai,
        batchSize: options.batchSize,
      });
    } catch (error) {
      const calls =
        error instanceof ModelEmbeddingInitializationError ? error.calls : ([] as const);
      const initializationError = summarizeError(error);
      modelRuns.push({
        status: "initialization_failed",
        result: {
          model,
          status: "initialization_failed",
          initialization_error: initializationError,
          dimensions: null,
          episode_vectors: null,
          gold_question_vectors: null,
          real_query_vectors: null,
          gold: null,
          real_queries: null,
          latency: summarizeLatency(calls),
        },
      });
      log?.(
        `Warning: embedding model ${model} could not be initialized; continuing with the remaining models (${initializationError.name}: ${initializationError.message}).`,
      );
      continue;
    }

    const cache = new VectorCache(outDir, model, runtime.dimensions);
    const episodeEmbeddings = await embedItems({
      items: bank.episodes.map((episode) => ({
        key: episode.id,
        text: episode.embedding_text,
      })),
      runtime,
      cache,
      purpose: "episode",
      batchSize: options.batchSize,
      onBatch: progressLogger(log, `${model} episodes`),
    });
    const goldEmbeddings = await embedItems({
      items: usableGoldQuestions.map((question) => ({
        key: question.source_episode_id,
        text: question.question,
      })),
      runtime,
      cache,
      purpose: "gold_question",
      batchSize: options.batchSize,
      onBatch: progressLogger(log, `${model} gold queries`),
    });
    const realQueryEmbeddings = await embedItems({
      items: options.queries.map((query, index) => ({ key: String(index + 1), text: query })),
      runtime,
      cache,
      purpose: "real_query",
      batchSize: options.batchSize,
      onBatch: progressLogger(log, `${model} real queries`),
    });

    modelRuns.push({
      status: "prepared",
      prepared: {
        model,
        runtime,
        episodeEmbeddings,
        goldEmbeddings,
        realQueryEmbeddings,
      },
    });
  }

  const preparedModels = modelRuns.flatMap((run) =>
    run.status === "prepared" ? [run.prepared] : [],
  );
  const commonIds = commonEpisodeIds(
    bank.episodes.map((episode) => episode.id),
    preparedModels.map((model) => model.episodeEmbeddings.vectors),
  );
  const commonIdSet = new Set(commonIds);
  const commonEpisodes = bank.episodes.filter((episode) => commonIdSet.has(episode.id));
  const excludedEpisodeIds = bank.episodes
    .filter((episode) => !commonIdSet.has(episode.id))
    .map((episode) => episode.id);
  const coverageComplete =
    preparedModels.length > 0 && commonEpisodes.length === bank.episodes.length;

  const internalModels: InternalModelResult[] = [];
  const modelResults = modelRuns.map((run) => {
    if (run.status === "initialization_failed") {
      return run.result;
    }

    const { model, runtime, episodeEmbeddings, goldEmbeddings, realQueryEmbeddings } = run.prepared;

    const episodeVectors: EpisodeVector[] = commonEpisodes.map((episode) => {
      const vector = episodeEmbeddings.vectors.get(episode.id);
      if (vector === undefined) {
        throw new Error(
          `Common-corpus invariant failed: ${model} has no vector for episode ${episode.id}`,
        );
      }
      return { episode, vector };
    });
    const goldRanks: GoldQuestionRank[] = usableGoldQuestions.map((question) => {
      const queryVector = goldEmbeddings.vectors.get(question.source_episode_id);
      if (queryVector === undefined) {
        return {
          index: question.index,
          source_episode_id: question.source_episode_id,
          rank: null,
          source_cosine_similarity: null,
          error:
            failureError(goldEmbeddings.coverage.failures, question.source_episode_id) ??
            missingVectorError("query"),
        };
      }
      if (!commonIdSet.has(question.source_episode_id)) {
        return {
          index: question.index,
          source_episode_id: question.source_episode_id,
          rank: null,
          source_cosine_similarity: null,
          error: commonCorpusExclusionError(),
        };
      }
      const ranking = rankEpisodes(queryVector, episodeVectors);
      const source = ranking.find(
        (candidate) => candidate.episode_id === question.source_episode_id,
      );
      return {
        index: question.index,
        source_episode_id: question.source_episode_id,
        rank: source?.rank ?? null,
        source_cosine_similarity: source?.cosine_similarity ?? null,
        ...(source === undefined
          ? {
              error: missingVectorError("source episode"),
            }
          : {}),
      };
    });

    const fullRealQueryRankings: RankedEpisode[][] = [];
    const realQueries = options.queries.map((_, index) => {
      const key = String(index + 1);
      const vector = realQueryEmbeddings.vectors.get(key);
      if (vector === undefined) {
        fullRealQueryRankings.push([]);
        return {
          query_index: index + 1,
          error:
            failureError(realQueryEmbeddings.coverage.failures, key) ?? missingVectorError("query"),
          top_10: [],
        };
      }
      const ranking = rankEpisodes(vector, episodeVectors);
      fullRealQueryRankings.push(ranking);
      return {
        query_index: index + 1,
        top_10: ranking.slice(0, 10),
      };
    });

    const result: CompletedModelEvaluationResult = {
      model,
      status: "completed",
      initialization_error: null,
      dimensions: runtime.dimensions,
      episode_vectors: episodeEmbeddings.coverage,
      gold_question_vectors: goldEmbeddings.coverage,
      real_query_vectors: realQueryEmbeddings.coverage,
      gold: {
        metrics: summarizeRanks(goldRanks.map((entry) => entry.rank)),
        per_question: goldRanks,
      },
      real_queries: realQueries,
      latency: summarizeLatency(runtime.transport.calls),
    };
    internalModels.push({ result, fullRealQueryRankings });
    return result;
  });

  const replayComparisons = buildReplayComparisons(internalModels, options.queries.length);
  let judging: EmbeddingAbResults["judging"] = null;

  if (judgeModel !== undefined && llmClient !== undefined) {
    const episodeById = new Map(bank.episodes.map((episode) => [episode.id, episode]));
    const judgmentCache = new JsonlValueCache<CachedJudgment>(
      join(outDir, "cache", "judge-ratings.jsonl"),
      parseCachedJudgment,
      join(outDir, "cache"),
    );
    const queryJudgments: JudgeQueryResult[] = [];
    for (let queryIndex = 0; queryIndex < options.queries.length; queryIndex += 1) {
      const rankings = internalModels.map((model) => model.fullRealQueryRankings[queryIndex] ?? []);
      const candidateIds = uniqueTopFiveCandidateIds(rankings);
      const candidates = candidateIds.flatMap((episodeId) => {
        const episode = episodeById.get(episodeId);
        return episode === undefined ? [] : [episode];
      });
      const query = options.queries[queryIndex];
      if (query === undefined) {
        continue;
      }
      queryJudgments.push(
        await judgeRealQuery({
          queryIndex: queryIndex + 1,
          query,
          candidates,
          llmClient,
          model: judgeModel,
          cache: judgmentCache,
        }),
      );
      log?.(`Judge: ${queryIndex + 1}/${options.queries.length}`);
    }

    const perModel: JudgeModelSummary[] = internalModels.map((model) => {
      const ratings: JudgeRating[] = [];
      for (let queryIndex = 0; queryIndex < queryJudgments.length; queryIndex += 1) {
        const judgment = queryJudgments[queryIndex];
        if (judgment === undefined || judgment.error !== undefined) {
          continue;
        }
        const ratingById = new Map(
          judgment.ratings.map((rating) => [rating.episode_id, rating] as const),
        );
        const ranking = model.fullRealQueryRankings[queryIndex] ?? [];
        for (const candidate of ranking.slice(0, 5)) {
          const rating = ratingById.get(candidate.episode_id);
          if (rating !== undefined) {
            ratings.push(rating);
          }
        }
      }
      return {
        model: model.result.model,
        rated_result_count: ratings.length,
        mean_relevance: meanRelevance(ratings),
      };
    });
    judging = {
      model: judgeModel,
      prompt_version: JUDGE_PROMPT_VERSION,
      queries: queryJudgments,
      per_model: perModel,
    };
  }

  return {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    inputs: {
      data_dir: bank.dataDir,
      out_dir: outDir,
      models: [...options.models],
      queries_source: options.queriesSource,
      queries: [...options.queries],
      requested_gold_size: options.goldSize,
      batch_size: options.batchSize,
      embedding_timeout_ms: EMBEDDING_TIMEOUT_MS,
      judge_requested: options.judgeRequested,
      requested_judge_model: options.judgeModel ?? null,
    },
    bank: {
      all_episode_count: bank.allEpisodeCount,
      active_episode_count: bank.episodes.length,
      inactive_episode_count: bank.allEpisodeCount - bank.episodes.length,
      source_embedding_dimensions: bank.sourceEmbeddingDimensions,
      active_corpus_sha256: bank.activeCorpusSha256,
    },
    comparison_corpus: {
      participating_models: preparedModels.map((model) => model.model),
      active_bank_episode_count: bank.episodes.length,
      common_episode_count: commonEpisodes.length,
      excluded_episode_count: excludedEpisodeIds.length,
      episode_ids: [...commonIds],
      excluded_episode_ids: excludedEpisodeIds,
      coverage_complete: coverageComplete,
      comparative_metrics_comparable_to_full_bank_recall: coverageComplete,
    },
    gateway: {
      base_url: baseUrl,
      available_models: availableModels.map((model) => model.id),
      gold_generation_model: goldModel,
    },
    gold_set: {
      seed: GOLD_SEED,
      prompt_version: GOLD_PROMPT_VERSION,
      requested_size: options.goldSize,
      sampled_size: sampledEpisodes.length,
      generated_question_count: usableGoldQuestions.length,
      questions: goldQuestions,
    },
    models: modelResults,
    replay_comparisons: replayComparisons,
    judging,
  };
}
