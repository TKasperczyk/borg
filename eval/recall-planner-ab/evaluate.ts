import { chmodSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";

import type { EntityRepository } from "../../src/memory/commitments/index.js";
import { RetrievalPipeline, type RetrievedEpisode } from "../../src/retrieval/index.js";
import type { RetrievalDegradation } from "../../src/retrieval/pipeline.js";
import type { RecallQueryPlan } from "../../src/retrieval/recall-expansion.js";
import { CallbackTracer, type CallbackTraceEntry } from "../../src/tracing/tracer.js";
import type { EntityId } from "../../src/util/ids.js";

import { JsonlValueCache, VectorCache } from "../embedding-ab/cache.js";
import {
  createGatewayLlmClient,
  createModelEmbeddingRuntime,
  createOpenAIClient,
  discoverGatewayModels,
  normalizeGatewayBaseUrl,
  selectStrongModel,
  summarizeError,
} from "../embedding-ab/gateway.js";
import {
  judgeRealQuery,
  JUDGE_PROMPT_VERSION,
  meanRelevance,
  parseCachedJudgment,
  uniqueTopFiveCandidateIds,
  type CachedJudgment,
} from "../embedding-ab/llm-tasks.js";
import { summarizeRanks } from "../embedding-ab/ranking.js";
import type { JudgeRating, RankedEpisode } from "../embedding-ab/types.js";

import { openRecallPlannerBank, rawFocusOnlyRepository } from "./bank.js";
import {
  ScratchCachingEmbeddingClient,
  ScratchPlannerLlmClient,
  parseCachedPlannerResponse,
} from "./instrumentation.js";
import {
  CASE_GENERATION_PROMPT_VERSION,
  generateRecallPlannerCases,
  parseCachedGeneratedCase,
  type CachedGeneratedCase,
} from "./llm-tasks.js";
import type {
  EmbeddingLogicalCallRecord,
  RecallPlannerAbResults,
  RecallPlannerCase,
  RecallPlannerCaseRun,
  RecallPlannerCompletedCaseRun,
  RecallPlannerConfiguration,
  RecallPlannerConfigurationSummary,
  RecallPlannerJudgeCase,
  RecallPlannerLaneRank,
  RecallPlannerTopResult,
} from "./types.js";

const RETRIEVAL_LIMIT = 10;
const EMBEDDING_BATCH_SIZE = 8;

export type RecallPlannerAbOptions = {
  dataDir: string;
  casesPath: string;
  cases: RecallPlannerCase[];
  outDir: string;
  variantCounts: number[];
  baseline: boolean;
  embeddingModel?: string;
  judgeRequested: boolean;
  judgeModel?: string;
  generateCases: number;
  baseUrl: string;
  apiKey: string;
  plannerModel?: string;
  log?: (message: string) => void;
};

type VisibilityResolution = {
  audienceEntityId: EntityId | null;
  visibleAudienceEntityIds: EntityId[];
  unresolvedNames: string[];
};

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function configurations(options: RecallPlannerAbOptions): RecallPlannerConfiguration[] {
  return [
    ...(options.baseline
      ? [
          {
            id: "baseline",
            label: "Baseline: raw FOCUS-blob lane only (no LLM expansion)",
            mode: "baseline_raw_focus_only" as const,
            semantic_variant_count: null,
          },
        ]
      : []),
    ...options.variantCounts.map((count) => ({
      id: `planner-n${count}`,
      label: `Planner N=${count}`,
      mode: "planner" as const,
      semantic_variant_count: count,
    })),
  ];
}

function effectiveCurrentAudienceName(item: RecallPlannerCase): string | undefined {
  return (
    item.identity.current_audience_name ??
    (item.identity.current_venue?.type === "personal"
      ? (item.identity.current_sender_name ?? item.identity.current_venue.name)
      : item.identity.current_venue?.name)
  );
}

function resolveVisibility(
  item: RecallPlannerCase,
  entityRepository: EntityRepository,
): VisibilityResolution {
  const currentAudienceName = effectiveCurrentAudienceName(item);
  const visibleNames = uniqueStrings([
    ...(currentAudienceName === undefined ? [] : [currentAudienceName]),
    ...item.owner_recent_activity.flatMap((activity) =>
      activity.venue.type === "personal" ? [] : [activity.venue.name],
    ),
  ]);
  const resolved = new Map<string, EntityId | null>(
    visibleNames.map((name) => [name, entityRepository.findByName(name)]),
  );
  const audienceEntityId =
    currentAudienceName === undefined ? null : (resolved.get(currentAudienceName) ?? null);
  const visibleAudienceEntityIds = uniqueStrings(
    [...resolved.values()].flatMap((entityId) => (entityId === null ? [] : [entityId])),
  ) as EntityId[];

  return {
    audienceEntityId,
    visibleAudienceEntityIds,
    unresolvedNames: [...resolved]
      .filter(([, entityId]) => entityId === null)
      .map(([name]) => name),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function plannerOutputFromTrace(events: readonly CallbackTraceEntry[]): RecallQueryPlan | null {
  const event = events.find((entry) => entry.event === "recall_expansion.completed");

  if (
    event === undefined ||
    typeof event.resolved_query !== "string" ||
    !Array.isArray(event.semantic_variants) ||
    !Array.isArray(event.named_terms) ||
    !Array.isArray(event.typed_queries)
  ) {
    return null;
  }

  // The trace carries the parsed cue (sinceTs/untilTs/label) under temporal_cue; the raw draft is
  // not traced, so the report shows the cue Borg actually used.
  const temporalCue =
    event.temporal_cue !== null && typeof event.temporal_cue === "object"
      ? (event.temporal_cue as RecallQueryPlan["temporalCue"])
      : null;

  return {
    resolved_query: event.resolved_query,
    semantic_variants: event.semantic_variants as RecallQueryPlan["semantic_variants"],
    named_terms: event.named_terms.filter((term): term is string => typeof term === "string"),
    typed_queries: event.typed_queries as RecallQueryPlan["typed_queries"],
    temporal_cue: null,
    temporalCue,
  };
}

function plannerLatencyFromTrace(events: readonly CallbackTraceEntry[]): number | null {
  const started = events.find(
    (entry) => entry.event === "llm_call.started" && entry.label === "recall_expansion",
  );
  const completed = events.find((entry) => entry.event === "recall_expansion.completed");
  const degraded = events.find(
    (entry) => entry.event === "retrieval.degraded" && entry.subsystem === "recall_expansion",
  );
  const ended = completed ?? degraded;

  return started === undefined || ended === undefined
    ? null
    : Math.max(0, ended.wallMs - started.wallMs);
}

function laneRanksFromTrace(
  events: readonly CallbackTraceEntry[],
  expectedEpisodeIds: readonly string[],
): RecallPlannerLaneRank[] {
  return events
    .filter((entry) => entry.event === "retrieval.intent_candidates")
    .map((entry) => {
      const rawCandidates = Array.isArray(entry.candidates) ? entry.candidates : [];
      const candidates = rawCandidates.flatMap((candidate, index) => {
        if (!isRecord(candidate) || typeof candidate.episode_id !== "string") {
          return [];
        }

        return [
          {
            rank: index + 1,
            episode_id: candidate.episode_id,
            score:
              typeof candidate.score === "number" && Number.isFinite(candidate.score)
                ? candidate.score
                : null,
            vector_score:
              typeof candidate.vector_score === "number" && Number.isFinite(candidate.vector_score)
                ? candidate.vector_score
                : null,
          },
        ];
      });

      return {
        intent_id: typeof entry.intent_id === "string" ? entry.intent_id : "unknown",
        intent_kind: typeof entry.intent_kind === "string" ? entry.intent_kind : "unknown",
        intent_source: typeof entry.intent_source === "string" ? entry.intent_source : "unknown",
        intent_priority:
          typeof entry.intent_priority === "number" && Number.isFinite(entry.intent_priority)
            ? entry.intent_priority
            : 0,
        intent_query: typeof entry.intent_query === "string" ? entry.intent_query : null,
        candidate_count:
          typeof entry.candidate_count === "number" ? entry.candidate_count : rawCandidates.length,
        candidates,
        expected_ranks: Object.fromEntries(
          expectedEpisodeIds.map((episodeId) => {
            const candidate = candidates.find((entry) => entry.episode_id === episodeId);
            return [episodeId, candidate?.rank ?? null];
          }),
        ),
      };
    });
}

function finalRanks(
  episodes: readonly RetrievedEpisode[],
  expectedEpisodeIds: readonly string[],
): Record<string, number | null> {
  const ranks = new Map<string, number>(
    episodes.map((episode, index): [string, number] => [episode.episode.id, index + 1]),
  );
  return Object.fromEntries(
    expectedEpisodeIds.map((episodeId) => [episodeId, ranks.get(episodeId) ?? null]),
  );
}

function bestRank(ranks: Readonly<Record<string, number | null>>): number | null {
  const present = Object.values(ranks).filter((rank): rank is number => rank !== null);
  return present.length === 0 ? null : Math.min(...present);
}

function topResults(episodes: readonly RetrievedEpisode[]): RecallPlannerTopResult[] {
  return episodes.slice(0, RETRIEVAL_LIMIT).map((item, index) => ({
    rank: index + 1,
    episode_id: item.episode.id,
    title: item.episode.title,
    score: item.score,
    raw_score: item.rawScore,
    vector_similarity: item.scoreBreakdown.similarity,
  }));
}

function cloneLogicalCalls(
  calls: readonly EmbeddingLogicalCallRecord[],
): EmbeddingLogicalCallRecord[] {
  return calls.map((call) => ({
    ...call,
    ...(call.error === undefined ? {} : { error: { ...call.error } }),
  }));
}

function embeddingSummary(input: {
  logicalCalls: readonly EmbeddingLogicalCallRecord[];
  gatewayAttempts: RecallPlannerCompletedCaseRun["embedding"]["gateway_attempts"];
}): RecallPlannerCompletedCaseRun["embedding"] {
  return {
    logical_call_count: input.logicalCalls.length,
    disk_cache_hit_count: input.logicalCalls.filter((call) => call.source === "disk_cache").length,
    pending_cache_hit_count: input.logicalCalls.filter((call) => call.source === "pending_cache")
      .length,
    gateway_logical_call_count: input.logicalCalls.filter((call) => call.source === "gateway")
      .length,
    gateway_attempt_count: input.gatewayAttempts.length,
    gateway_error_count: input.gatewayAttempts.filter((call) => call.outcome === "error").length,
    gateway_timeout_count: input.gatewayAttempts.filter((call) => call.outcome === "timeout")
      .length,
    logical_calls: cloneLogicalCalls(input.logicalCalls),
    gateway_attempts: input.gatewayAttempts.map((call) => ({
      ...call,
      ...(call.error === undefined ? {} : { error: { ...call.error } }),
    })),
  };
}

function publicVisibility(resolution: VisibilityResolution) {
  return {
    audience_entity_id: resolution.audienceEntityId,
    visible_audience_entity_ids: [...resolution.visibleAudienceEntityIds],
    unresolved_names: [...resolution.unresolvedNames],
  };
}

async function settlePlannerCache(input: {
  client: ScratchPlannerLlmClient;
  callIndex: number;
  accepted: boolean;
  degradations: RetrievalDegradation[];
}): Promise<void> {
  try {
    await input.client.settleCallsSince(input.callIndex, input.accepted);
  } catch (error) {
    input.degradations.push({
      subsystem: "planner_cache",
      reason: summarizeError(error).message,
    });
  }
}

async function runCaseConfiguration(input: {
  item: RecallPlannerCase;
  configuration: RecallPlannerConfiguration;
  bank: Awaited<ReturnType<typeof openRecallPlannerBank>>;
  embeddingClient: ScratchCachingEmbeddingClient;
  plannerClient: ScratchPlannerLlmClient;
  plannerModel: string;
  traceTurnId: string;
}): Promise<RecallPlannerCaseRun> {
  const traceEvents: CallbackTraceEntry[] = [];
  const degradations: RetrievalDegradation[] = [];
  const tracer = new CallbackTracer({
    includePayloads: true,
    sink: (entry) => traceEvents.push(entry),
  });
  const visibility = resolveVisibility(input.item, input.bank.entityRepository);
  const plannerCurrentAudienceName = effectiveCurrentAudienceName(input.item);
  const logicalStart = input.embeddingClient.calls.length;
  const gatewayStart = input.embeddingClient.gatewayCalls.length;
  const plannerStart = input.plannerClient.calls.length;
  const isBaseline = input.configuration.mode === "baseline_raw_focus_only";
  const pipeline = new RetrievalPipeline({
    embeddingClient: input.embeddingClient,
    ...(isBaseline ? {} : { llmClient: input.plannerClient }),
    recallExpansionModel: input.plannerModel,
    recallExpansionTimeoutMs: input.bank.config.retrieval.recallExpansionTimeoutMs,
    ...(input.configuration.semantic_variant_count === null
      ? {}
      : { recallExpansionSemanticVariantCount: input.configuration.semantic_variant_count }),
    episodicRepository: isBaseline
      ? rawFocusOnlyRepository(input.bank.episodicRepository)
      : input.bank.episodicRepository,
    dataDir: input.bank.metadata.dataDir,
    entryIndex: input.bank.entryIndex,
    entityRepository: input.bank.entityRepository,
    tracer,
    lexicalFusionEnabled: input.bank.config.retrieval.lexicalFusion.enabled,
    semanticUnderReviewMultiplier: input.bank.config.retrieval.semantic.underReviewMultiplier,
    semanticStatusMultipliers: input.bank.config.retrieval.semantic.statusMultipliers,
    semanticOverfetchMultiplier: input.bank.config.retrieval.semanticOverfetchMultiplier,
  });

  try {
    const episodes = await pipeline.searchEpisodesForDisclosure(input.item.focus, {
      limit: RETRIEVAL_LIMIT,
      audienceEntityId: visibility.audienceEntityId,
      ...(visibility.visibleAudienceEntityIds.length === 0
        ? {}
        : { visibleAudienceEntityIds: visibility.visibleAudienceEntityIds }),
      recordRetrieval: false,
      traceTurnId: input.traceTurnId,
      onDegraded: (degradation) => degradations.push({ ...degradation }),
      ...(isBaseline || input.item.identity.entity_terms === undefined
        ? {}
        : { entityTerms: input.item.identity.entity_terms }),
      ...(isBaseline || input.configuration.semantic_variant_count === null
        ? {}
        : {
            semanticVariantCount: input.configuration.semantic_variant_count,
            recallQueryPlannerContext: {
              contextTurns: input.item.context_turns.map((turn) => ({ ...turn })),
              identity: {
                memoryOwnerName: input.item.identity.memory_owner_name,
                ...(input.item.identity.current_sender_name === undefined
                  ? {}
                  : { currentSenderName: input.item.identity.current_sender_name }),
                ...(plannerCurrentAudienceName === undefined
                  ? {}
                  : { currentAudienceName: plannerCurrentAudienceName }),
                ...(input.item.identity.current_venue === undefined
                  ? {}
                  : { currentVenue: { ...input.item.identity.current_venue } }),
                ...(input.item.identity.entity_terms === undefined
                  ? {}
                  : { entityTerms: [...input.item.identity.entity_terms] }),
              },
              ownerRecentActivity: input.item.owner_recent_activity.map((activity) => ({
                excerpt: activity.excerpt,
                occurredAt: activity.occurred_at,
                venue: { ...activity.venue },
                ...(activity.counterparty_name === undefined
                  ? {}
                  : { counterpartyName: activity.counterparty_name }),
              })),
            },
          }),
    });
    const expectedFinalRanks = finalRanks(episodes, input.item.expected_episode_ids);
    const plannerOutput = isBaseline ? null : plannerOutputFromTrace(traceEvents);
    await settlePlannerCache({
      client: input.plannerClient,
      callIndex: plannerStart,
      accepted: plannerOutput !== null,
      degradations,
    });
    const logicalCalls = input.embeddingClient.calls.slice(logicalStart);
    const gatewayAttempts = input.embeddingClient.gatewayCalls.slice(gatewayStart);
    const plannerCalls = input.plannerClient.calls.slice(plannerStart);

    return {
      status: "completed",
      case_id: input.item.id,
      configuration_id: input.configuration.id,
      planner_output: plannerOutput,
      planner_latency_ms: isBaseline ? null : plannerLatencyFromTrace(traceEvents),
      planner_cache_hit: isBaseline ? null : (plannerCalls[0]?.cache_hit ?? null),
      lane_ranks: laneRanksFromTrace(traceEvents, input.item.expected_episode_ids),
      expected_final_ranks: expectedFinalRanks,
      best_expected_rank: bestRank(expectedFinalRanks),
      top_10: topResults(episodes),
      embedding: embeddingSummary({ logicalCalls, gatewayAttempts }),
      degradations,
      visibility: publicVisibility(visibility),
    };
  } catch (error) {
    const plannerOutput = isBaseline ? null : plannerOutputFromTrace(traceEvents);
    await settlePlannerCache({
      client: input.plannerClient,
      callIndex: plannerStart,
      accepted: plannerOutput !== null,
      degradations,
    });
    const logicalCalls = input.embeddingClient.calls.slice(logicalStart);
    const gatewayAttempts = input.embeddingClient.gatewayCalls.slice(gatewayStart);
    const plannerCalls = input.plannerClient.calls.slice(plannerStart);

    return {
      status: "failed",
      case_id: input.item.id,
      configuration_id: input.configuration.id,
      error: summarizeError(error),
      planner_output: plannerOutput,
      planner_latency_ms: isBaseline ? null : plannerLatencyFromTrace(traceEvents),
      planner_cache_hit: isBaseline ? null : (plannerCalls[0]?.cache_hit ?? null),
      lane_ranks: laneRanksFromTrace(traceEvents, input.item.expected_episode_ids),
      expected_final_ranks: Object.fromEntries(
        input.item.expected_episode_ids.map((episodeId) => [episodeId, null]),
      ),
      best_expected_rank: null,
      top_10: [],
      embedding: embeddingSummary({ logicalCalls, gatewayAttempts }),
      degradations,
      visibility: publicVisibility(visibility),
    };
  }
}

function percentile(values: readonly number[], fraction: number): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)] ?? null;
}

function mean(values: readonly number[]): number | null {
  return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function summarizeConfigurations(
  allConfigurations: readonly RecallPlannerConfiguration[],
  runs: readonly RecallPlannerCaseRun[],
): RecallPlannerConfigurationSummary[] {
  return allConfigurations.map((configuration) => {
    const configurationRuns = runs.filter((run) => run.configuration_id === configuration.id);
    const latencies = configurationRuns.flatMap((run) =>
      run.planner_latency_ms === null ? [] : [run.planner_latency_ms],
    );
    const gatewayLatencies = configurationRuns.flatMap((run) =>
      run.embedding.gateway_attempts.map((attempt) => attempt.latency_ms),
    );

    return {
      configuration_id: configuration.id,
      metrics: summarizeRanks(configurationRuns.map((run) => run.best_expected_rank)),
      completed_case_count: configurationRuns.filter((run) => run.status === "completed").length,
      failed_case_count: configurationRuns.filter((run) => run.status === "failed").length,
      degraded_case_count: configurationRuns.filter((run) => run.degradations.length > 0).length,
      planner_latency: {
        measured_count: latencies.length,
        cache_hit_count: configurationRuns.filter((run) => run.planner_cache_hit === true).length,
        cache_miss_count: configurationRuns.filter((run) => run.planner_cache_hit === false).length,
        p50_ms: percentile(latencies, 0.5),
        p95_ms: percentile(latencies, 0.95),
        max_ms: latencies.length === 0 ? null : Math.max(...latencies),
      },
      embedding: {
        logical_calls_per_query: mean(
          configurationRuns.map((run) => run.embedding.logical_call_count),
        ),
        gateway_attempts_per_query: mean(
          configurationRuns.map((run) => run.embedding.gateway_attempt_count),
        ),
        gateway_latency: {
          measured_count: gatewayLatencies.length,
          p50_ms: percentile(gatewayLatencies, 0.5),
          p95_ms: percentile(gatewayLatencies, 0.95),
          max_ms: gatewayLatencies.length === 0 ? null : Math.max(...gatewayLatencies),
        },
        disk_cache_hit_count: configurationRuns.reduce(
          (sum, run) => sum + run.embedding.disk_cache_hit_count,
          0,
        ),
        pending_cache_hit_count: configurationRuns.reduce(
          (sum, run) => sum + run.embedding.pending_cache_hit_count,
          0,
        ),
        gateway_error_count: configurationRuns.reduce(
          (sum, run) => sum + run.embedding.gateway_error_count,
          0,
        ),
        gateway_timeout_count: configurationRuns.reduce(
          (sum, run) => sum + run.embedding.gateway_timeout_count,
          0,
        ),
      },
    };
  });
}

function judgeQuery(item: RecallPlannerCase): string {
  return [
    "CONTEXT (oldest to newest):",
    ...item.context_turns.map((turn) => `${turn.role}: ${turn.content}`),
    "FOCUS:",
    item.focus,
    "IDENTITY HANDLES (JSON data):",
    JSON.stringify(item.identity),
    "OWNER RECENT ACTIVITY (JSON data):",
    JSON.stringify(item.owner_recent_activity),
  ].join("\n");
}

function asJudgeRanking(run: RecallPlannerCaseRun): RankedEpisode[] {
  return run.top_10.map((candidate) => ({
    rank: candidate.rank,
    episode_id: candidate.episode_id,
    title: candidate.title,
    cosine_similarity: candidate.vector_similarity,
  }));
}

async function judgeRuns(input: {
  model: string;
  cases: readonly RecallPlannerCase[];
  configurations: readonly RecallPlannerConfiguration[];
  runs: readonly RecallPlannerCaseRun[];
  episodes: Awaited<ReturnType<typeof openRecallPlannerBank>>["metadata"]["episodes"];
  llmClient: ReturnType<typeof createGatewayLlmClient>;
  cache: JsonlValueCache<CachedJudgment>;
  log?: (message: string) => void;
}): Promise<NonNullable<RecallPlannerAbResults["judging"]>> {
  const episodeById = new Map(input.episodes.map((episode) => [episode.id, episode]));
  const caseJudgments: RecallPlannerJudgeCase[] = [];

  for (let index = 0; index < input.cases.length; index += 1) {
    const item = input.cases[index];
    if (item === undefined) {
      continue;
    }
    const caseRuns = input.configurations.map((configuration) =>
      input.runs.find(
        (run) => run.case_id === item.id && run.configuration_id === configuration.id,
      ),
    );
    const candidateIds = uniqueTopFiveCandidateIds(
      caseRuns.map((run) => (run === undefined ? [] : asJudgeRanking(run))),
    );
    const candidates = candidateIds.flatMap((episodeId) => {
      const episode = episodeById.get(episodeId);
      return episode === undefined ? [] : [episode];
    });
    const judged = await judgeRealQuery({
      queryIndex: index + 1,
      query: judgeQuery(item),
      candidates,
      llmClient: input.llmClient,
      model: input.model,
      cache: input.cache,
    });
    caseJudgments.push({
      case_id: item.id,
      cache_hit: judged.cache_hit,
      ratings: judged.ratings,
      ...(judged.error === undefined ? {} : { error: judged.error }),
    });
    input.log?.(`Judge: ${index + 1}/${input.cases.length}`);
  }

  const perConfiguration = input.configurations.map((configuration) => {
    const ratings: JudgeRating[] = [];

    for (const item of input.cases) {
      const judgment = caseJudgments.find((entry) => entry.case_id === item.id);
      const run = input.runs.find(
        (entry) => entry.case_id === item.id && entry.configuration_id === configuration.id,
      );
      if (judgment === undefined || judgment.error !== undefined || run === undefined) {
        continue;
      }
      const ratingById = new Map(
        judgment.ratings.map((rating) => [rating.episode_id, rating] as const),
      );
      for (const candidate of run.top_10.slice(0, 5)) {
        const rating = ratingById.get(candidate.episode_id);
        if (rating !== undefined) {
          ratings.push(rating);
        }
      }
    }

    return {
      configuration_id: configuration.id,
      rated_result_count: ratings.length,
      mean_relevance: meanRelevance(ratings),
    };
  });

  return {
    model: input.model,
    prompt_version: JUDGE_PROMPT_VERSION,
    cases: caseJudgments,
    per_configuration: perConfiguration,
  };
}

function sanitizedBaseUrl(raw: string): string {
  const url = new URL(raw);
  url.username = "";
  url.password = "";
  url.search = "";
  url.hash = "";
  return url.toString().replace(/\/$/, "");
}

export async function runRecallPlannerAbEvaluation(
  options: RecallPlannerAbOptions,
): Promise<RecallPlannerAbResults> {
  const outDir = resolve(options.outDir);
  mkdirSync(outDir, { recursive: true, mode: 0o700 });
  chmodSync(outDir, 0o700);
  const baseUrl = normalizeGatewayBaseUrl(options.baseUrl);
  const log = options.log;

  log?.("Opening copied bank through read-only SQLite/LanceDB repository handles...");
  const bank = await openRecallPlannerBank(options.dataDir);

  try {
    const configuredEmbeddingModel = bank.config.embedding.model;
    const embeddingModel = options.embeddingModel ?? configuredEmbeddingModel;
    const plannerModel = options.plannerModel ?? bank.config.anthropic.models.recallExpansion;
    const openai = createOpenAIClient(baseUrl, options.apiKey);
    log?.("Discovering gateway models...");
    const availableModels = await discoverGatewayModels(openai);
    const availableIds = new Set(availableModels.map((model) => model.id));
    for (const model of [embeddingModel, plannerModel]) {
      if (!availableIds.has(model)) {
        log?.(`Warning: ${model} was not advertised by the gateway; trying it anyway.`);
      }
    }

    const needsStrongModel =
      options.generateCases > 0 || (options.judgeRequested && options.judgeModel === undefined);
    const strongModel = needsStrongModel ? selectStrongModel(availableModels) : null;
    const generationModel = options.generateCases > 0 ? strongModel : null;
    const judgeModel = options.judgeRequested
      ? (options.judgeModel ?? strongModel ?? undefined)
      : undefined;

    if (
      judgeModel !== undefined &&
      options.judgeModel !== undefined &&
      !availableIds.has(judgeModel)
    ) {
      throw new Error(`Requested judge model ${judgeModel} was not advertised by the gateway`);
    }

    const gatewayLlmClient = createGatewayLlmClient(baseUrl, options.apiKey);
    const plannerCache = new JsonlValueCache(
      join(outDir, "cache", "planner-outputs.jsonl"),
      parseCachedPlannerResponse,
      join(outDir, "cache"),
    );
    const plannerClient = new ScratchPlannerLlmClient(gatewayLlmClient, plannerCache);
    const runtime = await createModelEmbeddingRuntime({
      model: embeddingModel,
      models: availableModels,
      openai,
      batchSize: EMBEDDING_BATCH_SIZE,
    });
    const initializationAttempts = runtime.transport.calls.map((call) => ({ ...call }));

    if (runtime.dimensions !== bank.metadata.sourceEmbeddingDimensions) {
      throw new Error(
        `Embedding model ${embeddingModel} returns ${runtime.dimensions} dimensions, but the copied bank uses ${bank.metadata.sourceEmbeddingDimensions}`,
      );
    }

    const vectorCache = new VectorCache(outDir, embeddingModel, runtime.dimensions);
    const embeddingClient = new ScratchCachingEmbeddingClient(runtime, vectorCache);
    const memoryOwnerName =
      bank.memoryOwner?.canonical_name ??
      bank.config.defaultUser ??
      options.cases[0]?.identity.memory_owner_name ??
      "borg";
    const generatedCaseCache = new JsonlValueCache<CachedGeneratedCase>(
      join(outDir, "cache", "generated-cases.jsonl"),
      parseCachedGeneratedCase,
      join(outDir, "cache"),
    );
    const generatedRecords =
      generationModel === null
        ? []
        : await generateRecallPlannerCases({
            episodes: bank.metadata.episodes,
            count: options.generateCases,
            memoryOwnerName,
            llmClient: gatewayLlmClient,
            model: generationModel,
            cache: generatedCaseCache,
            onProgress: ({ completed, total }) => log?.(`Generated cases: ${completed}/${total}`),
          });
    const allCases = [
      ...options.cases,
      ...generatedRecords.flatMap((record) => (record.case === null ? [] : [record.case])),
    ];
    const duplicateCaseIds = allCases
      .map((item) => item.id)
      .filter((id, index, ids) => ids.indexOf(id) !== index);
    if (duplicateCaseIds.length > 0) {
      throw new Error(
        `Duplicate case id(s) after generation: ${uniqueStrings(duplicateCaseIds).join(", ")}`,
      );
    }

    const allConfigurations = configurations(options);
    const runs: RecallPlannerCaseRun[] = [];
    let sequence = 0;

    for (const item of allCases) {
      for (const configuration of allConfigurations) {
        sequence += 1;
        log?.(
          `Recall ${sequence}/${allCases.length * allConfigurations.length}: ${item.id} / ${configuration.id}`,
        );
        runs.push(
          await runCaseConfiguration({
            item,
            configuration,
            bank,
            embeddingClient,
            plannerClient,
            plannerModel,
            traceTurnId: `recall_planner_ab:${sequence}:${item.id}:${configuration.id}`,
          }),
        );
      }
    }

    let judging: RecallPlannerAbResults["judging"] = null;
    if (judgeModel !== undefined) {
      judging = await judgeRuns({
        model: judgeModel,
        cases: allCases,
        configurations: allConfigurations,
        runs,
        episodes: bank.metadata.episodes,
        llmClient: gatewayLlmClient,
        cache: new JsonlValueCache<CachedJudgment>(
          join(outDir, "cache", "judge-ratings.jsonl"),
          parseCachedJudgment,
          join(outDir, "cache"),
        ),
        log,
      });
    }

    const activeEpisodeIds = new Set(bank.metadata.episodes.map((episode) => episode.id));
    const missingExpectedEpisodeIds = uniqueStrings(
      allCases.flatMap((item) =>
        item.expected_episode_ids.filter((episodeId) => !activeEpisodeIds.has(episodeId)),
      ),
    );

    return {
      schema_version: 1,
      generated_at: new Date().toISOString(),
      inputs: {
        data_dir: bank.metadata.dataDir,
        cases_path: resolve(options.casesPath),
        out_dir: outDir,
        variant_counts: [...options.variantCounts],
        baseline_requested: options.baseline,
        embedding_model: embeddingModel,
        planner_model: plannerModel,
        judge_requested: options.judgeRequested,
        requested_judge_model: options.judgeModel ?? null,
        generate_cases: options.generateCases,
      },
      bank: {
        all_episode_count: bank.metadata.allEpisodeCount,
        active_episode_count: bank.metadata.episodes.length,
        inactive_episode_count: bank.metadata.allEpisodeCount - bank.metadata.episodes.length,
        embedding_dimensions: bank.metadata.sourceEmbeddingDimensions,
        active_corpus_sha256: bank.metadata.activeCorpusSha256,
        configured_embedding_model: configuredEmbeddingModel,
        memory_owner_name: memoryOwnerName,
        missing_expected_episode_ids: missingExpectedEpisodeIds,
      },
      gateway: {
        base_url: sanitizedBaseUrl(baseUrl),
        available_models: availableModels.map((model) => model.id),
        embedding_initialization_attempts: initializationAttempts,
        case_generation_model: generationModel,
      },
      generation: {
        requested: options.generateCases,
        generated: generatedRecords.filter((record) => record.case !== null).length,
        prompt_version: CASE_GENERATION_PROMPT_VERSION,
        records: generatedRecords,
      },
      configurations: allConfigurations,
      cases: allCases,
      runs,
      summaries: summarizeConfigurations(allConfigurations, runs),
      judging,
    };
  } finally {
    bank.close();
  }
}
