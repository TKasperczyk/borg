import { z } from "zod";

import { computeWeights } from "../../cognition/attention/index.js";
import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import {
  episodeIdSchema,
  isEpisodeInGlobalIdentityScope,
  isEpisodeVisibleToAudience,
} from "../../memory/episodic/index.js";
import {
  growthMarkerCategorySchema,
  growthMarkerIdSchema,
  growthMarkerSchema,
  openQuestionIdSchema,
  openQuestionSchema,
  type OpenQuestion,
} from "../../memory/self/index.js";
import { createOpenQuestionReopener } from "../../memory/self/open-questions.js";
import { computeRetrievalConfidence, type RetrievedEpisode } from "../../retrieval/index.js";
import { createGrowthMarkerId } from "../../util/ids.js";
import { BudgetExceededError, StorageError } from "../../util/errors.js";

import type { ReverserRegistry } from "../audit-log.js";
import { getBudgetErrorTokens, withBudget } from "../budget.js";
import { offlineProcessError } from "../process-errors.js";
import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessError,
  OfflineResult,
} from "../types.js";

const DAY_MS = 24 * 60 * 60 * 1_000;
const GROWTH_MARKER_CONFIDENCE_CEILING = 0.6;
const OPEN_QUESTION_ACTIVE_ACTION_STATES = ["considering", "committed_to_do", "scheduled"] as const;

const resolutionResponseSchema = z.object({
  resolution_note: z.string().min(1),
  growth_marker: z
    .object({
      what_changed: z.string().min(1),
      before_description: z.string().nullable().optional(),
      after_description: z.string().nullable().optional(),
      confidence: z.number().min(0).max(1),
      category: growthMarkerCategorySchema,
    })
    .nullable(),
});
const RUMINATOR_TOOL_NAME = "EmitRuminatorDecisions";
export const RUMINATOR_TOOL = {
  name: RUMINATOR_TOOL_NAME,
  description: "Emit a grounded open-question resolution note and optional growth marker.",
  inputSchema: toToolInputSchema(resolutionResponseSchema),
} satisfies LLMToolDefinition;

const serializableGrowthMarkerSchema = growthMarkerSchema.extend({
  evidence_episode_ids: z.array(episodeIdSchema).min(1),
});

const ruminatorPlanItemSchema = z.discriminatedUnion("action", [
  z.object({
    action: z.literal("resolve"),
    question_id: openQuestionIdSchema,
    previous: openQuestionSchema,
    resolution_evidence_episode_ids: z.array(episodeIdSchema).min(1),
    resolution_evidence_stream_entry_ids: z.array(z.never()).default([]),
    resolution_note: z.string().min(1),
    growth_marker: serializableGrowthMarkerSchema.nullable(),
  }),
  z.object({
    action: z.literal("bump_urgency"),
    question_id: openQuestionIdSchema,
    previous: openQuestionSchema,
    delta: z.number().finite(),
    next_urgency: z.number().min(0).max(1),
    next_unresolved_rumination_ticks: z.number().int().nonnegative(),
  }),
  z.object({
    action: z.literal("abandon"),
    question_id: openQuestionIdSchema,
    previous: openQuestionSchema,
    reason: z.string().min(1),
  }),
  z.object({
    action: z.literal("merge_duplicate"),
    primary_question_id: openQuestionIdSchema,
    duplicate_question_id: openQuestionIdSchema,
    previous_primary: openQuestionSchema,
    previous_duplicate: openQuestionSchema,
    similarity: z.number().min(0).max(1),
  }),
  z.object({
    action: z.literal("mark_unresolved"),
    question_id: openQuestionIdSchema,
    previous: openQuestionSchema,
    next_unresolved_rumination_ticks: z.number().int().nonnegative(),
  }),
]);

export const ruminatorPlanSchema = z.object({
  process: z.literal("ruminator"),
  items: z.array(ruminatorPlanItemSchema),
  errors: z
    .array(
      z.object({
        process: z.literal("ruminator"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
});

export type RuminatorPlan = z.infer<typeof ruminatorPlanSchema>;

type RuminatorReversal = {
  previous?: OpenQuestion;
  previous_primary?: OpenQuestion;
  previous_duplicate?: OpenQuestion;
  marker_id?: z.infer<typeof growthMarkerIdSchema>;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function buildResolutionPrompt(question: OpenQuestion, evidence: string): string {
  return [
    "Resolve the open question using only the evidence below.",
    `Emit your result by calling the ${RUMINATOR_TOOL_NAME} tool exactly once.`,
    "Only include a growth_marker if the evidence clearly shows new understanding.",
    `Question: ${question.question}`,
    `Source: ${question.source}`,
    "Evidence:",
    evidence,
  ].join("\n\n");
}

function parseResolutionResponse(result: LLMCompleteResult) {
  const call = result.tool_calls.find((toolCall) => toolCall.name === RUMINATOR_TOOL_NAME);

  if (call === undefined) {
    throw new StorageError(`Ruminator did not emit tool ${RUMINATOR_TOOL_NAME}`, {
      code: "RUMINATOR_INVALID",
    });
  }

  return resolutionResponseSchema.parse(call.input);
}

function isOfflineChange(change: OfflineChange | null): change is OfflineChange {
  return change !== null;
}

function buildChange(item: RuminatorPlan["items"][number]): OfflineChange | null {
  if (item.action === "resolve") {
    return {
      process: "ruminator",
      action: "resolve",
      targets: {
        question_id: item.question_id,
        resolution_evidence_episode_ids: item.resolution_evidence_episode_ids,
      },
      preview: {
        note: item.resolution_note,
        growth_marker: item.growth_marker?.what_changed ?? null,
      },
    };
  }

  if (item.action === "abandon") {
    return {
      process: "ruminator",
      action: "abandon",
      targets: {
        question_id: item.question_id,
      },
      preview: {
        reason: item.reason,
      },
    };
  }

  if (item.action === "merge_duplicate") {
    return {
      process: "ruminator",
      action: "merge_duplicate",
      targets: {
        primary_question_id: item.primary_question_id,
        duplicate_question_id: item.duplicate_question_id,
      },
      preview: {
        similarity: item.similarity,
      },
    };
  }

  if (item.action === "mark_unresolved") {
    return null;
  }

  return {
    process: "ruminator",
    action: "bump_urgency",
    targets: {
      question_id: item.question_id,
    },
    preview: {
      delta: item.delta,
      next_urgency: item.next_urgency,
    },
  };
}

function isGlobalIdentityQuestion(
  question: OpenQuestion,
  selfAudienceEntityId: ReturnType<OfflineContext["entityRepository"]["findByName"]>,
): boolean {
  return (
    question.audience_entity_id === null ||
    (selfAudienceEntityId !== null &&
      selfAudienceEntityId !== undefined &&
      question.audience_entity_id === selfAudienceEntityId)
  );
}

function isResolutionEvidenceVisibleToQuestion(
  episode: RetrievedEpisode["episode"],
  question: OpenQuestion,
  selfAudienceEntityId: ReturnType<OfflineContext["entityRepository"]["findByName"]>,
): boolean {
  if (isEpisodeInGlobalIdentityScope(episode, selfAudienceEntityId)) {
    return true;
  }

  if (question.audience_entity_id === null) {
    return false;
  }

  return isEpisodeVisibleToAudience(episode, question.audience_entity_id);
}

function mergeRetrievedEpisodes(episodeSets: readonly (readonly RetrievedEpisode[])[]) {
  const byId = new Map<RetrievedEpisode["episode"]["id"], RetrievedEpisode>();

  for (const episodes of episodeSets) {
    for (const result of episodes) {
      const current = byId.get(result.episode.id);

      if (
        current === undefined ||
        result.score > current.score ||
        (result.score === current.score && result.episode.updated_at > current.episode.updated_at)
      ) {
        byId.set(result.episode.id, result);
      }
    }
  }

  return [...byId.values()].sort(
    (left, right) => right.score - left.score || right.episode.updated_at - left.episode.updated_at,
  );
}

function shareOpenQuestionEntityScope(left: OpenQuestion, right: OpenQuestion): boolean {
  const rightNodeIds = new Set(right.related_semantic_node_ids);

  return left.related_semantic_node_ids.some((nodeId) => rightNodeIds.has(nodeId));
}

function olderQuestion(left: OpenQuestion, right: OpenQuestion): OpenQuestion {
  return left.created_at < right.created_at ||
    (left.created_at === right.created_at && left.id < right.id)
    ? left
    : right;
}

function newerQuestion(left: OpenQuestion, right: OpenQuestion): OpenQuestion {
  const older = olderQuestion(left, right);

  return older.id === left.id ? right : left;
}

function mergeQuestionIds(
  ...groups: readonly (readonly OpenQuestion["related_episode_ids"][number][])[]
): OpenQuestion["related_episode_ids"] {
  return [...new Set(groups.flatMap((group) => [...group]))];
}

function mergeSemanticNodeIds(
  left: readonly OpenQuestion["related_semantic_node_ids"][number][],
  right: readonly OpenQuestion["related_semantic_node_ids"][number][],
): OpenQuestion["related_semantic_node_ids"] {
  return [...new Set([...left, ...right])];
}

function mergeQuestionStreamIds(
  ...groups: readonly (readonly OpenQuestion["resolution_evidence_stream_entry_ids"][number][])[]
): OpenQuestion["resolution_evidence_stream_entry_ids"] {
  return [...new Set(groups.flatMap((group) => [...group]))];
}

function countOpenQuestionEvidenceHandles(question: OpenQuestion): number {
  return new Set([
    ...question.related_episode_ids.map((episodeId) => `episode:${episodeId}`),
    ...question.resolution_evidence_episode_ids.map((episodeId) => `episode:${episodeId}`),
    ...openQuestionProvenanceEpisodeIds(question).map((episodeId) => `episode:${episodeId}`),
    ...question.resolution_evidence_stream_entry_ids.map(
      (streamEntryId) => `stream:${streamEntryId}`,
    ),
    ...openQuestionProvenanceStreamEntryIds(question).map(
      (streamEntryId) => `stream:${streamEntryId}`,
    ),
  ]).size;
}

function openQuestionProvenanceEpisodeIds(
  question: OpenQuestion,
): OpenQuestion["related_episode_ids"] {
  if (question.provenance?.kind === "episodes") {
    return [...question.provenance.episode_ids];
  }

  if (question.provenance?.kind === "online_reflector") {
    return [...question.provenance.evidence_episode_ids];
  }

  return [];
}

function openQuestionProvenanceStreamEntryIds(
  question: OpenQuestion,
): OpenQuestion["resolution_evidence_stream_entry_ids"] {
  return question.provenance?.kind === "online_reflector"
    ? [...question.provenance.evidence_stream_entry_ids]
    : [];
}

function openQuestionCitationEpisodeIds(
  question: OpenQuestion,
): OpenQuestion["related_episode_ids"] {
  return [
    ...new Set([
      ...question.related_episode_ids,
      ...question.resolution_evidence_episode_ids,
      ...openQuestionProvenanceEpisodeIds(question),
    ]),
  ];
}

async function hasPostCreationCitation(
  ctx: OfflineContext,
  question: OpenQuestion,
): Promise<boolean> {
  const episodeIds = openQuestionCitationEpisodeIds(question);

  if (episodeIds.length === 0) {
    return false;
  }

  const episodes = await ctx.episodicRepository.getMany(episodeIds);

  return episodes.some((episode) => episode.created_at > question.created_at);
}

function hasActiveAssociatedAction(ctx: OfflineContext, question: OpenQuestion): boolean {
  if (
    ctx.actionRepository.list({
      openQuestionId: question.id,
      states: OPEN_QUESTION_ACTIVE_ACTION_STATES,
      limit: 1,
    }).length > 0
  ) {
    return true;
  }

  return (
    question.goal_id !== null &&
    ctx.actionRepository.list({
      goalId: question.goal_id,
      states: OPEN_QUESTION_ACTIVE_ACTION_STATES,
      limit: 1,
    }).length > 0
  );
}

async function shouldDismissStaleNoTraction(
  ctx: OfflineContext,
  question: OpenQuestion,
): Promise<boolean> {
  return (
    question.unresolved_rumination_ticks >= ctx.config.offline.ruminator.staleNoTractionTicks &&
    !(await hasPostCreationCitation(ctx, question)) &&
    !hasActiveAssociatedAction(ctx, question)
  );
}

async function planDuplicateMerges(
  ctx: OfflineContext,
  questions: readonly OpenQuestion[],
): Promise<RuminatorPlan["items"]> {
  const items: RuminatorPlan["items"] = [];
  const mergedDuplicateIds = new Set<OpenQuestion["id"]>();
  const ordered = [...questions].sort(
    (left, right) => left.created_at - right.created_at || left.id.localeCompare(right.id),
  );

  for (const question of ordered) {
    if (mergedDuplicateIds.has(question.id) || question.related_semantic_node_ids.length === 0) {
      continue;
    }

    const candidates = await ctx.openQuestionsRepository.searchSimilar(question, {
      limit: Math.max(10, ctx.config.offline.ruminator.maxQuestionsPerRun * 4),
      minSimilarity: ctx.config.offline.ruminator.duplicateSimilarityThreshold,
    });

    for (const candidate of candidates) {
      const match = candidate.question;

      if (
        match.status !== "open" ||
        mergedDuplicateIds.has(match.id) ||
        !shareOpenQuestionEntityScope(question, match)
      ) {
        continue;
      }

      const primary = olderQuestion(question, match);
      const duplicate = newerQuestion(question, match);

      if (primary.id !== question.id || mergedDuplicateIds.has(duplicate.id)) {
        continue;
      }

      items.push({
        action: "merge_duplicate",
        primary_question_id: primary.id,
        duplicate_question_id: duplicate.id,
        previous_primary: primary,
        previous_duplicate: duplicate,
        similarity: candidate.similarity,
      });
      mergedDuplicateIds.add(duplicate.id);
    }
  }

  return items;
}

async function searchResolutionEvidence(
  ctx: OfflineContext,
  question: OpenQuestion,
  maxQuestionsPerRun: number,
): Promise<{
  episodes: RetrievedEpisode[];
  expectedCount: number;
}> {
  const selfAudienceEntityId = ctx.entityRepository.findByName("self");
  const baseOptions = {
    limit: Math.max(3, maxQuestionsPerRun),
    attentionWeights: buildReflectionWeights(ctx),
    goalDescriptions: ctx.goalsRepository
      .list({ status: "active" })
      .map((goal) => goal.description),
    includeOpenQuestions: false,
  };
  const globalRetrieval = await ctx.retrievalPipeline.searchWithContext(question.question, {
    ...baseOptions,
    globalIdentitySelfAudienceEntityId: selfAudienceEntityId,
  });
  const globalIdentityEpisodes = globalRetrieval.episodes.filter((result) =>
    isEpisodeInGlobalIdentityScope(result.episode, selfAudienceEntityId),
  );

  if (isGlobalIdentityQuestion(question, selfAudienceEntityId)) {
    return {
      episodes: globalIdentityEpisodes,
      expectedCount: baseOptions.limit,
    };
  }

  // Reuse the retrieval pipeline's audienceEntityId path so the episodic layer
  // applies its existing audience-visible retrieval lanes in addition to the
  // global/self search above.
  const audienceRetrieval = await ctx.retrievalPipeline.searchWithContext(question.question, {
    ...baseOptions,
    audienceEntityId: question.audience_entity_id,
  });
  const episodes = mergeRetrievedEpisodes([
    globalIdentityEpisodes,
    audienceRetrieval.episodes.filter((result) =>
      isResolutionEvidenceVisibleToQuestion(result.episode, question, selfAudienceEntityId),
    ),
  ]);

  return {
    episodes,
    expectedCount: baseOptions.limit,
  };
}

function buildReflectionWeights(ctx: OfflineContext) {
  return computeWeights("reflective", {
    currentGoals: ctx.goalsRepository.list({ status: "active" }),
    hasActiveValues: ctx.valuesRepository.list().some((value) => value.state === "established"),
    hasTemporalCue: false,
  });
}

function emitOpenQuestionResolutionAttempt(
  ctx: OfflineContext,
  input: {
    oqId: OpenQuestion["id"];
    sourcePath: "offline_ruminator" | "retrieval_evidence_match";
    decision: string;
    decisionReason: string;
  },
): void {
  if (ctx.tracer?.enabled !== true) {
    return;
  }

  ctx.tracer.emit("open_question_resolution_attempt", {
    turnId: ctx.runId,
    oq_id: input.oqId,
    source_path: input.sourcePath,
    decision: input.decision,
    decision_reason: input.decisionReason,
  });
}

async function planResolution(
  ctx: OfflineContext,
  llmClient: LLMClient,
  question: OpenQuestion,
  maxQuestionsPerRun: number,
): Promise<RuminatorPlan["items"][number] | null> {
  const retrieval = await searchResolutionEvidence(ctx, question, maxQuestionsPerRun);
  const freshEvidence = retrieval.episodes.filter(
    (result) => result.episode.updated_at > question.last_touched,
  );
  const strongEvidence = freshEvidence.sort(
    (left, right) => right.score - left.score || right.episode.updated_at - left.episode.updated_at,
  )[0];

  if (strongEvidence === undefined) {
    emitOpenQuestionResolutionAttempt(ctx, {
      oqId: question.id,
      sourcePath: "retrieval_evidence_match",
      decision: "rejected",
      decisionReason: "no_new_visible_evidence",
    });
    return null;
  }

  // Recompute over the merged fresh visible evidence set instead of trusting
  // either retrieval lane's confidence. This prevents a strong global/self lane
  // from authorizing a weaker audience-scoped anchor after the lanes are merged.
  const mergedFreshConfidence = computeRetrievalConfidence({
    episodes: freshEvidence,
    contradictionPresent: false,
    nowMs: ctx.clock.now(),
    expectedCount: retrieval.expectedCount,
  });

  if (mergedFreshConfidence.overall < ctx.config.offline.ruminator.resolveConfidenceThreshold) {
    emitOpenQuestionResolutionAttempt(ctx, {
      oqId: question.id,
      sourcePath: "retrieval_evidence_match",
      decision: "rejected",
      decisionReason: "confidence_below_threshold",
    });
    return null;
  }

  emitOpenQuestionResolutionAttempt(ctx, {
    oqId: question.id,
    sourcePath: "retrieval_evidence_match",
    decision: "matched",
    decisionReason: "confidence_and_fresh_evidence",
  });

  const evidenceBlock = retrieval.episodes
    .slice(0, 3)
    .map((result) =>
      JSON.stringify({
        id: result.episode.id,
        title: result.episode.title,
        narrative: result.episode.narrative,
        tags: result.episode.tags,
        relevance_score: Number(result.score.toFixed(3)),
      }),
    )
    .join("\n");
  const response = parseResolutionResponse(
    await llmClient.complete({
      model: ctx.config.anthropic.models.background,
      system: "You update Borg's open questions conservatively and only from grounded evidence.",
      messages: [
        {
          role: "user",
          content: buildResolutionPrompt(question, evidenceBlock),
        },
      ],
      tools: [RUMINATOR_TOOL],
      tool_choice: { type: "tool", name: RUMINATOR_TOOL_NAME },
      max_tokens: 4_000,
      budget: "offline-ruminator",
    }),
  );
  const growthMarker =
    response.growth_marker === null
      ? null
      : serializableGrowthMarkerSchema.parse({
          id: createGrowthMarkerId(),
          ts: ctx.clock.now(),
          category: response.growth_marker.category,
          what_changed: response.growth_marker.what_changed,
          before_description: response.growth_marker.before_description ?? null,
          after_description: response.growth_marker.after_description ?? null,
          evidence_episode_ids: [strongEvidence.episode.id],
          confidence: Math.min(GROWTH_MARKER_CONFIDENCE_CEILING, response.growth_marker.confidence),
          source_process: "ruminator",
          provenance: {
            kind: "offline",
            process: "ruminator",
          },
          created_at: ctx.clock.now(),
        });

  emitOpenQuestionResolutionAttempt(ctx, {
    oqId: question.id,
    sourcePath: "offline_ruminator",
    decision: "planned_resolution",
    decisionReason: "llm_resolution_candidate",
  });

  return {
    action: "resolve",
    question_id: question.id,
    previous: question,
    resolution_evidence_episode_ids: [strongEvidence.episode.id],
    resolution_evidence_stream_entry_ids: [],
    resolution_note: response.resolution_note.trim(),
    growth_marker: growthMarker,
  };
}

function planFallbackAction(
  ctx: OfflineContext,
  question: OpenQuestion,
  nextUnresolvedRuminationTicks: number,
): RuminatorPlan["items"][number] | null {
  const ageMs = Math.max(0, ctx.clock.now() - question.last_touched);
  const stalenessTicks = ctx.config.offline.ruminator.stalenessTicks;
  const dayStalenessReached =
    ageMs >= ctx.config.offline.ruminator.stalenessDays * DAY_MS && question.urgency < 0.2;
  // Tick staleness is an observed count of unresolved maintenance passes; once
  // it reaches the configured threshold it should not wait for the urgency gate.
  const tickStalenessReached =
    stalenessTicks !== null && question.unresolved_rumination_ticks >= stalenessTicks;

  if (dayStalenessReached || tickStalenessReached) {
    return {
      action: "abandon",
      question_id: question.id,
      previous: question,
      reason: "No relevant new evidence surfaced before the staleness threshold.",
    };
  }

  if (ageMs >= 7 * DAY_MS) {
    const nextUrgency = clamp(question.urgency + 0.05, 0, 1);

    if (nextUrgency > question.urgency) {
      return {
        action: "bump_urgency",
        question_id: question.id,
        previous: question,
        delta: Number((nextUrgency - question.urgency).toFixed(3)),
        next_urgency: nextUrgency,
        next_unresolved_rumination_ticks: nextUnresolvedRuminationTicks,
      };
    }
  }

  return null;
}

export type RuminatorProcessOptions = {
  openQuestionsRepository: OfflineContext["openQuestionsRepository"];
  growthMarkersRepository: OfflineContext["growthMarkersRepository"];
  registry: ReverserRegistry;
};

export class RuminatorProcess implements OfflineProcess<RuminatorPlan> {
  readonly name = "ruminator" as const;

  constructor(private readonly options: RuminatorProcessOptions) {
    const reopenForReversal = createOpenQuestionReopener(this.options.openQuestionsRepository);

    this.options.registry.register(this.name, "resolve", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.previous !== undefined) {
        reopenForReversal(parsed.previous.id, parsed.previous.urgency);
      }
    });
    this.options.registry.register(this.name, "bump_urgency", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.previous !== undefined) {
        this.options.openQuestionsRepository.setUrgency(
          parsed.previous.id,
          parsed.previous.urgency,
        );
      }
    });
    this.options.registry.register(this.name, "abandon", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.previous !== undefined) {
        reopenForReversal(parsed.previous.id, parsed.previous.urgency);
      }
    });
    this.options.registry.register(this.name, "merge_duplicate", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.previous_primary !== undefined) {
        this.options.openQuestionsRepository.restore(parsed.previous_primary);
      }

      if (parsed.previous_duplicate !== undefined) {
        if (this.options.openQuestionsRepository.get(parsed.previous_duplicate.id) === null) {
          this.options.openQuestionsRepository.add({
            id: parsed.previous_duplicate.id,
            question: parsed.previous_duplicate.question,
            urgency: parsed.previous_duplicate.urgency,
            related_episode_ids: parsed.previous_duplicate.related_episode_ids,
            related_semantic_node_ids: parsed.previous_duplicate.related_semantic_node_ids,
            goal_id: parsed.previous_duplicate.goal_id,
            audience_entity_id: parsed.previous_duplicate.audience_entity_id,
            provenance: parsed.previous_duplicate.provenance,
            source: parsed.previous_duplicate.source,
            created_at: parsed.previous_duplicate.created_at,
            last_touched: parsed.previous_duplicate.last_touched,
          });
        }

        this.options.openQuestionsRepository.restore(parsed.previous_duplicate);
      }
    });
    this.options.registry.register(this.name, "add_growth_marker", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.marker_id !== undefined) {
        this.options.growthMarkersRepository.delete(parsed.marker_id);
      }
    });
  }

  async plan(
    ctx: OfflineContext,
    opts: { budget?: number; params?: Record<string, unknown> } = {},
  ) {
    const errors: OfflineProcessError[] = [];
    const items: RuminatorPlan["items"] = [];
    const budget = opts.budget ?? ctx.config.offline.ruminator.budget;
    const maxQuestionsRaw = opts.params?.maxQuestionsPerRun;
    const maxQuestionsPerRun =
      typeof maxQuestionsRaw === "number" &&
      Number.isInteger(maxQuestionsRaw) &&
      maxQuestionsRaw > 0
        ? maxQuestionsRaw
        : ctx.config.offline.ruminator.maxQuestionsPerRun;
    const questions = ctx.openQuestionsRepository.list({
      status: "open",
      limit: maxQuestionsPerRun,
    });
    const duplicateQuestionIds = new Set<OpenQuestion["id"]>();
    let tokensUsed = 0;
    let budgetExhausted = false;

    try {
      const duplicateMerges = await planDuplicateMerges(ctx, questions);
      items.push(...duplicateMerges);

      for (const item of duplicateMerges) {
        if (item.action === "merge_duplicate") {
          duplicateQuestionIds.add(item.duplicate_question_id);
        }
      }
    } catch (error) {
      errors.push(offlineProcessError(this.name, error));
    }

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient = wrapClient(ctx.llm.background);

        for (const question of questions) {
          if (duplicateQuestionIds.has(question.id)) {
            continue;
          }

          try {
            const resolution = await planResolution(ctx, llmClient, question, maxQuestionsPerRun);

            if (resolution !== null) {
              items.push(resolution);
              continue;
            }

            const nextUnresolvedRuminationTicks = question.unresolved_rumination_ticks + 1;
            if (await shouldDismissStaleNoTraction(ctx, question)) {
              items.push({
                action: "abandon",
                question_id: question.id,
                previous: question,
                reason: "stale_no_traction",
              });
              continue;
            }

            const fallback = planFallbackAction(ctx, question, nextUnresolvedRuminationTicks);

            if (fallback !== null) {
              items.push(fallback);
              continue;
            }

            items.push({
              action: "mark_unresolved",
              question_id: question.id,
              previous: question,
              next_unresolved_rumination_ticks: nextUnresolvedRuminationTicks,
            });
          } catch (error) {
            if (error instanceof BudgetExceededError) {
              throw error;
            }

            errors.push(offlineProcessError(this.name, error));
          }
        }
      });

      tokensUsed = budgeted.tokens_used;
    } catch (error) {
      tokensUsed = getBudgetErrorTokens(error);
      budgetExhausted = error instanceof BudgetExceededError;
      errors.push(offlineProcessError(this.name, error));
    }

    return ruminatorPlanSchema.parse({
      process: this.name,
      items,
      errors,
      tokens_used: tokensUsed,
      budget_exhausted: budgetExhausted,
    });
  }

  preview(plan: RuminatorPlan): OfflineResult {
    const parsed = ruminatorPlanSchema.parse(plan);

    return {
      process: this.name,
      dryRun: true,
      changes: parsed.items.map((item) => buildChange(item)).filter(isOfflineChange),
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
    };
  }

  async apply(ctx: OfflineContext, rawPlan: RuminatorPlan): Promise<OfflineResult> {
    const plan = ruminatorPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];
    const processProvenance = {
      kind: "offline" as const,
      process: this.name,
    };

    for (const item of plan.items) {
      if (item.action === "mark_unresolved") {
        ctx.openQuestionsRepository.markRuminated(
          item.question_id,
          item.next_unresolved_rumination_ticks,
        );
        continue;
      }

      if (item.action === "resolve") {
        const current = ctx.openQuestionsRepository.get(item.question_id);

        if (current === null) {
          throw new StorageError(`Missing open question for ruminator plan: ${item.question_id}`, {
            code: "RUMINATOR_PLAN_INVALID",
          });
        }

        if (
          current.status !== "resolved" ||
          current.resolution_evidence_episode_ids.length !==
            item.resolution_evidence_episode_ids.length ||
          current.resolution_evidence_episode_ids.some(
            (episodeId, index) => episodeId !== item.resolution_evidence_episode_ids[index],
          ) ||
          current.resolution_evidence_stream_entry_ids.length !==
            item.resolution_evidence_stream_entry_ids.length ||
          current.resolution_note !== item.resolution_note
        ) {
          ctx.identityService.resolveOpenQuestion(
            item.question_id,
            {
              resolution_evidence_episode_ids: item.resolution_evidence_episode_ids,
              resolution_evidence_stream_entry_ids: item.resolution_evidence_stream_entry_ids,
              resolution_note: item.resolution_note,
            },
            processProvenance,
            {
              throughReview: true,
            },
          );
          emitOpenQuestionResolutionAttempt(ctx, {
            oqId: item.question_id,
            sourcePath: "offline_ruminator",
            decision: "applied",
            decisionReason: "through_review",
          });
        } else {
          emitOpenQuestionResolutionAttempt(ctx, {
            oqId: item.question_id,
            sourcePath: "offline_ruminator",
            decision: "already_applied",
            decisionReason: "matching_resolution_present",
          });
        }

        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "resolve",
          targets: {
            question_id: item.question_id,
            resolution_evidence_episode_ids: item.resolution_evidence_episode_ids,
          },
          reversal: {
            previous: item.previous,
          } satisfies RuminatorReversal,
        });

        if (
          item.growth_marker !== null &&
          ctx.growthMarkersRepository.get(item.growth_marker.id) === null
        ) {
          ctx.identityService.addGrowthMarker(item.growth_marker);
          ctx.auditLog.record({
            run_id: ctx.runId,
            process: this.name,
            action: "add_growth_marker",
            targets: {
              marker_id: item.growth_marker.id,
              question_id: item.question_id,
            },
            reversal: {
              marker_id: item.growth_marker.id,
            } satisfies RuminatorReversal,
          });
        }

        const change = buildChange(item);

        if (change !== null) {
          changes.push(change);
        }
        continue;
      }

      if (item.action === "abandon") {
        const current = ctx.openQuestionsRepository.get(item.question_id);

        if (current === null) {
          throw new StorageError(`Missing open question for ruminator plan: ${item.question_id}`, {
            code: "RUMINATOR_PLAN_INVALID",
          });
        }

        if (current.status !== "abandoned" || current.abandoned_reason !== item.reason) {
          ctx.identityService.abandonOpenQuestion(
            item.question_id,
            item.reason,
            processProvenance,
            {
              throughReview: true,
            },
          );
        }

        if (item.reason === "stale_no_traction" && ctx.tracer?.enabled === true) {
          ctx.tracer.emit("open_question_stale_dismissed", {
            turnId: ctx.runId,
            question_id: item.question_id,
            reason: item.reason,
          });
        }

        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "abandon",
          targets: {
            question_id: item.question_id,
          },
          reversal: {
            previous: item.previous,
          } satisfies RuminatorReversal,
        });
        const change = buildChange(item);

        if (change !== null) {
          changes.push(change);
        }
        continue;
      }

      if (item.action === "merge_duplicate") {
        const primary = ctx.openQuestionsRepository.get(item.primary_question_id);
        const duplicate = ctx.openQuestionsRepository.get(item.duplicate_question_id);

        if (primary === null) {
          throw new StorageError(
            `Missing primary open question for ruminator plan: ${item.primary_question_id}`,
            {
              code: "RUMINATOR_PLAN_INVALID",
            },
          );
        }

        if (duplicate !== null && duplicate.status === "open") {
          ctx.identityService.updateOpenQuestion(
            primary.id,
            {
              urgency: Math.max(primary.urgency, duplicate.urgency),
              goal_id: primary.goal_id ?? duplicate.goal_id,
              related_episode_ids: mergeQuestionIds(
                primary.related_episode_ids,
                primary.resolution_evidence_episode_ids,
                duplicate.related_episode_ids,
                duplicate.resolution_evidence_episode_ids,
                openQuestionProvenanceEpisodeIds(duplicate),
              ),
              related_semantic_node_ids: mergeSemanticNodeIds(
                primary.related_semantic_node_ids,
                duplicate.related_semantic_node_ids,
              ),
              resolution_evidence_episode_ids: mergeQuestionIds(
                primary.resolution_evidence_episode_ids,
                duplicate.resolution_evidence_episode_ids,
              ),
              resolution_evidence_stream_entry_ids: mergeQuestionStreamIds(
                primary.resolution_evidence_stream_entry_ids,
                duplicate.resolution_evidence_stream_entry_ids,
                openQuestionProvenanceStreamEntryIds(duplicate),
              ),
            },
            processProvenance,
            {
              throughReview: true,
              reason: "open_question_duplicate_merge",
              preserveRecordProvenance: true,
            },
          );
          await ctx.openQuestionsRepository.delete(duplicate.id);

          if (ctx.tracer?.enabled === true) {
            ctx.tracer.emit("open_question_merged", {
              turnId: ctx.runId,
              kept_oq_id: primary.id,
              deleted_oq_id: duplicate.id,
              similarity_score: item.similarity,
              evidence_folded_count: countOpenQuestionEvidenceHandles(duplicate),
            });
          }
        }

        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "merge_duplicate",
          targets: {
            primary_question_id: item.primary_question_id,
            duplicate_question_id: item.duplicate_question_id,
          },
          reversal: {
            previous_primary: item.previous_primary,
            previous_duplicate: item.previous_duplicate,
          } satisfies RuminatorReversal,
        });
        const change = buildChange(item);

        if (change !== null) {
          changes.push(change);
        }
        continue;
      }

      const current = ctx.openQuestionsRepository.get(item.question_id);

      if (current === null) {
        throw new StorageError(`Missing open question for ruminator plan: ${item.question_id}`, {
          code: "RUMINATOR_PLAN_INVALID",
        });
      }

      if (Math.abs(current.urgency - item.next_urgency) > 1e-6) {
        ctx.identityService.bumpOpenQuestionUrgency(
          item.question_id,
          item.next_urgency - current.urgency,
          processProvenance,
          {
            throughReview: true,
          },
        );
      }

      ctx.openQuestionsRepository.markRuminated(
        item.question_id,
        item.next_unresolved_rumination_ticks,
      );

      ctx.auditLog.record({
        run_id: ctx.runId,
        process: this.name,
        action: "bump_urgency",
        targets: {
          question_id: item.question_id,
        },
        reversal: {
          previous: item.previous,
        } satisfies RuminatorReversal,
      });
      const change = buildChange(item);

      if (change !== null) {
        changes.push(change);
      }
    }

    return {
      process: this.name,
      dryRun: false,
      changes,
      tokens_used: plan.tokens_used,
      errors: plan.errors,
      budget_exhausted: plan.budget_exhausted,
    };
  }

  async run(
    ctx: OfflineContext,
    opts: { dryRun?: boolean; budget?: number; params?: Record<string, unknown> },
  ): Promise<OfflineResult> {
    const plan = await this.plan(ctx, opts);
    return opts.dryRun === true ? this.preview(plan) : this.apply(ctx, plan);
  }
}
