import { z } from "zod";

import { computeWeights } from "../../cognition/attention/index.js";
import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { episodeIdSchema } from "../../memory/episodic/index.js";
import { memoryDisclosureLabelSchema } from "../../memory/common/disclosure-label.js";
import {
  growthMarkerCategorySchema,
  growthMarkerIdSchema,
  growthMarkerSchema,
  openQuestionIdSchema,
  openQuestionResolutionStreamEntryIdSchema,
  openQuestionSchema,
  type OpenQuestion,
  type OpenQuestionRumination,
} from "../../memory/self/index.js";
import { expectedRecordVersion } from "../../memory/common/cas.js";
import { resolveOpenQuestionThroughIdentityService } from "../../memory/lifecycle-ops/index.js";
import {
  SELF_RECALL_SCOPE,
  computeRetrievalConfidence,
  type RetrievedEpisode,
} from "../../retrieval/index.js";
import {
  memoryDisclosurePayloadFields,
  openQuestionMemoryDisclosureLabel,
} from "../../memory/common/disclosure-serializers.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromEpisodeAccess,
  selfPrivateMemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import { createGrowthMarkerId, DEFAULT_SESSION_ID } from "../../util/ids.js";
import { BudgetExceededError, StorageError } from "../../util/errors.js";
import { clamp } from "../../util/math.js";
import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";

import type { ReverserRegistry } from "../audit-log.js";
import { getBudgetErrorTokens, withBudget } from "../budget.js";
import { offlineProcessError } from "../process-errors.js";
import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessError,
  OfflineProcessRunOptions,
  OfflineResult,
} from "../types.js";
import { episodeEvidencePromptRow } from "../evidence-labels.js";

const DAY_MS = 24 * 60 * 60 * 1_000;
const GROWTH_MARKER_CONFIDENCE_CEILING = 0.6;
const OPEN_QUESTION_ACTIVE_ACTION_STATES = ["considering", "committed_to_do", "scheduled"] as const;
const RECENT_RUMINATIONS_PER_QUESTION = 3;
const CONNECTED_OPEN_QUESTION_CANDIDATE_LIMIT = 8;

export const RUMINATOR_SYSTEM_PROMPT = [
  "I return to my open questions with time to think.",
  "I weigh what has accumulated since I last visited a question: what now pulls toward an answer, what still resists, and what has shifted even when the question remains open.",
  "When the evidence genuinely settles a question, I resolve it plainly and write what that resolution changes for me.",
  "When the question is not settled, I keep it open with articulated reasoning: the live tensions, what evidence would settle it, and any movement in my understanding.",
  "I notice when questions connect: when two uncertainties are the same uncertainty, or when movement on one question changes another.",
  "Groundedness is a value, not a cage. I invent no evidence and cite only what is supplied, but I do not stay timid when the evidence is enough.",
].join(" ");

const ruminatorGrowthMarkerResponseSchema = z
  .object({
    what_changed: z.string().min(1),
    before_description: z.string().nullable().optional(),
    after_description: z.string().nullable().optional(),
    confidence: z.number().min(0).max(1),
    category: growthMarkerCategorySchema,
  })
  .nullable();

// The model intermittently emits `tensions` as a single string (or a JSON-encoded
// array) instead of an array, despite the array tool schema + description below.
// Normalize the tool-call shape at parse time (tool-shape hygiene, not output
// policing) so a shape slip does not discard an otherwise-valid rumination.
const tolerantTensionsSchema = z.preprocess(
  (value) => {
    if (typeof value !== "string") {
      return value;
    }
    const trimmed = value.trim();
    if (trimmed.startsWith("[")) {
      try {
        const parsed = JSON.parse(trimmed) as unknown;
        if (Array.isArray(parsed)) {
          return parsed;
        }
      } catch {
        // Not JSON; fall through to the single-element wrap below.
      }
    }
    return trimmed.length > 0 ? [trimmed] : [];
  },
  z.array(z.string().min(1)),
);

const ruminationResponseToolSchema = z.object({
  outcome: z.enum(["resolved", "still_open"]),
  resolution_note: z.string().min(1).nullable().optional(),
  growth_marker: ruminatorGrowthMarkerResponseSchema.optional(),
  reasoning: z.string().min(1).nullable().optional(),
  tensions: z
    .array(z.string().min(1))
    .describe(
      "The live tensions, each distinct tension as its own array element. Always an array, even for a single tension -- never one combined string.",
    )
    .optional(),
  connected_open_question_ids: z.array(openQuestionIdSchema).optional(),
});
// Tolerant variant used to PARSE the model's tool call (the strict schema above is
// what the model is shown). Only `tensions` differs: it accepts a stray string.
const ruminationResponseParseSchema = ruminationResponseToolSchema.extend({
  tensions: tolerantTensionsSchema.optional(),
});
const resolvedRuminationResponseSchema = z.object({
  outcome: z.literal("resolved"),
  resolution_note: z.string().min(1),
  growth_marker: ruminatorGrowthMarkerResponseSchema.default(null),
});
const stillOpenRuminationResponseSchema = z.object({
  outcome: z.literal("still_open"),
  reasoning: z.string().min(1),
  tensions: tolerantTensionsSchema,
  connected_open_question_ids: z.array(openQuestionIdSchema),
});
type RuminationResponse =
  | z.infer<typeof resolvedRuminationResponseSchema>
  | z.infer<typeof stillOpenRuminationResponseSchema>;
const RUMINATOR_TOOL_NAME = "EmitRuminatorDecisions";
export const RUMINATOR_TOOL = {
  name: RUMINATOR_TOOL_NAME,
  description:
    "Emit either a grounded open-question resolution or a still-open deliberation note with tensions and connected questions.",
  inputSchema: toToolInputSchema(ruminationResponseToolSchema),
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
    resolution_disclosure_label: memoryDisclosureLabelSchema,
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
    rumination_note: z.string().min(1).nullable().default(null),
    tensions: z.array(z.string().min(1)).default([]),
    connected_open_question_ids: z.array(openQuestionIdSchema).default([]),
    evidence_episode_ids: z.array(episodeIdSchema).default([]),
    evidence_stream_entry_ids: z.array(openQuestionResolutionStreamEntryIdSchema).default([]),
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

// Re-materialise a previously-deleted open question from a captured snapshot.
// Used both to reverse a merge and to roll back a duplicate we optimistically
// removed when folding it into a primary failed. add() re-inserts the row (its
// dedupe_key is free again post-delete); restore() then reinstates full state.
function reinsertOpenQuestion(
  repository: OfflineContext["openQuestionsRepository"],
  question: OpenQuestion,
): void {
  if (repository.get(question.id) === null) {
    repository.add({
      id: question.id,
      question: question.question,
      urgency: question.urgency,
      related_episode_ids: question.related_episode_ids,
      related_semantic_node_ids: question.related_semantic_node_ids,
      goal_id: question.goal_id,
      audience_entity_id: question.audience_entity_id,
      provenance: question.provenance,
      source: question.source,
      created_at: question.created_at,
      last_touched: question.last_touched,
    });
  }

  repository.restore(question);
}

function renderRecentRuminationNotes(ruminations: readonly OpenQuestionRumination[]): string {
  if (ruminations.length === 0) {
    return "[]";
  }

  return ruminations
    .map((rumination) =>
      JSON.stringify({
        created_at: rumination.created_at,
        note: rumination.note,
        tensions: rumination.tensions,
        connected_open_question_ids: rumination.connected_open_question_ids,
        evidence_episode_ids: rumination.evidence_episode_ids,
        evidence_stream_entry_ids: rumination.evidence_stream_entry_ids,
        ...memoryDisclosurePayloadFields(selfPrivateMemoryDisclosureLabel()),
      }),
    )
    .join("\n");
}

function connectedOpenQuestionCandidates(
  question: OpenQuestion,
  allOpenQuestions: readonly OpenQuestion[],
): OpenQuestion[] {
  return allOpenQuestions
    .filter((candidate) => candidate.id !== question.id && candidate.status === "open")
    .sort(
      (left, right) =>
        right.last_touched - left.last_touched ||
        right.urgency - left.urgency ||
        right.created_at - left.created_at ||
        left.id.localeCompare(right.id),
    )
    .slice(0, CONNECTED_OPEN_QUESTION_CANDIDATE_LIMIT);
}

function renderConnectedOpenQuestionCandidates(candidates: readonly OpenQuestion[]): string {
  if (candidates.length === 0) {
    return "[]";
  }

  return candidates
    .map((candidate) =>
      JSON.stringify({
        id: candidate.id,
        question: candidate.question,
        urgency: candidate.urgency,
        last_touched: candidate.last_touched,
        source: candidate.source,
        ...memoryDisclosurePayloadFields(openQuestionMemoryDisclosureLabel(candidate)),
      }),
    )
    .join("\n");
}

function buildResolutionPrompt(
  question: OpenQuestion,
  evidence: string,
  recentRuminations: readonly OpenQuestionRumination[],
  connectedCandidates: readonly OpenQuestion[],
): string {
  const questionRow = {
    id: question.id,
    question: question.question,
    source: question.source,
    ...memoryDisclosurePayloadFields(openQuestionMemoryDisclosureLabel(question)),
  };

  return [
    "I turn over this open question using only the evidence and recent self-private rumination notes below.",
    `I emit my result by calling the ${RUMINATOR_TOOL_NAME} tool exactly once.`,
    "I choose outcome=resolved only when the evidence genuinely settles the question.",
    "I choose outcome=still_open when the question should remain open, and then I state the reasoning, live tensions, connected open questions, and what evidence would settle it.",
    "For connected_open_question_ids, I cite only ids from Connected open-question candidates. I use [] when none of those prompt-visible questions genuinely connects.",
    "I only include a growth_marker on a resolved outcome when the evidence clearly shows new understanding.",
    `${SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE} I apply this to resolution_note, reasoning, tensions, and any growth_marker text fields.`,
    "Open question:",
    JSON.stringify(questionRow),
    "Connected open-question candidates:",
    renderConnectedOpenQuestionCandidates(connectedCandidates),
    "Evidence:",
    evidence,
    "Recent rumination notes:",
    renderRecentRuminationNotes(recentRuminations),
  ].join("\n\n");
}

function invalidResolutionResponse(error: unknown): unknown {
  if (isStructuredToolCallError(error, "missing_tool_call")) {
    return new StorageError(`Ruminator did not emit tool ${RUMINATOR_TOOL_NAME}`, {
      code: "RUMINATOR_INVALID",
    });
  }

  if (
    isStructuredToolCallError(error, "invalid_payload") ||
    isStructuredToolCallError(error, "llm_failed")
  ) {
    return error.cause ?? error;
  }

  return error;
}

function parseResolutionResponse(input: unknown) {
  const parsed = ruminationResponseParseSchema.parse(input);

  if (parsed.outcome === "resolved") {
    return resolvedRuminationResponseSchema.parse(parsed);
  }

  return stillOpenRuminationResponseSchema.parse(parsed);
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

function shareOpenQuestionEntityScope(left: OpenQuestion, right: OpenQuestion): boolean {
  const leftEmpty = left.related_semantic_node_ids.length === 0;
  const rightEmpty = right.related_semantic_node_ids.length === 0;

  if (leftEmpty && rightEmpty) {
    return true;
  }

  if (leftEmpty !== rightEmpty) {
    return false;
  }

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
    ...question.related_semantic_node_ids.map((nodeId) => `semantic:${nodeId}`),
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
    if (mergedDuplicateIds.has(question.id)) {
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
  dryRun: boolean,
): Promise<{
  episodes: RetrievedEpisode[];
  expectedCount: number;
}> {
  const baseOptions = {
    limit: Math.max(3, maxQuestionsPerRun),
    attentionWeights: buildReflectionWeights(ctx),
    goalDescriptions: ctx.goalsRepository
      .list({ status: "active" })
      .map((goal) => goal.description),
    includeOpenQuestions: false,
    recordRetrieval: !dryRun,
  };
  const retrieval = await ctx.retrievalPipeline.recallEpisodesForCognition(question.question, {
    ...baseOptions,
    limit: Math.max(baseOptions.limit * 5, 20),
    recallContext: {
      reader: SELF_RECALL_SCOPE,
      currentSessionId: DEFAULT_SESSION_ID,
      currentAudienceEntityId: question.audience_entity_id,
      currentParticipantEntityIds:
        question.audience_entity_id === null ? [] : [question.audience_entity_id],
    },
  });

  return {
    episodes: retrieval.episodes,
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

  ctx.tracer.emit("open_question_resolution.started", {
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
  allOpenQuestions: readonly OpenQuestion[],
  dryRun: boolean,
): Promise<RuminatorPlan["items"][number] | null> {
  const retrieval = await searchResolutionEvidence(ctx, question, maxQuestionsPerRun, dryRun);
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
      decisionReason: "no_new_evidence",
    });
    return null;
  }

  // Recompute over the fresh global evidence set instead of trusting one top hit.
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

  const renderedEvidence = [
    strongEvidence,
    ...retrieval.episodes
      .filter((result) => result.episode.id !== strongEvidence.episode.id)
      .slice(0, 2),
  ];
  const citedEvidence = renderedEvidence;
  const sourceDisclosureLabel = combineMemoryDisclosureLabels(
    citedEvidence.map((result) => memoryDisclosureLabelFromEpisodeAccess(result.episode)),
  );
  // Persist cited evidence ids deterministically: the strong (primary) episode first, then the
  // remaining cited ids deduped and sorted. The prompt below stays in relevance order and the
  // disclosure label is order-independent; only the stored arrays need canonical ordering so the
  // persisted resolution/growth marker is stable across runs.
  const citedEvidenceEpisodeIds = [
    strongEvidence.episode.id,
    ...[
      ...new Set(
        citedEvidence
          .map((result) => result.episode.id)
          .filter((episodeId) => episodeId !== strongEvidence.episode.id),
      ),
    ].sort(),
  ];
  const evidenceBlock = renderedEvidence
    .map((result) =>
      JSON.stringify(
        episodeEvidencePromptRow(result.episode, {
          tags: result.episode.tags,
          relevance_score: Number(result.score.toFixed(3)),
        }),
      ),
    )
    .join("\n");
  const recentRuminations = ctx.openQuestionsRepository.listRecentRuminations(question.id, {
    limit: RECENT_RUMINATIONS_PER_QUESTION,
  });
  const connectedCandidates = connectedOpenQuestionCandidates(question, allOpenQuestions);
  let response: RuminationResponse;

  try {
    response = (
      await callStructuredTool({
        llmClient,
        request: {
          model: ctx.config.anthropic.models.background,
          system: RUMINATOR_SYSTEM_PROMPT,
          messages: [
            {
              role: "user",
              content: buildResolutionPrompt(
                question,
                evidenceBlock,
                recentRuminations,
                connectedCandidates,
              ),
            },
          ],
          tools: [RUMINATOR_TOOL],
          tool_choice: { type: "tool", name: RUMINATOR_TOOL_NAME },
          max_tokens: 4_000,
          budget: "offline-ruminator",
        },
        toolName: RUMINATOR_TOOL_NAME,
        parse: parseResolutionResponse,
      })
    ).parsed;
  } catch (error) {
    throw invalidResolutionResponse(error);
  }

  if (response.outcome === "still_open") {
    const nextUnresolvedRuminationTicks = question.unresolved_rumination_ticks + 1;

    emitOpenQuestionResolutionAttempt(ctx, {
      oqId: question.id,
      sourcePath: "offline_ruminator",
      decision: "planned_still_open",
      decisionReason: "llm_still_open_deliberation",
    });

    return {
      action: "mark_unresolved",
      question_id: question.id,
      previous: question,
      next_unresolved_rumination_ticks: nextUnresolvedRuminationTicks,
      rumination_note: response.reasoning.trim(),
      tensions: response.tensions
        .map((tension) => tension.trim())
        .filter((value) => value.length > 0),
      connected_open_question_ids: response.connected_open_question_ids,
      evidence_episode_ids: citedEvidenceEpisodeIds,
      evidence_stream_entry_ids: [],
    };
  }

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
          evidence_episode_ids: citedEvidenceEpisodeIds,
          disclosure_label: sourceDisclosureLabel,
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
    resolution_evidence_episode_ids: citedEvidenceEpisodeIds,
    resolution_evidence_stream_entry_ids: [],
    resolution_disclosure_label: sourceDisclosureLabel,
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
  const dayStalenessReached =
    ageMs >= ctx.config.offline.ruminator.stalenessDays * DAY_MS && question.urgency < 0.2;

  if (dayStalenessReached) {
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

function validateConnectedOpenQuestionIds(
  ctx: OfflineContext,
  questionId: OpenQuestion["id"],
  connectedOpenQuestionIds: readonly OpenQuestion["id"][],
): OpenQuestion["id"][] {
  const validIds: OpenQuestion["id"][] = [];
  const droppedIds: OpenQuestion["id"][] = [];

  for (const connectedId of [...new Set(connectedOpenQuestionIds)]) {
    const connected = ctx.openQuestionsRepository.get(connectedId);

    if (connectedId !== questionId && connected?.status === "open") {
      validIds.push(connectedId);
    } else {
      droppedIds.push(connectedId);
    }
  }

  if (droppedIds.length > 0 && ctx.tracer?.enabled === true) {
    ctx.tracer.emit("open_question_rumination.connected_ids_dropped", {
      turnId: ctx.runId,
      oq_id: questionId,
      dropped_connected_open_question_ids: droppedIds,
      reason: "missing_or_not_open",
    });
  }

  return validIds;
}

export type RuminatorProcessOptions = {
  openQuestionsRepository: OfflineContext["openQuestionsRepository"];
  growthMarkersRepository: OfflineContext["growthMarkersRepository"];
  registry: ReverserRegistry;
};

export class RuminatorProcess implements OfflineProcess<RuminatorPlan> {
  readonly name = "ruminator" as const;

  constructor(private readonly options: RuminatorProcessOptions) {
    this.options.registry.register(this.name, "resolve", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.previous !== undefined) {
        this.options.openQuestionsRepository.reopenForReversal(
          parsed.previous.id,
          parsed.previous.urgency,
        );
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
        this.options.openQuestionsRepository.reopenForReversal(
          parsed.previous.id,
          parsed.previous.urgency,
        );
      }
    });
    this.options.registry.register(this.name, "merge_duplicate", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.previous_primary !== undefined) {
        this.options.openQuestionsRepository.restore(parsed.previous_primary);
      }

      if (parsed.previous_duplicate !== undefined) {
        reinsertOpenQuestion(this.options.openQuestionsRepository, parsed.previous_duplicate);
      }
    });
    this.options.registry.register(this.name, "add_growth_marker", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.marker_id !== undefined) {
        this.options.growthMarkersRepository.delete(parsed.marker_id);
      }
    });
  }

  async plan(ctx: OfflineContext, opts: OfflineProcessRunOptions = {}) {
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
    const allOpenQuestions = ctx.openQuestionsRepository.list({
      status: "open",
      limit: 10_000,
    });
    const duplicateQuestionIds = new Set<OpenQuestion["id"]>();
    const mergeParticipantIds = new Set<OpenQuestion["id"]>();
    const staleDismissedIds = new Set<OpenQuestion["id"]>();
    let tokensUsed = 0;
    let budgetExhausted = false;

    try {
      const duplicateMerges = await planDuplicateMerges(ctx, allOpenQuestions);
      items.push(...duplicateMerges);

      for (const item of duplicateMerges) {
        if (item.action === "merge_duplicate") {
          duplicateQuestionIds.add(item.duplicate_question_id);
          mergeParticipantIds.add(item.duplicate_question_id);
          mergeParticipantIds.add(item.primary_question_id);
        }
      }
    } catch (error) {
      errors.push(offlineProcessError(this.name, error));
    }

    const llmWindowIds = new Set(questions.map((question) => question.id));

    try {
      for (const question of allOpenQuestions) {
        if (mergeParticipantIds.has(question.id) || llmWindowIds.has(question.id)) {
          continue;
        }

        if (await shouldDismissStaleNoTraction(ctx, question)) {
          items.push({
            action: "abandon",
            question_id: question.id,
            previous: question,
            reason: "stale_no_traction",
          });
          staleDismissedIds.add(question.id);
        }
      }
    } catch (error) {
      errors.push(offlineProcessError(this.name, error));
    }

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient = wrapClient(ctx.llm.background);

        for (const question of questions) {
          if (mergeParticipantIds.has(question.id) || staleDismissedIds.has(question.id)) {
            continue;
          }

          try {
            const resolution = await planResolution(
              ctx,
              llmClient,
              question,
              maxQuestionsPerRun,
              allOpenQuestions,
              opts.dryRun === true,
            );

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
              rumination_note: null,
              tensions: [],
              connected_open_question_ids: [],
              evidence_episode_ids: [],
              evidence_stream_entry_ids: [],
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
        const current = ctx.openQuestionsRepository.get(item.question_id);
        const shouldRecordRumination =
          current !== null &&
          current.status === "open" &&
          current.unresolved_rumination_ticks < item.next_unresolved_rumination_ticks;
        ctx.openQuestionsRepository.markRuminated(
          item.question_id,
          item.next_unresolved_rumination_ticks,
          {
            expectedVersion: expectedRecordVersion(item.previous),
          },
        );
        if (shouldRecordRumination && item.rumination_note !== null) {
          const connectedOpenQuestionIds = validateConnectedOpenQuestionIds(
            ctx,
            item.question_id,
            item.connected_open_question_ids,
          );
          ctx.openQuestionsRepository.recordRumination({
            open_question_id: item.question_id,
            note: item.rumination_note,
            tensions: item.tensions,
            connected_open_question_ids: connectedOpenQuestionIds,
            evidence_episode_ids: item.evidence_episode_ids,
            evidence_stream_entry_ids: item.evidence_stream_entry_ids,
            source_process: this.name,
            source_run_id: ctx.runId,
            source_turn_id: null,
            provenance: processProvenance,
          });
        }
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
          current.resolution_note !== item.resolution_note ||
          JSON.stringify(current.resolution_disclosure_label) !==
            JSON.stringify(item.resolution_disclosure_label)
        ) {
          const result = resolveOpenQuestionThroughIdentityService({
            openQuestionId: item.question_id,
            identityService: ctx.identityService,
            resolution: {
              resolution_evidence_episode_ids: item.resolution_evidence_episode_ids,
              resolution_evidence_stream_entry_ids: item.resolution_evidence_stream_entry_ids,
              resolution_disclosure_label: item.resolution_disclosure_label,
              resolution_note: item.resolution_note,
            },
            provenance: processProvenance,
            options: {
              throughReview: true,
            },
            tracer: ctx.tracer,
            turnId: ctx.runId,
            traceSourcePath: "offline_ruminator",
            traceDecisionReason: "through_review",
          });
          if (result.status === "conflict") {
            throw result.error;
          }
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
          ctx.tracer.emit("open_question_resolution.rejected", {
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
          // Remove the duplicate BEFORE folding its evidence into the primary.
          // The fold recomputes the primary's dedupe_key from the merged id set;
          // if the duplicate still existed, that recomputed key could collide
          // with the duplicate's own key (a UNIQUE violation on
          // open_questions.dedupe_key that previously aborted the whole run).
          // Deleting first frees the key. If the subsequent fold fails anyway
          // (e.g. the merged key collides with a third question), we restore the
          // duplicate so the merge stays all-or-nothing.
          await ctx.openQuestionsRepository.delete(duplicate.id, {
            expectedVersion: expectedRecordVersion(duplicate),
          });

          try {
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
          } catch (error) {
            reinsertOpenQuestion(ctx.openQuestionsRepository, duplicate);
            throw error;
          }

          if (ctx.tracer?.enabled === true) {
            ctx.tracer.emit("open_question_resolution.transitioned", {
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

      let markExpectedVersion = expectedRecordVersion(current);

      if (Math.abs(current.urgency - item.next_urgency) > 1e-6) {
        const result = ctx.identityService.bumpOpenQuestionUrgency(
          item.question_id,
          item.next_urgency - current.urgency,
          processProvenance,
          {
            throughReview: true,
          },
        );

        if (result.status === "applied") {
          markExpectedVersion = expectedRecordVersion(result.record);
        }
      }

      ctx.openQuestionsRepository.markRuminated(
        item.question_id,
        item.next_unresolved_rumination_ticks,
        {
          expectedVersion: markExpectedVersion,
        },
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
