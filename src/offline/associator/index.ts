import { z } from "zod";

import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { episodeIdSchema, type Episode, type EpisodeStats } from "../../memory/episodic/index.js";
import { memoryDisclosureLabelSchema } from "../../memory/common/disclosure-label.js";
import {
  buildOpenQuestionDedupeKey,
  buildOpenQuestionDuplicatePresentation,
  buildOpenQuestionReinforcementPatch,
  findOpenQuestionDuplicateBackstop,
  openQuestionIdSchema,
  openQuestionSchema,
  type OpenQuestionDuplicatePresentation,
} from "../../memory/self/index.js";
import {
  semanticEdgeIdSchema,
  semanticNodeCorrectionRefSchema,
  semanticNodeIdSchema,
  semanticNodeKindSchema,
  semanticNodeSchema,
  semanticNodeStatusSchema,
  type SemanticNode,
} from "../../memory/semantic/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { BudgetExceededError, SemanticError } from "../../util/errors.js";
import { createSemanticEdgeId, createSemanticNodeId } from "../../util/ids.js";
import { clamp } from "../../util/math.js";

import type { ReverserRegistry } from "../audit-log.js";
import { getBudgetErrorTokens, withBudget } from "../budget.js";
import { disclosureLabelForEpisodeIds, episodeEvidencePromptRow } from "../evidence-labels.js";
import { offlineProcessError } from "../process-errors.js";
import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessError,
  OfflineResult,
} from "../types.js";

export const ASSOCIATOR_PROMPT = [
  "I am in unstructured associative time.",
  "The episodes before me were drawn deliberately from across my life, not because they already resemble one another.",
  "I let distant experiences sit together and notice only what genuinely connects.",
  "A real connection earns its keep structurally: shared vocabulary is not shared structure, and a parallel that holds only at the level of words is not a finding.",
  "Finding nothing is an honest and common outcome of a sweep like this. Emptiness is not failure.",
  "When something does connect, I decide its weight honestly.",
  "If it is a weak hypothesis worth holding, I write it as an open question with low urgency by default.",
  "Before writing an open question, I compare it with every supplied open-question candidate. The candidates are global across audiences and sources. Rephrasing, translation, audience, source, and elapsed time do not make a second question when the underlying unresolved uncertainty is the same.",
  "When a supplied open question already covers the proposed uncertainty, I set duplicate_of_open_question_id to that supplied id. I never name an id outside the supplied candidate rows.",
  "If it is a pattern I would assert subject to review, I write it as an insight with conservative confidence.",
  "I cite only the episode ids that actually ground the finding.",
  "I do not force symmetry, analogy, or closure. I return an empty findings list when the supplied episodes do not responsibly support a finding.",
].join("\n");

const ASSOCIATOR_TOOL_NAME = "EmitAssociations";

const associatorOpenQuestionFindingSchema = z.object({
  kind: z.literal("open_question"),
  question: z.string().min(1),
  urgency: z.number().min(0).max(1),
  source_episode_ids: z.array(z.string().min(1)).min(1),
  duplicate_of_open_question_id: openQuestionIdSchema.nullable().default(null),
});

const associatorNewInsightFindingSchema = z.object({
  kind: z.literal("new_insight"),
  label: z.string().min(1),
  description: z.string().min(1),
  confidence: z.number().min(0).max(1),
  source_episode_ids: z.array(z.string().min(1)).min(1),
});

const associatorFindingResponseSchema = z.discriminatedUnion("kind", [
  associatorOpenQuestionFindingSchema,
  associatorNewInsightFindingSchema,
]);

const associatorResponseSchema = z.object({
  // Required, not defaulted: the model explicitly returns [] when nothing
  // genuinely connects -- emptiness is a stated outcome, not a fallback.
  findings: z.array(associatorFindingResponseSchema),
});

export const ASSOCIATOR_TOOL = {
  name: ASSOCIATOR_TOOL_NAME,
  description:
    "Emit cross-life associative findings as either open questions or review-gated insights.",
  inputSchema: toToolInputSchema(associatorResponseSchema),
} satisfies LLMToolDefinition;

const serializableSemanticNodeSchema = z.object({
  id: semanticNodeIdSchema,
  kind: semanticNodeKindSchema,
  label: z.string().min(1),
  description: z.string().min(1),
  domain: z.string().min(1).nullable().default(null),
  aliases: z.array(z.string().min(1)),
  confidence: z.number().min(0).max(1),
  source_episode_ids: z.array(episodeIdSchema).min(1),
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
  last_verified_at: z.number().finite(),
  embedding: z.array(z.number().finite()),
  archived: z.boolean(),
  superseded_by: semanticNodeIdSchema.nullable(),
  status: semanticNodeStatusSchema.default("active"),
  corrected_by: semanticNodeCorrectionRefSchema.nullable().default(null),
  superseded_at: z.number().finite().nullable().default(null),
});

const associatorTargetSchema = z.discriminatedUnion("mode", [
  z.object({
    mode: z.literal("insert"),
    node: serializableSemanticNodeSchema,
  }),
  z.object({
    mode: z.literal("update"),
    node_id: semanticNodeIdSchema,
    patch: z.object({
      description: z.string().min(1),
      confidence: z.number().min(0).max(1),
      source_episode_ids: z.array(episodeIdSchema).min(1),
      last_verified_at: z.number().finite(),
      embedding: z.array(z.number().finite()),
      archived: z.boolean(),
    }),
  }),
]);

const associatorSupportEdgeCandidateSchema = z.object({
  id: semanticEdgeIdSchema,
  insight_node_id: semanticNodeIdSchema,
  target_node_id: semanticNodeIdSchema,
  source_episode_ids: z.array(episodeIdSchema).min(1),
  confidence: z.number().min(0).max(1),
});

const associatorPlanFindingSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("open_question"),
    question: z.string().min(1),
    urgency: z.number().min(0).max(1),
    episode_ids: z.array(episodeIdSchema).min(1),
    source_disclosure_label: memoryDisclosureLabelSchema,
    duplicate_of_open_question_id: openQuestionIdSchema.nullable().default(null),
  }),
  z.object({
    kind: z.literal("new_insight"),
    episode_ids: z.array(episodeIdSchema).min(1),
    source_disclosure_label: memoryDisclosureLabelSchema,
    target: associatorTargetSchema,
    candidate_support_edges: z.array(associatorSupportEdgeCandidateSchema).default([]),
    review: z.object({
      kind: z.literal("new_insight"),
      reason: z.string().min(1),
    }),
  }),
]);

const associatorPlanSampleSchema = z.object({
  sample_id: z.string().min(1),
  seed: z.string().min(1),
  episode_ids: z.array(episodeIdSchema).min(1),
  source_disclosure_label: memoryDisclosureLabelSchema,
  presented_open_question_ids: z.array(openQuestionIdSchema).default([]),
  open_question_candidate_set_complete: z.boolean().default(true),
  open_question_candidates_omitted: z.number().int().nonnegative().default(0),
  findings: z.array(associatorPlanFindingSchema).default([]),
});

export const associatorPlanSchema = z.object({
  process: z.literal("associator"),
  samples: z.array(associatorPlanSampleSchema),
  errors: z
    .array(
      z.object({
        process: z.literal("associator"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
  truncated_findings: z.number().int().nonnegative().default(0),
});

export type AssociatorPlan = z.infer<typeof associatorPlanSchema>;

const ABSOLUTE_CONFIDENCE_CEILING = 0.5;
const MAX_EPISODES_PER_SAMPLE = 8;
const MAX_SAMPLES_PER_RUN = 2;
const MAX_FINDINGS_PER_RUN = 4;
const OPEN_QUESTION_REINFORCEMENT_DELTA = 0.02;
const FALLBACK_BUCKET_COUNT = 4;

type AssociationSample = {
  sampleId: string;
  seed: string;
  episodes: Episode[];
};

type EpisodeSamplingRecord = {
  episode: Episode;
  stats: EpisodeStats | null;
  bucketKey: string;
  anchorScore: number;
  longTailScore: number;
};

const associatorInsightReversalSchema = z.object({
  nodeId: semanticNodeIdSchema,
  nodeCreated: z.boolean(),
  previousNode: serializableSemanticNodeSchema.optional(),
  edgeIds: z.array(semanticEdgeIdSchema),
  reviewItemId: z.number().int().positive().optional(),
});

const associatorOpenQuestionReversalSchema = z.discriminatedUnion("mode", [
  z.object({
    mode: z.literal("created"),
    question: openQuestionSchema,
  }),
  z.object({
    mode: z.literal("reinforced"),
    previous: openQuestionSchema,
  }),
]);

type AssociatorInsightReversal = z.infer<typeof associatorInsightReversalSchema>;
type AssociatorOpenQuestionReversal = z.infer<typeof associatorOpenQuestionReversalSchema>;

function serializeSemanticNode(node: SemanticNode) {
  return serializableSemanticNodeSchema.parse({
    ...node,
    embedding: Array.from(node.embedding),
  });
}

function deserializeSemanticNode(node: unknown): SemanticNode {
  const parsed = serializableSemanticNodeSchema.parse(node);

  return semanticNodeSchema.parse({
    ...parsed,
    embedding: Float32Array.from(parsed.embedding),
  });
}

function semanticNodeSnapshotMatches(
  node: SemanticNode,
  snapshot: z.infer<typeof serializableSemanticNodeSchema>,
): boolean {
  return JSON.stringify(serializeSemanticNode(node)) === JSON.stringify(snapshot);
}

function hashString(input: string): number {
  let hash = 2_166_136_261;

  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16_777_619);
  }

  return hash >>> 0;
}

function createPrng(seed: string): () => number {
  let state = hashString(seed) || 1;

  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0xffffffff;
  };
}

function clampPositiveInteger(value: number, max: number): number {
  return Math.max(1, Math.min(max, Math.floor(value)));
}

function shuffleDeterministic<T>(items: readonly T[], random: () => number): T[] {
  const shuffled = [...items];

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    const current = shuffled[index] as T;
    shuffled[index] = shuffled[swapIndex] as T;
    shuffled[swapIndex] = current;
  }

  return shuffled;
}

function episodeTimestamp(episode: Episode): number {
  return (episode.start_time + episode.end_time) / 2;
}

function tierAnchorWeight(stats: EpisodeStats | null): number {
  switch (stats?.tier) {
    case "T1":
      return 1;
    case "T2":
      return 0.7;
    case "T3":
      return 0.35;
    case "T4":
      return 0.1;
    default:
      return 0.5;
  }
}

function tierLongTailWeight(stats: EpisodeStats | null): number {
  switch (stats?.tier) {
    case "T4":
      return 1;
    case "T3":
      return 0.7;
    case "T2":
      return 0.35;
    case "T1":
      return 0.1;
    default:
      return 0.5;
  }
}

function scoreAnchor(episode: Episode, stats: EpisodeStats | null): number {
  return (
    episode.significance * 2 +
    episode.confidence * 0.2 +
    tierAnchorWeight(stats) +
    Math.log1p((stats?.retrieval_count ?? 0) + (stats?.use_count ?? 0)) * 0.1 +
    (stats?.heat_multiplier ?? 1) * 0.2
  );
}

function scoreLongTail(episode: Episode, stats: EpisodeStats | null, now: number): number {
  const heat = stats?.heat_multiplier ?? 1;
  const retrievals = stats?.retrieval_count ?? 0;
  const uses = stats?.use_count ?? 0;
  const lastRetrieved = stats?.last_retrieved;
  const ageScore =
    lastRetrieved === null || lastRetrieved === undefined
      ? 1
      : clamp((now - lastRetrieved) / (30 * 24 * 60 * 60 * 1_000), 0, 1);

  return (
    1 / (1 + retrievals + uses) +
    1 / (1 + heat) +
    tierLongTailWeight(stats) +
    ageScore +
    (1 - episode.significance) * 0.2
  );
}

function buildFallbackTimeBucketMap(episodes: readonly Episode[]): Map<Episode["id"], string> {
  const sorted = [...episodes].sort((left, right) => {
    const timeDelta = episodeTimestamp(left) - episodeTimestamp(right);
    return timeDelta === 0 ? left.id.localeCompare(right.id) : timeDelta;
  });
  const bucketSize = Math.max(1, Math.ceil(sorted.length / FALLBACK_BUCKET_COUNT));
  const bucketByEpisodeId = new Map<Episode["id"], string>();

  sorted.forEach((episode, index) => {
    bucketByEpisodeId.set(episode.id, `time:${Math.floor(index / bucketSize)}`);
  });

  return bucketByEpisodeId;
}

async function buildSamplingRecords(
  ctx: OfflineContext,
  episodes: readonly Episode[],
): Promise<EpisodeSamplingRecord[]> {
  const statsById = ctx.episodicRepository.getStatsMany(episodes.map((episode) => episode.id));
  const periods = ctx.autobiographicalRepository.listPeriods({ limit: 1_000 });
  const fallbackBucketByEpisodeId = buildFallbackTimeBucketMap(episodes);
  const now = ctx.clock.now();

  return episodes.map((episode) => {
    const timestamp = episodeTimestamp(episode);
    const period = periods.find(
      (candidate) =>
        candidate.start_ts <= timestamp &&
        (candidate.end_ts === null || candidate.end_ts >= timestamp),
    );
    const stats = statsById.get(episode.id) ?? null;

    return {
      episode,
      stats,
      bucketKey: period?.id ?? fallbackBucketByEpisodeId.get(episode.id) ?? "time:0",
      anchorScore: scoreAnchor(episode, stats),
      longTailScore: scoreLongTail(episode, stats, now),
    };
  });
}

function selectFromBucket(input: {
  bucket: readonly EpisodeSamplingRecord[];
  selectedIds: ReadonlySet<Episode["id"]>;
  usedIds: ReadonlySet<Episode["id"]>;
  prefer: "anchor" | "long_tail";
  random: () => number;
}): EpisodeSamplingRecord | null {
  const available = input.bucket.filter(
    (record) => !input.selectedIds.has(record.episode.id) && !input.usedIds.has(record.episode.id),
  );

  if (available.length === 0) {
    return null;
  }

  const ranked = [...available].sort((left, right) => {
    const scoreDelta =
      input.prefer === "anchor"
        ? right.anchorScore - left.anchorScore
        : right.longTailScore - left.longTailScore;

    return scoreDelta === 0 ? left.episode.id.localeCompare(right.episode.id) : scoreDelta;
  });
  const windowSize = Math.min(3, ranked.length);
  const index = Math.floor(input.random() * windowSize);

  return ranked[index] ?? ranked[0] ?? null;
}

async function collectAssociationSamples(ctx: OfflineContext): Promise<AssociationSample[]> {
  const episodes = await ctx.episodicRepository.listEffectivelyVisible();
  const episodesPerSample = clampPositiveInteger(
    ctx.config.offline.associator.episodesPerSample,
    MAX_EPISODES_PER_SAMPLE,
  );
  const maxSamplesPerRun = clampPositiveInteger(
    ctx.config.offline.associator.maxSamplesPerRun,
    MAX_SAMPLES_PER_RUN,
  );

  if (episodes.length === 0) {
    return [];
  }

  const records = await buildSamplingRecords(ctx, episodes);
  const byBucket = new Map<string, EpisodeSamplingRecord[]>();

  for (const record of records) {
    byBucket.set(record.bucketKey, [...(byBucket.get(record.bucketKey) ?? []), record]);
  }

  const seed = `${ctx.runId}:${ctx.clock.now()}:${episodes.length}:${episodesPerSample}:${maxSamplesPerRun}`;
  const random = createPrng(seed);
  const bucketKeys = shuffleDeterministic([...byBucket.keys()].sort(), random);
  const usedIds = new Set<Episode["id"]>();
  const samples: AssociationSample[] = [];

  for (let sampleIndex = 0; sampleIndex < maxSamplesPerRun; sampleIndex += 1) {
    const selected: EpisodeSamplingRecord[] = [];
    const selectedIds = new Set<Episode["id"]>();
    const sampleBucketKeys = shuffleDeterministic(
      bucketKeys,
      createPrng(`${seed}:sample:${sampleIndex}`),
    );
    let pass = 0;

    while (selected.length < episodesPerSample && pass < episodesPerSample * 2) {
      const prefer = pass % 2 === 0 ? "anchor" : "long_tail";
      let addedThisPass = false;

      for (const bucketKey of sampleBucketKeys) {
        if (selected.length >= episodesPerSample) {
          break;
        }

        const bucket = byBucket.get(bucketKey) ?? [];
        const picked = selectFromBucket({
          bucket,
          selectedIds,
          usedIds,
          prefer,
          random,
        });

        if (picked === null) {
          continue;
        }

        selected.push(picked);
        selectedIds.add(picked.episode.id);
        addedThisPass = true;
      }

      if (!addedThisPass) {
        break;
      }

      pass += 1;
    }

    if (selected.length === 0) {
      break;
    }

    for (const record of selected) {
      usedIds.add(record.episode.id);
    }

    samples.push({
      sampleId: `association:${sampleIndex}:${hashString(`${seed}:${sampleIndex}`).toString(16)}`,
      seed,
      episodes: selected
        .map((record) => record.episode)
        .sort((left, right) => {
          const timeDelta = episodeTimestamp(left) - episodeTimestamp(right);
          return timeDelta === 0 ? left.id.localeCompare(right.id) : timeDelta;
        }),
    });
  }

  return samples;
}

function renderSamplePrompt(
  sample: AssociationSample,
  statsById: ReadonlyMap<Episode["id"], EpisodeStats>,
  duplicatePresentation: OpenQuestionDuplicatePresentation,
): string {
  return [
    ASSOCIATOR_PROMPT,
    `Sample id: ${sample.sampleId}`,
    `Sampling seed: ${sample.seed}`,
    "Episodes:",
    ...sample.episodes.map((episode) =>
      JSON.stringify(
        episodeEvidencePromptRow(episode, {
          start_time: episode.start_time,
          end_time: episode.end_time,
          significance: episode.significance,
          confidence: episode.confidence,
          tags: episode.tags,
          participants: episode.participants,
          episode_kind: episode.episode_kind ?? "raw",
          stats: statsById.get(episode.id) ?? null,
        }),
      ),
    ),
    "Open-question duplicate candidate set:",
    JSON.stringify({
      complete: duplicatePresentation.complete,
      total_open_questions: duplicatePresentation.total_open_questions,
      presented_count: duplicatePresentation.presented_count,
      omitted_count: duplicatePresentation.omitted_count,
    }),
    ...duplicatePresentation.rows.map((row) => JSON.stringify(row)),
  ].join("\n");
}

function invalidAssociationResponse(error: unknown): unknown {
  if (isStructuredToolCallError(error, "missing_tool_call")) {
    return new SemanticError(`Associator did not emit tool ${ASSOCIATOR_TOOL_NAME}`, {
      code: "ASSOCIATOR_INVALID",
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

function parseAssociationResponse(input: unknown) {
  return associatorResponseSchema.parse(input);
}

function uniqueValidEpisodeIds(input: readonly string[], allowedIds: ReadonlySet<Episode["id"]>) {
  const validIds: Episode["id"][] = [];
  const invalidIds: string[] = [];
  const seen = new Set<string>();

  for (const rawId of input) {
    if (!allowedIds.has(rawId as Episode["id"])) {
      invalidIds.push(rawId);
      continue;
    }

    if (!seen.has(rawId)) {
      seen.add(rawId);
      validIds.push(rawId as Episode["id"]);
    }
  }

  if (validIds.length === 0 || invalidIds.length > 0) {
    throw new SemanticError("Associator referenced episodes outside the sample", {
      code: "ASSOCIATOR_INVALID_REF",
      cause: {
        invalid_episode_ids: invalidIds,
      },
    });
  }

  return validIds;
}

async function buildSupportEdgeCandidates(
  ctx: OfflineContext,
  insightNodeId: SemanticNode["id"],
  sourceEpisodeIds: readonly Episode["id"][],
  confidence: number,
): Promise<Array<z.infer<typeof associatorSupportEdgeCandidateSchema>>> {
  const sourceEpisodeIdSet = new Set(sourceEpisodeIds);
  const candidateNodes = await ctx.semanticNodeRepository.list({
    includeArchived: false,
    limit: 1_000,
  });
  const evidenceByTargetNodeId = new Map<SemanticNode["id"], Episode["id"][]>();

  for (const node of candidateNodes) {
    if (node.id === insightNodeId) {
      continue;
    }

    const evidenceEpisodeIds = node.source_episode_ids.filter((episodeId) =>
      sourceEpisodeIdSet.has(episodeId),
    );

    if (evidenceEpisodeIds.length === 0) {
      continue;
    }

    evidenceByTargetNodeId.set(node.id, [
      ...(evidenceByTargetNodeId.get(node.id) ?? []),
      ...evidenceEpisodeIds,
    ]);
  }

  return [...evidenceByTargetNodeId.entries()].map(([targetNodeId, evidenceEpisodeIds]) =>
    associatorSupportEdgeCandidateSchema.parse({
      id: createSemanticEdgeId(),
      insight_node_id: insightNodeId,
      target_node_id: targetNodeId,
      source_episode_ids: [...new Set(evidenceEpisodeIds)],
      confidence,
    }),
  );
}

async function buildInsightFinding(input: {
  ctx: OfflineContext;
  finding: z.infer<typeof associatorNewInsightFindingSchema>;
  episodeIds: readonly Episode["id"][];
}) {
  const embedding = await input.ctx.embeddingClient.embed(
    `${input.finding.label}\n${input.finding.description}`,
  );
  const timestamp = input.ctx.clock.now();
  const ceiling = Math.min(
    ABSOLUTE_CONFIDENCE_CEILING,
    input.ctx.config.offline.associator.ceilingConfidence,
  );
  const confidence = Math.min(input.finding.confidence, ceiling);
  const target = {
    mode: "insert" as const,
    node: serializableSemanticNodeSchema.parse({
      id: createSemanticNodeId(),
      kind: "proposition",
      label: input.finding.label.trim(),
      description: input.finding.description.trim(),
      aliases: [],
      confidence,
      source_episode_ids: input.episodeIds,
      created_at: timestamp,
      updated_at: timestamp,
      last_verified_at: timestamp,
      embedding: Array.from(embedding),
      archived: false,
      superseded_by: null,
    }),
  };
  const sourceDisclosureLabel = await disclosureLabelForEpisodeIds(
    input.ctx.episodicRepository,
    input.episodeIds,
  );
  const candidateSupportEdges = await buildSupportEdgeCandidates(
    input.ctx,
    target.node.id,
    input.episodeIds,
    confidence,
  );

  return associatorPlanFindingSchema.parse({
    kind: "new_insight",
    episode_ids: input.episodeIds,
    source_disclosure_label: sourceDisclosureLabel,
    target,
    candidate_support_edges: candidateSupportEdges,
    review: {
      kind: "new_insight",
      reason: "New conservative associative insight extracted from distant episodic evidence",
    },
  });
}

async function buildSampleFindings(input: {
  ctx: OfflineContext;
  llmClient: LLMClient;
  sample: AssociationSample;
  remainingFindings: number;
}): Promise<{
  findings: AssociatorPlan["samples"][number]["findings"];
  truncatedFindings: number;
  duplicatePresentation: OpenQuestionDuplicatePresentation;
}> {
  const allowedIds = new Set(input.sample.episodes.map((episode) => episode.id));
  const statsById = input.ctx.episodicRepository.getStatsMany(
    input.sample.episodes.map((episode) => episode.id),
  );
  const duplicatePresentation = await buildOpenQuestionDuplicatePresentation({
    repository: input.ctx.openQuestionsRepository,
    sourceTextProxy: input.sample.episodes
      .map((episode) => `${episode.title}\n${episode.narrative}`)
      .join("\n"),
    onSearchFailure: (error) => {
      console.warn("Associator open-question candidate search failed open", {
        run_id: input.ctx.runId,
        error,
      });
    },
  });
  const presentedOpenQuestionIds = new Set(duplicatePresentation.rows.map((row) => row.id));
  let response: z.infer<typeof associatorResponseSchema>;

  try {
    response = (
      await callStructuredTool({
        llmClient: input.llmClient,
        request: {
          model: input.ctx.config.anthropic.models.background,
          system: "I make cross-life associative findings from deliberately distant memories.",
          messages: [
            {
              role: "user",
              content: renderSamplePrompt(input.sample, statsById, duplicatePresentation),
            },
          ],
          tools: [ASSOCIATOR_TOOL],
          tool_choice: { type: "tool", name: ASSOCIATOR_TOOL_NAME },
          max_tokens: 4_000,
          budget: "offline-associator",
        },
        toolName: ASSOCIATOR_TOOL_NAME,
        parse: parseAssociationResponse,
      })
    ).parsed;
  } catch (error) {
    throw invalidAssociationResponse(error);
  }

  const findings: AssociatorPlan["samples"][number]["findings"] = [];
  const cappedFindings = response.findings.slice(0, input.remainingFindings);
  const truncatedFindings = Math.max(0, response.findings.length - cappedFindings.length);

  for (const finding of cappedFindings) {
    const episodeIds = uniqueValidEpisodeIds(finding.source_episode_ids, allowedIds);

    if (finding.kind === "open_question") {
      findings.push(
        associatorPlanFindingSchema.parse({
          kind: "open_question",
          question: finding.question.trim(),
          urgency: finding.urgency,
          episode_ids: episodeIds,
          source_disclosure_label: await disclosureLabelForEpisodeIds(
            input.ctx.episodicRepository,
            episodeIds,
          ),
          duplicate_of_open_question_id:
            finding.duplicate_of_open_question_id !== null &&
            presentedOpenQuestionIds.has(finding.duplicate_of_open_question_id)
              ? finding.duplicate_of_open_question_id
              : null,
        }),
      );
      continue;
    }

    findings.push(
      await buildInsightFinding({
        ctx: input.ctx,
        finding,
        episodeIds,
      }),
    );
  }

  return {
    findings,
    truncatedFindings,
    duplicatePresentation,
  };
}

function candidateStats(input: {
  findings: number;
  errors: number;
  truncatedFindings: number;
  rejectedPostPlan?: number;
  accepted: number;
}) {
  const rejected = input.errors + input.truncatedFindings + (input.rejectedPostPlan ?? 0);

  return {
    proposed: input.findings + input.errors + input.truncatedFindings,
    accepted: input.accepted,
    rejected,
    ...(input.truncatedFindings === 0 ? {} : { truncated: input.truncatedFindings }),
  };
}

function emitEpisodeArchivedPostPlan(ctx: OfflineContext): void {
  if (ctx.tracer?.enabled !== true) {
    return;
  }

  ctx.tracer.emit("semantic_insert.skipped", {
    turnId: ctx.runId,
    kind: "episode",
    reason: "episode_archived_post_plan",
  });
}

async function refsArchivedPostPlan(
  ctx: OfflineContext,
  episodeIds: readonly Episode["id"][],
): Promise<boolean> {
  const uniqueIds = [...new Set(episodeIds)];
  const episodes = await Promise.all(uniqueIds.map((episodeId) => ctx.episodicRepository.get(episodeId)));

  return episodes.some((episode) => episode === null);
}

function sampleChangeTargets(
  sample: AssociatorPlan["samples"][number],
  episodeIds: readonly Episode["id"][],
) {
  return {
    sample_id: sample.sample_id,
    sample_episode_ids: sample.episode_ids,
    episode_ids: episodeIds,
  };
}

function buildChange(
  sample: AssociatorPlan["samples"][number],
  finding: AssociatorPlan["samples"][number]["findings"][number],
): OfflineChange {
  if (finding.kind === "open_question") {
    return {
      process: "associator",
      action: "open_question",
      targets: sampleChangeTargets(sample, finding.episode_ids),
      preview: {
        question: finding.question,
        urgency: finding.urgency,
      },
    };
  }

  const nodeLabel =
    finding.target.mode === "insert"
      ? finding.target.node.label
      : `${finding.target.node_id} (update)`;

  return {
    process: "associator",
    action: "insight",
    targets: sampleChangeTargets(sample, finding.episode_ids),
    preview: {
      label: nodeLabel,
      confidence:
        finding.target.mode === "insert"
          ? finding.target.node.confidence
          : finding.target.patch.confidence,
    },
  };
}

function allFindings(plan: AssociatorPlan) {
  return plan.samples.flatMap((sample) =>
    sample.findings.map((finding) => ({
      sample,
      finding,
    })),
  );
}

export type AssociatorProcessOptions = {
  semanticNodeRepository: OfflineContext["semanticNodeRepository"];
  semanticEdgeRepository: OfflineContext["semanticEdgeRepository"];
  reviewQueueRepository: OfflineContext["reviewQueueRepository"];
  openQuestionsRepository: OfflineContext["openQuestionsRepository"];
  registry: ReverserRegistry;
  clock?: Clock;
};

export class AssociatorProcess implements OfflineProcess<AssociatorPlan> {
  readonly name = "associator" as const;
  private readonly clock: Clock;

  constructor(private readonly options: AssociatorProcessOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.options.registry.register(this.name, "insight", async ({ reversal }) => {
      const parsed = associatorInsightReversalSchema.parse(reversal);

      for (const edgeId of parsed.edgeIds) {
        this.options.semanticEdgeRepository.invalidateEdge(edgeId, {
          at: this.clock.now(),
          by_process: "maintenance",
          reason: "associator_audit_reversal",
        });
      }

      if (parsed.nodeCreated) {
        await this.options.semanticNodeRepository.update(parsed.nodeId, {
          archived: true,
        });
      } else if (parsed.previousNode !== undefined) {
        const previousNode = deserializeSemanticNode(parsed.previousNode);
        const current = await this.options.semanticNodeRepository.get(previousNode.id);

        if (current === null || !semanticNodeSnapshotMatches(current, parsed.previousNode)) {
          await this.options.semanticNodeRepository.restore(previousNode);
        }
      }

      if (typeof parsed.reviewItemId === "number") {
        this.options.reviewQueueRepository.delete(parsed.reviewItemId);
      }
    });
    this.options.registry.register(this.name, "open_question", async ({ reversal }) => {
      const parsed = associatorOpenQuestionReversalSchema.parse(reversal);

      if (parsed.mode === "created") {
        await this.options.openQuestionsRepository.delete(parsed.question.id);
        return;
      }

      this.options.openQuestionsRepository.restore(parsed.previous);
    });
  }

  async plan(ctx: OfflineContext, opts: { budget?: number } = {}): Promise<AssociatorPlan> {
    const errors: OfflineProcessError[] = [];
    const budget = opts.budget ?? ctx.config.offline.associator.budget;
    const samples = await collectAssociationSamples(ctx);
    const planSamples: AssociatorPlan["samples"] = [];
    let plannedFindings = 0;
    let tokensUsed = 0;
    let budgetExhausted = false;
    let truncatedFindings = 0;
    const maxFindingsPerRun = clampPositiveInteger(
      ctx.config.offline.associator.maxFindingsPerRun,
      MAX_FINDINGS_PER_RUN,
    );

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient = wrapClient(ctx.llm.background);

        for (const sample of samples) {
          const remainingFindings = maxFindingsPerRun - plannedFindings;

          if (remainingFindings <= 0) {
            break;
          }

          try {
            const sampleSourceDisclosureLabel = await disclosureLabelForEpisodeIds(
              ctx.episodicRepository,
              sample.episodes.map((episode) => episode.id),
            );
            const result = await buildSampleFindings({
              ctx,
              llmClient,
              sample,
              remainingFindings,
            });
            const findings = result.findings;

            plannedFindings += findings.length;
            truncatedFindings += result.truncatedFindings;
            planSamples.push({
              sample_id: sample.sampleId,
              seed: sample.seed,
              episode_ids: sample.episodes.map((episode) => episode.id),
              source_disclosure_label: sampleSourceDisclosureLabel,
              presented_open_question_ids: result.duplicatePresentation.rows.map((row) => row.id),
              open_question_candidate_set_complete: result.duplicatePresentation.complete,
              open_question_candidates_omitted: result.duplicatePresentation.omitted_count,
              findings,
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

    return associatorPlanSchema.parse({
      process: this.name,
      samples: planSamples,
      errors,
      tokens_used: tokensUsed,
      budget_exhausted: budgetExhausted,
      truncated_findings: truncatedFindings,
    });
  }

  preview(plan: AssociatorPlan): OfflineResult {
    const parsed = associatorPlanSchema.parse(plan);
    const findings = allFindings(parsed);

    return {
      process: this.name,
      dryRun: true,
      changes: findings.map(({ sample, finding }) => buildChange(sample, finding)),
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
      run_capped: parsed.truncated_findings > 0,
      candidate_stats: candidateStats({
        findings: findings.length,
        errors: parsed.errors.length,
        truncatedFindings: parsed.truncated_findings,
        accepted: 0,
      }),
    };
  }

  async apply(ctx: OfflineContext, rawPlan: AssociatorPlan): Promise<OfflineResult> {
    const plan = associatorPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];
    const processProvenance = {
      kind: "offline" as const,
      process: this.name,
    };
    let rejectedPostPlan = 0;

    for (const sample of plan.samples) {
      for (const finding of sample.findings) {
        if (await refsArchivedPostPlan(ctx, [...sample.episode_ids, ...finding.episode_ids])) {
          rejectedPostPlan += 1;
          emitEpisodeArchivedPostPlan(ctx);
          continue;
        }

        if (finding.kind === "open_question") {
          const existingByDedupeKey = ctx.openQuestionsRepository.getByDedupeKey(
            buildOpenQuestionDedupeKey({
              question: finding.question,
              relatedEpisodeIds: finding.episode_ids,
              relatedSemanticNodeIds: [],
              audienceEntityId: null,
            }),
          );
          const advisoryDuplicate =
            finding.duplicate_of_open_question_id !== null &&
            sample.presented_open_question_ids.includes(finding.duplicate_of_open_question_id)
              ? ctx.openQuestionsRepository.get(finding.duplicate_of_open_question_id)
              : null;
          const backstop =
            advisoryDuplicate?.status === "open" || existingByDedupeKey?.status === "open"
              ? null
              : await findOpenQuestionDuplicateBackstop({
                  repository: ctx.openQuestionsRepository,
                  question: finding.question,
                  onSearchFailure: (error) => {
                    console.warn("Associator open-question duplicate backstop failed open", {
                      run_id: ctx.runId,
                      error,
                    });
                  },
                });
          const similar =
            advisoryDuplicate?.status === "open"
              ? { question: advisoryDuplicate, similarity: null }
              : existingByDedupeKey?.status === "open"
                ? { question: existingByDedupeKey, similarity: 1 }
                : backstop;

          if (similar !== null) {
            const previous = similar.question;
            const result = ctx.identityService.updateOpenQuestion(
              previous.id,
              buildOpenQuestionReinforcementPatch({
                existing: previous,
                incomingRelatedEpisodeIds: finding.episode_ids,
                incomingRelatedSemanticNodeIds: [],
                incomingDisclosureLabel: finding.source_disclosure_label,
                urgencyDelta: OPEN_QUESTION_REINFORCEMENT_DELTA,
              }),
              processProvenance,
              {
                throughReview: true,
                reason: "associator_open_question_reinforcement",
                preserveRecordProvenance: true,
              },
            );

            if (result.status === "applied") {
              ctx.auditLog.record({
                run_id: ctx.runId,
                process: this.name,
                action: "open_question",
                targets: {
                  question_id: previous.id,
                  reinforced: true,
                },
                reversal: {
                  mode: "reinforced",
                  previous,
                } satisfies AssociatorOpenQuestionReversal,
              });
              changes.push(buildChange(sample, finding));
            }
            continue;
          }

          if (existingByDedupeKey !== null) {
            continue;
          }

          const created = ctx.identityService.addOpenQuestion({
            question: finding.question,
            urgency: finding.urgency,
            related_episode_ids: finding.episode_ids,
            related_semantic_node_ids: [],
            disclosure_label: finding.source_disclosure_label,
            provenance: processProvenance,
            source: this.name,
          });

          ctx.auditLog.record({
            run_id: ctx.runId,
            process: this.name,
            action: "open_question",
            targets: {
              question_id: created.id,
              reinforced: false,
            },
            reversal: {
              mode: "created",
              question: created,
            } satisfies AssociatorOpenQuestionReversal,
          });
          changes.push(buildChange(sample, finding));
          continue;
        }

        let nodeId: SemanticNode["id"];
        let nodeCreated = false;
        let previousNode: z.infer<typeof serializableSemanticNodeSchema> | undefined;

        if (finding.target.mode === "insert") {
          nodeId = finding.target.node.id;
          nodeCreated = true;
        } else {
          const current = await ctx.semanticNodeRepository.get(finding.target.node_id);

          if (current === null) {
            throw new SemanticError(
              `Missing semantic node for associator plan: ${finding.target.node_id}`,
              {
                code: "ASSOCIATOR_PLAN_INVALID",
              },
            );
          }

          previousNode = serializeSemanticNode(current);
          nodeId = current.id;
        }

        const reviewItem = ctx.reviewQueueRepository.enqueue({
          kind: finding.review.kind,
          refs: {
            node_ids: [nodeId],
            episode_ids: finding.episode_ids,
            evidence_cluster_key: sample.sample_id,
            evidence_cluster_size: finding.episode_ids.length,
            source_disclosure_label: finding.source_disclosure_label,
            reflector_pending_insight: {
              target: finding.target,
              candidate_support_edges: finding.candidate_support_edges,
              evidence_cluster: {
                key: sample.sample_id,
                episode_ids: finding.episode_ids,
                size: finding.episode_ids.length,
              },
            },
          },
          reason: finding.review.reason,
          sourceProcess: this.name,
          traceTurnId: ctx.runId,
        });

        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "insight",
          targets: {
            nodeId,
            reviewItemId: reviewItem.id,
          },
          reversal: {
            nodeId,
            nodeCreated,
            ...(previousNode === undefined ? {} : { previousNode }),
            edgeIds: finding.candidate_support_edges.map((edge) => edge.id),
            reviewItemId: reviewItem.id,
          } satisfies AssociatorInsightReversal,
        });
        changes.push(buildChange(sample, finding));
      }
    }

    return {
      process: this.name,
      dryRun: false,
      changes,
      tokens_used: plan.tokens_used,
      errors: plan.errors,
      budget_exhausted: plan.budget_exhausted,
      run_capped: plan.truncated_findings > 0,
      candidate_stats: candidateStats({
        findings: allFindings(plan).length,
        errors: plan.errors.length,
        truncatedFindings: plan.truncated_findings,
        rejectedPostPlan,
        accepted: changes.length,
      }),
    };
  }

  async run(
    ctx: OfflineContext,
    opts: { dryRun?: boolean; budget?: number },
  ): Promise<OfflineResult> {
    const plan = await this.plan(ctx, opts);
    return opts.dryRun === true ? this.preview(plan) : this.apply(ctx, plan);
  }
}
