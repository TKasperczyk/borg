import { z } from "zod";

import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { episodeIdSchema, type Episode } from "../../memory/episodic/index.js";
import {
  memoryDisclosureLabelFromEpisodeAccess,
  memoryDisclosureLabelSchema,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import {
  proceduralEvidenceIdSchema,
  proceduralEvidenceSchema,
  skillContextStatsSchema,
  skillIdSchema,
  skillSchema,
  type ProceduralContextMetadata,
  type ProceduralEvidenceRecord,
  type SkillContextStatsRecord,
  type SkillRecord,
} from "../../memory/procedural/index.js";
import type { ReviewQueueItem, SkillSplitReviewPayload } from "../../memory/review-queue/index.js";
import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";
import {
  combineMemoryDisclosureLabels,
  mapWithDisclosureConcurrency,
  memoryDisclosureLabelForEpisodeIds,
} from "../../retrieval/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { BudgetExceededError, StorageError } from "../../util/errors.js";
import { type EpisodeId } from "../../util/ids.js";

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
import { detectDivergentSkillSplits, type SkillSplitCandidate } from "./split-detector.js";

const CLUSTER_SIMILARITY_THRESHOLD = 0.85;
const SYNTHESIZER_TOOL_NAME = "EmitProceduralSkillCandidate";
const SKILL_SPLIT_TOOL_NAME = "EmitSkillSplit";

const proceduralSynthesizerRejectionReasonSchema = z.enum([
  "unusable_abstraction",
  "centered_proper_noun",
]);

const skillCandidateSchema = z.object({
  applies_when: z.string().min(1),
  approach: z.string().min(1),
  abstraction_fit: z.enum(["too_narrow", "usable", "too_broad"]),
  rejection_reason: proceduralSynthesizerRejectionReasonSchema.nullable().default(null),
});

const skillSplitPartSchema = z.object({
  applies_when: z.string().min(1),
  approach: z.string().min(1),
  target_contexts: z.array(z.string().min(1)).min(1),
});

const skillSplitProposalSchema = z.object({
  decision: z.enum(["split", "no_split", "refine_in_place"]),
  parts: z.array(skillSplitPartSchema).optional(),
  rationale: z.string().min(1),
});

export const PROCEDURAL_SYNTHESIZER_TOOL = {
  name: SYNTHESIZER_TOOL_NAME,
  description: "Emit a reusable procedural skill candidate from repeated successful attempts.",
  inputSchema: toToolInputSchema(skillCandidateSchema),
} satisfies LLMToolDefinition;

export const PROCEDURAL_SKILL_SPLIT_TOOL = {
  name: SKILL_SPLIT_TOOL_NAME,
  description: "Emit a conservative split proposal for a skill with divergent context outcomes.",
  inputSchema: toToolInputSchema(skillSplitProposalSchema),
} satisfies LLMToolDefinition;

const proceduralSynthesizerDedupDecisionSchema = z.object({
  skill_id: skillIdSchema.nullable(),
  similarity: z.number().min(0).max(1).nullable(),
});

const proceduralSynthesizerPlanItemSchema = z.object({
  cluster_key: z.string().min(1),
  evidence: z.array(proceduralEvidenceSchema).min(2),
  source_episode_ids: z.array(episodeIdSchema),
  source_disclosure_label: memoryDisclosureLabelSchema,
  candidate: skillCandidateSchema,
  dedup_decision: proceduralSynthesizerDedupDecisionSchema,
  rejection_reason: proceduralSynthesizerRejectionReasonSchema.nullable(),
});

const proceduralSkillSplitBucketSchema = z.object({
  stats: skillContextStatsSchema,
  posterior_mean: z.number().min(0).max(1),
});

const proceduralSynthesizerSplitItemSchema = z.object({
  skill: skillSchema,
  buckets: z.array(proceduralSkillSplitBucketSchema).min(2),
  min_posterior_mean: z.number().min(0).max(1),
  max_posterior_mean: z.number().min(0).max(1),
  divergence: z.number().min(0).max(1),
  split_claimed_at: z.number().finite().nullable().optional(),
  proposal: skillSplitProposalSchema,
});

export const proceduralSynthesizerPlanSchema = z.object({
  process: z.literal("procedural-synthesizer"),
  items: z.array(proceduralSynthesizerPlanItemSchema),
  split_items: z.array(proceduralSynthesizerSplitItemSchema).default([]),
  budget: z.number().int().positive(),
  errors: z
    .array(
      z.object({
        process: z.literal("procedural-synthesizer"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
});

export type ProceduralSynthesizerPlan = z.infer<typeof proceduralSynthesizerPlanSchema>;

const proceduralSynthesizerReversalSchema = z.object({
  skillId: skillIdSchema,
  inserted: z.boolean(),
  previousSkill: skillSchema.optional(),
  newSkill: skillSchema.optional(),
  evidenceIds: z.array(proceduralEvidenceIdSchema),
});

const proceduralSkillSplitReversalSchema = z.object({
  originalSkill: skillSchema,
  createdSkills: z.array(skillSchema),
  movedContextStats: z.array(skillContextStatsSchema),
});

type EvidenceCluster = {
  key: string;
  evidence: ProceduralEvidenceRecord[];
  evidenceRows: LabeledProceduralEvidenceRow[];
  sourceEpisodeIds: EpisodeId[];
  sourceEpisodes: Episode[];
  sourceDisclosureLabel: MemoryDisclosureLabel;
};

type LabeledProceduralEvidenceRow = {
  evidence: ProceduralEvidenceRecord;
  sourceEpisodeIds: EpisodeId[];
  disclosureLabel: MemoryDisclosureLabel;
};

type LabeledSkillSplitCandidate = SkillSplitCandidate & {
  sourceEpisodes: Episode[];
  sourceDisclosureLabel: MemoryDisclosureLabel;
};

function evidenceEmbeddingText(evidence: ProceduralEvidenceRecord): string {
  return [
    evidence.pending_attempt_snapshot.problem_text,
    evidence.pending_attempt_snapshot.approach_summary,
  ].join("\n");
}

function uniqueEpisodeIds(ids: readonly EpisodeId[]): EpisodeId[] {
  return [...new Set(ids)];
}

async function resolveEvidenceEpisodeIds(
  ctx: OfflineContext,
  evidence: ProceduralEvidenceRecord,
): Promise<EpisodeId[]> {
  const resolved = [...evidence.resolved_episode_ids];
  const exact = await ctx.episodicRepository.findBySourceStreamIds(
    evidence.pending_attempt_snapshot.source_stream_ids,
  );

  if (exact !== null && ctx.episodicRepository.isEpisodeEffectivelyVisible(exact.id)) {
    resolved.push(exact.id);
  } else {
    const sourceIds = evidence.pending_attempt_snapshot.source_stream_ids;
    const episodes = await ctx.episodicRepository.listEffectivelyVisible();
    const matchingEpisode = episodes.find((episode) =>
      sourceIds.every((streamId) => episode.source_stream_ids.includes(streamId)),
    );

    if (matchingEpisode !== undefined) {
      resolved.push(matchingEpisode.id);
    }
  }

  return uniqueEpisodeIds(resolved);
}

async function collectEvidenceClusters(
  ctx: OfflineContext,
  evidenceRows: readonly ProceduralEvidenceRecord[],
): Promise<EvidenceCluster[]> {
  const embeddings = await ctx.embeddingClient.embedBatch(
    evidenceRows.map((evidence) => evidenceEmbeddingText(evidence)),
  );
  const remaining = new Set(evidenceRows.map((_, index) => index));
  const clusters: EvidenceCluster[] = [];

  for (let seedIndex = 0; seedIndex < evidenceRows.length; seedIndex += 1) {
    if (!remaining.has(seedIndex)) {
      continue;
    }

    const seed = evidenceRows[seedIndex];
    const seedEmbedding = embeddings[seedIndex];

    if (seed === undefined || seedEmbedding === undefined) {
      continue;
    }

    remaining.delete(seedIndex);
    const clusterEvidence = [seed];

    for (const candidateIndex of [...remaining]) {
      const candidateEmbedding = embeddings[candidateIndex];

      if (
        candidateEmbedding !== undefined &&
        cosineSimilarity(seedEmbedding, candidateEmbedding) >= CLUSTER_SIMILARITY_THRESHOLD
      ) {
        const candidate = evidenceRows[candidateIndex];

        if (candidate !== undefined) {
          clusterEvidence.push(candidate);
          remaining.delete(candidateIndex);
        }
      }
    }

    const resolvedEvidenceRows = await mapWithDisclosureConcurrency(
      clusterEvidence,
      async (evidence) => ({
        evidence,
        sourceEpisodeIds: await resolveEvidenceEpisodeIds(ctx, evidence),
      }),
    );
    const sourceEpisodeIds = uniqueEpisodeIds(
      resolvedEvidenceRows.flatMap((evidenceRow) => evidenceRow.sourceEpisodeIds),
    );
    const sourceEpisodes = await ctx.episodicRepository.getMany(sourceEpisodeIds);
    const labelsByEpisodeId = new Map(
      sourceEpisodes.map((episode) => [
        episode.id,
        memoryDisclosureLabelFromEpisodeAccess(episode),
      ]),
    );
    const labeledEvidenceRows: LabeledProceduralEvidenceRow[] = resolvedEvidenceRows.map(
      ({ evidence, sourceEpisodeIds: evidenceSourceEpisodeIds }) => ({
        evidence,
        sourceEpisodeIds: evidenceSourceEpisodeIds,
        disclosureLabel: memoryDisclosureLabelForEpisodeIds(
          evidenceSourceEpisodeIds,
          labelsByEpisodeId,
        ),
      }),
    );

    clusters.push({
      key: `procedural:${seed.id}`,
      evidence: clusterEvidence,
      evidenceRows: labeledEvidenceRows,
      sourceEpisodeIds,
      sourceEpisodes,
      sourceDisclosureLabel: combineMemoryDisclosureLabels(
        labeledEvidenceRows.map((evidenceRow) => evidenceRow.disclosureLabel),
      ),
    });
  }

  return clusters.sort(
    (left, right) =>
      right.evidence.length - left.evidence.length ||
      left.evidence[0]!.created_at - right.evidence[0]!.created_at,
  );
}

function buildPrompt(cluster: EvidenceCluster): string {
  return [
    "I synthesize one reusable procedural skill from repeated successful problem-solving attempts.",
    `I emit my result by calling the ${SYNTHESIZER_TOOL_NAME} tool exactly once.`,
    "The skill I emit describes a reusable problem class and concrete checks or moves.",
    "I mark abstraction_fit as too_narrow when the skill is tied to a specific named project, person, or incident.",
    "I mark abstraction_fit as too_broad when it is generic advice rather than a reusable procedure.",
    "I set rejection_reason to centered_proper_noun when an otherwise usable candidate remains centered on a project/person/product name instead of a reusable class; I set unusable_abstraction when abstraction_fit is not usable; otherwise null.",
    `Cluster: ${cluster.key}`,
    "Evidence:",
    ...cluster.evidenceRows.map((row) =>
      JSON.stringify({
        id: row.evidence.id,
        problem_text: row.evidence.pending_attempt_snapshot.problem_text,
        approach_summary: row.evidence.pending_attempt_snapshot.approach_summary,
        classification: row.evidence.classification,
        outcome_evidence: row.evidence.evidence_text,
        source_episode_ids: row.sourceEpisodeIds,
        ...memoryDisclosurePayloadFields(row.disclosureLabel),
      }),
    ),
    "Source episodes:",
    ...cluster.sourceEpisodes.map((episode) => JSON.stringify(episodeEvidencePromptRow(episode))),
  ].join("\n");
}

function parseSkillCandidate(input: unknown) {
  return skillCandidateSchema.parse(input);
}

function parseSkillSplitProposal(input: unknown) {
  return skillSplitProposalSchema.parse(input);
}

function proceduralSynthesizerStructuredError(
  error: unknown,
  toolName: typeof SYNTHESIZER_TOOL_NAME | typeof SKILL_SPLIT_TOOL_NAME,
): unknown {
  if (isStructuredToolCallError(error, "missing_tool_call")) {
    return new StorageError(`Procedural synthesizer did not emit tool ${toolName}`, {
      code:
        toolName === SYNTHESIZER_TOOL_NAME
          ? "PROCEDURAL_SYNTHESIZER_INVALID"
          : "PROCEDURAL_SKILL_SPLIT_INVALID",
    });
  }

  return isStructuredToolCallError(error) ? (error.cause ?? error) : error;
}

function isSkillSplitParseFailure(error: unknown): boolean {
  if (error instanceof z.ZodError) {
    return true;
  }

  return error instanceof StorageError && error.code === "PROCEDURAL_SKILL_SPLIT_INVALID";
}

function resolveContextMetadata(stats: SkillContextStatsRecord): ProceduralContextMetadata | null {
  if (stats.procedural_context !== null && stats.procedural_context !== undefined) {
    return stats.procedural_context;
  }

  return null;
}

function sketchContextStats(stats: SkillContextStatsRecord): string {
  const parsed = resolveContextMetadata(stats);

  if (parsed === null) {
    return "unknown; no domain tags; audience=unknown";
  }

  const tagText =
    parsed.domain_tags.length === 0 ? "no domain tags" : parsed.domain_tags.join(", ");

  return `${parsed.problem_kind}; ${tagText}; audience=${parsed.audience_scope}`;
}

function buildSplitPrompt(candidate: LabeledSkillSplitCandidate): string {
  return [
    "One of my procedural skills has divergent outcomes across context buckets.",
    `I emit my result by calling the ${SKILL_SPLIT_TOOL_NAME} tool exactly once.`,
    "I prefer no_split when the evidence does not clearly imply distinct reusable procedures.",
    "I use refine_in_place only when one narrower skill text would cover the divergence without creating child skills.",
    "I use split only when each part is narrower than the original and can own one or more listed context_keys.",
    "I set target_contexts to exact context_key strings from the listed buckets.",
    "Original skill:",
    JSON.stringify({
      id: candidate.skill.id,
      applies_when: candidate.skill.applies_when,
      approach: candidate.skill.approach,
      source_episode_ids: candidate.skill.source_episode_ids,
      alpha: candidate.skill.alpha,
      beta: candidate.skill.beta,
      attempts: candidate.skill.attempts,
      successes: candidate.skill.successes,
      failures: candidate.skill.failures,
      ...memoryDisclosurePayloadFields(candidate.sourceDisclosureLabel),
    }),
    "Source episodes:",
    ...candidate.sourceEpisodes.map((episode) => JSON.stringify(episodeEvidencePromptRow(episode))),
    "Divergent context buckets:",
    ...candidate.buckets.map((bucket) =>
      JSON.stringify({
        context_key: bucket.stats.context_key,
        sketch: sketchContextStats(bucket.stats),
        posterior_mean: Number(bucket.posterior_mean.toFixed(3)),
        alpha: bucket.stats.alpha,
        beta: bucket.stats.beta,
        attempts: bucket.stats.attempts,
        successes: bucket.stats.successes,
        failures: bucket.stats.failures,
        last_used: bucket.stats.last_used,
        last_successful: bucket.stats.last_successful,
      }),
    ),
    JSON.stringify({
      divergence: Number(candidate.divergence.toFixed(3)),
      min_posterior_mean: Number(candidate.min_posterior_mean.toFixed(3)),
      max_posterior_mean: Number(candidate.max_posterior_mean.toFixed(3)),
    }),
  ].join("\n");
}

async function labeledSplitCandidate(
  ctx: OfflineContext,
  candidate: SkillSplitCandidate,
): Promise<LabeledSkillSplitCandidate> {
  const sourceEpisodeIds = uniqueEpisodeIds(candidate.skill.source_episode_ids);
  const episodes = (await ctx.episodicRepository.getMany(sourceEpisodeIds)).filter(
    (episode): episode is Episode => episode !== null,
  );
  const sourceDisclosureLabel = await disclosureLabelForEpisodeIds(
    ctx.episodicRepository,
    sourceEpisodeIds,
  );

  return {
    ...candidate,
    sourceEpisodes: episodes,
    sourceDisclosureLabel,
    skill: {
      ...candidate.skill,
      source_episode_ids: sourceEpisodeIds,
      disclosure_label: sourceDisclosureLabel,
    },
  };
}

function buildPlanChange(item: ProceduralSynthesizerPlan["items"][number]): OfflineChange {
  return {
    process: "procedural-synthesizer",
    action: "synthesize",
    targets: {
      cluster: item.cluster_key,
      evidence_ids: item.evidence.map((evidence) => evidence.id),
    },
    preview: {
      applies_when: item.candidate.applies_when,
      approach: item.candidate.approach,
      abstraction_fit: item.candidate.abstraction_fit,
      deduped: item.dedup_decision.skill_id !== null,
      rejected_reason: item.rejection_reason,
      support: item.evidence.length,
      source_episode_ids: item.source_episode_ids,
      source_disclosure_label: memoryDisclosurePayloadFields(item.source_disclosure_label)
        .disclosure_label,
    },
  };
}

function buildSplitPlanChange(
  item: ProceduralSynthesizerPlan["split_items"][number],
): OfflineChange {
  return {
    process: "procedural-synthesizer",
    action: "skill_split_proposal",
    targets: {
      skill_id: item.skill.id,
      context_keys: item.buckets.map((bucket) => bucket.stats.context_key),
    },
    preview: {
      decision: item.proposal.decision,
      rationale: item.proposal.rationale,
      divergence: item.divergence,
      buckets: item.buckets.map((bucket) => ({
        context_key: bucket.stats.context_key,
        posterior_mean: bucket.posterior_mean,
        attempts: bucket.stats.attempts,
      })),
      parts: item.proposal.parts ?? [],
    },
  };
}

function buildQueuedSplitChange(input: {
  item: ProceduralSynthesizerPlan["split_items"][number];
  reviewItem: ReviewQueueItem;
}): OfflineChange {
  return {
    ...buildSplitPlanChange(input.item),
    targets: {
      skill_id: input.item.skill.id,
      review_item_id: input.reviewItem.id,
      context_keys: input.item.buckets.map((bucket) => bucket.stats.context_key),
    },
  };
}

function buildAppliedChange(input: {
  item: ProceduralSynthesizerPlan["items"][number];
  skill: SkillRecord;
  deduped: boolean;
}): OfflineChange {
  return {
    process: "procedural-synthesizer",
    action: "skill_synthesis",
    targets: {
      skill_id: input.skill.id,
      evidence_ids: input.item.evidence.map((evidence) => evidence.id),
    },
    preview: {
      applies_when: input.skill.applies_when,
      approach: input.skill.approach,
      deduped: input.deduped,
      support: input.item.evidence.length,
    },
  };
}

function buildSkillSplitReviewPayload(input: {
  item: ProceduralSynthesizerPlan["split_items"][number];
  parts: Array<{ applies_when: string; approach: string; target_contexts: string[] }>;
  proposedAt: number;
  splitCooldownDays: number;
  splitClaimStaleSec: number;
}): SkillSplitReviewPayload {
  const bucketByContext = new Map(
    input.item.buckets.map((bucket) => [bucket.stats.context_key, bucket]),
  );
  const childSpecs = input.parts.map((part) => ({
    label: part.applies_when,
    problem: part.applies_when,
    approach: part.approach,
    context_stats: part.target_contexts.map((contextKey) => bucketByContext.get(contextKey)!.stats),
  }));

  return {
    target_type: "skill",
    target_id: input.item.skill.id,
    original_skill_id: input.item.skill.id,
    proposed_children: childSpecs,
    rationale: input.item.proposal.rationale,
    evidence_summary: {
      source_episode_ids: input.item.skill.source_episode_ids,
      source_disclosure_label: memoryDisclosurePayloadFields(
        input.item.skill.disclosure_label ?? unknownMemoryDisclosureLabel(),
      ).disclosure_label,
      divergence: input.item.divergence,
      min_posterior_mean: input.item.min_posterior_mean,
      max_posterior_mean: input.item.max_posterior_mean,
      buckets: input.item.buckets.map((bucket) => ({
        context_key: bucket.stats.context_key,
        posterior_mean: bucket.posterior_mean,
        alpha: bucket.stats.alpha,
        beta: bucket.stats.beta,
        attempts: bucket.stats.attempts,
        successes: bucket.stats.successes,
        failures: bucket.stats.failures,
        last_used: bucket.stats.last_used,
        last_successful: bucket.stats.last_successful,
      })),
    },
    cooldown: {
      proposed_at: input.proposedAt,
      claimed_at: input.item.split_claimed_at ?? null,
      claim_expires_at:
        input.item.split_claimed_at === undefined || input.item.split_claimed_at === null
          ? null
          : input.item.split_claimed_at + input.splitClaimStaleSec * 1_000,
      split_cooldown_days: input.splitCooldownDays,
      split_claim_stale_sec: input.splitClaimStaleSec,
      last_split_attempt_at: input.item.skill.last_split_attempt_at ?? null,
      split_failure_count: input.item.skill.split_failure_count,
      last_split_error: input.item.skill.last_split_error,
    },
  };
}

function hasOpenSkillSplitReview(
  reviewItems: readonly ReviewQueueItem[],
  skillId: SkillRecord["id"],
): boolean {
  return reviewItems.some(
    (item) => item.kind === "skill_split" && item.refs.original_skill_id === skillId,
  );
}

function shouldSkipSynthesizedOutcome(
  evidence: ProceduralEvidenceRecord,
  skill: SkillRecord,
): boolean {
  return (
    evidence.pending_attempt_snapshot.selected_skill_id === skill.id &&
    (evidence.classification === "success" || evidence.classification === "failure")
  );
}

function recordClusterOutcomes(
  ctx: OfflineContext,
  skill: SkillRecord,
  item: ProceduralSynthesizerPlan["items"][number],
): SkillRecord {
  let current = skill;

  for (const evidence of item.evidence) {
    if (evidence.classification === "unclear" || shouldSkipSynthesizedOutcome(evidence, skill)) {
      continue;
    }

    current = ctx.skillRepository.recordOutcome(
      skill.id,
      evidence.classification === "success",
      uniqueEpisodeIds([...evidence.resolved_episode_ids, ...item.source_episode_ids]),
      evidence.procedural_context ?? evidence.pending_attempt_snapshot.procedural_context ?? null,
    );
  }

  return current;
}

function validateSplitParts(
  item: ProceduralSynthesizerPlan["split_items"][number],
): Array<{ applies_when: string; approach: string; target_contexts: string[] }> {
  if (item.proposal.decision !== "split") {
    return [];
  }

  const parts = item.proposal.parts ?? [];

  if (parts.length < 2) {
    throw new StorageError("Skill split decision must include at least two parts", {
      code: "PROCEDURAL_SKILL_SPLIT_PARTS_INVALID",
    });
  }

  const statsByContext = new Map(
    item.buckets.map((bucket) => [bucket.stats.context_key, bucket.stats]),
  );
  const allowedContexts = new Set(statsByContext.keys());
  const assignedContexts = new Set<string>();

  const normalizedParts = parts.map((part) => {
    const targetContexts = [...new Set(part.target_contexts.map((contextKey) => contextKey.trim()))]
      .filter((contextKey) => contextKey.length > 0)
      .sort();

    if (targetContexts.length === 0) {
      throw new StorageError("Skill split part has no target contexts", {
        code: "PROCEDURAL_SKILL_SPLIT_TARGETS_EMPTY",
      });
    }

    for (const contextKey of targetContexts) {
      if (!allowedContexts.has(contextKey)) {
        throw new StorageError(`Unknown split target context: ${contextKey}`, {
          code: "PROCEDURAL_SKILL_SPLIT_TARGET_UNKNOWN",
        });
      }

      if (assignedContexts.has(contextKey)) {
        throw new StorageError(`Split target context assigned more than once: ${contextKey}`, {
          code: "PROCEDURAL_SKILL_SPLIT_TARGET_DUPLICATE",
        });
      }

      assignedContexts.add(contextKey);
    }

    return {
      applies_when: part.applies_when.trim(),
      approach: part.approach.trim(),
      target_contexts: targetContexts,
    };
  });

  if (assignedContexts.size !== allowedContexts.size) {
    const missingContexts = [...allowedContexts].filter(
      (contextKey) => !assignedContexts.has(contextKey),
    );

    throw new StorageError(
      `Skill split did not cover all divergent context buckets: ${missingContexts.join(", ")}`,
      {
        code: "PROCEDURAL_SKILL_SPLIT_TARGETS_INCOMPLETE",
      },
    );
  }

  return normalizedParts;
}

async function appendSkillSplitInternalEvent(
  ctx: OfflineContext,
  content: Record<string, unknown>,
  errors: OfflineProcessError[],
): Promise<void> {
  try {
    await ctx.streamWriter.append({
      kind: "internal_event",
      content,
    });
  } catch (error) {
    errors.push(
      offlineProcessError("procedural-synthesizer", error, {
        code: "procedural_skill_split_log_failed",
      }),
    );
  }
}

export type ProceduralSynthesizerProcessOptions = {
  skillRepository: OfflineContext["skillRepository"];
  proceduralEvidenceRepository: OfflineContext["proceduralEvidenceRepository"];
  registry: ReverserRegistry;
  clock?: Clock;
};

export class ProceduralSynthesizerProcess implements OfflineProcess<ProceduralSynthesizerPlan> {
  readonly name = "procedural-synthesizer" as const;
  private readonly clock: Clock;

  constructor(private readonly options: ProceduralSynthesizerProcessOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.options.registry.register(this.name, "skill_synthesis", async ({ reversal }) => {
      const parsed = proceduralSynthesizerReversalSchema.parse(reversal);

      if (parsed.inserted) {
        await this.options.skillRepository.delete(parsed.skillId);
      } else if (parsed.previousSkill !== undefined) {
        await this.options.skillRepository.replace(parsed.previousSkill);
      }

      this.options.proceduralEvidenceRepository.markUnconsumed(parsed.evidenceIds);
    });
    this.options.registry.register(this.name, "skill_split", async ({ reversal }) => {
      const parsed = proceduralSkillSplitReversalSchema.parse(reversal);

      for (const skill of parsed.createdSkills) {
        await this.options.skillRepository.delete(skill.id);
      }

      await this.options.skillRepository.replace(parsed.originalSkill);
      this.options.skillRepository.restoreContextStats(parsed.movedContextStats);
    });
  }

  async plan(
    ctx: OfflineContext,
    opts: { dryRun?: boolean; budget?: number; params?: Record<string, unknown> } = {},
  ): Promise<ProceduralSynthesizerPlan> {
    const errors: OfflineProcessError[] = [];
    const dryRun = opts.dryRun === true;
    const budget = opts.budget ?? ctx.config.offline.proceduralSynthesizer.budget;
    const minSupport = ctx.config.offline.proceduralSynthesizer.minSupport;
    const synthesizerConfig = ctx.config.offline.proceduralSynthesizer;
    const sourceEvidence = ctx.proceduralEvidenceRepository
      .listUnconsumed()
      .filter(
        (evidence) =>
          evidence.classification === "success" &&
          evidence.grounded &&
          evidence.skill_actually_applied,
      );
    const clusters = await collectEvidenceClusters(ctx, sourceEvidence);
    const candidateClusters = clusters
      .filter((cluster) => cluster.evidence.length >= minSupport)
      .slice(0, synthesizerConfig.maxSkillsPerRun);
    const skills = ctx.skillRepository.list(500);
    const contextStatsBySkillId = ctx.skillRepository.batchListContextStatsForSkills(
      skills.map((skill) => skill.id),
    );
    const splitCandidates = detectDivergentSkillSplits({
      skills,
      contextStatsBySkillId,
      nowMs: this.clock.now(),
      minContextAttemptsForSplit: synthesizerConfig.minContextAttemptsForSplit,
      minDivergenceForSplit: synthesizerConfig.minDivergenceForSplit,
      splitCooldownDays: synthesizerConfig.splitCooldownDays,
      splitClaimStaleSec: synthesizerConfig.splitClaimStaleSec,
      maxSplitParseFailures: synthesizerConfig.maxSplitParseFailures,
    }).slice(0, synthesizerConfig.maxSkillsPerRun);
    const items: ProceduralSynthesizerPlan["items"] = [];
    const splitItems: ProceduralSynthesizerPlan["split_items"] = [];
    let tokensUsed = 0;
    let budgetExhausted = false;

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient: LLMClient = wrapClient(ctx.llm.background);

        for (const cluster of candidateClusters) {
          try {
            const candidate = (
              await callStructuredTool({
                llmClient,
                request: {
                  model: ctx.config.anthropic.models.background,
                  system:
                    "You synthesize reusable procedural memory from successful evidence clusters. Be conservative and reject narrow or generic candidates.",
                  messages: [
                    {
                      role: "user",
                      content: buildPrompt(cluster),
                    },
                  ],
                  tools: [PROCEDURAL_SYNTHESIZER_TOOL],
                  tool_choice: { type: "tool", name: SYNTHESIZER_TOOL_NAME },
                  max_tokens: 1_500,
                  budget: "offline-procedural-synthesizer",
                },
                toolName: SYNTHESIZER_TOOL_NAME,
                parse: parseSkillCandidate,
              })
            ).parsed;
            const rejectionReason =
              candidate.abstraction_fit !== "usable"
                ? "unusable_abstraction"
                : candidate.rejection_reason;
            const similarSkill =
              rejectionReason !== null
                ? undefined
                : (await ctx.skillRepository.searchByContext(candidate.applies_when, 3)).find(
                    (item) =>
                      item.similarity >= ctx.config.offline.proceduralSynthesizer.dedupThreshold,
                  );

            items.push({
              cluster_key: cluster.key,
              evidence: cluster.evidence,
              source_episode_ids: cluster.sourceEpisodeIds,
              source_disclosure_label: cluster.sourceDisclosureLabel,
              candidate,
              dedup_decision: {
                skill_id: similarSkill?.skill.id ?? null,
                similarity: similarSkill?.similarity ?? null,
              },
              rejection_reason: rejectionReason,
            });
          } catch (error) {
            if (error instanceof BudgetExceededError) {
              throw error;
            }

            errors.push(
              offlineProcessError(
                this.name,
                proceduralSynthesizerStructuredError(error, SYNTHESIZER_TOOL_NAME),
              ),
            );
          }
        }

        for (const candidate of splitCandidates) {
          const claimedAt = dryRun ? null : this.clock.now();

          if (
            claimedAt !== null &&
            !ctx.skillRepository.claimSplit({
              skillId: candidate.skill.id,
              claimedAt,
              staleBefore: claimedAt - synthesizerConfig.splitClaimStaleSec * 1_000,
            })
          ) {
            await appendSkillSplitInternalEvent(
              ctx,
              {
                hook: "skill_split_skipped",
                reason: "claim_not_acquired",
                skill_id: candidate.skill.id,
              },
              errors,
            );
            continue;
          }

          try {
            const labeledCandidate = await labeledSplitCandidate(ctx, candidate);

            const proposal = (
              await callStructuredTool({
                llmClient,
                request: {
                  model: ctx.config.anthropic.models.background,
                  system:
                    "You refactor procedural memory only when context-conditioned outcomes clearly show that one skill is hiding different reusable procedures. Be conservative.",
                  messages: [
                    {
                      role: "user",
                      content: buildSplitPrompt(labeledCandidate),
                    },
                  ],
                  tools: [PROCEDURAL_SKILL_SPLIT_TOOL],
                  tool_choice: { type: "tool", name: SKILL_SPLIT_TOOL_NAME },
                  max_tokens: 1_500,
                  budget: "offline-procedural-synthesizer",
                },
                toolName: SKILL_SPLIT_TOOL_NAME,
                parse: parseSkillSplitProposal,
              })
            ).parsed;
            const splitItem = {
              skill: labeledCandidate.skill,
              buckets: labeledCandidate.buckets,
              min_posterior_mean: labeledCandidate.min_posterior_mean,
              max_posterior_mean: labeledCandidate.max_posterior_mean,
              divergence: labeledCandidate.divergence,
              split_claimed_at: claimedAt,
              proposal,
            } satisfies ProceduralSynthesizerPlan["split_items"][number];

            if (proposal.decision === "split") {
              try {
                validateSplitParts(splitItem);
              } catch (error) {
                const message = error instanceof Error ? error.message : String(error);

                if (!dryRun) {
                  ctx.skillRepository.recordSplitAttemptAndClearClaim({
                    skillId: candidate.skill.id,
                    attemptedAt: this.clock.now(),
                    claimedAt,
                  });
                }
                splitItems.push({
                  ...splitItem,
                  proposal: {
                    decision: "no_split",
                    rationale: `Rejected split proposal: ${message}`,
                  },
                });
                continue;
              }
            }

            if (!dryRun && proposal.decision !== "split") {
              ctx.skillRepository.recordSplitAttemptAndClearClaim({
                skillId: candidate.skill.id,
                attemptedAt: this.clock.now(),
                claimedAt,
              });
            }

            splitItems.push(splitItem);
          } catch (error) {
            if (error instanceof BudgetExceededError) {
              if (claimedAt !== null) {
                ctx.skillRepository.clearSplitClaim({
                  skillId: candidate.skill.id,
                  claimedAt,
                  clearedAt: this.clock.now(),
                });
              }
              throw error;
            }

            const splitError = proceduralSynthesizerStructuredError(error, SKILL_SPLIT_TOOL_NAME);
            const message = splitError instanceof Error ? splitError.message : String(splitError);
            const failedSkill =
              !dryRun && isSkillSplitParseFailure(splitError)
                ? ctx.skillRepository.recordSplitFailureAndClearClaim({
                    skillId: candidate.skill.id,
                    attemptedAt: this.clock.now(),
                    claimedAt,
                    error: message,
                    manualReviewThreshold: synthesizerConfig.maxSplitParseFailures,
                  })
                : null;

            if (!dryRun && failedSkill === null) {
              ctx.skillRepository.recordSplitAttemptAndClearClaim({
                skillId: candidate.skill.id,
                attemptedAt: this.clock.now(),
                claimedAt,
              });
            }
            errors.push(offlineProcessError(this.name, splitError));
            if (!dryRun) {
              await appendSkillSplitInternalEvent(
                ctx,
                {
                  hook: "skill_split_failed",
                  error: message,
                  skill_id: candidate.skill.id,
                  split_failure_count: failedSkill?.split_failure_count ?? null,
                  split_suppressed:
                    (failedSkill?.split_failure_count ?? 0) >=
                    synthesizerConfig.maxSplitParseFailures,
                },
                errors,
              );
            }
          }
        }
      });

      tokensUsed += budgeted.tokens_used;
    } catch (error) {
      tokensUsed += getBudgetErrorTokens(error);
      budgetExhausted = error instanceof BudgetExceededError;
      errors.push(offlineProcessError(this.name, error));
    }

    return proceduralSynthesizerPlanSchema.parse({
      process: this.name,
      items,
      split_items: splitItems,
      budget,
      errors,
      tokens_used: tokensUsed,
      budget_exhausted: budgetExhausted,
    });
  }

  preview(plan: ProceduralSynthesizerPlan): OfflineResult {
    const parsed = proceduralSynthesizerPlanSchema.parse(plan);

    return {
      process: this.name,
      dryRun: true,
      changes: [
        ...parsed.items.map((item) => buildPlanChange(item)),
        ...parsed.split_items.map((item) => buildSplitPlanChange(item)),
      ],
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
    };
  }

  async apply(ctx: OfflineContext, rawPlan: ProceduralSynthesizerPlan): Promise<OfflineResult> {
    const plan = proceduralSynthesizerPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];
    const errors: OfflineProcessError[] = [...plan.errors];
    let tokensUsed = plan.tokens_used;
    let budgetExhausted = plan.budget_exhausted;

    for (const item of plan.items) {
      if (item.rejection_reason !== null) {
        continue;
      }

      try {
        const evidenceIds = item.evidence.map((evidence) => evidence.id);
        const similarSkill =
          item.dedup_decision.skill_id === null
            ? null
            : ctx.skillRepository.get(item.dedup_decision.skill_id);
        let skill: SkillRecord;
        let previousSkill: SkillRecord | undefined;
        let inserted = false;

        if (item.dedup_decision.skill_id !== null && similarSkill === null) {
          throw new StorageError(`Unknown dedup skill id: ${item.dedup_decision.skill_id}`, {
            code: "PROCEDURAL_SYNTHESIZER_DEDUP_MISSING",
          });
        }

        if (similarSkill === null) {
          skill = await ctx.skillRepository.add({
            applies_when: item.candidate.applies_when.trim(),
            approach: item.candidate.approach.trim(),
            sourceEpisodes: item.source_episode_ids,
            disclosureLabel: item.source_disclosure_label,
            priorAlpha: 2,
            priorBeta: 1,
          });
          inserted = true;
        } else {
          previousSkill = similarSkill;
          skill = similarSkill;
        }

        for (const evidence of item.evidence) {
          ctx.proceduralEvidenceRepository.updateResolvedEpisodeIds(
            evidence.id,
            item.source_episode_ids,
          );
        }

        skill = recordClusterOutcomes(ctx, skill, item);
        ctx.proceduralEvidenceRepository.markConsumed(evidenceIds, this.clock.now());
        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "skill_synthesis",
          targets: {
            skillId: skill.id,
            evidenceIds,
          },
          reversal: {
            skillId: skill.id,
            inserted,
            ...(previousSkill === undefined ? {} : { previousSkill }),
            newSkill: skill,
            evidenceIds,
          },
        });
        changes.push(buildAppliedChange({ item, skill, deduped: !inserted }));
      } catch (error) {
        errors.push(offlineProcessError(this.name, error));
      }
    }

    for (const item of plan.split_items) {
      try {
        if (item.proposal.decision !== "split") {
          await appendSkillSplitInternalEvent(
            ctx,
            {
              hook: "skill_split_decision",
              skill_id: item.skill.id,
              decision: item.proposal.decision,
              rationale: item.proposal.rationale,
              divergence: item.divergence,
            },
            errors,
          );
          ctx.skillRepository.recordSplitAttemptAndClearClaim({
            skillId: item.skill.id,
            attemptedAt: this.clock.now(),
            claimedAt: item.split_claimed_at ?? null,
          });
          continue;
        }

        const parts = validateSplitParts(item);
        const existingOpenSplitReviews = ctx.reviewQueueRepository.list({
          kind: "skill_split",
          openOnly: true,
        });

        if (hasOpenSkillSplitReview(existingOpenSplitReviews, item.skill.id)) {
          continue;
        }

        const proposedAt = this.clock.now();
        const reviewItem = ctx.reviewQueueRepository.enqueue({
          kind: "skill_split",
          refs: buildSkillSplitReviewPayload({
            item,
            parts,
            proposedAt,
            splitCooldownDays: ctx.config.offline.proceduralSynthesizer.splitCooldownDays,
            splitClaimStaleSec: ctx.config.offline.proceduralSynthesizer.splitClaimStaleSec,
          }),
          reason: `Skill split proposed for divergent context outcomes on ${item.skill.applies_when}`,
        });
        changes.push(buildQueuedSplitChange({ item, reviewItem }));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        ctx.skillRepository.recordSplitAttemptAndClearClaim({
          skillId: item.skill.id,
          attemptedAt: this.clock.now(),
          claimedAt: item.split_claimed_at ?? null,
        });
        errors.push(offlineProcessError(this.name, error));
        await appendSkillSplitInternalEvent(
          ctx,
          {
            hook: "skill_split_failed",
            error: message,
            skill_id: item.skill.id,
          },
          errors,
        );
      }
    }

    return {
      process: this.name,
      dryRun: false,
      changes,
      tokens_used: tokensUsed,
      errors,
      budget_exhausted: budgetExhausted,
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
