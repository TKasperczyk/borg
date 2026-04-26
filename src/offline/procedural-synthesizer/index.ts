import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import {
  episodeIdSchema,
  isEpisodeInGlobalIdentityScope,
  type Episode,
} from "../../memory/episodic/index.js";
import {
  isProceduralOutcomeEvidenceGrounded,
  proceduralEvidenceIdSchema,
  proceduralEvidenceSchema,
  skillIdSchema,
  skillSchema,
  type ProceduralEvidenceRecord,
  type SkillRecord,
} from "../../memory/procedural/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { BudgetExceededError, StorageError } from "../../util/errors.js";
import type { EpisodeId } from "../../util/ids.js";

import type { ReverserRegistry } from "../audit-log.js";
import { getBudgetErrorTokens, withBudget } from "../budget.js";
import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessError,
  OfflineResult,
} from "../types.js";

const CLUSTER_SIMILARITY_THRESHOLD = 0.85;
const SYNTHESIZER_TOOL_NAME = "EmitProceduralSkillCandidate";

const skillCandidateSchema = z.object({
  applies_when: z.string().min(1),
  approach: z.string().min(1),
  abstraction_fit: z.enum(["too_narrow", "usable", "too_broad"]),
});

export const PROCEDURAL_SYNTHESIZER_TOOL = {
  name: SYNTHESIZER_TOOL_NAME,
  description: "Emit a reusable procedural skill candidate from repeated successful attempts.",
  inputSchema: toToolInputSchema(skillCandidateSchema),
} satisfies LLMToolDefinition;

const proceduralSynthesizerRejectionReasonSchema = z.enum([
  "unusable_abstraction",
  "centered_proper_noun",
]);

const proceduralSynthesizerDedupDecisionSchema = z.object({
  skill_id: skillIdSchema.nullable(),
  similarity: z.number().min(0).max(1).nullable(),
});

const proceduralSynthesizerPlanItemSchema = z.object({
  cluster_key: z.string().min(1),
  evidence: z.array(proceduralEvidenceSchema).min(2),
  source_episode_ids: z.array(episodeIdSchema).min(1),
  candidate: skillCandidateSchema,
  dedup_decision: proceduralSynthesizerDedupDecisionSchema,
  rejection_reason: proceduralSynthesizerRejectionReasonSchema.nullable(),
});

export const proceduralSynthesizerPlanSchema = z.object({
  process: z.literal("procedural-synthesizer"),
  items: z.array(proceduralSynthesizerPlanItemSchema),
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

type EvidenceCluster = {
  key: string;
  evidence: ProceduralEvidenceRecord[];
  sourceEpisodeIds: EpisodeId[];
};

const COMMON_PROPER_NOUNS = new Set([
  "I",
  "The",
  "A",
  "An",
  "User",
  "Agent",
  "Assistant",
  "Borg",
  "This",
  "That",
  "When",
  "If",
  "Use",
  "Try",
]);

function cosineSimilarity(left: Float32Array, right: Float32Array): number {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;

  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    const leftValue = left[index] ?? 0;
    const rightValue = right[index] ?? 0;
    dot += leftValue * rightValue;
    leftNorm += leftValue * leftValue;
    rightNorm += rightValue * rightValue;
  }

  if (leftNorm === 0 || rightNorm === 0) {
    return 0;
  }

  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

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

  if (exact !== null) {
    resolved.push(exact.id);
  } else {
    const sourceIds = evidence.pending_attempt_snapshot.source_stream_ids;
    const episodes = await ctx.episodicRepository.listAll();
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
  const embeddings = await Promise.all(
    evidenceRows.map((evidence) => ctx.embeddingClient.embed(evidenceEmbeddingText(evidence))),
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

    const sourceEpisodeIds = uniqueEpisodeIds(
      (
        await Promise.all(
          clusterEvidence.map((evidence) => resolveEvidenceEpisodeIds(ctx, evidence)),
        )
      ).flat(),
    );

    if (sourceEpisodeIds.length === 0) {
      continue;
    }

    clusters.push({
      key: `procedural:${seed.id}`,
      evidence: clusterEvidence,
      sourceEpisodeIds,
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
    "Synthesize one reusable procedural skill from repeated successful problem-solving attempts.",
    `Emit your result by calling the ${SYNTHESIZER_TOOL_NAME} tool exactly once.`,
    "The skill should describe a reusable problem class and concrete checks or moves.",
    "Mark abstraction_fit as too_narrow when the skill is tied to a specific named project, person, or incident.",
    "Mark abstraction_fit as too_broad when it is generic advice rather than a reusable procedure.",
    `Cluster: ${cluster.key}`,
    "Evidence:",
    ...cluster.evidence.map((evidence) =>
      JSON.stringify({
        id: evidence.id,
        problem_text: evidence.pending_attempt_snapshot.problem_text,
        approach_summary: evidence.pending_attempt_snapshot.approach_summary,
        classification: evidence.classification,
        outcome_evidence: evidence.evidence_text,
        source_episode_ids: evidence.resolved_episode_ids,
      }),
    ),
  ].join("\n");
}

function parseSkillCandidate(result: LLMCompleteResult) {
  const call = result.tool_calls.find((toolCall) => toolCall.name === SYNTHESIZER_TOOL_NAME);

  if (call === undefined) {
    throw new StorageError(`Procedural synthesizer did not emit tool ${SYNTHESIZER_TOOL_NAME}`, {
      code: "PROCEDURAL_SYNTHESIZER_INVALID",
    });
  }

  return skillCandidateSchema.parse(call.input);
}

function extractCenteredProperNouns(cluster: EvidenceCluster): string[] {
  const counts = new Map<string, number>();

  for (const evidence of cluster.evidence) {
    const text = evidenceEmbeddingText(evidence);
    const names = new Set(text.match(/\b[A-Z][A-Za-z0-9]{2,}\b/g) ?? []);

    for (const name of names) {
      if (COMMON_PROPER_NOUNS.has(name)) {
        continue;
      }

      counts.set(name, (counts.get(name) ?? 0) + 1);
    }
  }

  return [...counts.entries()]
    .filter(([, count]) => count >= Math.min(2, cluster.evidence.length))
    .map(([name]) => name);
}

function containsCenteredProperNoun(
  candidateAppliesWhen: string,
  cluster: EvidenceCluster,
): boolean {
  return extractCenteredProperNouns(cluster).some((properNoun) =>
    new RegExp(`\\b${properNoun.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(
      candidateAppliesWhen,
    ),
  );
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
    );
  }

  return current;
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
  }

  async plan(
    ctx: OfflineContext,
    opts: { budget?: number; params?: Record<string, unknown> } = {},
  ): Promise<ProceduralSynthesizerPlan> {
    const errors: OfflineProcessError[] = [];
    const budget = opts.budget ?? ctx.config.offline.proceduralSynthesizer.budget;
    const selfAudienceEntityId = ctx.entityRepository.findByName("self");
    const minSupport = ctx.config.offline.proceduralSynthesizer.minSupport;
    const sourceEvidence = ctx.proceduralEvidenceRepository
      .listUnconsumed()
      .filter(
        (evidence) =>
          evidence.classification === "success" &&
          isProceduralOutcomeEvidenceGrounded(evidence) &&
          isEpisodeInGlobalIdentityScope(
            { audience_entity_id: evidence.audience_entity_id },
            selfAudienceEntityId,
          ),
      );
    const clusters = await collectEvidenceClusters(ctx, sourceEvidence);
    const candidateClusters = clusters
      .filter((cluster) => cluster.evidence.length >= minSupport)
      .slice(0, ctx.config.offline.proceduralSynthesizer.maxSkillsPerRun);
    const items: ProceduralSynthesizerPlan["items"] = [];
    let tokensUsed = 0;
    let budgetExhausted = false;

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient: LLMClient = wrapClient(ctx.llm.background);

        for (const cluster of candidateClusters) {
          try {
            const candidate = parseSkillCandidate(
              await llmClient.complete({
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
              }),
            );
            const rejectionReason =
              candidate.abstraction_fit !== "usable"
                ? "unusable_abstraction"
                : containsCenteredProperNoun(candidate.applies_when, cluster)
                  ? "centered_proper_noun"
                  : null;
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

            errors.push({
              process: this.name,
              message: error instanceof Error ? error.message : String(error),
              code: error instanceof Error && "code" in error ? String(error.code) : undefined,
            });
          }
        }
      });

      tokensUsed += budgeted.tokens_used;
    } catch (error) {
      tokensUsed += getBudgetErrorTokens(error);
      budgetExhausted = error instanceof BudgetExceededError;
      errors.push({
        process: this.name,
        message: error instanceof Error ? error.message : String(error),
        code: error instanceof Error && "code" in error ? String(error.code) : undefined,
      });
    }

    return proceduralSynthesizerPlanSchema.parse({
      process: this.name,
      items,
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
      changes: parsed.items.map((item) => buildPlanChange(item)),
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
        errors.push({
          process: this.name,
          message: error instanceof Error ? error.message : String(error),
          code: error instanceof Error && "code" in error ? String(error.code) : undefined,
        });
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
