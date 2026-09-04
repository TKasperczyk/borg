import { z } from "zod";

import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { OFFLINE_REFLECTOR_PROMPT_PREAMBLE } from "../../cognition/prompts/reflector.js";
import {
  goalMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
} from "../../memory/common/disclosure-serializers.js";
import { episodeIdSchema, type Episode } from "../../memory/episodic/index.js";
import { memoryDisclosureLabelSchema } from "../../memory/common/disclosure-label.js";
import type { GoalRecord } from "../../memory/self/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import {
  semanticEdgeIdSchema,
  semanticNodeCorrectionRefSchema,
  semanticNodeIdSchema,
  semanticNodeKindSchema,
  semanticNodeSchema,
  semanticNodeStatusSchema,
  type SemanticNode,
} from "../../memory/semantic/index.js";
import { bestVectorSimilarity, cosineSimilarity } from "../../retrieval/embedding-similarity.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { createSemanticEdgeId, createSemanticNodeId } from "../../util/ids.js";
import { BudgetExceededError, SemanticError } from "../../util/errors.js";

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
import { disclosureLabelForEpisodeIds, episodeEvidencePromptRow } from "../evidence-labels.js";
import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";

const insightResponseSchema = z.object({
  label: z.string().min(1),
  description: z.string().min(1),
  confidence: z.number().min(0).max(1),
  source_episode_ids: z.array(z.string().min(1)).min(1),
});
const REFLECTOR_TOOL_NAME = "EmitReflectorInsights";
export const REFLECTOR_TOOL = {
  name: REFLECTOR_TOOL_NAME,
  description: "Emit a grounded semantic insight from repeated episodic evidence.",
  inputSchema: toToolInputSchema(insightResponseSchema),
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

const reflectorTargetSchema = z.discriminatedUnion("mode", [
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

const reflectorSupportEdgeCandidateSchema = z.object({
  id: semanticEdgeIdSchema,
  insight_node_id: semanticNodeIdSchema,
  target_node_id: semanticNodeIdSchema,
  source_episode_ids: z.array(episodeIdSchema).min(1),
  confidence: z.number().min(0).max(1),
});

const reflectorPlanItemSchema = z.object({
  cluster_key: z.string().min(1),
  episode_ids: z.array(episodeIdSchema).min(1),
  source_disclosure_label: memoryDisclosureLabelSchema,
  target: reflectorTargetSchema,
  candidate_support_edges: z.array(reflectorSupportEdgeCandidateSchema).default([]),
  review: z.object({
    kind: z.literal("new_insight"),
    reason: z.string().min(1),
  }),
});

export const reflectorPlanSchema = z.object({
  process: z.literal("reflector"),
  items: z.array(reflectorPlanItemSchema),
  errors: z
    .array(
      z.object({
        process: z.literal("reflector"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
});

export type ReflectorPlan = z.infer<typeof reflectorPlanSchema>;

const ABSOLUTE_CONFIDENCE_CEILING = 0.5;
const DEDUP_THRESHOLD = 0.88;

type ReflectionCluster = {
  key: string;
  episodes: Episode[];
};

type ReflectionGoalVector = {
  key: string;
  vector: Float32Array;
};

type ReflectionTagVector = {
  tag: string;
  vector: Float32Array;
};

type ReflectionTagGroup = {
  key: string;
  tags: readonly string[];
};

const reflectorReversalSchema = z.object({
  nodeId: semanticNodeIdSchema,
  nodeCreated: z.boolean(),
  previousNode: serializableSemanticNodeSchema.optional(),
  anchorNodeId: semanticNodeIdSchema.optional(),
  edgeIds: z.array(semanticEdgeIdSchema),
  reviewItemId: z.number().int().positive().optional(),
});

type ReflectorReversal = z.infer<typeof reflectorReversalSchema>;

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

function renderActiveGoalPromptRow(goal: GoalRecord): string {
  return JSON.stringify({
    id: goal.id,
    description: goal.description,
    counterparty_entity_id: goal.counterparty_entity_id ?? null,
    ...memoryDisclosurePayloadFields(goalMemoryDisclosureLabel(goal)),
  });
}

function buildPrompt(cluster: ReflectionCluster, activeGoals: readonly GoalRecord[]): string {
  const goals = activeGoals.map((goal) => renderActiveGoalPromptRow(goal)).join(" | ") || "none";

  return [
    OFFLINE_REFLECTOR_PROMPT_PREAMBLE,
    `Cluster key: ${cluster.key}`,
    `Active goals: ${goals}`,
    "Episodes:",
    ...cluster.episodes.map((episode) =>
      JSON.stringify(
        episodeEvidencePromptRow(episode, {
          tags: episode.tags,
          participants: episode.participants,
        }),
      ),
    ),
  ].join("\n");
}

function invalidInsight(error: unknown): unknown {
  if (isStructuredToolCallError(error, "missing_tool_call")) {
    return new SemanticError(`Reflector did not emit tool ${REFLECTOR_TOOL_NAME}`, {
      code: "REFLECTOR_INVALID",
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

function parseInsight(input: unknown) {
  return insightResponseSchema.parse(input);
}

function collectReflectionClusters(
  episodes: readonly Episode[],
  goalVectors: readonly ReflectionGoalVector[],
  tagGroups: readonly ReflectionTagGroup[],
  minSupport: number,
  maxInsightsPerRun: number,
  goalSimilarityThreshold: number,
): ReflectionCluster[] {
  const byKey = new Map<string, Episode[]>();
  const tagGroupByTag = new Map<string, string>();

  for (const group of tagGroups) {
    for (const tag of group.tags) {
      tagGroupByTag.set(tag, group.key);
    }
  }

  for (const episode of episodes) {
    for (const tag of episode.tags) {
      const groupKey = tagGroupByTag.get(tag.trim());

      if (groupKey !== undefined) {
        const key = `tag:${groupKey}`;
        byKey.set(key, [...(byKey.get(key) ?? []), episode]);
      }
    }

    for (const goal of goalVectors) {
      const similarity = bestVectorSimilarity(episode.embedding, [goal.vector]);

      if (similarity < goalSimilarityThreshold) {
        continue;
      }

      const key = `goal:${goal.key}`;
      byKey.set(key, [...(byKey.get(key) ?? []), episode]);
    }
  }

  return [...byKey.entries()]
    .map(([key, clusterEpisodes]) => ({
      key,
      episodes: clusterEpisodes
        .sort((left, right) => right.updated_at - left.updated_at)
        .filter(
          (episode, index, items) => items.findIndex((item) => item.id === episode.id) === index,
        ),
    }))
    .filter((cluster) => cluster.episodes.length >= minSupport)
    .sort((left, right) => right.episodes.length - left.episodes.length)
    .slice(0, maxInsightsPerRun);
}

function uniqueEpisodeTags(episodes: readonly Episode[]): string[] {
  const tags: string[] = [];
  const seen = new Set<string>();

  for (const episode of episodes) {
    for (const tag of episode.tags) {
      const trimmed = tag.trim();

      if (trimmed.length === 0 || seen.has(trimmed)) {
        continue;
      }

      seen.add(trimmed);
      tags.push(trimmed);
    }
  }

  return tags;
}

async function buildReflectionTagGroups(input: {
  embeddingClient: EmbeddingClient;
  episodes: readonly Episode[];
  similarityThreshold: number;
}): Promise<ReflectionTagGroup[]> {
  const tags = uniqueEpisodeTags(input.episodes);

  if (tags.length === 0) {
    return [];
  }

  const embeddings = await input.embeddingClient.embedBatch(tags);
  const tagVectors = tags.flatMap((tag, index): ReflectionTagVector[] => {
    const vector = embeddings[index];
    return vector === undefined ? [] : [{ tag, vector }];
  });
  const remaining = new Set(tagVectors.map((_, index) => index));
  const groups: ReflectionTagGroup[] = [];

  for (let seedIndex = 0; seedIndex < tagVectors.length; seedIndex += 1) {
    if (!remaining.has(seedIndex)) {
      continue;
    }

    const seed = tagVectors[seedIndex];

    if (seed === undefined) {
      continue;
    }

    remaining.delete(seedIndex);
    const group = [seed];

    for (const candidateIndex of [...remaining]) {
      const candidate = tagVectors[candidateIndex];

      if (
        candidate !== undefined &&
        cosineSimilarity(seed.vector, candidate.vector) >= input.similarityThreshold
      ) {
        group.push(candidate);
        remaining.delete(candidateIndex);
      }
    }

    groups.push({
      key: group.map((item) => item.tag).join("+"),
      tags: group.map((item) => item.tag),
    });
  }

  return groups;
}

function semanticNodeMatchesReflectorCandidate(
  node: SemanticNode,
  candidateEmbedding: Float32Array,
): boolean {
  return (
    node.kind === "proposition" &&
    cosineSimilarity(node.embedding, candidateEmbedding) >= DEDUP_THRESHOLD
  );
}

function uniqueSemanticNodes(nodes: readonly SemanticNode[]): SemanticNode[] {
  const byId = new Map<SemanticNode["id"], SemanticNode>();

  for (const node of nodes) {
    byId.set(node.id, node);
  }

  return [...byId.values()];
}

async function buildInsightCandidate(
  ctx: OfflineContext,
  llmClient: LLMClient,
  cluster: ReflectionCluster,
  activeGoals: readonly GoalRecord[],
): Promise<{
  label: string;
  description: string;
  confidence: number;
  sourceEpisodeIds: Episode["id"][];
  embedding: Float32Array;
}> {
  let insight: z.infer<typeof insightResponseSchema>;

  try {
    insight = (
      await callStructuredTool({
        llmClient,
        request: {
          model: ctx.config.anthropic.models.background,
          system: [
            "I propose low-confidence semantic propositions grounded in repeated episodic evidence.",
            `${SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE} Apply this to a proposition's label and description when the proposition is about myself -- my own dispositions, patterns, beliefs, reflexes, or what I did or decided. Keep propositions about the world or about other people in their natural third-person form, naming those people.`,
          ].join(" "),
          messages: [
            {
              role: "user",
              content: buildPrompt(cluster, activeGoals),
            },
          ],
          tools: [REFLECTOR_TOOL],
          tool_choice: { type: "tool", name: REFLECTOR_TOOL_NAME },
          max_tokens: 4_000,
          budget: "offline-reflector",
        },
        toolName: REFLECTOR_TOOL_NAME,
        parse: parseInsight,
      })
    ).parsed;
  } catch (error) {
    throw invalidInsight(error);
  }

  const allowedIds = new Set(cluster.episodes.map((episode) => episode.id));
  // The model occasionally over-cites episodes outside the cluster. The stored insight
  // is grounded in the cluster's episodes regardless (clusterEpisodeIds at the call
  // site), so drop out-of-support refs (reference-validity hygiene) and only reject
  // when NONE of the cited episodes are in the support set -- i.e. the insight is
  // grounded entirely outside its own cluster.
  const groundedEpisodeIds = insight.source_episode_ids.filter((episodeId) =>
    allowedIds.has(episodeId as Episode["id"]),
  );

  if (groundedEpisodeIds.length === 0) {
    throw new SemanticError("Reflector referenced episodes outside the support set", {
      code: "REFLECTOR_INVALID_REF",
    });
  }

  const embedding = await ctx.embeddingClient.embed(`${insight.label}\n${insight.description}`);
  const ceiling = Math.min(
    ABSOLUTE_CONFIDENCE_CEILING,
    ctx.config.offline.reflector.ceilingConfidence,
  );

  return {
    label: insight.label.trim(),
    description: insight.description.trim(),
    confidence: Math.min(insight.confidence, ceiling),
    sourceEpisodeIds: groundedEpisodeIds as Episode["id"][],
    embedding,
  };
}

function buildChange(item: ReflectorPlan["items"][number]): OfflineChange {
  const nodeLabel =
    item.target.mode === "insert" ? item.target.node.label : `${item.target.node_id} (update)`;

  return {
    process: "reflector",
    action: "insight",
    targets: {
      cluster: item.cluster_key,
      episode_ids: item.episode_ids,
    },
    preview: {
      label: nodeLabel,
      confidence:
        item.target.mode === "insert" ? item.target.node.confidence : item.target.patch.confidence,
      deduped: item.target.mode === "update",
    },
  };
}

async function buildSupportEdgeCandidates(
  ctx: OfflineContext,
  insightNodeId: SemanticNode["id"],
  sourceEpisodeIds: readonly Episode["id"][],
  confidence: number,
): Promise<Array<z.infer<typeof reflectorSupportEdgeCandidateSchema>>> {
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
    reflectorSupportEdgeCandidateSchema.parse({
      id: createSemanticEdgeId(),
      insight_node_id: insightNodeId,
      target_node_id: targetNodeId,
      source_episode_ids: [...new Set(evidenceEpisodeIds)],
      confidence,
    }),
  );
}

export type ReflectorProcessOptions = {
  semanticNodeRepository: OfflineContext["semanticNodeRepository"];
  semanticEdgeRepository: OfflineContext["semanticEdgeRepository"];
  reviewQueueRepository: OfflineContext["reviewQueueRepository"];
  registry: ReverserRegistry;
  clock?: Clock;
};

export class ReflectorProcess implements OfflineProcess {
  readonly name = "reflector" as const;
  private readonly clock: Clock;

  constructor(private readonly options: ReflectorProcessOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.options.registry.register(this.name, "insight", async ({ reversal }) => {
      const parsed = reflectorReversalSchema.parse(reversal);

      for (const edgeId of parsed.edgeIds) {
        this.options.semanticEdgeRepository.invalidateEdge(edgeId, {
          at: this.clock.now(),
          by_process: "maintenance",
          reason: "reflector_audit_reversal",
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

      if (parsed.anchorNodeId !== undefined) {
        await this.options.semanticNodeRepository.update(parsed.anchorNodeId, {
          archived: true,
        });
      }

      if (typeof parsed.reviewItemId === "number") {
        this.options.reviewQueueRepository.delete(parsed.reviewItemId);
      }
    });
  }

  async plan(ctx: OfflineContext, opts: { budget?: number } = {}): Promise<ReflectorPlan> {
    const errors: OfflineProcessError[] = [];
    const items: ReflectorPlan["items"] = [];
    const budget = opts.budget ?? ctx.config.offline.reflector.budget;
    const episodes = await ctx.episodicRepository.listEffectivelyVisible();
    const rawActiveGoals = ctx.goalsRepository.list({ status: "active" });
    const activeGoals =
      ctx.sourceStreamAudienceDisclosureResolver?.resolve({ goalTrees: rawActiveGoals })
        .goalTrees ?? rawActiveGoals;
    let goalVectors: ReflectionGoalVector[] = [];
    let tagGroups: ReflectionTagGroup[] = [];

    if (activeGoals.length > 0) {
      try {
        const embeddings = await ctx.embeddingClient.embedBatch(
          activeGoals.map((goal) => goal.description),
        );
        goalVectors = activeGoals.flatMap((goal, index) => {
          const vector = embeddings[index];

          if (vector === undefined) {
            return [];
          }

          return [
            {
              key: goal.id,
              vector,
            },
          ];
        });
      } catch (error) {
        errors.push(offlineProcessError(this.name, error));
      }
    }
    if (episodes.some((episode) => episode.tags.length > 0)) {
      try {
        tagGroups = await buildReflectionTagGroups({
          embeddingClient: ctx.embeddingClient,
          episodes,
          similarityThreshold: ctx.config.offline.reflector.goalSimilarityThreshold,
        });
      } catch (error) {
        errors.push(offlineProcessError(this.name, error));
      }
    }
    const clusters = collectReflectionClusters(
      episodes,
      goalVectors,
      tagGroups,
      ctx.config.offline.reflector.minSupport,
      ctx.config.offline.reflector.maxInsightsPerRun,
      ctx.config.offline.reflector.goalSimilarityThreshold,
    );
    let tokensUsed = 0;
    let budgetExhausted = false;

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient = wrapClient(ctx.llm.background);

        for (const cluster of clusters) {
          try {
            const candidate = await buildInsightCandidate(ctx, llmClient, cluster, activeGoals);
            const byLabel = await ctx.semanticNodeRepository.findByExactLabelOrAlias(
              candidate.label,
              3,
              {
                includeArchived: true,
              },
            );
            const byVector = await ctx.semanticNodeRepository.searchByVector(candidate.embedding, {
              limit: 3,
              minSimilarity: DEDUP_THRESHOLD,
              kindFilter: ["proposition"],
              includeArchived: false,
            });
            const eligibleByLabel: SemanticNode[] = [];

            for (const node of byLabel) {
              if (semanticNodeMatchesReflectorCandidate(node, candidate.embedding)) {
                eligibleByLabel.push(node);
              }
            }

            const eligibleByVector: SemanticNode[] = [];

            for (const item of byVector) {
              if (semanticNodeMatchesReflectorCandidate(item.node, candidate.embedding)) {
                eligibleByVector.push(item.node);
              }
            }

            const eligibleNodes = uniqueSemanticNodes([...eligibleByLabel, ...eligibleByVector]);
            const existing = eligibleNodes.length === 1 ? eligibleNodes[0] : undefined;
            const dedupeDecision =
              existing === undefined
                ? "none"
                : eligibleByLabel.some((node) => node.id === existing.id)
                  ? "exact_label_or_alias"
                  : "vector_similarity";
            const timestamp = ctx.clock.now();
            const clusterEpisodeIds = cluster.episodes.map((episode) => episode.id);
            const target =
              existing === undefined
                ? {
                    mode: "insert" as const,
                    node: serializableSemanticNodeSchema.parse({
                      id: createSemanticNodeId(),
                      kind: "proposition",
                      label: candidate.label,
                      description: candidate.description,
                      aliases: [],
                      confidence: candidate.confidence,
                      source_episode_ids: clusterEpisodeIds,
                      created_at: timestamp,
                      updated_at: timestamp,
                      last_verified_at: timestamp,
                      embedding: Array.from(candidate.embedding),
                      archived: false,
                      superseded_by: null,
                    }),
                  }
                : {
                    mode: "update" as const,
                    node_id: existing.id,
                    patch: {
                      description:
                        candidate.confidence >= existing.confidence
                          ? candidate.description
                          : existing.description,
                      confidence: Math.max(existing.confidence * 0.99, candidate.confidence),
                      source_episode_ids: clusterEpisodeIds,
                      last_verified_at: timestamp,
                      embedding: Array.from(candidate.embedding),
                      archived: false,
                    },
                  };
            const nodeId = target.mode === "insert" ? target.node.id : target.node_id;
            const sourceDisclosureEpisodeIds =
              existing === undefined
                ? clusterEpisodeIds
                : [...new Set([...existing.source_episode_ids, ...clusterEpisodeIds])];
            const sourceDisclosureLabel = await disclosureLabelForEpisodeIds(
              ctx.episodicRepository,
              sourceDisclosureEpisodeIds,
            );
            const candidateSupportEdges = await buildSupportEdgeCandidates(
              ctx,
              nodeId,
              candidate.sourceEpisodeIds,
              candidate.confidence,
            );
            items.push({
              cluster_key: cluster.key,
              episode_ids: cluster.episodes.map((episode) => episode.id),
              source_disclosure_label: sourceDisclosureLabel,
              target,
              candidate_support_edges: candidateSupportEdges,
              review: {
                kind: "new_insight",
                reason:
                  existing === undefined
                    ? `New low-confidence insight extracted from ${cluster.key}`
                    : `Existing insight revisited from ${cluster.key}`,
              },
            });
            if (ctx.tracer?.enabled === true) {
              ctx.tracer.emit("reflector.candidate.completed", {
                turnId: ctx.runId,
                kind: "new_insight",
                dedupe_decision: dedupeDecision,
                support_count: cluster.episodes.length,
                persistence_decision: "review_queue_enqueue_planned",
              });
            }
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

    return reflectorPlanSchema.parse({
      process: this.name,
      items,
      errors,
      tokens_used: tokensUsed,
      budget_exhausted: budgetExhausted,
    });
  }

  preview(plan: ReflectorPlan): OfflineResult {
    const parsed = reflectorPlanSchema.parse(plan);

    return {
      process: this.name,
      dryRun: true,
      changes: parsed.items.map((item) => buildChange(item)),
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
      candidate_stats: {
        proposed: parsed.items.length + parsed.errors.length,
        accepted: 0,
        rejected: parsed.errors.length,
      },
    };
  }

  async apply(ctx: OfflineContext, rawPlan: ReflectorPlan): Promise<OfflineResult> {
    const plan = reflectorPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];

    for (const item of plan.items) {
      let nodeId: SemanticNode["id"];
      let nodeCreated = false;
      let previousNode: z.infer<typeof serializableSemanticNodeSchema> | undefined;

      if (item.target.mode === "insert") {
        nodeId = item.target.node.id;
        nodeCreated = true;
      } else {
        const current = await ctx.semanticNodeRepository.get(item.target.node_id);

        if (current === null) {
          throw new SemanticError(
            `Missing semantic node for reflector plan: ${item.target.node_id}`,
            {
              code: "REFLECTOR_PLAN_INVALID",
            },
          );
        }

        previousNode = serializeSemanticNode(current);
        nodeId = current.id;
      }

      const reviewItem = ctx.reviewQueueRepository.enqueue({
        kind: item.review.kind,
        refs: {
          node_ids: [nodeId],
          episode_ids: item.episode_ids,
          evidence_cluster_key: item.cluster_key,
          evidence_cluster_size: item.episode_ids.length,
          source_disclosure_label: item.source_disclosure_label,
          reflector_pending_insight: {
            target: item.target,
            candidate_support_edges: item.candidate_support_edges,
            evidence_cluster: {
              key: item.cluster_key,
              episode_ids: item.episode_ids,
              size: item.episode_ids.length,
            },
          },
        },
        reason: item.review.reason,
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
          edgeIds: item.candidate_support_edges.map((edge) => edge.id),
          reviewItemId: reviewItem.id,
        } satisfies ReflectorReversal,
      });
      changes.push(buildChange(item));
    }

    return {
      process: this.name,
      dryRun: false,
      changes,
      tokens_used: plan.tokens_used,
      errors: plan.errors,
      budget_exhausted: plan.budget_exhausted,
      candidate_stats: {
        proposed: plan.items.length + plan.errors.length,
        accepted: changes.length,
        rejected: plan.errors.length,
      },
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
