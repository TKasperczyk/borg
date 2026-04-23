import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import {
  episodeAccessScopeKey,
  episodeIdSchema,
  type Episode,
} from "../../memory/episodic/index.js";
import { semanticNodeIdSchema, type SemanticNode } from "../../memory/semantic/index.js";
import { semanticRelationSchema } from "../../memory/semantic/types.js";
import { createSemanticEdgeId, createSemanticNodeId } from "../../util/ids.js";
import { BudgetExceededError, SemanticError } from "../../util/errors.js";
import { tokenizeText } from "../../util/text/tokenize.js";

import type { ReverserRegistry } from "../audit-log.js";
import { getBudgetErrorTokens, withBudget } from "../budget.js";
import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessError,
  OfflineResult,
} from "../types.js";

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
  kind: z.enum(["concept", "entity", "proposition"]),
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
});

const serializableSemanticEdgeSchema = z.object({
  id: z.string().min(1),
  from_node_id: semanticNodeIdSchema,
  to_node_id: semanticNodeIdSchema,
  relation: semanticRelationSchema,
  confidence: z.number().min(0).max(1),
  evidence_episode_ids: z.array(episodeIdSchema).min(1),
  created_at: z.number().finite(),
  last_verified_at: z.number().finite(),
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

const reflectorPlanItemSchema = z.object({
  cluster_key: z.string().min(1),
  episode_ids: z.array(episodeIdSchema).min(1),
  target: reflectorTargetSchema,
  anchor_node: serializableSemanticNodeSchema,
  edge: serializableSemanticEdgeSchema,
  review: z.object({
    kind: z.enum(["duplicate", "new_insight"]),
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

type ReflectorReversal = {
  nodeId: string;
  nodeCreated: boolean;
  previousNode?: z.infer<typeof serializableSemanticNodeSchema>;
  anchorNodeId?: string;
  edgeIds: string[];
  reviewItemId?: number;
};

function serializeSemanticNode(node: SemanticNode) {
  return serializableSemanticNodeSchema.parse({
    ...node,
    embedding: Array.from(node.embedding),
  });
}

function deserializeSemanticNode(
  node: z.infer<typeof serializableSemanticNodeSchema>,
): SemanticNode {
  return {
    ...node,
    embedding: Float32Array.from(node.embedding),
  };
}

function buildPrompt(cluster: ReflectionCluster, goalDescriptions: readonly string[]): string {
  return [
    "Infer one modest semantic proposition from the supporting episodes.",
    `Emit your result by calling the ${REFLECTOR_TOOL_NAME} tool exactly once.`,
    "Use only source_episode_ids from the provided episodes.",
    "Keep confidence conservative.",
    `Cluster key: ${cluster.key}`,
    `Active goals: ${goalDescriptions.join(" | ") || "none"}`,
    "Episodes:",
    ...cluster.episodes.map((episode) =>
      JSON.stringify({
        id: episode.id,
        title: episode.title,
        narrative: episode.narrative,
        tags: episode.tags,
        participants: episode.participants,
      }),
    ),
  ].join("\n");
}

function parseInsight(result: LLMCompleteResult) {
  const call = result.tool_calls.find((toolCall) => toolCall.name === REFLECTOR_TOOL_NAME);

  if (call === undefined) {
    throw new SemanticError(`Reflector did not emit tool ${REFLECTOR_TOOL_NAME}`, {
      code: "REFLECTOR_INVALID",
    });
  }

  return insightResponseSchema.parse(call.input);
}

function collectReflectionClusters(
  episodes: readonly Episode[],
  goalDescriptions: readonly string[],
  minSupport: number,
  maxInsightsPerRun: number,
): ReflectionCluster[] {
  const byKey = new Map<string, Episode[]>();

  for (const episode of episodes) {
    for (const tag of episode.tags) {
      const key = `${episodeAccessScopeKey(episode)}|tag:${tag.toLowerCase()}`;
      byKey.set(key, [...(byKey.get(key) ?? []), episode]);
    }

    const episodeTokens = tokenizeText(
      `${episode.title} ${episode.narrative} ${episode.tags.join(" ")}`,
    );

    for (const description of goalDescriptions) {
      const goalTokens = tokenizeText(description);

      if (![...goalTokens].some((token) => episodeTokens.has(token))) {
        continue;
      }

      const key = `${episodeAccessScopeKey(episode)}|goal:${description.toLowerCase()}`;
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

async function semanticNodeMatchesClusterScope(
  ctx: OfflineContext,
  node: SemanticNode,
  cluster: ReflectionCluster,
): Promise<boolean> {
  const scopeKey = episodeAccessScopeKey(cluster.episodes[0] ?? {});
  const sourceEpisodes = await ctx.episodicRepository.getMany(node.source_episode_ids);

  return (
    sourceEpisodes.length === node.source_episode_ids.length &&
    sourceEpisodes.every((episode) => episodeAccessScopeKey(episode) === scopeKey)
  );
}

async function buildInsightCandidate(
  ctx: OfflineContext,
  llmClient: LLMClient,
  cluster: ReflectionCluster,
): Promise<{
  label: string;
  description: string;
  confidence: number;
  sourceEpisodeIds: Episode["id"][];
  embedding: Float32Array;
}> {
  const insight = parseInsight(
    await llmClient.complete({
      model: ctx.config.anthropic.models.background,
      system:
        "You propose low-confidence semantic propositions grounded in repeated episodic evidence.",
      messages: [
        {
          role: "user",
          content: buildPrompt(
            cluster,
            ctx.goalsRepository.list({ status: "active" }).map((goal) => goal.description),
          ),
        },
      ],
      tools: [REFLECTOR_TOOL],
      tool_choice: { type: "tool", name: REFLECTOR_TOOL_NAME },
      max_tokens: 4_000,
      budget: "offline-reflector",
    }),
  );
  const allowedIds = new Set(cluster.episodes.map((episode) => episode.id));

  if (
    !insight.source_episode_ids.every((episodeId) => allowedIds.has(episodeId as Episode["id"]))
  ) {
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
    sourceEpisodeIds: insight.source_episode_ids as Episode["id"][],
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

export type ReflectorProcessOptions = {
  semanticNodeRepository: OfflineContext["semanticNodeRepository"];
  semanticEdgeRepository: OfflineContext["semanticEdgeRepository"];
  reviewQueueRepository: OfflineContext["reviewQueueRepository"];
  registry: ReverserRegistry;
};

export class ReflectorProcess implements OfflineProcess {
  readonly name = "reflector" as const;

  constructor(private readonly options: ReflectorProcessOptions) {
    this.options.registry.register(this.name, "insight", async ({ reversal }) => {
      const parsed = reversal as Partial<ReflectorReversal>;

      if (typeof parsed.nodeId === "string") {
        if (parsed.nodeCreated) {
          await this.options.semanticNodeRepository.delete(parsed.nodeId as SemanticNode["id"]);
        } else if (parsed.previousNode !== undefined) {
          const previousNode = deserializeSemanticNode(parsed.previousNode);
          await this.options.semanticNodeRepository.update(previousNode.id, {
            label: previousNode.label,
            description: previousNode.description,
            aliases: previousNode.aliases,
            confidence: previousNode.confidence,
            source_episode_ids: previousNode.source_episode_ids,
            last_verified_at: previousNode.last_verified_at,
            embedding: previousNode.embedding,
            archived: previousNode.archived,
            superseded_by: previousNode.superseded_by,
          });
        }
      }

      for (const edgeId of parsed.edgeIds ?? []) {
        if (typeof edgeId === "string") {
          this.options.semanticEdgeRepository.delete(edgeId as never);
        }
      }

      if (typeof parsed.anchorNodeId === "string") {
        await this.options.semanticNodeRepository.delete(parsed.anchorNodeId as SemanticNode["id"]);
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
    const episodes = (await ctx.episodicRepository.listAll()).filter(
      (episode) => !(ctx.episodicRepository.getStats(episode.id)?.archived ?? false),
    );
    const goalDescriptions = ctx.goalsRepository
      .list({ status: "active" })
      .map((goal) => goal.description);
    const clusters = collectReflectionClusters(
      episodes,
      goalDescriptions,
      ctx.config.offline.reflector.minSupport,
      ctx.config.offline.reflector.maxInsightsPerRun,
    );
    let tokensUsed = 0;
    let budgetExhausted = false;

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient = wrapClient(ctx.llm.background);

        for (const cluster of clusters) {
          try {
            const candidate = await buildInsightCandidate(ctx, llmClient, cluster);
            const byLabel = await ctx.semanticNodeRepository.findByLabelOrAlias(candidate.label, 3, {
              includeArchived: true,
            });
            const byVector = await ctx.semanticNodeRepository.searchByVector(candidate.embedding, {
              limit: 3,
              minSimilarity: DEDUP_THRESHOLD,
              kindFilter: ["proposition"],
              includeArchived: false,
            });
            const eligibleByLabel: SemanticNode[] = [];

            for (const node of byLabel) {
              if (await semanticNodeMatchesClusterScope(ctx, node, cluster)) {
                eligibleByLabel.push(node);
              }
            }

            const eligibleByVector: SemanticNode[] = [];

            for (const item of byVector) {
              if (await semanticNodeMatchesClusterScope(ctx, item.node, cluster)) {
                eligibleByVector.push(item.node);
              }
            }

            const existing = eligibleByLabel[0] ?? eligibleByVector[0];
            const timestamp = ctx.clock.now();
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
                      source_episode_ids: candidate.sourceEpisodeIds,
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
                      source_episode_ids: candidate.sourceEpisodeIds,
                      last_verified_at: timestamp,
                      embedding: Array.from(candidate.embedding),
                      archived: false,
                    },
                  };
            const targetNodeId = target.mode === "insert" ? target.node.id : target.node_id;
            const anchorLabel = `Evidence cluster ${cluster.key}`;
            const anchorNode = serializableSemanticNodeSchema.parse({
              id: createSemanticNodeId(),
              kind: "proposition",
              label: anchorLabel,
              description: `Supporting evidence from ${cluster.episodes.length} episodes for ${candidate.label}.`,
              aliases: [],
              confidence: 0.4,
              source_episode_ids: cluster.episodes.map((episode) => episode.id),
              created_at: timestamp,
              updated_at: timestamp,
              last_verified_at: timestamp,
              embedding: Array.from(
                await ctx.embeddingClient.embed(
                  `${anchorLabel}\n${cluster.episodes.map((episode) => episode.title).join("\n")}`,
                ),
              ),
              archived: false,
              superseded_by: null,
            });

            items.push({
              cluster_key: cluster.key,
              episode_ids: cluster.episodes.map((episode) => episode.id),
              target,
              anchor_node: anchorNode,
              edge: serializableSemanticEdgeSchema.parse({
                id: createSemanticEdgeId(),
                from_node_id: anchorNode.id,
                to_node_id: targetNodeId,
                relation: "supports",
                confidence: 0.6,
                evidence_episode_ids: cluster.episodes.map((episode) => episode.id),
                created_at: timestamp,
                last_verified_at: timestamp,
              }),
              review: {
                kind: existing === undefined ? "new_insight" : "duplicate",
                reason:
                  existing === undefined
                    ? `New low-confidence insight extracted from ${cluster.key}`
                    : `Existing insight revisited from ${cluster.key}`,
              },
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

      tokensUsed = budgeted.tokens_used;
    } catch (error) {
      tokensUsed = getBudgetErrorTokens(error);
      budgetExhausted = error instanceof BudgetExceededError;
      errors.push({
        process: this.name,
        message: error instanceof Error ? error.message : String(error),
        code: error instanceof Error && "code" in error ? String(error.code) : undefined,
      });
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
        const inserted = await ctx.semanticNodeRepository.insert(
          deserializeSemanticNode(item.target.node),
        );
        nodeId = inserted.id;
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
        const updated = await ctx.semanticNodeRepository.update(item.target.node_id, {
          description: item.target.patch.description,
          confidence: item.target.patch.confidence,
          source_episode_ids: item.target.patch.source_episode_ids,
          last_verified_at: item.target.patch.last_verified_at,
          embedding: Float32Array.from(item.target.patch.embedding),
          archived: item.target.patch.archived,
        });

        if (updated === null) {
          throw new SemanticError(`Failed to update semantic node ${item.target.node_id}`, {
            code: "REFLECTOR_PLAN_INVALID",
          });
        }

        nodeId = updated.id;
      }

      const anchorNode = await ctx.semanticNodeRepository.insert(
        deserializeSemanticNode(item.anchor_node),
      );
      const edge = ctx.semanticEdgeRepository.addEdge({
        id: item.edge.id as never,
        from_node_id: anchorNode.id,
        to_node_id: nodeId,
        relation: item.edge.relation,
        confidence: item.edge.confidence,
        evidence_episode_ids: item.edge.evidence_episode_ids,
        created_at: item.edge.created_at,
        last_verified_at: item.edge.last_verified_at,
      });
      const reviewItem = ctx.reviewQueueRepository.enqueue({
        kind: item.review.kind,
        refs: {
          node_ids: [nodeId],
          episode_ids: item.episode_ids,
        },
        reason: item.review.reason,
      });

      ctx.auditLog.record({
        run_id: ctx.runId,
        process: this.name,
        action: "insight",
        targets: {
          nodeId,
          edgeIds: [edge.id],
        },
        reversal: {
          nodeId,
          nodeCreated,
          ...(previousNode === undefined ? {} : { previousNode }),
          anchorNodeId: anchorNode.id,
          edgeIds: [edge.id],
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
