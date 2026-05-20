import { z } from "zod";

import { episodeIdSchema, type Episode } from "../../memory/episodic/index.js";
import {
  SemanticExtractor,
  semanticEdgeIdSchema,
  semanticEdgeSchema,
  semanticNodeIdSchema,
  semanticNodeSchema,
  type ReviewQueueInsertInput,
  type SemanticEdge,
  type SemanticNode,
} from "../../memory/semantic/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
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

const semanticExtractorPlanSchema = z.object({
  process: z.literal("semantic-extractor"),
  episode_ids: z.array(episodeIdSchema),
  budget: z.number().int().positive(),
  errors: z
    .array(
      z.object({
        process: z.literal("semantic-extractor"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
});

export const semanticExtractorProcessPlanSchema = semanticExtractorPlanSchema;
export type SemanticExtractorProcessPlan = z.infer<typeof semanticExtractorPlanSchema>;

const serializableSemanticNodeSchema = semanticNodeSchema.extend({
  embedding: z.array(z.number().finite()),
});

const semanticExtractorReversalSchema = z.object({
  created_node_ids: z.array(semanticNodeIdSchema),
  updated_nodes: z.array(serializableSemanticNodeSchema),
  created_edge_ids: z.array(semanticEdgeIdSchema),
  updated_edges: z.array(semanticEdgeSchema).default([]),
});

type SemanticExtractorReversal = z.infer<typeof semanticExtractorReversalSchema>;

export type SemanticExtractorProcessOptions = {
  semanticNodeRepository: OfflineContext["semanticNodeRepository"];
  semanticEdgeRepository: OfflineContext["semanticEdgeRepository"];
  registry: ReverserRegistry;
  clock?: Clock;
};

function serializeSemanticNode(node: SemanticNode): z.infer<typeof serializableSemanticNodeSchema> {
  return serializableSemanticNodeSchema.parse({
    ...node,
    embedding: Array.from(node.embedding),
  });
}

function deserializeSemanticNode(
  node: z.infer<typeof serializableSemanticNodeSchema>,
): SemanticNode {
  return semanticNodeSchema.parse({
    ...node,
    embedding: Float32Array.from(node.embedding),
  });
}

function semanticNodeSnapshot(node: SemanticNode): string {
  return JSON.stringify(serializeSemanticNode(node));
}

function buildChange(input: {
  episodeIds: readonly Episode["id"][];
  insertedNodes: number;
  updatedNodes: number;
  skippedNodes: number;
  insertedEdges: number;
  updatedEdges: number;
  skippedEdges: number;
}): OfflineChange {
  return {
    process: "semantic-extractor",
    action: "extract",
    targets: {
      episode_ids: input.episodeIds,
    },
    preview: {
      inserted_nodes: input.insertedNodes,
      updated_nodes: input.updatedNodes,
      skipped_nodes: input.skippedNodes,
      inserted_edges: input.insertedEdges,
      updated_edges: input.updatedEdges,
      skipped_edges: input.skippedEdges,
    },
  };
}

function semanticEdgeSnapshot(edge: SemanticEdge): string {
  return JSON.stringify(semanticEdgeSchema.parse(edge));
}

async function snapshotSemanticNodes(
  ctx: OfflineContext,
): Promise<Map<SemanticNode["id"], SemanticNode>> {
  const nodes = await ctx.semanticNodeRepository.list({
    includeArchived: true,
    limit: 100_000,
  });

  return new Map(nodes.map((node) => [node.id, node]));
}

function snapshotSemanticEdges(ctx: OfflineContext): Map<SemanticEdge["id"], SemanticEdge> {
  const edges = ctx.semanticEdgeRepository.listEdges({
    includeInvalid: true,
  });

  return new Map(edges.map((edge) => [edge.id, edge]));
}

async function restoreSemanticSnapshot(input: {
  ctx: OfflineContext;
  beforeNodes: ReadonlyMap<SemanticNode["id"], SemanticNode>;
  beforeEdges: ReadonlyMap<SemanticEdge["id"], SemanticEdge>;
}): Promise<void> {
  const afterEdges = snapshotSemanticEdges(input.ctx);

  for (const [edgeId, edge] of afterEdges) {
    const before = input.beforeEdges.get(edgeId);

    if (before === undefined) {
      input.ctx.semanticEdgeRepository.delete(edgeId);
      continue;
    }

    if (semanticEdgeSnapshot(before) !== semanticEdgeSnapshot(edge)) {
      input.ctx.semanticEdgeRepository.restoreEdge(before);
    }
  }

  const afterNodes = await snapshotSemanticNodes(input.ctx);

  for (const [nodeId, node] of afterNodes) {
    const before = input.beforeNodes.get(nodeId);

    if (before === undefined) {
      await input.ctx.semanticNodeRepository.delete(nodeId);
      continue;
    }

    if (semanticNodeSnapshot(before) !== semanticNodeSnapshot(node)) {
      await input.ctx.semanticNodeRepository.restore(before);
    }
  }
}

function emitSemanticInsertSkipped(
  ctx: OfflineContext,
  input: { kind: "episode"; reason: "episode_archived_post_plan" },
): void {
  if (ctx.tracer?.enabled !== true) {
    return;
  }

  ctx.tracer.emit("semantic_insert.skipped", {
    turnId: ctx.runId,
    kind: input.kind,
    reason: input.reason,
  });
}

async function representedEpisodeIds(ctx: OfflineContext): Promise<Set<Episode["id"]>> {
  const represented = new Set<Episode["id"]>();
  const nodes = await ctx.semanticNodeRepository.list({
    includeArchived: true,
    limit: 100_000,
  });

  for (const node of nodes) {
    for (const episodeId of node.source_episode_ids) {
      represented.add(episodeId);
    }
  }

  for (const edge of ctx.semanticEdgeRepository.listEdges({ includeInvalid: true })) {
    for (const episodeId of edge.evidence_episode_ids) {
      represented.add(episodeId);
    }
  }

  return represented;
}

function auditedExtractionEpisodeIds(ctx: OfflineContext): Set<Episode["id"]> {
  const audited = new Set<Episode["id"]>();

  for (const audit of ctx.auditLog.list({ process: "semantic-extractor", reverted: false })) {
    const episodeIds = audit.targets.episode_ids;

    if (!Array.isArray(episodeIds)) {
      continue;
    }

    for (const episodeId of episodeIds) {
      const parsed = episodeIdSchema.safeParse(episodeId);

      if (parsed.success) {
        audited.add(parsed.data);
      }
    }
  }

  return audited;
}

async function processedEpisodeIds(ctx: OfflineContext): Promise<Set<Episode["id"]>> {
  return new Set([...(await representedEpisodeIds(ctx)), ...auditedExtractionEpisodeIds(ctx)]);
}

async function selectEpisodesForExtraction(ctx: OfflineContext): Promise<Episode[]> {
  const maxEpisodesPerRun = ctx.config.offline.semanticExtractor.maxEpisodesPerRun;
  const processed = await processedEpisodeIds(ctx);
  const episodes = await ctx.episodicRepository.listAll();
  const selected: Episode[] = [];

  for (const episode of episodes) {
    if (ctx.episodicRepository.getStats(episode.id)?.archived === true) {
      continue;
    }

    if (processed.has(episode.id)) {
      continue;
    }

    selected.push(episode);

    if (selected.length >= maxEpisodesPerRun) {
      break;
    }
  }

  return selected;
}

function resultCandidateStats(input: {
  insertedNodes: number;
  updatedNodes: number;
  skippedNodes: number;
  insertedEdges: number;
  updatedEdges: number;
  skippedEdges: number;
}): NonNullable<OfflineResult["candidate_stats"]> {
  const accepted =
    input.insertedNodes + input.updatedNodes + input.insertedEdges + input.updatedEdges;
  const rejected = input.skippedNodes + input.skippedEdges;

  return {
    proposed: accepted + rejected,
    accepted,
    rejected,
  };
}

/*
 * SemanticExtractor is a deliberate direct-write maintenance process. Unlike
 * the reflector, which proposes speculative cross-cluster insight and gates it
 * through review, this process mines grounded concept structure from already
 * accepted episodes. The extracted graph writes are therefore accepted by
 * definition, matching borg.semantic.extract() facade behavior. If semantic
 * extraction should become review-routed later, make that an explicit config
 * option rather than blending the reflector review path into this process.
 */
export class SemanticExtractorProcess implements OfflineProcess<SemanticExtractorProcessPlan> {
  readonly name = "semantic-extractor" as const;
  private readonly clock: Clock;

  constructor(private readonly options: SemanticExtractorProcessOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.options.registry.register(this.name, "extract", async ({ reversal }) => {
      const parsed = semanticExtractorReversalSchema.parse(reversal);

      for (const edgeId of parsed.created_edge_ids) {
        const edge = this.options.semanticEdgeRepository.getEdge(edgeId);
        this.options.semanticEdgeRepository.invalidateEdge(edgeId, {
          at: Math.max(this.clock.now(), edge?.valid_from ?? this.clock.now()),
          by_process: "maintenance",
          reason: "semantic_extractor_audit_reversal",
        });
      }

      for (const previousEdge of parsed.updated_edges) {
        this.options.semanticEdgeRepository.restoreEdge(previousEdge);
      }

      for (const nodeId of parsed.created_node_ids) {
        await this.options.semanticNodeRepository.update(nodeId, {
          archived: true,
        });
      }

      for (const previousNode of parsed.updated_nodes) {
        await this.options.semanticNodeRepository.restore(deserializeSemanticNode(previousNode));
      }
    });
  }

  async plan(
    ctx: OfflineContext,
    opts: { budget?: number } = {},
  ): Promise<SemanticExtractorProcessPlan> {
    const errors: OfflineProcessError[] = [];
    let episodeIds: Episode["id"][] = [];
    const budget = opts.budget ?? ctx.config.offline.semanticExtractor.budget;

    if (!ctx.config.offline.semanticExtractor.enabled) {
      return semanticExtractorPlanSchema.parse({
        process: this.name,
        episode_ids: [],
        budget,
        errors,
        tokens_used: 0,
        budget_exhausted: false,
      });
    }

    try {
      episodeIds = (await selectEpisodesForExtraction(ctx)).map((episode) => episode.id);
    } catch (error) {
      errors.push(offlineProcessError(this.name, error));
    }

    return semanticExtractorPlanSchema.parse({
      process: this.name,
      episode_ids: episodeIds,
      budget,
      errors,
      tokens_used: 0,
      budget_exhausted: false,
    });
  }

  preview(plan: SemanticExtractorProcessPlan): OfflineResult {
    const parsed = semanticExtractorPlanSchema.parse(plan);

    return {
      process: this.name,
      dryRun: true,
      changes:
        parsed.episode_ids.length === 0
          ? []
          : [
              {
                process: this.name,
                action: "extract",
                targets: {
                  episode_ids: parsed.episode_ids,
                },
                preview: {
                  planned_episode_count: parsed.episode_ids.length,
                },
              },
            ],
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
      candidate_stats: {
        proposed: parsed.episode_ids.length,
        accepted: 0,
        rejected: 0,
      },
    };
  }

  async apply(ctx: OfflineContext, rawPlan: SemanticExtractorProcessPlan): Promise<OfflineResult> {
    const plan = semanticExtractorPlanSchema.parse(rawPlan);
    const errors: OfflineProcessError[] = [...plan.errors];

    if (plan.episode_ids.length === 0 || errors.length > 0) {
      return {
        process: this.name,
        dryRun: false,
        changes: [],
        tokens_used: plan.tokens_used,
        errors,
        budget_exhausted: plan.budget_exhausted,
        candidate_stats: {
          proposed: 0,
          accepted: 0,
          rejected: errors.length,
        },
      };
    }

    const plannedEpisodes = (await ctx.episodicRepository.getMany(plan.episode_ids)).filter(
      (episode): episode is Episode => episode !== null,
    );

    if (plannedEpisodes.length !== plan.episode_ids.length) {
      throw new StorageError("Semantic extractor plan references missing episodes", {
        code: "SEMANTIC_EXTRACTOR_PLAN_INVALID",
      });
    }

    const archivedPostPlanIds: Episode["id"][] = [];
    const episodes: Episode[] = [];

    for (const episode of plannedEpisodes) {
      if (ctx.episodicRepository.getStats(episode.id)?.archived === true) {
        archivedPostPlanIds.push(episode.id);
        emitSemanticInsertSkipped(ctx, {
          kind: "episode",
          reason: "episode_archived_post_plan",
        });
        continue;
      }

      episodes.push(episode);
    }

    if (episodes.length === 0) {
      return {
        process: this.name,
        dryRun: false,
        changes: [],
        tokens_used: plan.tokens_used,
        errors,
        budget_exhausted: plan.budget_exhausted,
        candidate_stats: {
          proposed: archivedPostPlanIds.length,
          accepted: 0,
          rejected: archivedPostPlanIds.length,
        },
      };
    }

    const beforeNodes = await snapshotSemanticNodes(ctx);
    const beforeEdges = snapshotSemanticEdges(ctx);
    const deferredReviews: ReviewQueueInsertInput[] = [];
    let tokensUsed = plan.tokens_used;
    let budgetExhausted = plan.budget_exhausted;

    try {
      const budgeted = await withBudget(this.name, plan.budget, async ({ wrapClient }) => {
        const extractor = new SemanticExtractor({
          nodeRepository: ctx.semanticNodeRepository,
          edgeRepository: ctx.semanticEdgeRepository,
          embeddingClient: ctx.embeddingClient,
          episodicRepository: ctx.episodicRepository,
          llmClient: wrapClient(ctx.llm.extraction),
          model: ctx.config.anthropic.models.extraction,
          reviewEnqueue: (input) => {
            deferredReviews.push({
              ...input,
              sourceProcess: input.sourceProcess ?? this.name,
              traceTurnId: input.traceTurnId ?? ctx.runId,
            });
          },
          clock: ctx.clock,
          tracer: ctx.tracer,
          traceTurnId: ctx.runId,
        });

        return extractor.extractFromEpisodes(episodes);
      });
      const extraction = budgeted.result;
      tokensUsed = budgeted.tokens_used;
      const afterNodes = await snapshotSemanticNodes(ctx);
      const afterEdges = snapshotSemanticEdges(ctx);
      const createdNodeIds: SemanticNode["id"][] = [];
      const updatedNodes: Array<z.infer<typeof serializableSemanticNodeSchema>> = [];
      const createdEdgeIds: SemanticEdge["id"][] = [];
      const updatedEdges: SemanticEdge[] = [];

      for (const [nodeId, node] of afterNodes) {
        const before = beforeNodes.get(nodeId);

        if (before === undefined) {
          createdNodeIds.push(nodeId);
          continue;
        }

        if (semanticNodeSnapshot(before) !== semanticNodeSnapshot(node)) {
          updatedNodes.push(serializeSemanticNode(before));
        }
      }

      for (const [edgeId, edge] of afterEdges) {
        const before = beforeEdges.get(edgeId);

        if (before === undefined) {
          createdEdgeIds.push(edgeId);
          continue;
        }

        if (semanticEdgeSnapshot(before) !== semanticEdgeSnapshot(edge)) {
          updatedEdges.push(before);
        }
      }

      ctx.auditLog.record({
        run_id: ctx.runId,
        process: this.name,
        action: "extract",
        targets: {
          episode_ids: episodes.map((episode) => episode.id),
          created_node_ids: createdNodeIds,
          updated_node_count: updatedNodes.length,
          created_edge_ids: createdEdgeIds,
          updated_edge_count: updatedEdges.length,
        },
        reversal: {
          created_node_ids: createdNodeIds,
          updated_nodes: updatedNodes,
          created_edge_ids: createdEdgeIds,
          updated_edges: updatedEdges,
        } satisfies SemanticExtractorReversal,
      });

      for (const nodeId of createdNodeIds) {
        const node = afterNodes.get(nodeId);

        if (node !== undefined) {
          ctx.semanticReviewService?.queueDuplicateReview(node, {
            sourceProcess: this.name,
            traceTurnId: ctx.runId,
          });
        }
      }

      for (const review of deferredReviews) {
        try {
          ctx.reviewQueueRepository.enqueue(review);
        } catch (error) {
          errors.push(offlineProcessError(this.name, error));
        }
      }

      const change = buildChange({
        episodeIds: episodes.map((episode) => episode.id),
        insertedNodes: extraction.insertedNodes,
        updatedNodes: extraction.updatedNodes,
        skippedNodes: extraction.skippedNodes,
        insertedEdges: extraction.insertedEdges,
        updatedEdges: extraction.updatedEdges,
        skippedEdges: extraction.skippedEdges,
      });
      const candidateStats = resultCandidateStats({
        insertedNodes: extraction.insertedNodes,
        updatedNodes: extraction.updatedNodes,
        skippedNodes: extraction.skippedNodes,
        insertedEdges: extraction.insertedEdges,
        updatedEdges: extraction.updatedEdges,
        skippedEdges: extraction.skippedEdges,
      });

      return {
        process: this.name,
        dryRun: false,
        changes: [change],
        tokens_used: tokensUsed,
        errors,
        budget_exhausted: budgetExhausted,
        candidate_stats: {
          proposed: candidateStats.proposed + archivedPostPlanIds.length,
          accepted: candidateStats.accepted,
          rejected: candidateStats.rejected + archivedPostPlanIds.length + errors.length,
        },
      };
    } catch (error) {
      await restoreSemanticSnapshot({
        ctx,
        beforeNodes,
        beforeEdges,
      });
      tokensUsed = getBudgetErrorTokens(error);
      budgetExhausted = error instanceof BudgetExceededError;
      errors.push(offlineProcessError(this.name, error));

      return {
        process: this.name,
        dryRun: false,
        changes: [],
        tokens_used: tokensUsed,
        errors,
        budget_exhausted: budgetExhausted,
        candidate_stats: {
          proposed: plan.episode_ids.length,
          accepted: 0,
          rejected: errors.length,
        },
      };
    }
  }

  async run(
    ctx: OfflineContext,
    opts: { dryRun?: boolean; budget?: number; params?: Record<string, unknown> } = {},
  ): Promise<OfflineResult> {
    const plan = await this.plan(ctx, opts);
    return opts.dryRun === true ? this.preview(plan) : this.apply(ctx, plan);
  }
}
