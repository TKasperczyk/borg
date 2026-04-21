import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { emotionalArcSchema } from "../../memory/affective/index.js";
import {
  episodeIdSchema,
  episodeLineageSchema,
  episodeStatsSchema,
  episodeTierSchema,
  type Episode,
  type EpisodeStats,
  type EpisodeTier,
} from "../../memory/episodic/index.js";
import { streamEntryIdSchema } from "../../memory/episodic/types.js";
import { createEpisodeId } from "../../util/ids.js";
import { BudgetExceededError, StorageError } from "../../util/errors.js";

import type { ReverserRegistry } from "../audit-log.js";
import { getBudgetErrorTokens, withBudget } from "../budget.js";
import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessError,
  OfflineResult,
} from "../types.js";

const mergeResponseSchema = z.object({
  title: z.string().min(1),
  narrative: z.string().min(1),
});
const MERGE_TOOL_NAME = "EmitConsolidation";
export const MERGE_TOOL = {
  name: MERGE_TOOL_NAME,
  description: "Emit the merged episode title and narrative for a redundant cluster.",
  inputSchema: toToolInputSchema(mergeResponseSchema),
} satisfies LLMToolDefinition;

const serializableEpisodeSchema = z.object({
  id: episodeIdSchema,
  title: z.string().min(1),
  narrative: z.string().min(1),
  participants: z.array(z.string().min(1)),
  location: z.string().min(1).nullable(),
  start_time: z.number().finite(),
  end_time: z.number().finite(),
  source_stream_ids: z.array(streamEntryIdSchema).min(1),
  significance: z.number().min(0).max(1),
  tags: z.array(z.string().min(1)),
  confidence: z.number().min(0).max(1),
  lineage: episodeLineageSchema,
  emotional_arc: emotionalArcSchema.nullable(),
  embedding: z.array(z.number().finite()),
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
});

const consolidatorPlanItemSchema = z.object({
  source_episode_ids: z.array(episodeIdSchema).min(2),
  merged_episode: serializableEpisodeSchema,
  inherited_tier: episodeTierSchema,
});

export const consolidatorPlanSchema = z.object({
  process: z.literal("consolidator"),
  items: z.array(consolidatorPlanItemSchema),
  errors: z
    .array(
      z.object({
        process: z.literal("consolidator"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
});

export type ConsolidatorPlan = z.infer<typeof consolidatorPlanSchema>;

const HOUR_MS = 60 * 60 * 1_000;
const TIER_ORDER: Record<EpisodeTier, number> = {
  T1: 1,
  T2: 2,
  T3: 3,
  T4: 4,
};

type EpisodeCluster = {
  episodes: Episode[];
  stats: EpisodeStats[];
};

type ConsolidationReversal = {
  newEpisodeId: string;
  sourceEpisodes: Array<{
    id: string;
    lineage: Episode["lineage"];
  }>;
  sourceStats: EpisodeStats[];
};

function cosineSimilarity(left: Float32Array, right: Float32Array): number {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  const size = Math.min(left.length, right.length);

  for (let index = 0; index < size; index += 1) {
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

function shareTagFamily(left: Episode, right: Episode): boolean {
  if (left.tags.length === 0 || right.tags.length === 0) {
    return false;
  }

  const rightTags = new Set(right.tags.map((tag) => tag.toLowerCase()));
  return left.tags.some((tag) => rightTags.has(tag.toLowerCase()));
}

function compareTier(left: EpisodeTier, right: EpisodeTier): number {
  return TIER_ORDER[left] - TIER_ORDER[right];
}

function maxTier(stats: readonly EpisodeStats[]): EpisodeTier {
  return stats.reduce<EpisodeTier>(
    (best, current) => (compareTier(current.tier, best) > 0 ? current.tier : best),
    "T1",
  );
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))];
}

function parseMergeResponse(result: LLMCompleteResult) {
  const call = result.tool_calls.find((toolCall) => toolCall.name === MERGE_TOOL_NAME);

  if (call === undefined) {
    throw new StorageError(`Consolidator did not emit tool ${MERGE_TOOL_NAME}`, {
      code: "CONSOLIDATOR_INVALID",
    });
  }

  return mergeResponseSchema.parse(call.input);
}

function buildMergePrompt(cluster: EpisodeCluster): string {
  return [
    "Merge the redundant episodes into one grounded episode.",
    `Emit your result by calling the ${MERGE_TOOL_NAME} tool exactly once.`,
    "Preserve facts from all inputs. Keep the narrative to 2-5 sentences.",
    "Episodes:",
    ...cluster.episodes.map((episode) =>
      JSON.stringify({
        id: episode.id,
        title: episode.title,
        narrative: episode.narrative,
        participants: episode.participants,
        location: episode.location,
        start_time: episode.start_time,
        end_time: episode.end_time,
        tags: episode.tags,
        source_stream_ids: episode.source_stream_ids,
      }),
    ),
  ].join("\n");
}

function buildClusters(
  episodes: readonly Episode[],
  statsById: ReadonlyMap<string, EpisodeStats>,
  similarityThreshold: number,
  minClusterSize: number,
  maxClusters: number,
): EpisodeCluster[] {
  const adjacency = new Map<Episode["id"], Set<Episode["id"]>>();

  for (const episode of episodes) {
    adjacency.set(episode.id, new Set());
  }

  for (let leftIndex = 0; leftIndex < episodes.length; leftIndex += 1) {
    const left = episodes[leftIndex];

    if (left === undefined) {
      continue;
    }

    for (let rightIndex = leftIndex + 1; rightIndex < episodes.length; rightIndex += 1) {
      const right = episodes[rightIndex];

      if (right === undefined) {
        continue;
      }

      if (!shareTagFamily(left, right)) {
        continue;
      }

      const similarity = cosineSimilarity(left.embedding, right.embedding);

      if (similarity < similarityThreshold) {
        continue;
      }

      adjacency.get(left.id)?.add(right.id);
      adjacency.get(right.id)?.add(left.id);
    }
  }

  const byId = new Map(episodes.map((episode) => [episode.id, episode]));
  const visited = new Set<string>();
  const clusters: EpisodeCluster[] = [];

  for (const episode of episodes) {
    if (visited.has(episode.id)) {
      continue;
    }

    const queue = [episode.id];
    const component: Episode[] = [];

    while (queue.length > 0) {
      const nextId = queue.shift();

      if (nextId === undefined || visited.has(nextId)) {
        continue;
      }

      visited.add(nextId);
      const next = byId.get(nextId);

      if (next !== undefined) {
        component.push(next);
      }

      for (const neighbor of adjacency.get(nextId) ?? []) {
        if (!visited.has(neighbor)) {
          queue.push(neighbor);
        }
      }
    }

    if (component.length < minClusterSize) {
      continue;
    }

    const stats = component
      .map((item) => statsById.get(item.id))
      .filter((item): item is EpisodeStats => item !== undefined);

    if (stats.length !== component.length) {
      continue;
    }

    clusters.push({
      episodes: component.sort((left, right) => left.created_at - right.created_at),
      stats,
    });
  }

  return clusters
    .sort(
      (left, right) =>
        right.episodes.length - left.episodes.length ||
        (right.episodes[0]?.updated_at ?? 0) - (left.episodes[0]?.updated_at ?? 0),
    )
    .slice(0, maxClusters);
}

async function buildMergedEpisode(
  ctx: OfflineContext,
  llmClient: LLMClient,
  cluster: EpisodeCluster,
): Promise<{ episode: Episode; inheritedTier: EpisodeTier }> {
  const merged = parseMergeResponse(
    await llmClient.complete({
      model: ctx.config.anthropic.models.background,
      system:
        "You merge overlapping autobiographical episodes. Keep only grounded facts from the inputs.",
      messages: [
        {
          role: "user",
          content: buildMergePrompt(cluster),
        },
      ],
      tools: [MERGE_TOOL],
      tool_choice: { type: "tool", name: MERGE_TOOL_NAME },
      max_tokens: 700,
      budget: "offline-consolidator",
    }),
  );
  const participants = uniqueStrings(cluster.episodes.flatMap((episode) => episode.participants));
  const sourceStreamIds = uniqueStrings(
    cluster.episodes.flatMap((episode) => episode.source_stream_ids),
  ) as Episode["source_stream_ids"];
  const tags = uniqueStrings(cluster.episodes.flatMap((episode) => episode.tags));
  const startTime = Math.min(...cluster.episodes.map((episode) => episode.start_time));
  const endTime = Math.max(...cluster.episodes.map((episode) => episode.end_time));
  const significance = Math.max(...cluster.episodes.map((episode) => episode.significance));
  const confidence = Math.min(...cluster.episodes.map((episode) => episode.confidence));
  const locationValues = uniqueStrings(
    cluster.episodes.flatMap((episode) => (episode.location === null ? [] : [episode.location])),
  );
  const nowMs = ctx.clock.now();
  const embedding = await ctx.embeddingClient.embed(
    `${merged.title}\n${merged.narrative}\n${tags.join(" ")}\n${participants.join(" ")}`,
  );

  return {
    episode: {
      id: createEpisodeId(),
      title: merged.title.trim(),
      narrative: merged.narrative.trim(),
      participants,
      location: locationValues.length === 1 ? (locationValues[0] ?? null) : null,
      start_time: startTime,
      end_time: endTime,
      source_stream_ids: sourceStreamIds,
      significance,
      tags,
      confidence,
      lineage: {
        derived_from: cluster.episodes.map((episode) => episode.id),
        supersedes: [],
      },
      emotional_arc:
        cluster.episodes.find((episode) => episode.emotional_arc !== null)?.emotional_arc ?? null,
      embedding,
      created_at: nowMs,
      updated_at: nowMs,
    },
    inheritedTier: maxTier(cluster.stats),
  };
}

function serializeEpisode(episode: Episode) {
  return serializableEpisodeSchema.parse({
    ...episode,
    embedding: Array.from(episode.embedding),
  });
}

function deserializeEpisode(episode: z.infer<typeof serializableEpisodeSchema>): Episode {
  return {
    ...episode,
    embedding: Float32Array.from(episode.embedding),
  };
}

function buildChange(item: ConsolidatorPlan["items"][number]): OfflineChange {
  return {
    process: "consolidator",
    action: "consolidate",
    targets: {
      source_ids: item.source_episode_ids,
    },
    preview: {
      title: item.merged_episode.title,
      narrative: item.merged_episode.narrative,
      source_ids: item.source_episode_ids,
    },
  };
}

export type ConsolidatorProcessOptions = {
  episodicRepository: OfflineContext["episodicRepository"];
  registry: ReverserRegistry;
};

export class ConsolidatorProcess implements OfflineProcess {
  readonly name = "consolidator" as const;

  constructor(private readonly options: ConsolidatorProcessOptions) {
    this.options.registry.register(this.name, "consolidate", async ({ reversal }) => {
      const parsed = reversal as Partial<ConsolidationReversal>;

      if (typeof parsed.newEpisodeId === "string") {
        await this.options.episodicRepository.delete(parsed.newEpisodeId as Episode["id"]);
      }

      if (Array.isArray(parsed.sourceEpisodes)) {
        for (const entry of parsed.sourceEpisodes) {
          if (
            entry !== null &&
            typeof entry === "object" &&
            typeof entry.id === "string" &&
            entry.lineage !== undefined
          ) {
            await this.options.episodicRepository.update(entry.id as Episode["id"], {
              lineage: entry.lineage,
            });
          }
        }
      }

      if (Array.isArray(parsed.sourceStats)) {
        for (const stats of parsed.sourceStats) {
          const parsedStats = episodeStatsSchema.safeParse(stats);

          if (!parsedStats.success) {
            continue;
          }

          this.options.episodicRepository.updateStats(
            parsedStats.data.episode_id,
            parsedStats.data,
          );
        }
      }
    });
  }

  async plan(ctx: OfflineContext, opts: { budget?: number } = {}): Promise<ConsolidatorPlan> {
    const errors: OfflineProcessError[] = [];
    const items: ConsolidatorPlan["items"] = [];
    const budget = opts.budget ?? ctx.config.offline.consolidator.budget;
    const episodes = await ctx.episodicRepository.listAll();
    const statsById = new Map(
      ctx.episodicRepository.listStats().map((stats) => [stats.episode_id, stats] as const),
    );
    const clusters = buildClusters(
      episodes.filter((episode) => !(statsById.get(episode.id)?.archived ?? false)),
      statsById,
      ctx.config.offline.consolidator.similarityThreshold,
      ctx.config.offline.consolidator.minClusterSize,
      ctx.config.offline.consolidator.maxClustersPerRun,
    );
    let tokensUsed = 0;
    let budgetExhausted = false;

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient = wrapClient(ctx.llm.background);

        for (const cluster of clusters) {
          try {
            const merged = await buildMergedEpisode(ctx, llmClient, cluster);
            items.push({
              source_episode_ids: cluster.episodes.map((episode) => episode.id),
              merged_episode: serializeEpisode(merged.episode),
              inherited_tier: merged.inheritedTier,
            });
          } catch (error) {
            if (error instanceof BudgetExceededError) {
              throw error;
            }

            errors.push({
              process: this.name,
              message: error instanceof Error ? error.message : String(error),
              code: error instanceof StorageError ? error.code : undefined,
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

    return consolidatorPlanSchema.parse({
      process: this.name,
      items,
      errors,
      tokens_used: tokensUsed,
      budget_exhausted: budgetExhausted,
    });
  }

  preview(plan: ConsolidatorPlan): OfflineResult {
    const parsed = consolidatorPlanSchema.parse(plan);

    return {
      process: this.name,
      dryRun: true,
      changes: parsed.items.map((item) => buildChange(item)),
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
    };
  }

  async apply(ctx: OfflineContext, rawPlan: ConsolidatorPlan): Promise<OfflineResult> {
    const plan = consolidatorPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];

    for (const item of plan.items) {
      const mergedEpisode = deserializeEpisode(item.merged_episode);
      const sourceEpisodes = await ctx.episodicRepository.getMany(item.source_episode_ids);
      const sourceStats = item.source_episode_ids
        .map((episodeId) => ctx.episodicRepository.getStats(episodeId))
        .filter((stats): stats is EpisodeStats => stats !== null);

      if (
        sourceEpisodes.length !== item.source_episode_ids.length ||
        sourceStats.length !== item.source_episode_ids.length
      ) {
        throw new StorageError("Consolidator plan references missing source episodes", {
          code: "CONSOLIDATOR_PLAN_INVALID",
        });
      }

      const previousSourceEpisodes = sourceEpisodes.map((episode) => ({
        id: episode.id,
        lineage: episode.lineage,
      }));
      const previousSourceStats = sourceStats.map((stats) => ({ ...stats }));

      try {
        await ctx.episodicRepository.insert(mergedEpisode);
        ctx.episodicRepository.updateStats(mergedEpisode.id, {
          tier: item.inherited_tier,
          promoted_at: ctx.clock.now(),
          promoted_from: "consolidator",
          archived: false,
        });

        for (const sourceEpisode of sourceEpisodes) {
          await ctx.episodicRepository.update(sourceEpisode.id, {
            lineage: {
              derived_from: sourceEpisode.lineage.derived_from,
              supersedes: uniqueStrings([
                ...sourceEpisode.lineage.supersedes,
                mergedEpisode.id,
              ]) as Episode["lineage"]["supersedes"],
            },
          });
        }

        for (const sourceStat of sourceStats) {
          ctx.episodicRepository.updateStats(sourceStat.episode_id, {
            archived: true,
          });
        }
      } catch (error) {
        await ctx.episodicRepository.delete(mergedEpisode.id);

        for (const sourceEpisode of previousSourceEpisodes) {
          await ctx.episodicRepository.update(sourceEpisode.id as Episode["id"], {
            lineage: sourceEpisode.lineage,
          });
        }

        for (const sourceStat of previousSourceStats) {
          ctx.episodicRepository.updateStats(sourceStat.episode_id, sourceStat);
        }

        throw error;
      }

      ctx.auditLog.record({
        run_id: ctx.runId,
        process: this.name,
        action: "consolidate",
        targets: {
          newEpisodeId: mergedEpisode.id,
          sourceIds: item.source_episode_ids,
        },
        reversal: {
          newEpisodeId: mergedEpisode.id,
          sourceEpisodes: previousSourceEpisodes,
          sourceStats: previousSourceStats,
        } satisfies ConsolidationReversal,
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
