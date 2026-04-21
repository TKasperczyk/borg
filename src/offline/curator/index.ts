import { z } from "zod";

import { computeEpisodeHeat } from "../../memory/episodic/heat.js";
import {
  episodeStatsPatchSchema,
  episodeStatsSchema,
  episodeTierSchema,
  type Episode,
  type EpisodeStats,
  type EpisodeTier,
} from "../../memory/episodic/types.js";
import type { EpisodicRepository } from "../../memory/episodic/index.js";
import { episodeIdSchema } from "../../memory/episodic/types.js";
import type { Clock } from "../../util/clock.js";

import type { ReverserRegistry } from "../audit-log.js";
import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessError,
  OfflineResult,
} from "../types.js";

const HOUR_MS = 60 * 60 * 1_000;
const DAY_MS = 24 * HOUR_MS;
const TIER_ORDER: Record<EpisodeTier, number> = {
  T1: 1,
  T2: 2,
  T3: 3,
  T4: 4,
};

const curatorPlanActionSchema = z.enum(["promote", "demote", "archive", "decay"]);

const curatorPlanItemSchema = z.object({
  action: curatorPlanActionSchema,
  episode_id: episodeIdSchema,
  patch: episodeStatsPatchSchema,
  previous: episodeStatsSchema,
});

export const curatorPlanSchema = z.object({
  process: z.literal("curator"),
  items: z.array(curatorPlanItemSchema),
  errors: z
    .array(
      z.object({
        process: z.literal("curator"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.literal(0).default(0),
  budget_exhausted: z.literal(false).default(false),
});

export type CuratorPlan = z.infer<typeof curatorPlanSchema>;

function compareTiers(left: EpisodeTier, right: EpisodeTier): number {
  return TIER_ORDER[left] - TIER_ORDER[right];
}

function buildChange(item: CuratorPlan["items"][number]): OfflineChange {
  return {
    process: "curator",
    action: item.action,
    targets: {
      episode_id: item.episode_id,
    },
    preview: item.patch,
  };
}

function buildItems(ctx: OfflineContext, episodes: readonly Episode[]): CuratorPlan["items"] {
  const nowMs = ctx.clock.now();
  const items: CuratorPlan["items"] = [];

  for (const episode of episodes) {
    const stats = ctx.episodicRepository.getStats(episode.id);

    if (stats === null) {
      continue;
    }

    const heat = computeEpisodeHeat(episode, stats, nowMs);
    const ageMs = Math.max(0, nowMs - episode.created_at);
    const lastUsedAt = stats.last_retrieved ?? stats.promoted_at;

    if (stats.last_decayed_at !== nowMs) {
      items.push({
        action: "decay",
        episode_id: episode.id,
        patch: {
          last_decayed_at: nowMs,
        },
        previous: stats,
      });
    }

    if (stats.tier === "T1" && heat >= ctx.config.offline.curator.t1Heat && ageMs >= 2 * HOUR_MS) {
      items.push({
        action: "promote",
        episode_id: episode.id,
        patch: {
          tier: episodeTierSchema.parse("T2"),
          promoted_at: nowMs,
          promoted_from: "curator",
        },
        previous: stats,
      });
      continue;
    }

    if (stats.tier === "T2" && heat >= ctx.config.offline.curator.t2Heat && ageMs >= 7 * DAY_MS) {
      items.push({
        action: "promote",
        episode_id: episode.id,
        patch: {
          tier: episodeTierSchema.parse("T3"),
          promoted_at: nowMs,
          promoted_from: "curator",
        },
        previous: stats,
      });
      continue;
    }

    if (
      stats.tier === "T3" &&
      heat < ctx.config.offline.curator.t3DemoteHeat &&
      nowMs - lastUsedAt >= 30 * DAY_MS
    ) {
      items.push({
        action: "demote",
        episode_id: episode.id,
        patch: {
          tier: episodeTierSchema.parse("T2"),
          promoted_at: nowMs,
          promoted_from: "curator",
        },
        previous: stats,
      });
      continue;
    }

    if (
      !stats.archived &&
      compareTiers(stats.tier, "T2") <= 0 &&
      heat < ctx.config.offline.curator.archiveMinHeat &&
      ageMs >= ctx.config.offline.curator.archiveAgeDays * DAY_MS
    ) {
      items.push({
        action: "archive",
        episode_id: episode.id,
        patch: {
          archived: true,
        },
        previous: stats,
      });
    }
  }

  return items;
}

export type CuratorProcessOptions = {
  episodicRepository: EpisodicRepository;
  registry: ReverserRegistry;
  clock?: Clock;
};

export class CuratorProcess implements OfflineProcess {
  readonly name = "curator" as const;

  constructor(private readonly options: CuratorProcessOptions) {
    const revertStats = async (input: { reversal: Record<string, unknown> }): Promise<void> => {
      const previous = input.reversal.previous;

      if (!Array.isArray(previous)) {
        return;
      }

      for (const item of previous) {
        const parsed = episodeStatsSchema.safeParse(item);

        if (!parsed.success) {
          continue;
        }

        this.options.episodicRepository.updateStats(parsed.data.episode_id, parsed.data);
      }
    };

    this.options.registry.register(this.name, "promote", revertStats);
    this.options.registry.register(this.name, "demote", revertStats);
    this.options.registry.register(this.name, "archive", revertStats);
    this.options.registry.register(this.name, "decay", revertStats);
  }

  async plan(ctx: OfflineContext, _opts: { budget?: number } = {}): Promise<CuratorPlan> {
    const episodes = await ctx.episodicRepository.listAll();

    return curatorPlanSchema.parse({
      process: this.name,
      items: buildItems(ctx, episodes),
      errors: [] satisfies OfflineProcessError[],
      tokens_used: 0,
      budget_exhausted: false,
    });
  }

  preview(plan: CuratorPlan): OfflineResult {
    return {
      process: this.name,
      dryRun: true,
      changes: plan.items.map((item) => buildChange(item)),
      tokens_used: plan.tokens_used,
      errors: plan.errors,
      budget_exhausted: plan.budget_exhausted,
    };
  }

  async apply(ctx: OfflineContext, rawPlan: CuratorPlan): Promise<OfflineResult> {
    const plan = curatorPlanSchema.parse(rawPlan);
    const previousByAction = new Map<string, EpisodeStats[]>();

    for (const item of plan.items) {
      ctx.episodicRepository.updateStats(item.episode_id, item.patch);
      previousByAction.set(item.action, [
        ...(previousByAction.get(item.action) ?? []),
        item.previous,
      ]);
    }

    for (const [action, previous] of previousByAction.entries()) {
      if (previous.length === 0) {
        continue;
      }

      ctx.auditLog.record({
        run_id: ctx.runId,
        process: this.name,
        action,
        targets: {
          episode_ids: previous.map((stats) => stats.episode_id),
        },
        reversal: {
          previous,
        },
      });
    }

    return {
      process: this.name,
      dryRun: false,
      changes: plan.items.map((item) => buildChange(item)),
      tokens_used: 0,
      errors: plan.errors,
      budget_exhausted: false,
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
