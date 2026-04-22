import { z } from "zod";

import { moodHistoryEntrySchema, type MoodHistoryEntry } from "../../memory/affective/index.js";
import { computeEpisodeHeat } from "../../memory/episodic/heat.js";
import type { EpisodicRepository } from "../../memory/episodic/index.js";
import {
  episodeIdSchema,
  episodeStatsPatchSchema,
  episodeStatsSchema,
  episodeTierSchema,
  type Episode,
  type EpisodeStats,
  type EpisodeTier,
} from "../../memory/episodic/types.js";
import { traitIdSchema, traitSchema } from "../../memory/self/index.js";
import { socialProfileSchema, type SocialProfile } from "../../memory/social/index.js";
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

const episodePlanItemSchema = z.object({
  action: z.enum(["promote", "demote", "archive", "decay"]),
  episode_id: episodeIdSchema,
  patch: episodeStatsPatchSchema,
  previous: episodeStatsSchema,
});

const trimMoodHistoryPlanItemSchema = z.object({
  action: z.literal("trim_mood_history"),
  removed: z.array(moodHistoryEntrySchema),
});

const socialPlanItemSchema = z.object({
  action: z.literal("refresh_social_profile"),
  entity_id: socialProfileSchema.shape.entity_id,
  previous: socialProfileSchema,
  next: socialProfileSchema,
});

const traitDecayPlanItemSchema = z.object({
  action: z.literal("decay_trait"),
  trait_id: traitIdSchema,
  previous: traitSchema,
});

const curatorPlanItemSchema = z.discriminatedUnion("action", [
  episodePlanItemSchema,
  trimMoodHistoryPlanItemSchema,
  socialPlanItemSchema,
  traitDecayPlanItemSchema,
]);

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

function buildEpisodeChange(item: z.infer<typeof episodePlanItemSchema>): OfflineChange {
  return {
    process: "curator",
    action: item.action,
    targets: {
      episode_id: item.episode_id,
    },
    preview: item.patch,
  };
}

function buildChange(item: CuratorPlan["items"][number]): OfflineChange {
  if ("episode_id" in item) {
    return buildEpisodeChange(item);
  }

  if ("trait_id" in item) {
    return {
      process: "curator",
      action: item.action,
      targets: {
        trait_id: item.trait_id,
      },
      preview: {
        label: item.previous.label,
        strength: item.previous.strength,
      },
    };
  }

  if (item.action === "trim_mood_history") {
    return {
      process: "curator",
      action: item.action,
      targets: {
        removed: item.removed.length,
      },
    };
  }

  return {
    process: "curator",
    action: item.action,
    targets: {
      entity_id: item.entity_id,
    },
    preview: {
      commitment_count: item.next.commitment_count,
      interaction_count: item.next.interaction_count,
      sentiment_points: item.next.sentiment_history.length,
    },
  };
}

function buildEpisodeItems(
  ctx: OfflineContext,
  episodes: readonly Episode[],
): CuratorPlan["items"] {
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

function buildMoodItems(ctx: OfflineContext): CuratorPlan["items"] {
  if ((ctx as Partial<OfflineContext>).moodRepository === undefined) {
    return [];
  }

  const nowMs = ctx.clock.now();
  const items: CuratorPlan["items"] = [];

  const threshold = nowMs - ctx.config.affective.moodHistoryRetentionDays * DAY_MS;
  const removed = ctx.moodRepository.historyBefore(threshold);

  if (removed.length > 0) {
    items.push({
      action: "trim_mood_history",
      removed,
    });
  }

  return items;
}

function buildSocialItems(ctx: OfflineContext): CuratorPlan["items"] {
  if (
    (ctx as Partial<OfflineContext>).socialRepository === undefined ||
    (ctx as Partial<OfflineContext>).commitmentRepository === undefined
  ) {
    return [];
  }

  return ctx.socialRepository
    .list(500)
    .map((previous) => {
      const activeCommitmentCount = ctx.commitmentRepository
        .list({ activeOnly: true })
        .filter(
          (commitment) =>
            commitment.made_to_entity === previous.entity_id ||
            commitment.restricted_audience === previous.entity_id ||
            commitment.about_entity === previous.entity_id,
        ).length;
      const nextSentimentHistory = previous.sentiment_history.slice(-50);
      const changed =
        activeCommitmentCount !== previous.commitment_count ||
        nextSentimentHistory.length !== previous.sentiment_history.length;

      if (!changed) {
        return null;
      }

      return {
        action: "refresh_social_profile",
        entity_id: previous.entity_id,
        previous,
        next: socialProfileSchema.parse({
          ...previous,
          commitment_count: activeCommitmentCount,
          sentiment_history: nextSentimentHistory,
          updated_at: ctx.clock.now(),
        }),
      } satisfies z.infer<typeof socialPlanItemSchema>;
    })
    .filter((item): item is z.infer<typeof socialPlanItemSchema> => item !== null);
}

function buildTraitDecayItems(ctx: OfflineContext): CuratorPlan["items"] {
  const nowMs = ctx.clock.now();
  const staleThreshold = nowMs - 7 * DAY_MS;

  return ctx.traitsRepository
    .list()
    .filter((trait) => {
      const lastTouched = Math.max(trait.last_reinforced, trait.last_decayed ?? 0);
      return trait.last_reinforced <= staleThreshold && lastTouched < nowMs;
    })
    .map((trait) => ({
      action: "decay_trait",
      trait_id: trait.id,
      previous: trait,
    }));
}

function buildItems(ctx: OfflineContext, episodes: readonly Episode[]): CuratorPlan["items"] {
  return [
    ...buildEpisodeItems(ctx, episodes),
    ...buildTraitDecayItems(ctx),
    ...buildMoodItems(ctx),
    ...buildSocialItems(ctx),
  ];
}

export type CuratorProcessOptions = {
  episodicRepository: EpisodicRepository;
  moodRepository: OfflineContext["moodRepository"];
  socialRepository: OfflineContext["socialRepository"];
  registry: ReverserRegistry;
  clock?: Clock;
};

export class CuratorProcess implements OfflineProcess<CuratorPlan> {
  readonly name = "curator" as const;

  constructor(private readonly options: CuratorProcessOptions) {
    const revertStats = async (input: { reversal: Record<string, unknown> }): Promise<void> => {
      const previous = input.reversal.previous;

      if (!Array.isArray(previous)) {
        return;
      }

      for (const item of previous) {
        const parsed = episodeStatsSchema.safeParse(item);

        if (parsed.success) {
          this.options.episodicRepository.updateStats(parsed.data.episode_id, parsed.data);
        }
      }
    };

    this.options.registry.register(this.name, "promote", revertStats);
    this.options.registry.register(this.name, "demote", revertStats);
    this.options.registry.register(this.name, "archive", revertStats);
    this.options.registry.register(this.name, "decay", revertStats);
    this.options.registry.register(this.name, "trim_mood_history", async ({ reversal }) => {
      const parsed = z.array(moodHistoryEntrySchema).safeParse(reversal.removed);

      if (parsed.success) {
        this.options.moodRepository.restoreHistory(parsed.data);
      }
    });
    this.options.registry.register(this.name, "refresh_social_profile", async ({ reversal }) => {
      const parsed = socialProfileSchema.safeParse(reversal.previous);

      if (parsed.success) {
        this.options.socialRepository.restoreProfile(parsed.data);
      }
    });
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
    const traitDecayIds = plan.items
      .filter((item): item is z.infer<typeof traitDecayPlanItemSchema> => item.action === "decay_trait")
      .map((item) => item.trait_id);

    for (const item of plan.items) {
      if ("episode_id" in item) {
        ctx.episodicRepository.updateStats(item.episode_id, item.patch);
        previousByAction.set(item.action, [
          ...(previousByAction.get(item.action) ?? []),
          item.previous,
        ]);
        continue;
      }

      if (item.action === "decay_trait") {
        continue;
      }

      if (item.action === "trim_mood_history") {
        ctx.moodRepository.trimHistory(
          ctx.config.affective.moodHistoryRetentionDays,
          ctx.clock.now(),
        );
        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: item.action,
          targets: {
            removed: item.removed.length,
          },
          reversal: {
            removed: item.removed,
          },
        });
        continue;
      }

      ctx.socialRepository.restoreProfile(item.next);
      ctx.auditLog.record({
        run_id: ctx.runId,
        process: this.name,
        action: item.action,
        targets: {
          entity_id: item.entity_id,
        },
        reversal: {
          previous: item.previous,
        },
      });
    }

    if (traitDecayIds.length > 0) {
      ctx.traitsRepository.decay(
        ctx.config.offline.curator.traitHalfLifeDays * 24,
        ctx.clock.now(),
        { traitIds: traitDecayIds },
      );
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
