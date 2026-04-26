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
  GROWTH_MARKER_CATEGORIES,
  autobiographicalPeriodIdSchema,
  autobiographicalPeriodSchema,
  growthMarkerSchema,
  type AutobiographicalPeriod,
} from "../../memory/self/index.js";
import { createAutobiographicalPeriodId, createGrowthMarkerId } from "../../util/ids.js";
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

const DAY_MS = 24 * 60 * 60 * 1_000;
const GROWTH_MARKER_CONFIDENCE_CEILING = 0.6;

const selfNarratorObservationSchema = z.object({
  observation: z
    .object({
      category: z.enum(GROWTH_MARKER_CATEGORIES),
      what_changed: z.string().min(1),
      before_description: z.string().nullable().optional(),
      after_description: z.string().nullable().optional(),
      confidence: z.number().min(0).max(1),
      evidence_episode_ids: z.array(z.string().min(1)).min(2),
    })
    .nullable(),
});
const SELF_NARRATOR_TOOL_NAME = "EmitSelfNarratorObservations";
export const SELF_NARRATOR_TOOL = {
  name: SELF_NARRATOR_TOOL_NAME,
  description: "Emit a grounded autobiographical growth observation or null.",
  inputSchema: toToolInputSchema(selfNarratorObservationSchema),
} satisfies LLMToolDefinition;

const serializableGrowthMarkerSchema = growthMarkerSchema.extend({
  evidence_episode_ids: z.array(episodeIdSchema).min(1),
});

const selfNarratorPlanItemSchema = z.discriminatedUnion("action", [
  z.object({
    action: z.literal("open_period"),
    period: autobiographicalPeriodSchema,
  }),
  z.object({
    action: z.literal("close_period"),
    previous: autobiographicalPeriodSchema,
    end_ts: z.number().finite(),
  }),
  z.object({
    action: z.literal("update_period_narrative"),
    period_id: autobiographicalPeriodIdSchema,
    previous: autobiographicalPeriodSchema,
    narrative: z.string(),
    key_episode_ids: z.array(episodeIdSchema),
    themes: z.array(z.string().min(1)),
  }),
  z.object({
    action: z.literal("add_growth_marker"),
    marker: serializableGrowthMarkerSchema,
  }),
]);

export const selfNarratorPlanSchema = z.object({
  process: z.literal("self-narrator"),
  items: z.array(selfNarratorPlanItemSchema),
  errors: z
    .array(
      z.object({
        process: z.literal("self-narrator"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
});

export type SelfNarratorPlan = z.infer<typeof selfNarratorPlanSchema>;

type NarrativeCluster = {
  key: string;
  episodes: Episode[];
};

type SelfNarratorReversal = {
  previous?: AutobiographicalPeriod;
  period_id?: string;
  marker_id?: string;
};

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))];
}

function uniqueValues<T>(values: readonly T[]): T[] {
  return [...new Set(values)];
}

function defaultQuarterLabel(nowMs: number): string {
  const date = new Date(nowMs);
  const year = date.getUTCFullYear();
  const quarter = Math.floor(date.getUTCMonth() / 3) + 1;
  return `${year}-Q${quarter}`;
}

function buildObservationPrompt(cluster: NarrativeCluster): string {
  return [
    "Infer one concise growth observation from this episode cluster if it shows meaningful change.",
    `Emit your result by calling the ${SELF_NARRATOR_TOOL_NAME} tool exactly once.`,
    "Return null if there is no grounded growth signal.",
    "Only cite evidence_episode_ids from the provided episodes.",
    `Cluster: ${cluster.key}`,
    "Episodes:",
    ...cluster.episodes.map((episode) =>
      JSON.stringify({
        id: episode.id,
        title: episode.title,
        narrative: episode.narrative,
        participants: episode.participants,
        tags: episode.tags,
      }),
    ),
  ].join("\n");
}

function parseObservationResponse(result: LLMCompleteResult) {
  const call = result.tool_calls.find((toolCall) => toolCall.name === SELF_NARRATOR_TOOL_NAME);

  if (call === undefined) {
    throw new StorageError(`Self-narrator did not emit tool ${SELF_NARRATOR_TOOL_NAME}`, {
      code: "SELF_NARRATOR_INVALID",
    });
  }

  return selfNarratorObservationSchema.parse(call.input);
}

function collectClusters(
  episodes: readonly Episode[],
  minSupportEpisodes: number,
  maxObservationsPerRun: number,
): NarrativeCluster[] {
  const byKey = new Map<string, Episode[]>();

  for (const episode of episodes) {
    const key = episode.tags[0]?.trim().toLowerCase() || "untagged";
    byKey.set(key, [...(byKey.get(key) ?? []), episode]);
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
    .filter((cluster) => cluster.episodes.length >= minSupportEpisodes)
    .sort((left, right) => right.episodes.length - left.episodes.length)
    .slice(0, maxObservationsPerRun);
}

function shouldStartNewPeriod(
  currentPeriod: AutobiographicalPeriod | null,
  themes: readonly string[],
  nowMs: number,
  cadenceHintDays: number,
): boolean {
  if (currentPeriod === null) {
    return true;
  }

  const ageMs = Math.max(0, nowMs - currentPeriod.start_ts);

  if (ageMs < cadenceHintDays * DAY_MS || themes.length === 0) {
    return false;
  }

  const currentThemes = new Set(currentPeriod.themes.map((theme) => theme.toLowerCase()));
  return themes.some((theme) => !currentThemes.has(theme.toLowerCase()));
}

function buildNarrative(existingNarrative: string | null, observations: readonly string[]): string {
  const parts = uniqueStrings([
    existingNarrative?.trim() ?? "",
    ...observations.map((observation) => observation.trim()),
  ]);

  if (parts.length === 0) {
    return "A new autobiographical period began.";
  }

  return parts.join(" ");
}

function buildChange(item: SelfNarratorPlan["items"][number]): OfflineChange {
  if (item.action === "open_period") {
    return {
      process: "self-narrator",
      action: "open_period",
      targets: {
        period_id: item.period.id,
        label: item.period.label,
      },
    };
  }

  if (item.action === "close_period") {
    return {
      process: "self-narrator",
      action: "close_period",
      targets: {
        period_id: item.previous.id,
        label: item.previous.label,
      },
      preview: {
        end_ts: item.end_ts,
      },
    };
  }

  if (item.action === "update_period_narrative") {
    return {
      process: "self-narrator",
      action: "update_period_narrative",
      targets: {
        period_id: item.period_id,
      },
      preview: {
        narrative: item.narrative,
        themes: item.themes,
      },
    };
  }

  return {
    process: "self-narrator",
    action: "add_growth_marker",
    targets: {
      marker_id: item.marker.id,
      category: item.marker.category,
    },
    preview: {
      what_changed: item.marker.what_changed,
    },
  };
}

export type SelfNarratorProcessOptions = {
  autobiographicalRepository: OfflineContext["autobiographicalRepository"];
  growthMarkersRepository: OfflineContext["growthMarkersRepository"];
  registry: ReverserRegistry;
};

export class SelfNarratorProcess implements OfflineProcess<SelfNarratorPlan> {
  readonly name = "self-narrator" as const;

  constructor(private readonly options: SelfNarratorProcessOptions) {
    this.options.registry.register(this.name, "open_period", async ({ reversal }) => {
      const parsed = reversal as Partial<SelfNarratorReversal>;

      if (typeof parsed.period_id === "string") {
        this.options.autobiographicalRepository.deletePeriod(parsed.period_id as never);
      }
    });
    this.options.registry.register(this.name, "close_period", async ({ reversal }) => {
      const parsed = reversal as Partial<SelfNarratorReversal>;

      if (parsed.previous !== undefined) {
        this.options.autobiographicalRepository.upsertPeriod(parsed.previous);
      }
    });
    this.options.registry.register(this.name, "update_period_narrative", async ({ reversal }) => {
      const parsed = reversal as Partial<SelfNarratorReversal>;

      if (parsed.previous !== undefined) {
        this.options.autobiographicalRepository.upsertPeriod(parsed.previous);
      }
    });
    this.options.registry.register(this.name, "add_growth_marker", async ({ reversal }) => {
      const parsed = reversal as Partial<SelfNarratorReversal>;

      if (typeof parsed.marker_id === "string") {
        this.options.growthMarkersRepository.delete(parsed.marker_id as never);
      }
    });
  }

  async plan(
    ctx: OfflineContext,
    opts: { budget?: number; params?: Record<string, unknown> } = {},
  ): Promise<SelfNarratorPlan> {
    const errors: OfflineProcessError[] = [];
    const items: SelfNarratorPlan["items"] = [];
    const budget = opts.budget ?? ctx.config.offline.selfNarrator.budget;
    const nowMs = ctx.clock.now();
    const configuredLabel = typeof opts.params?.label === "string" ? opts.params.label.trim() : "";
    const currentPeriod = ctx.autobiographicalRepository.currentPeriod();
    const selfAudienceEntityId = ctx.entityRepository.findByName("self");
    // Existing global periods and growth markers may already cite older
    // audience-scoped evidence; this write-side guard only prevents new ones.
    const sourceEpisodes = (await ctx.episodicRepository.listAll()).filter(
      (episode) =>
        isEpisodeInGlobalIdentityScope(episode, selfAudienceEntityId) &&
        (currentPeriod === null ||
          episode.start_time >= currentPeriod.start_ts ||
          episode.end_time >= currentPeriod.start_ts),
    );
    const clusters = collectClusters(
      sourceEpisodes,
      ctx.config.offline.selfNarrator.minSupportEpisodes,
      ctx.config.offline.selfNarrator.maxObservationsPerRun,
    );
    const markerCandidates: Array<z.infer<typeof serializableGrowthMarkerSchema>> = [];
    let tokensUsed = 0;
    let budgetExhausted = false;

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient: LLMClient = wrapClient(ctx.llm.background);

        for (const cluster of clusters) {
          try {
            const response = parseObservationResponse(
              await llmClient.complete({
                model: ctx.config.anthropic.models.background,
                system:
                  "You identify grounded autobiographical growth markers. Return null when the evidence is weak.",
                messages: [
                  {
                    role: "user",
                    content: buildObservationPrompt(cluster),
                  },
                ],
                tools: [SELF_NARRATOR_TOOL],
                tool_choice: { type: "tool", name: SELF_NARRATOR_TOOL_NAME },
                max_tokens: 8_000,
                budget: "offline-self-narrator",
              }),
            );

            if (response.observation === null) {
              continue;
            }

            const allowedIds = new Set<string>(cluster.episodes.map((episode) => episode.id));

            if (!response.observation.evidence_episode_ids.every((id) => allowedIds.has(id))) {
              throw new StorageError("Self-narrator referenced episodes outside the cluster", {
                code: "SELF_NARRATOR_INVALID_REF",
              });
            }

            const evidenceEpisodeIds = response.observation.evidence_episode_ids.map(
              (id) => id as Episode["id"],
            );

            markerCandidates.push(
              serializableGrowthMarkerSchema.parse({
                id: createGrowthMarkerId(),
                ts: nowMs,
                category: response.observation.category,
                what_changed: response.observation.what_changed,
                before_description: response.observation.before_description ?? null,
                after_description: response.observation.after_description ?? null,
                evidence_episode_ids: evidenceEpisodeIds,
                confidence: Math.min(
                  GROWTH_MARKER_CONFIDENCE_CEILING,
                  response.observation.confidence,
                ),
                source_process: "self-narrator",
                provenance: {
                  kind: "offline",
                  process: "self-narrator",
                },
                created_at: nowMs,
              }),
            );
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

    const observations = markerCandidates.map((marker) => marker.what_changed);
    const keyEpisodeIds = uniqueValues(
      markerCandidates.flatMap((marker) => marker.evidence_episode_ids),
    ).slice(0, 8);
    const themes = uniqueStrings([
      ...markerCandidates.map((marker) => marker.category),
      ...clusters.map((cluster) => cluster.key),
    ]).slice(0, 6);
    const nextLabel =
      configuredLabel.length > 0
        ? configuredLabel
        : (currentPeriod?.label ?? defaultQuarterLabel(nowMs));
    const openNewPeriod = shouldStartNewPeriod(
      currentPeriod,
      themes,
      nowMs,
      ctx.config.offline.selfNarrator.cadenceHintDays,
    );
    let targetPeriod = currentPeriod;

    if (currentPeriod === null || openNewPeriod) {
      if (currentPeriod !== null) {
        items.push({
          action: "close_period",
          previous: currentPeriod,
          end_ts: nowMs,
        });
      }

      const period = autobiographicalPeriodSchema.parse({
        id: createAutobiographicalPeriodId(),
        label: nextLabel,
        start_ts: sourceEpisodes.at(-1)?.start_time ?? nowMs,
        end_ts: null,
        narrative: buildNarrative(null, observations),
        key_episode_ids: keyEpisodeIds,
        themes,
        provenance: {
          kind: "offline",
          process: this.name,
        },
        created_at: nowMs,
        last_updated: nowMs,
      });

      items.push({
        action: "open_period",
        period,
      });
      targetPeriod = period;
    } else if (currentPeriod !== null) {
      const nextNarrative = buildNarrative(currentPeriod.narrative, observations);
      const nextThemes = uniqueStrings([...currentPeriod.themes, ...themes]).slice(0, 6);
      const nextKeyEpisodes = uniqueValues([
        ...currentPeriod.key_episode_ids,
        ...keyEpisodeIds,
      ]).slice(0, 8);

      if (
        nextNarrative !== currentPeriod.narrative ||
        JSON.stringify(nextThemes) !== JSON.stringify(currentPeriod.themes) ||
        JSON.stringify(nextKeyEpisodes) !== JSON.stringify(currentPeriod.key_episode_ids)
      ) {
        items.push({
          action: "update_period_narrative",
          period_id: currentPeriod.id,
          previous: currentPeriod,
          narrative: nextNarrative,
          key_episode_ids: nextKeyEpisodes,
          themes: nextThemes,
        });
      }
    }

    for (const marker of markerCandidates) {
      items.push({
        action: "add_growth_marker",
        marker,
      });
    }

    void targetPeriod;

    return selfNarratorPlanSchema.parse({
      process: this.name,
      items,
      errors,
      tokens_used: tokensUsed,
      budget_exhausted: budgetExhausted,
    });
  }

  preview(plan: SelfNarratorPlan): OfflineResult {
    const parsed = selfNarratorPlanSchema.parse(plan);

    return {
      process: this.name,
      dryRun: true,
      changes: parsed.items.map((item) => buildChange(item)),
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
    };
  }

  async apply(ctx: OfflineContext, rawPlan: SelfNarratorPlan): Promise<OfflineResult> {
    const plan = selfNarratorPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];
    const processProvenance = {
      kind: "offline" as const,
      process: this.name,
    };

    for (let index = 0; index < plan.items.length; index += 1) {
      const item = plan.items[index]!;
      const nextItem = plan.items[index + 1];

      if (item.action === "close_period" && nextItem?.action === "open_period") {
        const result = ctx.identityService.updatePeriod(
          item.previous.id,
          {
            end_ts: item.end_ts,
          },
          processProvenance,
        );

        if (result.status === "requires_review") {
          ctx.reviewQueueRepository.enqueue({
            kind: "identity_inconsistency",
            refs: {
              target_type: "autobiographical_period",
              target_id: item.previous.id,
              repair_op: "patch",
              patch: {
                end_ts: item.end_ts,
              },
              proposed_provenance: processProvenance,
              // Keep rollover close+open coupled so accepting the review applies both together.
              next_period_open_payload: nextItem.period,
            },
            reason: `self-narrator proposed closing autobiographical period ${item.previous.id}`,
          });
          index += 1;
          continue;
        }

        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "close_period",
          targets: {
            period_id: item.previous.id,
            label: item.previous.label,
          },
          reversal: {
            previous: item.previous,
          } satisfies SelfNarratorReversal,
        });
        ctx.autobiographicalRepository.upsertPeriod(nextItem.period);
        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "open_period",
          targets: {
            period_id: nextItem.period.id,
            label: nextItem.period.label,
          },
          reversal: {
            period_id: nextItem.period.id,
          } satisfies SelfNarratorReversal,
        });
        changes.push(buildChange(item), buildChange(nextItem));
        index += 1;
        continue;
      }

      if (item.action === "open_period") {
        ctx.autobiographicalRepository.upsertPeriod(item.period);
        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "open_period",
          targets: {
            period_id: item.period.id,
            label: item.period.label,
          },
          reversal: {
            period_id: item.period.id,
          } satisfies SelfNarratorReversal,
        });
        changes.push(buildChange(item));
        continue;
      }

      if (item.action === "close_period") {
        const result = ctx.identityService.updatePeriod(
          item.previous.id,
          {
            end_ts: item.end_ts,
          },
          processProvenance,
        );

        if (result.status === "requires_review") {
          ctx.reviewQueueRepository.enqueue({
            kind: "identity_inconsistency",
            refs: {
              target_type: "autobiographical_period",
              target_id: item.previous.id,
              repair_op: "patch",
              patch: {
                end_ts: item.end_ts,
              },
              proposed_provenance: processProvenance,
            },
            reason: `self-narrator proposed closing autobiographical period ${item.previous.id}`,
          });
          continue;
        }

        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "close_period",
          targets: {
            period_id: item.previous.id,
            label: item.previous.label,
          },
          reversal: {
            previous: item.previous,
          } satisfies SelfNarratorReversal,
        });
        changes.push(buildChange(item));
        continue;
      }

      if (item.action === "update_period_narrative") {
        const result = ctx.identityService.updatePeriod(
          item.period_id,
          {
            narrative: item.narrative,
            key_episode_ids: item.key_episode_ids,
            themes: item.themes,
          },
          processProvenance,
        );

        if (result.status === "requires_review") {
          ctx.reviewQueueRepository.enqueue({
            kind: "identity_inconsistency",
            refs: {
              target_type: "autobiographical_period",
              target_id: item.period_id,
              repair_op: "patch",
              patch: {
                narrative: item.narrative,
                key_episode_ids: item.key_episode_ids,
                themes: item.themes,
              },
              proposed_provenance: processProvenance,
              evidence_episode_ids: item.key_episode_ids,
            },
            reason: `self-narrator proposed revising autobiographical period ${item.period_id}`,
          });
          continue;
        }

        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "update_period_narrative",
          targets: {
            period_id: item.period_id,
          },
          reversal: {
            previous: item.previous,
          } satisfies SelfNarratorReversal,
        });
        changes.push(buildChange(item));
        continue;
      }

      if (ctx.growthMarkersRepository.get(item.marker.id) === null) {
        ctx.growthMarkersRepository.add(item.marker);
      }

      ctx.auditLog.record({
        run_id: ctx.runId,
        process: this.name,
        action: "add_growth_marker",
        targets: {
          marker_id: item.marker.id,
        },
        reversal: {
          marker_id: item.marker.id,
        } satisfies SelfNarratorReversal,
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
    opts: { dryRun?: boolean; budget?: number; params?: Record<string, unknown> },
  ): Promise<OfflineResult> {
    const plan = await this.plan(ctx, opts);
    return opts.dryRun === true ? this.preview(plan) : this.apply(ctx, plan);
  }
}
