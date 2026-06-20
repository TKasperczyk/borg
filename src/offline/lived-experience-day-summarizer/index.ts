import { z } from "zod";

import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import {
  combineMemoryDisclosureLabels,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import {
  livedExperienceDaySummarySchema,
  type ActivityAutobiographicalSourceEvent,
  type ActivityGlobalDailyDensityRow,
  type LivedExperienceDaySummary,
} from "../../memory/activity/index.js";
import { episodeIdSchema, type Episode } from "../../memory/episodic/index.js";
import type {
  SelfDecisionDailyDensityRow,
  SelfDecisionProjectionSourceEvent,
} from "../../memory/self-decisions/index.js";
import { BudgetExceededError, StorageError } from "../../util/errors.js";
import {
  createLivedExperienceDaySummaryId,
  parseLivedExperienceDaySummaryId,
  type EntityId,
  type EpisodeId,
  type LivedExperienceDaySummaryId,
  type StreamEntryId,
} from "../../util/ids.js";
import { streamEntryIdSchema } from "../../util/id-schemas.js";
import { type JsonValue } from "../../util/json-value.js";
import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";
import { formatUtcDayBoundary, timestampFromUtcDayKey, utcDayStartMs } from "../../util/utc-day.js";

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

const DAY_MS = 24 * 60 * 60 * 1_000;
const PROCESS_NAME = "lived-experience-day-summarizer";
const TOOL_NAME = "EmitLivedExperienceDaySummary";

const livedExperienceDaySummaryToolSchema = z.object({
  utc_day: z.string().min(1),
  gist: z.string().min(1),
  salience: z.number().min(0).max(1).default(0.5),
  cited_episode_ids: z.array(episodeIdSchema).default([]),
  cited_source_stream_entry_ids: z.array(streamEntryIdSchema).default([]),
});

export const LIVED_EXPERIENCE_DAY_SUMMARY_TOOL = {
  name: TOOL_NAME,
  description:
    "Emit one first-person experiential summary for a closed UTC day from self-private decisions, counts, and summarized episode evidence.",
  inputSchema: toToolInputSchema(livedExperienceDaySummaryToolSchema),
} satisfies LLMToolDefinition;

const livedExperienceDaySummarizerPlanItemSchema = z.object({
  action: z.literal("upsert_day_summary"),
  summary: livedExperienceDaySummarySchema,
  previous: livedExperienceDaySummarySchema.nullable(),
});

export const livedExperienceDaySummarizerPlanSchema = z.object({
  process: z.literal(PROCESS_NAME),
  items: z.array(livedExperienceDaySummarizerPlanItemSchema),
  errors: z
    .array(
      z.object({
        process: z.literal(PROCESS_NAME),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
});

export type LivedExperienceDaySummarizerPlan = z.infer<
  typeof livedExperienceDaySummarizerPlanSchema
>;

type LivedExperienceDaySummarizerReversal = {
  previous?: LivedExperienceDaySummary;
  summary_id?: LivedExperienceDaySummaryId;
};

type DayCandidate = {
  utcDay: string;
  dayStartMs: number;
  dayEndMs: number;
  activityDensity: ActivityGlobalDailyDensityRow | null;
  selfDecisionDensity: SelfDecisionDailyDensityRow | null;
};

function uniqueValues<T>(values: readonly T[]): T[] {
  return [...new Set(values)];
}

function parseSummaryId(value: unknown): LivedExperienceDaySummaryId | null {
  if (value === undefined) {
    return null;
  }

  if (typeof value !== "string") {
    throw new StorageError("Invalid lived-experience day summarizer reversal summary_id", {
      code: "LIVED_EXPERIENCE_DAY_SUMMARIZER_REVERSAL_INVALID",
    });
  }

  try {
    return parseLivedExperienceDaySummaryId(value);
  } catch (error) {
    throw new StorageError("Invalid lived-experience day summarizer reversal summary_id", {
      cause: error,
      code: "LIVED_EXPERIENCE_DAY_SUMMARIZER_REVERSAL_INVALID",
    });
  }
}

function sourceStreamEntryIdsFromEpisodes(episodes: readonly Episode[]): StreamEntryId[] {
  return uniqueValues(episodes.flatMap((episode) => episode.source_stream_ids));
}

function dayWindow(dayKey: string): { dayStartMs: number; dayEndMs: number } {
  const dayStartMs = timestampFromUtcDayKey(dayKey);

  return {
    dayStartMs,
    dayEndMs: dayStartMs + DAY_MS - 1,
  };
}

function densityCountsSnapshot(candidate: DayCandidate): JsonValue {
  return {
    utc_day: candidate.utcDay,
    day_start_ms: candidate.dayStartMs,
    day_end_ms: candidate.dayEndMs,
    activity: candidate.activityDensity
      ? {
          event_count: candidate.activityDensity.eventCount,
          conversation_turn_count: candidate.activityDensity.conversationTurnCount,
          distinct_session_count: candidate.activityDensity.distinctSessionCount,
          kind_counts: {
            user_contact: candidate.activityDensity.kindCounts.userContact,
            borg_replied: candidate.activityDensity.kindCounts.borgReplied,
            turn_completed: candidate.activityDensity.kindCounts.turnCompleted,
          },
          first_occurred_at: candidate.activityDensity.firstOccurredAt,
          last_occurred_at: candidate.activityDensity.lastOccurredAt,
        }
      : {
          event_count: 0,
          conversation_turn_count: 0,
          distinct_session_count: 0,
          kind_counts: {
            user_contact: 0,
            borg_replied: 0,
            turn_completed: 0,
          },
        },
    self_decisions: candidate.selfDecisionDensity
      ? {
          decision_count: candidate.selfDecisionDensity.decisionCount,
          distinct_decision_shape_count: candidate.selfDecisionDensity.distinctDecisionShapeCount,
          first_occurred_at: candidate.selfDecisionDensity.firstOccurredAt,
          last_occurred_at: candidate.selfDecisionDensity.lastOccurredAt,
        }
      : {
          decision_count: 0,
          distinct_decision_shape_count: 0,
        },
  };
}

function buildCandidateDays(input: {
  activityDensityRows: readonly ActivityGlobalDailyDensityRow[];
  selfDecisionDensityRows: readonly SelfDecisionDailyDensityRow[];
  existingSummaries: readonly LivedExperienceDaySummary[];
  currentOpenDayStartMs: number;
  maxDays: number;
}): DayCandidate[] {
  const activityByDay = new Map(input.activityDensityRows.map((row) => [row.dayKey, row]));
  const selfDecisionByDay = new Map(input.selfDecisionDensityRows.map((row) => [row.dayKey, row]));
  const summarizedDays = new Set(input.existingSummaries.map((summary) => summary.utc_day));
  const dayKeys = uniqueValues([
    ...input.activityDensityRows.map((row) => row.dayKey),
    ...input.selfDecisionDensityRows.map((row) => row.dayKey),
  ]).sort();
  const candidates: DayCandidate[] = [];

  for (const dayKey of dayKeys) {
    const { dayStartMs, dayEndMs } = dayWindow(dayKey);

    if (dayEndMs >= input.currentOpenDayStartMs || summarizedDays.has(dayKey)) {
      continue;
    }

    candidates.push({
      utcDay: dayKey,
      dayStartMs,
      dayEndMs,
      activityDensity: activityByDay.get(dayKey) ?? null,
      selfDecisionDensity: selfDecisionByDay.get(dayKey) ?? null,
    });

    if (candidates.length >= input.maxDays) {
      break;
    }
  }

  return candidates;
}

function buildDayPrompt(input: {
  candidate: DayCandidate;
  selfDecisionEvents: readonly SelfDecisionProjectionSourceEvent[];
  activityEvents: readonly ActivityAutobiographicalSourceEvent[];
  episodes: readonly Episode[];
  countsSnapshot: JsonValue;
  maxSelfDecisionEvents: number;
  maxActivityEvents: number;
  maxEpisodes: number;
}): string {
  return [
    "I consolidate one closed UTC day of my own cross-session lived experience.",
    `I emit exactly one ${TOOL_NAME} tool call.`,
    "I write one durable first-person experiential gist: what the day felt like as an arc, what distinct events stood out, and how many repeated wakes or decisions collapsed into a routine pattern.",
    "I preserve provided counts as counts. I do not invent counts.",
    "I do not decide whether any wake, silence, or decision should have happened, succeeded, failed, or deserved approval. I only narrate what the day was like from the evidence.",
    "I do not quote other-audience message text. Activity rows are structural context only.",
    `${SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE} I apply this to the gist.`,
    "I only cite episode IDs and source stream entry IDs from the provided candidates.",
    "Day:",
    JSON.stringify({
      utc_day: input.candidate.utcDay,
      label: formatUtcDayBoundary(input.candidate.dayStartMs),
      day_start_ms: input.candidate.dayStartMs,
      day_end_ms: input.candidate.dayEndMs,
    }),
    "Counts snapshot:",
    JSON.stringify(input.countsSnapshot),
    `Self-private autonomous decision summaries, capped at ${input.maxSelfDecisionEvents}:`,
    ...input.selfDecisionEvents.map((event) =>
      JSON.stringify({
        occurred_at: event.occurredAt,
        trigger_name: event.triggerName,
        trigger_type: event.triggerType,
        decision_summary: event.decisionSummary,
        decision_rationale: event.decisionRationale,
        source_stream_entry_ids: event.sourceStreamEntryIds,
        ...memoryDisclosurePayloadFields(selfPrivateMemoryDisclosureLabel()),
      }),
    ),
    `Global activity rows without message text, capped at ${input.maxActivityEvents}:`,
    ...input.activityEvents.map((event) =>
      JSON.stringify({
        kind: event.kind,
        occurred_at: event.occurredAt,
        session_id: event.sessionId,
        session_source_type: event.sessionSourceType,
        session_audience_role: event.sessionAudienceRole,
        session_label: event.sessionLabel,
        participant_label: event.participantLabel,
        audience_entity_id: event.audienceEntityId,
        participant_entity_ids: event.participantEntityIds,
        source_stream_entry_ids: event.sourceStreamEntryIds,
      }),
    ),
    `Episode summaries, capped at ${input.maxEpisodes}:`,
    ...input.episodes.map((episode) =>
      JSON.stringify(
        episodeEvidencePromptRow(episode, {
          start_time: episode.start_time,
          end_time: episode.end_time,
          source_stream_ids: episode.source_stream_ids,
          significance: episode.significance,
          tags: episode.tags,
        }),
      ),
    ),
  ].join("\n");
}

function invalidSummaryResponse(error: unknown): unknown {
  if (isStructuredToolCallError(error, "missing_tool_call")) {
    return new StorageError(`Lived-experience summarizer did not emit tool ${TOOL_NAME}`, {
      code: "LIVED_EXPERIENCE_DAY_SUMMARIZER_INVALID",
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

function parseSummaryResponse(input: unknown) {
  return livedExperienceDaySummaryToolSchema.parse(input);
}

function buildChange(item: LivedExperienceDaySummarizerPlan["items"][number]): OfflineChange {
  return {
    process: PROCESS_NAME,
    action: "upsert_day_summary",
    targets: {
      summary_id: item.summary.id,
      self_entity_id: item.summary.self_entity_id,
      utc_day: item.summary.utc_day,
    },
    preview: {
      gist: item.summary.gist,
    },
  };
}

async function combinedDisclosureLabel(input: {
  ctx: OfflineContext;
  activityEvents: readonly ActivityAutobiographicalSourceEvent[];
  sourceEpisodeIds: readonly EpisodeId[];
}): Promise<MemoryDisclosureLabel> {
  const originAudienceIds = uniqueValues(
    input.activityEvents
      .map((event) => event.audienceEntityId)
      .filter((entityId): entityId is EntityId => entityId !== null),
  );
  const labels: MemoryDisclosureLabel[] = [selfPrivateMemoryDisclosureLabel(originAudienceIds)];

  if (input.sourceEpisodeIds.length > 0) {
    labels.push(
      await disclosureLabelForEpisodeIds(input.ctx.episodicRepository, input.sourceEpisodeIds),
    );
  }

  return labels.length === 0
    ? unknownMemoryDisclosureLabel(originAudienceIds)
    : combineMemoryDisclosureLabels(labels);
}

function candidateSourceStreamIds(input: {
  selfDecisionEvents: readonly SelfDecisionProjectionSourceEvent[];
  activityEvents: readonly ActivityAutobiographicalSourceEvent[];
  episodes: readonly Episode[];
}): StreamEntryId[] {
  return uniqueValues([
    ...input.selfDecisionEvents.flatMap((event) => event.sourceStreamEntryIds),
    ...input.activityEvents.flatMap((event) => event.sourceStreamEntryIds),
    ...sourceStreamEntryIdsFromEpisodes(input.episodes),
  ]);
}

export type LivedExperienceDaySummarizerProcessOptions = {
  livedExperienceDaySummaryRepository: OfflineContext["livedExperienceDaySummaryRepository"];
  registry: ReverserRegistry;
};

export class LivedExperienceDaySummarizerProcess implements OfflineProcess<LivedExperienceDaySummarizerPlan> {
  readonly name = PROCESS_NAME;

  constructor(private readonly options: LivedExperienceDaySummarizerProcessOptions) {
    this.options.registry.register(this.name, "upsert_day_summary", async ({ reversal }) => {
      const parsed = reversal as Partial<LivedExperienceDaySummarizerReversal>;

      if (parsed.previous !== undefined) {
        this.options.livedExperienceDaySummaryRepository.upsert({
          id: parsed.previous.id,
          selfEntityId: parsed.previous.self_entity_id,
          utcDay: parsed.previous.utc_day,
          dayStartMs: parsed.previous.day_start_ms,
          dayEndMs: parsed.previous.day_end_ms,
          gist: parsed.previous.gist,
          salience: parsed.previous.salience,
          countsSnapshot: parsed.previous.counts_snapshot,
          sourceEpisodeIds: parsed.previous.source_episode_ids,
          sourceStreamEntryIds: parsed.previous.source_stream_entry_ids,
          disclosureLabel: parsed.previous.disclosure_label,
          provenance: parsed.previous.provenance,
          sourceRunId: parsed.previous.source_run_id,
          createdAt: parsed.previous.created_at,
          updatedAt: parsed.previous.updated_at,
        });
        return;
      }

      const summaryId = parseSummaryId(parsed.summary_id);

      if (summaryId !== null) {
        this.options.livedExperienceDaySummaryRepository.delete(summaryId);
      }
    });
  }

  async plan(
    ctx: OfflineContext,
    opts: { budget?: number; params?: Record<string, unknown> } = {},
  ): Promise<LivedExperienceDaySummarizerPlan> {
    const errors: OfflineProcessError[] = [];
    const items: LivedExperienceDaySummarizerPlan["items"] = [];
    const config = ctx.config.offline.livedExperienceDaySummarizer;
    const budget = opts.budget ?? config.budget;
    const nowMs = ctx.clock.now();
    const currentOpenDayStartMs = utcDayStartMs(nowMs);
    const windowDays =
      typeof opts.params?.windowDays === "number" ? opts.params.windowDays : config.windowDays;
    const maxDaysPerRun =
      typeof opts.params?.maxDaysPerRun === "number"
        ? opts.params.maxDaysPerRun
        : config.maxDaysPerRun;
    const sinceMs = currentOpenDayStartMs - Math.max(1, Math.floor(windowDays)) * DAY_MS;
    const untilMs = currentOpenDayStartMs - 1;
    const selfEntity = ctx.entityRepository.getSelf();
    let tokensUsed = 0;
    let budgetExhausted = false;

    if (selfEntity === null) {
      return livedExperienceDaySummarizerPlanSchema.parse({
        process: this.name,
        items,
        errors: [
          {
            process: this.name,
            message: "Lived-experience day summarizer could not find a self entity.",
            code: "LIVED_EXPERIENCE_DAY_SUMMARIZER_NO_SELF_ENTITY",
          },
        ],
        tokens_used: 0,
        budget_exhausted: false,
      });
    }

    const existingSummaries = ctx.livedExperienceDaySummaryRepository.listForWindow({
      selfEntityId: selfEntity.id,
      fromMs: sinceMs,
      toMs: untilMs,
      limit: Math.max(1, Math.floor(windowDays)) + 1,
    });
    const activityDensityRows = ctx.activityRepository.listDailyGlobalActiveDensity({
      sinceMs,
      untilMs,
      limit: Math.max(1, Math.floor(windowDays)) + 1,
    });
    const selfDecisionDensityRows =
      ctx.selfDecisionRepository.listDailyAutonomousSelfPrivateDensity({
        sinceMs,
        untilMs,
        limit: Math.max(1, Math.floor(windowDays)) + 1,
      });
    const candidates = buildCandidateDays({
      activityDensityRows,
      selfDecisionDensityRows,
      existingSummaries,
      currentOpenDayStartMs,
      maxDays: Math.max(1, Math.floor(maxDaysPerRun)),
    });

    if (candidates.length === 0) {
      return livedExperienceDaySummarizerPlanSchema.parse({
        process: this.name,
        items,
        errors,
        tokens_used: 0,
        budget_exhausted: false,
      });
    }

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient: LLMClient = wrapClient(ctx.llm.background);

        for (const candidate of candidates) {
          try {
            const selfDecisionEvents = ctx.selfDecisionRepository.listAutonomousSelfPrivateForRange(
              {
                sinceMs: candidate.dayStartMs,
                untilMs: candidate.dayEndMs,
                limit: config.maxSelfDecisionEventsPerDay,
              },
            );
            const activityEvents = ctx.activityRepository.listRecentGlobalEvents({
              sinceMs: candidate.dayStartMs,
              untilMs: candidate.dayEndMs,
              limit: config.maxActivityEventsPerDay,
            });
            const episodes = (
              await ctx.episodicRepository.recallByTimeRangeForCognition(
                {
                  start: candidate.dayStartMs,
                  end: candidate.dayEndMs,
                },
                { limit: config.maxEpisodesPerDay },
              )
            ).map((candidateEpisode) => candidateEpisode.episode);
            const countsSnapshot = densityCountsSnapshot(candidate);
            const response = (
              await callStructuredTool({
                llmClient,
                request: {
                  model: ctx.config.anthropic.models.background,
                  system: [
                    "I consolidate my own closed-day lived experience from self-private decision summaries, structural activity counts, and summarized episode evidence.",
                    "I produce experiential narrative only. I do not judge output quality or decide whether any wake, silence, or decision was justified.",
                    SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE,
                  ].join("\n"),
                  messages: [
                    {
                      role: "user",
                      content: buildDayPrompt({
                        candidate,
                        selfDecisionEvents,
                        activityEvents,
                        episodes,
                        countsSnapshot,
                        maxSelfDecisionEvents: config.maxSelfDecisionEventsPerDay,
                        maxActivityEvents: config.maxActivityEventsPerDay,
                        maxEpisodes: config.maxEpisodesPerDay,
                      }),
                    },
                  ],
                  tools: [LIVED_EXPERIENCE_DAY_SUMMARY_TOOL],
                  tool_choice: { type: "tool", name: TOOL_NAME },
                  max_tokens: 4_000,
                  budget: "offline-lived-experience-day-summarizer",
                },
                toolName: TOOL_NAME,
                parse: parseSummaryResponse,
              })
            ).parsed;
            const allowedEpisodeIds = new Set(episodes.map((episode) => episode.id));
            const allowedStreamEntryIds = new Set(
              candidateSourceStreamIds({
                selfDecisionEvents,
                activityEvents,
                episodes,
              }),
            );

            if (response.utc_day !== candidate.utcDay) {
              throw new StorageError("Lived-experience summary returned the wrong UTC day", {
                code: "LIVED_EXPERIENCE_DAY_SUMMARIZER_INVALID_REF",
              });
            }

            for (const episodeId of response.cited_episode_ids) {
              if (!allowedEpisodeIds.has(episodeId)) {
                throw new StorageError(
                  "Lived-experience summary referenced episodes outside candidates",
                  {
                    code: "LIVED_EXPERIENCE_DAY_SUMMARIZER_INVALID_REF",
                  },
                );
              }
            }

            for (const sourceStreamEntryId of response.cited_source_stream_entry_ids) {
              if (!allowedStreamEntryIds.has(sourceStreamEntryId)) {
                throw new StorageError(
                  "Lived-experience summary referenced source stream entries outside candidates",
                  {
                    code: "LIVED_EXPERIENCE_DAY_SUMMARIZER_INVALID_REF",
                  },
                );
              }
            }

            const citedEpisodeIds = response.cited_episode_ids;
            const sourceEpisodeIds = uniqueValues(
              citedEpisodeIds.length > 0 ? citedEpisodeIds : episodes.map((episode) => episode.id),
            );
            const citedSourceStreamEntryIds = response.cited_source_stream_entry_ids;
            const sourceStreamEntryIds = uniqueValues(
              citedSourceStreamEntryIds.length > 0
                ? citedSourceStreamEntryIds
                : [...allowedStreamEntryIds],
            );
            const disclosureLabel = await combinedDisclosureLabel({
              ctx,
              activityEvents,
              sourceEpisodeIds,
            });
            const previous = ctx.livedExperienceDaySummaryRepository.getByDay(
              selfEntity.id,
              candidate.utcDay,
            );
            const summary = livedExperienceDaySummarySchema.parse({
              id: previous?.id ?? createLivedExperienceDaySummaryId(),
              self_entity_id: selfEntity.id,
              utc_day: candidate.utcDay,
              day_start_ms: candidate.dayStartMs,
              day_end_ms: candidate.dayEndMs,
              gist: response.gist.trim(),
              salience: response.salience,
              counts_snapshot: countsSnapshot,
              source_episode_ids: sourceEpisodeIds,
              source_stream_entry_ids: sourceStreamEntryIds,
              disclosure_label: disclosureLabel,
              provenance: {
                kind: "offline",
                process: this.name,
              },
              source_run_id: ctx.runId,
              created_at: previous?.created_at ?? nowMs,
              updated_at: nowMs,
            });

            items.push({
              action: "upsert_day_summary",
              summary,
              previous,
            });
          } catch (error) {
            if (error instanceof BudgetExceededError) {
              throw error;
            }

            errors.push(offlineProcessError(this.name, invalidSummaryResponse(error)));
          }
        }
      });

      tokensUsed = budgeted.tokens_used;
    } catch (error) {
      tokensUsed = getBudgetErrorTokens(error);
      budgetExhausted = error instanceof BudgetExceededError;
      errors.push(offlineProcessError(this.name, error));
    }

    return livedExperienceDaySummarizerPlanSchema.parse({
      process: this.name,
      items,
      errors,
      tokens_used: tokensUsed,
      budget_exhausted: budgetExhausted,
    });
  }

  preview(plan: LivedExperienceDaySummarizerPlan): OfflineResult {
    const parsed = livedExperienceDaySummarizerPlanSchema.parse(plan);

    return {
      process: this.name,
      dryRun: true,
      changes: parsed.items.map((item) => buildChange(item)),
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
    };
  }

  async apply(
    ctx: OfflineContext,
    rawPlan: LivedExperienceDaySummarizerPlan,
  ): Promise<OfflineResult> {
    const plan = livedExperienceDaySummarizerPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];

    for (const item of plan.items) {
      const stored = ctx.livedExperienceDaySummaryRepository.upsert({
        id: item.summary.id,
        selfEntityId: item.summary.self_entity_id,
        utcDay: item.summary.utc_day,
        dayStartMs: item.summary.day_start_ms,
        dayEndMs: item.summary.day_end_ms,
        gist: item.summary.gist,
        salience: item.summary.salience,
        countsSnapshot: item.summary.counts_snapshot,
        sourceEpisodeIds: item.summary.source_episode_ids,
        sourceStreamEntryIds: item.summary.source_stream_entry_ids,
        disclosureLabel: item.summary.disclosure_label,
        provenance: item.summary.provenance,
        sourceRunId: item.summary.source_run_id,
        createdAt: item.summary.created_at,
        updatedAt: item.summary.updated_at,
      });

      ctx.auditLog.record({
        run_id: ctx.runId,
        process: this.name,
        action: "upsert_day_summary",
        targets: {
          summary_id: stored.id,
          self_entity_id: stored.self_entity_id,
          utc_day: stored.utc_day,
        },
        reversal:
          item.previous === null
            ? ({
                summary_id: stored.id,
              } satisfies LivedExperienceDaySummarizerReversal)
            : ({
                previous: item.previous,
              } satisfies LivedExperienceDaySummarizerReversal),
      });
      changes.push(buildChange({ ...item, summary: stored }));
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
