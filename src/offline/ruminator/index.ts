import { z } from "zod";

import { computeWeights } from "../../cognition/attention/index.js";
import type { LLMClient } from "../../llm/index.js";
import { episodeIdSchema } from "../../memory/episodic/index.js";
import {
  growthMarkerCategorySchema,
  growthMarkerIdSchema,
  growthMarkerSchema,
  openQuestionIdSchema,
  openQuestionSchema,
  type OpenQuestion,
} from "../../memory/self/index.js";
import { createOpenQuestionReopener } from "../../memory/self/open-questions.js";
import { createGrowthMarkerId } from "../../util/ids.js";
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

const resolutionResponseSchema = z.object({
  resolution_note: z.string().min(1),
  growth_marker: z
    .object({
      what_changed: z.string().min(1),
      before_description: z.string().nullable().optional(),
      after_description: z.string().nullable().optional(),
      confidence: z.number().min(0).max(1),
      category: growthMarkerCategorySchema.default("understanding"),
    })
    .nullable()
    .default(null),
});

const serializableGrowthMarkerSchema = growthMarkerSchema.extend({
  evidence_episode_ids: z.array(episodeIdSchema).min(1),
});

const ruminatorPlanItemSchema = z.discriminatedUnion("action", [
  z.object({
    action: z.literal("resolve"),
    question_id: openQuestionIdSchema,
    previous: openQuestionSchema,
    resolution_episode_id: episodeIdSchema,
    resolution_note: z.string().min(1),
    growth_marker: serializableGrowthMarkerSchema.nullable(),
  }),
  z.object({
    action: z.literal("bump_urgency"),
    question_id: openQuestionIdSchema,
    previous: openQuestionSchema,
    delta: z.number().finite(),
    next_urgency: z.number().min(0).max(1),
  }),
  z.object({
    action: z.literal("abandon"),
    question_id: openQuestionIdSchema,
    previous: openQuestionSchema,
    reason: z.string().min(1),
  }),
]);

export const ruminatorPlanSchema = z.object({
  process: z.literal("ruminator"),
  items: z.array(ruminatorPlanItemSchema),
  errors: z
    .array(
      z.object({
        process: z.literal("ruminator"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
});

export type RuminatorPlan = z.infer<typeof ruminatorPlanSchema>;

type RuminatorReversal = {
  previous?: OpenQuestion;
  marker_id?: z.infer<typeof growthMarkerIdSchema>;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function buildResolutionPrompt(question: OpenQuestion, evidence: string): string {
  return [
    "Resolve the open question using only the evidence below.",
    'Return strict JSON with shape {"resolution_note":"...","growth_marker":{"what_changed":"...","before_description":"...","after_description":"...","confidence":0.0,"category":"understanding"}|null} and no surrounding prose.',
    "Only include a growth_marker if the evidence clearly shows new understanding.",
    `Question: ${question.question}`,
    `Source: ${question.source}`,
    "Evidence:",
    evidence,
  ].join("\n\n");
}

function parseResolutionResponse(text: string) {
  let raw: unknown;

  try {
    raw = JSON.parse(text) as unknown;
  } catch (error) {
    throw new StorageError("Ruminator returned non-JSON output", {
      cause: error,
      code: "RUMINATOR_INVALID",
    });
  }

  return resolutionResponseSchema.parse(raw);
}

function buildChange(item: RuminatorPlan["items"][number]): OfflineChange {
  if (item.action === "resolve") {
    return {
      process: "ruminator",
      action: "resolve",
      targets: {
        question_id: item.question_id,
        resolution_episode_id: item.resolution_episode_id,
      },
      preview: {
        note: item.resolution_note,
        growth_marker: item.growth_marker?.what_changed ?? null,
      },
    };
  }

  if (item.action === "abandon") {
    return {
      process: "ruminator",
      action: "abandon",
      targets: {
        question_id: item.question_id,
      },
      preview: {
        reason: item.reason,
      },
    };
  }

  return {
    process: "ruminator",
    action: "bump_urgency",
    targets: {
      question_id: item.question_id,
    },
    preview: {
      delta: item.delta,
      next_urgency: item.next_urgency,
    },
  };
}

function buildReflectionWeights(ctx: OfflineContext) {
  return computeWeights("reflective", {
    currentGoals: ctx.goalsRepository.list({ status: "active" }),
    hasTemporalCue: false,
  });
}

async function planResolution(
  ctx: OfflineContext,
  llmClient: LLMClient,
  question: OpenQuestion,
  maxQuestionsPerRun: number,
): Promise<RuminatorPlan["items"][number] | null> {
  const retrieval = await ctx.retrievalPipeline.searchWithContext(question.question, {
    limit: Math.max(3, maxQuestionsPerRun),
    attentionWeights: buildReflectionWeights(ctx),
    goalDescriptions: ctx.goalsRepository
      .list({ status: "active" })
      .map((goal) => goal.description),
    includeOpenQuestions: false,
  });
  const strongEvidence = retrieval.episodes
    .filter(
      (result) =>
        result.score >= ctx.config.offline.ruminator.resolveConfidenceThreshold &&
        result.episode.updated_at > question.last_touched,
    )
    .sort(
      (left, right) =>
        right.score - left.score || right.episode.updated_at - left.episode.updated_at,
    )[0];

  if (strongEvidence === undefined) {
    return null;
  }

  const evidenceBlock = retrieval.episodes
    .slice(0, 3)
    .map((result) =>
      JSON.stringify({
        id: result.episode.id,
        title: result.episode.title,
        narrative: result.episode.narrative,
        tags: result.episode.tags,
        score: Number(result.score.toFixed(3)),
      }),
    )
    .join("\n");
  const response = parseResolutionResponse(
    (
      await llmClient.complete({
        model: ctx.config.anthropic.models.background,
        system: "You update Borg's open questions conservatively and only from grounded evidence.",
        messages: [
          {
            role: "user",
            content: buildResolutionPrompt(question, evidenceBlock),
          },
        ],
        max_tokens: 300,
        budget: "offline-ruminator",
      })
    ).text,
  );
  const growthMarker =
    response.growth_marker === null
      ? null
      : serializableGrowthMarkerSchema.parse({
          id: createGrowthMarkerId(),
          ts: ctx.clock.now(),
          category: response.growth_marker.category,
          what_changed: response.growth_marker.what_changed,
          before_description: response.growth_marker.before_description ?? null,
          after_description: response.growth_marker.after_description ?? null,
          evidence_episode_ids: [strongEvidence.episode.id],
          confidence: Math.min(GROWTH_MARKER_CONFIDENCE_CEILING, response.growth_marker.confidence),
          source_process: "ruminator",
          created_at: ctx.clock.now(),
        });

  return {
    action: "resolve",
    question_id: question.id,
    previous: question,
    resolution_episode_id: strongEvidence.episode.id,
    resolution_note: response.resolution_note.trim(),
    growth_marker: growthMarker,
  };
}

function planFallbackAction(
  ctx: OfflineContext,
  question: OpenQuestion,
): RuminatorPlan["items"][number] | null {
  const ageMs = Math.max(0, ctx.clock.now() - question.last_touched);

  if (ageMs >= ctx.config.offline.ruminator.stalenessDays * DAY_MS && question.urgency < 0.2) {
    return {
      action: "abandon",
      question_id: question.id,
      previous: question,
      reason: "No relevant new evidence surfaced before the staleness threshold.",
    };
  }

  if (ageMs >= 7 * DAY_MS) {
    const nextUrgency = clamp(question.urgency + 0.05, 0, 1);

    if (nextUrgency > question.urgency) {
      return {
        action: "bump_urgency",
        question_id: question.id,
        previous: question,
        delta: Number((nextUrgency - question.urgency).toFixed(3)),
        next_urgency: nextUrgency,
      };
    }
  }

  return null;
}

export type RuminatorProcessOptions = {
  openQuestionsRepository: OfflineContext["openQuestionsRepository"];
  growthMarkersRepository: OfflineContext["growthMarkersRepository"];
  registry: ReverserRegistry;
};

export class RuminatorProcess implements OfflineProcess<RuminatorPlan> {
  readonly name = "ruminator" as const;

  constructor(private readonly options: RuminatorProcessOptions) {
    const reopenForReversal = createOpenQuestionReopener(this.options.openQuestionsRepository);

    this.options.registry.register(this.name, "resolve", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.previous !== undefined) {
        reopenForReversal(parsed.previous.id, parsed.previous.urgency);
      }
    });
    this.options.registry.register(this.name, "bump_urgency", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.previous !== undefined) {
        this.options.openQuestionsRepository.setUrgency(
          parsed.previous.id,
          parsed.previous.urgency,
        );
      }
    });
    this.options.registry.register(this.name, "abandon", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.previous !== undefined) {
        reopenForReversal(parsed.previous.id, parsed.previous.urgency);
      }
    });
    this.options.registry.register(this.name, "add_growth_marker", async ({ reversal }) => {
      const parsed = reversal as Partial<RuminatorReversal>;

      if (parsed.marker_id !== undefined) {
        this.options.growthMarkersRepository.delete(parsed.marker_id);
      }
    });
  }

  async plan(
    ctx: OfflineContext,
    opts: { budget?: number; params?: Record<string, unknown> } = {},
  ) {
    const errors: OfflineProcessError[] = [];
    const items: RuminatorPlan["items"] = [];
    const budget = opts.budget ?? ctx.config.offline.ruminator.budget;
    const maxQuestionsRaw = opts.params?.maxQuestionsPerRun;
    const maxQuestionsPerRun =
      typeof maxQuestionsRaw === "number" &&
      Number.isInteger(maxQuestionsRaw) &&
      maxQuestionsRaw > 0
        ? maxQuestionsRaw
        : ctx.config.offline.ruminator.maxQuestionsPerRun;
    const questions = ctx.openQuestionsRepository.list({
      status: "open",
      limit: maxQuestionsPerRun,
    });
    let tokensUsed = 0;
    let budgetExhausted = false;

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient = wrapClient(ctx.llm.background);

        for (const question of questions) {
          try {
            const resolution = await planResolution(ctx, llmClient, question, maxQuestionsPerRun);

            if (resolution !== null) {
              items.push(resolution);
              continue;
            }

            const fallback = planFallbackAction(ctx, question);

            if (fallback !== null) {
              items.push(fallback);
            }
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

    return ruminatorPlanSchema.parse({
      process: this.name,
      items,
      errors,
      tokens_used: tokensUsed,
      budget_exhausted: budgetExhausted,
    });
  }

  preview(plan: RuminatorPlan): OfflineResult {
    const parsed = ruminatorPlanSchema.parse(plan);

    return {
      process: this.name,
      dryRun: true,
      changes: parsed.items.map((item) => buildChange(item)),
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
    };
  }

  async apply(ctx: OfflineContext, rawPlan: RuminatorPlan): Promise<OfflineResult> {
    const plan = ruminatorPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];

    for (const item of plan.items) {
      if (item.action === "resolve") {
        const current = ctx.openQuestionsRepository.get(item.question_id);

        if (current === null) {
          throw new StorageError(`Missing open question for ruminator plan: ${item.question_id}`, {
            code: "RUMINATOR_PLAN_INVALID",
          });
        }

        if (
          current.status !== "resolved" ||
          current.resolution_episode_id !== item.resolution_episode_id ||
          current.resolution_note !== item.resolution_note
        ) {
          ctx.openQuestionsRepository.resolve(item.question_id, {
            resolution_episode_id: item.resolution_episode_id,
            resolution_note: item.resolution_note,
          });
        }

        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "resolve",
          targets: {
            question_id: item.question_id,
            resolution_episode_id: item.resolution_episode_id,
          },
          reversal: {
            previous: item.previous,
          } satisfies RuminatorReversal,
        });

        if (
          item.growth_marker !== null &&
          ctx.growthMarkersRepository.get(item.growth_marker.id) === null
        ) {
          ctx.growthMarkersRepository.add(item.growth_marker);
          ctx.auditLog.record({
            run_id: ctx.runId,
            process: this.name,
            action: "add_growth_marker",
            targets: {
              marker_id: item.growth_marker.id,
              question_id: item.question_id,
            },
            reversal: {
              marker_id: item.growth_marker.id,
            } satisfies RuminatorReversal,
          });
        }

        changes.push(buildChange(item));
        continue;
      }

      if (item.action === "abandon") {
        const current = ctx.openQuestionsRepository.get(item.question_id);

        if (current === null) {
          throw new StorageError(`Missing open question for ruminator plan: ${item.question_id}`, {
            code: "RUMINATOR_PLAN_INVALID",
          });
        }

        if (current.status !== "abandoned" || current.abandoned_reason !== item.reason) {
          ctx.openQuestionsRepository.abandon(item.question_id, item.reason);
        }

        ctx.auditLog.record({
          run_id: ctx.runId,
          process: this.name,
          action: "abandon",
          targets: {
            question_id: item.question_id,
          },
          reversal: {
            previous: item.previous,
          } satisfies RuminatorReversal,
        });
        changes.push(buildChange(item));
        continue;
      }

      const current = ctx.openQuestionsRepository.get(item.question_id);

      if (current === null) {
        throw new StorageError(`Missing open question for ruminator plan: ${item.question_id}`, {
          code: "RUMINATOR_PLAN_INVALID",
        });
      }

      if (Math.abs(current.urgency - item.next_urgency) > 1e-6) {
        ctx.openQuestionsRepository.setUrgency(item.question_id, item.next_urgency);
      }

      ctx.auditLog.record({
        run_id: ctx.runId,
        process: this.name,
        action: "bump_urgency",
        targets: {
          question_id: item.question_id,
        },
        reversal: {
          previous: item.previous,
        } satisfies RuminatorReversal,
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
