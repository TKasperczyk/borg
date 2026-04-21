import { z } from "zod";

import type { LLMClient } from "../../llm/index.js";
import { episodeIdSchema, type Episode } from "../../memory/episodic/index.js";
import {
  reviewKindSchema,
  semanticNodeIdSchema,
  type SemanticNode,
} from "../../memory/semantic/index.js";
import { BudgetExceededError } from "../../util/errors.js";

import type { ReverserRegistry } from "../audit-log.js";
import { getBudgetErrorTokens, withBudget } from "../budget.js";
import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessError,
  OfflineResult,
} from "../types.js";

const overseerFlagKindSchema = z.enum([
  reviewKindSchema.enum.misattribution,
  reviewKindSchema.enum.temporal_drift,
  reviewKindSchema.enum.identity_inconsistency,
]);

const reviewFlagSchema = z.object({
  kind: overseerFlagKindSchema,
  reason: z.string().min(1),
  confidence: z.number().min(0).max(1),
});

const overseerResponseSchema = z.object({
  flags: z.array(reviewFlagSchema).default([]),
});

const HOUR_MS = 60 * 60 * 1_000;

const overseerTargetSchema = z.discriminatedUnion("target_type", [
  z.object({
    target_type: z.literal("episode"),
    target_id: episodeIdSchema,
  }),
  z.object({
    target_type: z.literal("semantic_node"),
    target_id: semanticNodeIdSchema,
  }),
]);

const overseerPlanItemBaseSchema = z.object({
  kind: overseerFlagKindSchema,
  reason: z.string().min(1),
  confidence: z.number().min(0).max(1),
});

const overseerPlanItemSchema = z
  .discriminatedUnion("target_type", [
    z.object({
      target_type: z.literal("episode"),
      target_id: episodeIdSchema,
    }),
    z.object({
      target_type: z.literal("semantic_node"),
      target_id: semanticNodeIdSchema,
    }),
  ])
  .and(overseerPlanItemBaseSchema);

export const overseerPlanSchema = z.object({
  process: z.literal("overseer"),
  items: z.array(overseerPlanItemSchema),
  errors: z
    .array(
      z.object({
        process: z.literal("overseer"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
});

export type OverseerPlan = z.infer<typeof overseerPlanSchema>;

type OverseerTarget =
  | {
      type: "episode";
      id: Episode["id"];
      created_at: number;
      content: Episode;
    }
  | {
      type: "semantic_node";
      id: SemanticNode["id"];
      created_at: number;
      content: SemanticNode;
    };

type OverseerReversal = {
  reviewItemId?: number;
};

function parseFlags(text: string) {
  let raw: unknown;

  try {
    raw = JSON.parse(text) as unknown;
  } catch (error) {
    throw new TypeError(`Overseer returned non-JSON output: ${String(error)}`);
  }

  return overseerResponseSchema.parse(raw);
}

function summarizeSelfState(ctx: OfflineContext): string {
  const values =
    ctx.valuesRepository
      .list()
      .map((value) => value.label)
      .join(", ") || "none";
  const goals =
    ctx.goalsRepository
      .list({ status: "active" })
      .map((goal) => goal.description)
      .join(" | ") || "none";
  const traits =
    ctx.traitsRepository
      .list()
      .map((trait) => `${trait.label}:${trait.strength.toFixed(2)}`)
      .join(", ") || "none";

  return `Values: ${values}\nGoals: ${goals}\nTraits: ${traits}`;
}

function buildPrompt(target: OverseerTarget, ctx: OfflineContext): string {
  return [
    "Check the memory item for misattribution, temporal drift, and identity inconsistency.",
    'Return strict JSON with shape {"flags":[{"kind":"misattribution|temporal_drift|identity_inconsistency","reason":"...","confidence":0.0}]} and no surrounding prose.',
    summarizeSelfState(ctx),
    "Memory item:",
    JSON.stringify({
      type: target.type,
      content: target.content,
    }),
  ].join("\n\n");
}

async function collectTargets(ctx: OfflineContext): Promise<OverseerTarget[]> {
  const [episodes, nodes] = await Promise.all([
    ctx.episodicRepository.listAll(),
    ctx.semanticNodeRepository.list({
      includeArchived: true,
      limit: 200,
    }),
  ]);

  return [
    ...episodes.map(
      (episode) =>
        ({
          type: "episode",
          id: episode.id,
          created_at: episode.created_at,
          content: episode,
        }) satisfies OverseerTarget,
    ),
    ...nodes.map(
      (node) =>
        ({
          type: "semantic_node",
          id: node.id,
          created_at: node.created_at,
          content: node,
        }) satisfies OverseerTarget,
    ),
  ];
}

function computeSinceTimestamp(ctx: OfflineContext): number {
  const priorAuditTs = ctx.auditLog.list({ process: "overseer" })[0]?.applied_at ?? 0;
  const lookbackTs = ctx.clock.now() - ctx.config.offline.overseer.lookbackHours * HOUR_MS;

  return Math.max(priorAuditTs, lookbackTs);
}

function buildChange(item: OverseerPlan["items"][number]): OfflineChange {
  return {
    process: "overseer",
    action: "flag",
    targets: {
      kind: item.kind,
      target_type: item.target_type,
      target_id: item.target_id,
    },
    preview: {
      reason: item.reason,
      confidence: item.confidence,
    },
  };
}

export type OverseerProcessOptions = {
  reviewQueueRepository: OfflineContext["reviewQueueRepository"];
  registry: ReverserRegistry;
};

export class OverseerProcess implements OfflineProcess<OverseerPlan> {
  readonly name = "overseer" as const;

  constructor(private readonly options: OverseerProcessOptions) {
    this.options.registry.register(this.name, "flag", async ({ reversal }) => {
      const parsed = reversal as Partial<OverseerReversal>;

      if (typeof parsed.reviewItemId === "number") {
        this.options.reviewQueueRepository.delete(parsed.reviewItemId);
      }
    });
  }

  async plan(ctx: OfflineContext, opts: { budget?: number } = {}): Promise<OverseerPlan> {
    const errors: OfflineProcessError[] = [];
    const items: OverseerPlan["items"] = [];
    const budget = opts.budget ?? ctx.config.offline.overseer.budget;
    const sinceTs = computeSinceTimestamp(ctx);
    const targets = (await collectTargets(ctx))
      .filter((target) => target.created_at >= sinceTs)
      .sort((left, right) => right.created_at - left.created_at)
      .slice(0, ctx.config.offline.overseer.maxChecksPerRun);
    let tokensUsed = 0;
    let budgetExhausted = false;

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient: LLMClient = wrapClient(ctx.llm.cognition);

        for (const target of targets) {
          try {
            const flags = parseFlags(
              (
                await llmClient.complete({
                  model: ctx.config.anthropic.models.cognition,
                  system:
                    "You audit recently formed memories. Flag only grounded QA concerns and keep false positives low.",
                  messages: [
                    {
                      role: "user",
                      content: buildPrompt(target, ctx),
                    },
                  ],
                  max_tokens: 500,
                  budget: "offline-overseer",
                })
              ).text,
            ).flags.filter((flag) => flag.confidence >= 0.5);

            for (const flag of flags) {
              if (target.type === "episode") {
                items.push({
                  target_type: "episode",
                  target_id: target.id,
                  kind: flag.kind,
                  reason: flag.reason,
                  confidence: flag.confidence,
                });
              } else {
                items.push({
                  target_type: "semantic_node",
                  target_id: target.id,
                  kind: flag.kind,
                  reason: flag.reason,
                  confidence: flag.confidence,
                });
              }
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

    return overseerPlanSchema.parse({
      process: this.name,
      items,
      errors,
      tokens_used: tokensUsed,
      budget_exhausted: budgetExhausted,
    });
  }

  preview(plan: OverseerPlan): OfflineResult {
    const parsed = overseerPlanSchema.parse(plan);

    return {
      process: this.name,
      dryRun: true,
      changes: parsed.items.map((item) => buildChange(item)),
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
    };
  }

  async apply(ctx: OfflineContext, rawPlan: OverseerPlan): Promise<OfflineResult> {
    const plan = overseerPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];

    for (const item of plan.items) {
      const reviewItem = ctx.reviewQueueRepository.enqueue({
        kind: item.kind,
        refs: {
          target_type: item.target_type,
          target_id: item.target_id,
        },
        reason: item.reason,
      });

      ctx.auditLog.record({
        run_id: ctx.runId,
        process: this.name,
        action: "flag",
        targets: {
          kind: item.kind,
          target_type: item.target_type,
          target_id: item.target_id,
        },
        reversal: {
          reviewItemId: reviewItem.id,
        } satisfies OverseerReversal,
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
