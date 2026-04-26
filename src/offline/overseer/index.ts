import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { episodeIdSchema, type Episode } from "../../memory/episodic/index.js";
import {
  reviewKindSchema,
  semanticEdgeIdSchema,
  semanticNodeIdSchema,
  type SemanticEdge,
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
  patch: z.record(z.string(), z.unknown()).optional(),
  corrected_start_time: z.number().finite().optional(),
  corrected_end_time: z.number().finite().optional(),
  patch_description: z.string().min(1).optional(),
  repair_target_type: z
    .enum(["trait", "value", "commitment", "goal", "autobiographical_period"])
    .optional(),
  repair_target_id: z.string().min(1).optional(),
  repair_op: z.enum(["reinforce", "contradict", "patch"]).optional(),
  evidence_episode_ids: z.array(z.string().min(1)).optional(),
  suggested_valid_to: z.number().finite().optional(),
  by_edge_id: semanticEdgeIdSchema.optional(),
});

const overseerResponseSchema = z.object({
  flags: z.array(reviewFlagSchema),
});
const OVERSEER_TOOL_NAME = "EmitOverseerFlags";
export const OVERSEER_TOOL = {
  name: OVERSEER_TOOL_NAME,
  description: "Emit grounded overseer review flags for a memory item.",
  inputSchema: toToolInputSchema(overseerResponseSchema),
} satisfies LLMToolDefinition;

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
  z.object({
    target_type: z.literal("semantic_edge"),
    target_id: semanticEdgeIdSchema,
  }),
]);

const overseerPlanItemBaseSchema = z.object({
  kind: overseerFlagKindSchema,
  reason: z.string().min(1),
  confidence: z.number().min(0).max(1),
  patch: z.record(z.string(), z.unknown()).optional(),
  corrected_start_time: z.number().finite().optional(),
  corrected_end_time: z.number().finite().optional(),
  patch_description: z.string().min(1).optional(),
  repair_target_type: z
    .enum(["trait", "value", "commitment", "goal", "autobiographical_period"])
    .optional(),
  repair_target_id: z.string().min(1).optional(),
  repair_op: z.enum(["reinforce", "contradict", "patch"]).optional(),
  evidence_episode_ids: z.array(z.string().min(1)).optional(),
  suggested_valid_to: z.number().finite().optional(),
  by_edge_id: semanticEdgeIdSchema.optional(),
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
    z.object({
      target_type: z.literal("semantic_edge"),
      target_id: semanticEdgeIdSchema,
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
    }
  | {
      type: "semantic_edge";
      id: SemanticEdge["id"];
      created_at: number;
      content: SemanticEdge;
    };

type OverseerReversal = {
  reviewItemId?: number;
};

function parseFlags(result: LLMCompleteResult) {
  const call = result.tool_calls.find((toolCall) => toolCall.name === OVERSEER_TOOL_NAME);

  if (call === undefined) {
    throw new TypeError(`Overseer did not emit tool ${OVERSEER_TOOL_NAME}`);
  }

  return overseerResponseSchema.parse(call.input);
}

function summarizeSelfState(ctx: OfflineContext): string {
  const values =
    ctx.valuesRepository
      .list()
      .map((value) => `${value.id}:${value.label}`)
      .join(", ") || "none";
  const goals =
    ctx.goalsRepository
      .list({ status: "active" })
      .map((goal) => `${goal.id}:${goal.description}`)
      .join(" | ") || "none";
  const traits =
    ctx.traitsRepository
      .list()
      .map((trait) => `${trait.id}:${trait.label}:${trait.strength.toFixed(2)}`)
      .join(", ") || "none";
  const commitments =
    ctx.commitmentRepository
      .list({ activeOnly: true })
      .map((commitment) => `${commitment.id}:${commitment.directive}`)
      .join(" | ") || "none";
  const currentPeriod = ctx.autobiographicalRepository.currentPeriod();

  return [
    `Values: ${values}`,
    `Goals: ${goals}`,
    `Traits: ${traits}`,
    `Commitments: ${commitments}`,
    `CurrentPeriod: ${currentPeriod === null ? "none" : `${currentPeriod.id}:${currentPeriod.label}`}`,
  ].join("\n");
}

function buildPrompt(target: OverseerTarget, ctx: OfflineContext): string {
  const serializedTarget =
    target.type === "episode"
      ? {
          type: target.type,
          content: {
            id: target.content.id,
            title: target.content.title,
            narrative: target.content.narrative,
            participants: target.content.participants,
            location: target.content.location,
            start_time: target.content.start_time,
            end_time: target.content.end_time,
            source_stream_ids: target.content.source_stream_ids,
            significance: target.content.significance,
            tags: target.content.tags,
            confidence: target.content.confidence,
            emotional_arc: target.content.emotional_arc,
          },
        }
      : target.type === "semantic_node"
        ? {
            type: target.type,
            content: {
              id: target.content.id,
              kind: target.content.kind,
              label: target.content.label,
              description: target.content.description,
              aliases: target.content.aliases,
              confidence: target.content.confidence,
              source_episode_ids: target.content.source_episode_ids,
              archived: target.content.archived,
              superseded_by: target.content.superseded_by,
            },
          }
        : {
            type: target.type,
            content: {
              id: target.content.id,
              from_node_id: target.content.from_node_id,
              to_node_id: target.content.to_node_id,
              relation: target.content.relation,
              confidence: target.content.confidence,
              evidence_episode_ids: target.content.evidence_episode_ids,
              created_at: target.content.created_at,
              last_verified_at: target.content.last_verified_at,
              valid_from: target.content.valid_from,
              valid_to: target.content.valid_to,
              invalidated_at: target.content.invalidated_at,
              invalidated_by_edge_id: target.content.invalidated_by_edge_id,
              invalidated_by_review_id: target.content.invalidated_by_review_id,
              invalidated_by_process: target.content.invalidated_by_process,
              invalidated_reason: target.content.invalidated_reason,
            },
          };

  return [
    "Check the memory item for misattribution, temporal drift, and identity inconsistency.",
    "If you flag an issue, include the concrete repair payload needed to fix it.",
    "For misattribution, provide patch fields that directly correct the target memory.",
    "For temporal drift, provide corrected timestamps and/or a replacement description.",
    "For semantic_edge temporal drift or identity inconsistency, provide suggested_valid_to and optional by_edge_id; only flag edges that should be reviewed for closure.",
    "For identity inconsistency, target a specific value, goal, trait, commitment, or autobiographical period by id and propose reinforce, contradict, or patch.",
    `Emit your result by calling the ${OVERSEER_TOOL_NAME} tool exactly once.`,
    summarizeSelfState(ctx),
    "Memory item:",
    JSON.stringify(serializedTarget),
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
  const edges = ctx.semanticEdgeRepository.listEdges();

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
    ...edges.map(
      (edge) =>
        ({
          type: "semantic_edge",
          id: edge.id,
          created_at: edge.created_at,
          content: edge,
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
      ...(item.patch === undefined ? {} : { patch: item.patch }),
      ...(item.patch_description === undefined
        ? {}
        : { patch_description: item.patch_description }),
      ...(item.suggested_valid_to === undefined
        ? {}
        : { suggested_valid_to: item.suggested_valid_to }),
      ...(item.by_edge_id === undefined ? {} : { by_edge_id: item.by_edge_id }),
      ...(item.repair_target_type === undefined
        ? {}
        : {
            repair_target_type: item.repair_target_type,
            repair_target_id: item.repair_target_id,
            repair_op: item.repair_op,
          }),
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
        const llmClient: LLMClient = wrapClient(ctx.llm.background);

        for (const target of targets) {
          try {
            const flags = parseFlags(
              await llmClient.complete({
                model: ctx.config.anthropic.models.background,
                system:
                  "You audit recently formed memories. Flag only grounded QA concerns and keep false positives low.",
                messages: [
                  {
                    role: "user",
                    content: buildPrompt(target, ctx),
                  },
                ],
                tools: [OVERSEER_TOOL],
                tool_choice: { type: "tool", name: OVERSEER_TOOL_NAME },
                max_tokens: 4_000,
                budget: "offline-overseer",
              }),
            ).flags.filter((flag) => flag.confidence >= 0.5);

            for (const flag of flags) {
              if (target.type === "semantic_edge" && flag.kind === "misattribution") {
                continue;
              }

              const baseItem = {
                kind: flag.kind,
                reason: flag.reason,
                confidence: flag.confidence,
                ...(flag.patch === undefined ? {} : { patch: flag.patch }),
                ...(flag.corrected_start_time === undefined
                  ? {}
                  : { corrected_start_time: flag.corrected_start_time }),
                ...(flag.corrected_end_time === undefined
                  ? {}
                  : { corrected_end_time: flag.corrected_end_time }),
                ...(flag.patch_description === undefined
                  ? {}
                  : { patch_description: flag.patch_description }),
                ...(flag.suggested_valid_to === undefined
                  ? {}
                  : { suggested_valid_to: flag.suggested_valid_to }),
                ...(flag.by_edge_id === undefined ? {} : { by_edge_id: flag.by_edge_id }),
                ...(flag.repair_target_type === undefined
                  ? {}
                  : {
                      repair_target_type: flag.repair_target_type,
                      repair_target_id: flag.repair_target_id,
                      repair_op: flag.repair_op,
                    }),
                ...(flag.evidence_episode_ids === undefined
                  ? {}
                  : { evidence_episode_ids: flag.evidence_episode_ids }),
              };

              if (target.type === "episode") {
                items.push({
                  target_type: "episode",
                  target_id: target.id,
                  ...baseItem,
                });
              } else if (target.type === "semantic_node") {
                items.push({
                  target_type: "semantic_node",
                  target_id: target.id,
                  ...baseItem,
                });
              } else {
                items.push({
                  target_type: "semantic_edge",
                  target_id: target.id,
                  ...baseItem,
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
    const proposedProvenance = {
      kind: "offline" as const,
      process: this.name,
    };

    for (const item of plan.items) {
      const refs =
        item.kind === "identity_inconsistency"
          ? item.target_type === "semantic_edge"
            ? {
                target_type: "semantic_edge",
                target_kind: "semantic_edge",
                target_id: item.target_id,
                ...(item.suggested_valid_to === undefined
                  ? {}
                  : { suggested_valid_to: item.suggested_valid_to }),
                ...(item.by_edge_id === undefined ? {} : { by_edge_id: item.by_edge_id }),
                reason: item.reason,
                proposed_provenance: proposedProvenance,
                source_target_type: item.target_type,
                source_target_id: item.target_id,
              }
            : {
                target_type: item.repair_target_type ?? item.target_type,
                target_id: item.repair_target_id ?? item.target_id,
                repair_op: item.repair_op ?? "patch",
                ...(item.patch === undefined ? {} : { patch: item.patch }),
                ...(item.evidence_episode_ids === undefined
                  ? {}
                  : { evidence_episode_ids: item.evidence_episode_ids }),
                proposed_provenance: proposedProvenance,
                source_target_type: item.target_type,
                source_target_id: item.target_id,
              }
          : item.kind === "misattribution"
            ? {
                target_type: item.target_type,
                target_id: item.target_id,
                ...(item.patch === undefined ? {} : { patch: item.patch }),
                proposed_provenance: proposedProvenance,
              }
            : item.kind === "temporal_drift"
              ? {
                  target_type: item.target_type,
                  target_id: item.target_id,
                  ...(item.corrected_start_time === undefined
                    ? {}
                    : { corrected_start_time: item.corrected_start_time }),
                  ...(item.corrected_end_time === undefined
                    ? {}
                    : { corrected_end_time: item.corrected_end_time }),
                  ...(item.patch_description === undefined
                    ? {}
                    : { patch_description: item.patch_description }),
                  ...(item.target_type === "semantic_edge"
                    ? {
                        target_kind: "semantic_edge",
                        reason: item.reason,
                      }
                    : {}),
                  ...(item.suggested_valid_to === undefined
                    ? {}
                    : { suggested_valid_to: item.suggested_valid_to }),
                  ...(item.by_edge_id === undefined ? {} : { by_edge_id: item.by_edge_id }),
                  proposed_provenance: proposedProvenance,
                }
              : {
                  target_type: item.target_type,
                  target_id: item.target_id,
                };
      const reviewItem = ctx.reviewQueueRepository.enqueue({
        kind: item.kind,
        refs,
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
