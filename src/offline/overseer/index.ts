import { z } from "zod";

import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import {
  commitmentMemoryDisclosureLabel,
  goalMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
} from "../../memory/common/disclosure-serializers.js";
import { selfPrivateMemoryDisclosureLabel } from "../../retrieval/index.js";
import { episodeIdSchema, type Episode } from "../../memory/episodic/index.js";
import {
  semanticEdgeIdSchema,
  semanticNodeIdSchema,
  type SemanticEdge,
  type SemanticNode,
} from "../../memory/semantic/index.js";
import { streamEntryIdSchema, type StreamEntry } from "../../stream/index.js";
import { dedupePreservingOrder } from "../../util/collections.js";
import { BudgetExceededError } from "../../util/errors.js";

import type { ReverserRegistry } from "../audit-log.js";
import { getBudgetErrorTokens, withBudget } from "../budget.js";
import { serializeDisclosureLabeledTargetPayload } from "../disclosure-target-serialization.js";
import { offlineProcessError } from "../process-errors.js";
import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessError,
  OfflineResult,
} from "../types.js";
import {
  buildOverseerFlagAuditPayload,
  gateMisattributionFlag,
  overseerFlagAuditPayloadSchema,
  overseerFlagKindSchema,
  overseerFlagPayloadSchema,
  renderSourceBundleForPrompt,
  resolveTargetSourceBundle,
  sourceAssessmentSchema,
  suppressedOverseerFlagSchema,
  type OverseerSourceBundle,
} from "./source-grounding.js";

const reviewFlagSchema = overseerFlagPayloadSchema;

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
  source_assessment: sourceAssessmentSchema.optional(),
  cited_stream_ids: z.array(streamEntryIdSchema).optional(),
  quoted_span: z.string().min(1).optional(),
  provenance_note: z.string().min(1).optional(),
  overseer_flag: overseerFlagAuditPayloadSchema,
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

const overseerCandidateStatsSchema = z.object({
  proposed: z.number().int().nonnegative(),
  accepted: z.number().int().nonnegative(),
  rejected: z.number().int().nonnegative(),
});

export const overseerPlanSchema = z.object({
  process: z.literal("overseer"),
  items: z.array(overseerPlanItemSchema),
  suppressed_flags: z.array(suppressedOverseerFlagSchema).default([]),
  errors: z
    .array(
      z.object({
        process: z.literal("overseer"),
        message: z.string(),
        code: z.string().optional(),
        target_type: z.enum(["episode", "semantic_node", "semantic_edge"]).optional(),
        target_id: z.string().min(1).optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
  candidate_stats: overseerCandidateStatsSchema.optional(),
});

export type OverseerPlan = z.infer<typeof overseerPlanSchema>;
type OverseerPlanItem = OverseerPlan["items"][number];
type OverseerProposedProvenance = {
  kind: "offline";
  process: "overseer";
};

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

function invalidFlagsResponse(error: unknown): unknown {
  if (isStructuredToolCallError(error, "missing_tool_call")) {
    return new TypeError(`Overseer did not emit tool ${OVERSEER_TOOL_NAME}`);
  }

  if (
    isStructuredToolCallError(error, "invalid_payload") ||
    isStructuredToolCallError(error, "llm_failed")
  ) {
    return error.cause ?? error;
  }

  return error;
}

function parseFlags(input: unknown) {
  return overseerResponseSchema.parse(input);
}

function isAssistantAuthoredReviewSource(entry: Pick<StreamEntry, "kind">): boolean {
  return (
    entry.kind === "agent_msg" ||
    entry.kind === "agent_observed" ||
    entry.kind === "agent_suppressed"
  );
}

function uniqueStreamIds(ids: readonly z.infer<typeof streamEntryIdSchema>[]) {
  return dedupePreservingOrder(ids);
}

async function assistantAuthoredCitedStreamEntryIds(
  ctx: OfflineContext,
  citedStreamIds: readonly z.infer<typeof streamEntryIdSchema>[] | undefined,
) {
  if (citedStreamIds === undefined || citedStreamIds.length === 0) {
    return [];
  }

  const resolved = await ctx.retrievalPipeline.resolveSourceEntries(citedStreamIds);

  return uniqueStreamIds(
    citedStreamIds.filter((streamEntryId) => {
      const entry = resolved.get(streamEntryId);

      return entry !== undefined && isAssistantAuthoredReviewSource(entry);
    }),
  );
}

function summarizeSelfState(ctx: OfflineContext): string {
  const values =
    ctx.valuesRepository
      .list()
      .map((value) =>
        JSON.stringify({
          id: value.id,
          label: value.label,
          ...memoryDisclosurePayloadFields(selfPrivateMemoryDisclosureLabel()),
        }),
      )
      .join(" | ") || "none";
  const goals =
    ctx.goalsRepository
      .list({ status: "active" })
      .map((goal) =>
        JSON.stringify({
          id: goal.id,
          description: goal.description,
          ...memoryDisclosurePayloadFields(goalMemoryDisclosureLabel(goal)),
        }),
      )
      .join(" | ") || "none";
  const traits =
    ctx.traitsRepository
      .list()
      .map((trait) =>
        JSON.stringify({
          id: trait.id,
          label: trait.label,
          strength: trait.strength,
          ...memoryDisclosurePayloadFields(selfPrivateMemoryDisclosureLabel()),
        }),
      )
      .join(" | ") || "none";
  const commitments =
    ctx.commitmentRepository
      .list({ activeOnly: true })
      .map((commitment) =>
        JSON.stringify({
          id: commitment.id,
          directive: commitment.directive,
          ...memoryDisclosurePayloadFields(commitmentMemoryDisclosureLabel(commitment)),
        }),
      )
      .join(" | ") || "none";
  const currentPeriod = ctx.autobiographicalRepository.currentPeriod();
  const currentPeriodRow =
    currentPeriod === null
      ? "none"
      : JSON.stringify({
          id: currentPeriod.id,
          label: currentPeriod.label,
          ...memoryDisclosurePayloadFields(selfPrivateMemoryDisclosureLabel()),
        });

  return [
    `Values: ${values}`,
    `Goals: ${goals}`,
    `Traits: ${traits}`,
    `Commitments: ${commitments}`,
    `CurrentPeriod: ${currentPeriodRow}`,
  ].join("\n");
}

async function buildPrompt(
  target: OverseerTarget,
  ctx: OfflineContext,
  sourceBundle: OverseerSourceBundle,
): Promise<string> {
  const serializedTarget = await serializeDisclosureLabeledTargetPayload(ctx, target);

  return [
    "Check the memory item for misattribution, temporal drift, and identity inconsistency.",
    "If you flag an issue, include the concrete repair payload needed to fix it.",
    "For misattribution, use only the resolved source entries below. Include quoted_span as the exact target text span being challenged, cited_stream_ids from those entries, source_assessment, and patch fields that directly correct the target memory.",
    "Set source_assessment to supports_flag only when the cited source entries support the flag, contradicts_flag when they refute the flag, and provenance_insufficient when the provided source entries are missing or inadequate.",
    "Audience entity metadata below is legitimate grounding for the listed display_name. If the target uses that exact display_name for the audience and a source episode is tagged with that audience entity_id, do not flag the audience-name reference merely because the raw source text omits the name.",
    "For temporal drift, provide corrected timestamps and/or a replacement description.",
    "For semantic_edge temporal drift or identity inconsistency, provide suggested_valid_to and optional by_edge_id; only flag edges that should be reviewed for closure.",
    "For identity inconsistency, target a specific value, goal, trait, commitment, or autobiographical period by id and propose reinforce, contradict, or patch.",
    `Emit your result by calling the ${OVERSEER_TOOL_NAME} tool exactly once.`,
    summarizeSelfState(ctx),
    "Memory item:",
    JSON.stringify(serializedTarget),
    "Raw source entries:",
    renderSourceBundleForPrompt(sourceBundle),
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
      ...(item.source_assessment === undefined
        ? {}
        : { source_assessment: item.source_assessment }),
      ...(item.cited_stream_ids === undefined ? {} : { cited_stream_ids: item.cited_stream_ids }),
      ...(item.provenance_note === undefined ? {} : { provenance_note: item.provenance_note }),
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

function candidateStatsForPlan(
  plan: Pick<OverseerPlan, "items" | "suppressed_flags" | "candidate_stats">,
): NonNullable<OfflineResult["candidate_stats"]> {
  return (
    plan.candidate_stats ?? {
      proposed: plan.items.length + plan.suppressed_flags.length,
      accepted: plan.items.length,
      rejected: plan.suppressed_flags.length,
    }
  );
}

function buildIdentityRepairRefs(
  item: OverseerPlanItem,
  proposedProvenance: OverseerProposedProvenance,
): Record<string, unknown> {
  return item.target_type === "semantic_edge"
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
      };
}

function buildMisattributionRepairRefs(input: {
  item: OverseerPlanItem;
  proposedProvenance: OverseerProposedProvenance;
  reviewedAssistantStreamEntryIds: readonly StreamEntry["id"][];
}): Record<string, unknown> {
  return {
    target_type: input.item.target_type,
    target_id: input.item.target_id,
    ...(input.item.patch === undefined ? {} : { patch: input.item.patch }),
    ...(input.item.cited_stream_ids === undefined
      ? {}
      : { evidence_stream_ids: input.item.cited_stream_ids }),
    ...(input.reviewedAssistantStreamEntryIds.length === 0
      ? {}
      : { reviewed_assistant_stream_entry_ids: input.reviewedAssistantStreamEntryIds }),
    proposed_provenance: input.proposedProvenance,
  };
}

function buildTemporalDriftRepairRefs(
  item: OverseerPlanItem,
  proposedProvenance: OverseerProposedProvenance,
): Record<string, unknown> {
  return {
    target_type: item.target_type,
    target_id: item.target_id,
    ...(item.corrected_start_time === undefined
      ? {}
      : { corrected_start_time: item.corrected_start_time }),
    ...(item.corrected_end_time === undefined
      ? {}
      : { corrected_end_time: item.corrected_end_time }),
    ...(item.patch_description === undefined ? {} : { patch_description: item.patch_description }),
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
  };
}

function buildRepairRefs(input: {
  item: OverseerPlanItem;
  proposedProvenance: OverseerProposedProvenance;
  reviewedAssistantStreamEntryIds: readonly StreamEntry["id"][];
}): Record<string, unknown> {
  switch (input.item.kind) {
    case "identity_inconsistency":
      return buildIdentityRepairRefs(input.item, input.proposedProvenance);
    case "misattribution":
      return buildMisattributionRepairRefs(input);
    case "temporal_drift":
      return buildTemporalDriftRepairRefs(input.item, input.proposedProvenance);
  }
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
    const suppressedFlags: OverseerPlan["suppressed_flags"] = [];
    const candidateStats: NonNullable<OfflineResult["candidate_stats"]> = {
      proposed: 0,
      accepted: 0,
      rejected: 0,
    };
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
            const sourceBundle = await resolveTargetSourceBundle(target, ctx);
            const flags = (
              await callStructuredTool({
                llmClient,
                request: {
                  model: ctx.config.anthropic.models.background,
                  system:
                    "You audit recently formed memories. Flag only grounded QA concerns and keep false positives low.",
                  messages: [
                    {
                      role: "user",
                      content: await buildPrompt(target, ctx, sourceBundle),
                    },
                  ],
                  tools: [OVERSEER_TOOL],
                  tool_choice: { type: "tool", name: OVERSEER_TOOL_NAME },
                  max_tokens: 4_000,
                  budget: "offline-overseer",
                },
                toolName: OVERSEER_TOOL_NAME,
                parse: parseFlags,
              })
            ).parsed.flags;

            for (const flag of flags) {
              candidateStats.proposed += 1;

              if (flag.confidence < 0.5) {
                candidateStats.rejected += 1;
                continue;
              }

              if (flag.kind === "misattribution") {
                const suppression = gateMisattributionFlag(flag, sourceBundle);

                if (suppression !== null) {
                  suppressedFlags.push(suppression);
                  candidateStats.rejected += 1;
                  continue;
                }
              }

              if (target.type === "semantic_edge" && flag.kind === "misattribution") {
                candidateStats.rejected += 1;
                continue;
              }

              const overseerFlag = buildOverseerFlagAuditPayload(flag, sourceBundle);
              const baseItem = {
                kind: flag.kind,
                reason: flag.reason,
                confidence: flag.confidence,
                overseer_flag: overseerFlag,
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
                ...(flag.source_assessment === undefined
                  ? {}
                  : { source_assessment: flag.source_assessment }),
                ...(flag.cited_stream_ids === undefined
                  ? {}
                  : { cited_stream_ids: flag.cited_stream_ids }),
                ...(flag.provenance_note === undefined
                  ? {}
                  : { provenance_note: flag.provenance_note }),
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

              candidateStats.accepted += 1;
            }
          } catch (error) {
            if (error instanceof BudgetExceededError) {
              throw error;
            }

            errors.push(
              offlineProcessError(this.name, invalidFlagsResponse(error), {
                target_type: target.type,
                target_id: target.id,
              }),
            );
          }
        }
      });

      tokensUsed = budgeted.tokens_used;
    } catch (error) {
      tokensUsed = getBudgetErrorTokens(error);
      budgetExhausted = error instanceof BudgetExceededError;
      errors.push(offlineProcessError(this.name, error));
    }

    return overseerPlanSchema.parse({
      process: this.name,
      items,
      suppressed_flags: suppressedFlags,
      errors,
      tokens_used: tokensUsed,
      budget_exhausted: budgetExhausted,
      candidate_stats: candidateStats,
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
      candidate_stats: candidateStatsForPlan(parsed),
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
      const reviewedAssistantStreamEntryIds =
        item.kind === "misattribution"
          ? await assistantAuthoredCitedStreamEntryIds(ctx, item.cited_stream_ids)
          : [];
      const repairRefs = buildRepairRefs({
        item,
        proposedProvenance,
        reviewedAssistantStreamEntryIds,
      });
      const refs = {
        ...repairRefs,
        overseer_flag: item.overseer_flag,
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
          overseer_flag: item.overseer_flag,
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
      candidate_stats: candidateStatsForPlan(plan),
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
