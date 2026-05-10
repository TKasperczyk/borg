import { z } from "zod";

import { episodeIdSchema, type Episode } from "../../memory/episodic/index.js";
import {
  semanticNodeIdSchema,
  type ReviewQueueItem,
  type SemanticNode,
} from "../../memory/semantic/index.js";
import type { OfflineContext } from "../types.js";
import {
  gateMisattributionFlag,
  overseerFlagAuditPayloadSchema,
  resolveTargetSourceBundle,
  type SuppressedOverseerFlag,
} from "./source-grounding.js";

const DAY_MS = 24 * 60 * 60 * 1_000;

const revalidatableReviewKindSchema = z.literal("misattribution");
const revalidationTargetRefsSchema = z.discriminatedUnion("target_type", [
  z
    .object({
      target_type: z.literal("episode"),
      target_id: episodeIdSchema,
    })
    .passthrough(),
  z
    .object({
      target_type: z.literal("semantic_node"),
      target_id: semanticNodeIdSchema,
    })
    .passthrough(),
]);

type ReviewRevalidationContext = Pick<
  OfflineContext,
  | "clock"
  | "entityRepository"
  | "episodicRepository"
  | "retrievalPipeline"
  | "reviewQueueRepository"
  | "semanticNodeRepository"
>;

type RevalidationTarget =
  | {
      type: "episode";
      id: Episode["id"];
      content: Episode;
    }
  | {
      type: "semantic_node";
      id: SemanticNode["id"];
      content: SemanticNode;
    };

export type ReviewRevalidationOptions = {
  kind: "misattribution";
  maxAgeDays?: number;
};

export type ReviewRevalidationResult = {
  kind: "misattribution";
  revalidated: number;
  dismissed_as_suppressed: number;
  skipped_legacy: number;
  unchanged: number;
  diagnostics: Record<string, number>;
  warnings: string[];
};

function hasPersistedFlagPayload(item: ReviewQueueItem): boolean {
  return Object.hasOwn(item.refs, "overseer_flag");
}

function incrementDiagnostic(
  diagnostics: Record<string, number>,
  suppression: SuppressedOverseerFlag,
): void {
  diagnostics[suppression.reason] = (diagnostics[suppression.reason] ?? 0) + 1;
}

function suppressionDiagnostic(suppression: SuppressedOverseerFlag): string {
  const citedIds = suppression.cited_ids.length === 0 ? "none" : suppression.cited_ids.join(", ");

  return `${suppression.reason}; cited_stream_ids=${citedIds}`;
}

function shouldRevalidateByAge(
  item: ReviewQueueItem,
  options: Pick<ReviewRevalidationOptions, "maxAgeDays">,
  now: number,
): boolean {
  if (options.maxAgeDays === undefined) {
    return true;
  }

  return item.created_at <= now - options.maxAgeDays * DAY_MS;
}

async function resolveRevalidationTarget(
  ctx: ReviewRevalidationContext,
  item: ReviewQueueItem,
  warnings: string[],
): Promise<RevalidationTarget | null> {
  const parsedRefs = revalidationTargetRefsSchema.safeParse(item.refs);

  if (!parsedRefs.success) {
    warnings.push(`review ${item.id} skipped: target refs are not revalidatable`);
    return null;
  }

  if (parsedRefs.data.target_type === "episode") {
    const episode = await ctx.episodicRepository.get(parsedRefs.data.target_id, {
      includeArchived: true,
    });

    if (episode === null) {
      warnings.push(`review ${item.id} skipped: target episode was not found`);
      return null;
    }

    return {
      type: "episode",
      id: episode.id,
      content: episode,
    };
  }

  const node = await ctx.semanticNodeRepository.get(parsedRefs.data.target_id);

  if (node === null) {
    warnings.push(`review ${item.id} skipped: target semantic node was not found`);
    return null;
  }

  return {
    type: "semantic_node",
    id: node.id,
    content: node,
  };
}

export async function revalidateReviewQueue(
  ctx: ReviewRevalidationContext,
  options: ReviewRevalidationOptions,
): Promise<ReviewRevalidationResult> {
  const kind = revalidatableReviewKindSchema.parse(options.kind);
  const result: ReviewRevalidationResult = {
    kind,
    revalidated: 0,
    dismissed_as_suppressed: 0,
    skipped_legacy: 0,
    unchanged: 0,
    diagnostics: {},
    warnings: [],
  };
  const now = ctx.clock.now();
  const items = ctx.reviewQueueRepository.list({
    kind,
    openOnly: true,
  });

  for (const item of items) {
    if (!shouldRevalidateByAge(item, options, now)) {
      continue;
    }

    if (!hasPersistedFlagPayload(item)) {
      result.skipped_legacy += 1;
      result.warnings.push(`review ${item.id} skipped: legacy item has no overseer_flag payload`);
      continue;
    }

    const parsedPayload = overseerFlagAuditPayloadSchema.safeParse(item.refs.overseer_flag);

    if (!parsedPayload.success) {
      result.skipped_legacy += 1;
      result.warnings.push(`review ${item.id} skipped: overseer_flag payload is invalid`);
      continue;
    }

    const target = await resolveRevalidationTarget(ctx, item, result.warnings);

    if (target === null) {
      result.unchanged += 1;
      continue;
    }

    const sourceBundle = await resolveTargetSourceBundle(target, ctx);
    const suppression = gateMisattributionFlag(parsedPayload.data, sourceBundle);

    result.revalidated += 1;

    if (suppression === null) {
      result.unchanged += 1;
      continue;
    }

    incrementDiagnostic(result.diagnostics, suppression);
    await ctx.reviewQueueRepository.resolve(
      item.id,
      {
        decision: "dismiss",
        reason: `revalidated -- now suppressed by current grounding logic: ${suppressionDiagnostic(suppression)}`,
      },
      {
        source: "auto",
        sourceProcess: "review-revalidate",
      },
    );
    result.dismissed_as_suppressed += 1;
  }

  return result;
}
