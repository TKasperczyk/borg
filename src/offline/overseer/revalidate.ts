import { z } from "zod";

import { episodeIdSchema } from "../../memory/episodic/index.js";
import { semanticNodeIdSchema, type ReviewQueueItem } from "../../memory/semantic/index.js";
import type { StreamEntry } from "../../stream/index.js";
import { DEFAULT_SESSION_ID, type EpisodeId, type StreamEntryId } from "../../util/ids.js";
import type { OfflineContext } from "../types.js";
import {
  gateMisattributionFlag,
  overseerFlagAuditPayloadSchema,
  type OverseerFlagAuditPayload,
  type OverseerResolvedSourceEntry,
  type OverseerSourceBundle,
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
  "clock" | "episodicRepository" | "retrievalPipeline" | "reviewQueueRepository"
>;

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

function parseRevalidationTargetRefs(
  item: ReviewQueueItem,
  warnings: string[],
): z.infer<typeof revalidationTargetRefsSchema> | null {
  const parsedRefs = revalidationTargetRefsSchema.safeParse(item.refs);

  if (!parsedRefs.success) {
    warnings.push(`review ${item.id} skipped: target refs are not revalidatable`);
    return null;
  }

  return parsedRefs.data;
}

function missingPersistedInputFields(payload: OverseerFlagAuditPayload): string[] {
  const missing: string[] = [];

  if (payload.quoted_span === undefined) {
    missing.push("quoted_span");
  }

  if (payload.cited_stream_ids === undefined) {
    missing.push("cited_stream_ids");
  }

  if (payload.source_assessment === undefined) {
    missing.push("source_assessment");
  }

  return missing;
}

function uniqueEpisodeIds(ids: readonly EpisodeId[]): EpisodeId[] {
  return [...new Set(ids)];
}

function uniqueStreamIds(ids: readonly StreamEntryId[]): StreamEntryId[] {
  return [...new Set(ids)];
}

function sourceEpisodeIdsFromPayload(payload: OverseerFlagAuditPayload): EpisodeId[] {
  return uniqueEpisodeIds([
    ...(payload.source_episode_ids ?? []),
    ...payload.audience_entities.flatMap((audience) => audience.source_episode_ids),
  ]);
}

function sourceStreamIdsFromPayload(
  payload: OverseerFlagAuditPayload & { cited_stream_ids: StreamEntryId[] },
): StreamEntryId[] {
  return uniqueStreamIds(payload.source_stream_ids ?? payload.cited_stream_ids);
}

function placeholderStreamEntry(id: StreamEntryId): StreamEntry {
  return {
    id,
    timestamp: 0,
    kind: "internal_event",
    content: {
      missing_persisted_citation: true,
    },
    session_id: DEFAULT_SESSION_ID,
    compressed: false,
  };
}

async function resolvePersistedSourceEntries(
  ctx: ReviewRevalidationContext,
  item: ReviewQueueItem,
  sourceStreamIds: readonly StreamEntryId[],
  sourceEpisodeIds: readonly EpisodeId[],
  warnings: string[],
): Promise<OverseerResolvedSourceEntry[]> {
  const resolvedEntries = await ctx.retrievalPipeline.resolveSourceEntries(sourceStreamIds);
  const missingStreamIds: StreamEntryId[] = [];
  const entries = sourceStreamIds.map((streamId) => {
    const entry = resolvedEntries.get(streamId);

    if (entry !== undefined) {
      return {
        source_episode_ids: [...sourceEpisodeIds],
        entry,
      };
    }

    missingStreamIds.push(streamId);
    return {
      source_episode_ids: [...sourceEpisodeIds],
      entry: placeholderStreamEntry(streamId),
    };
  });

  if (missingStreamIds.length > 0) {
    warnings.push(
      `review ${item.id}: persisted source stream entries no longer resolve: ${missingStreamIds.join(", ")}`,
    );
  }

  return entries;
}

function recordArchivedPersistedSourceEpisodes(
  ctx: ReviewRevalidationContext,
  item: ReviewQueueItem,
  sourceEpisodeIds: readonly EpisodeId[],
  warnings: string[],
): void {
  const stats = ctx.episodicRepository.getStatsMany(sourceEpisodeIds);
  const archived = sourceEpisodeIds.filter((episodeId) => stats.get(episodeId)?.archived === true);

  if (archived.length > 0) {
    warnings.push(
      `review ${item.id}: persisted source episodes are now archived: ${archived.join(", ")}`,
    );
  }
}

async function buildPersistedSourceBundle(
  ctx: ReviewRevalidationContext,
  item: ReviewQueueItem,
  payload: OverseerFlagAuditPayload & {
    cited_stream_ids: StreamEntryId[];
    quoted_span: string;
    source_assessment: NonNullable<OverseerFlagAuditPayload["source_assessment"]>;
  },
  warnings: string[],
): Promise<OverseerSourceBundle | null> {
  const targetRefs = parseRevalidationTargetRefs(item, warnings);

  if (targetRefs === null) {
    return null;
  }

  const sourceEpisodeIds = sourceEpisodeIdsFromPayload(payload);
  const sourceStreamIds = sourceStreamIdsFromPayload(payload);
  recordArchivedPersistedSourceEpisodes(ctx, item, sourceEpisodeIds, warnings);

  return {
    target_type: targetRefs.target_type,
    target_id: targetRefs.target_id,
    source_episode_ids: sourceEpisodeIds,
    source_stream_ids: sourceStreamIds,
    source_episodes: [],
    audience_entities: payload.audience_entities,
    entries: await resolvePersistedSourceEntries(
      ctx,
      item,
      sourceStreamIds,
      sourceEpisodeIds,
      warnings,
    ),
    missing_episode_ids: [],
    missing_stream_ids: [],
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

    const missingFields = missingPersistedInputFields(parsedPayload.data);

    if (missingFields.length > 0) {
      result.skipped_legacy += 1;
      result.warnings.push(
        `review ${item.id} skipped: overseer_flag payload is missing ${missingFields.join(", ")}`,
      );
      continue;
    }

    const sourceBundle = await buildPersistedSourceBundle(
      ctx,
      item,
      parsedPayload.data as OverseerFlagAuditPayload & {
        cited_stream_ids: StreamEntryId[];
        quoted_span: string;
        source_assessment: NonNullable<OverseerFlagAuditPayload["source_assessment"]>;
      },
      result.warnings,
    );

    if (sourceBundle === null) {
      result.skipped_legacy += 1;
      continue;
    }

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
        reason: `revalidated -- now suppressed by current gate logic against persisted enqueue-time inputs: ${suppressionDiagnostic(suppression)}`,
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
