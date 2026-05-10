import { z } from "zod";

import { entityIdSchema } from "../../memory/commitments/index.js";
import { episodeIdSchema, type Episode } from "../../memory/episodic/index.js";
import {
  reviewKindSchema,
  semanticEdgeIdSchema,
  type SemanticEdge,
  type SemanticNode,
} from "../../memory/semantic/index.js";
import { streamEntryIdSchema, type StreamEntry } from "../../stream/index.js";
import type { EntityId, EpisodeId, StreamEntryId } from "../../util/ids.js";
import { valueAppearsIn } from "../../util/text-presence.js";
import type { OfflineContext } from "../types.js";

export type OverseerSourceGroundingContext = Pick<
  OfflineContext,
  "entityRepository" | "episodicRepository" | "retrievalPipeline"
>;

export const overseerFlagKindSchema = z.enum([
  reviewKindSchema.enum.misattribution,
  reviewKindSchema.enum.temporal_drift,
  reviewKindSchema.enum.identity_inconsistency,
]);

export const sourceAssessmentSchema = z.enum([
  "supports_flag",
  "contradicts_flag",
  "provenance_insufficient",
]);

export const overseerFlagPayloadSchema = z.object({
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
});

export type OverseerFlagPayload = z.infer<typeof overseerFlagPayloadSchema>;

export const overseerAudienceMetadataSchema = z
  .object({
    entity_id: entityIdSchema,
    display_name: z.string().min(1),
    source_episode_ids: z.array(episodeIdSchema),
  })
  .strict();

export type OverseerAudienceMetadata = z.infer<typeof overseerAudienceMetadataSchema>;

export const overseerFlagAuditPayloadSchema = overseerFlagPayloadSchema
  .extend({
    flag_kind: overseerFlagKindSchema,
    audience_entities: z.array(overseerAudienceMetadataSchema),
  })
  .strict()
  .superRefine((payload, ctx) => {
    if (payload.flag_kind !== payload.kind) {
      ctx.addIssue({
        code: "custom",
        message: "flag_kind must match kind",
      });
    }
  });

export type OverseerFlagAuditPayload = z.infer<typeof overseerFlagAuditPayloadSchema>;

export const suppressedFlagReasonSchema = z.enum([
  "PROVENANCE-INSUFFICIENT",
  "INVALID-CITATION",
  "SOURCE-CONTRADICTS",
  "AUDIENCE-NAME-GROUNDED",
]);

export const suppressedOverseerFlagSchema = z.object({
  flag: overseerFlagPayloadSchema,
  reason: suppressedFlagReasonSchema,
  cited_ids: z.array(streamEntryIdSchema),
});

export type SuppressedOverseerFlag = z.infer<typeof suppressedOverseerFlagSchema>;

export type OverseerSourceTarget =
  | {
      type: "episode";
      id: Episode["id"];
      content: Episode;
    }
  | {
      type: "semantic_node";
      id: SemanticNode["id"];
      content: SemanticNode;
    }
  | {
      type: "semantic_edge";
      id: SemanticEdge["id"];
      content: SemanticEdge;
    };

export type OverseerResolvedSourceEntry = {
  source_episode_ids: EpisodeId[];
  entry: StreamEntry;
};

export type OverseerSourceEpisode = {
  id: EpisodeId;
  audience_entity_id: EntityId | null;
  shared: boolean;
};

export type OverseerSourceBundle = {
  target_type: OverseerSourceTarget["type"];
  target_id: OverseerSourceTarget["id"];
  source_episode_ids: EpisodeId[];
  source_stream_ids: StreamEntryId[];
  source_episodes: OverseerSourceEpisode[];
  audience_entities: OverseerAudienceMetadata[];
  entries: OverseerResolvedSourceEntry[];
  missing_episode_ids: EpisodeId[];
  missing_stream_ids: StreamEntryId[];
};

function suppressFlag(
  flag: OverseerFlagPayload,
  reason: z.infer<typeof suppressedFlagReasonSchema>,
  citedIds: readonly z.infer<typeof streamEntryIdSchema>[],
): SuppressedOverseerFlag {
  return {
    flag,
    reason,
    cited_ids: [...citedIds],
  };
}

function gateAudienceNameGrounding(
  flag: OverseerFlagPayload,
  sourceBundle: OverseerSourceBundle,
): SuppressedOverseerFlag | null {
  if (flag.quoted_span === undefined) {
    return null;
  }

  for (const audience of sourceBundle.audience_entities) {
    if (
      audience.source_episode_ids.length > 0 &&
      valueAppearsIn(flag.quoted_span, audience.display_name)
    ) {
      return suppressFlag(flag, "AUDIENCE-NAME-GROUNDED", flag.cited_stream_ids ?? []);
    }
  }

  return null;
}

export function gateMisattributionFlag(
  flag: OverseerFlagPayload,
  sourceBundle: OverseerSourceBundle,
): SuppressedOverseerFlag | null {
  const audienceNameSuppression = gateAudienceNameGrounding(flag, sourceBundle);

  if (audienceNameSuppression !== null) {
    return audienceNameSuppression;
  }

  const citedIds = flag.cited_stream_ids ?? [];

  if (citedIds.length === 0) {
    return suppressFlag(flag, "PROVENANCE-INSUFFICIENT", citedIds);
  }

  const validSourceIds = new Set(sourceBundle.entries.map((source) => source.entry.id));
  const invalidCitations = citedIds.filter((streamId) => !validSourceIds.has(streamId));

  if (invalidCitations.length > 0) {
    return suppressFlag(flag, "INVALID-CITATION", citedIds);
  }

  if (flag.source_assessment === "contradicts_flag") {
    return suppressFlag(flag, "SOURCE-CONTRADICTS", citedIds);
  }

  if (
    flag.source_assessment === undefined ||
    flag.source_assessment === "provenance_insufficient"
  ) {
    return suppressFlag(flag, "PROVENANCE-INSUFFICIENT", citedIds);
  }

  return null;
}

export function buildOverseerFlagAuditPayload(
  flag: OverseerFlagPayload,
  sourceBundle: OverseerSourceBundle,
): OverseerFlagAuditPayload {
  return overseerFlagAuditPayloadSchema.parse({
    ...flag,
    flag_kind: flag.kind,
    audience_entities: sourceBundle.audience_entities,
  });
}

function uniqueEpisodeIds(ids: readonly EpisodeId[]): EpisodeId[] {
  const seen = new Set<string>();
  const unique: EpisodeId[] = [];

  for (const id of ids) {
    if (seen.has(id)) {
      continue;
    }

    seen.add(id);
    unique.push(id);
  }

  return unique;
}

function uniqueStreamIds(ids: readonly StreamEntryId[]): StreamEntryId[] {
  const seen = new Set<string>();
  const unique: StreamEntryId[] = [];

  for (const id of ids) {
    if (seen.has(id)) {
      continue;
    }

    seen.add(id);
    unique.push(id);
  }

  return unique;
}

function appendStreamSources(
  streamSources: Map<string, EpisodeId[]>,
  episodeId: EpisodeId,
  streamIds: readonly StreamEntryId[],
): void {
  for (const streamId of streamIds) {
    const existing = streamSources.get(streamId);

    if (existing === undefined) {
      streamSources.set(streamId, [episodeId]);
      continue;
    }

    existing.push(episodeId);
  }
}

function sourceEpisodeMetadata(episodes: readonly Episode[]): OverseerSourceEpisode[] {
  return episodes.map((episode) => ({
    id: episode.id,
    audience_entity_id: episode.audience_entity_id ?? null,
    shared: episode.shared ?? (episode.audience_entity_id ?? null) === null,
  }));
}

function audienceMetadataForEpisodes(
  episodes: readonly Episode[],
  ctx: OverseerSourceGroundingContext,
): OverseerAudienceMetadata[] {
  const byEntityId = new Map<string, OverseerAudienceMetadata>();

  for (const episode of episodes) {
    const audienceEntityId = episode.audience_entity_id ?? null;

    if (audienceEntityId === null) {
      continue;
    }

    const entity = ctx.entityRepository.get(audienceEntityId);
    const displayName = entity?.canonical_name.trim() ?? "";

    if (displayName.length === 0) {
      continue;
    }

    const existing = byEntityId.get(audienceEntityId);

    if (existing === undefined) {
      byEntityId.set(audienceEntityId, {
        entity_id: audienceEntityId,
        display_name: displayName,
        source_episode_ids: [episode.id],
      });
      continue;
    }

    existing.source_episode_ids.push(episode.id);
  }

  return [...byEntityId.values()].map((metadata) => ({
    ...metadata,
    source_episode_ids: uniqueEpisodeIds(metadata.source_episode_ids),
  }));
}

async function sourceEpisodesForTarget(
  target: OverseerSourceTarget,
  ctx: OverseerSourceGroundingContext,
): Promise<{
  episodes: Episode[];
  sourceEpisodeIds: EpisodeId[];
  missingEpisodeIds: EpisodeId[];
}> {
  if (target.type === "episode") {
    return {
      episodes: [target.content],
      sourceEpisodeIds: [target.content.id],
      missingEpisodeIds: [],
    };
  }

  const sourceEpisodeIds = uniqueEpisodeIds(
    target.type === "semantic_node"
      ? target.content.source_episode_ids
      : target.content.evidence_episode_ids,
  );
  const episodes = await ctx.episodicRepository.getMany(sourceEpisodeIds);
  const foundEpisodeIds = new Set(episodes.map((episode) => episode.id));
  const missingEpisodeIds = sourceEpisodeIds.filter((episodeId) => !foundEpisodeIds.has(episodeId));

  return {
    episodes,
    sourceEpisodeIds,
    missingEpisodeIds,
  };
}

export async function resolveTargetSourceBundle(
  target: OverseerSourceTarget,
  ctx: OverseerSourceGroundingContext,
): Promise<OverseerSourceBundle> {
  const sourceEpisodes = await sourceEpisodesForTarget(target, ctx);
  const streamSources = new Map<string, EpisodeId[]>();
  const streamIds = uniqueStreamIds(
    sourceEpisodes.episodes.flatMap((episode) => {
      appendStreamSources(streamSources, episode.id, episode.source_stream_ids);
      return episode.source_stream_ids;
    }),
  );
  const resolvedEntries = await ctx.retrievalPipeline.resolveSourceEntries(streamIds);
  const entries = streamIds.flatMap((streamId) => {
    const entry = resolvedEntries.get(streamId);

    if (entry === undefined) {
      return [];
    }

    return [
      {
        source_episode_ids: uniqueEpisodeIds(streamSources.get(streamId) ?? []),
        entry,
      },
    ];
  });

  return {
    target_type: target.type,
    target_id: target.id,
    source_episode_ids: sourceEpisodes.sourceEpisodeIds,
    source_stream_ids: streamIds,
    source_episodes: sourceEpisodeMetadata(sourceEpisodes.episodes),
    audience_entities: audienceMetadataForEpisodes(sourceEpisodes.episodes, ctx),
    entries,
    missing_episode_ids: sourceEpisodes.missingEpisodeIds,
    missing_stream_ids: streamIds.filter((streamId) => !resolvedEntries.has(streamId)),
  };
}

function entryContent(entry: StreamEntry): string {
  if (typeof entry.content === "string") {
    return entry.content;
  }

  return JSON.stringify(entry.content) ?? String(entry.content);
}

function formatIdList(ids: readonly string[]): string {
  return ids.length === 0 ? "none" : ids.join(", ");
}

export function renderSourceBundleForPrompt(bundle: OverseerSourceBundle): string {
  const lines = [
    "Target source grounding:",
    `target_type: ${bundle.target_type}`,
    `target_id: ${bundle.target_id}`,
    `source_episode_ids: ${formatIdList(bundle.source_episode_ids)}`,
    `source_stream_ids: ${formatIdList(bundle.source_stream_ids)}`,
  ];

  if (bundle.source_episodes.length === 0) {
    lines.push("Source episode audience tags: none");
  } else {
    lines.push("Source episode audience tags:");

    for (const episode of bundle.source_episodes) {
      lines.push(
        `EPISODE episode_id=${episode.id} audience_entity_id=${episode.audience_entity_id ?? "none"} shared=${String(episode.shared)}`,
      );
    }
  }

  if (bundle.audience_entities.length === 0) {
    lines.push("Audience entity metadata: none");
  } else {
    lines.push("Audience entity metadata:");

    for (const audience of bundle.audience_entities) {
      lines.push(
        `AUDIENCE entity_id=${audience.entity_id} display_name=${JSON.stringify(audience.display_name)} source_episode_ids=${formatIdList(audience.source_episode_ids)}`,
      );
    }
  }

  if (bundle.missing_episode_ids.length > 0) {
    lines.push(
      `PROVENANCE-INSUFFICIENT missing source_episode_ids: ${formatIdList(bundle.missing_episode_ids)}`,
    );
  }

  if (bundle.missing_stream_ids.length > 0) {
    lines.push(
      `PROVENANCE-INSUFFICIENT missing source_stream_ids: ${formatIdList(bundle.missing_stream_ids)}`,
    );
  }

  if (bundle.entries.length === 0) {
    lines.push("Resolved source entries: none");
    return lines.join("\n");
  }

  lines.push("Resolved source entries:");

  for (const [index, source] of bundle.entries.entries()) {
    lines.push(
      [
        `SOURCE[${index}] source_episode_ids=${formatIdList(source.source_episode_ids)} session_id=${source.entry.session_id} timestamp=${source.entry.timestamp} stream_id=${source.entry.id} kind=${source.entry.kind}`,
        entryContent(source.entry),
      ].join("\n"),
    );
  }

  return lines.join("\n");
}
