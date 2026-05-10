import type { Episode } from "../../memory/episodic/index.js";
import type { SemanticEdge, SemanticNode } from "../../memory/semantic/index.js";
import type { StreamEntry } from "../../stream/index.js";
import type { EntityId, EpisodeId, StreamEntryId } from "../../util/ids.js";
import type { OfflineContext } from "../types.js";

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

export type OverseerAudienceMetadata = {
  entity_id: EntityId;
  display_name: string;
  source_episode_ids: EpisodeId[];
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
  ctx: OfflineContext,
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
  ctx: OfflineContext,
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
  ctx: OfflineContext,
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
