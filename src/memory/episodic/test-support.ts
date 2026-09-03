import type { Episode } from "./types.js";

export function episodeToRawLanceRowForTest(episode: Episode): Record<string, unknown> {
  const audienceEntityId = episode.audience_entity_id ?? null;

  return {
    id: episode.id,
    title: episode.title,
    narrative: episode.narrative,
    participants: JSON.stringify(episode.participants),
    location: episode.location,
    start_time: episode.start_time,
    end_time: episode.end_time,
    source_stream_ids: JSON.stringify(episode.source_stream_ids),
    significance: episode.significance,
    tags: JSON.stringify(episode.tags),
    confidence: episode.confidence,
    lineage_derived_from: JSON.stringify(episode.lineage.derived_from),
    lineage_supersedes: JSON.stringify(episode.lineage.supersedes),
    source_fingerprint: [...new Set(episode.source_stream_ids)].sort().join("\n"),
    audience_entity_id: audienceEntityId,
    origin_audience_entity_ids: JSON.stringify(episode.origin_audience_entity_ids ?? []),
    shared: episode.shared ?? audienceEntityId === null,
    emotional_arc: episode.emotional_arc === null ? null : JSON.stringify(episode.emotional_arc),
    episode_kind: episode.episode_kind ?? null,
    consolidation_family_id: episode.consolidation_family_id ?? null,
    consolidation_coverage_hash: episode.consolidation_coverage_hash ?? null,
    embedding: Array.from(episode.embedding),
    created_at: episode.created_at,
    updated_at: episode.updated_at,
  };
}
