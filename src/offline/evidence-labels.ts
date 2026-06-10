import { memoryDisclosurePayloadFields } from "../memory/common/disclosure-serializers.js";
import type { EpisodicRepository, Episode } from "../memory/episodic/index.js";
import {
  memoryDisclosureLabelFromEpisodeAccess,
  resolveMemoryDisclosureLabelForEpisodeIds,
  type MemoryDisclosureLabel,
} from "../retrieval/index.js";

export function episodeEvidencePromptRow(
  episode: Episode,
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  const disclosureLabel = memoryDisclosureLabelFromEpisodeAccess(episode);

  return {
    id: episode.id,
    title: episode.title,
    narrative: episode.narrative,
    ...extra,
    ...memoryDisclosurePayloadFields(disclosureLabel),
  };
}

export async function disclosureLabelForEpisodeIds(
  episodicRepository: EpisodicRepository,
  episodeIds: readonly Episode["id"][],
): Promise<MemoryDisclosureLabel> {
  return resolveMemoryDisclosureLabelForEpisodeIds(episodicRepository, episodeIds);
}
