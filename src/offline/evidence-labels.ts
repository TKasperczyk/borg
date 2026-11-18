import type { EpisodicRepository, Episode } from "../memory/episodic/index.js";
import {
  resolveMemoryDisclosureLabelForEpisodeIds,
  type MemoryDisclosureLabel,
} from "../retrieval/index.js";
export {
  episodeEvidencePromptRow,
  withEpisodeEvidenceContext,
  EVIDENCE_GUIDANCE,
  renderEvidencePromptJson,
  renderEvidencePromptLines,
  evidencePromptFields,
  withSemanticNodeEvidenceContext,
} from "../memory/common/episode-evidence.js";

export async function disclosureLabelForEpisodeIds(
  episodicRepository: EpisodicRepository,
  episodeIds: readonly Episode["id"][],
): Promise<MemoryDisclosureLabel> {
  return resolveMemoryDisclosureLabelForEpisodeIds(episodicRepository, episodeIds);
}
