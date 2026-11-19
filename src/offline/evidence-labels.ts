import { EVIDENCE_GUIDANCE, renderEvidencePromptLines } from "../memory/common/episode-evidence.js";
import { estimatePromptTokens } from "../util/token-estimate.js";
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

/** Sum the same per-row estimates as before, now including bounded envelopes and guidance. */
export function renderEvidenceRows(rows: readonly Record<string, unknown>[]): {
  text: string;
  tokens: number;
} {
  const text = renderEvidencePromptLines([EVIDENCE_GUIDANCE, ...rows]);
  // JSON rows escape embedded newlines. This splits serialized structure, not record text.
  return {
    text,
    tokens:
      text.length === 0
        ? 0
        : text.split("\n").reduce((sum, row) => sum + estimatePromptTokens(row), 0),
  };
}
