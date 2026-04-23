import { type EpisodeId } from "../../../util/ids.js";
import { isEpisodeProvenance, type Provenance } from "../../common/provenance.js";

export const EVIDENCE_EPISODE_LIMIT = 3;

const CONFIDENCE_ALPHA = 2;
const CONFIDENCE_BETA = 1;

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function computeConfidence(supportCount: number, contradictionCount: number): number {
  return clamp(
    (CONFIDENCE_ALPHA + supportCount) /
      (CONFIDENCE_ALPHA + CONFIDENCE_BETA + supportCount + contradictionCount),
    0,
    1,
  );
}

function toRecentDistinctEpisodeIds(
  events: Array<{ ts: number; provenance: Provenance }>,
): EpisodeId[] {
  const latestEpisodeTs = new Map<EpisodeId, number>();

  for (const event of events) {
    if (!isEpisodeProvenance(event.provenance)) {
      continue;
    }

    for (const episodeId of event.provenance.episode_ids) {
      const currentTs = latestEpisodeTs.get(episodeId) ?? Number.NEGATIVE_INFINITY;
      if (event.ts > currentTs) {
        latestEpisodeTs.set(episodeId, event.ts);
      }
    }
  }

  return [...latestEpisodeTs.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, EVIDENCE_EPISODE_LIMIT)
    .map(([episodeId]) => episodeId);
}

export type EvidenceSummary = {
  supportCount: number;
  contradictionCount: number;
  lastTestedAt: number | null;
  lastContradictedAt: number | null;
  evidenceEpisodeIds: EpisodeId[];
};

export function summarizeEvidence(
  supportEvents: Array<{ ts: number; provenance: Provenance }>,
  contradictionEvents: Array<{ ts: number; provenance: Provenance }>,
): EvidenceSummary {
  const episodeSupportEvents = supportEvents.filter((event) =>
    isEpisodeProvenance(event.provenance),
  );

  return {
    supportCount: episodeSupportEvents.length,
    contradictionCount: contradictionEvents.length,
    lastTestedAt:
      episodeSupportEvents.length === 0
        ? null
        : Math.max(...episodeSupportEvents.map((event) => event.ts)),
    lastContradictedAt:
      contradictionEvents.length === 0
        ? null
        : Math.max(...contradictionEvents.map((event) => event.ts)),
    evidenceEpisodeIds: toRecentDistinctEpisodeIds(episodeSupportEvents),
  };
}
