import { type EpisodeId } from "../../../util/ids.js";
import { isEpisodeProvenance, type Provenance } from "../../common/provenance.js";
import { type ValueRecord } from "../types.js";

export const VALUE_PROMOTION_THRESHOLD = 3;
export const TRAIT_PROMOTION_THRESHOLD = 5;

const PROMOTION_PROVENANCE_EPISODE_LIMIT = 3;

export type PromotionMetadata = Pick<ValueRecord, "state" | "established_at"> & {
  promotionProvenance: Provenance | null;
};

export function getPromotionMetadataFromEvents<T extends { ts: number; provenance: Provenance }>(
  events: T[],
  threshold: number,
): PromotionMetadata {
  const distinctEpisodeIds = new Set<EpisodeId>();
  const latestEpisodeTs = new Map<EpisodeId, number>();
  let establishedAt: number | null = null;

  for (const event of events) {
    if (!isEpisodeProvenance(event.provenance)) {
      continue;
    }

    for (const episodeId of event.provenance.episode_ids) {
      distinctEpisodeIds.add(episodeId);
      latestEpisodeTs.set(episodeId, event.ts);
    }

    if (distinctEpisodeIds.size >= threshold) {
      establishedAt = event.ts;
      break;
    }
  }

  if (establishedAt === null) {
    return {
      state: "candidate",
      established_at: null,
      promotionProvenance: null,
    };
  }

  const promotionEpisodeIds = [...latestEpisodeTs.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, PROMOTION_PROVENANCE_EPISODE_LIMIT)
    .map(([episodeId]) => episodeId);

  return {
    state: "established",
    established_at: establishedAt,
    promotionProvenance: {
      kind: "episodes",
      episode_ids: promotionEpisodeIds,
    },
  };
}

export function resolveValueInitialState(
  provenance: Provenance,
  timestamp: number,
): Pick<ValueRecord, "state" | "established_at"> {
  switch (provenance.kind) {
    case "manual":
    case "system":
      return {
        state: "established",
        established_at: timestamp,
      };
    case "episodes":
      return {
        state:
          new Set(provenance.episode_ids).size >= VALUE_PROMOTION_THRESHOLD
            ? "established"
            : "candidate",
        established_at:
          new Set(provenance.episode_ids).size >= VALUE_PROMOTION_THRESHOLD ? timestamp : null,
      };
    case "offline":
    case "online":
      return {
        state: "candidate",
        established_at: null,
      };
  }
}
