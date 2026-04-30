/* Episodic candidate merge helpers. Recall Core owns adapter fanout. */
import type { EpisodeSearchCandidate } from "../memory/episodic/types.js";

export type EpisodeCandidateSource =
  | "vector"
  | "temporal"
  | "audience"
  | "entity"
  | "recent"
  | "heat";

export type MergedEpisodeCandidate = {
  candidate: EpisodeSearchCandidate;
  sources: Set<EpisodeCandidateSource>;
};

export function tagCandidates(
  source: EpisodeCandidateSource,
  candidates: readonly EpisodeSearchCandidate[],
): MergedEpisodeCandidate[] {
  return candidates.map((candidate) => ({
    candidate,
    sources: new Set([source]),
  }));
}

export function mergeCandidates(
  candidateSets: readonly MergedEpisodeCandidate[][],
): MergedEpisodeCandidate[] {
  const merged = new Map<string, MergedEpisodeCandidate>();

  for (const candidateSet of candidateSets) {
    for (const entry of candidateSet) {
      const existing = merged.get(entry.candidate.episode.id);

      if (existing === undefined) {
        merged.set(entry.candidate.episode.id, {
          candidate: {
            ...entry.candidate,
          },
          sources: new Set(entry.sources),
        });
        continue;
      }

      existing.candidate = {
        ...existing.candidate,
        similarity: Math.max(existing.candidate.similarity, entry.candidate.similarity),
        stats: entry.candidate.stats,
      };

      for (const source of entry.sources) {
        existing.sources.add(source);
      }
    }
  }

  return [...merged.values()];
}
