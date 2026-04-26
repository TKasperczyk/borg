/* Episodic candidate generation for multi-lane retrieval. */
import type { EpisodicRepository } from "../memory/episodic/repository.js";
import type {
  Episode,
  EpisodeSearchCandidate,
  EpisodeSearchOptions,
} from "../memory/episodic/types.js";
import type { EntityId } from "../util/ids.js";

import {
  overlapsTimeRange,
  type ResolvedTimeRange,
  type ResolvedTimeSignals,
} from "./time-signals.js";

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

export type EpisodicCandidateSearchOptions = EpisodeSearchOptions & {
  entityTerms?: readonly string[];
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

export async function searchByVector(
  repository: EpisodicRepository,
  queryVector: Float32Array,
  options: EpisodeSearchOptions,
): Promise<EpisodeSearchCandidate[]> {
  return repository.searchByVector(queryVector, options);
}

export async function searchByTimeRange(
  repository: EpisodicRepository,
  range: ResolvedTimeRange,
  options: Pick<
    EpisodeSearchOptions,
    "audienceEntityId" | "crossAudience" | "globalIdentitySelfAudienceEntityId"
  > & {
    limit?: number;
  },
): Promise<EpisodeSearchCandidate[]> {
  return repository.searchByTimeRange(range, options);
}

async function generateAudienceCandidates(
  repository: EpisodicRepository,
  audienceEntityId: EntityId,
  limit: number,
  visibleEpisodes: Promise<readonly Episode[]>,
): Promise<MergedEpisodeCandidate[]> {
  const recentLimit = Math.max(1, Math.ceil(limit / 2));
  const heatLimit = Math.max(1, limit - recentLimit);
  const sharedVisibleEpisodes = await visibleEpisodes;
  const [recent, hottest] = await Promise.all([
    repository.listByAudience(audienceEntityId, {
      limit: recentLimit,
      orderBy: "recent",
      visibleEpisodes: sharedVisibleEpisodes,
    }),
    repository.listByAudience(audienceEntityId, {
      limit: heatLimit,
      orderBy: "heat",
      visibleEpisodes: sharedVisibleEpisodes,
    }),
  ]);

  return mergeCandidates([tagCandidates("audience", recent), tagCandidates("audience", hottest)]);
}

async function generateRecentAndHeatCandidates(
  repository: EpisodicRepository,
  options: EpisodicCandidateSearchOptions,
  limit: number,
  visibleEpisodes: Promise<readonly Episode[]>,
): Promise<MergedEpisodeCandidate[]> {
  const recentLimit = Math.max(1, Math.ceil(limit / 2));
  const heatLimit = Math.max(1, limit - recentLimit);
  const sharedVisibleEpisodes = await visibleEpisodes;
  const [recent, hottest] = await Promise.all([
    repository.listRecent({
      limit: recentLimit,
      audienceEntityId: options.audienceEntityId,
      crossAudience: options.crossAudience,
      visibleEpisodes: sharedVisibleEpisodes,
    }),
    repository.listHottest({
      limit: heatLimit,
      audienceEntityId: options.audienceEntityId,
      crossAudience: options.crossAudience,
      visibleEpisodes: sharedVisibleEpisodes,
    }),
  ]);

  return mergeCandidates([tagCandidates("recent", recent), tagCandidates("heat", hottest)]);
}

export async function generateEpisodicCandidates(params: {
  repository: EpisodicRepository;
  query: string;
  queryVector: Float32Array;
  options: EpisodicCandidateSearchOptions;
  limit: number;
  timeSignals: ResolvedTimeSignals;
}): Promise<MergedEpisodeCandidate[]> {
  const { repository, queryVector, options, limit, timeSignals } = params;
  const vectorBudget = Math.max(limit * 2, 12);
  const temporalBudget = Math.max(limit * 2, 8);
  const audienceBudget = Math.max(limit * 2, 8);
  const entityBudget = Math.max(limit * 2, 8);
  const recentHeatBudget = Math.max(limit, 4);
  const visibleEpisodes = repository.listVisibleEpisodes({
    audienceEntityId: options.audienceEntityId,
    crossAudience: options.crossAudience,
  });
  const vectorSearchOptions: EpisodeSearchOptions = {
    ...options,
    timeRange: timeSignals.strictFilterRange ?? undefined,
  };
  const generatorCalls: Array<Promise<MergedEpisodeCandidate[]>> = [
    searchByVector(repository, queryVector, {
      ...vectorSearchOptions,
      limit: vectorBudget,
    }).then((candidates) => tagCandidates("vector", candidates)),
    generateRecentAndHeatCandidates(repository, options, recentHeatBudget, visibleEpisodes),
  ];

  if (timeSignals.scoringRange !== null) {
    generatorCalls.push(
      searchByTimeRange(repository, timeSignals.scoringRange, {
        limit: temporalBudget,
        audienceEntityId: options.audienceEntityId,
        crossAudience: options.crossAudience,
        globalIdentitySelfAudienceEntityId: options.globalIdentitySelfAudienceEntityId,
      }).then((candidates) => tagCandidates("temporal", candidates)),
    );
  }

  if (
    options.audienceEntityId !== null &&
    options.audienceEntityId !== undefined &&
    options.crossAudience !== true
  ) {
    generatorCalls.push(
      generateAudienceCandidates(
        repository,
        options.audienceEntityId,
        audienceBudget,
        visibleEpisodes,
      ),
    );
  }

  if (options.entityTerms !== undefined && options.entityTerms.length > 0) {
    generatorCalls.push(
      repository
        .searchByParticipantsOrTags(options.entityTerms, {
          limit: entityBudget,
          audienceEntityId: options.audienceEntityId,
          crossAudience: options.crossAudience,
          visibleEpisodes: await visibleEpisodes,
        })
        .then((candidates) => tagCandidates("entity", candidates)),
    );
  }

  // The strict time filter producing zero candidates is a normal user-query
  // outcome ("asked about yesterday, nothing happened"). The retrieval_completed
  // tracer event already records episodeCount and the time range from options,
  // so this state is observable without spamming stdout.
  return mergeCandidates(await Promise.all(generatorCalls)).filter((entry) =>
    timeSignals.strictFilterRange === null
      ? true
      : overlapsTimeRange(entry.candidate.episode, timeSignals.strictFilterRange),
  );
}
