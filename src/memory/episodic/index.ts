export { applyEpisodeDecay, type DecayOptions, type DecayResult } from "./decay.js";
export {
  episodeAccessScopeKey,
  hasSameEpisodeAccessScope,
  isEpisodeInGlobalIdentityScope,
  isEpisodeVisibleToAudience,
  normalizeEpisodeAccess,
  type EpisodeAccessLike,
} from "./access.js";
export {
  EpisodicExtractor,
  type EpisodicExtractorOptions,
  type ExtractFromStreamOptions,
  type ExtractFromStreamResult,
} from "./extractor.js";
export { computeEpisodeHeat } from "./heat.js";
export { episodicMigrations } from "./migrations.js";
export {
  EpisodicRepository,
  createEpisodesTableSchema,
  type EpisodicRepositoryOptions,
  type ReconciliationReport,
} from "./repository.js";
export {
  EPISODE_TIERS,
  episodeIdSchema,
  episodeInsertSchema,
  episodeLineageSchema,
  episodePatchSchema,
  episodeSchema,
  episodeStatsPatchSchema,
  episodeStatsSchema,
  episodeTierSchema,
  type Episode,
  type EpisodeListOptions,
  type EpisodeListResult,
  type EpisodePatch,
  type EpisodeSearchCandidate,
  type EpisodeSearchOptions,
  type EpisodeStats,
  type EpisodeStatsPatch,
  type EpisodeTier,
} from "./types.js";
