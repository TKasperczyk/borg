export { applyEpisodeDecay, type DecayOptions, type DecayResult } from "./decay.js";
export {
  filterEpisodesByAudience,
  inferSinglePrivateAudience,
  isEpisodeAccessVisible,
  type AudienceEpisodeAccess,
  type AudienceFilterResult,
  type AudiencePolicy,
} from "./audience-filter.js";
export {
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
export {
  collectProtectedEpisodeTokenLines,
  preserveProtectedEpisodeTokenLines,
} from "./protected-lines.js";
export {
  episodeParticipantDisplayNames,
  episodeParticipantEntityIds,
  episodeParticipantEntityIdTerm,
  parseEpisodeParticipantEntityIdTerm,
} from "./participant-terms.js";
export { episodicMigrations } from "./migrations.js";
export {
  EpisodicRepository,
  buildConsolidationCoverageHash,
  createEpisodesTableSchema,
  type ConsolidationFamilyRecord,
  type ConsolidationMemberInput,
  type ConsolidationMemberRecord,
  type EpisodeGetOptions,
  type EpisodicRepositoryOptions,
  type ReconciliationReport,
} from "./repository.js";
export {
  EPISODE_TIERS,
  EPISODE_KINDS,
  consolidationFamilyIdSchema,
  episodeIdSchema,
  episodeInsertSchema,
  episodeKindSchema,
  episodeLineageSchema,
  episodePatchSchema,
  episodeSchema,
  episodeStatsPatchSchema,
  episodeStatsSchema,
  episodeTierSchema,
  type Episode,
  type EpisodeKind,
  type EpisodeListOptions,
  type EpisodeListResult,
  type EpisodePatch,
  type EpisodeSearchCandidate,
  type EpisodeSearchOptions,
  type EpisodeStats,
  type EpisodeStatsPatch,
  type EpisodeTier,
} from "./types.js";
