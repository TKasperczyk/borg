// Self-band data: the repositories for "who I am" records (values, goals,
// traits, autobiographical periods, growth markers, open questions).
// Governance over identity-bearing mutations (audit trail, guard, review-
// gated overwrites) does NOT live here -- it lives in memory/identity,
// which composes these repositories with memory/commitments and routes
// writes through IdentityService + IdentityGuard.

export {
  AutobiographicalRepository,
  autobiographicalPeriodIdSchema,
  autobiographicalPeriodPatchSchema,
  autobiographicalPeriodSchema,
  type AutobiographicalPeriod,
  type AutobiographicalPeriodPatch,
  type AutobiographicalRepositoryOptions,
} from "./autobiographical.js";
export {
  GROWTH_MARKER_CATEGORIES,
  GrowthMarkersRepository,
  growthMarkerCategorySchema,
  growthMarkerIdSchema,
  growthMarkerPatchSchema,
  growthMarkerSchema,
  type GrowthMarker,
  type GrowthMarkerCategory,
  type GrowthMarkerPatch,
  type GrowthMarkersRepositoryOptions,
  type GrowthMarkersSummary,
} from "./growth-markers.js";
export { selfMigrations } from "./migrations.js";
export {
  OPEN_QUESTION_SOURCES,
  OPEN_QUESTION_STATUSES,
  buildOpenQuestionDedupeKey,
  createOpenQuestionsTableSchema,
  OpenQuestionsRepository,
  openQuestionAudienceEntityIdSchema,
  openQuestionIdSchema,
  openQuestionPatchSchema,
  openQuestionSchema,
  openQuestionSourceSchema,
  openQuestionStatusSchema,
  type OpenQuestion,
  type OpenQuestionEmbeddingBackfillReport,
  type OpenQuestionEmbeddingFailureDetails,
  type OpenQuestionPatch,
  type OpenQuestionSearchCandidate,
  type OpenQuestionSource,
  type OpenQuestionsRepositoryOptions,
  type OpenQuestionStatus,
} from "./open-questions.js";
export {
  GoalsRepository,
  TraitsRepository,
  ValuesRepository,
  type GoalsRepositoryOptions,
  type TraitsRepositoryOptions,
  type ValuesRepositoryOptions,
} from "./repository.js";
export {
  appendInternalFailureEvent,
  enqueueOpenQuestionForReview,
  type ReviewOpenQuestionExtractorLike,
  type ReviewOpenQuestionHookOptions,
} from "./review-open-question-hook.js";
export {
  REVIEW_OPEN_QUESTION_TOOL,
  ReviewOpenQuestionExtractor,
  type OpenQuestionProposal,
  type ReviewOpenQuestionContext,
  type ReviewOpenQuestionExtractorDegradedEvent,
  type ReviewOpenQuestionExtractorOptions,
} from "./review-open-question-extractor.js";
export {
  goalIdSchema,
  goalPatchSchema,
  goalSchema,
  goalStatusSchema,
  traitIdSchema,
  traitPatchSchema,
  traitSchema,
  valueIdSchema,
  valuePatchSchema,
  valueSchema,
  type GoalRecord,
  type GoalPatch,
  type GoalStatus,
  type GoalTreeNode,
  type TraitPatch,
  type TraitRecord,
  type ValuePatch,
  type ValueRecord,
} from "./types.js";
