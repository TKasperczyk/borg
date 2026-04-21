export {
  AffectiveExtractor,
  analyzeAffectiveSignalHeuristically,
  type AffectiveExtractorOptions,
} from "./extractor.js";
export { affectiveMigrations } from "./migrations.js";
export { MoodRepository, type MoodRepositoryOptions } from "./mood.js";
export {
  DOMINANT_EMOTIONS,
  affectiveSignalSchema,
  createNeutralAffectiveSignal,
  createNeutralEmotionalArc,
  dominantEmotionSchema,
  emotionalArcSchema,
  moodHistoryEntrySchema,
  moodStateSchema,
  type AffectiveSignal,
  type DominantEmotion,
  type EmotionalArc,
  type MoodHistoryEntry,
  type MoodState,
} from "./types.js";
