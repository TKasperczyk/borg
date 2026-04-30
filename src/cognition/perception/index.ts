export { detectAffectiveSignal } from "./affective-signal.js";
export { EntityExtractor, type EntityExtractorOptions } from "./entity-extractor.js";
export {
  FACTUAL_CHALLENGE_TOOL,
  detectFactualChallenge,
  type FactualChallengeDetectorDegradedReason,
  type FactualChallengeDetectorOptions,
} from "./factual-challenge.js";
export { ModeDetector, type ModeDetectorOptions } from "./mode-detector.js";
export {
  Perceiver,
  perceive,
  runPerceptionClassifierSafely,
  type PerceiverOptions,
  type PerceptionClassifierFailure,
  type PerceptionClassifierFailureObserver,
  type PerceptionClassifierName,
} from "./perceive.js";
export { detectTemporalCue } from "./temporal-cue.js";
