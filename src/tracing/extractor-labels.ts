export const EXTRACTOR_MAX_TOKEN_LLM_LABELS = [
  "action_state_extractor",
  "action-state-extractor",
  "closure_loop_classifier",
  "closure-loop-classifier",
  "corrective_preference_extractor",
  "corrective-preference-extractor",
  "creator_directive_extractor",
  "creator_directive_rendered",
  "creator-directive-extractor",
  "decision_artifact_compiler",
  "decision_artifact_semantic_revision",
  "entity_extractor",
  "frame_anomaly_classifier",
  "frame-anomaly-classifier",
  "goal_promotion_extractor",
  "goal-promotion-extractor",
  "mode_detector",
  "pending_action_judge",
  "pending-action-judge",
  "perception-entity-fallback",
  "perception-mode-fallback",
  "perception-temporal-cue",
  "procedural_context_extractor",
  "procedural-context",
  "shared_state_compiler",
  "shared_state_semantic_revision",
  "temporal_cue_extractor",
] as const;

const EXTRACTOR_MAX_TOKEN_LLM_LABEL_SET: ReadonlySet<string> = new Set(
  EXTRACTOR_MAX_TOKEN_LLM_LABELS,
);

export function isExtractorMaxTokenLlmLabel(label: string): boolean {
  return EXTRACTOR_MAX_TOKEN_LLM_LABEL_SET.has(label);
}
