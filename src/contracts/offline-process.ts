export const OFFLINE_PROCESS_NAMES = [
  "consolidator",
  "reflector",
  "semantic-extractor",
  "curator",
  "overseer",
  "associator",
  "review-resolver",
  "ruminator",
  "self-narrator",
  "procedural-synthesizer",
  "belief-reviser",
  "creator-directive-reconciler",
  "lived-experience-day-summarizer",
  "commitment-reconciler",
] as const;

export type OfflineProcessName = (typeof OFFLINE_PROCESS_NAMES)[number];
