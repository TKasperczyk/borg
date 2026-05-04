import { computeGoalRelevanceFromEmbeddings } from "./embedding-relevance.js";

export function computeGoalRelevance(input: {
  episodeEmbedding: Float32Array;
  goalVectors: readonly Float32Array[];
  primaryGoalVector?: Float32Array;
}): number {
  return computeGoalRelevanceFromEmbeddings(input);
}
