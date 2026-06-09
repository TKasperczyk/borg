import { bestVectorSimilarity } from "./embedding-similarity.js";

// Tunes how much the selected primary goal can outrank the broad goal set.
const PRIMARY_GOAL_RELEVANCE_BOOST = 1.25;

export function computeGoalRelevanceFromEmbeddings(input: {
  episodeEmbedding: Float32Array;
  goalVectors: readonly Float32Array[];
  primaryGoalVector?: Float32Array;
}): number {
  const broad = bestVectorSimilarity(input.episodeEmbedding, input.goalVectors);
  const primary =
    input.primaryGoalVector === undefined
      ? 0
      : bestVectorSimilarity(input.episodeEmbedding, [input.primaryGoalVector]);

  return Math.min(1, Math.max(broad, primary * PRIMARY_GOAL_RELEVANCE_BOOST));
}

export function computeValueAlignmentFromEmbeddings(input: {
  episodeEmbedding: Float32Array;
  valueVectors: readonly Float32Array[];
}): number {
  return bestVectorSimilarity(input.episodeEmbedding, input.valueVectors);
}

export function computeGoalRelevance(input: {
  episodeEmbedding: Float32Array;
  goalVectors: readonly Float32Array[];
  primaryGoalVector?: Float32Array;
}): number {
  return computeGoalRelevanceFromEmbeddings(input);
}

export function computeValueAlignment(input: {
  episodeEmbedding: Float32Array;
  valueVectors: readonly Float32Array[];
}): number {
  return computeValueAlignmentFromEmbeddings(input);
}
