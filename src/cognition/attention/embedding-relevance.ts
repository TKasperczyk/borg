import { bestVectorSimilarity } from "../../retrieval/embedding-similarity.js";

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

  return Math.min(1, Math.max(broad, primary * 1.25));
}

export function computeValueAlignmentFromEmbeddings(input: {
  episodeEmbedding: Float32Array;
  valueVectors: readonly Float32Array[];
}): number {
  return bestVectorSimilarity(input.episodeEmbedding, input.valueVectors);
}
