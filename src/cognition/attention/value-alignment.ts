import { computeValueAlignmentFromEmbeddings } from "./embedding-relevance.js";

export function computeValueAlignment(input: {
  episodeEmbedding: Float32Array;
  valueVectors: readonly Float32Array[];
}): number {
  return computeValueAlignmentFromEmbeddings(input);
}
