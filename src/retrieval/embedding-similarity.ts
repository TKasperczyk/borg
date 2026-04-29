export function cosineSimilarity(left: Float32Array, right: Float32Array): number {
  if (left.length !== right.length) {
    throw new RangeError(
      `Cannot compare embeddings with different dimensions: ${left.length} !== ${right.length}`,
    );
  }

  let dot = 0;
  let leftNormSquared = 0;
  let rightNormSquared = 0;

  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index] ?? 0;
    const rightValue = right[index] ?? 0;
    dot += leftValue * rightValue;
    leftNormSquared += leftValue * leftValue;
    rightNormSquared += rightValue * rightValue;
  }

  if (leftNormSquared === 0 || rightNormSquared === 0) {
    return 0;
  }

  return dot / (Math.sqrt(leftNormSquared) * Math.sqrt(rightNormSquared));
}

export function cosineSimilarity01(left: Float32Array, right: Float32Array): number {
  return Math.min(1, Math.max(0, (cosineSimilarity(left, right) + 1) / 2));
}

export function bestVectorSimilarity(
  target: Float32Array,
  candidates: readonly Float32Array[],
): number {
  let best = 0;

  for (const candidate of candidates) {
    best = Math.max(best, cosineSimilarity(target, candidate));
  }

  return Math.max(0, best);
}
