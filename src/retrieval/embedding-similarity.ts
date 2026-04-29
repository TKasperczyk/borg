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
    const rawLeftValue = left[index] ?? 0;
    const rawRightValue = right[index] ?? 0;
    const leftValue = Number.isFinite(rawLeftValue) ? rawLeftValue : 0;
    const rightValue = Number.isFinite(rawRightValue) ? rawRightValue : 0;
    dot += leftValue * rightValue;
    leftNormSquared += leftValue * leftValue;
    rightNormSquared += rightValue * rightValue;
  }

  if (
    !Number.isFinite(dot) ||
    !Number.isFinite(leftNormSquared) ||
    !Number.isFinite(rightNormSquared) ||
    leftNormSquared === 0 ||
    rightNormSquared === 0
  ) {
    return 0;
  }

  const similarity = dot / (Math.sqrt(leftNormSquared) * Math.sqrt(rightNormSquared));

  return Number.isFinite(similarity) ? similarity : 0;
}

export function cosineSimilarity01(left: Float32Array, right: Float32Array): number {
  const similarity = cosineSimilarity(left, right);

  if (!Number.isFinite(similarity)) {
    return 0;
  }

  return Math.min(1, Math.max(0, (similarity + 1) / 2));
}

export function bestVectorSimilarity(
  target: Float32Array,
  candidates: readonly Float32Array[],
): number {
  let best = 0;

  for (const candidate of candidates) {
    const similarity = cosineSimilarity(target, candidate);

    if (Number.isFinite(similarity)) {
      best = Math.max(best, similarity);
    }
  }

  return Number.isFinite(best) ? Math.max(0, best) : 0;
}
