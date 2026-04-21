export type MmrCandidate<T> = {
  item: T;
  vector: Float32Array;
  relevanceScore: number;
};

function dot(left: Float32Array, right: Float32Array): number {
  let sum = 0;

  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index];
    const rightValue = right[index];

    if (leftValue === undefined || rightValue === undefined) {
      continue;
    }

    sum += leftValue * rightValue;
  }

  return sum;
}

function magnitude(vector: Float32Array): number {
  let sum = 0;

  for (const value of vector) {
    sum += value * value;
  }

  return Math.sqrt(sum);
}

function cosineSimilarity(left: Float32Array, right: Float32Array): number {
  const denominator = magnitude(left) * magnitude(right);

  if (denominator === 0) {
    return 0;
  }

  return dot(left, right) / denominator;
}

export function applyMmr<T>(
  candidates: readonly MmrCandidate<T>[],
  options: { limit: number; lambda?: number },
): MmrCandidate<T>[] {
  const limit = Math.max(0, options.limit);

  if (limit === 0 || candidates.length === 0) {
    return [];
  }

  const lambda = options.lambda ?? 0.7;
  const remaining = [...candidates];
  const selected: MmrCandidate<T>[] = [];

  while (remaining.length > 0 && selected.length < limit) {
    let bestIndex = 0;
    let bestScore = Number.NEGATIVE_INFINITY;

    for (const [index, candidate] of remaining.entries()) {
      const redundancy =
        selected.length === 0
          ? 0
          : Math.max(
              ...selected.map((chosen) => cosineSimilarity(candidate.vector, chosen.vector)),
            );
      const score = lambda * candidate.relevanceScore - (1 - lambda) * redundancy;

      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }

    const [next] = remaining.splice(bestIndex, 1);

    if (next !== undefined) {
      selected.push(next);
    }
  }

  return selected;
}
