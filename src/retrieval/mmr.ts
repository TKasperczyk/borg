import { cosineSimilarity } from "./embedding-similarity.js";

// Tunes the default balance between relevance and diversity in MMR selection.
export const DEFAULT_MMR_LAMBDA = 0.7;

export type MmrCandidate<T> = {
  item: T;
  vector: Float32Array;
  relevanceScore: number;
};

export function applyMmr<T>(
  candidates: readonly MmrCandidate<T>[],
  options: { limit: number; lambda?: number },
): MmrCandidate<T>[] {
  const limit = Math.max(0, options.limit);

  if (limit === 0 || candidates.length === 0) {
    return [];
  }

  const requestedLambda = options.lambda ?? DEFAULT_MMR_LAMBDA;
  const lambda = Number.isFinite(requestedLambda)
    ? Math.min(1, Math.max(0, requestedLambda))
    : DEFAULT_MMR_LAMBDA;
  const remaining = [...candidates];
  const selected: MmrCandidate<T>[] = [];

  while (remaining.length > 0 && selected.length < limit) {
    let bestIndex = 0;
    let bestScore = Number.NEGATIVE_INFINITY;

    for (const [index, candidate] of remaining.entries()) {
      // The first pick has no redundancy term, so it always maximizes raw
      // relevance — lambda only trades relevance against diversity for later
      // picks. Without this, lambda 0 zeroed every first-pick score and
      // degraded the top selection to pool order.
      if (selected.length === 0) {
        if (candidate.relevanceScore > bestScore) {
          bestScore = candidate.relevanceScore;
          bestIndex = index;
        }
        continue;
      }

      const redundancy = Math.max(
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
