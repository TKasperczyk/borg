/* Open-question scoring for retrieval context assembly. */
import type { EmbeddingClient } from "../embeddings/index.js";
import type { OpenQuestion, OpenQuestionsRepository } from "../memory/self/index.js";
import type { SemanticNode } from "../memory/semantic/types.js";
import type { EntityId } from "../util/ids.js";

const DEFAULT_OPEN_QUESTION_MIN_SIMILARITY = 0.01;

export async function retrieveOpenQuestionsForQuery(
  openQuestionsRepository: OpenQuestionsRepository | undefined,
  embeddingClient: EmbeddingClient | undefined,
  query: string,
  options: {
    relatedSemanticNodeIds?: readonly SemanticNode["id"][];
    audienceEntityId?: EntityId | null;
    limit?: number;
    queryVector?: Float32Array;
    onDegraded?: (reason: string, error?: unknown) => void | Promise<void>;
  } = {},
): Promise<OpenQuestion[]> {
  if (openQuestionsRepository === undefined) {
    return [];
  }

  const relatedNodeIds = new Set(options.relatedSemanticNodeIds ?? []);
  const limit = Math.max(1, options.limit ?? 3);
  const listLimit = Math.max(100, limit * 10);
  let vectorCandidates: Awaited<ReturnType<OpenQuestionsRepository["searchByVector"]>> = [];

  if (embeddingClient !== undefined) {
    try {
      vectorCandidates = await openQuestionsRepository.searchByVector(
        options.queryVector ?? (await embeddingClient.embed(query)),
        {
          status: "open",
          visibleToAudienceEntityId: options.audienceEntityId ?? null,
          limit: listLimit,
          minSimilarity: DEFAULT_OPEN_QUESTION_MIN_SIMILARITY,
        },
      );
    } catch (error) {
      await options.onDegraded?.("open_question_vector_search_failed", error);
    }
  } else {
    await options.onDegraded?.("open_question_embedding_unavailable");
  }

  const listCandidates = openQuestionsRepository.list({
    status: "open",
    visibleToAudienceEntityId: options.audienceEntityId ?? null,
    limit: listLimit,
  });
  const candidateById = new Map<OpenQuestion["id"], OpenQuestion>();
  const similarityById = new Map<OpenQuestion["id"], number>();

  for (const candidate of vectorCandidates) {
    candidateById.set(candidate.question.id, candidate.question);
    similarityById.set(candidate.question.id, candidate.similarity);
  }

  for (const question of listCandidates) {
    candidateById.set(question.id, question);
  }

  const scored = [...candidateById.values()]
    .map((question) => {
      const similarityScore = similarityById.get(question.id) ?? 0;
      const relatedScore = question.related_semantic_node_ids.some((id) => relatedNodeIds.has(id))
        ? 0.35
        : 0;
      const score =
        similarityScore === 0 && relatedScore === 0
          ? 0
          : similarityScore + relatedScore + question.urgency * 0.15;

      return {
        question,
        score,
      };
    })
    .filter((item) => item.score > 0)
    .sort(
      (left, right) =>
        right.score - left.score ||
        right.question.urgency - left.question.urgency ||
        right.question.last_touched - left.question.last_touched,
    );

  return scored.slice(0, limit).map((item) => item.question);
}
