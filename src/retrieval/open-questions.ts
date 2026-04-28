/* Open-question scoring for retrieval context assembly. */
import type { EmbeddingClient } from "../embeddings/index.js";
import type { OpenQuestion, OpenQuestionsRepository } from "../memory/self/index.js";
import type { SemanticNode } from "../memory/semantic/types.js";
import type { EntityId } from "../util/ids.js";
import { tokenizeText } from "../util/text/tokenize.js";

function tokenOverlapScore(queryTokens: ReadonlySet<string>, question: OpenQuestion): number {
  const questionTokens = tokenizeText(question.question);
  let overlap = 0;

  for (const token of queryTokens) {
    if (questionTokens.has(token)) {
      overlap += 1;
    }
  }

  const unionSize = new Set([...queryTokens, ...questionTokens]).size;
  return unionSize === 0 ? 0 : overlap / unionSize;
}

export async function retrieveOpenQuestionsForQuery(
  openQuestionsRepository: OpenQuestionsRepository | undefined,
  embeddingClient: EmbeddingClient | undefined,
  query: string,
  options: {
    relatedSemanticNodeIds?: readonly SemanticNode["id"][];
    audienceEntityId?: EntityId | null;
    limit?: number;
    queryVector?: Float32Array;
  } = {},
): Promise<OpenQuestion[]> {
  if (openQuestionsRepository === undefined) {
    return [];
  }

  const queryTokens = tokenizeText(query);
  const relatedNodeIds = new Set(options.relatedSemanticNodeIds ?? []);
  const limit = Math.max(1, options.limit ?? 3);
  const listLimit = Math.max(100, limit * 10);
  const vectorSearchAttempted = embeddingClient !== undefined;
  const vectorCandidates =
    embeddingClient === undefined
      ? []
      : await openQuestionsRepository.searchByVector(
          options.queryVector ?? (await embeddingClient.embed(query)),
          {
            status: "open",
            visibleToAudienceEntityId: options.audienceEntityId ?? null,
            limit: listLimit,
          },
        );
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

  const embeddedIds = await openQuestionsRepository.getEmbeddedQuestionIds([
    ...candidateById.keys(),
  ]);
  const scored = [...candidateById.values()]
    .map((question) => {
      const similarityScore = similarityById.get(question.id) ?? 0;
      const fallbackScore =
        vectorSearchAttempted && (similarityById.has(question.id) || embeddedIds.has(question.id))
          ? 0
          : tokenOverlapScore(queryTokens, question);
      const baseScore = similarityScore > 0 ? similarityScore : fallbackScore;
      const relatedScore = question.related_semantic_node_ids.some((id) => relatedNodeIds.has(id))
        ? 0.35
        : 0;
      const score =
        baseScore === 0 && relatedScore === 0
          ? 0
          : baseScore + relatedScore + question.urgency * 0.15;

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
