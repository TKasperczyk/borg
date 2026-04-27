/* Open-question scoring for retrieval context assembly. */
import type { OpenQuestion, OpenQuestionsRepository } from "../memory/self/index.js";
import type { SemanticNode } from "../memory/semantic/types.js";
import type { EntityId } from "../util/ids.js";
import { tokenizeText } from "../util/text/tokenize.js";

export function retrieveOpenQuestionsForQuery(
  openQuestionsRepository: OpenQuestionsRepository | undefined,
  query: string,
  options: {
    relatedSemanticNodeIds?: readonly SemanticNode["id"][];
    audienceEntityId?: EntityId | null;
    limit?: number;
  } = {},
): OpenQuestion[] {
  if (openQuestionsRepository === undefined) {
    return [];
  }

  const queryTokens = tokenizeText(query);
  const relatedNodeIds = new Set(options.relatedSemanticNodeIds ?? []);
  const limit = Math.max(1, options.limit ?? 3);
  const candidates = openQuestionsRepository.list({
    status: "open",
    visibleToAudienceEntityId: options.audienceEntityId ?? null,
    limit: 100,
  });
  const scored = candidates
    .map((question) => {
      const questionTokens = tokenizeText(question.question);
      let overlap = 0;

      for (const token of queryTokens) {
        if (questionTokens.has(token)) {
          overlap += 1;
        }
      }

      const unionSize = new Set([...queryTokens, ...questionTokens]).size;
      const tokenScore = unionSize === 0 ? 0 : overlap / unionSize;
      const relatedScore = question.related_semantic_node_ids.some((id) => relatedNodeIds.has(id))
        ? 0.35
        : 0;
      const score =
        tokenScore === 0 && relatedScore === 0
          ? 0
          : tokenScore + relatedScore + question.urgency * 0.15;

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
