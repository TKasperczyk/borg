/* Final retrieval context assembly helpers. */
import type { OpenQuestion } from "../memory/self/index.js";

import { computeRetrievalConfidence, type RetrievalConfidence } from "./confidence.js";
import type { RetrievedEpisode } from "./scoring.js";
import type { RetrievedSemantic } from "./semantic-retrieval.js";

export type RetrievedContext = {
  episodes: RetrievedEpisode[];
  semantic: RetrievedSemantic;
  open_questions: OpenQuestion[];
  contradiction_present: boolean;
  confidence: RetrievalConfidence;
};

export function assembleRetrievedContext(input: {
  episodes: RetrievedEpisode[];
  semantic: RetrievedSemantic;
  openQuestions: OpenQuestion[];
  contradictionPresent: boolean;
  expectedCount?: number;
}): RetrievedContext {
  const contradictionEdges = input.semantic.contradiction_hits.flatMap((hit) => hit.edgePath);
  const confidence = computeRetrievalConfidence({
    episodes: input.episodes,
    contradictionPresent: input.contradictionPresent,
    contradictionEdges: contradictionEdges.length === 0 ? undefined : contradictionEdges,
    asOf: input.semantic.as_of ?? undefined,
    expectedCount: input.expectedCount,
  });

  return {
    episodes: input.episodes,
    semantic: input.semantic,
    open_questions: input.openQuestions,
    contradiction_present: input.contradictionPresent,
    confidence,
  };
}
