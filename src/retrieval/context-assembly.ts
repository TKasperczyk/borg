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
  const confidence = computeRetrievalConfidence({
    episodes: input.episodes,
    contradictionPresent: input.contradictionPresent,
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
