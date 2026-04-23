/* Final retrieval context assembly helpers. */
import type { OpenQuestion } from "../memory/self/index.js";

import type { RetrievedEpisode } from "./scoring.js";
import type { RetrievedSemantic } from "./semantic-retrieval.js";

export type RetrievedContext = {
  episodes: RetrievedEpisode[];
  semantic: RetrievedSemantic;
  open_questions: OpenQuestion[];
  contradiction_present: boolean;
};

export function assembleRetrievedContext(input: {
  episodes: RetrievedEpisode[];
  semantic: RetrievedSemantic;
  openQuestions: OpenQuestion[];
  contradictionPresent: boolean;
}): RetrievedContext {
  return {
    episodes: input.episodes,
    semantic: input.semantic,
    open_questions: input.openQuestions,
    contradiction_present: input.contradictionPresent,
  };
}
