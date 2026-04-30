/* Final retrieval context assembly helpers. */
import type { OpenQuestion } from "../memory/self/index.js";

import { computeRetrievalConfidence, type RetrievalConfidence } from "./confidence.js";
import type { EvidenceItem, RecallIntent } from "./recall-types.js";
import type { RetrievedEpisode } from "./scoring.js";
import type { RetrievedSemantic } from "./semantic-retrieval.js";

export type RetrievedContext = {
  episodes: RetrievedEpisode[];
  semantic: RetrievedSemantic;
  open_questions: OpenQuestion[];
  evidence: EvidenceItem[];
  recall_intents: RecallIntent[];
  contradiction_present: boolean;
  confidence: RetrievalConfidence;
};

export function assembleRetrievedContext(input: {
  episodes: RetrievedEpisode[];
  semantic: RetrievedSemantic;
  openQuestions: OpenQuestion[];
  evidence: EvidenceItem[];
  recallIntents: RecallIntent[];
  contradictionPresent: boolean;
  nowMs: number;
  expectedCount?: number;
}): RetrievedContext {
  const contradictionEdges = input.semantic.contradiction_hits.flatMap((hit) => hit.edgePath);
  const confidence = computeRetrievalConfidence({
    episodes: input.episodes,
    contradictionPresent: input.contradictionPresent,
    contradictionEdges: contradictionEdges.length === 0 ? undefined : contradictionEdges,
    semanticEvidence: {
      matched_nodes: input.semantic.matched_nodes,
      support_hits: input.semantic.support_hits,
      causal_hits: input.semantic.causal_hits,
    },
    nowMs: input.nowMs,
    asOf: input.semantic.as_of ?? undefined,
    expectedCount: input.expectedCount,
  });

  return {
    episodes: input.episodes,
    semantic: input.semantic,
    open_questions: input.openQuestions,
    evidence: input.evidence,
    recall_intents: input.recallIntents,
    contradiction_present: input.contradictionPresent,
    confidence,
  };
}
