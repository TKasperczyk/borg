/* Final retrieval context assembly helpers. */
import { createHash } from "node:crypto";

import type { OpenQuestion } from "../memory/self/index.js";

import { computeRetrievalConfidence, type RetrievalConfidence } from "./confidence.js";
import type { EvidenceItem, RecallIntent } from "./recall-types.js";
import type { RetrievedEpisode } from "./scoring.js";
import type { RetrievedSemantic } from "./semantic-retrieval.js";

export type RetrievedContradictionSessionScope = "current" | "prior" | "unknown";

export type RetrievedContradictionRoutingItem = {
  edgeId?: string;
  nodeIds: string[];
  sourceEpisodeIds: string[];
  validUntil?: number | null;
  sessionScope?: RetrievedContradictionSessionScope;
  linkedOpenQuestionIds: string[];
  fingerprint: string;
};

export type RetrievedContradictionRouting = {
  contradictions: RetrievedContradictionRoutingItem[];
};

export type RetrievedContext = {
  episodes: RetrievedEpisode[];
  semantic: RetrievedSemantic;
  open_questions: OpenQuestion[];
  evidence: EvidenceItem[];
  recall_intents: RecallIntent[];
  contradiction_present: boolean;
  contradictionRouting: RetrievedContradictionRouting;
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
    contradictionRouting: buildContradictionRouting(input.semantic, input.openQuestions),
    confidence,
  };
}

function buildContradictionRouting(
  semantic: RetrievedSemantic,
  openQuestions: readonly OpenQuestion[],
): RetrievedContradictionRouting {
  const byFingerprint = new Map<string, RetrievedContradictionRoutingItem>();

  for (const hit of semantic.contradiction_hits) {
    const contradictionEdge = hit.edgePath.find((edge) => edge.relation === "contradicts");
    const edge = contradictionEdge ?? hit.edgePath.at(-1);
    const edgeId = edge?.id;
    const nodeIds = uniqueSorted([
      hit.root_node_id,
      hit.node.id,
      ...hit.edgePath.flatMap((pathEdge) => [pathEdge.from_node_id, pathEdge.to_node_id]),
    ]);
    const sourceEpisodeIds = uniqueSorted([
      ...hit.node.source_episode_ids,
      ...hit.edgePath.flatMap((pathEdge) => pathEdge.evidence_episode_ids),
    ]);
    const fingerprint = contradictionFingerprint(edgeId, nodeIds);
    const linkedOpenQuestionIds = linkedOpenQuestions(nodeIds, sourceEpisodeIds, openQuestions);
    const next: RetrievedContradictionRoutingItem = {
      ...(edgeId === undefined ? {} : { edgeId }),
      nodeIds,
      sourceEpisodeIds,
      validUntil: edge?.valid_to ?? null,
      sessionScope: "unknown",
      linkedOpenQuestionIds,
      fingerprint,
    };
    const existing = byFingerprint.get(fingerprint);

    byFingerprint.set(
      fingerprint,
      existing === undefined ? next : mergeContradictions(existing, next),
    );
  }

  return {
    contradictions: [...byFingerprint.values()].sort((left, right) =>
      left.fingerprint.localeCompare(right.fingerprint),
    ),
  };
}

function contradictionFingerprint(edgeId: string | undefined, nodeIds: readonly string[]): string {
  const handles = [
    ...(edgeId === undefined ? [] : [`edge:${edgeId}`]),
    ...nodeIds.map((id) => `node:${id}`),
  ]
    .sort()
    .join("|");

  return createHash("sha1").update(handles).digest("hex");
}

function linkedOpenQuestions(
  nodeIds: readonly string[],
  sourceEpisodeIds: readonly string[],
  openQuestions: readonly OpenQuestion[],
): string[] {
  const nodeIdSet = new Set(nodeIds);
  const episodeIdSet = new Set(sourceEpisodeIds);

  return uniqueSorted(
    openQuestions
      .filter(
        (question) =>
          question.source === "contradiction" &&
          (question.related_semantic_node_ids.some((id) => nodeIdSet.has(id)) ||
            question.related_episode_ids.some((id) => episodeIdSet.has(id))),
      )
      .map((question) => question.id),
  );
}

function mergeContradictions(
  left: RetrievedContradictionRoutingItem,
  right: RetrievedContradictionRoutingItem,
): RetrievedContradictionRoutingItem {
  return {
    ...(left.edgeId === undefined && right.edgeId === undefined
      ? {}
      : { edgeId: left.edgeId ?? right.edgeId }),
    nodeIds: uniqueSorted([...left.nodeIds, ...right.nodeIds]),
    sourceEpisodeIds: uniqueSorted([...left.sourceEpisodeIds, ...right.sourceEpisodeIds]),
    validUntil:
      left.validUntil === null || right.validUntil === null
        ? null
        : (left.validUntil ?? right.validUntil),
    sessionScope: mergeSessionScope(left.sessionScope, right.sessionScope),
    linkedOpenQuestionIds: uniqueSorted([
      ...left.linkedOpenQuestionIds,
      ...right.linkedOpenQuestionIds,
    ]),
    fingerprint: left.fingerprint,
  };
}

function mergeSessionScope(
  left: RetrievedContradictionSessionScope | undefined,
  right: RetrievedContradictionSessionScope | undefined,
): RetrievedContradictionSessionScope {
  if (left === "current" || right === "current") {
    return "current";
  }

  if (left === "prior" && right === "prior") {
    return "prior";
  }

  return "unknown";
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}
