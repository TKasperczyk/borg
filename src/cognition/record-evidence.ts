import type { EvidenceItem, RetrievedEpisode, RetrievedSemantic } from "../retrieval/index.js";
import { deriveRecallEvidenceHandle } from "../retrieval/recall-state.js";
import type { AnnotationTarget } from "../memory/identity/annotations.js";
import type { AnnotationPromptOptions } from "../memory/common/episode-evidence.js";
import {
  readRecordEvidenceContexts,
  recordEvidenceContext,
  type RecordEvidenceReader,
} from "../memory/common/record-evidence.js";

export function evidenceItemAnnotationTarget(item: EvidenceItem): AnnotationTarget | undefined {
  if (item.source === "warm_recall") {
    // The retained handle identifies the record itself. An edge's node id, raw-stream
    // parentEpisodeId and derived-source ids never transfer another record's annotation.
    const handle = deriveRecallEvidenceHandle(item);
    switch (handle?.source) {
      case "episode":
        return { record_type: "episode", record_id: handle.episodeId };
      case "semantic_node":
        return { record_type: "semantic_node", record_id: handle.nodeId };
      case "semantic_edge":
        return { record_type: "semantic_edge", record_id: handle.edgeId };
      default:
        return undefined;
    }
  }
  if (item.source === "episode" && item.provenance?.episodeId !== undefined)
    return { record_type: "episode", record_id: item.provenance.episodeId };
  if (
    (item.source === "semantic_node" || item.source === "semantic_edge") &&
    item.provenance?.nodeId !== undefined
  )
    return { record_type: "semantic_node", record_id: item.provenance.nodeId };
  return undefined;
}

export type RetrievedRecordEvidence = {
  episodes?: readonly RetrievedEpisode[];
  semantic?: RetrievedSemantic | null;
  evidence?: readonly EvidenceItem[];
};

export function readRetrievedRecordEvidence(
  reader: RecordEvidenceReader | undefined,
  input: RetrievedRecordEvidence,
  additionalTargets: readonly AnnotationTarget[] = [],
  options: AnnotationPromptOptions = {},
) {
  const semantic = input.semantic;
  const nodes =
    semantic == null
      ? []
      : [
          ...semantic.matched_nodes,
          ...semantic.supports,
          ...semantic.contradicts,
          ...semantic.categories,
          ...[
            ...semantic.support_hits,
            ...semantic.causal_hits,
            ...semantic.contradiction_hits,
            ...semantic.category_hits,
          ].map((hit) => hit.node),
        ];
  const targets: AnnotationTarget[] = [
    ...(input.episodes ?? []).map(({ episode }) => ({
      record_type: "episode" as const,
      record_id: episode.id,
    })),
    ...nodes.map((node) => ({
      record_type: "semantic_node" as const,
      record_id: node.id,
    })),
    ...(input.evidence ?? []).flatMap((item) => {
      const target = evidenceItemAnnotationTarget(item);
      return target === undefined ? [] : [target];
    }),
    ...additionalTargets,
  ];
  return readRecordEvidenceContexts(reader, targets, {
    ...options,
    readLoadedText: (target) => {
      if (target.record_type === "episode") {
        return input.episodes?.find(({ episode }) => episode.id === target.record_id)?.episode
          .narrative;
      }
      return target.record_type === "semantic_node"
        ? nodes.find((node) => node.id === target.record_id)?.description
        : undefined;
    },
  });
}

/** Live-cognition read projection. Stored records and retrieval ranking remain untouched. */
export async function withRetrievedRecordEvidence<T extends RetrievedRecordEvidence>(
  reader: RecordEvidenceReader | undefined,
  input: T,
): Promise<T> {
  const contexts = await readRetrievedRecordEvidence(reader, input);
  if (contexts.size === 0) return input;
  const decorate = <R>(record: R, target: AnnotationTarget | undefined): R => {
    const context = recordEvidenceContext(contexts, target);
    return context === undefined ? record : { ...record, evidence_context: context };
  };
  const semantic = input.semantic;
  const node = <N extends { id: string }>(value: N): N =>
    decorate(value, { record_type: "semantic_node", record_id: value.id });
  return {
    ...input,
    ...(input.episodes === undefined
      ? {}
      : {
          episodes: input.episodes.map((result) => ({
            ...result,
            episode: decorate(result.episode, {
              record_type: "episode",
              record_id: result.episode.id,
            }),
          })),
        }),
    ...(input.evidence === undefined
      ? {}
      : {
          evidence: input.evidence.map((item) =>
            decorate(item, evidenceItemAnnotationTarget(item)),
          ),
        }),
    ...(semantic == null
      ? {}
      : {
          semantic: {
            ...semantic,
            matched_nodes: semantic.matched_nodes.map(node),
            supports: semantic.supports.map(node),
            contradicts: semantic.contradicts.map(node),
            categories: semantic.categories.map(node),
            support_hits: semantic.support_hits.map((hit) => ({ ...hit, node: node(hit.node) })),
            causal_hits: semantic.causal_hits.map((hit) => ({ ...hit, node: node(hit.node) })),
            contradiction_hits: semantic.contradiction_hits.map((hit) => ({
              ...hit,
              node: node(hit.node),
            })),
            category_hits: semantic.category_hits.map((hit) => ({ ...hit, node: node(hit.node) })),
          },
        }),
  };
}
