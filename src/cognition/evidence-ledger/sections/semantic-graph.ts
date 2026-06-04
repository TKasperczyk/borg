import type { BuilderSectionContext } from "../builder-context.js";
import type { MemoryDisclosureLabel } from "../../../retrieval/index.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
  semanticNodeStateMetadata,
  semanticTaint,
} from "../entry-metadata.js";
import { SEMANTIC_TRUST_RANK, addEntry, cappedTrustRank } from "../section-buckets.js";
import { persistenceClassFromProvenance, scopeFromEpisodeIds } from "../scope-resolver.js";

function semanticNodeStatusAnnotation(node: {
  status: string;
  superseded_at?: number | null;
}): string {
  if (node.status === "active") {
    return "";
  }

  const supersededAt = node.superseded_at == null ? "" : `, t=${Math.trunc(node.superseded_at)}`;

  return `[status=${node.status}${supersededAt}] `;
}

function semanticNodeState(node: { kind: string; status: string; under_review?: unknown }): string {
  if (node.status !== "active") {
    return `${node.status}:${node.kind}`;
  }

  return node.under_review === undefined ? node.kind : `under_review:${node.kind}`;
}

function semanticDisclosureState(label: MemoryDisclosureLabel | undefined): string {
  const state = appendMemoryDisclosureState({
    state: "",
    disclosureLabel: label,
    renderContext: "semantic_source",
  });

  return state === undefined || state.length === 0 ? "" : ` ${state}`;
}

function semanticDisclosureMetadata(
  label: MemoryDisclosureLabel | undefined,
): Record<string, unknown> {
  if (label === undefined) {
    return {};
  }

  return (
    appendMemoryDisclosureStateMetadata({
      stateMetadata: undefined,
      disclosureLabel: label,
      renderContext: "semantic_source",
    }) ?? {}
  );
}

export function addSemanticGraphSection(context: BuilderSectionContext): void {
  const semantic = context.input.retrievedSemantic;

  if (semantic === null || semantic === undefined) {
    return;
  }

  for (const node of semantic.matched_nodes) {
    addEntry(
      context.buckets,
      "semantic_graph",
      cappedTrustRank({
        id: `semantic_node:${node.id}`,
        source_type: "semantic_node",
        session_scope: scopeFromEpisodeIds(node.source_episode_ids, context.resolver),
        actor: "memory",
        trust_rank: SEMANTIC_TRUST_RANK,
        text: `${semanticNodeStatusAnnotation(node)}${node.description}`,
        value: node.label,
        state: `${semanticNodeState(node)}${semanticDisclosureState(node.disclosureLabel)}`,
        state_metadata: {
          ...(semanticNodeStateMetadata(node) ?? {}),
          ...semanticDisclosureMetadata(node.disclosureLabel),
        },
        taint: semanticTaint({ underReview: node.under_review, status: node.status }),
        ...persistenceClassFromProvenance(
          { episodeIds: node.source_episode_ids },
          context.resolver,
        ),
      }),
    );
  }

  for (const hit of [
    ...semantic.support_hits,
    ...semantic.causal_hits,
    ...semantic.contradiction_hits,
    ...semantic.category_hits,
  ]) {
    addEntry(
      context.buckets,
      "semantic_graph",
      cappedTrustRank({
        id: `semantic_node:${hit.node.id}`,
        source_type: "semantic_node",
        session_scope: scopeFromEpisodeIds(hit.node.source_episode_ids, context.resolver),
        actor: "memory",
        trust_rank: SEMANTIC_TRUST_RANK,
        text: `${semanticNodeStatusAnnotation(hit.node)}${hit.node.description}`,
        value: hit.node.label,
        state: `${semanticNodeState(hit.node)}${semanticDisclosureState(hit.node.disclosureLabel)}`,
        state_metadata: {
          ...(semanticNodeStateMetadata(hit.node) ?? {}),
          ...semanticDisclosureMetadata(hit.node.disclosureLabel),
        },
        taint: semanticTaint({ underReview: hit.node.under_review, status: hit.node.status }),
        ...persistenceClassFromProvenance(
          { episodeIds: hit.node.source_episode_ids },
          context.resolver,
        ),
      }),
    );

    for (const edge of hit.edgePath) {
      addEntry(
        context.buckets,
        "semantic_graph",
        cappedTrustRank({
          id: `semantic_edge:${edge.id}`,
          source_type: "semantic_edge",
          session_scope: scopeFromEpisodeIds(edge.evidence_episode_ids, context.resolver),
          actor: "memory",
          trust_rank: SEMANTIC_TRUST_RANK,
          text: `${edge.from_node_id} ${edge.relation} ${edge.to_node_id}`,
          value: edge.relation,
          state: `${edge.valid_to === null ? "valid" : "closed"}${semanticDisclosureState(edge.disclosureLabel)}`,
          state_metadata: {
            edge_id: edge.id,
            evidence_episode_ids: [...edge.evidence_episode_ids],
            ...semanticDisclosureMetadata(edge.disclosureLabel),
          },
          taint: semanticTaint({
            validTo: edge.valid_to,
            invalidatedAt: edge.invalidated_at,
          }),
          ...persistenceClassFromProvenance(
            { episodeIds: edge.evidence_episode_ids },
            context.resolver,
          ),
        }),
      );
    }
  }
}
