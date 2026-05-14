import type { BuilderSectionContext } from "../builder-context.js";
import {
  semanticNodeStateMetadata,
  semanticTaint,
} from "../entry-metadata.js";
import {
  SEMANTIC_TRUST_RANK,
  addEntry,
  cappedTrustRank,
} from "../section-buckets.js";
import {
  persistenceClassFromProvenance,
  scopeFromEpisodeIds,
} from "../scope-resolver.js";

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
        text: node.description,
        value: node.label,
        state: node.under_review === undefined ? node.kind : `under_review:${node.kind}`,
        state_metadata: semanticNodeStateMetadata(node),
        taint: semanticTaint({ underReview: node.under_review }),
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
        text: hit.node.description,
        value: hit.node.label,
        state:
          hit.node.under_review === undefined ? hit.node.kind : `under_review:${hit.node.kind}`,
        state_metadata: semanticNodeStateMetadata(hit.node),
        taint: semanticTaint({ underReview: hit.node.under_review }),
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
          state: edge.valid_to === null ? "valid" : "closed",
          state_metadata: {
            edge_id: edge.id,
            evidence_episode_ids: [...edge.evidence_episode_ids],
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
