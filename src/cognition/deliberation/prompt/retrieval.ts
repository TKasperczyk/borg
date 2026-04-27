// Summarizes episodic and semantic retrieval results for deliberation prompts.
import type { SemanticNode } from "../../../memory/semantic/index.js";
import type {
  RetrievalConfidence,
  RetrievedEpisode,
  RetrievedSemantic,
  RetrievedSemanticHit,
  RetrievedSemanticNode,
} from "../../../retrieval/index.js";
import { DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET } from "../constants.js";

export function summarizeRetrievalConfidence(
  confidence: RetrievalConfidence | null | undefined,
): string | null {
  if (confidence === null || confidence === undefined || confidence.sampleSize === 0) {
    return null;
  }

  const fragments: string[] = [
    `overall=${confidence.overall.toFixed(2)}`,
    `evidence=${confidence.evidenceStrength.toFixed(2)}`,
    `coverage=${confidence.coverage.toFixed(2)}`,
    `diversity=${confidence.sourceDiversity.toFixed(2)}`,
    `samples=${confidence.sampleSize}`,
  ];

  if (confidence.contradictionPresent) {
    fragments.push("contradictions=present");
  }

  // Internal hint: the being should speak more cautiously when overall is low.
  // Not user-facing -- the LLM phrases uncertainty naturally rather than
  // emitting the percentage. This is the signal, not the phrasing.
  return [
    "Retrieval confidence (internal, for calibrating certainty in your response):",
    fragments.join(" "),
  ].join("\n");
}

function estimatePromptTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

function summarizeCitationChain(result: RetrievedEpisode): string | null {
  if (result.citationChain.length === 0) {
    return null;
  }

  const snippets = result.citationChain.slice(0, 2).map((entry) => {
    const content =
      typeof entry.content === "string" ? entry.content : JSON.stringify(entry.content ?? null);
    const normalized = content.replace(/\s+/g, " ").trim();
    return normalized.length > 140 ? `${normalized.slice(0, 137).trimEnd()}...` : normalized;
  });

  return snippets.length === 0 ? null : `  citations: ${snippets.join(" | ")}`;
}

export function summarizeRetrievedEpisodes(
  label: string,
  retrievedEpisodes: readonly RetrievedEpisode[],
  maxTokens = DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET,
): string | null {
  if (retrievedEpisodes.length === 0) {
    return null;
  }

  const lines = [`${label}:`];
  let usedTokens = estimatePromptTokens(lines[0] ?? label);

  for (const result of retrievedEpisodes) {
    // This is the relevance ranking score. Epistemic retrieval confidence is
    // rendered separately in the retrieval-confidence prompt block.
    const normalizedNarrative = result.episode.narrative.replace(/\s+/g, " ").trim();
    const narrative =
      normalizedNarrative.length > 320
        ? `${normalizedNarrative.slice(0, 317).trimEnd()}...`
        : normalizedNarrative;
    const blockLines = [
      `- ${result.episode.title} [score=${result.score.toFixed(2)} sim=${result.scoreBreakdown.similarity.toFixed(2)} salience=${result.scoreBreakdown.decayedSalience.toFixed(2)}]`,
      `  narrative: ${narrative}`,
      `  participants: ${result.episode.participants.join(", ") || "none"}`,
      `  tags: ${result.episode.tags.join(", ") || "none"}`,
      summarizeCitationChain(result),
    ].filter((line): line is string => line !== null);
    const block = blockLines.join("\n");
    const blockTokens = estimatePromptTokens(block);

    if (usedTokens + blockTokens > maxTokens) {
      lines.push("- ... truncated");
      break;
    }

    lines.push(block);
    usedTokens += blockTokens;
  }

  return lines.join("\n");
}

function summarizeSemanticNodeDescription(node: SemanticNode): string {
  const normalizedDescription = node.description.replace(/\s+/g, " ").trim();
  return normalizedDescription.length > 96
    ? `${normalizedDescription.slice(0, 93).trimEnd()}...`
    : normalizedDescription;
}

function summarizeEpisodeIds(ids: readonly string[], limit = 3): string {
  const displayed = ids.slice(0, limit);
  const suffix = ids.length > limit ? `, +${ids.length - limit} more` : "";
  return `${displayed.join(", ")}${suffix}`;
}

function formatIsoDate(timestamp: number): string {
  return new Date(timestamp).toISOString().slice(0, 10);
}

function summarizeValidityTag(
  edge: RetrievedSemanticHit["edgePath"][number],
): string {
  if (edge.valid_to === null) {
    return "";
  }

  const closedAt = edge.invalidated_at ?? edge.valid_to;

  return ` [valid ${formatIsoDate(edge.valid_from)}..${formatIsoDate(edge.valid_to)}, closed ${formatIsoDate(closedAt)}]`;
}

function semanticHitHasClosedEdge(
  hit: RetrievedSemanticHit,
  asOf: number,
): boolean {
  return hit.edgePath.some((edge) => edge.valid_to !== null && edge.valid_to <= asOf);
}

function summarizeUnderReviewPrefix(node: {
  under_review?: RetrievedSemanticNode["under_review"];
}): string {
  if (node.under_review === undefined) {
    return "";
  }

  return `[under re-evaluation: ${node.under_review.reason_code}] `;
}

function summarizeSemanticNode(
  node: SemanticNode & { under_review?: RetrievedSemanticNode["under_review"] },
): string {
  return `${summarizeUnderReviewPrefix(node)}${node.label} - ${summarizeSemanticNodeDescription(node)} (conf ${node.confidence.toFixed(2)})`;
}

function summarizeSemanticNodeWithSources(
  node: RetrievedSemantic["matched_nodes"][number],
): string {
  const label = [
    `${summarizeUnderReviewPrefix(node)}${node.label}`,
    node.historical === true ? " [historical]" : "",
  ].join("");

  return `${label} - ${summarizeSemanticNodeDescription(node)} (conf ${node.confidence.toFixed(2)}, sources ${summarizeEpisodeIds(node.source_episode_ids)})`;
}

function summarizeSemanticHit(
  hit: RetrievedSemanticHit,
  rootNodesById: ReadonlyMap<string, SemanticNode>,
  options: { tagClosedEdges: boolean },
): string {
  const root = rootNodesById.get(hit.root_node_id);
  const rootLabel = root?.label ?? hit.root_node_id;
  let currentNodeId = hit.root_node_id;
  const pathParts: string[] = [rootLabel];

  for (const [index, edge] of hit.edgePath.entries()) {
    const evidence = summarizeEpisodeIds(edge.evidence_episode_ids);
    const validityTag = options.tagClosedEdges ? summarizeValidityTag(edge) : "";
    const relation =
      edge.from_node_id === currentNodeId
        ? `-[${edge.relation} conf=${edge.confidence.toFixed(2)} evidence=${evidence}]${validityTag}->`
        : `<-[${edge.relation} conf=${edge.confidence.toFixed(2)} evidence=${evidence}]${validityTag}-`;

    pathParts.push(relation);

    if (index === hit.edgePath.length - 1) {
      pathParts.push(hit.node.label);
      continue;
    }

    currentNodeId = edge.from_node_id === currentNodeId ? edge.to_node_id : edge.from_node_id;
    pathParts.push("...");
  }

  return `${summarizeUnderReviewPrefix(hit.node)}${hit.node.label} - ${summarizeSemanticNodeDescription(hit.node)} (node conf ${hit.node.confidence.toFixed(2)}, sources ${summarizeEpisodeIds(hit.node.source_episode_ids)}; path ${pathParts.join(" ")})`;
}

function summarizeSemanticBucket(
  label: string,
  nodes: readonly (SemanticNode & { under_review?: RetrievedSemanticNode["under_review"] })[],
  limit = 3,
): string | null {
  if (nodes.length === 0) {
    return null;
  }

  return `${label}: ${nodes
    .slice(0, limit)
    .map((node) => summarizeSemanticNode(node))
    .join("; ")}`;
}

function summarizeSemanticHitBucket(
  label: string,
  hits: readonly RetrievedSemanticHit[],
  rootNodesById: ReadonlyMap<string, SemanticNode>,
  options: { tagClosedEdges: boolean },
  limit = 3,
): string[] {
  if (hits.length === 0) {
    return [];
  }

  return [
    `${label}:`,
    ...hits.slice(0, limit).map((hit) => `- ${summarizeSemanticHit(hit, rootNodesById, options)}`),
  ];
}

export function summarizeSemanticContext(
  retrievedSemantic: RetrievedSemantic | null | undefined,
  maxContextTokens: number,
  nowMs = Date.now(),
): string | null {
  if (retrievedSemantic === null || retrievedSemantic === undefined) {
    return null;
  }

  const {
    supports,
    contradicts,
    categories,
    matched_nodes: matchedNodes,
    support_hits: supportHits,
    causal_hits: causalHits,
    contradiction_hits: contradictionHits,
    category_hits: categoryHits,
  } = retrievedSemantic;

  if (
    matchedNodes.length === 0 &&
    supportHits.length === 0 &&
    causalHits.length === 0 &&
    contradictionHits.length === 0 &&
    categoryHits.length === 0 &&
    supports.length === 0 &&
    contradicts.length === 0 &&
    categories.length === 0
  ) {
    return null;
  }

  // Budget: rougher than the episode-level rendering because this is a single
  // flat block rather than one-per-episode. Still caps both node count per
  // bucket (at the bucket helper) and overall char budget.
  const bucketLimit = maxContextTokens <= 2_000 ? 3 : maxContextTokens <= 8_000 ? 5 : 8;
  const maxChars = Math.max(480, Math.min(maxContextTokens * 6, 6_000));
  const rootNodesById = new Map(matchedNodes.map((node) => [node.id, node] as const));
  const historicalMode = retrievedSemantic.as_of !== undefined && retrievedSemantic.as_of !== null;
  const currentAsOf = nowMs;
  const visibleSupportHits = historicalMode
    ? supportHits
    : supportHits.filter((hit) => !semanticHitHasClosedEdge(hit, currentAsOf));
  const visibleCausalHits = historicalMode
    ? causalHits
    : causalHits.filter((hit) => !semanticHitHasClosedEdge(hit, currentAsOf));
  const visibleContradictionHits = historicalMode
    ? contradictionHits
    : contradictionHits.filter((hit) => !semanticHitHasClosedEdge(hit, currentAsOf));
  const visibleCategoryHits = historicalMode
    ? categoryHits
    : categoryHits.filter((hit) => !semanticHitHasClosedEdge(hit, currentAsOf));
  const initialLine = "Related semantic context:";
  const sections: string[] = [initialLine];
  let totalChars = initialLine.length;

  const directMatchLines =
    matchedNodes.length === 0
      ? []
      : [
          "Directly matched:",
          ...matchedNodes
            .slice(0, bucketLimit)
            .map((node) => `- ${summarizeSemanticNodeWithSources(node)}`),
        ];

  const bucketLines = [
    ...directMatchLines,
    ...(supportHits.length > 0
      ? summarizeSemanticHitBucket(
          "supports",
          visibleSupportHits,
          rootNodesById,
          {
            tagClosedEdges: historicalMode,
          },
          bucketLimit,
        )
      : [summarizeSemanticBucket("supports", supports, bucketLimit)].filter(
          (value): value is string => value !== null,
        )),
    ...summarizeSemanticHitBucket(
      "causal",
      visibleCausalHits,
      rootNodesById,
      {
        tagClosedEdges: historicalMode,
      },
      bucketLimit,
    ),
    ...(contradictionHits.length > 0
      ? summarizeSemanticHitBucket(
          "contradicts",
          visibleContradictionHits,
          rootNodesById,
          {
            tagClosedEdges: historicalMode,
          },
          bucketLimit,
        )
      : [summarizeSemanticBucket("contradicts", contradicts, bucketLimit)].filter(
          (value): value is string => value !== null,
        )),
    ...(categoryHits.length > 0
      ? summarizeSemanticHitBucket(
          "categories",
          visibleCategoryHits,
          rootNodesById,
          {
            tagClosedEdges: historicalMode,
          },
          bucketLimit,
        )
      : [summarizeSemanticBucket("categories", categories, bucketLimit)].filter(
          (value): value is string => value !== null,
        )),
  ];

  for (const line of bucketLines) {
    if (totalChars + line.length > maxChars) {
      sections.push("... truncated");
      break;
    }

    sections.push(line);
    totalChars += line.length;
  }

  return sections.join("\n");
}
