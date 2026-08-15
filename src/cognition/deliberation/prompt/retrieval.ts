// Summarizes episodic and semantic retrieval results for deliberation prompts.
import type { SemanticNode } from "../../../memory/semantic/index.js";
import { openQuestionMemoryDisclosureLabel } from "../../../memory/common/disclosure-serializers.js";
import {
  memoryDisclosureLabelFromEpisodeAccess,
  renderMemoryDisclosureLabelForModel,
  unknownMemoryDisclosureLabel,
  type EvidenceItem,
  type MemoryDisclosureLabel,
  type RetrievalConfidence,
  type RetrievedContradictionRouting,
  type RetrievedEpisode,
  type RetrievedContext,
  type RetrievedSemantic,
  type RetrievedSemanticHit,
  type RetrievedSemanticNode,
} from "../../../retrieval/index.js";
import { estimatePromptTokens } from "../../../util/token-estimate.js";
import type { EntityId } from "../../../util/ids.js";
import { escapeXmlText } from "../../../util/prompt-tags.js";
import { renderEvidenceItemDisclosureLabel } from "../../evidence-item-disclosure.js";
import { DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET } from "../constants.js";
import type { ContradictionRoutingTier } from "../types.js";

const LOW_RETRIEVAL_CONFIDENCE_THRESHOLD = 0.45;

export type RetrievedEvidenceSummaryInput = {
  evidence?: readonly EvidenceItem[];
  episodes?: readonly RetrievedEpisode[];
  semantic?: RetrievedSemantic | null | undefined;
  openQuestions?: readonly {
    id: string;
    question: string;
    urgency: number;
    audience_entity_id?: EntityId | null;
  }[];
};

export function summarizeRetrievalConfidence(
  confidence: RetrievalConfidence | null | undefined,
): string | null {
  if (confidence === null || confidence === undefined) {
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

  const lines = [
    "Retrieval confidence (internal, for calibrating certainty in my response):",
    fragments.join(" "),
  ];

  // Policy text lives in EPISTEMIC_POSTURE_SECTION at the system-prompt
  // level (not here), because policy in the untrusted-data block is
  // explicitly told not to be treated as instruction. Here we just
  // surface the empty-state evidence so the LLM sees retrieval ran.
  if (confidence.sampleSize === 0) {
    lines.push("No relevant memory was retrieved for this turn.");
  } else if (confidence.overall < LOW_RETRIEVAL_CONFIDENCE_THRESHOLD) {
    lines.push("Retrieval confidence is low; specific claims here are weakly supported.");
  }

  // Internal hint: the being should speak more cautiously when overall is low.
  // Not user-facing -- the LLM phrases uncertainty naturally rather than
  // emitting the percentage. This is the signal, not the phrasing.
  return lines.join("\n");
}

export function summarizeContradictionSignal(
  routing: RetrievedContradictionRouting | null | undefined,
  tier: ContradictionRoutingTier | null | undefined,
  confidence: RetrievalConfidence | null | undefined,
  path: "system_1" | "system_2" | null | undefined,
): string | null {
  const contradictions = routing?.contradictions ?? [];

  if (
    path !== "system_1" ||
    contradictions.length === 0 ||
    tier === "none" ||
    tier === "s2_forced"
  ) {
    return null;
  }

  const localEdgeHandles = contradictions
    .filter((contradiction) => contradiction.edgeId !== undefined)
    .slice(0, 5)
    .map((_, index) => `contradiction_${index + 1}_edge`);
  const localHandles =
    localEdgeHandles.length === 0
      ? contradictions.slice(0, 5).map((_, index) => `contradiction_${index + 1}`)
      : localEdgeHandles;
  const omittedCount = Math.max(0, contradictions.length - localHandles.length);
  const handleSummary =
    omittedCount === 0
      ? localHandles.join(", ")
      : `${localHandles.join(", ")}, +${omittedCount} more`;
  const noun = contradictions.length === 1 ? "contradiction" : "contradictions";
  const confidencePenalty =
    confidence?.contradictionPresent === true
      ? "Confidence penalty applied."
      : "Confidence penalty not applied.";

  return `${contradictions.length} retrieved ${noun} present (edges: ${handleSummary}). ${confidencePenalty} Not routing to S2.`;
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
    return "No episodes retrieved for this turn.";
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
      `  disclosure: ${renderMemoryDisclosureLabelForModel(result.disclosureLabel ?? memoryDisclosureLabelFromEpisodeAccess(result.episode))}`,
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

export function summarizeRetrievedEvidence(
  label: string,
  input: RetrievedEvidenceSummaryInput,
  maxTokens = DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET,
): string | null {
  const evidence = input.evidence ?? [];

  if (evidence.length > 0) {
    return summarizeEvidenceItems(label, evidence, maxTokens);
  }

  const fallbackSections = [
    summarizeRetrievedEpisodes(label, input.episodes ?? [], maxTokens),
    summarizeSemanticContext(input.semantic, Math.max(500, Math.floor(maxTokens / 2))),
    summarizeOpenQuestionEvidence(input.openQuestions ?? []),
  ].filter((section): section is string => section !== null && section.length > 0);

  if (fallbackSections.length === 0) {
    return "No retrieved evidence for this turn.";
  }

  return fallbackSections.join("\n\n");
}

type VerificationRetrievalCandidate = {
  handle: string;
  sourceClass: "evidence" | "episode" | "semantic_node" | "semantic_edge" | "open_question";
  disclosure: string;
  structuralFields: Record<string, string | number | boolean | null>;
  payload: unknown;
};

function verificationXmlAttribute(value: string): string {
  return escapeXmlText(value)
    .replaceAll('"', "&quot;")
    .replaceAll("\r", "&#13;")
    .replaceAll("\n", "&#10;")
    .replaceAll("\t", "&#9;");
}

function verificationDisclosure(label: MemoryDisclosureLabel | undefined): string {
  const exact = label ?? unknownMemoryDisclosureLabel();
  const list = (values: readonly string[]) => (values.length === 0 ? "none" : values.join(","));
  return [
    `disclosure_class=${exact.disclosureClass}`,
    `origin_audience=${list(exact.originAudienceEntityIds)}`,
    `private-to=${list(exact.privateToEntityIds)}`,
    `public-to=${list(exact.publicToEntityIds)}`,
  ].join(" ");
}

function verificationEvidenceCandidates(
  evidence: readonly EvidenceItem[],
): VerificationRetrievalCandidate[] {
  return evidence.map((item) => ({
    handle: item.id,
    sourceClass: "evidence",
    disclosure: verificationDisclosure(item.disclosureLabel),
    structuralFields: {
      source: item.source,
      recall_intent_id: item.recallIntentId,
      score: item.score,
      provenance_episode_id: item.provenance?.episodeId ?? null,
      provenance_node_id: item.provenance?.nodeId ?? null,
      provenance_edge_id: item.provenance?.edgeId ?? null,
      provenance_commitment_id: item.provenance?.commitmentId ?? null,
      provenance_open_question_id: item.provenance?.openQuestionId ?? null,
      provenance_stream_ids: item.provenance?.streamIds?.join(",") ?? "none",
      partial_source_visibility: item.partial_source_visibility === true,
      source_visibility_fraction: item.source_visibility_fraction ?? null,
    },
    payload: {
      text: item.text,
      matched_terms: item.matchedTerms,
      image_label: item.imageLabel ?? null,
      image_origin_frame: item.imageOriginFrame ?? null,
      image_unavailable_reason: item.imageUnavailableReason ?? null,
    },
  }));
}

function verificationFallbackCandidates(
  input: Pick<RetrievedContext, "episodes" | "semantic" | "open_questions">,
): VerificationRetrievalCandidate[] {
  const episodes: VerificationRetrievalCandidate[] = input.episodes.map((result) => ({
    handle: result.episode.id,
    sourceClass: "episode",
    disclosure: verificationDisclosure(
      result.disclosureLabel ?? memoryDisclosureLabelFromEpisodeAccess(result.episode),
    ),
    structuralFields: {
      score: result.score,
      source_stream_ids: result.episode.source_stream_ids.join(","),
      start_time: result.episode.start_time,
      end_time: result.episode.end_time,
    },
    payload: {
      title: result.episode.title,
      narrative: result.episode.narrative,
      participants: result.episode.participants,
      tags: result.episode.tags,
      citations: result.citationChain.map((entry) => ({ id: entry.id, content: entry.content })),
    },
  }));
  const hits = [
    ...input.semantic.support_hits,
    ...input.semantic.causal_hits,
    ...input.semantic.contradiction_hits,
    ...input.semantic.category_hits,
  ];
  const nodesById = new Map(
    [...input.semantic.matched_nodes, ...hits.map((hit) => hit.node)].map((node) => [
      node.id,
      node,
    ]),
  );
  const edgesById = new Map(hits.flatMap((hit) => hit.edgePath.map((edge) => [edge.id, edge])));
  const nodes: VerificationRetrievalCandidate[] = [...nodesById.values()].map((node) => ({
    handle: node.id,
    sourceClass: "semantic_node",
    disclosure: verificationDisclosure(node.disclosureLabel),
    structuralFields: {
      kind: node.kind,
      status: node.status,
      confidence: node.confidence,
      source_episode_ids: node.source_episode_ids.join(","),
      partial_source_visibility: node.partial_source_visibility === true,
      source_visibility_fraction: node.source_visibility_fraction ?? null,
    },
    payload: {
      label: node.label,
      description: node.description,
      domain: node.domain,
      aliases: node.aliases,
      observation_metadata: node.observation_metadata,
      under_review_reason: node.under_review?.reason ?? null,
    },
  }));
  const edges: VerificationRetrievalCandidate[] = [...edgesById.values()].map((edge) => ({
    handle: edge.id,
    sourceClass: "semantic_edge",
    disclosure: verificationDisclosure(edge.disclosureLabel),
    structuralFields: {
      from_node_id: edge.from_node_id,
      to_node_id: edge.to_node_id,
      relation: edge.relation,
      confidence: edge.confidence,
      evidence_episode_ids: edge.evidence_episode_ids.join(","),
      valid_from: edge.valid_from,
      valid_to: edge.valid_to,
    },
    payload: { invalidated_reason: edge.invalidated_reason },
  }));
  const openQuestions: VerificationRetrievalCandidate[] = input.open_questions.map((question) => ({
    handle: question.id,
    sourceClass: "open_question",
    disclosure: verificationDisclosure(openQuestionMemoryDisclosureLabel(question)),
    structuralFields: {
      status: question.status,
      urgency: question.urgency,
      source: question.source,
      audience_entity_id: question.audience_entity_id,
      goal_id: question.goal_id,
    },
    payload: {
      question: question.question,
      resolution_note: question.resolution_note,
      abandoned_reason: question.abandoned_reason,
    },
  }));
  return [...episodes, ...nodes, ...edges, ...openQuestions];
}

function renderVerificationRetrievalCandidate(
  candidate: VerificationRetrievalCandidate,
  includePayload: boolean,
): string {
  const payloadJson = JSON.stringify(candidate.payload) ?? "null";
  const structural = Object.entries(candidate.structuralFields)
    .map(([key, value]) => `${key}="${verificationXmlAttribute(String(value ?? "none"))}"`)
    .join(" ");
  return [
    `<verification_source handle="${verificationXmlAttribute(candidate.handle)}"`,
    `source_class="${candidate.sourceClass}"`,
    `disclosure="${verificationXmlAttribute(candidate.disclosure)}"`,
    structural,
    `payload_status="${includePayload ? "exact" : "check_not_completed_budget"}"`,
    `payload_included_chars="${includePayload ? payloadJson.length : 0}"`,
    `payload_total_chars="${payloadJson.length}"`,
    `payload_json="${includePayload ? verificationXmlAttribute(payloadJson) : ""}" />`,
  ].join(" ");
}

function renderVerificationRetrievalRows(
  candidates: readonly VerificationRetrievalCandidate[],
  included: ReadonlySet<number>,
  maxTokens: number,
): string {
  const rows = candidates.map((candidate, index) =>
    renderVerificationRetrievalCandidate(candidate, included.has(index)),
  );
  const incompleteCount = candidates.length - included.size;
  return [
    `<plan_requested_verification_retrieval complete_membership="true" rows_total="${rows.length}" target_tokens="${maxTokens}" check_not_completed_count="${incompleteCount}">`,
    "  <interpretation>This retrieval was requested by the advisory plan. Every returned source handle is present. A payload_status=exact row carries its complete payload with no excerpt; payload_status=check_not_completed_budget carries the handle and structural fields but zero payload, so the requested check is explicitly incomplete rather than silently truncated.</interpretation>",
    ...rows.map((row) => `  ${row}`),
    "  <omitted_count>0</omitted_count>",
    `  <check_not_completed_count>${incompleteCount}</check_not_completed_count>`,
    "</plan_requested_verification_retrieval>",
  ].join("\n");
}

/**
 * Compact-terminal rendering for secondary retrieval driven structurally by
 * an S2 plan's non-empty verification_steps. Payloads are all-or-nothing:
 * exact when they fit, otherwise an explicit incomplete-check row.
 */
export function renderPlanRequestedVerificationRetrieval(
  input: Pick<RetrievedContext, "evidence" | "episodes" | "semantic" | "open_questions">,
  maxTokens = DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET,
): string {
  const candidates = [
    ...verificationEvidenceCandidates(input.evidence),
    ...verificationFallbackCandidates(input),
  ];
  const included = new Set<number>();

  for (let index = 0; index < candidates.length; index += 1) {
    included.add(index);
    if (
      estimatePromptTokens(renderVerificationRetrievalRows(candidates, included, maxTokens)) >
      maxTokens
    ) {
      included.delete(index);
    }
  }

  return renderVerificationRetrievalRows(candidates, included, maxTokens);
}

export function renderPlanRequestedVerificationNotCompleted(): string {
  return [
    '<plan_requested_verification_retrieval complete_membership="false" rows_total="1" check_not_completed_count="1">',
    "  <interpretation>The advisory plan requested a verification pass, but secondary retrieval was unavailable. The request handle remains visible with an explicit incomplete status and zero payload characters.</interpretation>",
    '  <verification_source handle="plan:verification_steps" source_class="verification_request" payload_status="check_not_completed_retrieval_unavailable" payload_included_chars="0" payload_total_chars="0" payload_json="" />',
    "  <omitted_count>0</omitted_count>",
    "  <check_not_completed_count>1</check_not_completed_count>",
    "</plan_requested_verification_retrieval>",
  ].join("\n");
}

function summarizeEvidenceItems(
  label: string,
  evidence: readonly EvidenceItem[],
  maxTokens: number,
): string {
  const lines = [`${label}:`];
  let usedTokens = estimatePromptTokens(lines[0] ?? label);

  for (const item of evidence) {
    const block = summarizeEvidenceItem(item);
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

function summarizeEvidenceItem(item: EvidenceItem): string {
  const text = truncatePromptText(item.text, 360);
  const provenance = summarizeEvidenceProvenance(item);
  const terms = item.matchedTerms.length === 0 ? "" : ` terms=${item.matchedTerms.join(", ")}`;
  const sourceVisibility = summarizeEvidenceSourceVisibility(item);
  const disclosure =
    item.disclosureLabel === undefined ? "" : ` ${renderEvidenceItemDisclosureLabel(item)}`;

  return [
    `- ${item.source} [score=${item.score.toFixed(2)} intent=${item.recallIntentId}${terms}${sourceVisibility}${disclosure}]${provenance}`,
    `  ${text}`,
  ].join("\n");
}

function summarizeEvidenceSourceVisibility(item: EvidenceItem): string {
  const parts = [
    item.source_episode_ids === undefined || item.source_episode_ids.length === 0
      ? null
      : `sources=${summarizeEpisodeIds(item.source_episode_ids)}`,
    item.partial_source_visibility === true ? "partial_sources=true" : null,
    item.source_visibility_fraction === undefined
      ? null
      : `visible_fraction=${item.source_visibility_fraction.toFixed(2)}`,
  ].filter((part): part is string => part !== null);

  return parts.length === 0 ? "" : ` ${parts.join(" ")}`;
}

function summarizeEvidenceProvenance(item: EvidenceItem): string {
  const provenance = item.provenance;

  if (provenance === undefined) {
    return "";
  }

  const parts = [
    provenance.episodeId === undefined ? null : `episode=${provenance.episodeId}`,
    provenance.nodeId === undefined ? null : `node=${provenance.nodeId}`,
    provenance.edgeId === undefined ? null : `edge=${provenance.edgeId}`,
    provenance.commitmentId === undefined ? null : `commitment=${provenance.commitmentId}`,
    provenance.openQuestionId === undefined ? null : `open_question=${provenance.openQuestionId}`,
    provenance.streamIds === undefined || provenance.streamIds.length === 0
      ? null
      : `streams=${provenance.streamIds.slice(0, 3).join(", ")}`,
  ].filter((part): part is string => part !== null);

  return parts.length === 0 ? "" : ` (${parts.join("; ")})`;
}

function summarizeOpenQuestionEvidence(
  openQuestions: readonly {
    id: string;
    question: string;
    urgency: number;
    audience_entity_id?: EntityId | null;
  }[],
): string | null {
  if (openQuestions.length === 0) {
    return null;
  }

  return [
    "Open questions:",
    ...openQuestions
      .slice(0, 4)
      .map(
        (question) =>
          `- ${question.question} [open_question=${question.id} urgency=${question.urgency.toFixed(2)} ${renderMemoryDisclosureLabelForModel(openQuestionMemoryDisclosureLabel({ audience_entity_id: question.audience_entity_id ?? null }))}]`,
      ),
  ].join("\n");
}

function truncatePromptText(text: string, maxChars: number): string {
  const normalized = text.replace(/\s+/g, " ").trim();
  return normalized.length > maxChars
    ? `${normalized.slice(0, maxChars - 3).trimEnd()}...`
    : normalized;
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

function summarizeValidityTag(edge: RetrievedSemanticHit["edgePath"][number]): string {
  if (edge.valid_to === null) {
    return "";
  }

  const closedAt = edge.invalidated_at ?? edge.valid_to;

  return ` [valid ${formatIsoDate(edge.valid_from)}..${formatIsoDate(edge.valid_to)}, closed ${formatIsoDate(closedAt)}]`;
}

function semanticHitHasClosedEdge(hit: RetrievedSemanticHit, asOf: number): boolean {
  return hit.edgePath.some((edge) => edge.valid_to !== null && edge.valid_to <= asOf);
}

function summarizeUnderReviewPrefix(node: {
  under_review?: RetrievedSemanticNode["under_review"];
}): string {
  if (node.under_review === undefined) {
    return "";
  }

  const disclosure =
    node.under_review.disclosureLabel.disclosureClass === "public"
      ? ""
      : ` ${renderMemoryDisclosureLabelForModel(node.under_review.disclosureLabel, { context: "semantic_source" })}`;

  return `[under re-evaluation: ${node.under_review.reason_code}]${disclosure} `;
}

function summarizeSemanticStatusPrefix(
  node: Pick<SemanticNode, "status" | "superseded_at">,
): string {
  if (node.status === "active") {
    return "";
  }

  const supersededAt = node.superseded_at === null ? "" : `, t=${Math.trunc(node.superseded_at)}`;

  return `[status=${node.status}${supersededAt}] `;
}

function summarizeSemanticNodePrefixes(
  node: Pick<SemanticNode, "status" | "superseded_at"> & {
    under_review?: RetrievedSemanticNode["under_review"];
  },
): string {
  return `${summarizeSemanticStatusPrefix(node)}${summarizeUnderReviewPrefix(node)}`;
}

function summarizePartialSourceTag(node: {
  partial_source_visibility?: RetrievedSemanticNode["partial_source_visibility"];
}): string {
  return node.partial_source_visibility === true ? ", partial sources" : "";
}

function summarizePartialEvidenceTag(edge: {
  partial_source_visibility?: RetrievedSemanticHit["edgePath"][number]["partial_source_visibility"];
  source_visibility_fraction?: RetrievedSemanticHit["edgePath"][number]["source_visibility_fraction"];
}): string {
  if (edge.partial_source_visibility !== true) {
    return "";
  }

  const fraction =
    edge.source_visibility_fraction === undefined
      ? ""
      : ` visible_fraction=${edge.source_visibility_fraction.toFixed(2)}`;
  return ` partial_sources=true${fraction}`;
}

function summarizeSemanticDisclosureTag(input: {
  disclosureLabel?: RetrievedSemanticNode["disclosureLabel"];
}): string {
  return input.disclosureLabel === undefined
    ? ""
    : `, ${renderMemoryDisclosureLabelForModel(input.disclosureLabel, { context: "semantic_source" })}`;
}

function summarizeSemanticNode(
  node: SemanticNode & {
    partial_source_visibility?: RetrievedSemanticNode["partial_source_visibility"];
    under_review?: RetrievedSemanticNode["under_review"];
    disclosureLabel?: RetrievedSemanticNode["disclosureLabel"];
  },
): string {
  return `${summarizeSemanticNodePrefixes(node)}${node.label} - ${summarizeSemanticNodeDescription(node)} (conf ${node.confidence.toFixed(2)}${summarizePartialSourceTag(node)}${summarizeSemanticDisclosureTag(node)})`;
}

function summarizeSemanticNodeWithSources(
  node: RetrievedSemantic["matched_nodes"][number],
): string {
  const label = [
    `${summarizeSemanticNodePrefixes(node)}${node.label}`,
    node.historical === true ? " [historical]" : "",
  ].join("");

  return `${label} - ${summarizeSemanticNodeDescription(node)} (conf ${node.confidence.toFixed(2)}, sources ${summarizeEpisodeIds(node.source_episode_ids)}${summarizePartialSourceTag(node)}${summarizeSemanticDisclosureTag(node)})`;
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
    const evidenceVisibility = summarizePartialEvidenceTag(edge);
    const evidenceDisclosure = summarizeSemanticDisclosureTag(edge);
    const validityTag = options.tagClosedEdges ? summarizeValidityTag(edge) : "";
    const relation =
      edge.from_node_id === currentNodeId
        ? `-[${edge.relation} conf=${edge.confidence.toFixed(2)} evidence=${evidence}${evidenceVisibility}${evidenceDisclosure}]${validityTag}->`
        : `<-[${edge.relation} conf=${edge.confidence.toFixed(2)} evidence=${evidence}${evidenceVisibility}${evidenceDisclosure}]${validityTag}-`;

    pathParts.push(relation);

    if (index === hit.edgePath.length - 1) {
      pathParts.push(hit.node.label);
      continue;
    }

    currentNodeId = edge.from_node_id === currentNodeId ? edge.to_node_id : edge.from_node_id;
    pathParts.push("...");
  }

  return `${summarizeSemanticNodePrefixes(hit.node)}${hit.node.label} - ${summarizeSemanticNodeDescription(hit.node)} (node conf ${hit.node.confidence.toFixed(2)}, sources ${summarizeEpisodeIds(hit.node.source_episode_ids)}${summarizePartialSourceTag(hit.node)}${summarizeSemanticDisclosureTag(hit.node)}; path ${pathParts.join(" ")})`;
}

function summarizeSemanticBucket(
  label: string,
  nodes: readonly (SemanticNode & {
    partial_source_visibility?: RetrievedSemanticNode["partial_source_visibility"];
    under_review?: RetrievedSemanticNode["under_review"];
    disclosureLabel?: RetrievedSemanticNode["disclosureLabel"];
  })[],
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
