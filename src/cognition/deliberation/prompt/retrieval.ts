// Summarizes episodic and semantic retrieval results for deliberation prompts.
import type { SemanticNode } from "../../../memory/semantic/index.js";
import { openQuestionMemoryDisclosureLabel } from "../../../memory/common/disclosure-serializers.js";
import {
  SEMANTIC_EVIDENCE_STRENGTH_SCALE,
  memoryDisclosureLabelFromEpisodeAccess,
  renderMemoryDisclosureLabelForModel,
  unknownMemoryDisclosureLabel,
  type EvidenceItem,
  type MemoryDisclosureClass,
  type MemoryDisclosureLabel,
  type RetrievalConfidence,
  type RetrievedContradictionRouting,
  type RetrievedEpisode,
  type RetrievedContext,
  type RetrievedSemantic,
  type RetrievedSemanticHit,
  type RetrievedSemanticNode,
} from "../../../retrieval/index.js";
import {
  estimatePromptTokens,
  estimatePromptTokensFromLength,
} from "../../../util/token-estimate.js";
import type { EntityId } from "../../../util/ids.js";
import { escapeXmlText } from "../../../util/prompt-tags.js";
import { renderEvidenceItemDisclosureLabel } from "../../evidence-item-disclosure.js";
import {
  DEFAULT_PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_TOKEN_BUDGET,
  DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET,
} from "../constants.js";
import type { ContradictionRoutingTier } from "../types.js";

const LOW_RETRIEVAL_CONFIDENCE_THRESHOLD = 0.45;
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER =
  "membership_not_enumerated_budget";
export const PLAN_REQUESTED_VERIFICATION_ROWS_TOTAL_AS_OF_ATTRIBUTE = "rows_total_as_of";
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER_ATTRIBUTE = "membership_order";
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER =
  "critical_commitments_first_then_retrieval_pipeline_order";
export const PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER_ATTRIBUTE = "payload_order";
export const PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER = "enumerated_membership_order";
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER =
  "membership_not_enumerated_by_disclosure_class";
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_REMAINDER_TOTAL_ATTRIBUTE = "total";
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ERROR_ATTRIBUTE = "membership_error";
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_OVERFLOW_ERROR =
  "carve_out_exceeds_budget";
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_OVERFLOW_MARKER =
  "membership_carve_out_overflow_error";
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_ROWS_ATTRIBUTE =
  "membership_carve_out_rows_total";
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_REQUIRED_TOKENS_ATTRIBUTE =
  "membership_carve_out_required_tokens";
export const PLAN_REQUESTED_VERIFICATION_RETRIEVAL_STATUS_ATTRIBUTE = "retrieval_status";
export const PLAN_REQUESTED_VERIFICATION_RETRIEVAL_UNAVAILABLE_STATUS = "unavailable";
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_STATUS_ATTRIBUTE = "membership_status";
export const PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_NOT_OBSERVED_STATUS = "not_observed";

const VERIFICATION_COMMITMENT_ENFORCEMENT_CLASS_FIELD = "commitment_enforcement_class";
const VERIFICATION_COMMITMENT_CRITICAL_DOMAIN_FIELD = "commitment_critical_domain";
const VERIFICATION_DISCLOSURE_CLASS_FIELD = "disclosure_class";

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

  // `evidenceStrength` is a clamped sum, not a ratio, so it saturates the same
  // way the two ratios below do and has no fraction to give it away. Print the
  // addends, and when their sum exceeded 1 print the sum and say it was
  // clamped -- otherwise a ceiling hit and a measurement are the same two
  // characters. Rendered on every turn, including when nothing was clamped,
  // because an absent marker would be indistinguishable from an unmarked one.
  // The addends are not commensurable and printing them bare implies they are.
  // `ep` is a mean of clamped saliences, so its bound is 1.00 -- the same bound
  // as the field it feeds. `sem` is a fixed scale times a sigmoid that saturates
  // once a few supported matches exist, so its bound is that scale and it sits
  // on it most turns. Print `sem` against that bound: it is what lets a reader
  // see the addend is pinned rather than measured, and derive the threshold
  // (`ep >= 1 - sem`) past which the whole field stops discriminating.
  const evidenceRaw = confidence.evidenceEpisodeStrength + confidence.evidenceSemanticStrength;
  const evidenceComponents =
    `ep=${confidence.evidenceEpisodeStrength.toFixed(2)}` +
    `+sem=${confidence.evidenceSemanticStrength.toFixed(2)}` +
    `/${SEMANTIC_EVIDENCE_STRENGTH_SCALE.toFixed(2)}` +
    (evidenceRaw > 1 ? `,raw=${evidenceRaw.toFixed(2)},clamped` : "");

  const fragments: string[] = [
    `overall=${confidence.overall.toFixed(2)}`,
    `evidence=${confidence.evidenceStrength.toFixed(2)}(${evidenceComponents})`,
    // Both ratios print with the fraction they came from. Their denominators
    // are neither `samples` nor each other: coverage divides by the retrieval
    // limit for this turn (which also capped the episode half of its own
    // numerator), diversity by the top-N slice plus semantic hits. A quotient
    // alone cannot say whether a 1.00 measured breadth or merely hit a ceiling
    // it could not miss.
    `coverage=${confidence.coverage.toFixed(2)}(${confidence.sampleSize}/${confidence.coverageExpected})`,
    `diversity=${confidence.sourceDiversity.toFixed(2)}(${confidence.diversitySources}/${confidence.diversitySampleSize})`,
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
  disclosureLabel: MemoryDisclosureLabel;
  structuralFields: Record<string, string | number | boolean | null>;
  payload: unknown;
};

type IndexedVerificationRetrievalCandidate = {
  candidate: VerificationRetrievalCandidate;
  originalIndex: number;
};

export type PlanRequestedVerificationMembershipCarveOutOverflow = {
  rowsTotal: number;
  carveOutRowsTotal: number;
  carveOutRequiredTokens: number;
  membershipTargetTokens: number;
};

export type PlanRequestedVerificationRetrievalRenderOptions = {
  rowsTotalReadAtMs?: number;
  currentTimeMs?: number;
  onMembershipCarveOutOverflow?: (
    overflow: PlanRequestedVerificationMembershipCarveOutOverflow,
  ) => void;
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
    `${VERIFICATION_DISCLOSURE_CLASS_FIELD}=${exact.disclosureClass}`,
    `origin_audience=${list(exact.originAudienceEntityIds)}`,
    `private-to=${list(exact.privateToEntityIds)}`,
    `public-to=${list(exact.publicToEntityIds)}`,
  ].join(" ");
}

function verificationEvidenceCandidates(
  evidence: readonly EvidenceItem[],
): VerificationRetrievalCandidate[] {
  return evidence.map((item) => {
    const disclosureLabel = item.disclosureLabel ?? unknownMemoryDisclosureLabel();
    return {
      handle: item.id,
      sourceClass: "evidence",
      disclosureLabel,
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
        ...(item.provenance?.commitmentId === undefined
          ? {}
          : {
              [VERIFICATION_COMMITMENT_ENFORCEMENT_CLASS_FIELD]:
                item.commitment_enforcement_class ?? null,
              [VERIFICATION_COMMITMENT_CRITICAL_DOMAIN_FIELD]:
                item.commitment_critical_domain ?? null,
            }),
      },
      payload: {
        text: item.text,
        matched_terms: item.matchedTerms,
        image_label: item.imageLabel ?? null,
        image_origin_frame: item.imageOriginFrame ?? null,
        image_unavailable_reason: item.imageUnavailableReason ?? null,
      },
    };
  });
}

function verificationFallbackCandidates(
  input: Pick<RetrievedContext, "episodes" | "semantic" | "open_questions">,
): VerificationRetrievalCandidate[] {
  const episodes: VerificationRetrievalCandidate[] = input.episodes.map((result) => {
    const disclosureLabel =
      result.disclosureLabel ?? memoryDisclosureLabelFromEpisodeAccess(result.episode);
    return {
      handle: result.episode.id,
      sourceClass: "episode",
      disclosureLabel,
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
    };
  });
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
  const nodes: VerificationRetrievalCandidate[] = [...nodesById.values()].map((node) => {
    const disclosureLabel = node.disclosureLabel ?? unknownMemoryDisclosureLabel();
    return {
      handle: node.id,
      sourceClass: "semantic_node",
      disclosureLabel,
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
    };
  });
  const edges: VerificationRetrievalCandidate[] = [...edgesById.values()].map((edge) => {
    const disclosureLabel = edge.disclosureLabel ?? unknownMemoryDisclosureLabel();
    return {
      handle: edge.id,
      sourceClass: "semantic_edge",
      disclosureLabel,
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
    };
  });
  const openQuestions: VerificationRetrievalCandidate[] = input.open_questions.map((question) => {
    const disclosureLabel = openQuestionMemoryDisclosureLabel(question);
    return {
      handle: question.id,
      sourceClass: "open_question",
      disclosureLabel,
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
    };
  });
  return [...episodes, ...nodes, ...edges, ...openQuestions];
}

function verificationPayloadJson(candidate: VerificationRetrievalCandidate): string {
  return JSON.stringify(candidate.payload) ?? "null";
}

function renderVerificationRetrievalCandidate(
  candidate: VerificationRetrievalCandidate,
  includePayload: boolean,
): string {
  const payloadJson = verificationPayloadJson(candidate);
  const structural = Object.entries(candidate.structuralFields)
    .map(([key, value]) => `${key}="${verificationXmlAttribute(String(value ?? "none"))}"`)
    .join(" ");
  return [
    `<verification_source handle="${verificationXmlAttribute(candidate.handle)}"`,
    `source_class="${candidate.sourceClass}"`,
    `disclosure="${verificationXmlAttribute(verificationDisclosure(candidate.disclosureLabel))}"`,
    structural,
    `payload_status="${includePayload ? "exact" : "check_not_completed_budget"}"`,
    `payload_included_chars="${includePayload ? payloadJson.length : 0}"`,
    `payload_total_chars="${payloadJson.length}"`,
    `payload_json="${includePayload ? verificationXmlAttribute(payloadJson) : ""}" />`,
  ].join(" ");
}

function isVerificationMembershipCarveOut(candidate: VerificationRetrievalCandidate): boolean {
  const commitmentId = candidate.structuralFields.provenance_commitment_id;
  const criticalDomain = candidate.structuralFields[VERIFICATION_COMMITMENT_CRITICAL_DOMAIN_FIELD];
  return (
    typeof commitmentId === "string" &&
    (candidate.structuralFields[VERIFICATION_COMMITMENT_ENFORCEMENT_CLASS_FIELD] === "critical" ||
      typeof criticalDomain === "string")
  );
}

function estimateVerificationMembershipTokens(characters: number): number {
  return characters === 0 ? 0 : estimatePromptTokensFromLength(characters);
}

function verificationDisclosureRemainders(
  candidates: readonly IndexedVerificationRetrievalCandidate[],
): ReadonlyMap<MemoryDisclosureClass, number> {
  const counts = new Map<MemoryDisclosureClass, number>();

  for (const { candidate } of candidates) {
    const disclosureClass = candidate.disclosureLabel.disclosureClass;
    counts.set(disclosureClass, (counts.get(disclosureClass) ?? 0) + 1);
  }

  return counts;
}

function renderVerificationMembershipBudgetMarker(
  candidates: readonly IndexedVerificationRetrievalCandidate[],
): string[] {
  if (candidates.length === 0) {
    return [];
  }

  const disclosureRemainderAttributes = [...verificationDisclosureRemainders(candidates).entries()]
    .map(([disclosureClass, count]) => `${disclosureClass}="${count}"`)
    .join(" ");

  return [
    `  <${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER} ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_REMAINDER_TOTAL_ATTRIBUTE}="${candidates.length}">`,
    `    <${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER} ${disclosureRemainderAttributes} />`,
    `  </${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}>`,
  ];
}

function renderVerificationRetrievalRows(
  candidates: readonly IndexedVerificationRetrievalCandidate[],
  omittedCandidates: readonly IndexedVerificationRetrievalCandidate[],
  rowsTotal: number,
  included: ReadonlySet<number>,
  payloadTokens: number,
  maxTokens: number,
  membershipMaxTokens: number,
  membershipTokens: number,
  carveOutRowsTotal: number,
  rowsTotalReadAtMs: number,
  currentTimeMs: number,
): string {
  const rows = candidates.map(({ candidate, originalIndex }) =>
    renderVerificationRetrievalCandidate(candidate, included.has(originalIndex)),
  );
  const incompleteCount = candidates.reduce(
    (count, { originalIndex }) => count + (included.has(originalIndex) ? 0 : 1),
    0,
  );
  const membershipNotEnumeratedCount = omittedCandidates.length;
  const rowsTotalAsOf = new Date(rowsTotalReadAtMs).toISOString();
  const readOffsetFromCurrentTimeMs = Math.trunc(rowsTotalReadAtMs - currentTimeMs);
  const relativeReadTime =
    readOffsetFromCurrentTimeMs === 0
      ? ""
      : `, ${Math.abs(readOffsetFromCurrentTimeMs)}ms ${
          readOffsetFromCurrentTimeMs > 0 ? "after" : "before"
        } the current_time_ms at the top of this prompt`;
  const membershipBudgetMarkerAttribute =
    membershipNotEnumeratedCount === 0
      ? ""
      : ` ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}="${membershipNotEnumeratedCount}"`;
  const membershipBudgetMarker = renderVerificationMembershipBudgetMarker(omittedCandidates);
  return [
    `<plan_requested_verification_retrieval complete_membership="${membershipNotEnumeratedCount === 0}" rows_total="${rowsTotal}" ${PLAN_REQUESTED_VERIFICATION_ROWS_TOTAL_AS_OF_ATTRIBUTE}="${rowsTotalAsOf}" ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER_ATTRIBUTE}="${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER}" ${PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER_ATTRIBUTE}="${PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER}" payload_target_tokens="${maxTokens}" payload_tokens_included="${payloadTokens}" membership_target_tokens="${membershipMaxTokens}" membership_tokens="${membershipTokens}" ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_ROWS_ATTRIBUTE}="${carveOutRowsTotal}"${membershipBudgetMarkerAttribute} check_not_completed_count="${incompleteCount}">`,
    `  <interpretation>This retrieval was requested by the advisory plan. Read at ${rowsTotalAsOf}${relativeReadTime}: rows_total is exact as of that read, not as of now. Membership rows are priced against membership_target_tokens alone. ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER_ATTRIBUTE}=${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER} means commitment rows with ${VERIFICATION_COMMITMENT_ENFORCEMENT_CLASS_FIELD}=critical or a structural ${VERIFICATION_COMMITMENT_CRITICAL_DOMAIN_FIELD} are emitted first. The protected critical-commitment partition and then all ordinary rows preserve the existing retrieval-pipeline candidate order: ranked unified evidence, projected episodes, semantic nodes and edges in projected first-seen order, then projected open questions. Disclosure labels do not affect membership admission or ordering. The renderer performs no content-based selection or new ranking. complete_membership=true means every one of rows_total handles and its structural fields is enumerated. When complete_membership=false, ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}=N and its same-named marker's ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_REMAINDER_TOTAL_ATTRIBUTE}=N carry the exact un-enumerated remainder; the child ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER} has one attribute per ${VERIFICATION_DISCLOSURE_CLASS_FIELD} present among omitted rows, with exact counts whose sum is N. Omitted membership is never silent. If the critical-commitment carve-out alone exceeds membership_target_tokens, ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ERROR_ATTRIBUTE}=${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_OVERFLOW_ERROR} replaces a partial list with an explicit failed-check state. A payload_status=exact row carries its complete payload with no excerpt; payload_status=check_not_completed_budget carries an enumerated handle and structural fields but zero payload, so that requested check is explicitly incomplete rather than silently truncated. Payloads are priced against payload_target_tokens alone and never consume the membership budget, and membership never consumes the payload budget. ${PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER_ATTRIBUTE}=${PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER} means payload admission walks these enumerated rows in the order they are rendered here, so the critical-commitment carve-out is priced first on this axis too and rows that are not enumerated never consume payload budget. Admission does not stop at the first refusal: a later small payload can still fit after a larger one is refused. A check_not_completed_budget row therefore means that row's payload did not fit in the budget still free when its turn came, which depends on the payloads of the rows above it, not on that row's size alone; payload_total_chars states what was withheld. payload_tokens_included is the admitted total that gated, summed per row exactly as the gate summed it.</interpretation>`,
    ...rows.map((row) => `  ${row}`),
    ...membershipBudgetMarker,
    `  <omitted_count>${membershipNotEnumeratedCount}</omitted_count>`,
    `  <check_not_completed_count>${incompleteCount}</check_not_completed_count>`,
    "</plan_requested_verification_retrieval>",
  ].join("\n");
}

function renderVerificationMembershipCarveOutOverflow(
  details: PlanRequestedVerificationMembershipCarveOutOverflow,
  omittedCandidates: readonly IndexedVerificationRetrievalCandidate[],
  maxTokens: number,
  rowsTotalReadAtMs: number,
  currentTimeMs: number,
): string {
  const rowsTotalAsOf = new Date(rowsTotalReadAtMs).toISOString();
  const readOffsetFromCurrentTimeMs = Math.trunc(rowsTotalReadAtMs - currentTimeMs);
  const relativeReadTime =
    readOffsetFromCurrentTimeMs === 0
      ? ""
      : `, ${Math.abs(readOffsetFromCurrentTimeMs)}ms ${
          readOffsetFromCurrentTimeMs > 0 ? "after" : "before"
        } the current_time_ms at the top of this prompt`;
  return [
    `<plan_requested_verification_retrieval complete_membership="false" rows_total="${details.rowsTotal}" ${PLAN_REQUESTED_VERIFICATION_ROWS_TOTAL_AS_OF_ATTRIBUTE}="${rowsTotalAsOf}" ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER_ATTRIBUTE}="${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER}" payload_target_tokens="${maxTokens}" payload_tokens_included="0" membership_target_tokens="${details.membershipTargetTokens}" membership_tokens="0" ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_ROWS_ATTRIBUTE}="${details.carveOutRowsTotal}" ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_REQUIRED_TOKENS_ATTRIBUTE}="${details.carveOutRequiredTokens}" ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ERROR_ATTRIBUTE}="${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_OVERFLOW_ERROR}" ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}="${details.rowsTotal}" check_not_completed_count="0">`,
    `  <interpretation>This retrieval was requested by the advisory plan. Read at ${rowsTotalAsOf}${relativeReadTime}: rows_total is exact as of that read, not as of now. The structurally protected critical-commitment membership rows alone require ${details.carveOutRequiredTokens} tokens against membership_target_tokens=${details.membershipTargetTokens}. Commitments with ${VERIFICATION_COMMITMENT_ENFORCEMENT_CLASS_FIELD}=critical or a structural ${VERIFICATION_COMMITMENT_CRITICAL_DOMAIN_FIELD} may not be shortened, so no verification_source rows or payloads are rendered; this verification check failed explicitly instead of presenting a partial critical-commitment carve-out as complete. Disclosure labels do not affect membership admission or ordering. ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER} and its per-${VERIFICATION_DISCLOSURE_CLASS_FIELD} child carry the exact un-enumerated remainder. check_not_completed_count is zero because it counts rendered rows with incomplete payload status, and no rows were rendered. Payload pricing remains independent and was not attempted for this structurally failed membership check.</interpretation>`,
    ...renderVerificationMembershipBudgetMarker(omittedCandidates),
    `  <omitted_count>${details.rowsTotal}</omitted_count>`,
    "  <check_not_completed_count>0</check_not_completed_count>",
    `  <${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_OVERFLOW_MARKER} ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_ROWS_ATTRIBUTE}="${details.carveOutRowsTotal}" ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_REQUIRED_TOKENS_ATTRIBUTE}="${details.carveOutRequiredTokens}" membership_target_tokens="${details.membershipTargetTokens}" />`,
    "</plan_requested_verification_retrieval>",
  ].join("\n");
}

/**
 * Compact-terminal rendering for secondary retrieval driven structurally by
 * an S2 plan's non-empty verification_steps. Payloads are all-or-nothing:
 * exact when they fit, otherwise an explicit incomplete-check row.
 *
 * Membership and payloads have independent quotas. Critical-commitment rows
 * form a stable first partition that must fit whole; ordinary membership then
 * takes an unchanged prefix of its stable partition. Disclosure classes are
 * accounting dimensions for omitted rows, not admission gates.
 * Payload admission runs over the enumerated rows in rendered order, so the
 * critical-commitment carve-out is priced first on the payload axis as well and
 * rows omitted from membership never consume payload budget. It keeps scanning
 * past a refusal, so a small payload later in the rendered membership can still
 * fit after a large one is refused.
 */
export function renderPlanRequestedVerificationRetrieval(
  input: Pick<RetrievedContext, "evidence" | "episodes" | "semantic" | "open_questions">,
  maxTokens = DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET,
  membershipMaxTokens = DEFAULT_PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_TOKEN_BUDGET,
  options: PlanRequestedVerificationRetrievalRenderOptions = {},
): string {
  const candidates = [
    ...verificationEvidenceCandidates(input.evidence),
    ...verificationFallbackCandidates(input),
  ];
  const rowsTotalReadAtMs = options.rowsTotalReadAtMs ?? Date.now();
  const currentTimeMs = options.currentTimeMs ?? rowsTotalReadAtMs;
  const indexedCandidates = candidates.map((candidate, originalIndex) => ({
    candidate,
    originalIndex,
  }));
  const carveOutCandidates = indexedCandidates.filter(({ candidate }) =>
    isVerificationMembershipCarveOut(candidate),
  );
  const ordinaryCandidates = indexedCandidates.filter(
    ({ candidate }) => !isVerificationMembershipCarveOut(candidate),
  );
  const carveOutCharacters = carveOutCandidates.reduce(
    (sum, { candidate }) => sum + renderVerificationRetrievalCandidate(candidate, false).length,
    0,
  );
  const carveOutRequiredTokens = estimateVerificationMembershipTokens(carveOutCharacters);

  if (carveOutRequiredTokens > membershipMaxTokens) {
    const overflow: PlanRequestedVerificationMembershipCarveOutOverflow = {
      rowsTotal: candidates.length,
      carveOutRowsTotal: carveOutCandidates.length,
      carveOutRequiredTokens,
      membershipTargetTokens: membershipMaxTokens,
    };
    options.onMembershipCarveOutOverflow?.(overflow);
    return renderVerificationMembershipCarveOutOverflow(
      overflow,
      indexedCandidates,
      maxTokens,
      rowsTotalReadAtMs,
      currentTimeMs,
    );
  }

  let membershipCharacters = carveOutCharacters;
  const enumeratedCandidates = [...carveOutCandidates];
  for (const indexedCandidate of ordinaryCandidates) {
    const { candidate } = indexedCandidate;
    const rowLength = renderVerificationRetrievalCandidate(candidate, false).length;
    const nextMembershipCharacters = membershipCharacters + rowLength;
    if (estimateVerificationMembershipTokens(nextMembershipCharacters) > membershipMaxTokens) {
      break;
    }
    membershipCharacters = nextMembershipCharacters;
    enumeratedCandidates.push(indexedCandidate);
  }
  const omittedCandidates = ordinaryCandidates.slice(
    enumeratedCandidates.length - carveOutCandidates.length,
  );

  const included = new Set<number>();
  let payloadTokens = 0;
  for (const { candidate, originalIndex } of enumeratedCandidates) {
    const cost = estimatePromptTokens(verificationPayloadJson(candidate));
    if (payloadTokens + cost > maxTokens) {
      continue;
    }
    payloadTokens += cost;
    included.add(originalIndex);
  }

  return renderVerificationRetrievalRows(
    enumeratedCandidates,
    omittedCandidates,
    candidates.length,
    included,
    payloadTokens,
    maxTokens,
    membershipMaxTokens,
    estimateVerificationMembershipTokens(membershipCharacters),
    carveOutCandidates.length,
    rowsTotalReadAtMs,
    currentTimeMs,
  );
}

export function renderPlanRequestedVerificationNotCompleted(): string {
  return [
    `<plan_requested_verification_retrieval ${PLAN_REQUESTED_VERIFICATION_RETRIEVAL_STATUS_ATTRIBUTE}="${PLAN_REQUESTED_VERIFICATION_RETRIEVAL_UNAVAILABLE_STATUS}" ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_STATUS_ATTRIBUTE}="${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_NOT_OBSERVED_STATUS}" check_not_completed_count="1">`,
    "  <interpretation>The advisory plan requested a verification pass, but secondary retrieval was unavailable. No membership read occurred, so complete_membership, rows_total, omitted_count, and membership omission markers are intentionally absent. The request handle remains visible with an explicit incomplete payload status and zero payload characters; check_not_completed_count counts that one rendered incomplete row.</interpretation>",
    '  <verification_source handle="plan:verification_steps" source_class="verification_request" payload_status="check_not_completed_retrieval_unavailable" payload_included_chars="0" payload_total_chars="0" payload_json="" />',
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
