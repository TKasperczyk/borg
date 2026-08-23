import type { LLMSystemBlock } from "../../../llm/index.js";
import {
  effectiveCommitmentCriticalDomain,
  effectiveCommitmentEnforcementClass,
  type CommitmentRecord,
} from "../../../memory/commitments/index.js";
import { summarizeProvenanceForPrompt } from "../../../memory/common/index.js";
import {
  commitmentMemoryDisclosureLabel,
  relationalSlotMemoryDisclosureLabel,
} from "../../../memory/common/disclosure-serializers.js";
import {
  combineMemoryDisclosureLabels,
  MEMORY_DISCLOSURE_GUIDANCE_FOR_MODEL,
  memoryDisclosureLabelFromMetadata,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../../../retrieval/index.js";
import { escapeXmlText } from "../../../util/prompt-tags.js";
import { formatRelativeAge } from "../../../util/relative-time.js";
import { estimatePromptTokens } from "../../../util/token-estimate.js";
import { renderAutonomousOutboundActionAvailabilitySection } from "../../../outbound/outbound-prompt.js";
import {
  CURRENT_USER_MESSAGE_REMINDER,
  UNTRUSTED_DATA_PREAMBLE,
} from "../../prompts/base-identity.js";
import type { PromptSurfaceAdditionalSection } from "../../prompts/prompt-surface-registry.js";
import { GROUP_CHAT_SENDER_SCOPING_REMINDER } from "../../prompts/participation.js";
import type { EvidenceLedgerEntry } from "../../evidence-ledger/index.js";
import type {
  CreatorDirectiveBriefingDirective,
  CreatorDirectiveBriefingPrivateDirective,
  DeliberationContext,
} from "../types.js";
import {
  headTailPlannerExcerpt,
  renderGoalDigest,
  renderLivedExperienceDigest,
  type RenderedPlannerSection,
} from "./planner-context.js";
import {
  buildAutonomousOutboundAuthorizationSection,
  buildBaseSystemPromptSections,
  renderAuthorityContextLines,
  renderCreatorIdentity,
  renderCurrentTimeSection,
  renderPromptSection,
  renderSessionStatusSnapshotLines,
  INTERIM_CREATOR_DIRECTIVE_BOUNDARY_PROMPT,
  type BaseSystemPromptSections,
  type BuildBaseSystemPromptOptions,
} from "./system-prompt.js";

export type FinalizerResolvedSurfaceVariant = "compact" | "legacy";
export type FinalizerSurfaceVariant = FinalizerResolvedSurfaceVariant | "compact_conversational";
export type FinalizerCacheTier =
  | "terminal_static_head"
  | "terminal_durable_global"
  | "terminal_durable_audience"
  | "terminal_turn_context";

export type FinalizerSectionTraceSummary = {
  chars: number;
  estimatedTokens: number;
  rowCount: number;
  truncationCount: number;
  omissionCount: number;
  cacheTier: FinalizerCacheTier;
};

export type FinalizerContextTraceSummary = {
  variant: FinalizerResolvedSurfaceVariant;
  path: "system_1" | "system_2";
  sections: Record<string, FinalizerSectionTraceSummary>;
  blocks: Record<FinalizerCacheTier, { chars: number; estimatedTokens: number; ttl: "1h" | "5m" }>;
  totalChars: number;
  totalEstimatedTokens: number;
  rowCount: number;
  truncationCount: number;
  omissionCount: number;
};

export type CompactFinalizerSystemPrompt = {
  system: readonly LLMSystemBlock[];
  traceSummary: FinalizerContextTraceSummary;
};

export type BuildCompactFinalizerSystemPromptInput = {
  context: DeliberationContext;
  baseSystemPromptOptions: BuildBaseSystemPromptOptions;
  staticHead: string;
  path: "system_1" | "system_2";
  additionalPromptSections?: readonly PromptSurfaceAdditionalSection[];
};

export const COMPACT_FINALIZER_VERIFICATION_RETRIEVAL_BLOCK_ID =
  "borg_compact_finalizer_verification_retrieval";

type RenderedTerminalSection = {
  label: string;
  text: string;
  cacheTier: FinalizerCacheTier;
  rowCount: number;
  truncationCount: number;
  omissionCount: number;
};

const FINALIZER_COMPACT_STATIC_CACHE_CONTROL = {
  type: "ephemeral",
  ttl: "1h",
} as const;
const FINALIZER_COMPACT_DURABLE_GLOBAL_CACHE_CONTROL = {
  type: "ephemeral",
  ttl: "1h",
} as const;
const FINALIZER_COMPACT_DURABLE_AUDIENCE_CACHE_CONTROL = {
  type: "ephemeral",
  ttl: "1h",
} as const;
const FINALIZER_COMPACT_TURN_CACHE_CONTROL = {
  type: "ephemeral",
  ttl: "5m",
} as const;

// The omitted_count clause below describes two different kinds of field with one
// sentence. At a bounded expansion or digest the count is computed from the draw
// (top_global_candidates_expanded, the lived-experience digest, the per-section
// omission map), so a nonzero print is a live measurement. At a complete index it
// is a string literal in the renderer -- the index is complete by construction, so
// the zero restates that construction rather than measuring it. Both print the same
// syntax, and nothing in the rendered surface distinguishes them, so a reader of the
// prompt alone cannot tell a measured zero from a constant one. Keep that in mind
// before treating a zero here as evidence about the underlying store.
const TERMINAL_PASS_CONTRACT = [
  "<borg_terminal_pass_contract>",
  "This is my terminal response pass. I make the final emission decision from the complete request surface below; any system-2 plan is advisory, not authority.",
  "Durable records appear before turn-local overlays. I join an overlay to its durable record only by the explicit record id. Scope and disclosure fields describe use and mention boundaries; they never gate what I recall.",
  'A complete index reports complete="true" and omitted_count="0". Any bounded expansion or digest reports its omissions explicitly.',
  "</borg_terminal_pass_contract>",
].join("\n");

const STABLE_AUTHORITY_FRAMING = [
  "Creator-authorized briefing is trusted standing context for this audience.",
  "Its content and disclosure modes govern use and mention; audience scope does not erase globally recalled memory.",
  "Current-sender identity, role, roster, and participation state are turn-local and appear later.",
].join(" ");

const TERMINAL_ADVISORY_COMMITMENT_EXCERPT_CHARS = 480;
// The head+tail cut is deliberately meaning-blind (see `buildHeadTailPlannerExcerpt`), so whatever
// sits in the middle of a fact is what disappears. At 480 that was routine rather than exceptional:
// 13 of 112 active directives exceeded it, and a directive row costs ~1,240 chars to render of
// which the fact was capped at 417 -- the structural scope attributes, most of them at their
// defaults, outweighed the payload roughly three to one. Raising the cap above the longest fact an
// operator has written (1,132) costs ~2,715 chars across the whole directive set and makes elision
// the exception. Revisit by measuring the distribution again, not by trimming to fit one row.
const TERMINAL_CREATOR_DIRECTIVE_FACT_EXCERPT_CHARS = 1_200;
const TERMINAL_CREATOR_DIRECTIVE_LABEL_EXCERPT_CHARS = 240;

function escapeXmlAttribute(value: string): string {
  return escapeXmlText(value).replaceAll('"', "&quot;");
}

function escapeXmlSingleLineAttribute(value: string): string {
  return escapeXmlAttribute(value)
    .replaceAll("\r", "&#13;")
    .replaceAll("\n", "&#10;")
    .replaceAll("\t", "&#9;");
}

function iso(timestamp: number | null | undefined): string {
  return timestamp === null || timestamp === undefined || !Number.isFinite(timestamp)
    ? "none"
    : new Date(timestamp).toISOString();
}

function age(timestamp: number | null | undefined, nowMs: number | undefined): string {
  return timestamp === null ||
    timestamp === undefined ||
    nowMs === undefined ||
    !Number.isFinite(timestamp) ||
    !Number.isFinite(nowMs)
    ? "unknown"
    : formatRelativeAge(timestamp, nowMs);
}

function terminalSection(
  label: string,
  cacheTier: FinalizerCacheTier,
  text: string,
  counts: Partial<
    Pick<RenderedTerminalSection, "rowCount" | "truncationCount" | "omissionCount">
  > = {},
): RenderedTerminalSection {
  return {
    label,
    text,
    cacheTier,
    rowCount: counts.rowCount ?? 0,
    truncationCount: counts.truncationCount ?? 0,
    omissionCount: counts.omissionCount ?? 0,
  };
}

function tagged(tag: string, content: string | null): string | null {
  return content === null ? null : `<${tag}>\n${content}\n</${tag}>`;
}

function compactDisclosure(label: MemoryDisclosureLabel): string {
  const list = (values: readonly string[]) => (values.length === 0 ? "none" : values.join(","));
  return [
    `disclosure_class=${label.disclosureClass}`,
    `origin_audience=${list(label.originAudienceEntityIds)}`,
    `private-to=${list(label.privateToEntityIds)}`,
    `public-to=${list(label.publicToEntityIds)}`,
  ].join(" ");
}

function evidenceEntryDisclosure(entry: EvidenceLedgerEntry): MemoryDisclosureLabel {
  return (
    memoryDisclosureLabelFromMetadata(entry.state_metadata?.disclosure_label) ??
    unknownMemoryDisclosureLabel()
  );
}

function joinedAttribute(values: readonly string[] | undefined): string {
  return values === undefined || values.length === 0 ? "none" : values.join(",");
}

function metadataAttribute(metadata: EvidenceLedgerEntry["state_metadata"]): string {
  return metadata === undefined ? "none" : (JSON.stringify(metadata) ?? "none");
}

function commitmentStatus(commitment: CommitmentRecord): string {
  if (commitment.revoked_at !== null) return "revoked";
  if (commitment.expired_at !== null) return "expired";
  return "active";
}

function combinedCommitmentDisclosure(
  commitment: CommitmentRecord,
  ledgerEntry: EvidenceLedgerEntry | undefined,
): MemoryDisclosureLabel {
  // A canonical record and its standing-ledger projection are two provenance
  // claims about the same row. Combining their labels is deliberately
  // fail-closed: an absent/malformed ledger label contributes `unknown`.
  return ledgerEntry === undefined
    ? commitmentMemoryDisclosureLabel(commitment)
    : combineMemoryDisclosureLabels([
        commitmentMemoryDisclosureLabel(commitment),
        evidenceEntryDisclosure(ledgerEntry),
      ]);
}

function renderCommitmentIdentityFields(commitment: CommitmentRecord, ordinal: number): string[] {
  return [
    `<commitment id="${escapeXmlAttribute(commitment.id)}"`,
    `ordinal="${ordinal}"`,
    `canonical_record="true"`,
    `status="${commitmentStatus(commitment)}"`,
    `enforcement_class="${effectiveCommitmentEnforcementClass(commitment)}"`,
    `critical_domain="${escapeXmlAttribute(effectiveCommitmentCriticalDomain(commitment) ?? "none")}"`,
    `kind="${escapeXmlAttribute(commitment.kind)}"`,
    `type="${escapeXmlAttribute(commitment.type)}"`,
    `family="${escapeXmlAttribute(commitment.directive_family)}"`,
    `closure_pressure="${escapeXmlAttribute(commitment.closure_pressure_relevance)}"`,
    `priority="${commitment.priority}"`,
    `record_version="${commitment.record_version ?? "unknown"}"`,
  ];
}

function renderCommitmentTimelineAndScopeFields(
  commitment: CommitmentRecord,
  disclosure: MemoryDisclosureLabel,
): string[] {
  const provenance = escapeXmlSingleLineAttribute(
    summarizeProvenanceForPrompt(commitment.provenance, Number.MAX_SAFE_INTEGER),
  );
  const sourceIds = commitment.source_stream_entry_ids?.join(",") ?? "none";
  return [
    `created_at="${iso(commitment.created_at)}"`,
    `updated_at="${iso(commitment.updated_at)}"`,
    `last_reinforced_at="${iso(commitment.last_reinforced_at)}"`,
    `expires_at="${iso(commitment.expires_at)}"`,
    `expired_at="${iso(commitment.expired_at)}"`,
    `revoked_at="${iso(commitment.revoked_at)}"`,
    `superseded_by="${escapeXmlAttribute(commitment.superseded_by ?? "none")}"`,
    `made_to_entity_id="${escapeXmlAttribute(commitment.made_to_entity ?? "none")}"`,
    `restricted_audience_id="${escapeXmlAttribute(commitment.restricted_audience ?? "none")}"`,
    `about_entity_id="${escapeXmlAttribute(commitment.about_entity ?? "none")}"`,
    `committed_by_entity_id="${escapeXmlAttribute(commitment.committed_by_entity_id ?? "none")}"`,
    `disclosure="${escapeXmlAttribute(compactDisclosure(disclosure))}"`,
    `provenance="${provenance}"`,
    `source_stream_entry_ids="${escapeXmlAttribute(sourceIds)}"`,
    `canonicalized_by="${escapeXmlAttribute(commitment.canonicalized_by_artifact_entry_id ?? "none")}"`,
  ];
}

function renderCommitmentLedgerFields(
  commitment: CommitmentRecord,
  ledgerEntry: EvidenceLedgerEntry | undefined,
): string[] {
  return [
    `ledger_ref="${escapeXmlAttribute(ledgerEntry?.id ?? "none")}"`,
    `ledger_source_type="${escapeXmlAttribute(ledgerEntry?.source_type ?? "commitment")}"`,
    `ledger_actor="${escapeXmlAttribute(ledgerEntry?.actor ?? "memory")}"`,
    `ledger_trust_rank="${ledgerEntry?.trust_rank ?? "unknown"}"`,
    `ledger_state="${escapeXmlAttribute(ledgerEntry?.state ?? commitmentStatus(commitment))}"`,
    `ledger_salience_class="${escapeXmlAttribute(ledgerEntry?.salience_class ?? "none")}"`,
    `ledger_taint="${escapeXmlAttribute(ledgerEntry?.taint ?? "none")}"`,
    `ledger_value="${escapeXmlSingleLineAttribute(ledgerEntry === undefined ? "none" : ledgerEntry.value === commitment.directive_family ? "same_as_family" : (ledgerEntry.value ?? "none"))}"`,
    `ledger_text="${escapeXmlSingleLineAttribute(ledgerEntry === undefined ? "none" : ledgerEntry.text === commitment.directive ? "same_as_directive" : (ledgerEntry.text ?? "none"))}"`,
    `persistence_class="${escapeXmlAttribute(ledgerEntry?.persistence_class ?? "unknown")}"`,
    `stream_index="${ledgerEntry?.stream_index ?? "none"}"`,
    `citation_type="${escapeXmlAttribute(ledgerEntry?.citation_type ?? "none")}"`,
    `citations="${escapeXmlSingleLineAttribute(joinedAttribute(ledgerEntry?.citations))}"`,
  ];
}

function renderCommitmentRecord(
  commitment: CommitmentRecord,
  ledgerEntry: EvidenceLedgerEntry | undefined,
  ordinal: number,
): { row: string; truncationCount: number } {
  const disclosure = combinedCommitmentDisclosure(commitment, ledgerEntry);
  const critical = effectiveCommitmentEnforcementClass(commitment) === "critical";
  const directive = critical
    ? headTailPlannerExcerpt(commitment.directive, Number.MAX_SAFE_INTEGER)
    : headTailPlannerExcerpt(commitment.directive, TERMINAL_ADVISORY_COMMITMENT_EXCERPT_CHARS);
  return {
    row: [
      ...renderCommitmentIdentityFields(commitment, ordinal),
      ...renderCommitmentTimelineAndScopeFields(commitment, disclosure),
      ...renderCommitmentLedgerFields(commitment, ledgerEntry),
      `directive_exact="${!directive.truncated}"`,
      `directive_excerpt_shape="${directive.truncated ? "head+tail" : "full"}"`,
      `directive_included_chars="${directive.renderedChars}"`,
      `directive_total_chars="${directive.totalChars}"`,
      `directive="${escapeXmlSingleLineAttribute(directive.text)}" />`,
    ].join(" "),
    truncationCount: directive.truncated ? 1 : 0,
  };
}

function renderLedgerOnlyCommitmentRecord(entry: EvidenceLedgerEntry): {
  row: string;
  truncationCount: number;
} {
  const metadata = entry.state_metadata ?? {};
  const directive = entry.text ?? "";
  const disclosure = compactDisclosure(
    memoryDisclosureLabelFromMetadata(metadata.disclosure_label) ?? unknownMemoryDisclosureLabel(),
  );
  const attribute = (key: string, fallback = "unknown") => {
    const value = metadata[key];
    return typeof value === "string" || typeof value === "number" ? String(value) : fallback;
  };
  const critical = attribute("commitment_enforcement_class") === "critical";
  const excerpt = critical
    ? headTailPlannerExcerpt(directive, Number.MAX_SAFE_INTEGER)
    : headTailPlannerExcerpt(directive, TERMINAL_ADVISORY_COMMITMENT_EXCERPT_CHARS);
  return {
    row: [
      `<commitment id="${escapeXmlAttribute(entry.id)}" canonical_record="false"`,
      `status="${escapeXmlAttribute(entry.state ?? "unknown")}"`,
      `enforcement_class="${escapeXmlAttribute(attribute("commitment_enforcement_class"))}"`,
      `critical_domain="${escapeXmlAttribute(attribute("commitment_critical_domain", "none"))}"`,
      `kind="${escapeXmlAttribute(attribute("commitment_kind"))}"`,
      `type="${escapeXmlAttribute(attribute("commitment_type"))}"`,
      `family="${escapeXmlSingleLineAttribute(entry.value ?? "unknown")}"`,
      `created_at="${escapeXmlAttribute(attribute("created_at"))}"`,
      `last_reinforced_at="${escapeXmlAttribute(attribute("last_reinforced_at"))}"`,
      `made_to_entity_id="${escapeXmlAttribute(attribute("made_to_entity_id", "none"))}"`,
      `restricted_audience_id="${escapeXmlAttribute(attribute("restricted_audience_id", "none"))}"`,
      `about_entity_id="${escapeXmlAttribute(attribute("about_entity_id", "none"))}"`,
      `committed_by_entity_id="${escapeXmlAttribute(attribute("committed_by_entity_id", "none"))}"`,
      `disclosure="${escapeXmlAttribute(disclosure)}"`,
      `ledger_ref="${escapeXmlAttribute(entry.id)}"`,
      `ledger_source_type="${escapeXmlAttribute(entry.source_type)}"`,
      `ledger_actor="${escapeXmlAttribute(entry.actor)}"`,
      `ledger_trust_rank="${entry.trust_rank}"`,
      `ledger_state="${escapeXmlAttribute(entry.state ?? "unknown")}"`,
      `ledger_salience_class="${escapeXmlAttribute(entry.salience_class ?? "none")}"`,
      `ledger_taint="${escapeXmlAttribute(entry.taint ?? "none")}"`,
      `ledger_value="same_as_family"`,
      `ledger_text="same_as_directive"`,
      `persistence_class="${escapeXmlAttribute(entry.persistence_class ?? "unknown")}"`,
      `stream_index="${entry.stream_index ?? "none"}"`,
      `citation_type="${escapeXmlAttribute(entry.citation_type ?? "none")}"`,
      `citations="${escapeXmlSingleLineAttribute(joinedAttribute(entry.citations))}"`,
      `directive_exact="${!excerpt.truncated}"`,
      `directive_excerpt_shape="${excerpt.truncated ? "head+tail" : "full"}"`,
      `directive_included_chars="${excerpt.renderedChars}"`,
      `directive_total_chars="${excerpt.totalChars}"`,
      `directive="${escapeXmlSingleLineAttribute(excerpt.text)}" />`,
    ].join(" "),
    truncationCount: excerpt.truncated ? 1 : 0,
  };
}

// applicableCommitments arrives from recallActiveCommitmentsForCognition, whose
// predicate (CommitmentRepository.isActiveCommitment) is exactly "no retirement mark
// set". So expires_at / expired_at / revoked_at / superseded_by render as `none` on
// every row here by construction: carrying a value on any of the four is what removes
// a row from this draw. That makes the four columns a survivorship artifact of the
// render, not a report on the store -- reading `none` across all rows says nothing
// about whether the retirement paths ever fire. complete="true" and rows_total are
// likewise statements about the active set, not about the commitments table.
function renderCommitments(context: DeliberationContext): RenderedTerminalSection {
  const commitments = context.applicableCommitments ?? [];
  const ledgerEntries = context.evidenceLedger?.audienceStanding?.commitmentEntries ?? [];
  const ledgerByCanonicalId = new Map(ledgerEntries.map((entry) => [entry.id, entry]));
  const matchedLedgerIds = new Set<string>();
  const canonicalRendered = commitments.map((commitment, index) => {
    const ledgerEntry = ledgerByCanonicalId.get(`commitment:${commitment.id}`);
    if (ledgerEntry !== undefined) matchedLedgerIds.add(ledgerEntry.id);
    return renderCommitmentRecord(commitment, ledgerEntry, index + 1);
  });
  const ledgerOnlyRendered = ledgerEntries
    .filter((entry) => !matchedLedgerIds.has(entry.id))
    .map(renderLedgerOnlyCommitmentRecord);
  const rows = [...canonicalRendered, ...ledgerOnlyRendered].map((entry) => entry.row);
  const truncationCount = [...canonicalRendered, ...ledgerOnlyRendered].reduce(
    (sum, entry) => sum + entry.truncationCount,
    0,
  );
  return terminalSection(
    "commitments",
    "terminal_durable_global",
    [
      `<borg_terminal_commitments complete="true" rows_total="${rows.length}" canonical_rows="${canonicalRendered.length}" ledger_only_rows="${ledgerOnlyRendered.length}" advisory_excerpt_budget_chars="${TERMINAL_ADVISORY_COMMITMENT_EXCERPT_CHARS}">`,
      "  <interpretation>One row per globally assembled commitment: canonical records first, then any distinct standing-ledger-only records. Critical directives are exact. A long advisory directive is a visibly annotated mechanical head+tail cut carrying both included and total source-character counts, never a clean-looking summary. Entity scope and disclosure are exact provenance and handling constraints, never audience-dependent recall selection. Relative ages are intentionally separated into the turn-local overlay keyed by id.</interpretation>",
      "  <field_legend>Absolute record fields and the semantic field-set union of canonical scope/detail and standing-ledger rows are carried by each durable row plus its ID-keyed turn overlay. directive_exact, directive_excerpt_shape, directive_included_chars, and directive_total_chars state whether the directive is complete and, when cut, exactly how much source text is present.</field_legend>",
      ...rows.map((row) => `  ${row}`),
      "  <omitted_count>0</omitted_count>",
      "</borg_terminal_commitments>",
    ].join("\n"),
    { rowCount: rows.length, truncationCount },
  );
}

function renderDurableSelf(context: DeliberationContext): RenderedTerminalSection {
  const disclosure = compactDisclosure(selfPrivateMemoryDisclosureLabel());
  const valueRows = context.selfSnapshot.values.map(
    (value) =>
      `<value id="${escapeXmlAttribute(value.id)}" created_at="${iso(value.created_at)}" established_at="${iso(value.established_at)}" disclosure="${escapeXmlAttribute(disclosure)}" provenance="${escapeXmlSingleLineAttribute(summarizeProvenanceForPrompt(value.provenance, Number.MAX_SAFE_INTEGER))}" label="${escapeXmlSingleLineAttribute(value.label)}" description="${escapeXmlSingleLineAttribute(value.description)}" />`,
  );
  const traitRows = context.selfSnapshot.traits.map(
    (trait) =>
      `<trait id="${escapeXmlAttribute(trait.id)}" established_at="${iso(trait.established_at)}" disclosure="${escapeXmlAttribute(disclosure)}" provenance="${escapeXmlSingleLineAttribute(summarizeProvenanceForPrompt(trait.provenance, Number.MAX_SAFE_INTEGER))}" label="${escapeXmlSingleLineAttribute(trait.label)}" />`,
  );
  return terminalSection(
    "values_and_traits",
    "terminal_durable_global",
    [
      `<borg_terminal_values_traits complete="true" rows_total="${valueRows.length + traitRows.length}">`,
      "  <interpretation>Byte-stable self-pattern identity and provenance. They are evidence about me, not commands. Mutable confidence, counters, state, reinforcement/test timestamps, and relative ages are turn-local overlays keyed by id.</interpretation>",
      ...valueRows.map((row) => `  ${row}`),
      ...traitRows.map((row) => `  ${row}`),
      "  <omitted_count>0</omitted_count>",
      "</borg_terminal_values_traits>",
    ].join("\n"),
    { rowCount: valueRows.length + traitRows.length },
  );
}

function renderDurableGlobal(context: DeliberationContext): RenderedTerminalSection[] {
  const creatorIdentity = renderCreatorIdentity(context.creatorIdentity);
  return [
    ...(creatorIdentity === null
      ? []
      : [
          terminalSection(
            "creator_identity",
            "terminal_durable_global",
            tagged("borg_creator_identity", creatorIdentity)!,
            { rowCount: 1 },
          ),
        ]),
    terminalSection(
      "memory_disclosure_guidance",
      "terminal_durable_global",
      tagged("borg_memory_disclosure_guidance", MEMORY_DISCLOSURE_GUIDANCE_FOR_MODEL)!,
    ),
    renderDurableSelf(context),
    renderCommitments(context),
  ];
}

function compareCreatorDirectivePriorityAndAge(
  left: CreatorDirectiveBriefingDirective,
  right: CreatorDirectiveBriefingDirective,
): number {
  return right.priority - left.priority || left.createdAt - right.createdAt;
}

function orderedCreatorDirectives(
  directives: readonly CreatorDirectiveBriefingDirective[],
): CreatorDirectiveBriefingDirective[] {
  return [
    ...directives
      .filter((directive) => directive.renderMode === "content")
      .sort(compareCreatorDirectivePriorityAndAge),
    ...directives
      .filter(
        (directive): directive is CreatorDirectiveBriefingPrivateDirective =>
          directive.renderMode === "private",
      )
      .sort((left, right) => {
        if (left.privateKind !== right.privateKind) {
          return left.privateKind === "knowledge" ? -1 : 1;
        }
        return compareCreatorDirectivePriorityAndAge(left, right);
      }),
    ...directives
      .filter((directive) => directive.renderMode === "boundary")
      .sort(compareCreatorDirectivePriorityAndAge),
  ];
}

function creatorDirectivePayload(directive: CreatorDirectiveBriefingDirective): {
  kind: "boundary_prompt" | "operational_directive" | "semantic_value" | "canonical_fact";
  payloadText: string | null;
  exactRequired: boolean;
} {
  if (directive.renderMode === "boundary") {
    return {
      kind: "boundary_prompt",
      payloadText: INTERIM_CREATOR_DIRECTIVE_BOUNDARY_PROMPT,
      exactRequired: true,
    };
  }
  if (directive.renderMode === "private" && directive.privateKind === "operation") {
    return {
      kind: "operational_directive",
      payloadText: directive.operationalDirective,
      exactRequired: true,
    };
  }
  if (directive.kind === "response_policy" || directive.kind === "routing_instruction") {
    return {
      kind: "operational_directive",
      payloadText: directive.operationalDirective,
      exactRequired: true,
    };
  }
  if (directive.semanticSlot !== null) {
    return {
      kind: "semantic_value",
      payloadText: directive.semanticValue,
      exactRequired: false,
    };
  }
  return {
    kind: "canonical_fact",
    payloadText: directive.canonicalFact,
    exactRequired: false,
  };
}

function directiveMode(directive: CreatorDirectiveBriefingDirective): string {
  return directive.renderMode !== "private"
    ? directive.renderMode
    : directive.privateKind === "operation"
      ? "private_operation"
      : "private_knowledge";
}

function directiveKind(directive: CreatorDirectiveBriefingDirective): string {
  return directive.renderMode === "boundary" ? "boundary" : directive.kind;
}

function directiveSubjectFields(directive: CreatorDirectiveBriefingDirective): {
  kind: string;
  label: string;
  semanticSlot: string;
} {
  if (
    directive.renderMode === "boundary" ||
    (directive.renderMode === "private" && directive.privateKind === "operation")
  ) {
    return { kind: "none", label: "", semanticSlot: "none" };
  }
  return {
    kind: directive.subjectKind,
    label: directive.subjectLabel,
    semanticSlot: directive.semanticSlot ?? "none",
  };
}

function compactCreatorDirectivePayloadAttributes(directive: CreatorDirectiveBriefingDirective): {
  attributes: string[];
  truncationCount: number;
} {
  const source = creatorDirectivePayload(directive);
  const value = source.payloadText ?? "";
  const payload = headTailPlannerExcerpt(
    value,
    source.exactRequired ? Number.MAX_SAFE_INTEGER : TERMINAL_CREATOR_DIRECTIVE_FACT_EXCERPT_CHARS,
  );
  const status =
    source.payloadText === null ? "missing" : payload.truncated ? "head+tail_excerpt" : "exact";
  return {
    attributes: [
      `payload_kind="${source.kind}"`,
      `payload_status="${status}"`,
      `payload_included_chars="${payload.renderedChars}"`,
      `payload_total_chars="${payload.totalChars}"`,
      `payload="${escapeXmlSingleLineAttribute(payload.text)}"`,
    ],
    truncationCount: Number(payload.truncated),
  };
}

function compactCreatorDirectiveSubjectAttributes(directive: CreatorDirectiveBriefingDirective): {
  attributes: string[];
  truncationCount: number;
} {
  const subject = directiveSubjectFields(directive);
  const label = headTailPlannerExcerpt(
    subject.label,
    TERMINAL_CREATOR_DIRECTIVE_LABEL_EXCERPT_CHARS,
  );
  return {
    attributes: [
      `subject_kind="${escapeXmlAttribute(subject.kind)}"`,
      `subject_label_exact="${!label.truncated}"`,
      `subject_label_included_chars="${label.renderedChars}"`,
      `subject_label_total_chars="${label.totalChars}"`,
      `subject_label="${escapeXmlSingleLineAttribute(label.text)}"`,
      `semantic_slot="${escapeXmlAttribute(subject.semanticSlot)}"`,
    ],
    truncationCount: Number(label.truncated),
  };
}

function compactCreatorDirectiveScopeAttributes(
  directive: CreatorDirectiveBriefingDirective,
): string[] {
  const scope = directive.scope;
  const scopeList = (values: readonly string[] | undefined) =>
    scope === undefined ? "unknown" : joinedAttribute(values);
  const fallbackMentionPolicy =
    directive.renderMode === "boundary" ||
    (directive.renderMode === "private" && directive.privateKind === "operation")
      ? "unknown"
      : directive.mentionPolicy;
  return [
    `scope_status="${scope === undefined ? "not_captured" : "exact"}"`,
    `directive_id="${escapeXmlAttribute(scope?.directiveId ?? "unknown")}"`,
    `created_by_entity_id="${escapeXmlAttribute(scope?.createdByEntityId ?? "unknown")}"`,
    `source_session_id="${escapeXmlAttribute(scope?.sourceSessionId ?? "unknown")}"`,
    `content_scope="${escapeXmlAttribute(scope?.contentScope ?? "unknown")}"`,
    `allowed_entity_ids="${escapeXmlAttribute(scopeList(scope?.allowedEntityIds))}"`,
    `excluded_entity_ids="${escapeXmlAttribute(scopeList(scope?.excludedEntityIds))}"`,
    `subject_may_know="${scope === undefined ? "unknown" : String(scope.subjectMayKnow ?? "null")}"`,
    `mention_policy="${escapeXmlAttribute(scope?.mentionPolicy ?? fallbackMentionPolicy)}"`,
    `denied_audience_behavior="${escapeXmlAttribute(scope?.deniedAudienceBehavior ?? "unknown")}"`,
    `activation_scope="${escapeXmlAttribute(scope?.activationScope ?? "unknown")}"`,
    `activation_allowed_entity_ids="${escapeXmlAttribute(scopeList(scope?.activationAllowedEntityIds))}"`,
    `activation_excluded_entity_ids="${escapeXmlAttribute(scopeList(scope?.activationExcludedEntityIds))}"`,
  ];
}

function renderCompactCreatorDirectiveRow(
  directive: CreatorDirectiveBriefingDirective,
  index: number,
): { row: string; truncationCount: number } {
  const scopeAttributes = compactCreatorDirectiveScopeAttributes(directive);
  const subject = compactCreatorDirectiveSubjectAttributes(directive);
  const payload = compactCreatorDirectivePayloadAttributes(directive);
  return {
    row: [
      `<creator_directive id_alias="cd_${index + 1}"`,
      `mode="${directiveMode(directive)}"`,
      `kind="${escapeXmlAttribute(directiveKind(directive))}"`,
      `priority="${directive.priority}"`,
      `created_at="${iso(directive.createdAt)}"`,
      ...scopeAttributes,
      ...subject.attributes,
      ...payload.attributes,
      "/>",
    ].join(" "),
    truncationCount: subject.truncationCount + payload.truncationCount,
  };
}

function renderCompactCreatorDirectives(
  briefing: DeliberationContext["creatorDirectiveBriefing"],
): { lines: string[]; rowCount: number; truncationCount: number } {
  if (briefing === null || briefing === undefined || briefing.directives.length === 0) {
    return {
      lines: [
        '  <creator_directive_index complete_for_current_audience="true" rows_total_for_current_audience="0"><omitted_count>0</omitted_count></creator_directive_index>',
      ],
      rowCount: 0,
      truncationCount: 0,
    };
  }
  let truncationCount = 0;
  const rows = orderedCreatorDirectives(briefing.directives).map((directive, index) => {
    const rendered = renderCompactCreatorDirectiveRow(directive, index);
    truncationCount += rendered.truncationCount;
    return rendered.row;
  });
  return {
    lines: [
      `  <creator_directive_index complete_for_current_audience="true" rows_total_for_current_audience="${rows.length}" fact_excerpt_budget_chars="${TERMINAL_CREATOR_DIRECTIVE_FACT_EXCERPT_CHARS}">`,
      "    <interpretation>This index is complete for the current audience: it lists every active directive this audience's disclosure policy admits. Directives scoped away from this audience are omitted, so absence here is not evidence one does not exist. Boundary and operational directives are exact. Fact-bearing payloads may be visibly annotated mechanical head+tail excerpts with included and total source-character counts. Every structural disclosure and activation scope field is exact; none is inferred from payload language.</interpretation>",
      ...rows.map((row) => `    ${row}`),
      "    <omitted_count>0</omitted_count>",
      "  </creator_directive_index>",
    ],
    rowCount: rows.length,
    truncationCount,
  };
}

function renderDurableAudience(context: DeliberationContext): RenderedTerminalSection[] {
  const directives = renderCompactCreatorDirectives(context.creatorDirectiveBriefing ?? null);
  return [
    terminalSection(
      "audience_authority_and_directives",
      "terminal_durable_audience",
      [
        `<borg_terminal_audience_durable audience_entity_id="${escapeXmlAttribute(context.audienceEntityId ?? "none")}" self_audience="${context.isSelfAudience === true}">`,
        `  <authority_framing>${escapeXmlText(STABLE_AUTHORITY_FRAMING)}</authority_framing>`,
        ...directives.lines,
        "</borg_terminal_audience_durable>",
      ].join("\n"),
      { rowCount: directives.rowCount, truncationCount: directives.truncationCount },
    ),
  ];
}

function assembledEntityLabel(
  context: DeliberationContext,
  entityId: CommitmentRecord["made_to_entity"],
): string {
  if (entityId === null) return "none";
  const commitmentLabel = context.commitmentEntityLabels?.[entityId];
  if (commitmentLabel !== undefined) return commitmentLabel;
  const participant = [
    ...(context.activeParticipants ?? []),
    ...(context.participantProfiles ?? []),
  ].find((candidate) => candidate.entityId === entityId);
  if (participant?.displayName !== null && participant?.displayName !== undefined) {
    return participant.displayName;
  }
  if (context.creatorContext?.currentSenderEntityId === entityId) {
    return context.creatorContext.currentSenderDisplayName ?? entityId;
  }
  if (context.audienceEntityId === entityId) return context.audience ?? entityId;
  return "unknown";
}

function renderRelativeAgeOverlay(context: DeliberationContext): RenderedTerminalSection {
  const nowMs =
    context.nowMs !== undefined && Number.isFinite(context.nowMs) ? context.nowMs : undefined;
  const rows: string[] = [];
  const commitmentLedgerEntries = context.evidenceLedger?.audienceStanding?.commitmentEntries ?? [];
  const commitmentLedgerById = new Map(commitmentLedgerEntries.map((entry) => [entry.id, entry]));
  for (const commitment of context.applicableCommitments ?? []) {
    const ledgerEntry = commitmentLedgerById.get(`commitment:${commitment.id}`);
    rows.push(
      `<commitment_age id="${escapeXmlAttribute(commitment.id)}" created="${age(commitment.created_at, nowMs)}" updated="${age(commitment.updated_at, nowMs)}" reinforced="${age(commitment.last_reinforced_at, nowMs)}" expires="${age(commitment.expires_at, nowMs)}" expired="${age(commitment.expired_at, nowMs)}" revoked="${age(commitment.revoked_at, nowMs)}" ledger_scope="${escapeXmlAttribute(ledgerEntry?.session_scope ?? "global")}" via_retrieval="${ledgerEntry?.via_retrieval === true}" ledger_state_metadata="${escapeXmlSingleLineAttribute(metadataAttribute(ledgerEntry?.state_metadata))}" made_to_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, commitment.made_to_entity))}" restricted_audience_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, commitment.restricted_audience))}" about_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, commitment.about_entity))}" committed_by_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, commitment.committed_by_entity_id ?? null))}" />`,
    );
  }
  const canonicalLedgerIds = new Set(
    (context.applicableCommitments ?? []).map((commitment) => `commitment:${commitment.id}`),
  );
  const metadataTimestamp = (entry: EvidenceLedgerEntry, key: string): number | null => {
    const value = entry.state_metadata?.[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value !== "string") return null;
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : null;
  };
  for (const entry of commitmentLedgerEntries) {
    if (canonicalLedgerIds.has(entry.id)) continue;
    rows.push(
      `<commitment_age id="${escapeXmlAttribute(entry.id)}" created="${age(metadataTimestamp(entry, "created_at"), nowMs)}" reinforced="${age(metadataTimestamp(entry, "last_reinforced_at"), nowMs)}" ledger_scope="${escapeXmlAttribute(entry.session_scope)}" via_retrieval="${entry.via_retrieval === true}" ledger_state_metadata="${escapeXmlSingleLineAttribute(metadataAttribute(entry.state_metadata))}" made_to_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, typeof entry.state_metadata?.made_to_entity_id === "string" ? (entry.state_metadata.made_to_entity_id as CommitmentRecord["made_to_entity"]) : null))}" committed_by_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, typeof entry.state_metadata?.committed_by_entity_id === "string" ? (entry.state_metadata.committed_by_entity_id as CommitmentRecord["made_to_entity"]) : null))}" />`,
    );
  }
  for (const value of context.selfSnapshot.values) {
    rows.push(
      `<value_age id="${escapeXmlAttribute(value.id)}" record_version="${value.record_version ?? "unknown"}" state="${escapeXmlAttribute(value.state)}" priority="${value.priority}" confidence="${value.confidence}" support_count="${value.support_count}" contradiction_count="${value.contradiction_count}" evidence_episode_ids="${escapeXmlAttribute(joinedAttribute(value.evidence_episode_ids))}" last_affirmed_at="${iso(value.last_affirmed)}" last_tested_at="${iso(value.last_tested_at)}" last_contradicted_at="${iso(value.last_contradicted_at)}" created="${age(value.created_at, nowMs)}" affirmed="${age(value.last_affirmed, nowMs)}" established="${age(value.established_at, nowMs)}" tested="${age(value.last_tested_at, nowMs)}" contradicted="${age(value.last_contradicted_at, nowMs)}" />`,
    );
  }
  for (const trait of context.selfSnapshot.traits) {
    rows.push(
      `<trait_age id="${escapeXmlAttribute(trait.id)}" record_version="${trait.record_version ?? "unknown"}" state="${escapeXmlAttribute(trait.state)}" strength="${trait.strength}" confidence="${trait.confidence}" support_count="${trait.support_count}" contradiction_count="${trait.contradiction_count}" evidence_episode_ids="${escapeXmlAttribute(joinedAttribute(trait.evidence_episode_ids))}" last_reinforced_at="${iso(trait.last_reinforced)}" last_decayed_at="${iso(trait.last_decayed)}" last_tested_at="${iso(trait.last_tested_at)}" last_contradicted_at="${iso(trait.last_contradicted_at)}" reinforced="${age(trait.last_reinforced, nowMs)}" decayed="${age(trait.last_decayed, nowMs)}" established="${age(trait.established_at, nowMs)}" tested="${age(trait.last_tested_at, nowMs)}" contradicted="${age(trait.last_contradicted_at, nowMs)}" />`,
    );
  }
  return terminalSection(
    "relative_age_overlay",
    "terminal_turn_context",
    [
      `<borg_terminal_relative_age_overlay complete="true" rows_total="${rows.length}">`,
      "  <interpretation>Turn-local mutable state, session/retrieval scope, assembled entity labels, and recency labels keyed to durable record ids. These fields supplement, and never replace, the stable identity/provenance and absolute timestamps on durable records.</interpretation>",
      ...rows.map((row) => `  ${row}`),
      "  <omitted_count>0</omitted_count>",
      "</borg_terminal_relative_age_overlay>",
    ].join("\n"),
    { rowCount: rows.length },
  );
}

function renderSenderAuthority(context: DeliberationContext): RenderedTerminalSection {
  const participants = context.activeParticipants ?? [];
  const rows = participants.map(
    (participant) =>
      `<participant entity_id="${escapeXmlAttribute(participant.entityId)}" role="${escapeXmlAttribute(participant.role)}" display_name="${escapeXmlSingleLineAttribute(participant.displayName ?? participant.entityId)}" />`,
  );
  return terminalSection(
    "sender_roster_authority",
    "terminal_turn_context",
    [
      `<borg_terminal_sender_authority audience="${escapeXmlSingleLineAttribute(context.audience ?? "unknown")}" audience_entity_id="${escapeXmlAttribute(context.audienceEntityId ?? "none")}" sender_entity_id="${escapeXmlAttribute(context.senderEntityId ?? "none")}">`,
      ...renderAuthorityContextLines(context, "  "),
      ...rows.map((row) => `  ${row}`),
      "</borg_terminal_sender_authority>",
    ].join("\n"),
    { rowCount: rows.length },
  );
}

function renderSessionSnapshot(context: DeliberationContext): RenderedTerminalSection {
  const snapshot = context.operatorSessionSnapshot ?? null;
  return terminalSection(
    "session_status_snapshot",
    "terminal_turn_context",
    renderSessionStatusSnapshotLines(snapshot, "").join("\n"),
    {
      rowCount: snapshot?.sessions.length ?? 0,
      omissionCount: snapshot?.omitted_count ?? 0,
    },
  );
}

const TRUSTED_TURN_BASE_SECTION_IDS = [
  "borg_participation_policy",
  "borg_procedural_guidance",
  "borg_mechanism_evidence",
  "borg_discourse_control",
  "borg_frame_anomaly_gate",
] as const;

const UNTRUSTED_TURN_BASE_SECTION_IDS = [
  "borg_working_state",
  "borg_affective_trajectory",
  "borg_audience_profile",
  "borg_thread_roster",
  "borg_retrieved_evidence",
  "borg_retrieval_confidence",
  "contradiction_signal",
  "borg_open_questions",
  "borg_pending_corrections",
  "borg_autonomy_trigger",
  "borg_current_period",
  "borg_recent_growth",
  "borg_recent_completed_actions",
] as const;

const TERMINAL_STANDING_INDEX_FIELD_CHARS = 240;
const TERMINAL_STANDING_INDEX_METADATA_CHARS = 320;

type TerminalIndexRows = { rows: string[]; truncationCount: number };

function boundedIndexAttribute(value: string, maxChars: number) {
  return headTailPlannerExcerpt(value, maxChars);
}

function renderCompleteLedgerIndexRows(
  entries: readonly EvidenceLedgerEntry[],
  rowTag: string,
): TerminalIndexRows {
  let truncationCount = 0;
  const rows = entries.map((entry) => {
    const text = boundedIndexAttribute(entry.text ?? "", TERMINAL_STANDING_INDEX_FIELD_CHARS);
    const value = boundedIndexAttribute(entry.value ?? "", TERMINAL_STANDING_INDEX_FIELD_CHARS);
    const metadata = boundedIndexAttribute(
      entry.state_metadata === undefined ? "none" : JSON.stringify(entry.state_metadata),
      TERMINAL_STANDING_INDEX_METADATA_CHARS,
    );
    truncationCount += [text, value, metadata].filter((field) => field.truncated).length;
    return [
      `<${rowTag} id="${escapeXmlAttribute(entry.id)}"`,
      `source_type="${escapeXmlAttribute(entry.source_type)}"`,
      `scope="${escapeXmlAttribute(entry.session_scope)}"`,
      `actor="${escapeXmlAttribute(entry.actor)}"`,
      `trust_rank="${entry.trust_rank}"`,
      `state="${escapeXmlAttribute(entry.state ?? "none")}"`,
      `salience_class="${escapeXmlAttribute(entry.salience_class ?? "none")}"`,
      `taint="${escapeXmlAttribute(entry.taint ?? "none")}"`,
      `persistence_class="${escapeXmlAttribute(entry.persistence_class ?? "unknown")}"`,
      `via_retrieval="${entry.via_retrieval === true}"`,
      `stream_index="${entry.stream_index ?? "none"}"`,
      `citation_type="${escapeXmlAttribute(entry.citation_type ?? "none")}"`,
      `citations="${escapeXmlSingleLineAttribute(joinedAttribute(entry.citations))}"`,
      `disclosure="${escapeXmlSingleLineAttribute(compactDisclosure(evidenceEntryDisclosure(entry)))}"`,
      `text="${escapeXmlSingleLineAttribute(text.text)}"`,
      `value="${escapeXmlSingleLineAttribute(value.text)}"`,
      `state_metadata="${escapeXmlSingleLineAttribute(metadata.text)}" />`,
    ].join(" ");
  });
  return { rows, truncationCount };
}

function renderCompleteRelationalSlotRows(context: DeliberationContext): TerminalIndexRows {
  let truncationCount = 0;
  const rows = (context.relationalSlots ?? []).map((slot) => {
    const key = boundedIndexAttribute(slot.slot_key, TERMINAL_STANDING_INDEX_FIELD_CHARS);
    const value = boundedIndexAttribute(slot.value, TERMINAL_STANDING_INDEX_FIELD_CHARS);
    const alternates = boundedIndexAttribute(
      slot.alternate_values.map((alternate) => alternate.value).join(" | "),
      TERMINAL_STANDING_INDEX_FIELD_CHARS,
    );
    truncationCount += [key, value, alternates].filter((field) => field.truncated).length;
    return [
      `<relational_slot_row id="${escapeXmlAttribute(slot.id)}"`,
      `subject_entity_id="${escapeXmlAttribute(slot.subject_entity_id)}"`,
      `subject_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, slot.subject_entity_id))}"`,
      `state="${escapeXmlAttribute(slot.state)}"`,
      `created_at="${iso(slot.created_at)}"`,
      `updated_at="${iso(slot.updated_at)}"`,
      `updated_age="${escapeXmlAttribute(age(slot.updated_at, context.nowMs))}"`,
      `alternate_count="${slot.alternate_values.length}"`,
      `evidence_stream_entry_ids="${escapeXmlAttribute(joinedAttribute(slot.evidence_stream_entry_ids))}"`,
      `contradicted_by_stream_entry_ids="${escapeXmlAttribute(joinedAttribute(slot.contradicted_by_stream_entry_ids))}"`,
      `disclosure="${escapeXmlSingleLineAttribute(compactDisclosure(relationalSlotMemoryDisclosureLabel(slot)))}"`,
      `slot_key="${escapeXmlSingleLineAttribute(key.text)}"`,
      `value="${escapeXmlSingleLineAttribute(value.text)}"`,
      `alternate_values="${escapeXmlSingleLineAttribute(alternates.text)}" />`,
    ].join(" ");
  });
  return { rows, truncationCount };
}

function renderCompleteStandingMemoryIndexes(
  context: DeliberationContext,
): RenderedTerminalSection {
  const standing = context.evidenceLedger?.audienceStanding;
  const relationalSlots = renderCompleteRelationalSlotRows(context);
  const relationalStanding = renderCompleteLedgerIndexRows(
    standing?.relationalEntries ?? [],
    "relational_standing_row",
  );
  const socialStanding = renderCompleteLedgerIndexRows(
    standing?.observedEventIntrospectionEntries ?? [],
    "social_standing_row",
  );
  const crossSession = renderCompleteLedgerIndexRows(
    standing?.recentLivedExperienceEntries ?? [],
    "cross_session_row",
  );
  const groups = [
    { tag: "relational_slots", rows: relationalSlots.rows },
    { tag: "relational_standing", rows: relationalStanding.rows },
    { tag: "social_standing", rows: socialStanding.rows },
    { tag: "cross_session_entries", rows: crossSession.rows },
  ];
  const rowCount = groups.reduce((sum, group) => sum + group.rows.length, 0);
  return terminalSection(
    "standing_memory_indexes",
    "terminal_turn_context",
    [
      `<borg_terminal_standing_memory_indexes complete="true" rows_total="${rowCount}" standing_cadence_due="${standing?.renderRecentLivedExperience === true}">`,
      "  <interpretation>Complete membership indexes for relational slots, relational standing, social/observed-event memory, and cross-session lived entries. Payload fields are mechanical head+tail excerpts; an excerpt is never a summary. Disclosure labels survive on every row and govern mention, not recall.</interpretation>",
      ...groups.flatMap((group) => [
        `  <${group.tag} complete="true" rows_total="${group.rows.length}">`,
        ...group.rows.map((row) => `    ${row}`),
        "    <omitted_count>0</omitted_count>",
        `  </${group.tag}>`,
      ]),
      "  <omitted_count>0</omitted_count>",
      "</borg_terminal_standing_memory_indexes>",
    ].join("\n"),
    {
      rowCount,
      truncationCount:
        relationalSlots.truncationCount +
        relationalStanding.truncationCount +
        socialStanding.truncationCount +
        crossSession.truncationCount,
    },
  );
}

function plannerSectionToTerminal(
  section: RenderedPlannerSection,
  label = section.label,
): RenderedTerminalSection {
  return terminalSection(label, "terminal_turn_context", section.text, {
    rowCount: section.rowCount,
    truncationCount: section.truncationCount,
    omissionCount: section.omissionCount,
  });
}

const FINALIZER_ADDITIONAL_SECTION_ORDER: Readonly<Record<string, number>> = {
  borg_evidence_ledger: 30,
  borg_additional_retrieval: 40,
  [COMPACT_FINALIZER_VERIFICATION_RETRIEVAL_BLOCK_ID]: 40,
  borg_s2_plan: 50,
};

function orderedUntrustedAdditionalSections(
  sections: readonly PromptSurfaceAdditionalSection[],
): PromptSurfaceAdditionalSection[] {
  const hasCompactVerificationRetrieval = sections.some(
    (section) => section.blockId === COMPACT_FINALIZER_VERIFICATION_RETRIEVAL_BLOCK_ID,
  );
  return sections
    .filter(
      (section) =>
        !hasCompactVerificationRetrieval || section.blockId !== "borg_additional_retrieval",
    )
    .map((section, inputIndex) => ({ section, inputIndex }))
    .sort((left, right) => {
      const leftOrder = FINALIZER_ADDITIONAL_SECTION_ORDER[left.section.blockId] ?? 45;
      const rightOrder = FINALIZER_ADDITIONAL_SECTION_ORDER[right.section.blockId] ?? 45;
      return leftOrder - rightOrder || left.inputIndex - right.inputIndex;
    })
    .map(({ section }) => section);
}

function renderTurnContext(
  input: BuildCompactFinalizerSystemPromptInput,
  baseSections: BaseSystemPromptSections,
): RenderedTerminalSection[] {
  const nowMs =
    input.baseSystemPromptOptions.nowMs !== undefined &&
    Number.isFinite(input.baseSystemPromptOptions.nowMs)
      ? input.baseSystemPromptOptions.nowMs
      : input.context.nowMs;
  const currentTime = renderCurrentTimeSection(nowMs, input.context.currentTimeContext ?? null);
  const renderBaseSections = (ids: readonly string[]) =>
    ids.flatMap((id) => {
      const rendered = renderPromptSection(baseSections.promptSectionsById.get(id));
      return rendered === null ? [] : [terminalSection(id, "terminal_turn_context", rendered)];
    });
  const trustedBaseSections = renderBaseSections(TRUSTED_TURN_BASE_SECTION_IDS);
  const untrustedBaseSections = renderBaseSections(UNTRUSTED_TURN_BASE_SECTION_IDS);
  const autonomous = buildAutonomousOutboundAuthorizationSection(
    input.context.autonomousOutbound ?? null,
    input.context.turnOrigin,
    input.context.autonomousFinalizerToolMenu,
  );
  const autonomousOutboundAction = renderAutonomousOutboundActionAvailabilitySection(
    input.context.autonomousOutbound ?? null,
    input.context.autonomousFinalizerToolMenu,
    input.context.turnOrigin,
  );
  const trustedAdditional = (input.additionalPromptSections ?? [])
    .filter((entry) => entry.blockId === "borg_session_reentry_continuity")
    .map((entry) => terminalSection(entry.blockId, "terminal_turn_context", entry.text));
  const untrustedAdditional = orderedUntrustedAdditionalSections(
    (input.additionalPromptSections ?? []).filter(
      (entry) => entry.blockId !== "borg_session_reentry_continuity",
    ),
  ).map((entry) => terminalSection(entry.blockId, "terminal_turn_context", entry.text));

  return [
    ...(currentTime === null
      ? []
      : [
          terminalSection(
            "current_time",
            "terminal_turn_context",
            tagged("borg_current_time", currentTime)!,
          ),
        ]),
    renderRelativeAgeOverlay({ ...input.context, nowMs }),
    renderSenderAuthority(input.context),
    renderSessionSnapshot(input.context),
    ...((input.context.participantRoster?.participants.length ??
      input.context.activeParticipants?.length ??
      0) > 1
      ? [
          terminalSection(
            "group_chat_sender_scoping_reminder",
            "terminal_turn_context",
            GROUP_CHAT_SENDER_SCOPING_REMINDER,
          ),
        ]
      : []),
    ...trustedBaseSections,
    ...(autonomous === null
      ? []
      : [terminalSection("borg_autonomous_reflection", "terminal_turn_context", autonomous)]),
    ...(autonomousOutboundAction === null
      ? []
      : [
          terminalSection(
            "borg_directed_outbound_instruction",
            "terminal_turn_context",
            autonomousOutboundAction,
          ),
        ]),
    ...trustedAdditional,
    terminalSection("turn_data_boundary", "terminal_turn_context", UNTRUSTED_DATA_PREAMBLE),
    ...untrustedBaseSections,
    renderCompleteStandingMemoryIndexes({ ...input.context, nowMs }),
    plannerSectionToTerminal(renderGoalDigest({ ...input.context, nowMs }), "goal_index"),
    plannerSectionToTerminal(
      renderLivedExperienceDigest({ ...input.context, nowMs }),
      "lived_experience",
    ),
    ...untrustedAdditional,
  ];
}

function joinSections(sections: readonly RenderedTerminalSection[]): string {
  return sections.map((entry) => entry.text).join("\n\n");
}

function traceSummary(
  input: BuildCompactFinalizerSystemPromptInput,
  sections: readonly RenderedTerminalSection[],
): FinalizerContextTraceSummary {
  const summaries = Object.fromEntries(
    sections.map((entry) => [
      entry.label,
      {
        chars: entry.text.length,
        estimatedTokens: estimatePromptTokens(entry.text),
        rowCount: entry.rowCount,
        truncationCount: entry.truncationCount,
        omissionCount: entry.omissionCount,
        cacheTier: entry.cacheTier,
      },
    ]),
  );
  const tierText = (tier: FinalizerCacheTier) =>
    joinSections(sections.filter((entry) => entry.cacheTier === tier));
  const blocks = {
    terminal_static_head: tierText("terminal_static_head"),
    terminal_durable_global: tierText("terminal_durable_global"),
    terminal_durable_audience: tierText("terminal_durable_audience"),
    terminal_turn_context: tierText("terminal_turn_context"),
  };
  const totalText = Object.values(blocks).join("\n\n");
  return {
    variant: "compact",
    path: input.path,
    sections: summaries,
    blocks: {
      terminal_static_head: {
        chars: blocks.terminal_static_head.length,
        estimatedTokens: estimatePromptTokens(blocks.terminal_static_head),
        ttl: "1h",
      },
      terminal_durable_global: {
        chars: blocks.terminal_durable_global.length,
        estimatedTokens: estimatePromptTokens(blocks.terminal_durable_global),
        ttl: "1h",
      },
      terminal_durable_audience: {
        chars: blocks.terminal_durable_audience.length,
        estimatedTokens: estimatePromptTokens(blocks.terminal_durable_audience),
        ttl: "1h",
      },
      terminal_turn_context: {
        chars: blocks.terminal_turn_context.length,
        estimatedTokens: estimatePromptTokens(blocks.terminal_turn_context),
        ttl: "5m",
      },
    },
    totalChars: totalText.length,
    totalEstimatedTokens: estimatePromptTokens(totalText),
    rowCount: sections.reduce((sum, entry) => sum + entry.rowCount, 0),
    truncationCount: sections.reduce((sum, entry) => sum + entry.truncationCount, 0),
    omissionCount: sections.reduce((sum, entry) => sum + entry.omissionCount, 0),
  };
}

export function buildCompactFinalizerSystemPrompt(
  input: BuildCompactFinalizerSystemPromptInput,
): CompactFinalizerSystemPrompt {
  // Partition stability is intentional and local to this site:
  // - static_head: protocol/framing plus deployment-stable host capabilities only;
  // - durable_global: creator identity, the canonical commitment union except turn-resolved
  //   scope/labels, and census-stable value/trait identity, text, timestamps, and provenance;
  // - durable_audience: only the already-resolved audience directive briefing and stable frame;
  // - turn_context: every clock, sender, roster, working-state, retrieval, ledger, and plan input.
  // Mutable self confidence/strength/priority/state/counters and reinforcement/test timestamps,
  // plus retrieval/session scope, entity labels, and every relative age stay in turn_context.
  // Moving a field between these tiers requires a stability census; a cache hit must never
  // substitute stale sender/session state merely because the text looked durable in one trace.
  const baseSections = buildBaseSystemPromptSections(input.context, {
    ...input.baseSystemPromptOptions,
    omitStandingAssembly: true,
  });
  const staticSections = [
    terminalSection(
      "terminal_contract",
      "terminal_static_head",
      // The production staticHead is the registry-rendered finalizer static
      // surface and already ends with trusted framing + host capabilities.
      // Re-rendering either here duplicates authoritative instructions.
      [input.staticHead, CURRENT_USER_MESSAGE_REMINDER, TERMINAL_PASS_CONTRACT].join("\n\n"),
    ),
  ];
  const durableGlobalSections = renderDurableGlobal(input.context);
  const durableAudienceSections = renderDurableAudience(input.context);
  const turnSections = renderTurnContext(input, baseSections);
  const allSections = [
    ...staticSections,
    ...durableGlobalSections,
    ...durableAudienceSections,
    ...turnSections,
  ];

  return {
    system: [
      {
        type: "text",
        text: joinSections(staticSections),
        cache_control: FINALIZER_COMPACT_STATIC_CACHE_CONTROL,
      },
      {
        type: "text",
        text: joinSections(durableGlobalSections),
        cache_control: FINALIZER_COMPACT_DURABLE_GLOBAL_CACHE_CONTROL,
      },
      {
        type: "text",
        text: joinSections(durableAudienceSections),
        cache_control: FINALIZER_COMPACT_DURABLE_AUDIENCE_CACHE_CONTROL,
      },
      {
        type: "text",
        text: joinSections(turnSections),
        cache_control: FINALIZER_COMPACT_TURN_CACHE_CONTROL,
      },
    ],
    traceSummary: traceSummary(input, allSections),
  };
}
