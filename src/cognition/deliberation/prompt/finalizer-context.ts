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
import { appendMemoryDisclosureState } from "../../evidence-ledger/entry-metadata.js";
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
  toolAvailability: FinalizerToolAvailabilityState;
  path: "system_1" | "system_2";
  additionalPromptSections?: readonly PromptSurfaceAdditionalSection[];
};

export type FinalizerToolAvailabilityState = {
  turnOrigin: NonNullable<DeliberationContext["turnOrigin"]>;
  participationPolicy: NonNullable<DeliberationContext["participationPolicy"]>;
  enabledTerminalEmissions: readonly string[];
  outboundPostAvailable: boolean;
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

// The omitted_count clause below describes more than one kind of field with one
// sentence. At a bounded expansion or digest the count is computed from the draw
// (top_global_candidates_expanded, the lived-experience digest, the per-section
// omission map), so a nonzero print is a live measurement. At most complete indexes
// it is still a string literal in the renderer -- the index is complete by
// construction, so the zero restates that construction rather than measuring it, and
// nothing in the rendered surface distinguishes such a zero from a measured one.
// borg_terminal_values_traits is the exception: its complete attribute and its
// omitted_count are both derived from stored row counts each store reports by its own
// statement, so there the zero is a measurement and a filter appearing in the draw
// would flip the attribute rather than pass unnoticed. Everywhere else, keep in mind
// before treating a zero here as evidence about the underlying store that a constant
// true stays true for whatever reason once made it true, and says nothing when that
// reason stops holding.
const TERMINAL_PASS_CONTRACT = [
  "<borg_terminal_pass_contract>",
  "This is my terminal response pass. I make the final emission decision from the complete request surface below; any system-2 plan is advisory, not authority.",
  "Durable records appear before turn-local overlays. I join an overlay to its durable record only by the explicit record id. Scope and disclosure fields describe use and mention boundaries; they never gate what I recall.",
  "Cacheable block 1 contains only durable commitment, value, and trait record fields plus the durable ledger join identity. The current ledger projection -- actor, trust rank, salience, taint, scope, persistence, retrieval state, stream and citation data, divergent state/value/text, and resolved disclosure -- lives in turn-local block 3.",
  "The commitment membership denominator is commitment_rows_total in the turn-local relative-age overlay. It stays out of cacheable block 1 so that block's header remains byte-stable across turns.",
  "Where an exact absolute timestamp is present, I derive its relative age by subtracting it from the borg_current_time current_time_ms value; a duplicate relative-age label is intentionally omitted.",
  'A completeness claim rides on a complete="true" attribute beside omitted_count="0"; where a container is drawn narrower than the record it names its draw in an attribute instead. An element name is a label and never a claim of coverage, whatever word it contains. A complete="unmeasured" attribute with no omitted_count is not a weaker yes: it means the container checked nothing, so I read it as making no coverage claim in either direction.',
  "Any bounded expansion or digest reports its omissions explicitly.",
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

export function renderFinalizerToolAvailability(
  availability: FinalizerToolAvailabilityState,
): string {
  return `<borg_finalizer_tool_availability turn_origin="${escapeXmlAttribute(availability.turnOrigin)}" participation_policy="${escapeXmlAttribute(availability.participationPolicy)}" outbound_post="${availability.outboundPostAvailable ? "available" : "unavailable"}" enabled_terminal_emissions="${escapeXmlAttribute(joinedAttribute(availability.enabledTerminalEmissions))}" />`;
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
    `last_reinforced_at="${iso(commitment.last_reinforced_at)}"`,
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

function renderTurnLedgerProjectionFields(
  ledgerEntry: EvidenceLedgerEntry | undefined,
  resolvedDisclosure: MemoryDisclosureLabel,
): string[] {
  return [
    `ledger_actor="${escapeXmlAttribute(ledgerEntry?.actor ?? "memory")}"`,
    `ledger_trust_rank="${ledgerEntry?.trust_rank ?? "unknown"}"`,
    `ledger_salience_class="${escapeXmlAttribute(ledgerEntry?.salience_class ?? "none")}"`,
    `ledger_taint="${escapeXmlAttribute(ledgerEntry?.taint ?? "none")}"`,
    `stream_index="${ledgerEntry?.stream_index ?? "none"}"`,
    `citation_type="${escapeXmlAttribute(ledgerEntry?.citation_type ?? "none")}"`,
    `citations="${escapeXmlSingleLineAttribute(joinedAttribute(ledgerEntry?.citations))}"`,
    `resolved_disclosure="${escapeXmlAttribute(compactDisclosure(resolvedDisclosure))}"`,
  ];
}

function renderCanonicalCommitmentLedgerDifferenceFields(
  commitment: CommitmentRecord,
  ledgerEntry: EvidenceLedgerEntry | undefined,
): string[] {
  const status = commitmentStatus(commitment);
  const duplicateLedgerStates = new Set([
    status,
    appendMemoryDisclosureState({
      state: status,
      disclosureLabel: ledgerEntry === undefined ? undefined : evidenceEntryDisclosure(ledgerEntry),
    }),
  ]);
  const fields: string[] = [];
  if (ledgerEntry?.state !== undefined && !duplicateLedgerStates.has(ledgerEntry.state)) {
    fields.push(`ledger_state="${escapeXmlAttribute(ledgerEntry.state)}"`);
  }
  if (ledgerEntry !== undefined && ledgerEntry.value !== commitment.directive_family) {
    fields.push(`ledger_value="${escapeXmlSingleLineAttribute(ledgerEntry.value ?? "missing")}"`);
  }
  if (ledgerEntry !== undefined && ledgerEntry.text !== commitment.directive) {
    fields.push(`ledger_text="${escapeXmlSingleLineAttribute(ledgerEntry.text ?? "missing")}"`);
  }
  return fields;
}

function renderCommitmentRecord(
  commitment: CommitmentRecord,
  ordinal: number,
): { row: string; truncationCount: number } {
  const disclosure = commitmentMemoryDisclosureLabel(commitment);
  const critical = effectiveCommitmentEnforcementClass(commitment) === "critical";
  const directive = critical
    ? headTailPlannerExcerpt(commitment.directive, Number.MAX_SAFE_INTEGER)
    : headTailPlannerExcerpt(commitment.directive, TERMINAL_ADVISORY_COMMITMENT_EXCERPT_CHARS);
  return {
    row: [
      ...renderCommitmentIdentityFields(commitment, ordinal),
      ...renderCommitmentTimelineAndScopeFields(commitment, disclosure),
      `ledger_ref="${escapeXmlAttribute(`commitment:${commitment.id}`)}"`,
      'ledger_source_type="commitment"',
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
  return {
    row: [
      `<commitment id="${escapeXmlAttribute(entry.id)}" canonical_record="false"`,
      `ledger_ref="${escapeXmlAttribute(entry.id)}"`,
      `ledger_source_type="${escapeXmlAttribute(entry.source_type)}"`,
      "/>",
    ].join(" "),
    truncationCount: 0,
  };
}

// applicableCommitments arrives from recallActiveCommitmentsForCognition, whose
// predicate (CommitmentRepository.isActiveCommitment) excludes retirement marks but permits
// a future expires_at. Retirement timestamps are therefore absent by construction, while the
// exact optional future expires_at moves to the ID-keyed turn overlay. Canonical block-1 rows are
// rendered without accepting an EvidenceLedgerEntry at all: only their deterministic ledger_ref
// and fixed ledger_source_type join identity remain beside durable commitment fields. This block
// changes only when membership or durable commitment content changes.
//
// The tag used to carry rows_total, canonical_rows, and ledger_only_rows; fcd95cb6 removed
// them from this early cacheable header. The equivalent commitment-prefixed counts now live
// on the turn-local overlay, leaving this block's prefix unchanged when membership changes.
//
// The invariant those attributes documented still holds and is worth keeping written down:
// evidenceLedger.audienceStanding.commitmentEntries is a straight 1:1 map of the very
// applicableCommitments this render already walks (buildCommitmentEntries in
// evidence-ledger/audience-standing.ts), so every ledger entry matches a canonical row
// by id and ledgerOnlyRendered is empty by construction. The turn-local ledger-only count is a
// divergence check on those inputs, not a second population's contribution: it says nothing
// about what an audience-scoped standing ledger would add because nothing in this path can add it.
function renderCommitments(context: DeliberationContext): RenderedTerminalSection {
  const commitments = context.applicableCommitments ?? [];
  const ledgerEntries = context.evidenceLedger?.audienceStanding?.commitmentEntries ?? [];
  const canonicalLedgerIds = new Set(
    commitments.map((commitment) => `commitment:${commitment.id}`),
  );
  const canonicalRendered = commitments.map((commitment, index) =>
    renderCommitmentRecord(commitment, index + 1),
  );
  const ledgerOnlyRendered = ledgerEntries
    .filter((entry) => !canonicalLedgerIds.has(entry.id))
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
      `<borg_terminal_commitments complete="true" advisory_excerpt_budget_chars="${TERMINAL_ADVISORY_COMMITMENT_EXCERPT_CHARS}">`,
      "  <interpretation>One row per commitment: canonical records first, then an identity-only pointer for any standing-ledger record that matched none of them by id. Canonical rows contain only durable commitment-record fields plus the durable ledger join identity; every turn-derived field projected from the current ledger is in the block-3 overlay with the same id. Critical directives are exact. A long advisory directive is a visibly annotated mechanical head+tail cut carrying both included and total source-character counts, never a clean-looking summary. Durable entity scope and disclosure are exact provenance and handling constraints, never audience-dependent recall selection.</interpretation>",
      '  <field_legend>Canonical rows omit canonical_record; only a ledger-only fallback pointer carries canonical_record=false. ledger_ref is the durable join key and ledger_source_type is durable source identity; they are the only ledger-named attributes in this block. The membership denominator is commitment_rows_total on the turn-local borg_terminal_relative_age_overlay header; commitment_canonical_rows and commitment_ledger_only_rows partition it exactly. Those counts live in turn block 3 rather than this cacheable block 1 so this header stays byte-stable across turns. Canonical disclosure comes only from the durable commitment record; the fail-closed disclosure resolved with the standing-ledger projection is resolved_disclosure in the overlay. ledger_actor, ledger_trust_rank, ledger_salience_class, ledger_taint, ledger_state, ledger_value, ledger_text, ledger_scope, persistence_class, via_retrieval, stream_index, citation_type, and citations are likewise turn-local overlay fields. A ledger_value or ledger_text exactly equal to family or directive is omitted there; a divergent projection retains its exact value, and a present projection with no value or text prints "missing" explicitly. ledger_state duplicates status plus disclosure and is omitted there; a structurally divergent state is retained exactly. Active canonical rows omit expired_at and revoked_at because their absence follows membership; a future expires_at, when present, is exact in the ID-keyed overlay, as is updated_at. Relative ages follow the terminal pass contract. directive_exact, directive_excerpt_shape, directive_included_chars, and directive_total_chars state whether a durable canonical directive is complete and, when cut, exactly how much source text is present. advisory_excerpt_budget_chars is the whole width a cut advisory directive is rendered into, including its head+tail marker. directive_exact reports elision only, not byte-fidelity of the XML-encoded attribute. On a critical row directive_exact is true by construction, directive_excerpt_shape is full, and directive_included_chars equals directive_total_chars. A search over this block reaches only its rendered characters: on a cut row the elided middle is present here in no form, so a string not found in this block may still be in the stored directive, and the honest form of any such negative names the width it did not reach. That width is the per-row difference between directive_total_chars and directive_included_chars summed over the rows above; no total for it is printed here, and complete="true" is a claim about membership only, never about how much of each row is present.</field_legend>',
      ...rows.map((row) => `  ${row}`),
      "  <omitted_count>0</omitted_count>",
      "</borg_terminal_commitments>",
    ].join("\n"),
    { rowCount: rows.length, truncationCount },
  );
}

function renderDurableSelf(context: DeliberationContext): RenderedTerminalSection {
  const disclosure = compactDisclosure(selfPrivateMemoryDisclosureLabel());
  const valueRows = [...context.selfSnapshot.values]
    .sort((left, right) => left.created_at - right.created_at || left.id.localeCompare(right.id))
    .map(
      (value) =>
        `<value id="${escapeXmlAttribute(value.id)}" created_at="${iso(value.created_at)}" established_at="${iso(value.established_at)}" disclosure="${escapeXmlAttribute(disclosure)}" provenance="${escapeXmlSingleLineAttribute(summarizeProvenanceForPrompt(value.provenance, Number.MAX_SAFE_INTEGER))}" label="${escapeXmlSingleLineAttribute(value.label)}" description="${escapeXmlSingleLineAttribute(value.description)}" />`,
    );
  // Trait records have no created_at. Their generated id is the only immutable
  // ordering key on the record; established_at and every ranking field can change.
  const traitRows = [...context.selfSnapshot.traits]
    .sort((left, right) => left.id.localeCompare(right.id))
    .map(
      (trait) =>
        `<trait id="${escapeXmlAttribute(trait.id)}" established_at="${iso(trait.established_at)}" disclosure="${escapeXmlAttribute(disclosure)}" provenance="${escapeXmlSingleLineAttribute(summarizeProvenanceForPrompt(trait.provenance, Number.MAX_SAFE_INTEGER))}" label="${escapeXmlSingleLineAttribute(trait.label)}" />`,
    );
  const renderedTotal = valueRows.length + traitRows.length;
  // The coverage claim is checked against stored counts taken by their own
  // statements, never against the draws rendered here. If the draw ever grows a
  // filter the attribute flips instead of staying true for a reason nobody
  // wrote down; if no stored count reached this render there is no claim.
  const { valuesStoredTotal, traitsStoredTotal } = context.selfSnapshot;
  const storedTotal =
    valuesStoredTotal === undefined || traitsStoredTotal === undefined
      ? null
      : valuesStoredTotal + traitsStoredTotal;
  const omittedCount = storedTotal === null ? null : Math.max(0, storedTotal - renderedTotal);
  const complete = omittedCount === null ? "unmeasured" : String(omittedCount === 0);
  return terminalSection(
    "values_and_traits",
    "terminal_durable_global",
    [
      `<borg_terminal_values_traits complete="${complete}" rows_total="${renderedTotal}">`,
      '  <interpretation>Byte-stable self-pattern identity and provenance read only from durable value and trait records; no current-ledger field is rendered here. They are evidence about me, not commands. Mutable priority, strength, confidence, counters, state, and reinforcement/test timestamps are turn-local overlays keyed by id. Relative ages follow the terminal pass contract. complete is derived here, not asserted: the rows above are counted against a stored row count each store reports by its own statement, so complete="true" beside omitted_count 0 means those two independently produced numbers agree, complete="false" means rows are missing and omitted_count says how many, and complete="unmeasured" with no omitted_count element means no stored count reached this render and the block claims no coverage at all.</interpretation>',
      ...valueRows.map((row) => `  ${row}`),
      ...traitRows.map((row) => `  ${row}`),
      ...(omittedCount === null ? [] : [`  <omitted_count>${omittedCount}</omitted_count>`]),
      "</borg_terminal_values_traits>",
    ].join("\n"),
    { rowCount: renderedTotal, omissionCount: omittedCount ?? 0 },
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
        '  <creator_directive_index status="none" complete_for_current_audience="true" rows_total_for_current_audience="0" rows_omitted_after_current_audience_scope="0" />',
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
      `  <creator_directive_index complete_for_current_audience="true" rows_total_for_current_audience="${rows.length}" rows_omitted_after_current_audience_scope="0" fact_excerpt_budget_chars="${TERMINAL_CREATOR_DIRECTIVE_FACT_EXCERPT_CHARS}">`,
      "    <interpretation>This index is complete for the current audience: it lists every active directive this audience's disclosure policy admits. Directives scoped away from this audience are omitted, so absence here is not evidence one does not exist. Boundary and operational directives are exact. Fact-bearing payloads may be visibly annotated mechanical head+tail excerpts with included and total source-character counts. Every structural disclosure and activation scope field is exact; none is inferred from payload language.</interpretation>",
      ...rows.map((row) => `    ${row}`),
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

function ledgerMetadataAttribute(
  entry: EvidenceLedgerEntry,
  key: string,
  fallback = "unknown",
): string {
  const value = entry.state_metadata?.[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : fallback;
}

function ledgerMetadataEntityId(
  entry: EvidenceLedgerEntry,
  key: string,
): CommitmentRecord["made_to_entity"] {
  const value = entry.state_metadata?.[key];
  return typeof value === "string" ? (value as CommitmentRecord["made_to_entity"]) : null;
}

function renderCanonicalCommitmentOverlayRow(
  context: DeliberationContext,
  commitment: CommitmentRecord,
  ledgerEntry: EvidenceLedgerEntry | undefined,
): string {
  const expiresAt =
    commitment.expires_at === null || commitment.expires_at === undefined
      ? []
      : [`expires_at="${iso(commitment.expires_at)}"`];
  return [
    `<commitment_age id="${escapeXmlAttribute(commitment.id)}"`,
    `updated_at="${iso(commitment.updated_at)}"`,
    ...expiresAt,
    `ledger_scope="${escapeXmlAttribute(ledgerEntry?.session_scope ?? "global")}"`,
    `persistence_class="${escapeXmlAttribute(ledgerEntry?.persistence_class ?? "unknown")}"`,
    `via_retrieval="${ledgerEntry?.via_retrieval === true}"`,
    ...renderTurnLedgerProjectionFields(
      ledgerEntry,
      combinedCommitmentDisclosure(commitment, ledgerEntry),
    ),
    ...renderCanonicalCommitmentLedgerDifferenceFields(commitment, ledgerEntry),
    `made_to_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, commitment.made_to_entity))}"`,
    `restricted_audience_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, commitment.restricted_audience))}"`,
    `about_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, commitment.about_entity))}"`,
    `committed_by_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, commitment.committed_by_entity_id ?? null))}"`,
    "/>",
  ].join(" ");
}

function renderLedgerOnlyCommitmentOverlayRow(
  context: DeliberationContext,
  entry: EvidenceLedgerEntry,
): { row: string; truncationCount: number } {
  const directive = entry.text ?? "";
  const critical = ledgerMetadataAttribute(entry, "commitment_enforcement_class") === "critical";
  const excerpt = critical
    ? headTailPlannerExcerpt(directive, Number.MAX_SAFE_INTEGER)
    : headTailPlannerExcerpt(directive, TERMINAL_ADVISORY_COMMITMENT_EXCERPT_CHARS);
  return {
    row: [
      `<commitment_age id="${escapeXmlAttribute(entry.id)}" canonical_record="false"`,
      `status="${escapeXmlAttribute(entry.state ?? "unknown")}"`,
      `enforcement_class="${escapeXmlAttribute(ledgerMetadataAttribute(entry, "commitment_enforcement_class"))}"`,
      `critical_domain="${escapeXmlAttribute(ledgerMetadataAttribute(entry, "commitment_critical_domain", "none"))}"`,
      `kind="${escapeXmlAttribute(ledgerMetadataAttribute(entry, "commitment_kind"))}"`,
      `type="${escapeXmlAttribute(ledgerMetadataAttribute(entry, "commitment_type"))}"`,
      `family="${escapeXmlSingleLineAttribute(entry.value ?? "unknown")}"`,
      `created_at="${escapeXmlSingleLineAttribute(ledgerMetadataAttribute(entry, "created_at"))}"`,
      `last_reinforced_at="${escapeXmlSingleLineAttribute(ledgerMetadataAttribute(entry, "last_reinforced_at"))}"`,
      `made_to_entity_id="${escapeXmlAttribute(ledgerMetadataAttribute(entry, "made_to_entity_id", "none"))}"`,
      `restricted_audience_id="${escapeXmlAttribute(ledgerMetadataAttribute(entry, "restricted_audience_id", "none"))}"`,
      `about_entity_id="${escapeXmlAttribute(ledgerMetadataAttribute(entry, "about_entity_id", "none"))}"`,
      `committed_by_entity_id="${escapeXmlAttribute(ledgerMetadataAttribute(entry, "committed_by_entity_id", "none"))}"`,
      `ledger_scope="${escapeXmlAttribute(entry.session_scope)}"`,
      `persistence_class="${escapeXmlAttribute(entry.persistence_class ?? "unknown")}"`,
      `via_retrieval="${entry.via_retrieval === true}"`,
      ...renderTurnLedgerProjectionFields(entry, evidenceEntryDisclosure(entry)),
      `made_to_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, ledgerMetadataEntityId(entry, "made_to_entity_id")))}"`,
      `restricted_audience_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, ledgerMetadataEntityId(entry, "restricted_audience_id")))}"`,
      `about_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, ledgerMetadataEntityId(entry, "about_entity_id")))}"`,
      `committed_by_entity_label="${escapeXmlSingleLineAttribute(assembledEntityLabel(context, ledgerMetadataEntityId(entry, "committed_by_entity_id")))}"`,
      `directive_exact="${!excerpt.truncated}"`,
      `directive_excerpt_shape="${excerpt.truncated ? "head+tail" : "full"}"`,
      `directive_included_chars="${excerpt.renderedChars}"`,
      `directive_total_chars="${excerpt.totalChars}"`,
      `directive="${escapeXmlSingleLineAttribute(excerpt.text)}" />`,
    ].join(" "),
    truncationCount: excerpt.truncated ? 1 : 0,
  };
}

function renderRelativeAgeOverlay(context: DeliberationContext): RenderedTerminalSection {
  const rows: string[] = [];
  let truncationCount = 0;
  const commitments = context.applicableCommitments ?? [];
  const commitmentLedgerEntries = context.evidenceLedger?.audienceStanding?.commitmentEntries ?? [];
  const commitmentLedgerById = new Map(commitmentLedgerEntries.map((entry) => [entry.id, entry]));
  for (const commitment of commitments) {
    const ledgerEntry = commitmentLedgerById.get(`commitment:${commitment.id}`);
    rows.push(renderCanonicalCommitmentOverlayRow(context, commitment, ledgerEntry));
  }
  const canonicalLedgerIds = new Set(
    commitments.map((commitment) => `commitment:${commitment.id}`),
  );
  let ledgerOnlyCommitmentRows = 0;
  for (const entry of commitmentLedgerEntries) {
    if (canonicalLedgerIds.has(entry.id)) continue;
    ledgerOnlyCommitmentRows += 1;
    const rendered = renderLedgerOnlyCommitmentOverlayRow(context, entry);
    rows.push(rendered.row);
    truncationCount += rendered.truncationCount;
  }
  for (const value of context.selfSnapshot.values) {
    rows.push(
      `<value_age id="${escapeXmlAttribute(value.id)}" record_version="${value.record_version ?? "unknown"}" state="${escapeXmlAttribute(value.state)}" priority="${value.priority}" confidence="${value.confidence}" support_count="${value.support_count}" contradiction_count="${value.contradiction_count}" evidence_episode_ids="${escapeXmlAttribute(joinedAttribute(value.evidence_episode_ids))}" last_affirmed_at="${iso(value.last_affirmed)}" last_tested_at="${iso(value.last_tested_at)}" last_contradicted_at="${iso(value.last_contradicted_at)}" />`,
    );
  }
  for (const trait of context.selfSnapshot.traits) {
    rows.push(
      `<trait_age id="${escapeXmlAttribute(trait.id)}" record_version="${trait.record_version ?? "unknown"}" state="${escapeXmlAttribute(trait.state)}" strength="${trait.strength}" confidence="${trait.confidence}" support_count="${trait.support_count}" contradiction_count="${trait.contradiction_count}" evidence_episode_ids="${escapeXmlAttribute(joinedAttribute(trait.evidence_episode_ids))}" last_reinforced_at="${iso(trait.last_reinforced)}" last_decayed_at="${iso(trait.last_decayed)}" last_tested_at="${iso(trait.last_tested_at)}" last_contradicted_at="${iso(trait.last_contradicted_at)}" />`,
    );
  }
  return terminalSection(
    "relative_age_overlay",
    "terminal_turn_context",
    [
      `<borg_terminal_relative_age_overlay complete="true" rows_total="${rows.length}" commitment_rows_total="${commitments.length + ledgerOnlyCommitmentRows}" commitment_canonical_rows="${commitments.length}" commitment_ledger_only_rows="${ledgerOnlyCommitmentRows}">`,
      "  <interpretation>Turn-local mutable state, the turn-derived portion of the current-ledger projection, assembled entity labels, and exact mutable timestamps keyed to durable record ids. For commitments this includes ledger actor, trust rank, salience, taint, scope, persistence, retrieval state, stream and citation data, divergent state/value/text, and fail-closed resolved_disclosure. rows_total counts every overlay row. commitment_rows_total is the complete commitment membership denominator for cacheable block 1; commitment_canonical_rows and commitment_ledger_only_rows partition it exactly. Commitment updated_at lives here; optional expires_at appears only when a scheduled expiry exists, and its absence means no scheduled expiry. A canonical_record=false row carries the ledger-only fallback projection whose block-1 row is only its durable join pointer. Relative ages follow the terminal pass contract.</interpretation>",
      ...rows.map((row) => `  ${row}`),
      "  <omitted_count>0</omitted_count>",
      "</borg_terminal_relative_age_overlay>",
    ].join("\n"),
    { rowCount: rows.length, truncationCount },
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

export const CROSS_SESSION_ENTRIES_DRAW_SCOPE = "other_sessions_recent_window";

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
  // Each group states the predicate of its OWN draw, because the groups do not share
  // one -- and because a draw's scope is not recoverable from the rows it produced.
  // Rows carrying foreign origin_audience labels are equally consistent with a global
  // draw and with a scoped draw that kept foreign provenance: origin_audience records
  // where a memory came from, which is a different axis from which assembly selected
  // it. So the predicate is named here rather than left to be inferred from contents.
  //
  // relational_slots (context.relationalSlots) and relational_standing
  // (audienceStanding.relationalEntries) both list per active participant, filtering on
  // subject_entity_id, and fall back to an unfiltered list when the roster is empty --
  // hence a computed value rather than a fixed one. social_standing is a global
  // observed-event draw (listRecentGlobal + listRecurringGlobal); current participants
  // add a by-speaker lane and a score boost, which widen and rank but never filter.
  // cross_session_entries is the self's own cross-session activity in a time window;
  // the current audience enters it only as a label on the return-silence row. Neither
  // of the last two is audience-scoped, and saying they were was wrong in the direction
  // that understates what the entity is holding.
  //
  // But "global" as this block defines it -- no filter on audience, participant, or
  // session -- was false for cross_session_entries in the session dimension, and the
  // falsehood is legible on the page as a hole. listRecentOtherActiveSessionEvents
  // filters e.session_id <> currentSessionId, requires both the event and its session
  // to be unarchived, bounds by the recency window, and admits a turn_completed row
  // only when that same session carried a user_contact or borg_replied on the same UTC
  // day. So an hours-long stretch can render with no rows while the store holds plenty:
  // the current session's own events are excluded by construction (they are the
  // transcript), and an autonomous-only session-day is excluded by the same-day gate.
  // The scope token now names that predicate instead of claiming the draw took
  // everything. Two further bounds were legible on the page as the same kind of hole,
  // so the token's definition names them too. First, the group is a merge and not a
  // draw: selectRecentLivedExperienceRows unions the cross-session event lane, the
  // self-decision lane, the day-level rows and the period rows, each drawn under its
  // own upstream cap with no budget shared between them -- so one kind's count there
  // says nothing about another's, and rows_total is a sum of separate limits rather
  // than one limit's output. Second, the event lane spends its cap in kind order
  // (contacts, then replies, then turn completions), and past the window where events
  // are listed individually those events are dropped in favour of the day-level row
  // for their day and session. The two compose: only the kind the cap admits deepest
  // reaches back far enough to be folded, so the surviving mix is an artefact of that
  // order rather than a sample of the store's. The render site still holds the rows
  // and not the limits that produced them, so both are stated as shape, not numbers.
  //
  // Two corrections to the above, both of the same shape: a true sentence that reads
  // as a stronger one. "Each under its own cap" holds for the day-level and period
  // lanes and fails for the two that carry the page -- retrieval-phase hands
  // selectCrossSessionSelfActivity and selectSelfDecisionIntrospection the same
  // recentLivedExperienceConfig.cap. So those two coming back equal at their limit is
  // one configured number applied twice, not two independently-bounded lanes
  // agreeing: the individuation error of counting one construction's two outputs as
  // two witnesses, here written into the string that warns against reading across
  // lanes. And a self-decision row's stamp is the timestamp of the end-of-turn action
  // entry the scheduler appends once the turn returns, not the moment its trigger
  // fired. On a backed-up queue that lands minutes later -- close enough to the NEXT
  // wake to align with it -- so joining these stamps against wake times pairs a
  // decision with the wrong wake rather than failing visibly. The row's own field is
  // what it is; what the token now says is what that field means.
  const relationalDrawScope =
    (context.activeParticipants ?? []).length === 0 ? "global" : "active_participant_subjects";
  const groups = [
    { tag: "relational_slots", rows: relationalSlots.rows, drawScope: relationalDrawScope },
    { tag: "relational_standing", rows: relationalStanding.rows, drawScope: relationalDrawScope },
    { tag: "social_standing", rows: socialStanding.rows, drawScope: "global" },
    {
      tag: "cross_session_entries",
      rows: crossSession.rows,
      drawScope: CROSS_SESSION_ENTRIES_DRAW_SCOPE,
    },
  ];
  const rowCount = groups.reduce((sum, group) => sum + group.rows.length, 0);
  return terminalSection(
    "standing_memory_indexes",
    "terminal_turn_context",
    [
      `<borg_terminal_standing_memory_indexes rows_total_across_groups="${rowCount}" standing_cadence_due="${standing?.renderRecentLivedExperience === true}">`,
      "  <interpretation>Complete membership indexes for relational slots, relational standing, social/observed-event memory, and cross-session lived entries. The groups are drawn by different predicates, so each carries draw_scope naming its own: active_participant_subjects means the draw filtered on subject_entity_id against the current roster; global means it did not filter by audience, participant, or session at all; other_sessions_recent_window means it ran over unarchived sessions other than this one, inside the recent-lived-experience window and under a row cap, and took a turn-completion row only on a day its own session also carried a contact or a reply -- the current session is absent from that group because it is the transcript, so a stretch of time with no rows there is not evidence that nothing happened in it. That group is a merge and not one draw: its individual events, its autonomous self-decisions and its day-level rows are drawn separately with no budget shared between them, so one kind's count there is not evidence about another's -- and the event lane and the self-decision lane are handed one configured cap value rather than two, so those two arriving equal at their limit is one number applied twice and not two lanes agreeing. A self-decision row is stamped when its decision was recorded at the end of its turn, not when its trigger fired, so lining those stamps up against wake times pairs a decision with a later wake. Its event cap is spent on contacts first, then replies, then turn completions, so which kinds survive is an artefact of that order rather than a sample of the mix; and past the window where events are listed individually, a day is carried by its day-level row while its own events are dropped, so a day present only as a day row is a compressed day and not a quiet one. Scope is not inferable from the rows -- a row whose origin_audience is elsewhere is consistent with any of them -- so read draw_scope, not the contents. Where draw_scope is global, the current audience may still rank or annotate; ranking is never a filter. rows_total is per group and rows_total_across_groups is their sum, which is therefore not a total at any single scope. Each group's complete and omitted_count describe the rows its own draw produced; every one of these draws is separately capped upstream, so neither field is evidence about what the store holds. Payload fields are mechanical head+tail excerpts; an excerpt is never a summary. Disclosure labels survive on every row and govern mention, not recall.</interpretation>",
      ...groups.flatMap((group) => {
        return [
          `  <${group.tag} complete="true" rows_total="${group.rows.length}" draw_scope="${group.drawScope}">`,
          ...group.rows.map((row) => `    ${row}`),
          "    <omitted_count>0</omitted_count>",
          `  </${group.tag}>`,
        ];
      }),
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
    terminalSection(
      "finalizer_tool_availability",
      "terminal_turn_context",
      renderFinalizerToolAvailability(input.toolAvailability),
    ),
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
  // - durable_global: creator identity, durable commitment records plus ledger join identity,
  //   and census-stable value/trait identity, text, timestamps, and provenance;
  // - durable_audience: only the already-resolved audience directive briefing and stable frame;
  // - turn_context: live tool availability, every clock, sender, roster, working-state,
  //   retrieval, ledger, and plan input.
  // Mutable self confidence/strength/priority/state/counters and reinforcement/test timestamps,
  // plus every rendered turn-derived commitment ledger field, entity labels, and mutable exact
  // timestamps stay in turn_context. Relative ages derive from those stamps and the turn clock
  // instead of printing.
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
