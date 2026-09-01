import type { EvidenceLedgerEntry, EvidenceLedgerSection } from "./types.js";

export function renderEntry(entry: EvidenceLedgerEntry): string {
  const stateMetadata =
    entry.state_metadata === undefined ? undefined : JSON.stringify(entry.state_metadata);
  const metadata = [
    `id=${entry.id}`,
    `source_type=${entry.source_type}`,
    `scope=${entry.session_scope}`,
    `actor=${entry.actor}`,
    `trust_rank=${entry.trust_rank}`,
    entry.citations === undefined || entry.citations.length === 0
      ? null
      : `[citation: ${entry.citations.join(", ")}]`,
    entry.citation_type === undefined ? null : `citation_type=${entry.citation_type}`,
    entry.stream_index === undefined ? null : `stream_index=${entry.stream_index}`,
    entry.state === undefined ? null : `state=${entry.state}`,
    entry.salience_class === undefined ? null : `salience_class=${entry.salience_class}`,
    stateMetadata === undefined ? null : `state_metadata=${stateMetadata}`,
    entry.taint === undefined ? null : `taint=${entry.taint}`,
    entry.persistence_class === undefined ? null : `persistence_class=${entry.persistence_class}`,
    entry.via_retrieval === true ? "via_retrieval=true" : null,
  ].filter((part): part is string => part !== null);
  const body = [
    entry.value === undefined ? null : `  value: ${entry.value}`,
    entry.text === undefined ? null : `  text:\n${entry.text}`,
  ].filter((part): part is string => part !== null);

  return [`- ${metadata.join(" ")}`, ...body].join("\n");
}

// Framing counts are taken when a section is assembled, upstream of both reductions the ledger
// then applies to it -- provenance dedupe folding an entry into a higher-priority section, and
// budget omission -- and are never recomputed against what survived. The per-entry
// autobiographical_recall_cap.rendered_count IS recomputed, twice, so the two numbers on the same
// section are measured at different stages and can disagree without either being wrong. Only the
// budget reduction announces itself, as an evidence_ledger_omitted entry; the dedupe half leaves
// no count anywhere, so the shortfall is not always accountable from the page. And a section can
// carry an omission entry from a stage earlier than either -- recent_lived_experience prints the
// compiler's own detail-omission breadcrumb -- so the last clause has to name which reduction the
// reported number belongs to, or it reads as accounting for a gap it does not cover. Naming the
// stage on the count's own line is the whole fix: the number stays the only surface carrying the
// pre-reduction figure, and stops reading as a description of the rows printed under it.
const FRAMING_COUNTS_SCOPE =
  "framing_counts_scope: counted over the rows this section was assembled from, before the ledger folded overlapping provenance into higher-priority sections and before it dropped rows for budget; the rows below survived both, so a shortfall against this count is a row removed after it was taken. Only the budget removal is reported, by an omitted-entries row naming the finalizer ledger budget; an omission row naming any other stage counts a different reduction again.";

export function renderSection(section: EvidenceLedgerSection): string {
  const framingLines =
    section.framing === undefined
      ? []
      : [
          `framing: ${section.framing.text}`,
          ...(section.framing.counts === undefined
            ? []
            : [
                `framing_counts: ${JSON.stringify(section.framing.counts)}`,
                FRAMING_COUNTS_SCOPE,
              ]),
        ];

  if (section.entries.length === 0) {
    return [`## ${section.label}`, ...framingLines, "No entries."].join("\n");
  }

  const sourceTypes = [...new Set(section.entries.map((entry) => entry.source_type))].join(", ");
  const scopes = [...new Set(section.entries.map((entry) => entry.session_scope))].join(", ");

  return [
    `## ${section.label}`,
    ...framingLines,
    `source_types: ${sourceTypes}`,
    `scopes: ${scopes}`,
    ...section.entries.map((entry) => renderEntry(entry)),
  ].join("\n");
}
