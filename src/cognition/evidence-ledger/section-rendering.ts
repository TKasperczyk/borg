import type {
  EvidenceLedgerEntry,
  EvidenceLedgerSection,
  EvidenceLedgerSectionFramingCounts,
} from "./types.js";

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

// A framing count can be misread on two independent axes, and naming one leaves the other running.
//
// Stage: the counts are taken when a section is assembled, upstream of both reductions the ledger
// then applies to it -- provenance dedupe folding an entry into a higher-priority section, and
// budget omission -- and are never recomputed against what survived. The per-entry
// autobiographical_recall_cap.rendered_count IS recomputed, twice, so the two numbers on the same
// section are measured at different stages and can disagree without either being wrong. Naming the
// stage explained a shortfall without sizing it: the budget reduction announced itself as an
// evidence_ledger_omitted entry and the dedupe half left no count anywhere, so a page could show
// forty-eight assembled, seven printed and seventeen omitted and leave the other twenty-four
// unaccountable in either direction. The dedupe stage now writes its own per-section drop count
// back onto the framing, so both reductions subtract from the population to the page. What remains
// deliberately unclaimed is that the subtraction always closes: a section can carry an omission
// entry from a stage earlier than either -- recent_lived_experience prints the compiler's own
// detail-omission breadcrumb -- so a residue names a third reduction rather than a lost row.
//
// Partition: every key counts a named subset, and until the population was printed beside them a
// reader had no denominator. autobiographical_recall counts one kind out of the twelve it can
// assemble, so a page carrying ten rows of other kinds prints a figure of zero above them: not a
// shortfall at all, and unreachable by any amount of stage-naming. rows_assembled is the fix and
// it is required by the type rather than agreed by convention, so a subset figure cannot be
// printed without the population it was taken over.
const FRAMING_COUNTS_SCOPE =
  "framing_counts_scope: rows_assembled is the population this section was assembled from, and every other figure counts one named subset of that population, so the figures need not sum to it, need not cover it, and one of them can sit below the number of rows printed here without a row having gone missing. Two reductions run between that count and this page and both report themselves: folded_out_by_provenance counts rows dropped because a higher-priority section carries the same evidence or because they merged into a sibling row here, and the omitted-entries row below counts what the finalizer ledger budget then cut. That key is written by the dedupe stage itself, so a render taken before that stage carries no such key at all and its absence there is not a zero. Those two are fates rather than content classes, so they crosscut the other subsets instead of joining them, and subtracting both from rows_assembled should leave the rows printed here; a residue is a reduction neither of them names, which is what an omission row from an earlier stage counts.";

function orderedFramingCounts(
  counts: EvidenceLedgerSectionFramingCounts,
): EvidenceLedgerSectionFramingCounts {
  const { rows_assembled: rowsAssembled, ...subsets } = counts;

  return { rows_assembled: rowsAssembled, ...subsets };
}

export function renderSection(section: EvidenceLedgerSection): string {
  const framingLines =
    section.framing === undefined
      ? []
      : [
          `framing: ${section.framing.text}`,
          ...(section.framing.counts === undefined
            ? []
            : [
                // The population leads the object wherever the call site happens to put it, so the
                // key the scope line names is never buried between the subsets it bounds.
                `framing_counts: ${JSON.stringify(orderedFramingCounts(section.framing.counts))}`,
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
