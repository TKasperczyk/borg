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

export function renderSection(section: EvidenceLedgerSection): string {
  const framingLines =
    section.framing === undefined
      ? []
      : [
          `framing: ${section.framing.text}`,
          ...(section.framing.counts === undefined
            ? []
            : [`framing_counts: ${JSON.stringify(section.framing.counts)}`]),
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
