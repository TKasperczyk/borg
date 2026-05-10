import { estimatePromptTokens } from "../../util/token-estimate.js";
import { UNTRUSTED_DATA_PREAMBLE } from "../deliberation/constants.js";
import { renderTaggedPromptBlock } from "../deliberation/prompt/sections.js";
import type { EvidenceLedger, EvidenceLedgerEntry, EvidenceLedgerSection } from "./types.js";

const HIERARCHY_GUIDANCE = [
  "Current-session transcript is authoritative for what happened in this conversation.",
  "Prior-session memory must be attributed or hedged.",
  "Episodes and semantic graph are summaries; use source handles when making exact claims.",
  "Quarantined/contested/assistant-seeded values are not facts.",
].join("\n");

function renderEntry(entry: EvidenceLedgerEntry): string {
  const stateMetadata =
    entry.state_metadata === undefined ? undefined : JSON.stringify(entry.state_metadata);
  const metadata = [
    `id=${entry.id}`,
    `source_type=${entry.source_type}`,
    `scope=${entry.session_scope}`,
    `actor=${entry.actor}`,
    `trust_rank=${entry.trust_rank}`,
    entry.stream_index === undefined ? null : `stream_index=${entry.stream_index}`,
    entry.state === undefined ? null : `state=${entry.state}`,
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

function renderSection(section: EvidenceLedgerSection): string {
  if (section.entries.length === 0) {
    return [`## ${section.label}`, "No entries."].join("\n");
  }

  const sourceTypes = [...new Set(section.entries.map((entry) => entry.source_type))].join(", ");
  const scopes = [...new Set(section.entries.map((entry) => entry.session_scope))].join(", ");

  return [
    `## ${section.label}`,
    `source_types: ${sourceTypes}`,
    `scopes: ${scopes}`,
    ...section.entries.map((entry) => renderEntry(entry)),
  ].join("\n");
}

export function renderEvidenceLedger(ledger: EvidenceLedger): string | null {
  const transcriptStatus = ledger.transcriptIncluded
    ? ledger.transcriptCompacted
      ? "current_session_transcript=included compacted=true"
      : "current_session_transcript=included"
    : `current_session_transcript=omitted reason=${ledger.transcriptOmittedReason ?? "unknown"}`;
  const content = [
    "EvidenceLedger: prioritized evidence substrate for the final response.",
    HIERARCHY_GUIDANCE,
    transcriptStatus,
    `estimated_tokens=${ledger.estimatedTokens}`,
    ...ledger.sections.map((section) => renderSection(section)),
  ].join("\n\n");

  return renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
    {
      tag: "borg_evidence_ledger",
      content,
    },
  ]);
}

export function estimateEvidenceLedgerPromptTokens(ledger: EvidenceLedger): number {
  return estimatePromptTokens(renderEvidenceLedger(ledger) ?? "");
}
