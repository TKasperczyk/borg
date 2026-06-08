import { estimatePromptTokens } from "../../util/token-estimate.js";
import { emptySectionCountRecord } from "./budget.js";
import type {
  EvidenceLedger,
  EvidenceLedgerEntry,
  EvidenceLedgerSection,
  EvidenceLedgerSectionId,
  EvidenceLedgerTraceSummary,
} from "./types.js";

function entryFlatText(section: EvidenceLedgerSection, entry: EvidenceLedgerEntry): string {
  return [
    section.label,
    entry.id,
    entry.source_type,
    entry.session_scope,
    entry.actor,
    String(entry.trust_rank),
    entry.stream_index === undefined ? "" : String(entry.stream_index),
    entry.state ?? "",
    entry.taint ?? "",
    entry.persistence_class ?? "",
    entry.via_retrieval === true ? "via_retrieval" : "",
    entry.value ?? "",
    entry.text ?? "",
  ].join("\n");
}

function sectionFramingFlatText(section: EvidenceLedgerSection): string {
  if (section.framing === undefined) {
    return "";
  }

  return [
    section.label,
    section.framing.text,
    section.framing.counts === undefined ? "" : JSON.stringify(section.framing.counts),
  ].join("\n");
}

function estimateSectionTokens(section: EvidenceLedgerSection): number {
  const text = [
    sectionFramingFlatText(section),
    ...section.entries.map((entry) => entryFlatText(section, entry)),
  ]
    .filter((part) => part.length > 0)
    .join("\n");
  return text.length === 0 ? 0 : estimatePromptTokens(text);
}

export function estimateLedgerTokens(sections: readonly EvidenceLedgerSection[]): number {
  const text = sections
    .flatMap((section) => [
      sectionFramingFlatText(section),
      ...section.entries.map((entry) => entryFlatText(section, entry)),
    ])
    .filter((part) => part.length > 0)
    .join("\n");

  return text.length === 0 ? 0 : estimatePromptTokens(text);
}

export function summarizeEvidenceLedgerTrace(ledger: EvidenceLedger): EvidenceLedgerTraceSummary {
  // Sprint 8d.3: per-section token accounting. v36 mean input tokens were
  // ~113k -- without per-section breakdown there is no way to attribute
  // that load to specific bands. Surfacing it in the trace lets us tell
  // ledger-side bloat (transcript, action records) apart from
  // retrieval-side bloat (semantic walks, episodes).
  const estimatedTokensBySection = emptySectionCountRecord();
  const entryCountsBySection = emptySectionCountRecord();

  for (const section of ledger.sections) {
    estimatedTokensBySection[section.id] = estimateSectionTokens(section);
    entryCountsBySection[section.id] = section.entries.length;
  }

  return {
    entryCountsBySection,
    estimatedTokensBySection,
    transcriptIncluded: ledger.transcriptIncluded,
    transcriptCompacted: ledger.transcriptCompacted,
    transcriptOmittedReason: ledger.transcriptOmittedReason,
    originalTranscriptTokenEstimate: ledger.originalTranscriptTokenEstimate,
    compactedTranscriptTokenEstimate: estimatedTokensBySection.current_session_transcript,
    compactedEntryCount: ledger.compactedTranscriptEntryCount,
    rawPreservedUserEntryCount: ledger.rawPreservedUserTranscriptEntryCount,
    totalEstimatedTokens: ledger.estimatedTokens,
  };
}
