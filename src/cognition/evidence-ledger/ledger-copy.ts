import { estimatePromptTokens } from "../../util/token-estimate.js";
import { renderEvidenceLedger } from "./finalizer-ledger.js";
import type {
  EvidenceLedger,
  EvidenceLedgerAudienceStanding,
  EvidenceLedgerEntry,
  EvidenceLedgerSection,
} from "./types.js";

function cloneEntry(entry: EvidenceLedgerEntry): EvidenceLedgerEntry {
  return {
    ...entry,
    citations: entry.citations === undefined ? undefined : [...entry.citations],
    state_metadata: entry.state_metadata === undefined ? undefined : { ...entry.state_metadata },
    planner_metadata:
      entry.planner_metadata === undefined ? undefined : { ...entry.planner_metadata },
  };
}

function cloneAudienceStanding(
  standing: EvidenceLedgerAudienceStanding | undefined,
): EvidenceLedgerAudienceStanding | undefined {
  if (standing === undefined) {
    return undefined;
  }

  return {
    recentLivedExperienceEntries: standing.recentLivedExperienceEntries.map(cloneEntry),
    renderRecentLivedExperience: standing.renderRecentLivedExperience,
    observedEventIntrospectionEntries: standing.observedEventIntrospectionEntries.map(cloneEntry),
    commitmentEntries: standing.commitmentEntries.map(cloneEntry),
    relationalEntries: standing.relationalEntries.map(cloneEntry),
  };
}

export function estimateEvidenceLedgerTokens(ledger: EvidenceLedger): number {
  return estimatePromptTokens(
    renderEvidenceLedger({
      ...ledger,
      estimatedTokens: 0,
    }) ?? "",
  );
}

export function cloneLedgerWithSections(
  ledger: EvidenceLedger,
  sections: readonly EvidenceLedgerSection[],
): EvidenceLedger {
  const next = {
    ...ledger,
    sections: sections.map((section) => ({
      ...section,
      entries: section.entries.map(cloneEntry),
    })),
    audienceStanding: cloneAudienceStanding(ledger.audienceStanding),
  };

  return {
    ...next,
    estimatedTokens: estimateEvidenceLedgerTokens(next),
  };
}
