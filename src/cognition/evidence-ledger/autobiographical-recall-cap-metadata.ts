import type { AutobiographicalRecallCapMetadata } from "../autobiographical-recall.js";
import { cloneLedgerWithSections } from "./ledger-copy.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "./types.js";

export const AUTOBIOGRAPHICAL_RECALL_CAP_METADATA_KEY = "autobiographical_recall_cap";

type CandidateCountStateMetadata =
  | {
      candidate_count: number;
      candidate_count_lower_bound?: never;
    }
  | {
      candidate_count?: never;
      candidate_count_lower_bound: number;
    };

type SourceGroupCapStateMetadata = CandidateCountStateMetadata & {
  rendered_count: number;
  candidate_scope?: "scanned_sessions";
};

type TotalCapStateMetadata = {
  rendered_count: number;
  candidate_count: number;
  candidate_scope: "post_source_caps";
};

type AutobiographicalRecallCapStateMetadata = {
  source_group?: SourceGroupCapStateMetadata;
  total?: TotalCapStateMetadata;
};

export function autobiographicalRecallCapStateMetadata(
  capMetadata: AutobiographicalRecallCapMetadata | undefined,
): Record<string, unknown> {
  if (capMetadata === undefined) {
    return {};
  }

  return {
    [AUTOBIOGRAPHICAL_RECALL_CAP_METADATA_KEY]: {
      ...(capMetadata.sourceGroup === undefined
        ? {}
        : {
            source_group: {
              rendered_count: capMetadata.sourceGroup.renderedCount,
              ...(capMetadata.sourceGroup.candidateCount === undefined
                ? {
                    candidate_count_lower_bound: capMetadata.sourceGroup.candidateCountLowerBound,
                  }
                : { candidate_count: capMetadata.sourceGroup.candidateCount }),
              ...(capMetadata.sourceGroup.candidateScope === undefined
                ? {}
                : { candidate_scope: capMetadata.sourceGroup.candidateScope }),
            },
          }),
      ...(capMetadata.total === undefined
        ? {}
        : {
            total: {
              rendered_count: capMetadata.total.renderedCount,
              candidate_count: capMetadata.total.candidateCount,
              candidate_scope: capMetadata.total.candidateScope,
            },
          }),
    } satisfies AutobiographicalRecallCapStateMetadata,
  };
}

function recallGroupId(entry: EvidenceLedgerEntry): string | null {
  const groupId = entry.state_metadata?.group_id;
  return typeof groupId === "string" ? groupId : null;
}

function capStateMetadata(
  entry: EvidenceLedgerEntry,
): AutobiographicalRecallCapStateMetadata | null {
  const value = entry.state_metadata?.[AUTOBIOGRAPHICAL_RECALL_CAP_METADATA_KEY];
  return typeof value === "object" && value !== null
    ? (value as AutobiographicalRecallCapStateMetadata)
    : null;
}

function withoutCapStateMetadata(entry: EvidenceLedgerEntry): EvidenceLedgerEntry {
  if (
    entry.state_metadata === undefined ||
    !(AUTOBIOGRAPHICAL_RECALL_CAP_METADATA_KEY in entry.state_metadata)
  ) {
    return entry;
  }

  const stateMetadata = { ...entry.state_metadata };
  delete stateMetadata[AUTOBIOGRAPHICAL_RECALL_CAP_METADATA_KEY];

  return {
    ...entry,
    state_metadata: Object.keys(stateMetadata).length === 0 ? undefined : stateMetadata,
  };
}

function withCapStateMetadata(
  entry: EvidenceLedgerEntry,
  capMetadata: AutobiographicalRecallCapStateMetadata,
): EvidenceLedgerEntry {
  const withoutCap = withoutCapStateMetadata(entry);
  return {
    ...withoutCap,
    state_metadata: {
      ...(withoutCap.state_metadata ?? {}),
      [AUTOBIOGRAPHICAL_RECALL_CAP_METADATA_KEY]: capMetadata,
    },
  };
}

// Candidate totals originate before provenance dedupe. Reassign them onto the final section so a
// surviving group always retains one carrier and rendered_count describes model-visible rows.
export function reconcileAutobiographicalRecallCapMetadata(input: {
  factLedger: EvidenceLedger;
  renderedLedger: EvidenceLedger;
}): EvidenceLedger {
  const factSection = input.factLedger.sections.find(
    (section) => section.id === "autobiographical_recall",
  );
  const renderedSection = input.renderedLedger.sections.find(
    (section) => section.id === "autobiographical_recall",
  );

  if (factSection === undefined || renderedSection === undefined) {
    return input.renderedLedger;
  }

  const sourceGroupFacts = new Map<string, SourceGroupCapStateMetadata>();
  let totalFacts: TotalCapStateMetadata | undefined;

  for (const entry of factSection.entries) {
    const groupId = recallGroupId(entry);
    const capMetadata = capStateMetadata(entry);

    if (
      groupId !== null &&
      capMetadata?.source_group !== undefined &&
      !sourceGroupFacts.has(groupId)
    ) {
      sourceGroupFacts.set(groupId, capMetadata.source_group);
    }

    if (totalFacts === undefined && capMetadata?.total !== undefined) {
      totalFacts = capMetadata.total;
    }
  }

  if (sourceGroupFacts.size === 0 && totalFacts === undefined) {
    return input.renderedLedger;
  }

  const renderedCountsByGroup = new Map<string, number>();
  let renderedRecallCount = 0;

  for (const entry of renderedSection.entries) {
    const groupId = recallGroupId(entry);

    if (groupId === null) {
      continue;
    }

    renderedRecallCount += 1;
    renderedCountsByGroup.set(groupId, (renderedCountsByGroup.get(groupId) ?? 0) + 1);
  }

  const annotatedGroups = new Set<string>();
  let totalAnnotated = false;
  const entries = renderedSection.entries.map((entry) => {
    const groupId = recallGroupId(entry);
    const sourceGroupFact = groupId === null ? undefined : sourceGroupFacts.get(groupId);
    const includeSourceGroup =
      groupId !== null && sourceGroupFact !== undefined && !annotatedGroups.has(groupId);
    const includeTotal = groupId !== null && totalFacts !== undefined && !totalAnnotated;

    if (includeSourceGroup) {
      annotatedGroups.add(groupId);
    }

    if (includeTotal) {
      totalAnnotated = true;
    }

    if (!includeSourceGroup && !includeTotal) {
      return withoutCapStateMetadata(entry);
    }

    return withCapStateMetadata(entry, {
      ...(includeSourceGroup && groupId !== null && sourceGroupFact !== undefined
        ? {
            source_group: {
              ...sourceGroupFact,
              rendered_count: renderedCountsByGroup.get(groupId) ?? 0,
            },
          }
        : {}),
      ...(includeTotal && totalFacts !== undefined
        ? {
            total: {
              ...totalFacts,
              rendered_count: renderedRecallCount,
            },
          }
        : {}),
    });
  });

  return cloneLedgerWithSections(
    input.renderedLedger,
    input.renderedLedger.sections.map((section) =>
      section.id === "autobiographical_recall" ? { ...section, entries } : section,
    ),
  );
}
