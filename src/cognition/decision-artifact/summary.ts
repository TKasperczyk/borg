import { estimatePromptTokens } from "../../util/token-estimate.js";
import {
  DECISION_ARTIFACT_ENTRY_KINDS,
  type DecisionArtifact,
  type DecisionArtifactEntry,
  type DecisionArtifactEntryKind,
} from "../../memory/decision-artifacts/index.js";
import { normalizePositiveInteger } from "../evidence-ledger/budget.js";
import {
  activeDecisionArtifactEntries,
  compareDecisionArtifactEntriesByRecency,
  countDecisionArtifactEntriesByKind,
  emptyDecisionArtifactKindCounts,
  onePerKindTokenDropFloor,
  selectDecisionArtifactEntriesForRender,
  subtractDecisionArtifactKindCounts,
  tokenDropIndexForKinds,
  type DecisionArtifactKindCounts,
} from "./selection.js";
import { truncateDecisionArtifactText } from "./render.js";

const DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_MAX_ENTRIES = {
  locked: 14,
  live: 8,
  pending: 6,
  invalidated: 4,
  tentative: 2,
} as const satisfies Record<DecisionArtifactEntryKind, number>;
const DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_TOKEN_BUDGET = 6_000;
const DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_ENTRY_TEXT_TOKENS = 1_000;

export type DecisionArtifactPromptSummaryOptions = {
  maxEntries?: Partial<Record<DecisionArtifactEntryKind, number>>;
  summaryTokenBudget?: number;
  maxEntryTextTokens?: number;
};

export type DecisionArtifactPromptSummaryEntry = {
  id: DecisionArtifactEntry["id"];
  text: string;
  owner_entity_id?: NonNullable<DecisionArtifactEntry["owner_entity_id"]>;
  last_updated_stream_entry_id:
    | DecisionArtifactEntry["last_updated_stream_entry_ids"][number]
    | null;
  canonicalizes_ids_count: number;
};

export type DecisionArtifactPromptSummarySupersededEntry = {
  id: DecisionArtifactEntry["id"];
  text: string;
  superseded_by_id: NonNullable<DecisionArtifactEntry["superseded_by_id"]>;
};

export type DecisionArtifactPromptSummary = {
  audience_entity_id: DecisionArtifact["audience_entity_id"];
  record_version: DecisionArtifact["record_version"];
  active_counts_by_kind: DecisionArtifactKindCounts;
  active_entries: Record<DecisionArtifactEntryKind, DecisionArtifactPromptSummaryEntry[]>;
  omitted_counts_by_kind: DecisionArtifactKindCounts;
  recent_superseded: DecisionArtifactPromptSummarySupersededEntry[];
};

function decisionArtifactPromptSummaryOptions(options: DecisionArtifactPromptSummaryOptions = {}): {
  maxEntries: Record<DecisionArtifactEntryKind, number>;
  summaryTokenBudget: number;
  maxEntryTextTokens: number;
} {
  const maxEntries: Record<DecisionArtifactEntryKind, number> = {
    ...DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_MAX_ENTRIES,
  };

  for (const kind of DECISION_ARTIFACT_ENTRY_KINDS) {
    const configured = options.maxEntries?.[kind];

    if (configured !== undefined && Number.isFinite(configured)) {
      maxEntries[kind] = Math.max(0, Math.floor(configured));
    }
  }

  return {
    maxEntries,
    summaryTokenBudget: normalizePositiveInteger(
      options.summaryTokenBudget,
      DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_TOKEN_BUDGET,
    ),
    maxEntryTextTokens: normalizePositiveInteger(
      options.maxEntryTextTokens,
      DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_ENTRY_TEXT_TOKENS,
    ),
  };
}

function emptyDecisionArtifactPromptSummaryEntries(): Record<
  DecisionArtifactEntryKind,
  DecisionArtifactPromptSummaryEntry[]
> {
  return {
    locked: [],
    live: [],
    pending: [],
    invalidated: [],
    tentative: [],
  };
}

function decisionArtifactCanonicalizesIdCount(entry: DecisionArtifactEntry): number {
  return (
    entry.canonicalizes.goal_ids.length +
    entry.canonicalizes.commitment_ids.length +
    entry.canonicalizes.action_ids.length +
    entry.canonicalizes.open_question_ids.length
  );
}

function lastUpdatedStreamEntryId(
  entry: DecisionArtifactEntry,
): DecisionArtifactPromptSummaryEntry["last_updated_stream_entry_id"] {
  return (
    entry.last_updated_stream_entry_ids[entry.last_updated_stream_entry_ids.length - 1] ?? null
  );
}

function toDecisionArtifactPromptSummaryEntry(
  entry: DecisionArtifactEntry,
  maxEntryTextTokens: number,
): DecisionArtifactPromptSummaryEntry {
  const text =
    estimatePromptTokens(entry.text) <= maxEntryTextTokens
      ? entry.text
      : truncateDecisionArtifactText(entry.text, maxEntryTextTokens);

  return {
    id: entry.id,
    text,
    ...(entry.owner_entity_id === null ? {} : { owner_entity_id: entry.owner_entity_id }),
    last_updated_stream_entry_id: lastUpdatedStreamEntryId(entry),
    canonicalizes_ids_count: decisionArtifactCanonicalizesIdCount(entry),
  };
}

function toDecisionArtifactPromptSummarySupersededEntry(
  entry: DecisionArtifactEntry,
): DecisionArtifactPromptSummarySupersededEntry | null {
  if (entry.superseded_by_id === null) {
    return null;
  }

  return {
    id: entry.id,
    text: entry.text,
    superseded_by_id: entry.superseded_by_id,
  };
}

function selectedDecisionArtifactPromptSummaryEntries(input: {
  activeEntries: readonly DecisionArtifactEntry[];
  maxEntries: Record<DecisionArtifactEntryKind, number>;
}): DecisionArtifactEntry[] {
  const totalMaxEntries = DECISION_ARTIFACT_ENTRY_KINDS.reduce(
    (sum, kind) => sum + input.maxEntries[kind],
    0,
  );
  const cappedActiveEntries = DECISION_ARTIFACT_ENTRY_KINDS.flatMap((kind) =>
    input.activeEntries
      .filter((entry) => entry.kind === kind)
      .sort(compareDecisionArtifactEntriesByRecency)
      .slice(0, input.maxEntries[kind]),
  );
  const selected = selectDecisionArtifactEntriesForRender({
    entries: cappedActiveEntries,
    maxEntries: totalMaxEntries,
    reservedSlots: {
      live: input.maxEntries.live,
      pending: input.maxEntries.pending,
      invalidated: input.maxEntries.invalidated,
    },
    lockedMaxEntries: input.maxEntries.locked,
  });
  const counts = emptyDecisionArtifactKindCounts();

  return selected.filter((entry) => {
    if (counts[entry.kind] >= input.maxEntries[entry.kind]) {
      return false;
    }

    counts[entry.kind] += 1;
    return true;
  });
}

function buildDecisionArtifactPromptSummaryFromEntries(input: {
  artifact: DecisionArtifact;
  activeEntries: readonly DecisionArtifactEntry[];
  selectedEntries: readonly DecisionArtifactEntry[];
  recentSuperseded: readonly DecisionArtifactPromptSummarySupersededEntry[];
  maxEntryTextTokens: number;
}): DecisionArtifactPromptSummary {
  const activeEntriesByKind = emptyDecisionArtifactPromptSummaryEntries();

  for (const entry of input.selectedEntries) {
    activeEntriesByKind[entry.kind].push(
      toDecisionArtifactPromptSummaryEntry(entry, input.maxEntryTextTokens),
    );
  }

  return {
    audience_entity_id: input.artifact.audience_entity_id,
    record_version: input.artifact.record_version,
    active_counts_by_kind: countDecisionArtifactEntriesByKind(input.activeEntries),
    active_entries: activeEntriesByKind,
    omitted_counts_by_kind: subtractDecisionArtifactKindCounts(
      countDecisionArtifactEntriesByKind(input.activeEntries),
      countDecisionArtifactEntriesByKind(input.selectedEntries),
    ),
    recent_superseded: [...input.recentSuperseded],
  };
}

function decisionArtifactPromptSummaryTokenEstimate(
  summary: DecisionArtifactPromptSummary,
): number {
  return estimatePromptTokens(JSON.stringify(summary));
}

function decisionArtifactPromptSummaryDropIndex(input: {
  entries: readonly DecisionArtifactEntry[];
  activeCounts: DecisionArtifactKindCounts;
}): number | null {
  const dropTentative = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["tentative"],
    minimumForKind: () => 0,
  });

  if (dropTentative !== null) {
    return dropTentative;
  }

  const dropInvalidated = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["invalidated"],
    minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
  });

  if (dropInvalidated !== null) {
    return dropInvalidated;
  }

  return tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["live", "pending", "locked"],
    minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
  });
}

export function buildDecisionArtifactPromptSummary(
  artifact: DecisionArtifact | null | undefined,
  options?: DecisionArtifactPromptSummaryOptions,
): DecisionArtifactPromptSummary | null {
  if (artifact === null || artifact === undefined) {
    return null;
  }

  const normalizedOptions = decisionArtifactPromptSummaryOptions(options);
  const activeEntries = activeDecisionArtifactEntries(artifact);
  let selectedEntries = selectedDecisionArtifactPromptSummaryEntries({
    activeEntries,
    maxEntries: normalizedOptions.maxEntries,
  });
  let recentSuperseded = artifact.entries
    .filter((entry) => entry.superseded_by_id !== null)
    .sort(compareDecisionArtifactEntriesByRecency)
    .slice(0, 5)
    .flatMap((entry) => {
      const summarized = toDecisionArtifactPromptSummarySupersededEntry(entry);

      return summarized === null ? [] : [summarized];
    });
  let summary = buildDecisionArtifactPromptSummaryFromEntries({
    artifact,
    activeEntries,
    selectedEntries,
    recentSuperseded,
    maxEntryTextTokens: normalizedOptions.maxEntryTextTokens,
  });

  while (
    decisionArtifactPromptSummaryTokenEstimate(summary) > normalizedOptions.summaryTokenBudget &&
    recentSuperseded.length > 0
  ) {
    recentSuperseded = recentSuperseded.slice(0, -1);
    summary = buildDecisionArtifactPromptSummaryFromEntries({
      artifact,
      activeEntries,
      selectedEntries,
      recentSuperseded,
      maxEntryTextTokens: normalizedOptions.maxEntryTextTokens,
    });
  }

  while (
    decisionArtifactPromptSummaryTokenEstimate(summary) > normalizedOptions.summaryTokenBudget &&
    selectedEntries.length > 0
  ) {
    const dropIndex = decisionArtifactPromptSummaryDropIndex({
      entries: selectedEntries,
      activeCounts: countDecisionArtifactEntriesByKind(activeEntries),
    });

    if (dropIndex === null) {
      break;
    }

    selectedEntries = [
      ...selectedEntries.slice(0, dropIndex),
      ...selectedEntries.slice(dropIndex + 1),
    ];
    summary = buildDecisionArtifactPromptSummaryFromEntries({
      artifact,
      activeEntries,
      selectedEntries,
      recentSuperseded,
      maxEntryTextTokens: normalizedOptions.maxEntryTextTokens,
    });
  }

  return summary;
}
