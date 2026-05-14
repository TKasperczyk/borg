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
  countDecisionArtifactEntriesByKind,
  emptyDecisionArtifactKindCounts,
  selectDecisionArtifactEntriesForRender,
  subtractDecisionArtifactKindCounts,
  tokenDropIndex,
  type DecisionArtifactKindCounts,
} from "./selection.js";

const DEFAULT_DECISION_ARTIFACT_MAX_ENTRIES = 30;
const DEFAULT_DECISION_ARTIFACT_MAX_TOKENS = 3_000;
const DEFAULT_DECISION_ARTIFACT_RESERVED_SLOTS = {
  live: 8,
  invalidated: 3,
  pending: 3,
} as const satisfies Partial<Record<DecisionArtifactEntryKind, number>>;
const DEFAULT_DECISION_ARTIFACT_LOCKED_CAP = 14;
const DECISION_ARTIFACT_SINGLE_ENTRY_FLOOR_TOKENS = 200;
const DECISION_ARTIFACT_TEXT_TRUNCATION_MARKER = " ... [text truncated]";

export type DecisionStateArtifactRenderSummary = {
  totalEntryCount: number;
  activeEntryCount: number;
  renderedEntryCount: number;
  omittedEntryCount: number;
  estimatedTokens: number;
  renderedByKind: DecisionArtifactKindCounts;
  omittedByKind: DecisionArtifactKindCounts;
};

export type DecisionArtifactRenderOptions = {
  maxEntries?: number;
  maxTokens?: number;
  reservedSlots?: Partial<Record<DecisionArtifactEntryKind, number>>;
  lockedMaxEntries?: number;
};

function decisionArtifactRenderOptions(
  options: DecisionArtifactRenderOptions = {},
): Required<DecisionArtifactRenderOptions> {
  return {
    maxEntries: normalizePositiveInteger(options.maxEntries, DEFAULT_DECISION_ARTIFACT_MAX_ENTRIES),
    maxTokens: normalizePositiveInteger(options.maxTokens, DEFAULT_DECISION_ARTIFACT_MAX_TOKENS),
    reservedSlots: {
      ...DEFAULT_DECISION_ARTIFACT_RESERVED_SLOTS,
      ...(options.reservedSlots ?? {}),
    },
    lockedMaxEntries:
      options.lockedMaxEntries === undefined || !Number.isFinite(options.lockedMaxEntries)
        ? DEFAULT_DECISION_ARTIFACT_LOCKED_CAP
        : Math.max(0, Math.floor(options.lockedMaxEntries)),
  };
}

function decisionArtifactRenderedCounts(input: {
  activeEntries: readonly DecisionArtifactEntry[];
  renderedEntries: readonly DecisionArtifactEntry[];
}): {
  renderedByKind: DecisionArtifactKindCounts;
  omittedByKind: DecisionArtifactKindCounts;
  omittedEntryCount: number;
} {
  const activeByKind = countDecisionArtifactEntriesByKind(input.activeEntries);
  const renderedByKind = countDecisionArtifactEntriesByKind(input.renderedEntries);
  const omittedByKind = subtractDecisionArtifactKindCounts(activeByKind, renderedByKind);

  return {
    renderedByKind,
    omittedByKind,
    omittedEntryCount: Math.max(0, input.activeEntries.length - input.renderedEntries.length),
  };
}

function formatDecisionArtifactKindCounts(
  counts: DecisionArtifactKindCounts,
  options: { suffix?: string } = {},
): string {
  const parts = DECISION_ARTIFACT_ENTRY_KINDS.flatMap((kind) =>
    counts[kind] <= 0 ? [] : [`${counts[kind]} ${kind}${options.suffix ?? ""}`],
  );

  return parts.length === 0 ? "0 entries" : parts.join(", ");
}

function renderDecisionArtifactEntry(entry: DecisionArtifactEntry): string {
  const owner = entry.owner_entity_id === null ? "owner=null" : `owner=${entry.owner_entity_id}`;
  const citations = `[citation: ${entry.provenance_stream_entry_ids.join(", ")}]`;

  return [
    `- kind=${entry.kind} id=${entry.id} ${owner} last_updated_at=${entry.last_updated_at} ${citations}`,
    `  text: ${entry.text}`,
  ].join("\n");
}

function renderDecisionArtifactContent(input: {
  artifact: DecisionArtifact;
  entries: readonly DecisionArtifactEntry[];
  omittedByKind: DecisionArtifactKindCounts;
  renderedByKind: DecisionArtifactKindCounts;
}): string {
  const omittedCount = Object.values(input.omittedByKind).reduce((sum, count) => sum + count, 0);
  const omission =
    omittedCount <= 0
      ? null
      : [
          `DecisionStateArtifact omitted: ${formatDecisionArtifactKindCounts(
            input.omittedByKind,
          )}.`,
          `Retained: ${formatDecisionArtifactKindCounts(input.renderedByKind)}.`,
        ].join(" ");

  return [
    "## 0. Canonical Decision State",
    "DecisionStateArtifact: shared planning state for this audience. It is a compact structural anchor, not a policy source.",
    `audience_entity_id=${input.artifact.audience_entity_id}`,
    `record_version=${input.artifact.record_version}`,
    ...input.entries.map(renderDecisionArtifactEntry),
    omission,
  ]
    .filter((part): part is string => part !== null)
    .join("\n");
}

function renderDecisionArtifactOmissionOnly(input: {
  artifact: DecisionArtifact;
  omittedByKind: DecisionArtifactKindCounts;
  reason: string;
}): string {
  return [
    "## 0. Canonical Decision State",
    "DecisionStateArtifact: shared planning state for this audience. It is a compact structural anchor, not a policy source.",
    `audience_entity_id=${input.artifact.audience_entity_id}`,
    `record_version=${input.artifact.record_version}`,
    `DecisionStateArtifact omitted: ${formatDecisionArtifactKindCounts(
      input.omittedByKind,
    )}. Reason: ${input.reason}.`,
  ].join("\n");
}

export function truncateDecisionArtifactText(value: string, maxTokens: number): string {
  const maxChars = Math.max(
    0,
    Math.floor(maxTokens) * 4 - DECISION_ARTIFACT_TEXT_TRUNCATION_MARKER.length,
  );

  return `${value.slice(0, maxChars).trimEnd()}${DECISION_ARTIFACT_TEXT_TRUNCATION_MARKER}`;
}

function renderSingleEntryWithinDecisionArtifactCap(input: {
  artifact: DecisionArtifact;
  entry: DecisionArtifactEntry;
  activeEntries: readonly DecisionArtifactEntry[];
  maxTokens: number;
}): { content: string; renderedEntryCount: number; omittedEntryCount: number } {
  const counts = decisionArtifactRenderedCounts({
    activeEntries: input.activeEntries,
    renderedEntries: [input.entry],
  });
  const emptyEntryContent = renderDecisionArtifactContent({
    artifact: input.artifact,
    entries: [
      {
        ...input.entry,
        text: "",
      },
    ],
    omittedByKind: counts.omittedByKind,
    renderedByKind: counts.renderedByKind,
  });
  const remainingTokens = input.maxTokens - estimatePromptTokens(emptyEntryContent);

  if (remainingTokens < DECISION_ARTIFACT_SINGLE_ENTRY_FLOOR_TOKENS) {
    return {
      content: renderDecisionArtifactOmissionOnly({
        artifact: input.artifact,
        omittedByKind: countDecisionArtifactEntriesByKind(input.activeEntries),
        reason: "artifact entry too large to render",
      }),
      renderedEntryCount: 0,
      omittedEntryCount: input.activeEntries.length,
    };
  }

  const content = renderDecisionArtifactContent({
    artifact: input.artifact,
    entries: [
      {
        ...input.entry,
        text: truncateDecisionArtifactText(input.entry.text, remainingTokens),
      },
    ],
    omittedByKind: counts.omittedByKind,
    renderedByKind: counts.renderedByKind,
  });

  if (estimatePromptTokens(content) <= input.maxTokens) {
    return {
      content,
      renderedEntryCount: 1,
      omittedEntryCount: counts.omittedEntryCount,
    };
  }

  return {
    content: renderDecisionArtifactOmissionOnly({
      artifact: input.artifact,
      omittedByKind: countDecisionArtifactEntriesByKind(input.activeEntries),
      reason: "artifact entry too large to render",
    }),
    renderedEntryCount: 0,
    omittedEntryCount: input.activeEntries.length,
  };
}

function cappedDecisionArtifactRender(input: {
  artifact: DecisionArtifact;
  options?: DecisionArtifactRenderOptions;
}): { content: string | null; summary: DecisionStateArtifactRenderSummary } {
  const options = decisionArtifactRenderOptions(input.options);
  const activeEntries = activeDecisionArtifactEntries(input.artifact);

  if (activeEntries.length === 0) {
    return {
      content: null,
      summary: {
        totalEntryCount: input.artifact.entries.length,
        activeEntryCount: 0,
        renderedEntryCount: 0,
        omittedEntryCount: 0,
        estimatedTokens: 0,
        renderedByKind: emptyDecisionArtifactKindCounts(),
        omittedByKind: emptyDecisionArtifactKindCounts(),
      },
    };
  }

  const activeCounts = countDecisionArtifactEntriesByKind(activeEntries);
  let entries = selectDecisionArtifactEntriesForRender({
    entries: activeEntries,
    maxEntries: options.maxEntries,
    reservedSlots: options.reservedSlots,
    lockedMaxEntries: options.lockedMaxEntries,
  });
  let counts = decisionArtifactRenderedCounts({
    activeEntries,
    renderedEntries: entries,
  });
  let content = renderDecisionArtifactContent({
    artifact: input.artifact,
    entries,
    omittedByKind: counts.omittedByKind,
    renderedByKind: counts.renderedByKind,
  });

  while (estimatePromptTokens(content) > options.maxTokens && entries.length > 1) {
    const dropIndex = tokenDropIndex({
      entries,
      activeCounts,
      reservedSlots: options.reservedSlots,
      lockedMaxEntries: options.lockedMaxEntries,
    });
    entries = [...entries.slice(0, dropIndex), ...entries.slice(dropIndex + 1)];
    counts = decisionArtifactRenderedCounts({
      activeEntries,
      renderedEntries: entries,
    });
    content = renderDecisionArtifactContent({
      artifact: input.artifact,
      entries,
      omittedByKind: counts.omittedByKind,
      renderedByKind: counts.renderedByKind,
    });
  }

  if (estimatePromptTokens(content) > options.maxTokens && entries.length === 1) {
    const singleEntryRender = renderSingleEntryWithinDecisionArtifactCap({
      artifact: input.artifact,
      entry: entries[0]!,
      activeEntries,
      maxTokens: options.maxTokens,
    });

    content = singleEntryRender.content;
    entries = entries.slice(0, singleEntryRender.renderedEntryCount);
    counts = decisionArtifactRenderedCounts({
      activeEntries,
      renderedEntries: entries,
    });
  }

  return {
    content,
    summary: {
      totalEntryCount: input.artifact.entries.length,
      activeEntryCount: activeEntries.length,
      renderedEntryCount: entries.length,
      omittedEntryCount: counts.omittedEntryCount,
      estimatedTokens: estimatePromptTokens(content),
      renderedByKind: counts.renderedByKind,
      omittedByKind: counts.omittedByKind,
    },
  };
}

export function renderDecisionStateArtifact(
  artifact: DecisionArtifact | null | undefined,
  options?: DecisionArtifactRenderOptions,
): string | null {
  if (artifact === null || artifact === undefined) {
    return null;
  }

  return cappedDecisionArtifactRender({
    artifact,
    options,
  }).content;
}

export function summarizeDecisionStateArtifactRender(
  artifact: DecisionArtifact | null | undefined,
  options?: DecisionArtifactRenderOptions,
): DecisionStateArtifactRenderSummary {
  if (artifact === null || artifact === undefined) {
    return {
      totalEntryCount: 0,
      activeEntryCount: 0,
      renderedEntryCount: 0,
      omittedEntryCount: 0,
      estimatedTokens: 0,
      renderedByKind: emptyDecisionArtifactKindCounts(),
      omittedByKind: emptyDecisionArtifactKindCounts(),
    };
  }

  return cappedDecisionArtifactRender({
    artifact,
    options,
  }).summary;
}
