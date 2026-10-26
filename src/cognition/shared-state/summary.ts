import { estimatePromptTokens } from "../../util/token-estimate.js";
import {
  SHARED_STATE_ENTRY_KINDS,
  type SharedStateArtifact,
  type SharedStateEntry,
  type SharedStateEntryKind,
} from "../../memory/shared-state/index.js";
import {
  memoryDisclosurePayloadFields,
  sharedStateMemoryDisclosureLabel,
} from "../../memory/common/disclosure-serializers.js";
import { coercePositiveIntegerOrFallback } from "../../util/math.js";
import {
  activeSharedStateArtifactEntries,
  compareSharedStateArtifactEntriesByRecency,
  countSharedStateArtifactEntriesByKind,
  emptySharedStateKindCounts,
  onePerKindTokenDropFloor,
  selectSharedStateArtifactEntriesForRender,
  subtractSharedStateKindCounts,
  tokenDropIndexForKinds,
  type SharedStateKindCounts,
} from "./selection.js";
import { sharedStateKeyBucket, tokenizeStateKey } from "./state-key.js";
import { truncateSharedStateArtifactText } from "./render.js";

const DEFAULT_SHARED_STATE_PROMPT_SUMMARY_MAX_ENTRIES = {
  locked: 14,
  live: 8,
  low_salience_live: 2,
  dormant_live: 0,
  invalidated: 4,
  tentative: 2,
} as const satisfies Partial<Record<SharedStateEntryKind, number>>;
const SHARED_STATE_PROMPT_SUMMARY_CONFIGURABLE_KINDS = [
  "locked",
  "live",
  "low_salience_live",
  "dormant_live",
  "invalidated",
  "tentative",
] as const satisfies readonly SharedStateEntryKind[];
const DEFAULT_SHARED_STATE_PROMPT_SUMMARY_TOKEN_BUDGET = 6_000;
const DEFAULT_SHARED_STATE_PROMPT_SUMMARY_ENTRY_TEXT_TOKENS = 1_000;

export type SharedStatePromptSummaryOptions = {
  maxEntries?: Partial<Record<SharedStateEntryKind, number>>;
  summaryTokenBudget?: number;
  maxEntryTextTokens?: number;
};

type MemoryDisclosurePromptFields = ReturnType<typeof memoryDisclosurePayloadFields>;

export type SharedStatePromptSummaryEntry = {
  id: SharedStateEntry["id"];
  state_key: SharedStateEntry["state_key"];
  kind: SharedStateEntryKind;
  text: string;
  body_completeness: "full" | "truncated";
  disclosure: MemoryDisclosurePromptFields["disclosure"];
  disclosure_label: MemoryDisclosurePromptFields["disclosure_label"];
  owner_entity_id?: NonNullable<SharedStateEntry["owner_entity_id"]>;
  last_updated_stream_entry_id: SharedStateEntry["last_updated_stream_entry_ids"][number] | null;
  canonicalizes_ids_count: number;
};

export type SharedStatePromptSummarySupersededEntry = {
  id: SharedStateEntry["id"];
  text: string;
  body_completeness: "full";
  disclosure: MemoryDisclosurePromptFields["disclosure"];
  disclosure_label: MemoryDisclosurePromptFields["disclosure_label"];
  superseded_by_id: NonNullable<SharedStateEntry["superseded_by_id"]>;
};

export type SharedStatePromptSummary = {
  audience_entity_id: SharedStateArtifact["audience_entity_id"];
  record_version: SharedStateArtifact["record_version"];
  active_counts_by_kind: SharedStateKindCounts;
  active_entries: Record<SharedStateEntryKind, SharedStatePromptSummaryEntry[]>;
  active_entries_by_state_key: Record<string, SharedStatePromptSummaryEntry[]>;
  omitted_counts_by_kind: SharedStateKindCounts;
  recent_superseded: SharedStatePromptSummarySupersededEntry[];
};

/** Mechanical render facts for this artifact version, never inferred from model output. */
export type SharedStatePromptExposure = {
  audience_entity_id: SharedStateArtifact["audience_entity_id"];
  artifact_version: SharedStateArtifact["record_version"];
  rows: Array<{
    id: SharedStateEntry["id"];
    state_key: SharedStateEntry["state_key"];
    kind: SharedStateEntryKind;
    // Lengths use JavaScript UTF-16 code units, like the existing token estimator.
    original_chars: number;
    rendered_chars: number;
    completeness: "full" | "truncated" | "omitted";
  }>;
  omitted_entry_ids: SharedStateEntry["id"][];
  budget: {
    max_entries: Record<SharedStateEntryKind, number>;
    max_entry_text_tokens: number;
    summary_token_budget: number;
    initial_summary_token_estimate: number;
    summary_token_estimate: number;
    active_rows_dropped: number;
    superseded_rows_dropped: number;
    decision: "within_budget" | "selection_floor_exceeded";
  };
};

// The registry names every active key, including ids whose bodies the summary omits.
// text_visible_entry_ids records body presence, not completeness. Replacing locked text
// additionally requires a full body in this compile's SharedStatePromptExposure.
export type ExistingStateKeyRegistryEntry = {
  state_key: string;
  bucket: string;
  active_entry_ids: SharedStateEntry["id"][];
  active_entry_count: number;
  kinds: SharedStateEntryKind[];
  most_recent_update_at: number;
  most_recent_stream_entry_id: SharedStateEntry["last_updated_stream_entry_ids"][number] | null;
  // Null when no summary was supplied to compare against: absent evidence, not "nothing here is
  // visible". Empty means the summary gave this key no body. Locked entries then permit
  // non-text updates only; a visible but truncated body also does not permit replacement.
  text_visible_entry_ids: SharedStateEntry["id"][] | null;
};

function sharedStatePromptSummaryOptions(options: SharedStatePromptSummaryOptions = {}): {
  maxEntries: Record<SharedStateEntryKind, number>;
  summaryTokenBudget: number;
  maxEntryTextTokens: number;
} {
  const maxEntries = Object.fromEntries(
    SHARED_STATE_ENTRY_KINDS.map((kind) => [kind, 0]),
  ) as Record<SharedStateEntryKind, number>;

  for (const kind of SHARED_STATE_PROMPT_SUMMARY_CONFIGURABLE_KINDS) {
    const configured = options.maxEntries?.[kind];

    if (configured !== undefined && Number.isFinite(configured)) {
      maxEntries[kind] = Math.max(0, Math.floor(configured));
    } else {
      maxEntries[kind] = DEFAULT_SHARED_STATE_PROMPT_SUMMARY_MAX_ENTRIES[kind] ?? 0;
    }
  }

  return {
    maxEntries,
    summaryTokenBudget: coercePositiveIntegerOrFallback(
      options.summaryTokenBudget,
      DEFAULT_SHARED_STATE_PROMPT_SUMMARY_TOKEN_BUDGET,
    ),
    maxEntryTextTokens: coercePositiveIntegerOrFallback(
      options.maxEntryTextTokens,
      DEFAULT_SHARED_STATE_PROMPT_SUMMARY_ENTRY_TEXT_TOKENS,
    ),
  };
}

function emptySharedStatePromptSummaryEntries(): Record<
  SharedStateEntryKind,
  SharedStatePromptSummaryEntry[]
> {
  return Object.fromEntries(
    SHARED_STATE_ENTRY_KINDS.map((kind) => [kind, []]),
  ) as unknown as Record<SharedStateEntryKind, SharedStatePromptSummaryEntry[]>;
}

function sharedStateCanonicalizesIdCount(entry: SharedStateEntry): number {
  return (
    entry.canonicalizes.goal_ids.length +
    entry.canonicalizes.commitment_ids.length +
    entry.canonicalizes.action_ids.length +
    entry.canonicalizes.open_question_ids.length
  );
}

function lastUpdatedStreamEntryId(
  entry: SharedStateEntry,
): SharedStatePromptSummaryEntry["last_updated_stream_entry_id"] {
  return (
    entry.last_updated_stream_entry_ids[entry.last_updated_stream_entry_ids.length - 1] ?? null
  );
}

function stateKeyRegistryBucket(stateKey: string): string {
  const tokens = tokenizeStateKey(stateKey);

  if (tokens.length === 0) {
    return stateKey;
  }

  return tokens.slice(0, 2).join(".");
}

function stateKeyRegistryKinds(entries: readonly SharedStateEntry[]): SharedStateEntryKind[] {
  return SHARED_STATE_ENTRY_KINDS.filter((kind) => entries.some((entry) => entry.kind === kind));
}

export function buildExistingStateKeyRegistry(
  artifact: SharedStateArtifact | null | undefined,
  summary?: SharedStatePromptSummary | null,
): ExistingStateKeyRegistryEntry[] {
  const bodyCarriedEntryIds =
    summary === undefined || summary === null
      ? null
      : new Set(
          Object.values(summary.active_entries_by_state_key ?? {}).flatMap((entries) =>
            (entries ?? []).map((entry) => entry.id),
          ),
        );
  const groups = new Map<string, SharedStateEntry[]>();

  for (const entry of activeSharedStateArtifactEntries(artifact)) {
    if (entry.state_key === null) {
      continue;
    }

    groups.set(entry.state_key, [...(groups.get(entry.state_key) ?? []), entry]);
  }

  return [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([stateKey, entries]) => {
      const entriesByRecency = [...entries].sort(compareSharedStateArtifactEntriesByRecency);
      const mostRecent = entriesByRecency[0]!;

      return {
        state_key: stateKey,
        bucket: stateKeyRegistryBucket(stateKey),
        active_entry_ids: entriesByRecency.map((entry) => entry.id),
        active_entry_count: entries.length,
        kinds: stateKeyRegistryKinds(entries),
        most_recent_update_at: mostRecent.last_updated_at,
        most_recent_stream_entry_id: lastUpdatedStreamEntryId(mostRecent),
        text_visible_entry_ids:
          bodyCarriedEntryIds === null
            ? null
            : entriesByRecency
                .map((entry) => entry.id)
                .filter((entryId) => bodyCarriedEntryIds.has(entryId)),
      };
    });
}

function toSharedStatePromptSummaryEntry(
  entry: SharedStateEntry,
  maxEntryTextTokens: number,
): SharedStatePromptSummaryEntry {
  const full = estimatePromptTokens(entry.text) <= maxEntryTextTokens;
  const text = full ? entry.text : truncateSharedStateArtifactText(entry.text, maxEntryTextTokens);

  return {
    id: entry.id,
    state_key: entry.state_key,
    // Carry the lock/salience state on the row itself, including in the by-key view.
    kind: entry.kind,
    text,
    body_completeness: full ? "full" : "truncated",
    ...memoryDisclosurePayloadFields(sharedStateMemoryDisclosureLabel(entry)),
    ...(entry.owner_entity_id === null ? {} : { owner_entity_id: entry.owner_entity_id }),
    last_updated_stream_entry_id: lastUpdatedStreamEntryId(entry),
    canonicalizes_ids_count: sharedStateCanonicalizesIdCount(entry),
  };
}

function toSharedStatePromptSummarySupersededEntry(
  entry: SharedStateEntry,
): SharedStatePromptSummarySupersededEntry | null {
  if (entry.superseded_by_id === null) {
    return null;
  }

  return {
    id: entry.id,
    text: entry.text,
    body_completeness: "full",
    ...memoryDisclosurePayloadFields(sharedStateMemoryDisclosureLabel(entry)),
    superseded_by_id: entry.superseded_by_id,
  };
}

function selectedSharedStatePromptSummaryEntries(input: {
  activeEntries: readonly SharedStateEntry[];
  maxEntries: Record<SharedStateEntryKind, number>;
}): SharedStateEntry[] {
  const totalMaxEntries = SHARED_STATE_ENTRY_KINDS.reduce(
    (sum, kind) => sum + input.maxEntries[kind],
    0,
  );
  const cappedActiveEntries = SHARED_STATE_ENTRY_KINDS.flatMap((kind) =>
    input.activeEntries
      .filter((entry) => entry.kind === kind)
      .sort(compareSharedStateArtifactEntriesByRecency)
      .slice(0, input.maxEntries[kind]),
  );
  const selected = selectSharedStateArtifactEntriesForRender({
    entries: cappedActiveEntries,
    maxEntries: totalMaxEntries,
    reservedSlots: {
      live: input.maxEntries.live,
      invalidated: input.maxEntries.invalidated,
    },
    lockedMaxEntries: input.maxEntries.locked,
  });
  const counts = emptySharedStateKindCounts();

  return selected.filter((entry) => {
    if (counts[entry.kind] >= input.maxEntries[entry.kind]) {
      return false;
    }

    counts[entry.kind] += 1;
    return true;
  });
}

function buildSharedStateArtifactPromptSummaryFromEntries(input: {
  artifact: SharedStateArtifact;
  activeEntries: readonly SharedStateEntry[];
  selectedEntries: readonly SharedStateEntry[];
  recentSuperseded: readonly SharedStatePromptSummarySupersededEntry[];
  maxEntryTextTokens: number;
}): SharedStatePromptSummary {
  const activeEntriesByKind = emptySharedStatePromptSummaryEntries();
  const activeEntriesByStateKey: Record<string, SharedStatePromptSummaryEntry[]> = {};

  for (const entry of input.selectedEntries) {
    const summaryEntry = toSharedStatePromptSummaryEntry(entry, input.maxEntryTextTokens);

    activeEntriesByKind[entry.kind].push(summaryEntry);
    const key = sharedStateKeyBucket(entry.state_key);
    activeEntriesByStateKey[key] = [...(activeEntriesByStateKey[key] ?? []), summaryEntry];
  }

  return {
    audience_entity_id: input.artifact.audience_entity_id,
    record_version: input.artifact.record_version,
    active_counts_by_kind: countSharedStateArtifactEntriesByKind(input.activeEntries),
    active_entries: activeEntriesByKind,
    active_entries_by_state_key: Object.fromEntries(
      Object.entries(activeEntriesByStateKey).sort(([left], [right]) => left.localeCompare(right)),
    ),
    omitted_counts_by_kind: subtractSharedStateKindCounts(
      countSharedStateArtifactEntriesByKind(input.activeEntries),
      countSharedStateArtifactEntriesByKind(input.selectedEntries),
    ),
    recent_superseded: [...input.recentSuperseded],
  };
}

function sharedStatePromptSummaryTokenEstimate(summary: SharedStatePromptSummary): number {
  return estimatePromptTokens(JSON.stringify(summary));
}

function sharedStatePromptSummaryDropIndex(input: {
  entries: readonly SharedStateEntry[];
  activeCounts: SharedStateKindCounts;
}): number | null {
  const dropTentative = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["tentative", "dormant_live", "low_salience_live"],
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
    kinds: ["live", "locked"],
    minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
  });
}

export function buildSharedStateArtifactPromptSummary(
  artifact: SharedStateArtifact | null | undefined,
  options?: SharedStatePromptSummaryOptions,
): SharedStatePromptSummary | null {
  return buildSharedStateArtifactPromptSummaryWithExposure(artifact, options).summary;
}

export function buildSharedStateArtifactPromptSummaryWithExposure(
  artifact: SharedStateArtifact | null | undefined,
  options?: SharedStatePromptSummaryOptions,
): { summary: SharedStatePromptSummary | null; exposure: SharedStatePromptExposure | null } {
  if (artifact === null || artifact === undefined) {
    return { summary: null, exposure: null };
  }

  const normalizedOptions = sharedStatePromptSummaryOptions(options);
  const activeEntries = activeSharedStateArtifactEntries(artifact);
  let selectedEntries = selectedSharedStatePromptSummaryEntries({
    activeEntries,
    maxEntries: normalizedOptions.maxEntries,
  });
  let recentSuperseded = artifact.entries
    .filter((entry) => entry.superseded_by_id !== null)
    .sort(compareSharedStateArtifactEntriesByRecency)
    .slice(0, 5)
    .flatMap((entry) => {
      const summarized = toSharedStatePromptSummarySupersededEntry(entry);

      return summarized === null ? [] : [summarized];
    });
  let summary = buildSharedStateArtifactPromptSummaryFromEntries({
    artifact,
    activeEntries,
    selectedEntries,
    recentSuperseded,
    maxEntryTextTokens: normalizedOptions.maxEntryTextTokens,
  });
  const initialTokenEstimate = sharedStatePromptSummaryTokenEstimate(summary);
  const initialActiveCount = selectedEntries.length;
  const initialSupersededCount = recentSuperseded.length;

  while (
    sharedStatePromptSummaryTokenEstimate(summary) > normalizedOptions.summaryTokenBudget &&
    recentSuperseded.length > 0
  ) {
    recentSuperseded = recentSuperseded.slice(0, -1);
    summary = buildSharedStateArtifactPromptSummaryFromEntries({
      artifact,
      activeEntries,
      selectedEntries,
      recentSuperseded,
      maxEntryTextTokens: normalizedOptions.maxEntryTextTokens,
    });
  }

  while (
    sharedStatePromptSummaryTokenEstimate(summary) > normalizedOptions.summaryTokenBudget &&
    selectedEntries.length > 0
  ) {
    const dropIndex = sharedStatePromptSummaryDropIndex({
      entries: selectedEntries,
      activeCounts: countSharedStateArtifactEntriesByKind(activeEntries),
    });

    if (dropIndex === null) {
      break;
    }

    selectedEntries = [
      ...selectedEntries.slice(0, dropIndex),
      ...selectedEntries.slice(dropIndex + 1),
    ];
    summary = buildSharedStateArtifactPromptSummaryFromEntries({
      artifact,
      activeEntries,
      selectedEntries,
      recentSuperseded,
      maxEntryTextTokens: normalizedOptions.maxEntryTextTokens,
    });
  }

  // Read the final render, after both count caps and token-budget drops. The two active
  // summary views contain the same bodies; recent_superseded also carries full bodies.
  const renderedById = new Map(
    [...Object.values(summary.active_entries).flat(), ...summary.recent_superseded].map(
      (entry) => [entry.id, entry] as const,
    ),
  );
  const rows: SharedStatePromptExposure["rows"] = artifact.entries.map((entry) => {
    const rendered = renderedById.get(entry.id);
    return {
      id: entry.id,
      state_key: entry.state_key,
      kind: entry.kind,
      original_chars: entry.text.length,
      rendered_chars: rendered?.text.length ?? 0,
      completeness: rendered?.body_completeness ?? "omitted",
    };
  });
  const finalTokenEstimate = sharedStatePromptSummaryTokenEstimate(summary);

  return {
    summary,
    exposure: {
      audience_entity_id: artifact.audience_entity_id,
      artifact_version: artifact.record_version,
      rows,
      omitted_entry_ids: rows.filter((row) => row.completeness === "omitted").map((row) => row.id),
      budget: {
        max_entries: normalizedOptions.maxEntries,
        max_entry_text_tokens: normalizedOptions.maxEntryTextTokens,
        summary_token_budget: normalizedOptions.summaryTokenBudget,
        initial_summary_token_estimate: initialTokenEstimate,
        summary_token_estimate: finalTokenEstimate,
        active_rows_dropped: initialActiveCount - selectedEntries.length,
        superseded_rows_dropped: initialSupersededCount - recentSuperseded.length,
        decision:
          finalTokenEstimate <= normalizedOptions.summaryTokenBudget
            ? "within_budget"
            : "selection_floor_exceeded",
      },
    },
  };
}
