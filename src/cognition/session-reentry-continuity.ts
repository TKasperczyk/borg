import {
  SHARED_STATE_ENTRY_KINDS,
  type SharedStateArtifact,
  type SharedStateEntry,
  type SharedStateEntryKind,
} from "../memory/decision-artifacts/index.js";
import type { EntityId, StreamEntryId } from "../util/ids.js";
import { renderTaggedPromptBlock } from "./deliberation/prompt/sections.js";
import { TRUSTED_GUIDANCE_PREAMBLE } from "./prompts/base-identity.js";
import {
  activeSharedStateArtifactEntries,
  compareSharedStateArtifactEntriesByRecency,
  countSharedStateArtifactEntriesByKind,
  emptySharedStateKindCounts,
  type SharedStateKindCounts,
} from "./shared-state/selection.js";
import { countSharedStateEntriesByKey, sharedStateKeyBucket } from "./shared-state/state-key.js";

export const SESSION_REENTRY_CONTINUITY_TAG = "borg_session_reentry_continuity";

export type SessionReentryContinuityStatus =
  | "not_user_turn"
  | "not_first_user_turn"
  | "missing_audience"
  | "blank_audience"
  | "rendered";

export type SessionReentryContinuitySummary = {
  audienceEntityId: EntityId | null;
  status: SessionReentryContinuityStatus;
  activeEntryCount: number;
  activeKeyedEntryCount: number;
  activeLegacyEntryCount: number;
  activeStateKeyCount: number;
  activeCountsByKind: SharedStateKindCounts;
  activeEntriesByKey: Record<string, number>;
  mostRecentUpdate: {
    entryId: SharedStateEntry["id"];
    stateKey: string;
    kind: SharedStateEntryKind;
    lastUpdatedAt: number;
    lastUpdatedStreamEntryId: StreamEntryId | null;
  } | null;
};

export type SessionReentryContinuityPrompt = {
  promptSection: string | null;
  summary: SessionReentryContinuitySummary;
};

function latestStreamEntryId(entry: SharedStateEntry): StreamEntryId | null {
  return entry.last_updated_stream_entry_ids.at(-1) ?? null;
}

function mostRecentSharedStateEntry(
  entries: readonly SharedStateEntry[],
): SessionReentryContinuitySummary["mostRecentUpdate"] {
  const mostRecent = [...entries].sort(compareSharedStateArtifactEntriesByRecency)[0];

  if (mostRecent === undefined) {
    return null;
  }

  return {
    entryId: mostRecent.id,
    stateKey: sharedStateKeyBucket(mostRecent.state_key),
    kind: mostRecent.kind,
    lastUpdatedAt: mostRecent.last_updated_at,
    lastUpdatedStreamEntryId: latestStreamEntryId(mostRecent),
  };
}

function formatKindCounts(counts: SharedStateKindCounts): string {
  return SHARED_STATE_ENTRY_KINDS.map((kind) => `${kind}=${counts[kind]}`).join(" ");
}

function kindCountsForKey(entries: readonly SharedStateEntry[]): SharedStateKindCounts {
  return countSharedStateArtifactEntriesByKind(entries);
}

function stateKeyBucketSource(entries: readonly SharedStateEntry[]): string {
  const keyedCount = entries.filter((entry) => entry.state_key !== null).length;

  if (keyedCount === 0) {
    return "unkeyed_legacy_state";
  }

  return keyedCount === entries.length ? "keyed_thread" : "mixed_keyed_and_legacy";
}

function stateKeyLines(entries: readonly SharedStateEntry[]): string[] {
  const groups = new Map<string, SharedStateEntry[]>();

  for (const entry of entries) {
    const key = sharedStateKeyBucket(entry.state_key);
    groups.set(key, [...(groups.get(key) ?? []), entry]);
  }

  return [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([stateKey, groupEntries]) => {
      const mostRecent = mostRecentSharedStateEntry(groupEntries);
      const latestRef =
        mostRecent?.lastUpdatedStreamEntryId === null ||
        mostRecent?.lastUpdatedStreamEntryId === undefined
          ? "null"
          : mostRecent.lastUpdatedStreamEntryId;
      const latestAt = mostRecent?.lastUpdatedAt ?? "null";

      return `- state_key_bucket=${stateKey} bucket_source=${stateKeyBucketSource(
        groupEntries,
      )} entries=${groupEntries.length} kinds=${formatKindCounts(
        kindCountsForKey(groupEntries),
      )} most_recent_update_at=${latestAt} most_recent_ref=${latestRef}`;
    });
}

function renderSessionReentryContinuityContent(
  summary: SessionReentryContinuitySummary,
  activeEntries: readonly SharedStateEntry[],
): string {
  const mostRecent = summary.mostRecentUpdate;
  const mostRecentLine =
    mostRecent === null
      ? "most_recent_update=null"
      : [
          `most_recent_update_at=${mostRecent.lastUpdatedAt}`,
          `most_recent_entry_id=${mostRecent.entryId}`,
          `most_recent_state_key=${mostRecent.stateKey}`,
          `most_recent_kind=${mostRecent.kind}`,
          `most_recent_ref=${mostRecent.lastUpdatedStreamEntryId ?? "null"}`,
        ].join(" ");

  return [
    "SessionReentryContinuity: this is the first user-origin turn of a new session for this audience.",
    "Instruction: the audience state is not blank; surface existing state before accepting fresh-start framing; only treat as blank if the user explicitly requests a reset.",
    `audience_entity_id=${summary.audienceEntityId ?? "null"}`,
    `matched_state_key_buckets=all_active_state_key_buckets active_state_key_bucket_count=${summary.activeStateKeyCount}`,
    `active_entry_count=${summary.activeEntryCount} active_keyed_entry_count=${summary.activeKeyedEntryCount} active_legacy_unkeyed_entry_count=${summary.activeLegacyEntryCount}`,
    `active_counts_by_kind=${formatKindCounts(summary.activeCountsByKind)}`,
    mostRecentLine,
    "state_key_buckets:",
    ...stateKeyLines(activeEntries),
  ].join("\n");
}

function baseSummary(input: {
  audienceEntityId: EntityId | null;
  status: SessionReentryContinuityStatus;
  activeEntries: readonly SharedStateEntry[];
}): SessionReentryContinuitySummary {
  const activeKeyedEntryCount = input.activeEntries.filter(
    (entry) => entry.state_key !== null,
  ).length;

  return {
    audienceEntityId: input.audienceEntityId,
    status: input.status,
    activeEntryCount: input.activeEntries.length,
    activeKeyedEntryCount,
    activeLegacyEntryCount: input.activeEntries.length - activeKeyedEntryCount,
    activeStateKeyCount: Object.keys(countSharedStateEntriesByKey(input.activeEntries)).length,
    activeCountsByKind:
      input.activeEntries.length === 0
        ? emptySharedStateKindCounts()
        : countSharedStateArtifactEntriesByKind(input.activeEntries),
    activeEntriesByKey: countSharedStateEntriesByKey(input.activeEntries),
    mostRecentUpdate: mostRecentSharedStateEntry(input.activeEntries),
  };
}

export function buildSessionReentryContinuityPrompt(input: {
  isUserTurn: boolean;
  priorUserTurnCount: number;
  audienceEntityId: EntityId | null;
  artifact: SharedStateArtifact | null;
}): SessionReentryContinuityPrompt {
  const activeEntries = activeSharedStateArtifactEntries(input.artifact);

  if (!input.isUserTurn) {
    return {
      promptSection: null,
      summary: baseSummary({
        audienceEntityId: input.audienceEntityId,
        status: "not_user_turn",
        activeEntries,
      }),
    };
  }

  if (input.priorUserTurnCount > 0) {
    return {
      promptSection: null,
      summary: baseSummary({
        audienceEntityId: input.audienceEntityId,
        status: "not_first_user_turn",
        activeEntries,
      }),
    };
  }

  if (input.audienceEntityId === null) {
    return {
      promptSection: null,
      summary: baseSummary({
        audienceEntityId: input.audienceEntityId,
        status: "missing_audience",
        activeEntries,
      }),
    };
  }

  if (activeEntries.length === 0) {
    return {
      promptSection: null,
      summary: baseSummary({
        audienceEntityId: input.audienceEntityId,
        status: "blank_audience",
        activeEntries,
      }),
    };
  }

  const summary = baseSummary({
    audienceEntityId: input.audienceEntityId,
    status: "rendered",
    activeEntries,
  });

  return {
    promptSection: renderTaggedPromptBlock(TRUSTED_GUIDANCE_PREAMBLE, [
      {
        tag: SESSION_REENTRY_CONTINUITY_TAG,
        content: renderSessionReentryContinuityContent(summary, activeEntries),
      },
    ]),
    summary,
  };
}
