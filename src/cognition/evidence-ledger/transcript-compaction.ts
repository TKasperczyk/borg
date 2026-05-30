import { type TranscriptStreamEntry } from "../../stream/index.js";
import { estimatePromptTokens, stringifyPromptContent } from "../../util/token-estimate.js";
import type { SpeakerEntityRepository } from "../speaker-tags.js";
import {
  actorForStreamEntry,
  optionalStateMetadata,
  replyTargetStateMetadata,
  streamPersistenceClass,
  transcriptState,
} from "./entry-metadata.js";
import { TRANSCRIPT_TRUST_RANK } from "./section-buckets.js";
import type { ScopeResolver } from "./scope-resolver.js";
import type { EvidenceLedgerEntry } from "./types.js";

const TRANSCRIPT_RAW_TAIL_MIN_ENTRIES = 8;
const TRANSCRIPT_RAW_TAIL_BUDGET_FRACTION = 0.6;

export type TranscriptCompactionResult = {
  entries: EvidenceLedgerEntry[];
  // Stream IDs rendered as raw transcript rows. Retrieved raw-stream evidence
  // dedupes against this exact set, so compacted stream IDs are intentionally absent.
  rawStreamIds: Set<string>;
  compacted: boolean;
  originalTokenEstimate: number;
  compactedEntryCount: number;
  rawPreservedUserEntryCount: number;
};

function estimateTranscriptTokens(entries: readonly TranscriptStreamEntry[]): number {
  if (entries.length === 0) {
    return 0;
  }

  return estimatePromptTokens(
    entries.map((entry) => stringifyPromptContent(entry.content)).join("\n"),
  );
}

function estimateTranscriptEntryTokens(entry: TranscriptStreamEntry): number {
  return estimatePromptTokens(stringifyPromptContent(entry.content));
}

function transcriptRawEntry(
  entry: TranscriptStreamEntry,
  resolver: ScopeResolver,
  entityRepository: SpeakerEntityRepository | undefined,
): EvidenceLedgerEntry {
  const stateMetadata = replyTargetStateMetadata(entry, entityRepository);

  return {
    id: `current_session_stream:${entry.id}`,
    source_type: "current_session_stream",
    session_scope: "current_session",
    actor: actorForStreamEntry(entry),
    trust_rank: TRANSCRIPT_TRUST_RANK,
    text: stringifyPromptContent(entry.content),
    stream_index: resolver.streamOrderById.get(entry.id),
    state: transcriptState(entry),
    ...optionalStateMetadata(stateMetadata),
    taint: "none",
    ...streamPersistenceClass(entry),
  };
}

function compactedTranscriptRunEntry(
  entries: readonly TranscriptStreamEntry[],
  resolver: ScopeResolver,
): EvidenceLedgerEntry {
  const first = entries[0] as TranscriptStreamEntry;
  const last = entries[entries.length - 1] as TranscriptStreamEntry;
  const streamIds = entries.map((entry) => entry.id).join(", ");
  const firstIndex = resolver.streamOrderById.get(first.id);
  const lastIndex = resolver.streamOrderById.get(last.id);
  const indexRange =
    firstIndex === undefined || lastIndex === undefined ? "unknown" : `${firstIndex}..${lastIndex}`;

  return {
    id: `current_session_compacted:${first.id}`,
    source_type: "system_metadata",
    session_scope: "current_session",
    actor: "system",
    trust_rank: TRANSCRIPT_TRUST_RANK,
    text: `Earlier assistant/system transcript entries compacted: entries=${entries.length}, stream_indexes=${indexRange}, stream_ids=${streamIds}.`,
    stream_index: firstIndex,
    state: "compacted",
    taint: "none",
  };
}

function compactedCurrentUserTranscriptEntry(
  entry: TranscriptStreamEntry,
  resolver: ScopeResolver,
): EvidenceLedgerEntry {
  return {
    id: `current_session_compacted_current_user:${entry.id}`,
    source_type: "system_metadata",
    session_scope: "current_session",
    actor: "system",
    trust_rank: TRANSCRIPT_TRUST_RANK,
    text: `Current user transcript duplicate compacted; full text is rendered in section 1 as current_user_message:${entry.id}.`,
    stream_index: resolver.streamOrderById.get(entry.id),
    state: "compacted",
    taint: "none",
  };
}

function rawTailStreamIds(
  entries: readonly TranscriptStreamEntry[],
  budget: number,
  currentUserEntryId: string | undefined,
  currentUserEntryIds: ReadonlySet<string>,
): Set<string> {
  const tailBudget = Math.max(1, Math.floor(budget * TRANSCRIPT_RAW_TAIL_BUDGET_FRACTION));
  const ids = new Set<string>();
  let tokens = 0;

  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index];

    if (
      entry === undefined ||
      entry.id === currentUserEntryId ||
      currentUserEntryIds.has(entry.id)
    ) {
      continue;
    }

    const entryTokens = estimateTranscriptEntryTokens(entry);

    if (ids.size >= TRANSCRIPT_RAW_TAIL_MIN_ENTRIES && tokens + entryTokens > tailBudget) {
      break;
    }

    ids.add(entry.id);
    tokens += entryTokens;
  }

  return ids;
}

function shouldKeepRawCompactedTranscriptEntry(
  entry: TranscriptStreamEntry,
  tailIds: ReadonlySet<string>,
  currentUserEntryId: string | undefined,
  currentUserEntryIds: ReadonlySet<string>,
): boolean {
  if (entry.id === currentUserEntryId || currentUserEntryIds.has(entry.id)) {
    return false;
  }

  return (
    tailIds.has(entry.id) ||
    entry.kind === "user_msg" ||
    entry.persistence_class === "assistant_self_report"
  );
}

export function compactTranscriptEntries(input: {
  entries: readonly TranscriptStreamEntry[];
  budget: number;
  currentUserEntryId?: string;
  currentUserEntryIds?: readonly string[];
  resolver: ScopeResolver;
  entityRepository?: SpeakerEntityRepository;
}): TranscriptCompactionResult {
  const currentUserEntryIds = new Set(input.currentUserEntryIds ?? []);
  const transcriptTokens = estimateTranscriptTokens(input.entries);

  if (transcriptTokens <= input.budget) {
    const rawEntries = input.entries.filter((entry) => !currentUserEntryIds.has(entry.id));

    return {
      entries: rawEntries.map((entry) =>
        transcriptRawEntry(entry, input.resolver, input.entityRepository),
      ),
      rawStreamIds: new Set(rawEntries.map((entry) => entry.id)),
      compacted: false,
      originalTokenEstimate: transcriptTokens,
      compactedEntryCount: 0,
      rawPreservedUserEntryCount: rawEntries.filter((entry) => entry.kind === "user_msg").length,
    };
  }

  const tailIds = rawTailStreamIds(
    input.entries,
    input.budget,
    input.currentUserEntryId,
    currentUserEntryIds,
  );
  const renderedEntries: EvidenceLedgerEntry[] = [];
  const rawStreamIds = new Set<string>();
  let compactedRun: TranscriptStreamEntry[] = [];
  let compactedEntryCount = 0;
  let rawPreservedUserEntryCount = 0;

  const flushCompactedRun = () => {
    if (compactedRun.length === 0) {
      return;
    }

    renderedEntries.push(compactedTranscriptRunEntry(compactedRun, input.resolver));
    compactedEntryCount += compactedRun.length;
    compactedRun = [];
  };

  for (const entry of input.entries) {
    if (
      shouldKeepRawCompactedTranscriptEntry(
        entry,
        tailIds,
        input.currentUserEntryId,
        currentUserEntryIds,
      )
    ) {
      flushCompactedRun();
      renderedEntries.push(transcriptRawEntry(entry, input.resolver, input.entityRepository));
      rawStreamIds.add(entry.id);
      if (entry.kind === "user_msg") {
        rawPreservedUserEntryCount += 1;
      }
      continue;
    }

    if (currentUserEntryIds.has(entry.id)) {
      flushCompactedRun();
      compactedEntryCount += 1;
      continue;
    }

    if (entry.id === input.currentUserEntryId) {
      flushCompactedRun();
      renderedEntries.push(compactedCurrentUserTranscriptEntry(entry, input.resolver));
      compactedEntryCount += 1;
      continue;
    }

    compactedRun.push(entry);
  }

  flushCompactedRun();

  return {
    entries: renderedEntries,
    rawStreamIds,
    compacted: true,
    originalTokenEstimate: transcriptTokens,
    compactedEntryCount,
    rawPreservedUserEntryCount,
  };
}
