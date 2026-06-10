import type { ActionRecord } from "../../memory/actions/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import type { RelationalSlot } from "../../memory/relational-slots/index.js";
import type { ReviewQueueItem } from "../../memory/review-queue/index.js";
import type { RetrievedEpisode } from "../../retrieval/index.js";
import type { StreamEntry } from "../../stream/index.js";
import type { EpisodeId, SessionId, StreamEntryId } from "../../util/ids.js";
import type { EvidenceLedgerSessionScope } from "./types.js";

export type ScopeResolver = {
  currentSessionId: SessionId;
  streamEntriesById: ReadonlyMap<string, StreamEntry>;
  streamOrderById: ReadonlyMap<string, number>;
  episodeScopesById: ReadonlyMap<string, EvidenceLedgerSessionScope>;
  episodeSourceStreamIdsById: ReadonlyMap<string, readonly string[]>;
};

export function scopeFromStreamIds(
  streamEntryIds: readonly StreamEntryId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  if (streamEntryIds === undefined || streamEntryIds.length === 0) {
    return "global";
  }

  let currentSessionCount = 0;
  let priorSessionCount = 0;

  for (const streamEntryId of streamEntryIds) {
    const entry = resolver.streamEntriesById.get(streamEntryId);

    if (entry === undefined) {
      return "global";
    }

    if (entry.session_id === resolver.currentSessionId) {
      currentSessionCount += 1;
    } else {
      priorSessionCount += 1;
    }
  }

  if (currentSessionCount === streamEntryIds.length) {
    return "current_session";
  }

  if (priorSessionCount === streamEntryIds.length) {
    return "prior_session";
  }

  return "global";
}

export function streamIndexFromSingleCurrentSessionStreamId(
  streamEntryIds: readonly StreamEntryId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
): number | undefined {
  if (streamEntryIds === undefined || streamEntryIds.length !== 1) {
    return undefined;
  }

  const streamEntryId = streamEntryIds[0];

  if (streamEntryId === undefined) {
    return undefined;
  }

  return resolver.streamOrderById.get(streamEntryId);
}

function scopeFromStreamEntries(
  entries: readonly StreamEntry[],
  currentSessionId: SessionId,
): EvidenceLedgerSessionScope {
  if (entries.length === 0) {
    return "global";
  }

  if (entries.every((entry) => entry.session_id === currentSessionId)) {
    return "current_session";
  }

  return entries.every((entry) => entry.session_id !== currentSessionId)
    ? "prior_session"
    : "global";
}

export function scopeFromEpisodeIds(
  episodeIds: readonly EpisodeId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  if (episodeIds === undefined || episodeIds.length === 0) {
    return "global";
  }

  let sawPriorSession = false;

  for (const episodeId of episodeIds) {
    const scope = resolver.episodeScopesById.get(episodeId);

    if (scope === "current_session") {
      return "current_session";
    }

    if (scope === "prior_session") {
      sawPriorSession = true;
    }
  }

  return sawPriorSession ? "prior_session" : "global";
}

function streamIdsFromEpisodeIds(
  episodeIds: readonly EpisodeId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
): string[] {
  if (episodeIds === undefined) {
    return [];
  }

  return episodeIds.flatMap((episodeId) => [
    ...(resolver.episodeSourceStreamIdsById.get(episodeId) ?? []),
  ]);
}

export function persistenceClassFromProvenance(
  input: {
    streamEntryIds?: readonly StreamEntryId[] | readonly string[];
    episodeIds?: readonly EpisodeId[] | readonly string[];
  },
  resolver: ScopeResolver,
) {
  return persistenceClassFromStreamIds(
    [...(input.streamEntryIds ?? []), ...streamIdsFromEpisodeIds(input.episodeIds, resolver)],
    resolver,
  );
}

export function combineScopes(
  scopes: readonly EvidenceLedgerSessionScope[],
): EvidenceLedgerSessionScope {
  if (scopes.some((scope) => scope === "current_session")) {
    return "current_session";
  }

  if (scopes.some((scope) => scope === "prior_session")) {
    return "prior_session";
  }

  return "global";
}

export function commitmentScope(
  commitment: CommitmentRecord,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return scopeFromStreamIds(commitment.source_stream_entry_ids ?? [], resolver);
}

export function actionScope(
  action: ActionRecord,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return combineScopes([
    scopeFromStreamIds(action.provenance_stream_entry_ids, resolver),
    scopeFromEpisodeIds(action.provenance_episode_ids, resolver),
  ]);
}

export function slotScope(
  slot: RelationalSlot,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return scopeFromStreamIds(
    [
      ...slot.evidence_stream_entry_ids,
      ...slot.contradicted_by_stream_entry_ids,
      ...slot.alternate_values.flatMap((alternate) => alternate.evidence_stream_entry_ids),
    ],
    resolver,
  );
}

export function reviewQueueScope(
  item: ReviewQueueItem,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return scopeFromStreamIds(reviewQueueStreamIds(item), resolver);
}

export function reviewQueueStreamIds(item: ReviewQueueItem): string[] {
  return Object.values(item.refs).flatMap((value) => {
    if (Array.isArray(value)) {
      return value.filter((candidate): candidate is string => typeof candidate === "string");
    }

    return typeof value === "string" ? [value] : [];
  });
}

export function persistenceClassFromStreamIds(
  streamEntryIds: readonly StreamEntryId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
) {
  if (streamEntryIds === undefined) {
    return {};
  }

  return streamEntryIds.some(
    (streamEntryId) =>
      resolver.streamEntriesById.get(streamEntryId)?.persistence_class === "assistant_self_report",
  )
    ? { persistence_class: "assistant_self_report" as const }
    : {};
}

export function buildEpisodeScopeMap(
  retrievedEpisodes: readonly RetrievedEpisode[],
  resolverBase: Omit<ScopeResolver, "episodeScopesById" | "episodeSourceStreamIdsById">,
): Map<string, EvidenceLedgerSessionScope> {
  const episodeScopes = new Map<string, EvidenceLedgerSessionScope>();

  for (const result of retrievedEpisodes) {
    const citationScope = scopeFromStreamEntries(
      result.citationChain,
      resolverBase.currentSessionId,
    );
    const sourceScope =
      citationScope === "global"
        ? scopeFromStreamIds(result.episode.source_stream_ids, {
            ...resolverBase,
            episodeScopesById: new Map(),
            episodeSourceStreamIdsById: new Map(),
          })
        : citationScope;

    episodeScopes.set(result.episode.id, sourceScope);
  }

  return episodeScopes;
}

export function buildEpisodeSourceStreamIdMap(
  retrievedEpisodes: readonly RetrievedEpisode[],
): Map<string, readonly string[]> {
  const episodeSourceStreamIds = new Map<string, readonly string[]>();

  for (const result of retrievedEpisodes) {
    episodeSourceStreamIds.set(result.episode.id, result.episode.source_stream_ids);
  }

  return episodeSourceStreamIds;
}
