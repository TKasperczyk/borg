import type { StreamEntry, StreamEntryIndexRepository, StreamReader } from "../stream/index.js";
import type { SessionId, StreamEntryId } from "../util/ids.js";

export type RelationshipEvidenceStreamEntryTrustResult = {
  allowed: boolean;
  reason?: "missing" | "not_user_msg" | "untrusted" | "unavailable";
};

export type RelationshipEvidenceStreamEntryTrustValidator = (
  streamEntryId: StreamEntryId,
) =>
  | RelationshipEvidenceStreamEntryTrustResult
  | Promise<RelationshipEvidenceStreamEntryTrustResult>;

export type SyncRelationshipEvidenceStreamEntryTrustValidator = (
  streamEntryId: StreamEntryId,
) => RelationshipEvidenceStreamEntryTrustResult;

export function createUserStreamEntryRelationshipEvidenceTrustValidator(options: {
  entryIndex: Pick<StreamEntryIndexRepository, "lookup">;
  createStreamReader: (sessionId: SessionId) => StreamReader;
}): RelationshipEvidenceStreamEntryTrustValidator {
  const cache = new Map<StreamEntryId, RelationshipEvidenceStreamEntryTrustResult>();

  return async (streamEntryId) => {
    const cached = cache.get(streamEntryId);

    if (cached !== undefined) {
      return cached;
    }

    const indexed = options.entryIndex.lookup(streamEntryId);

    if (indexed === null) {
      const result = { allowed: false, reason: "missing" } as const;
      cache.set(streamEntryId, result);
      return result;
    }

    for await (const entry of options.createStreamReader(indexed.session_id).iterate({
      sinceTs: indexed.timestamp,
      untilTs: indexed.timestamp,
    })) {
      if (entry.id !== streamEntryId) {
        continue;
      }

      const result =
        entry.kind === "user_msg"
          ? ({ allowed: true } as const)
          : ({ allowed: false, reason: "not_user_msg" } as const);
      cache.set(streamEntryId, result);
      return result;
    }

    const result = { allowed: false, reason: "missing" } as const;
    cache.set(streamEntryId, result);
    return result;
  };
}

export function createLoadedUserStreamEntryRelationshipEvidenceTrustValidator(options: {
  entries: readonly Pick<StreamEntry, "id" | "kind">[];
  isTrusted?: (streamEntryId: StreamEntryId) => boolean;
}): SyncRelationshipEvidenceStreamEntryTrustValidator {
  const entriesById = new Map(options.entries.map((entry) => [entry.id, entry]));

  return (streamEntryId) => {
    if (options.isTrusted?.(streamEntryId) === false) {
      return {
        allowed: false,
        reason: "untrusted",
      };
    }

    const entry = entriesById.get(streamEntryId);

    if (entry === undefined) {
      return {
        allowed: false,
        reason: "missing",
      };
    }

    return entry.kind === "user_msg"
      ? { allowed: true }
      : {
          allowed: false,
          reason: "not_user_msg",
        };
  };
}
