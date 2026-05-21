import type { StreamEntryIndexRepository, StreamReader } from "../../stream/index.js";
import type { SessionId, StreamEntryId } from "../../util/ids.js";
import type {
  SemanticRelationshipEvidenceStreamEntryTrustResult,
  SemanticRelationshipEvidenceStreamEntryTrustValidator,
} from "./extractor.js";

export function createUserStreamEntryRelationshipEvidenceTrustValidator(options: {
  entryIndex: Pick<StreamEntryIndexRepository, "lookup">;
  createStreamReader: (sessionId: SessionId) => StreamReader;
}): SemanticRelationshipEvidenceStreamEntryTrustValidator {
  const cache = new Map<StreamEntryId, SemanticRelationshipEvidenceStreamEntryTrustResult>();

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
