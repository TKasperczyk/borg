import {
  type SharedStateSourceTrustRejectionReason,
  type SharedStateSourceTrustValidator,
} from "../../memory/decision-artifacts/index.js";
import { streamEntryIdHelpers, type StreamEntryId } from "../../util/ids.js";
import type { PatchRejection } from "./schema.js";

export function parseSourceStreamEntryIds(
  values: readonly string[],
  allowedSourceStreamEntryIds: ReadonlySet<StreamEntryId> | null,
  sourceTrustValidator: SharedStateSourceTrustValidator | undefined,
): {
  streamEntryIds: StreamEntryId[];
  reason: PatchRejection["reason"] | null;
  rejectedStreamEntryId?: string;
  sourceTrustReason?: SharedStateSourceTrustRejectionReason | "unknown";
} {
  if (values.length === 0) {
    return { streamEntryIds: [], reason: "missing_citation" };
  }

  const streamEntryIds: StreamEntryId[] = [];

  for (const value of values) {
    if (!streamEntryIdHelpers.is(value)) {
      return {
        streamEntryIds: [],
        reason: "invalid_source_stream_entry_id",
        rejectedStreamEntryId: value,
      };
    }

    const trust = sourceTrustValidator?.(value);

    if (trust?.allowed === false) {
      return {
        streamEntryIds: [],
        reason:
          trust.reason === "quarantined"
            ? "quarantined_source_stream_entry_id"
            : "inactive_source_stream_entry_id",
        rejectedStreamEntryId: value,
        sourceTrustReason: trust.reason ?? "unknown",
      };
    }

    if (allowedSourceStreamEntryIds !== null && !allowedSourceStreamEntryIds.has(value)) {
      return {
        streamEntryIds: [],
        reason: "disallowed_source_stream_entry_id",
        rejectedStreamEntryId: value,
      };
    }

    if (!streamEntryIds.some((entryId) => entryId === value)) {
      streamEntryIds.push(value);
    }
  }

  return { streamEntryIds, reason: null };
}
