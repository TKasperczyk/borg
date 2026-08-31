import { describe, expect, it } from "vitest";

import { createStreamEntryId } from "../../util/ids.js";
import { parseSourceStreamEntryIds } from "./source-trust.js";

describe("parseSourceStreamEntryIds", () => {
  it("discards every eligible co-citation when one cited id is outside the allowed set", () => {
    const eligible = createStreamEntryId();
    const alsoEligible = createStreamEntryId();
    const outsideLedgerWindow = createStreamEntryId();

    const result = parseSourceStreamEntryIds(
      [eligible, outsideLedgerWindow, alsoEligible],
      new Set([eligible, alsoEligible]),
      undefined,
    );

    expect(result.reason).toBe("disallowed_source_stream_entry_id");
    expect(result.rejectedStreamEntryId).toBe(outsideLedgerWindow);
    expect(result.streamEntryIds).toEqual([]);
  });

  it("leaves an allowed-set rejection without a source trust reason", () => {
    const eligible = createStreamEntryId();
    const outsideLedgerWindow = createStreamEntryId();

    const result = parseSourceStreamEntryIds(
      [outsideLedgerWindow],
      new Set([eligible]),
      undefined,
    );

    expect(result.reason).toBe("disallowed_source_stream_entry_id");
    // The compile trace projects only rejections carrying sourceTrustReason, so an
    // allowed-set rejection reaches the trace as a bare reason string with no id.
    expect(result.sourceTrustReason).toBeUndefined();
  });

  it("reports source trust refusal ahead of allowed-set membership", () => {
    const quarantined = createStreamEntryId();
    const eligible = createStreamEntryId();

    const result = parseSourceStreamEntryIds([quarantined], new Set([eligible]), (streamEntryId) =>
      streamEntryId === quarantined ? { allowed: false, reason: "quarantined" } : { allowed: true },
    );

    expect(result.reason).toBe("quarantined_source_stream_entry_id");
    expect(result.sourceTrustReason).toBe("quarantined");
  });

  it("accepts every deduplicated citation when no allowed set constrains the compile", () => {
    const first = createStreamEntryId();
    const second = createStreamEntryId();

    expect(parseSourceStreamEntryIds([first, second, first], null, undefined)).toEqual({
      streamEntryIds: [first, second],
      reason: null,
    });
    expect(parseSourceStreamEntryIds([], null, undefined)).toEqual({
      streamEntryIds: [],
      reason: "missing_citation",
    });
  });
});
