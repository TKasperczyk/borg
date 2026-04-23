import { describe, expect, it } from "vitest";

import { parseStoredProvenance, toStoredProvenance } from "./provenance.js";

describe("provenance", () => {
  it("round-trips online provenance through stored columns", () => {
    const stored = toStoredProvenance({
      kind: "online",
      process: "reflector",
    });

    expect(stored).toEqual({
      provenance_kind: "online",
      provenance_episode_ids: "[]",
      provenance_process: "reflector",
    });
    expect(parseStoredProvenance(stored)).toEqual({
      kind: "online",
      process: "reflector",
    });
  });
});
