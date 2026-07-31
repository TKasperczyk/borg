import { describe, expect, it } from "vitest";

import {
  effectiveCommitmentCriticalDomain,
  effectiveCommitmentEnforcementClass,
  legacyCommitmentSchema,
  commitmentPatchSchema,
} from "./types.js";
import { createCommitmentId } from "../../util/ids.js";

describe("commitment patch schema", () => {
  it("rejects immutable commitment fields in patches", () => {
    expect(() =>
      commitmentPatchSchema.parse({
        created_at: 123,
      }),
    ).toThrow();
    expect(() =>
      commitmentPatchSchema.parse({
        updated_at: 123,
      }),
    ).toThrow();
  });
});

describe("legacy commitment enforcement normalization", () => {
  it("falls back from kind when enforcement fields are missing", () => {
    expect(
      effectiveCommitmentEnforcementClass({
        kind: "boundary",
      }),
    ).toBe("critical");
    expect(
      effectiveCommitmentCriticalDomain({
        kind: "boundary",
      }),
    ).toBe("audience_scope");
    expect(
      effectiveCommitmentEnforcementClass({
        kind: "process_norm",
      }),
    ).toBe("advisory");
  });

  it("parses legacy commitment values by defaulting enforcement fields", () => {
    const parsed = legacyCommitmentSchema.parse({
      id: createCommitmentId(),
      type: "boundary",
      kind: "boundary",
      directive_family: "legacy_boundary",
      closure_pressure_relevance: "neutral",
      directive: "Do not disclose the incident.",
      priority: 9,
      made_to_entity: null,
      restricted_audience: null,
      about_entity: null,
      provenance: { kind: "manual" },
      created_at: 1_000,
      expires_at: null,
      expired_at: null,
      revoked_at: null,
      revoked_reason: null,
      revoke_provenance: null,
      superseded_by: null,
      last_reinforced_at: 1_000,
    });

    expect(parsed.enforcement_class).toBe("critical");
    expect(parsed.critical_domain).toBe("audience_scope");
    expect(parsed.updated_at).toBe(parsed.created_at);
  });
});
