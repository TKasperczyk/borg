import { describe, expect, it } from "vitest";

import {
  emitManifestResponseSchema,
  flatEmitManifestResponseSchema,
  MANIFEST_STRUCTURED_OUTPUT_FORMAT,
  manifestClaimSchema,
  tightenManifestResponse,
  type FlatManifestClaim,
  type ManifestClaim,
} from "./manifest-schema.js";

const evidenceRef = {
  id: "current_user_message:strm_aaaaaaaaaaaaaaaa",
  source_type: "current_user_message",
} as const;

describe("manifest schema", () => {
  it.each([
    {
      kind: "discourse_only",
      rendered_span: "I can do that.",
    },
    {
      kind: "user_fact",
      rendered_span: "You prefer direct answers.",
      exact_values: ["direct answers"],
      evidence: [evidenceRef],
      confidence: "direct",
      scope_disclosure_span: "You prefer",
    },
    {
      kind: "prior_callback",
      rendered_span: "Earlier in this session, you asked about Sprint 3.",
      callback_scope: "current_session_prior",
      evidence: [evidenceRef],
      scope_disclosure_span: "Earlier in this session",
    },
    {
      kind: "action_state",
      rendered_span: "I have completed the check.",
      action_record_id: "act_aaaaaaaaaaaaaaaa",
      asserted_state: "completed",
      evidence: [evidenceRef],
    },
    {
      kind: "slot_fact",
      rendered_span: "The preferred style is concise.",
      slot_id: "slot_aaaaaaaaaaaaaaaa",
      exact_values: ["concise"],
      evidence: [evidenceRef],
    },
    {
      kind: "agent_self_provenance",
      rendered_span: "I found this in retrieved memory.",
      evidence: [evidenceRef],
    },
    {
      kind: "self_report",
      rendered_span: "The gap feels like a discontinuity with a remembered edge.",
      persistence_class: "assistant_self_report",
    },
    {
      kind: "interpretation",
      rendered_span: "That sounds like a preference, not a commitment.",
      evidence: [evidenceRef],
      confidence: "medium",
      persistence_allowed: false,
    },
    {
      kind: "hedge",
      rendered_span: "It looks like",
    },
  ] satisfies readonly ManifestClaim[])("parses $kind claims", (claim) => {
    expect(manifestClaimSchema.safeParse(claim).success).toBe(true);
  });

  it("parses the top-level manifest response shape", () => {
    const parsed = emitManifestResponseSchema.safeParse({
      final_text: "It looks like the import is missing.",
      discourse_act: "answer",
      claims: [
        {
          kind: "hedge",
          rendered_span: "It looks like",
        },
      ],
    });

    expect(parsed.success).toBe(true);
  });

  it("rejects unknown claim discriminants", () => {
    const parsed = manifestClaimSchema.safeParse({
      kind: "new_claim_kind",
      rendered_span: "unsupported",
    });

    expect(parsed.success).toBe(false);
  });

  it("rejects missing required fields for a known claim kind", () => {
    const parsed = manifestClaimSchema.safeParse({
      kind: "interpretation",
      rendered_span: "This seems important.",
      evidence: [evidenceRef],
      confidence: "high",
    });

    expect(parsed.success).toBe(false);
  });

  it("rejects evidence refs with non-ledger source types", () => {
    const parsed = manifestClaimSchema.safeParse({
      kind: "user_fact",
      rendered_span: "You prefer direct answers.",
      exact_values: ["direct answers"],
      evidence: [
        {
          id: "current_user_message:strm_aaaaaaaaaaaaaaaa",
          source_type: "made_up_source",
        },
      ],
      confidence: "direct",
    });

    expect(parsed.success).toBe(false);
  });

  it.each([
    {
      kind: "user_fact",
      claim: {
        kind: "user_fact",
        rendered_span: "You prefer direct answers.",
        exact_values: ["direct answers"],
        evidence: [],
        confidence: "direct",
      },
    },
    {
      kind: "prior_callback",
      claim: {
        kind: "prior_callback",
        rendered_span: "Earlier, you asked about Sprint 3.",
        callback_scope: "current_session_prior",
        evidence: [],
      },
    },
    {
      kind: "action_state",
      claim: {
        kind: "action_state",
        rendered_span: "I have completed the check.",
        action_record_id: "act_aaaaaaaaaaaaaaaa",
        asserted_state: "completed",
        evidence: [],
      },
    },
    {
      kind: "slot_fact",
      claim: {
        kind: "slot_fact",
        rendered_span: "The preferred style is concise.",
        slot_id: "slot_aaaaaaaaaaaaaaaa",
        exact_values: ["concise"],
        evidence: [],
      },
    },
    {
      kind: "agent_self_provenance",
      claim: {
        kind: "agent_self_provenance",
        rendered_span: "I found this in retrieved memory.",
        evidence: [],
      },
    },
  ])("rejects empty evidence arrays for $kind", ({ claim }) => {
    expect(manifestClaimSchema.safeParse(claim).success).toBe(false);
  });

  it.each([
    {
      kind: "user_fact",
      claim: {
        kind: "user_fact",
        rendered_span: "You prefer direct answers.",
        exact_values: [],
        evidence: [evidenceRef],
        confidence: "direct",
      },
    },
    {
      kind: "slot_fact",
      claim: {
        kind: "slot_fact",
        rendered_span: "The preferred style is concise.",
        slot_id: "slot_aaaaaaaaaaaaaaaa",
        exact_values: [],
        evidence: [evidenceRef],
      },
    },
  ])("rejects empty exact_values arrays for $kind", ({ claim }) => {
    expect(manifestClaimSchema.safeParse(claim).success).toBe(false);
  });
});

describe("flat manifest wire schema", () => {
  it("accepts a flat claim shape with the union of optional per-kind fields", () => {
    const flatUserFact: FlatManifestClaim = {
      kind: "user_fact",
      rendered_span: "You prefer direct answers.",
      exact_values: ["direct answers"],
      evidence: [evidenceRef],
      confidence: "direct",
    };

    const wire = flatEmitManifestResponseSchema.safeParse({
      final_text: "You prefer direct answers.",
      discourse_act: "answer",
      claims: [flatUserFact],
    });

    expect(wire.success).toBe(true);
  });

  it("tightens a wire response into the strict discriminated union", () => {
    const wireResponse = {
      final_text: "Acknowledged.",
      discourse_act: "continue_task" as const,
      claims: [
        {
          kind: "hedge" as const,
          rendered_span: "Acknowledged.",
          evidence: undefined,
        },
      ],
    };

    const wire = flatEmitManifestResponseSchema.safeParse(wireResponse);
    expect(wire.success).toBe(true);

    if (!wire.success) {
      throw new Error("wire parse failed");
    }

    const tightened = tightenManifestResponse(wire.data);

    expect(tightened.ok).toBe(true);
    if (tightened.ok) {
      expect(tightened.manifest.claims[0]?.kind).toBe("hedge");
      const claim = tightened.manifest.claims[0] as ManifestClaim;
      expect("evidence" in claim).toBe(false);
    }
  });

  it("rejects per-kind violations during tightening", () => {
    const wire = flatEmitManifestResponseSchema.safeParse({
      final_text: "Will check that for you.",
      discourse_act: "answer",
      claims: [
        {
          kind: "user_fact",
          rendered_span: "You prefer concise answers.",
        },
      ],
    });

    expect(wire.success).toBe(true);
    if (!wire.success) {
      throw new Error("wire parse failed");
    }

    const tightened = tightenManifestResponse(wire.data);

    expect(tightened.ok).toBe(false);
    if (!tightened.ok) {
      expect(tightened.offending_claim_index).toBe(0);
      expect(tightened.offending_claim?.kind).toBe("user_fact");
    }
  });

  it("strips fields that do not belong to a kind before strict parsing", () => {
    const wire = flatEmitManifestResponseSchema.safeParse({
      final_text: "Hedge wins.",
      discourse_act: "answer",
      claims: [
        {
          kind: "hedge",
          rendered_span: "Hedge wins.",
          confidence: "low",
          callback_scope: "prior_session",
          evidence: [evidenceRef],
        },
      ],
    });

    expect(wire.success).toBe(true);
    if (!wire.success) {
      throw new Error("wire parse failed");
    }

    const tightened = tightenManifestResponse(wire.data);

    expect(tightened.ok).toBe(true);
    if (tightened.ok) {
      const claim = tightened.manifest.claims[0];
      expect(claim?.kind).toBe("hedge");
      expect("confidence" in (claim ?? {})).toBe(false);
      expect("evidence" in (claim ?? {})).toBe(false);
    }
  });

  it("preserves addresses_audience_by_name across the tighten transformation", () => {
    const wire = flatEmitManifestResponseSchema.safeParse({
      final_text: "Goodnight, Tom.",
      discourse_act: "acknowledge",
      claims: [
        {
          kind: "discourse_only",
          rendered_span: "Goodnight, Tom.",
          addresses_audience_by_name: true,
        },
      ],
    });

    expect(wire.success).toBe(true);
    if (!wire.success) {
      throw new Error("wire parse failed");
    }

    const tightened = tightenManifestResponse(wire.data);

    expect(tightened.ok).toBe(true);
    if (tightened.ok) {
      const claim = tightened.manifest.claims[0];
      expect(claim?.addresses_audience_by_name).toBe(true);
    }
  });
});

describe("MANIFEST_STRUCTURED_OUTPUT_FORMAT (Anthropic wire schema)", () => {
  it("declares json_schema with a flat claim parent", () => {
    expect(MANIFEST_STRUCTURED_OUTPUT_FORMAT.type).toBe("json_schema");
    const claimSchema = (
      MANIFEST_STRUCTURED_OUTPUT_FORMAT.schema.properties as { claims: { items: unknown } }
    ).claims.items as Record<string, unknown>;
    expect(claimSchema.type).toBe("object");
    expect((claimSchema.properties as { kind: { enum: string[] } }).kind.enum).toContain(
      "user_fact",
    );
  });

  it("encodes per-kind required fields via allOf + if/then conditionals", () => {
    const serialized = JSON.stringify(MANIFEST_STRUCTURED_OUTPUT_FORMAT);

    // Each grounded kind enforces evidence at the API level.
    expect(serialized).toContain('"const":"user_fact"');
    expect(serialized).toContain('"const":"prior_callback"');
    expect(serialized).toContain('"const":"action_state"');
    expect(serialized).toContain('"const":"slot_fact"');
    expect(serialized).toContain('"const":"agent_self_provenance"');
    expect(serialized).toContain('"const":"interpretation"');
    expect(serialized).toContain('"const":"self_report"');

    // Defensive: persistence_class is required only on self_report, asserted
    // via const so the wire enforces it instead of treating it as a
    // descriptive string.
    expect(serialized).toContain('"const":"assistant_self_report"');

    // Every if/then branch has an "if" with the discriminator constant.
    const claimSchema = (
      MANIFEST_STRUCTURED_OUTPUT_FORMAT.schema.properties as { claims: { items: unknown } }
    ).claims.items as { allOf: Array<{ if: unknown; then: { required: string[] } }> };
    expect(claimSchema.allOf).toHaveLength(7);

    const userFactBranch = claimSchema.allOf.find(
      (entry) =>
        ((entry.if as { properties: { kind: { const: string } } }).properties.kind.const) ===
        "user_fact",
    );
    expect(userFactBranch?.then.required).toEqual([
      "evidence",
      "exact_values",
      "confidence",
    ]);
  });
});
