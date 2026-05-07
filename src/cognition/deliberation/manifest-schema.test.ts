import { describe, expect, it } from "vitest";

import {
  emitManifestResponseSchema,
  manifestClaimSchema,
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
