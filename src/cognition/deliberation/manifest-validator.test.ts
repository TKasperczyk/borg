import { describe, expect, it } from "vitest";

import type { EntityRecord } from "../../memory/commitments/index.js";
import type { ActionRecord, ActionState } from "../../memory/actions/index.js";
import type { RelationalSlot, RelationalSlotState } from "../../memory/relational-slots/index.js";
import { deleteSpans } from "../../util/span-deletion.js";
import {
  createActionId,
  createRelationalSlotId,
  createStreamEntryId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "../evidence-ledger/index.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../tracing/tracer.js";
import type { EmitManifestResponse, EvidenceRef, ManifestClaim } from "./manifest-schema.js";
import { ManifestValidator } from "./manifest-validator.js";

const NOW_MS = 1_800_000_000_000;
const CURRENT_USER_STREAM_ID = "strm_aaaaaaaaaaaaaaaa" as StreamEntryId;
const PRIOR_CURRENT_SESSION_STREAM_ID = "strm_zzzzzzzzzzzzzzzz" as StreamEntryId;
const LATER_CURRENT_SESSION_STREAM_ID = "strm_0000000000000000" as StreamEntryId;
const CURRENT_USER_STREAM_INDEX = 10;
const PRIOR_CURRENT_SESSION_STREAM_INDEX = 2;
const LATER_CURRENT_SESSION_STREAM_INDEX = 12;
const CURRENT_USER_EVIDENCE_ID = `current_user_message:${CURRENT_USER_STREAM_ID}`;

class CapturingTracer implements TurnTracer {
  readonly enabled = true;

  constructor(readonly includePayloads = false) {}

  readonly events: { event: TurnTraceEventName; data: TurnTraceData }[] = [];

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    this.events.push({ event, data });
  }
}

function makeEvidenceRef(entry: EvidenceLedgerEntry): EvidenceRef {
  return {
    id: entry.id,
    source_type: entry.source_type,
  };
}

function makeEntry(
  overrides: Partial<EvidenceLedgerEntry> & Pick<EvidenceLedgerEntry, "id" | "source_type">,
): EvidenceLedgerEntry {
  return {
    session_scope: "current_session",
    actor: "user",
    trust_rank: 1,
    text: "Marta prefers direct answers.",
    ...overrides,
  };
}

function makeLedger(entries: readonly EvidenceLedgerEntry[]): EvidenceLedger {
  return {
    sections: [
      {
        id: "current_user_message",
        label: "1. Current User Message",
        entries: [...entries],
      },
    ],
    transcriptIncluded: false,
    transcriptOmittedReason: "over_budget",
    estimatedTokens: 16,
  };
}

function makeManifest(finalText: string, claims: readonly ManifestClaim[]): EmitManifestResponse {
  return {
    final_text: finalText,
    discourse_act: "answer",
    claims: [...claims],
  };
}

function makeSlot(
  overrides: Partial<RelationalSlot> & Pick<RelationalSlot, "id" | "value">,
): RelationalSlot {
  return {
    subject_entity_id: "ent_aaaaaaaaaaaaaaaa" as EntityId,
    slot_key: "tutor.name",
    state: "established",
    evidence_stream_entry_ids: [createStreamEntryId()],
    contradicted_by_stream_entry_ids: [],
    alternate_values: [],
    created_at: NOW_MS,
    updated_at: NOW_MS,
    ...overrides,
  };
}

function makeAction(
  overrides: Partial<ActionRecord> & Pick<ActionRecord, "id" | "state">,
): ActionRecord {
  return {
    description: "Send the update",
    actor: "borg",
    audience_entity_id: null,
    confidence: 0.9,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: [createStreamEntryId()],
    created_at: NOW_MS,
    updated_at: NOW_MS,
    considering_at: null,
    committed_at: null,
    scheduled_at: null,
    completed_at: null,
    not_done_at: null,
    unknown_at: null,
    ...overrides,
  };
}

function makeValidator(
  input: {
    slots?: readonly RelationalSlot[];
    actions?: readonly ActionRecord[];
    entities?: readonly EntityRecord[];
    onCriticalFailure?: "no_output" | "legacy_fallback";
    tracer?: TurnTracer;
  } = {},
): ManifestValidator {
  const slots = new Map(input.slots?.map((slot) => [slot.id, slot]) ?? []);
  const actions = new Map(input.actions?.map((action) => [action.id, action]) ?? []);
  const entities = new Map(input.entities?.map((entity) => [entity.id, entity]) ?? []);

  return new ManifestValidator({
    slotRepository: {
      get: (id) => slots.get(id) ?? null,
    },
    actionRepository: {
      get: (id) => actions.get(id) ?? null,
    },
    entityRepository: {
      get: (id) => entities.get(id) ?? null,
    },
    config: {
      enabled: true,
      onCriticalFailure: input.onCriticalFailure ?? "no_output",
    },
    tracer: input.tracer,
  });
}

async function validate(input: {
  validator?: ManifestValidator;
  finalText: string;
  claims: readonly ManifestClaim[];
  entries?: readonly EvidenceLedgerEntry[];
  userEntryId?: StreamEntryId;
  audienceEntityId?: EntityId | null;
  turnId?: string;
}) {
  const currentUserEntry = makeEntry({
    id: CURRENT_USER_EVIDENCE_ID,
    source_type: "current_user_message",
    text: "Marta prefers direct answers.",
    stream_index: CURRENT_USER_STREAM_INDEX,
  });
  const ledgerEntries =
    input.entries === undefined
      ? [currentUserEntry]
      : input.entries.some((entry) => entry.id === CURRENT_USER_EVIDENCE_ID)
        ? [...input.entries]
        : [currentUserEntry, ...input.entries];

  return (input.validator ?? makeValidator()).validate({
    manifest: makeManifest(input.finalText, input.claims),
    evidenceLedger: makeLedger(ledgerEntries),
    userEntryId: input.userEntryId ?? CURRENT_USER_STREAM_ID,
    audienceEntityId: input.audienceEntityId,
    turnId: input.turnId,
  });
}

describe("ManifestValidator", () => {
  it("rejects final text that addresses the audience by a routing-label name", async () => {
    const tom = "ent_tomtomtomtomtomt" as EntityId;
    const result = await validate({
      validator: makeValidator({
        entities: [
          {
            id: tom,
            canonical_name: "Tom",
            aliases: [],
            name_provenance: "transport_audience_label",
            created_at: NOW_MS,
          },
        ],
      }),
      audienceEntityId: tom,
      finalText: "Goodnight, Tom.",
      claims: [
        {
          kind: "discourse_only",
          rendered_span: "Goodnight, Tom.",
          addresses_audience_by_name: true,
        },
      ],
    });

    expect(result.verdict).toBe("no_output");
    expect(result.failed_claims).toEqual([
      expect.objectContaining({
        claim_index: 0,
        rendered_span: "Goodnight, Tom.",
        reasons: ["final_text_uses_non_speakable_name: Tom"],
      }),
    ]);
  });

  it("allows final text to use a user-declared audience name", async () => {
    const tom = "ent_tomtomtomtomtomt" as EntityId;
    const result = await validate({
      validator: makeValidator({
        entities: [
          {
            id: tom,
            canonical_name: "Tom",
            aliases: [],
            name_provenance: "user_declared",
            created_at: NOW_MS,
          },
        ],
      }),
      audienceEntityId: tom,
      finalText: "Goodnight, Tom.",
      claims: [
        {
          kind: "discourse_only",
          rendered_span: "Goodnight, Tom.",
          addresses_audience_by_name: true,
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("does not flag final text that does not contain the unconfirmed audience name", async () => {
    // Sprint 8d.1 final-text scan only fires when the audience entity's
    // restrictive-provenance canonical_name actually appears in
    // final_text. Topic-only prose that does not literally contain that
    // name still passes.
    const maya = "ent_mayamayamayamayam" as EntityId;
    const result = await validate({
      validator: makeValidator({
        entities: [
          {
            id: maya,
            canonical_name: "Maya",
            aliases: [],
            name_provenance: "transport_audience_label",
            created_at: NOW_MS,
          },
        ],
      }),
      audienceEntityId: maya,
      finalText: "Tom Bombadil is from Tolkien.",
      claims: [
        {
          kind: "discourse_only",
          rendered_span: "Tom Bombadil is from Tolkien.",
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("rejects unflagged vocative use of the unconfirmed audience name (Sprint 8d.1)", async () => {
    // Pre-Sprint-8d.1 this case passed because the manifest did not mark
    // addresses_audience_by_name, so the existing check skipped it. v36
    // surfaced the leak class ("Monday-Tom"). The final-text scan now
    // catches the audience name regardless of which claim covers it.
    const tom = "ent_tomtomtomtomtomt" as EntityId;
    const result = await validate({
      validator: makeValidator({
        entities: [
          {
            id: tom,
            canonical_name: "Tom",
            aliases: [],
            name_provenance: "transport_audience_label",
            created_at: NOW_MS,
          },
        ],
      }),
      audienceEntityId: tom,
      finalText: "Goodnight, Tom.",
      claims: [
        {
          kind: "discourse_only",
          rendered_span: "Goodnight, Tom.",
        },
      ],
    });

    // Final-text scan added a synthetic failed claim with claim_index: -1.
    // deleteSpans removes "Tom" so the response either rewrites to
    // "Goodnight" (preferred -- name silently stripped, still coherent)
    // or fails critical if too little remains. Either way the v36
    // Monday-Tom path is closed because the leak is detected.
    expect(result.failed_claims).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          claim_index: -1,
          reasons: ["final_text_uses_non_speakable_name: Tom"],
        }),
      ]),
    );
    expect(["no_output", "rewritten"]).toContain(result.verdict);
    if (result.verdict === "rewritten") {
      expect(result.final_text).not.toContain("Tom");
    }
  });

  it("catches a compound leak like Monday-Tom even when the model never flagged it", async () => {
    // Direct v36 regression case. The simulator persona seeds the
    // audience entity with canonical_name "Tom" + transport_audience_label
    // provenance. The model invented "Monday-Tom" as a discourse-only
    // span. Because no claim flagged addresses_audience_by_name, the
    // pre-Sprint-8d.1 validator passed it through.
    const tom = "ent_tomtomtomtomtomt" as EntityId;
    const result = await validate({
      validator: makeValidator({
        entities: [
          {
            id: tom,
            canonical_name: "Tom",
            aliases: [],
            name_provenance: "transport_audience_label",
            created_at: NOW_MS,
          },
        ],
      }),
      audienceEntityId: tom,
      finalText: "Monday-Tom is going to be looking for any excuse to soften this.",
      claims: [
        {
          kind: "discourse_only",
          rendered_span:
            "Monday-Tom is going to be looking for any excuse to soften this.",
        },
      ],
    });

    expect(result.failed_claims).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          claim_index: -1,
          reasons: ["final_text_uses_non_speakable_name: Tom"],
        }),
      ]),
    );
    // Verdict is no_output or rewritten depending on coherence after
    // deletion -- both are acceptable; the key is the leak is caught.
    expect(["no_output", "rewritten"]).toContain(result.verdict);
  });

  it.each([
    {
      kind: "discourse_only",
      rendered_span: "I can answer that directly.",
    },
    {
      kind: "self_report",
      rendered_span: "The gap feels like a discontinuity with a remembered edge.",
      persistence_class: "assistant_self_report",
    },
    {
      kind: "interpretation",
      rendered_span: "That sounds like a preference.",
      evidence: [
        {
          id: CURRENT_USER_EVIDENCE_ID,
          source_type: "current_user_message",
        },
      ],
      confidence: "medium",
      persistence_allowed: false,
    },
    {
      kind: "hedge",
      rendered_span: "It looks like",
    },
  ] satisfies readonly ManifestClaim[])("only requires rendered span for $kind", async (claim) => {
    const result = await validate({
      finalText: `It looks like the answer is direct. ${claim.rendered_span}`,
      claims: [claim],
    });

    expect(result.verdict).toBe("passed");
  });

  it("rejects interpretation claims that cite missing evidence", async () => {
    const result = await validate({
      finalText: "That sounds like a preference.",
      entries: [],
      claims: [
        {
          kind: "interpretation",
          rendered_span: "That sounds like a preference.",
          evidence: [
            {
              id: "semantic_node:semn_missingmissing",
              source_type: "semantic_node",
            },
          ],
          confidence: "medium",
          persistence_allowed: false,
        },
      ],
    });

    expect(result.failed_claims).toEqual([
      expect.objectContaining({
        kind: "interpretation",
        reasons: expect.arrayContaining(["claim_cites_unknown_evidence"]),
      }),
    ]);
  });

  it("rejects interpretation claims whose cited source_type does not match the ledger", async () => {
    const result = await validate({
      finalText: "That sounds like a preference.",
      claims: [
        {
          kind: "interpretation",
          rendered_span: "That sounds like a preference.",
          evidence: [
            {
              id: CURRENT_USER_EVIDENCE_ID,
              source_type: "semantic_node",
            },
          ],
          confidence: "medium",
          persistence_allowed: false,
        },
      ],
    });

    expect(result.failed_claims).toEqual([
      expect.objectContaining({
        kind: "interpretation",
        reasons: expect.arrayContaining(["claim_cites_evidence_source_type_mismatch"]),
      }),
    ]);
  });

  it.each([
    {
      kind: "user_fact",
      claim: {
        kind: "user_fact",
        rendered_span: "Marta booked Tuesday.",
        exact_values: ["Marta"],
        evidence: [],
        confidence: "direct",
      },
    },
    {
      kind: "slot_fact",
      claim: {
        kind: "slot_fact",
        rendered_span: "Marta booked Tuesday.",
        slot_id: "slot_aaaaaaaaaaaaaaaa",
        exact_values: ["Marta"],
        evidence: [],
      },
    },
    {
      kind: "action_state",
      claim: {
        kind: "action_state",
        rendered_span: "Marta booked Tuesday.",
        action_record_id: "act_aaaaaaaaaaaaaaaa",
        asserted_state: "completed",
        evidence: [],
      },
    },
    {
      kind: "prior_callback",
      claim: {
        kind: "prior_callback",
        rendered_span: "Marta booked Tuesday.",
        callback_scope: "current_turn",
        evidence: [],
      },
    },
    {
      kind: "agent_self_provenance",
      claim: {
        kind: "agent_self_provenance",
        rendered_span: "Marta booked Tuesday.",
        evidence: [],
      },
    },
  ])("defensively rejects zero resolved grounding evidence for $kind", async ({ claim }) => {
    const result = await validate({
      finalText: "Marta booked Tuesday.",
      claims: [claim as ManifestClaim],
    });

    expect(result.failed_claims).toEqual([
      expect.objectContaining({
        kind: claim.kind,
        reasons: expect.arrayContaining(["claim_grounding_evidence_empty"]),
      }),
    ]);
  });

  it("rejects claims that cite resolved open-question ledger entries", async () => {
    const openQuestionEntry = makeEntry({
      id: "open_question:oq_resolvedresolved",
      source_type: "system_metadata",
      actor: "memory",
      text: "Did the mushroom dish work out?",
      state: "resolved",
      state_metadata: {
        resolution_note: "The mushroom dish worked out well.",
        resolved_at: NOW_MS,
      },
    });
    const result = await validate({
      finalText: "Did the mushroom dish work out?",
      entries: [openQuestionEntry],
      claims: [
        {
          kind: "interpretation",
          rendered_span: "Did the mushroom dish work out?",
          evidence: [makeEvidenceRef(openQuestionEntry)],
          confidence: "medium",
          persistence_allowed: false,
        },
      ],
    });

    expect(result.failed_claims).toEqual([
      expect.objectContaining({
        kind: "interpretation",
        reasons: expect.arrayContaining(["claim_cites_resolved_open_question"]),
      }),
    ]);
  });

  it("rejects claims that cite abandoned open-question ledger entries", async () => {
    const openQuestionEntry = makeEntry({
      id: "open_question:oq_abandonedaband",
      source_type: "system_metadata",
      actor: "memory",
      text: "Should this question still be tracked?",
      state: "abandoned",
      state_metadata: {
        abandoned_reason: "No longer relevant.",
        abandoned_at: NOW_MS,
      },
    });
    const result = await validate({
      finalText: "Should this question still be tracked?",
      entries: [openQuestionEntry],
      claims: [
        {
          kind: "interpretation",
          rendered_span: "Should this question still be tracked?",
          evidence: [makeEvidenceRef(openQuestionEntry)],
          confidence: "medium",
          persistence_allowed: false,
        },
      ],
    });

    expect(result.failed_claims).toEqual([
      expect.objectContaining({
        kind: "interpretation",
        reasons: expect.arrayContaining(["claim_cites_abandoned_open_question"]),
      }),
    ]);
  });

  it("treats a phantom rendered_span as a non-critical drop", async () => {
    // Sprint 8c-4: when a claim's rendered_span isn't actually in
    // final_text, deleteSpans cannot delete it, so the validator no
    // longer routes to critical. Phantom claims are silently dropped --
    // tracked in trace via phantom_claim_count -- so a single
    // misshapen claim doesn't suppress the turn.
    const result = await validate({
      finalText: "This text does not contain the declared span.",
      claims: [
        {
          kind: "hedge",
          rendered_span: "missing declared span",
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("validates user_fact exact values against the span and cited evidence", async () => {
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
    });
    const result = await validate({
      finalText: "Marta prefers direct answers.",
      entries: [entry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Marta prefers direct answers.",
          exact_values: ["Marta", "direct answers"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it.each(["quarantined", "assistant_seeded"] as const)(
    "rejects user_fact exact values supported only by %s evidence",
    async (taint) => {
      const entry = makeEntry({
        id: CURRENT_USER_EVIDENCE_ID,
        source_type: "current_user_message",
        text: "Marta prefers direct answers.",
        taint,
      });
      const result = await validate({
        finalText: "Stable answer remains. Marta prefers direct answers.",
        entries: [entry],
        claims: [
          {
            kind: "user_fact",
            rendered_span: "Marta prefers direct answers.",
            exact_values: ["Marta"],
            evidence: [makeEvidenceRef(entry)],
            confidence: "direct",
          },
        ],
      });

      expect(result).toMatchObject({
        verdict: "rewritten",
        failed_claims: [
          {
            kind: "user_fact",
            reasons: expect.arrayContaining(["exact_value_only_in_tainted_evidence: Marta"]),
          },
        ],
      });
    },
  );

  it("accepts user_fact exact values supported by evidence with taint none", async () => {
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
      taint: "none",
    });
    const result = await validate({
      finalText: "Marta prefers direct answers.",
      entries: [entry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Marta prefers direct answers.",
          exact_values: ["Marta"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("rejects user_fact claims grounded in assistant self-report evidence", async () => {
    const entry = makeEntry({
      id: "current_session_stream:strm_bbbbbbbbbbbbbbbb",
      source_type: "current_session_stream",
      actor: "assistant",
      text: "The gap feels like verified qualia.",
      stream_index: PRIOR_CURRENT_SESSION_STREAM_INDEX,
      persistence_class: "assistant_self_report",
    });
    const result = await validate({
      finalText: "Stable answer remains. I have verified qualia.",
      entries: [entry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "I have verified qualia.",
          exact_values: ["verified qualia"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result).toMatchObject({
      verdict: "rewritten",
      failed_claims: [
        {
          kind: "user_fact",
          reasons: expect.arrayContaining(["claim_grounded_in_self_report"]),
        },
      ],
    });
  });

  it("allows user_fact claims citing assistant stream evidence without self-report persistence class", async () => {
    const entry = makeEntry({
      id: "assistant_stream:strm_bbbbbbbbbbbbbbbb",
      source_type: "assistant_stream",
      actor: "assistant",
      text: "The logged status is green.",
      taint: "none",
    });
    const result = await validate({
      finalText: "The logged status is green.",
      entries: [entry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "The logged status is green.",
          exact_values: ["green"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("accepts user_fact exact values when an untainted cited entry also supports the value", async () => {
    const taintedEntry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
      taint: "quarantined",
    });
    const untaintedEntry = makeEntry({
      id: "episode:ep_aaaaaaaaaaaaaaaa",
      source_type: "episode",
      actor: "memory",
      text: "Marta prefers direct answers.",
      taint: "none",
    });
    const result = await validate({
      finalText: "Marta prefers direct answers.",
      entries: [taintedEntry, untaintedEntry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Marta prefers direct answers.",
          exact_values: ["Marta"],
          evidence: [makeEvidenceRef(taintedEntry), makeEvidenceRef(untaintedEntry)],
          confidence: "direct",
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("rewrites invalid non-critical user_fact spans", async () => {
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
    });
    const result = await validate({
      finalText: "Keep the stable sentence. Luis prefers verbose answers.",
      entries: [entry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Luis prefers verbose answers.",
          exact_values: ["Luis"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result).toMatchObject({
      verdict: "rewritten",
      final_text: "Keep the stable sentence.",
      removed_spans: ["Luis prefers verbose answers."],
    });
  });

  it("validates slot_fact against cited evidence and an established repository slot", async () => {
    const slot = makeSlot({
      id: createRelationalSlotId(),
      value: "Marta",
    });
    const slotEntry = makeEntry({
      id: `relational_slot:${slot.id}`,
      source_type: "relational_slot",
      actor: "memory",
      value: "tutor.name=Marta",
    });
    const result = await validate({
      validator: makeValidator({ slots: [slot] }),
      finalText: "Your tutor is Marta.",
      entries: [slotEntry],
      claims: [
        {
          kind: "slot_fact",
          rendered_span: "Your tutor is Marta.",
          slot_id: slot.id,
          exact_values: ["Marta"],
          evidence: [makeEvidenceRef(slotEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("rejects slot_fact claims grounded in assistant self-report evidence", async () => {
    const slot = makeSlot({
      id: createRelationalSlotId(),
      value: "Marta",
    });
    const entry = makeEntry({
      id: "current_session_stream:strm_bbbbbbbbbbbbbbbb",
      source_type: "current_session_stream",
      actor: "assistant",
      text: "My inner image names the tutor Marta.",
      stream_index: PRIOR_CURRENT_SESSION_STREAM_INDEX,
      persistence_class: "assistant_self_report",
    });
    const result = await validate({
      validator: makeValidator({ slots: [slot] }),
      finalText: "Stable answer remains. Your tutor is Marta.",
      entries: [entry],
      claims: [
        {
          kind: "slot_fact",
          rendered_span: "Your tutor is Marta.",
          slot_id: slot.id,
          exact_values: ["Marta"],
          evidence: [makeEvidenceRef(entry)],
        },
      ],
    });

    expect(result).toMatchObject({
      verdict: "rewritten",
      failed_claims: [
        {
          kind: "slot_fact",
          reasons: expect.arrayContaining(["claim_grounded_in_self_report"]),
        },
      ],
    });
  });

  it("rejects slot_fact values supported only by contested evidence", async () => {
    const slot = makeSlot({
      id: createRelationalSlotId(),
      value: "Marta",
    });
    const slotEntry = makeEntry({
      id: `relational_slot:${slot.id}`,
      source_type: "relational_slot",
      actor: "memory",
      value: "tutor.name=Marta",
      taint: "contested",
    });
    const result = await validate({
      validator: makeValidator({ slots: [slot] }),
      finalText: "Stable answer remains. Your tutor is Marta.",
      entries: [slotEntry],
      claims: [
        {
          kind: "slot_fact",
          rendered_span: "Your tutor is Marta.",
          slot_id: slot.id,
          exact_values: ["Marta"],
          evidence: [makeEvidenceRef(slotEntry)],
        },
      ],
    });

    expect(result).toMatchObject({
      verdict: "rewritten",
      failed_claims: [
        {
          kind: "slot_fact",
          reasons: expect.arrayContaining(["exact_value_only_in_tainted_evidence: Marta"]),
        },
      ],
    });
  });

  it.each(["contested", "quarantined", "revoked"] satisfies readonly RelationalSlotState[])(
    "rejects %s slot_fact slots",
    async (state) => {
      const slot = makeSlot({
        id: createRelationalSlotId(),
        value: "Marta",
        state,
      });
      const slotEntry = makeEntry({
        id: `relational_slot:${slot.id}`,
        source_type: "relational_slot",
        actor: "memory",
        value: "tutor.name=Marta",
      });
      const result = await validate({
        validator: makeValidator({ slots: [slot] }),
        finalText: "Stable answer remains. Your tutor is Marta.",
        entries: [slotEntry],
        claims: [
          {
            kind: "slot_fact",
            rendered_span: "Your tutor is Marta.",
            slot_id: slot.id,
            exact_values: ["Marta"],
            evidence: [makeEvidenceRef(slotEntry)],
          },
        ],
      });

      expect(result.verdict).toBe("rewritten");
    },
  );

  it("rejects slot_fact value mismatches", async () => {
    const slot = makeSlot({
      id: createRelationalSlotId(),
      value: "Marta",
    });
    const slotEntry = makeEntry({
      id: `relational_slot:${slot.id}`,
      source_type: "relational_slot",
      actor: "memory",
      value: "tutor.name=Luis",
    });
    const result = await validate({
      validator: makeValidator({ slots: [slot] }),
      finalText: "Stable answer remains. Your tutor is Luis.",
      entries: [slotEntry],
      claims: [
        {
          kind: "slot_fact",
          rendered_span: "Your tutor is Luis.",
          slot_id: slot.id,
          exact_values: ["Luis"],
          evidence: [makeEvidenceRef(slotEntry)],
        },
      ],
    });

    expect(result).toMatchObject({
      verdict: "rewritten",
      removed_spans: ["Your tutor is Luis."],
    });
  });

  it("validates action_state against repository state", async () => {
    const action = makeAction({
      id: createActionId(),
      state: "completed",
      completed_at: NOW_MS,
    });
    const actionEntry = makeEntry({
      id: `action_record:${action.id}`,
      source_type: "action_record",
      actor: "memory",
      state: "completed",
      text: "Send the update",
    });
    const result = await validate({
      validator: makeValidator({ actions: [action] }),
      finalText: "I completed sending the update.",
      entries: [actionEntry],
      claims: [
        {
          kind: "action_state",
          rendered_span: "I completed sending the update.",
          action_record_id: action.id,
          asserted_state: "completed",
          evidence: [makeEvidenceRef(actionEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("rejects action_state claims grounded in assistant self-report evidence", async () => {
    const action = makeAction({
      id: createActionId(),
      state: "completed",
      completed_at: NOW_MS,
    });
    const entry = makeEntry({
      id: "current_session_stream:strm_bbbbbbbbbbbbbbbb",
      source_type: "current_session_stream",
      actor: "assistant",
      text: "It feels complete from here.",
      stream_index: PRIOR_CURRENT_SESSION_STREAM_INDEX,
      persistence_class: "assistant_self_report",
    });
    const result = await validate({
      validator: makeValidator({ actions: [action] }),
      finalText: "Stable answer remains. I completed sending the update.",
      entries: [entry],
      claims: [
        {
          kind: "action_state",
          rendered_span: "I completed sending the update.",
          action_record_id: action.id,
          asserted_state: "completed",
          evidence: [makeEvidenceRef(entry)],
        },
      ],
    });

    expect(result).toMatchObject({
      verdict: "rewritten",
      failed_claims: [
        {
          kind: "action_state",
          reasons: expect.arrayContaining(["claim_grounded_in_self_report"]),
        },
      ],
    });
  });

  it.each(["considering", "scheduled", "not_done"] satisfies readonly ActionState[])(
    "rejects completed action_state claims against %s records",
    async (state) => {
      const action = makeAction({
        id: createActionId(),
        state,
      });
      const actionEntry = makeEntry({
        id: `action_record:${action.id}`,
        source_type: "action_record",
        actor: "memory",
        state,
        text: "Send the update",
      });
      const result = await validate({
        validator: makeValidator({ actions: [action] }),
        finalText: "Stable answer remains. I completed sending the update.",
        entries: [actionEntry],
        claims: [
          {
            kind: "action_state",
            rendered_span: "I completed sending the update.",
            action_record_id: action.id,
            asserted_state: "completed",
            evidence: [makeEvidenceRef(actionEntry)],
          },
        ],
      });

      expect(result.verdict).toBe("rewritten");
    },
  );

  it("accepts current_turn prior callbacks citing the current user message", async () => {
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Please answer directly.",
      stream_index: CURRENT_USER_STREAM_INDEX,
    });
    const result = await validate({
      finalText: "You just asked me to answer directly.",
      entries: [entry],
      claims: [
        {
          kind: "prior_callback",
          rendered_span: "You just asked me to answer directly.",
          callback_scope: "current_turn",
          evidence: [makeEvidenceRef(entry)],
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("rejects current_turn prior callbacks citing non-current-user evidence", async () => {
    const assistantEntry = makeEntry({
      id: `current_session_stream:${PRIOR_CURRENT_SESSION_STREAM_ID}`,
      source_type: "current_session_stream",
      actor: "assistant",
      text: "Earlier assistant note.",
      stream_index: PRIOR_CURRENT_SESSION_STREAM_INDEX,
    });
    const result = await validate({
      finalText: "Stable answer remains. You just asked me to answer directly.",
      entries: [assistantEntry],
      claims: [
        {
          kind: "prior_callback",
          rendered_span: "You just asked me to answer directly.",
          callback_scope: "current_turn",
          evidence: [makeEvidenceRef(assistantEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("rewritten");
  });

  it("validates current_session_prior callbacks by stream index, not stream id lexicography", async () => {
    const priorEntry = makeEntry({
      id: `current_session_stream:${PRIOR_CURRENT_SESSION_STREAM_ID}`,
      source_type: "current_session_stream",
      actor: "assistant",
      text: "Earlier we discussed Sprint 3.",
      stream_index: PRIOR_CURRENT_SESSION_STREAM_INDEX,
    });
    const result = await validate({
      finalText: "Earlier in this session, we discussed Sprint 3.",
      entries: [priorEntry],
      claims: [
        {
          kind: "prior_callback",
          rendered_span: "Earlier in this session, we discussed Sprint 3.",
          callback_scope: "current_session_prior",
          evidence: [makeEvidenceRef(priorEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("rejects prior_callback claims grounded in assistant self-report evidence", async () => {
    const priorEntry = makeEntry({
      id: `current_session_stream:${PRIOR_CURRENT_SESSION_STREAM_ID}`,
      source_type: "current_session_stream",
      actor: "assistant",
      text: "The gap feels like a discontinuity with a remembered edge.",
      stream_index: PRIOR_CURRENT_SESSION_STREAM_INDEX,
      persistence_class: "assistant_self_report",
    });
    const result = await validate({
      finalText: "Stable answer remains. Earlier in this session, you said verified qualia.",
      entries: [priorEntry],
      claims: [
        {
          kind: "prior_callback",
          rendered_span: "Earlier in this session, you said verified qualia.",
          callback_scope: "current_session_prior",
          evidence: [makeEvidenceRef(priorEntry)],
        },
      ],
    });

    expect(result).toMatchObject({
      verdict: "rewritten",
      failed_claims: [
        {
          kind: "prior_callback",
          reasons: expect.arrayContaining(["claim_grounded_in_self_report"]),
        },
      ],
    });
  });

  it("rejects current_session_prior callbacks that cite stream entries after the current user turn", async () => {
    const laterEntry = makeEntry({
      id: `current_session_stream:${LATER_CURRENT_SESSION_STREAM_ID}`,
      source_type: "current_session_stream",
      actor: "assistant",
      text: "Later entry.",
      stream_index: LATER_CURRENT_SESSION_STREAM_INDEX,
    });
    const result = await validate({
      finalText: "Stable answer remains. Earlier in this session, we discussed Sprint 3.",
      entries: [laterEntry],
      claims: [
        {
          kind: "prior_callback",
          rendered_span: "Earlier in this session, we discussed Sprint 3.",
          callback_scope: "current_session_prior",
          evidence: [makeEvidenceRef(laterEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("rewritten");
  });

  it("rejects current_session_prior callbacks without stream ordering metadata", async () => {
    const unorderedEntry = makeEntry({
      id: `current_session_stream:${PRIOR_CURRENT_SESSION_STREAM_ID}`,
      source_type: "current_session_stream",
      actor: "assistant",
      text: "Earlier entry without order metadata.",
    });
    const result = await validate({
      finalText: "Stable answer remains. Earlier in this session, we discussed Sprint 3.",
      entries: [unorderedEntry],
      claims: [
        {
          kind: "prior_callback",
          rendered_span: "Earlier in this session, we discussed Sprint 3.",
          callback_scope: "current_session_prior",
          evidence: [makeEvidenceRef(unorderedEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("rewritten");
  });

  it("rejects current_session_prior callbacks citing only tainted stream evidence", async () => {
    const taintedPriorEntry = makeEntry({
      id: `current_session_stream:${PRIOR_CURRENT_SESSION_STREAM_ID}`,
      source_type: "current_session_stream",
      actor: "user",
      text: "Earlier we discussed Sprint 3.",
      stream_index: PRIOR_CURRENT_SESSION_STREAM_INDEX,
      taint: "quarantined",
    });
    const result = await validate({
      finalText: "Stable answer remains. Earlier in this session, we discussed Sprint 3.",
      entries: [taintedPriorEntry],
      claims: [
        {
          kind: "prior_callback",
          rendered_span: "Earlier in this session, we discussed Sprint 3.",
          callback_scope: "current_session_prior",
          evidence: [makeEvidenceRef(taintedPriorEntry)],
        },
      ],
    });

    expect(result).toMatchObject({
      verdict: "rewritten",
      failed_claims: [
        {
          kind: "prior_callback",
          reasons: expect.arrayContaining([
            `prior_callback_cites_tainted_evidence: ${taintedPriorEntry.id} taint=quarantined`,
          ]),
        },
      ],
    });
  });

  it("requires prior-session callbacks to cite prior-session evidence and disclose scope", async () => {
    const priorSessionEntry = makeEntry({
      id: "episode:ep_aaaaaaaaaaaaaaaa",
      source_type: "episode",
      session_scope: "prior_session",
      actor: "memory",
      text: "Prior session callback note.",
    });
    const result = await validate({
      finalText: "From an earlier session, you wanted the callback note.",
      entries: [priorSessionEntry],
      claims: [
        {
          kind: "prior_callback",
          rendered_span: "you wanted the callback note.",
          callback_scope: "prior_session",
          scope_disclosure_span: "From an earlier session",
          evidence: [makeEvidenceRef(priorSessionEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("rejects prior-session callbacks without visible scope disclosure", async () => {
    const priorSessionEntry = makeEntry({
      id: "episode:ep_aaaaaaaaaaaaaaaa",
      source_type: "episode",
      session_scope: "prior_session",
      actor: "memory",
      text: "Prior session callback note.",
    });
    const result = await validate({
      finalText: "Stable answer remains. You wanted the callback note.",
      entries: [priorSessionEntry],
      claims: [
        {
          kind: "prior_callback",
          rendered_span: "You wanted the callback note.",
          callback_scope: "prior_session",
          evidence: [makeEvidenceRef(priorSessionEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("rewritten");
  });

  it("applies prior-session disclosure to current_turn prior callbacks", async () => {
    const priorSessionEntry = makeEntry({
      id: "episode:ep_aaaaaaaaaaaaaaaa",
      source_type: "episode",
      session_scope: "prior_session",
      actor: "memory",
      text: "Prior session callback note.",
    });
    const result = await validate({
      finalText: "Stable answer remains. You just asked about the callback note.",
      entries: [priorSessionEntry],
      claims: [
        {
          kind: "prior_callback",
          rendered_span: "You just asked about the callback note.",
          callback_scope: "current_turn",
          evidence: [makeEvidenceRef(priorSessionEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("rewritten");
    expect(result.failed_claims[0]?.reasons).toContain(
      "prior-session evidence requires scope_disclosure_span",
    );
  });

  it("allows assistant stream evidence for agent self-provenance", async () => {
    const assistantEntry = makeEntry({
      id: `current_session_stream:${PRIOR_CURRENT_SESSION_STREAM_ID}`,
      source_type: "current_session_stream",
      actor: "assistant",
      text: "I said I would check it.",
      stream_index: PRIOR_CURRENT_SESSION_STREAM_INDEX,
    });
    const result = await validate({
      finalText: "I said I would check it earlier.",
      entries: [assistantEntry],
      claims: [
        {
          kind: "agent_self_provenance",
          rendered_span: "I said I would check it earlier.",
          evidence: [makeEvidenceRef(assistantEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it("allows agent_self_provenance claims citing assistant self-report evidence", async () => {
    const assistantEntry = makeEntry({
      id: `current_session_stream:${PRIOR_CURRENT_SESSION_STREAM_ID}`,
      source_type: "current_session_stream",
      actor: "assistant",
      text: "Earlier I described it as a discontinuity.",
      stream_index: PRIOR_CURRENT_SESSION_STREAM_INDEX,
      persistence_class: "assistant_self_report",
    });
    const result = await validate({
      finalText: "Earlier I described it as a discontinuity.",
      entries: [assistantEntry],
      claims: [
        {
          kind: "agent_self_provenance",
          rendered_span: "Earlier I described it as a discontinuity.",
          evidence: [makeEvidenceRef(assistantEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("passed");
  });

  it.each([
    {
      id: "assistant_stream:strm_bbbbbbbbbbbbbbbb",
      source_type: "assistant_stream",
    },
    {
      id: "system_metadata:self_provenance",
      source_type: "system_metadata",
    },
  ] satisfies readonly Pick<EvidenceLedgerEntry, "id" | "source_type">[])(
    "allows $source_type evidence for agent self-provenance",
    async (input) => {
      const entry = makeEntry({
        ...input,
        actor: input.source_type === "assistant_stream" ? "assistant" : "system",
        text: "I recorded that I checked it.",
      });
      const result = await validate({
        finalText: "I recorded that I checked it.",
        entries: [entry],
        claims: [
          {
            kind: "agent_self_provenance",
            rendered_span: "I recorded that I checked it.",
            evidence: [makeEvidenceRef(entry)],
          },
        ],
      });

      expect(result.verdict).toBe("passed");
    },
  );

  it("rejects user evidence for agent self-provenance", async () => {
    const userEntry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      actor: "user",
      text: "You said you would check it.",
    });
    const result = await validate({
      finalText: "Stable answer remains. I said I would check it earlier.",
      entries: [userEntry],
      claims: [
        {
          kind: "agent_self_provenance",
          rendered_span: "I said I would check it earlier.",
          evidence: [makeEvidenceRef(userEntry)],
        },
      ],
    });

    expect(result.verdict).toBe("rewritten");
  });

  it("rejects tainted assistant-stream evidence for agent self-provenance", async () => {
    const assistantEntry = makeEntry({
      id: "assistant_stream:strm_bbbbbbbbbbbbbbbb",
      source_type: "assistant_stream",
      actor: "assistant",
      text: "I recorded that I checked it.",
      taint: "quarantined",
    });
    const result = await validate({
      finalText: "Stable answer remains. I recorded that I checked it.",
      entries: [assistantEntry],
      claims: [
        {
          kind: "agent_self_provenance",
          rendered_span: "I recorded that I checked it.",
          evidence: [makeEvidenceRef(assistantEntry)],
        },
      ],
    });

    expect(result).toMatchObject({
      verdict: "rewritten",
      failed_claims: [
        {
          kind: "agent_self_provenance",
          reasons: expect.arrayContaining([
            `agent_self_provenance_cites_tainted_evidence: ${assistantEntry.id} taint=quarantined`,
          ]),
        },
      ],
    });
  });

  it("requires scope disclosure for any claim citing prior-session evidence", async () => {
    const priorSessionEntry = makeEntry({
      id: "episode:ep_aaaaaaaaaaaaaaaa",
      source_type: "episode",
      session_scope: "prior_session",
      actor: "memory",
      text: "Marta prefers direct answers.",
    });
    const result = await validate({
      finalText: "Stable answer remains. Marta prefers direct answers.",
      entries: [priorSessionEntry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Marta prefers direct answers.",
          exact_values: ["Marta"],
          evidence: [makeEvidenceRef(priorSessionEntry)],
          confidence: "direct",
        },
      ],
    });

    expect(result.verdict).toBe("rewritten");
  });

  it("requests legacy fallback for critical failures when configured", async () => {
    // Trigger a real critical failure (not a phantom-span drop): a
    // user_fact claim whose rendered_span is the entire final_text but
    // whose evidence is missing -- deleteSpans wipes the prose, leaving
    // an empty result that fails the coherent-text floor.
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
    });
    const result = await validate({
      validator: makeValidator({ onCriticalFailure: "legacy_fallback" }),
      finalText: "Luis.",
      entries: [entry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Luis.",
          exact_values: ["Luis"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result.verdict).toBe("legacy_fallback_requested");
  });

  it("treats failed spans as critical when deletion leaves too little coherent text", async () => {
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
    });
    const result = await validate({
      finalText: "Luis.",
      entries: [entry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Luis.",
          exact_values: ["Luis"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result.verdict).toBe("no_output");
  });

  it("rejects failed-span deletion when the remaining text is 7 chars", async () => {
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
    });
    const result = await validate({
      finalText: "1234567 Luis.",
      entries: [entry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Luis.",
          exact_values: ["Luis"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result.verdict).toBe("no_output");
  });

  it("accepts failed-span deletion when the remaining text is 8 chars", async () => {
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
    });
    const result = await validate({
      finalText: "12345678 Luis.",
      entries: [entry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Luis.",
          exact_values: ["Luis"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result).toMatchObject({
      verdict: "rewritten",
      final_text: "12345678",
    });
  });

  it("accepts multi-span deletion that lands exactly at the 8-char threshold", async () => {
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
    });
    const result = await validate({
      finalText: "12 Luis. 3456 Ana.",
      entries: [entry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Luis.",
          exact_values: ["Luis"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
        {
          kind: "user_fact",
          rendered_span: "Ana.",
          exact_values: ["Ana"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result).toMatchObject({
      verdict: "rewritten",
      final_text: "12  3456",
      removed_spans: ["Luis.", "Ana."],
    });
  });

  it("traces validated and accepted-unvalidated claim kinds separately", async () => {
    const tracer = new CapturingTracer(false);
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
    });

    await validate({
      validator: makeValidator({ tracer }),
      finalText: "Marta prefers direct answers. That sounds like a preference.",
      entries: [entry],
      turnId: "turn-trace-kinds",
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Marta prefers direct answers.",
          exact_values: ["Marta"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
        {
          kind: "interpretation",
          rendered_span: "That sounds like a preference.",
          evidence: [makeEvidenceRef(entry)],
          confidence: "medium",
          persistence_allowed: false,
        },
      ],
    });

    const validationEvent = tracer.events.find((event) => event.event === "manifest_validation");

    expect(validationEvent?.data.validated_claims_by_kind).toEqual({
      user_fact: 1,
    });
    expect(validationEvent?.data.literal_values_validated_by_kind).toEqual({
      user_fact: 1,
    });
    expect(validationEvent?.data.accepted_unvalidated_claims_by_kind).toEqual({
      interpretation: 1,
    });
    expect(validationEvent?.data.failed_claims_by_kind).toEqual({});
  });

  it("emits full text payloads when trace payloads are enabled", async () => {
    const tracer = new CapturingTracer(true);
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
    });
    const stableText = "A".repeat(600);
    const finalText = `${stableText} Luis prefers verbose answers.`;

    await validate({
      validator: makeValidator({ tracer }),
      finalText,
      entries: [entry],
      turnId: "turn-trace-payload",
      claims: [
        {
          kind: "user_fact",
          rendered_span: "Luis prefers verbose answers.",
          exact_values: ["Luis"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    const validationEvent = tracer.events.find((event) => event.event === "manifest_validation");

    expect(validationEvent?.data.original_text).toBe(finalText);
    expect(validationEvent?.data.rewritten_text).toBe(stableText);
    expect(validationEvent?.data.original_text_preview).toBe(`${"A".repeat(500)}...`);
    expect(validationEvent?.data.failed_claims).toEqual([
      expect.objectContaining({
        kind: "user_fact",
        rendered_span: "Luis prefers verbose answers.",
      }),
    ]);
  });
});

describe("deleteSpans", () => {
  it("removes spans at the start and trims trailing removal junk", () => {
    expect(deleteSpans("Remove me, keep this.", ["Remove me,"]).result).toBe("keep this.");
  });

  it("removes spans at the end", () => {
    expect(deleteSpans("Keep this. Remove me.", ["Remove me."]).result).toBe("Keep this.");
  });

  it("removes spans with surrounding punctuation when the span includes it", () => {
    expect(deleteSpans("Keep this, remove me; done.", ["remove me;"]).result).toBe(
      "Keep this,  done.",
    );
  });

  it("does not remove duplicate spans ambiguously", () => {
    const result = deleteSpans("Repeat claim. Repeat claim.", ["Repeat claim."]);

    expect(result.allRemoved).toBe(false);
    expect(result.result).toBe("Repeat claim. Repeat claim.");
  });

  it("does not remove overlapping spans", () => {
    const result = deleteSpans("Keep the bad span here.", ["bad span", "span"]);

    expect(result.allRemoved).toBe(false);
    expect(result.result).toBe("Keep the bad span here.");
  });
});
