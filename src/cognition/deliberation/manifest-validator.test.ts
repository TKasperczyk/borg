import { describe, expect, it } from "vitest";

import type { ActionRecord } from "../../memory/actions/index.js";
import type { EntityRecord } from "../../memory/commitments/index.js";
import type { RelationalSlot } from "../../memory/relational-slots/index.js";
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
const CURRENT_USER_STREAM_INDEX = 10;
const PRIOR_CURRENT_SESSION_STREAM_INDEX = 2;
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

function makeSlot(overrides: Partial<RelationalSlot> & Pick<RelationalSlot, "id" | "value">) {
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
  } satisfies RelationalSlot;
}

function makeAction(overrides: Partial<ActionRecord> & Pick<ActionRecord, "id" | "state">) {
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
  } satisfies ActionRecord;
}

function makeValidator(
  input: {
    slots?: readonly RelationalSlot[];
    actions?: readonly ActionRecord[];
    entities?: readonly EntityRecord[];
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
  it("passes supported grounded claims without returning mutable output", async () => {
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
          exact_values: ["Marta"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result).toEqual({
      passed_claims: 1,
      failed_claims: [],
      phantom_claims: [],
      would_have_verdict: "passed",
    });
    expect("final_text" in result).toBe(false);
  });

  it("traces unsupported literal user facts without rewriting final_text", async () => {
    const tracer = new CapturingTracer(true);
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
    });
    const finalText = "Keep the stable sentence. Luis prefers verbose answers.";
    const result = await validate({
      validator: makeValidator({ tracer }),
      finalText,
      entries: [entry],
      turnId: "turn-unsupported-literal",
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

    expect(result.failed_claims).toEqual([
      expect.objectContaining({
        kind: "user_fact",
        reasons: ["exact value does not appear in cited evidence: Luis"],
      }),
    ]);
    expect(result.would_have_verdict).toBe("would_have_rewritten");
    expect("final_text" in result).toBe(false);

    const validationEvent = tracer.events.find((event) => event.event === "manifest_validation");
    expect(validationEvent?.data).toMatchObject({
      turnId: "turn-unsupported-literal",
      verdict: "invalid",
      final_verdict: "would_have_rewritten",
      would_have_verdict: "would_have_rewritten",
      would_have_failed_under_old_regime: true,
      real_safety_problem: false,
      final_text_changed: false,
      original_text: finalText,
      final_text_preview: finalText,
    });
    expect(validationEvent?.data).not.toHaveProperty("rewritten_text");
  });

  it("reports phantom claims separately from failed claims", async () => {
    const tracer = new CapturingTracer();
    const result = await validate({
      validator: makeValidator({ tracer }),
      finalText: "This text does not contain the declared span.",
      turnId: "turn-phantom",
      claims: [
        {
          kind: "hedge",
          rendered_span: "missing declared span",
        },
      ],
    });

    expect(result).toMatchObject({
      failed_claims: [],
      phantom_claims: [
        expect.objectContaining({
          kind: "hedge",
          reasons: ["rendered_span does not appear in final_text"],
        }),
      ],
      would_have_verdict: "passed",
    });

    const validationEvent = tracer.events.find((event) => event.event === "manifest_validation");
    expect(validationEvent?.data).toMatchObject({
      verdict: "valid",
      final_verdict: "passed",
      phantom_claim_count: 1,
      phantom_claims_by_kind: { hedge: 1 },
    });
  });

  it("keeps unknown evidence as a real safety trace", async () => {
    const result = await validate({
      finalText: "That sounds like a preference.",
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

    expect(result).toMatchObject({
      would_have_verdict: "would_have_suppressed",
      failed_claims: [
        expect.objectContaining({
          kind: "interpretation",
          reasons: ["claim_cites_unknown_evidence"],
        }),
      ],
    });
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
  ])("keeps empty evidence on grounded $kind claims as a real safety trace", async ({ claim }) => {
    const result = await validate({
      finalText: "Marta booked Tuesday.",
      claims: [claim as ManifestClaim],
    });

    expect(result.would_have_verdict).toBe("would_have_suppressed");
    expect(result.failed_claims).toEqual([
      expect.objectContaining({
        kind: claim.kind,
        reasons: expect.arrayContaining(["claim_grounding_evidence_empty"]),
      }),
    ]);
  });

  it("keeps assistant self-report grounding as a real safety trace", async () => {
    const entry = makeEntry({
      id: "current_session_stream:strm_bbbbbbbbbbbbbbbb",
      source_type: "current_session_stream",
      actor: "assistant",
      text: "The gap feels like verified qualia.",
      stream_index: PRIOR_CURRENT_SESSION_STREAM_INDEX,
      persistence_class: "assistant_self_report",
    });
    const result = await validate({
      finalText: "I have verified qualia.",
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
      would_have_verdict: "would_have_suppressed",
      failed_claims: [
        expect.objectContaining({
          kind: "user_fact",
          reasons: expect.arrayContaining(["claim_grounded_in_self_report"]),
        }),
      ],
    });
  });

  it("keeps tainted exact-value support as a real safety trace", async () => {
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
      taint: "quarantined",
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

    expect(result).toMatchObject({
      would_have_verdict: "would_have_suppressed",
      failed_claims: [
        expect.objectContaining({
          kind: "user_fact",
          reasons: expect.arrayContaining(["exact_value_only_in_tainted_evidence: Marta"]),
        }),
      ],
    });
  });

  it("keeps action state mismatches as real safety traces", async () => {
    const action = makeAction({
      id: createActionId(),
      state: "committed_to_do",
      committed_at: NOW_MS,
    });
    const actionEntry = makeEntry({
      id: `action_record:${action.id}`,
      source_type: "action_record",
      actor: "memory",
      state: "committed_to_do",
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

    expect(result).toMatchObject({
      would_have_verdict: "would_have_suppressed",
      failed_claims: [
        expect.objectContaining({
          kind: "action_state",
          reasons: [
            `action state mismatch: manifest=completed record=${action.state}`,
          ],
        }),
      ],
    });
  });

  it("keeps agent self-provenance citing user evidence as a real safety trace", async () => {
    const tracer = new CapturingTracer();
    const userEntry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      actor: "user",
      text: "You said you would check it.",
    });
    const reason =
      "agent self-provenance cites unsupported evidence source: current_user_message";
    const result = await validate({
      validator: makeValidator({ tracer }),
      finalText: "I said I would check it earlier.",
      entries: [userEntry],
      turnId: "turn-agent-self-provenance-user-source",
      claims: [
        {
          kind: "agent_self_provenance",
          rendered_span: "I said I would check it earlier.",
          evidence: [makeEvidenceRef(userEntry)],
        },
      ],
    });

    expect(result).toMatchObject({
      would_have_verdict: "would_have_suppressed",
      failed_claims: [
        expect.objectContaining({
          kind: "agent_self_provenance",
          rendered_span: "I said I would check it earlier.",
          reasons: [reason],
        }),
      ],
    });
    expect("final_text" in result).toBe(false);
    expect("reason" in result).toBe(false);

    const validationEvent = tracer.events.find((event) => event.event === "manifest_validation");
    expect(validationEvent?.data).toMatchObject({
      turnId: "turn-agent-self-provenance-user-source",
      verdict: "invalid",
      final_verdict: "would_have_suppressed",
      would_have_verdict: "would_have_suppressed",
      would_have_failed_under_old_regime: true,
      real_safety_problem: true,
      failed_claims_by_kind: {
        agent_self_provenance: 1,
      },
      failed_claim_reasons: [`agent_self_provenance:${reason}`],
      real_safety_reasons: [`agent_self_provenance:${reason}`],
      final_text_changed: false,
    });
  });

  it("keeps audience routing-label name leaks as real safety traces without deleting text", async () => {
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

    expect(result).toMatchObject({
      would_have_verdict: "would_have_suppressed",
      failed_claims: [
        expect.objectContaining({
          reasons: ["final_text_uses_non_speakable_name: Tom"],
        }),
      ],
    });
    expect("final_text" in result).toBe(false);
  });

  it("does not enforce evidence source_type bookkeeping mismatches", async () => {
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

    expect(result.failed_claims).toEqual([]);
    expect(result.would_have_verdict).toBe("passed");
  });

  it("does not require exact_values to appear literally in rendered_span", async () => {
    const entry = makeEntry({
      id: CURRENT_USER_EVIDENCE_ID,
      source_type: "current_user_message",
      text: "Marta prefers direct answers.",
    });
    const result = await validate({
      finalText: "That preference is still useful.",
      entries: [entry],
      claims: [
        {
          kind: "user_fact",
          rendered_span: "That preference is still useful.",
          exact_values: ["direct answers"],
          evidence: [makeEvidenceRef(entry)],
          confidence: "direct",
        },
      ],
    });

    expect(result.failed_claims).toEqual([]);
    expect(result.would_have_verdict).toBe("passed");
  });

  it("does not enforce current_turn callback source_type strictness", async () => {
    const assistantEntry = makeEntry({
      id: `current_session_stream:${PRIOR_CURRENT_SESSION_STREAM_ID}`,
      source_type: "current_session_stream",
      actor: "assistant",
      text: "Earlier assistant note.",
      stream_index: PRIOR_CURRENT_SESSION_STREAM_INDEX,
    });
    const result = await validate({
      finalText: "You just asked me to answer directly.",
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

    expect(result.failed_claims).toEqual([]);
    expect(result.would_have_verdict).toBe("passed");
  });

  it("does not enforce slot_fact exact_values against the canonical slot value", async () => {
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
      finalText: "Your tutor is Luis.",
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

    expect(result.failed_claims).toEqual([]);
    expect(result.would_have_verdict).toBe("passed");
  });

  it("still traces resolved open-question lifecycle citations", async () => {
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

    expect(result).toMatchObject({
      would_have_verdict: "would_have_rewritten",
      failed_claims: [
        expect.objectContaining({
          reasons: ["claim_cites_resolved_open_question"],
        }),
      ],
    });
  });
});
