import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import {
  CommitmentRepository,
  commitmentMigrations,
  commitmentSchema,
  type CommitmentRecord,
} from "../../memory/commitments/index.js";
import type { IdentityService } from "../../memory/identity/index.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import {
  DEFAULT_SESSION_ID,
  createCommitmentId,
  createEntityId,
  createRelationalSlotId,
  createStreamEntryId,
  type CommitmentId,
} from "../../util/ids.js";
import type { RelationshipClaim } from "../../memory/common/relationship-claims.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import { CorrectivePreferenceTurnService } from "./corrective-preference-service.js";

type AddCommitmentInput = Parameters<IdentityService["addCommitment"]>[0];

const defaultCorrectiveTurnContext = {
  isUserTurn: true,
  currentSenderEntityId: null,
  currentSenderBorgRole: null,
  sessionAudienceRole: "participant" as const,
};

function correctivePreferenceResponse(
  input: {
    supersedesCommitmentId?: CommitmentId | null;
    type?: "preference" | "rule" | "boundary";
    kind?: "audience_rule" | "participant_preference" | "boundary" | "process_norm";
    enforcementClass?: "critical" | "advisory";
    criticalDomain?:
      | "privacy"
      | "audience_scope"
      | "safety"
      | "explicit_no_disclosure"
      | "internal_tool_hygiene"
      | null;
    directive?: string;
    directiveFamily?: string;
    relationshipClaims?: RelationshipClaim[];
    appliesToAudienceEntityId?: string | null;
  } = {},
) {
  return {
    text: "",
    input_tokens: 6,
    output_tokens: 3,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_corrective",
        name: "EmitCorrectivePreference",
        input: {
          classification: "corrective_preference",
          type: input.type ?? "preference",
          kind: input.kind ?? "participant_preference",
          enforcement_class: input.enforcementClass ?? "advisory",
          critical_domain: input.criticalDomain ?? null,
          directive: input.directive ?? "Keep Alice's trip tasks separate from the group channel.",
          directive_family: input.directiveFamily ?? "separate_trip_tasks",
          closure_pressure_relevance: "neutral",
          priority: 8,
          reason: "The current speaker made a durable correction.",
          confidence: 0.91,
          supersedes_commitment_id: input.supersedesCommitmentId ?? null,
          retires_commitment_id: null,
          applies_to_audience_entity_id: input.appliesToAudienceEntityId ?? null,
          relationship_claims: input.relationshipClaims ?? [],
          slot_negations: [],
        },
      },
    ],
  };
}

function retireCommitmentResponse(input: {
  commitmentId: CommitmentId;
  reason?: string;
  confidence?: number;
}) {
  return {
    text: "",
    input_tokens: 6,
    output_tokens: 3,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_corrective_retire",
        name: "EmitCorrectivePreference",
        input: {
          classification: "retire_commitment",
          type: null,
          kind: null,
          enforcement_class: null,
          critical_domain: null,
          directive: null,
          directive_family: null,
          closure_pressure_relevance: null,
          priority: null,
          reason: input.reason ?? "The model judged the active commitment no longer applicable.",
          confidence: input.confidence ?? 0.91,
          supersedes_commitment_id: null,
          retires_commitment_id: input.commitmentId,
          applies_to_audience_entity_id: null,
          relationship_claims: [],
          slot_negations: [],
        },
      },
    ],
  };
}

function relationshipClaim(overrides: Partial<RelationshipClaim> = {}): RelationshipClaim {
  return {
    label_family: "kinship",
    subject_entity_id: null,
    object_entity_id: null,
    object_text: "relación familiar",
    requires_grounding: true,
    evidence_relational_slot_ids: [],
    evidence_stream_entry_ids: [],
    ...overrides,
  };
}

function persistedCommitmentFromInput(input: AddCommitmentInput): CommitmentRecord {
  return commitmentFixture({
    id: input.id ?? createCommitmentId(),
    kind: input.kind ?? "participant_preference",
    enforcement_class: input.enforcementClass ?? "advisory",
    critical_domain: input.criticalDomain ?? null,
    directive: input.directive,
    directive_family: input.directiveFamily,
    priority: input.priority,
    restricted_audience: input.restrictedAudience ?? null,
    committed_by_entity_id: input.committedByEntityId ?? null,
    created_at: input.createdAt,
  });
}

function commitmentFixture(
  overrides: Partial<CommitmentRecord> & Pick<CommitmentRecord, "id">,
): CommitmentRecord {
  return commitmentSchema.parse({
    id: overrides.id,
    record_version: overrides.record_version ?? 1,
    type: overrides.type ?? "preference",
    kind: overrides.kind ?? "participant_preference",
    enforcement_class: overrides.enforcement_class ?? "advisory",
    critical_domain: overrides.critical_domain ?? null,
    directive_family: overrides.directive_family ?? "separate_trip_tasks",
    closure_pressure_relevance: overrides.closure_pressure_relevance ?? "neutral",
    directive: overrides.directive ?? "Keep Alice's trip tasks separate.",
    priority: overrides.priority ?? 8,
    made_to_entity: overrides.made_to_entity ?? null,
    restricted_audience: overrides.restricted_audience ?? null,
    about_entity: overrides.about_entity ?? null,
    committed_by_entity_id: overrides.committed_by_entity_id ?? null,
    provenance: overrides.provenance ?? {
      kind: "online",
      process: "corrective-preference-extractor",
    },
    source_stream_entry_ids: overrides.source_stream_entry_ids,
    created_at: overrides.created_at ?? 2_000,
    expires_at: overrides.expires_at ?? null,
    expired_at: overrides.expired_at ?? null,
    revoked_at: overrides.revoked_at ?? null,
    revoked_reason: overrides.revoked_reason ?? null,
    revoke_provenance: overrides.revoke_provenance ?? null,
    superseded_by: overrides.superseded_by ?? null,
    canonicalized_by_artifact_entry_id: overrides.canonicalized_by_artifact_entry_id ?? null,
    last_reinforced_at: overrides.last_reinforced_at ?? 2_000,
  });
}

function createRetirementService(input: {
  activeCommitments: readonly CommitmentRecord[];
  getCommitment: CommitmentRecord | null;
  revoke?: CommitmentRepository["revoke"];
  addCommitment?: IdentityService["addCommitment"];
  tracer?: TurnTracer;
  clock?: FixedClock;
}) {
  const addCommitment = input.addCommitment ?? vi.fn<IdentityService["addCommitment"]>();
  const revoke =
    input.revoke ??
    vi.fn<CommitmentRepository["revoke"]>((id, reason, provenance) =>
      input.getCommitment === null
        ? null
        : commitmentFixture({
            ...input.getCommitment,
            id,
            revoked_at: input.clock?.now() ?? 2_000,
            revoked_reason: reason,
            revoke_provenance: provenance,
          }),
    );
  const service = new CorrectivePreferenceTurnService({
    model: "haiku",
    commitmentRepository: {
      get: () => input.getCommitment,
      getApplicable: () => [...input.activeCommitments],
      revoke,
      supersede: vi.fn(),
    },
    identityService: { addCommitment },
    relationalSlotRepository: {
      list: () => [],
      applyNegation: vi.fn(),
    },
    workingMemoryStore: {
      load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
      sanitizePendingActionsForRelationalSlot: vi.fn(),
    },
    clock: input.clock ?? new FixedClock(2_000),
    tracer: input.tracer ?? { enabled: false, includePayloads: false, emit: vi.fn() },
  });

  return { service, addCommitment, revoke };
}

async function extractAndPersistRetirement(input: {
  service: CorrectivePreferenceTurnService;
  llm: FakeLLMClient;
  turnId?: string;
}) {
  const turnId = input.turnId ?? "turn-retire-commitment";
  const result = await input.service.extractAndApply({
    llmClient: input.llm,
    turnId,
    ...defaultCorrectiveTurnContext,
    userMessage: "That standing commitment can be stood down now.",
    persistedUserEntryId: createStreamEntryId(),
    recentHistory: [],
    audienceEntityId: null,
    sessionId: DEFAULT_SESSION_ID,
    onHookFailure: vi.fn(),
    trackAppliedSlotNegation: vi.fn(),
  });

  await input.service.persistCommitment({
    commitment: result.commitment,
    supersession: result.commitmentSupersession,
    retirement: result.commitmentRetirement,
    turnId,
    sessionId: DEFAULT_SESSION_ID,
    onHookFailure: vi.fn(),
  });

  return result;
}

describe("CorrectivePreferenceTurnService", () => {
  it("builds group-chat corrective commitments with the speaker as committer", async () => {
    const group = createEntityId();
    const alice = createEntityId();
    const userEntryId = createStreamEntryId();
    const addCommitment = vi.fn();
    const llm = new FakeLLMClient({
      responses: [correctivePreferenceResponse()],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
        revoke: vi.fn(),
        supersede: vi.fn(),
      },
      identityService: { addCommitment },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer: { enabled: false, includePayloads: false, emit: vi.fn() },
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-group-commitment",
      ...defaultCorrectiveTurnContext,
      userMessage: "For me, keep my trip tasks separate from the channel.",
      persistedUserEntryId: userEntryId,
      recentHistory: [],
      audienceEntityId: group,
      committedByEntityId: alice,
      speakerDisplayName: "Alice",
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(result.commitment).toMatchObject({
      restricted_audience: group,
      committed_by_entity_id: alice,
      source_stream_entry_ids: [userEntryId],
    });

    await service.persistCommitment({
      commitment: result.commitment,
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
    });

    expect(addCommitment).toHaveBeenCalledWith(
      expect.objectContaining({
        restrictedAudience: group,
        committedByEntityId: alice,
      }),
    );
    expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain(
      `"speaker_entity_id":"${alice}"`,
    );
  });

  it("applies classification normalization before building a corrective commitment", async () => {
    const userEntryId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          type: "preference",
          kind: "participant_preference",
          enforcementClass: "critical",
          criticalDomain: "internal_tool_hygiene",
          directive:
            "Surface durable decisions and held context in explicit language at natural wrap points.",
          directiveFamily: "surface_durable_decisions",
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
        revoke: vi.fn(),
        supersede: vi.fn(),
      },
      identityService: { addCommitment: vi.fn() },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer: { enabled: false, includePayloads: false, emit: vi.fn() },
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-compaction-style-commitment",
      ...defaultCorrectiveTurnContext,
      userMessage:
        "From now on, surface durable decisions and held context explicitly at wrap points.",
      persistedUserEntryId: userEntryId,
      recentHistory: [],
      audienceEntityId: null,
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(result.commitment).toMatchObject({
      kind: "participant_preference",
      type: "preference",
      enforcement_class: "advisory",
      critical_domain: null,
      directive_family: "surface_durable_decisions",
    });
  });

  it.each([
    {
      label: "participant preference boundary explicit no-disclosure stays critical",
      type: "boundary",
      kind: "participant_preference",
      enforcementClass: "critical",
      criticalDomain: "explicit_no_disclosure",
      directive: "Do not disclose the deployment freeze discussion to the vendor channel.",
      directiveFamily: "e2e_explicit_no_disclosure_boundary",
      expectedEnforcementClass: "critical",
      expectedCriticalDomain: "explicit_no_disclosure",
      downgradeReason: null,
    },
    {
      label: "participant preference no-disclosure preference downgrades",
      type: "preference",
      kind: "participant_preference",
      enforcementClass: "critical",
      criticalDomain: "explicit_no_disclosure",
      directive:
        "Prefer not to phrase deployment freeze details as something for the vendor channel.",
      directiveFamily: "e2e_explicit_no_disclosure_preference",
      expectedEnforcementClass: "advisory",
      expectedCriticalDomain: null,
      downgradeReason: "explicit_no_disclosure_without_boundary_type",
    },
    {
      label: "process norm critical safety downgrades",
      type: "rule",
      kind: "process_norm",
      enforcementClass: "critical",
      criticalDomain: "safety",
      directive: "Record future rollback decisions in the shared state before summarizing.",
      directiveFamily: "e2e_process_norm_safety",
      expectedEnforcementClass: "advisory",
      expectedCriticalDomain: null,
      downgradeReason: "process_norm_classified_critical",
    },
    {
      label: "participant preference internal tool hygiene downgrades",
      type: "preference",
      kind: "participant_preference",
      enforcementClass: "critical",
      criticalDomain: "internal_tool_hygiene",
      directive:
        "Prefer concise references to prior notes without exposing internal tool mechanics.",
      directiveFamily: "e2e_internal_tool_hygiene_preference",
      expectedEnforcementClass: "advisory",
      expectedCriticalDomain: null,
      downgradeReason: "preference_with_internal_tool_hygiene",
    },
  ] as const)("persists normalized corrective classifications: $label", async (testCase) => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const clock = new FixedClock(2_000);
    const commitmentRepository = new CommitmentRepository({
      db,
      clock,
    });
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
    const userEntryId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          type: testCase.type,
          kind: testCase.kind,
          enforcementClass: testCase.enforcementClass,
          criticalDomain: testCase.criticalDomain,
          directive: testCase.directive,
          directiveFamily: testCase.directiveFamily,
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository,
      identityService: {
        addCommitment: (input) => commitmentRepository.add(input),
      },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock,
      tracer,
    });
    const turnId = `turn-${testCase.directiveFamily}`;

    try {
      const result = await service.extractAndApply({
        llmClient: llm,
        turnId,
        ...defaultCorrectiveTurnContext,
        userMessage: testCase.directive,
        persistedUserEntryId: userEntryId,
        recentHistory: [],
        audienceEntityId: null,
        sessionId: DEFAULT_SESSION_ID,
        onHookFailure: vi.fn(),
        trackAppliedSlotNegation: vi.fn(),
      });

      if (result.commitment === null) {
        throw new Error("Expected corrective commitment fixture");
      }

      await service.persistCommitment({
        commitment: result.commitment,
        supersession: result.commitmentSupersession,
        turnId,
        sessionId: DEFAULT_SESSION_ID,
        onHookFailure: vi.fn(),
      });

      const persisted = commitmentRepository.get(result.commitment.id);

      expect(persisted).toMatchObject({
        id: result.commitment.id,
        type: testCase.type,
        kind: testCase.kind,
        enforcement_class: testCase.expectedEnforcementClass,
        critical_domain: testCase.expectedCriticalDomain,
        directive_family: testCase.directiveFamily,
        source_stream_entry_ids: [userEntryId],
      });

      if (testCase.downgradeReason === null) {
        expect(tracer.emit).not.toHaveBeenCalledWith(
          "commitment_classification.downgraded",
          expect.anything(),
        );
      } else {
        expect(tracer.emit).toHaveBeenCalledWith(
          "commitment_classification.downgraded",
          expect.objectContaining({
            turnId,
            reason: testCase.downgradeReason,
            kind: testCase.kind,
            type: testCase.type,
            directive_family: testCase.directiveFamily,
            new_enforcement_class: "advisory",
            new_critical_domain: null,
          }),
        );
      }
    } finally {
      db.close();
    }
  });

  it("keeps corrective candidates with medical context nouns without relationship evidence", async () => {
    const userEntryId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          directive:
            "Do not mark the doctor appointment as booked until the user confirms it, and mention patient portal issues as administrative status.",
          directiveFamily: "appointment_status_confirmation",
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
        revoke: vi.fn(),
        supersede: vi.fn(),
      },
      identityService: { addCommitment: vi.fn() },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer: { enabled: false, includePayloads: false, emit: vi.fn() },
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-medical-context-noun-commitment",
      ...defaultCorrectiveTurnContext,
      userMessage: "Keep appointment status precise.",
      persistedUserEntryId: userEntryId,
      recentHistory: [],
      audienceEntityId: null,
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(result.commitment).toMatchObject({
      directive_family: "appointment_status_confirmation",
    });
  });

  it("skips corrective candidates with ungrounded relationship claims", async () => {
    const userEntryId = createStreamEntryId();
    const tracer = { enabled: true, includePayloads: false, emit: vi.fn() };
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          directive: "Use the parent constraint for future care-planning replies.",
          directiveFamily: "care_planning_parent_constraint",
          relationshipClaims: [relationshipClaim({ object_text: "mi madre" })],
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
        revoke: vi.fn(),
        supersede: vi.fn(),
      },
      identityService: { addCommitment: vi.fn() },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer,
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-ungrounded-relationship-commitment",
      ...defaultCorrectiveTurnContext,
      userMessage: "Make sure future replies respect the care-planning constraint.",
      persistedUserEntryId: userEntryId,
      recentHistory: [],
      audienceEntityId: null,
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(result.commitment).toBeNull();
    expect(result.commitmentSupersession).toBeNull();
    expect(tracer.emit).toHaveBeenCalledWith(
      "corrective_preference.candidate_rejected_ungrounded",
      expect.objectContaining({
        turnId: "turn-ungrounded-relationship-commitment",
        validationStatus: "rejected",
        reason: "relationship_claim_ungrounded",
        relationship_claim_label_families: ["kinship"],
        ungrounded_relationship_claims: [expect.objectContaining({ object_text: "mi madre" })],
      }),
    );
  });

  it("rejects corrective candidates grounded only by assistant stream evidence", async () => {
    const userEntryId = createStreamEntryId();
    const assistantEntryId = createStreamEntryId();
    const tracer = { enabled: true, includePayloads: false, emit: vi.fn() };
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          directive: "Use the parent constraint for future care-planning replies.",
          directiveFamily: "care_planning_parent_constraint",
          relationshipClaims: [
            relationshipClaim({
              evidence_stream_entry_ids: [assistantEntryId],
            }),
          ],
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
        revoke: vi.fn(),
        supersede: vi.fn(),
      },
      identityService: { addCommitment: vi.fn() },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer,
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-assistant-evidence-relationship-commitment",
      ...defaultCorrectiveTurnContext,
      userMessage: "Make sure future replies respect the care-planning constraint.",
      persistedUserEntryId: userEntryId,
      relationshipEvidenceStreamEntries: [{ id: assistantEntryId, kind: "agent_msg" }],
      recentHistory: [],
      audienceEntityId: null,
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(result.commitment).toBeNull();
    expect(tracer.emit).toHaveBeenCalledWith(
      "corrective_preference.candidate_rejected_ungrounded",
      expect.objectContaining({
        relationship_claim_label_families: ["kinship"],
        relationship_claims: [
          expect.objectContaining({
            evidence_stream_entry_ids: [assistantEntryId],
          }),
        ],
        rejected_relationship_claim_evidence_stream_entry_ids: [
          {
            id: assistantEntryId,
            reason: "not_user_msg",
          },
        ],
      }),
    );
  });

  it("rejects corrective candidates grounded only by out-of-bundle stream evidence", async () => {
    const userEntryId = createStreamEntryId();
    const outsideEntryId = createStreamEntryId();
    const tracer = { enabled: true, includePayloads: false, emit: vi.fn() };
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          directive: "Use the parent constraint for future care-planning replies.",
          directiveFamily: "care_planning_parent_constraint",
          relationshipClaims: [
            relationshipClaim({
              evidence_stream_entry_ids: [outsideEntryId],
            }),
          ],
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
        revoke: vi.fn(),
        supersede: vi.fn(),
      },
      identityService: { addCommitment: vi.fn() },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer,
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-outside-evidence-relationship-commitment",
      ...defaultCorrectiveTurnContext,
      userMessage: "Make sure future replies respect the care-planning constraint.",
      persistedUserEntryId: userEntryId,
      recentHistory: [],
      audienceEntityId: null,
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(result.commitment).toBeNull();
    expect(tracer.emit).toHaveBeenCalledWith(
      "corrective_preference.candidate_rejected_ungrounded",
      expect.objectContaining({
        relationship_claim_label_families: ["kinship"],
        relationship_claims: [
          expect.objectContaining({
            evidence_stream_entry_ids: [outsideEntryId],
          }),
        ],
        rejected_relationship_claim_evidence_stream_entry_ids: [
          {
            id: outsideEntryId,
            reason: "not_in_source_bundle",
          },
        ],
      }),
    );
  });

  it("keeps corrective candidates grounded by trusted user-message evidence", async () => {
    const userEntryId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          directive: "Use the parent constraint for future care-planning replies.",
          directiveFamily: "care_planning_parent_constraint",
          relationshipClaims: [
            relationshipClaim({
              evidence_stream_entry_ids: [userEntryId],
            }),
          ],
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
        revoke: vi.fn(),
        supersede: vi.fn(),
      },
      identityService: { addCommitment: vi.fn() },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit: vi.fn() },
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-grounded-relationship-commitment",
      ...defaultCorrectiveTurnContext,
      userMessage: "My current user message explicitly grounds the parent constraint.",
      persistedUserEntryId: userEntryId,
      recentHistory: [],
      audienceEntityId: null,
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(result.commitment).toMatchObject({
      directive: "Use the parent constraint for future care-planning replies.",
      source_stream_entry_ids: [userEntryId],
    });
  });

  it("keeps corrective candidates grounded by roster relational slot evidence", async () => {
    const slotId = createRelationalSlotId();
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          directive: "Use the parent constraint for future care-planning replies.",
          directiveFamily: "care_planning_parent_constraint",
          relationshipClaims: [
            relationshipClaim({
              evidence_relational_slot_ids: [slotId],
            }),
          ],
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
        revoke: vi.fn(),
        supersede: vi.fn(),
      },
      identityService: { addCommitment: vi.fn() },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit: vi.fn() },
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-roster-grounded-relationship-commitment",
      ...defaultCorrectiveTurnContext,
      userMessage: "The roster grounds this family role.",
      persistedUserEntryId: createStreamEntryId(),
      recentHistory: [],
      audienceEntityId: null,
      participantRoster: {
        participants: [
          {
            entity_id: createEntityId(),
            display_name: "Avery",
            known_relationships: ["parent.name:Robin"],
            audience_role: "speaker",
            relationship_source: `relational_slot:${slotId}`,
          },
        ],
        non_chat_subjects: [],
        unknown_or_uncertain: [],
      },
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(result.commitment).toMatchObject({
      directive: "Use the parent constraint for future care-planning replies.",
    });
  });

  it("supersedes a visible active commitment selected by the extractor", async () => {
    const supersededId = createCommitmentId();
    const target = commitmentFixture({ id: supersededId });
    const addCommitment = vi.fn((input: AddCommitmentInput) => persistedCommitmentFromInput(input));
    const supersede = vi.fn((_id: CommitmentId, nextId: CommitmentId) =>
      commitmentFixture({
        ...target,
        superseded_by: nextId,
      }),
    );
    const tracer = { enabled: true, includePayloads: false, emit: vi.fn() };
    const llm = new FakeLLMClient({
      responses: [correctivePreferenceResponse({ supersedesCommitmentId: supersededId })],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => target,
        getApplicable: () => [target],
        revoke: vi.fn(),
        supersede,
      },
      identityService: { addCommitment },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer,
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-valid-supersession",
      ...defaultCorrectiveTurnContext,
      userMessage: "Actually, keep Alice's trip tasks separate.",
      persistedUserEntryId: createStreamEntryId(),
      recentHistory: [],
      audienceEntityId: null,
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    await service.persistCommitment({
      commitment: result.commitment,
      supersession: result.commitmentSupersession,
      turnId: "turn-valid-supersession",
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
    });

    expect(supersede).toHaveBeenCalledWith(supersededId, addCommitment.mock.results[0]?.value.id);
    expect(addCommitment).toHaveBeenCalledWith(
      expect.objectContaining({
        skipDirectiveFamilyMerge: true,
      }),
    );
    expect(tracer.emit).toHaveBeenCalledWith("extraction.commitments.transitioned", {
      turnId: "turn-valid-supersession",
      session_id: DEFAULT_SESSION_ID,
      supersededId,
      newId: addCommitment.mock.results[0]?.value.id,
      validationStatus: "accepted",
    });
  });

  it("rejects a supersession id outside the active visible allowed list", async () => {
    const supersededId = createCommitmentId();
    const addCommitment = vi.fn((input: AddCommitmentInput) => persistedCommitmentFromInput(input));
    const supersede = vi.fn();
    const tracer = { enabled: true, includePayloads: false, emit: vi.fn() };
    const llm = new FakeLLMClient({
      responses: [correctivePreferenceResponse({ supersedesCommitmentId: supersededId })],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
        revoke: vi.fn(),
        supersede,
      },
      identityService: { addCommitment },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer,
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-invalid-supersession",
      ...defaultCorrectiveTurnContext,
      userMessage: "Actually, keep Alice's trip tasks separate.",
      persistedUserEntryId: createStreamEntryId(),
      recentHistory: [],
      audienceEntityId: null,
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    await service.persistCommitment({
      commitment: result.commitment,
      supersession: result.commitmentSupersession,
      turnId: "turn-invalid-supersession",
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
    });

    expect(supersede).not.toHaveBeenCalled();
    expect(addCommitment.mock.calls[0]?.[0]).not.toHaveProperty("skipDirectiveFamilyMerge");
    expect(tracer.emit).toHaveBeenCalledWith("extraction.commitments.rejected", {
      turnId: "turn-invalid-supersession",
      session_id: DEFAULT_SESSION_ID,
      supersededId,
      validationStatus: "rejected",
      reason: "not_in_allowed_active_commitments",
    });
  });

  it("rejects a visible supersession target that is no longer active at persistence", async () => {
    const supersededId = createCommitmentId();
    const visibleTarget = commitmentFixture({ id: supersededId });
    const revokedTarget = commitmentFixture({
      id: supersededId,
      revoked_at: 2_100,
      revoked_reason: "test revocation",
      revoke_provenance: { kind: "manual" },
    });
    const addCommitment = vi.fn((input: AddCommitmentInput) => persistedCommitmentFromInput(input));
    const supersede = vi.fn();
    const tracer = { enabled: true, includePayloads: false, emit: vi.fn() };
    const llm = new FakeLLMClient({
      responses: [correctivePreferenceResponse({ supersedesCommitmentId: supersededId })],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => revokedTarget,
        getApplicable: () => [visibleTarget],
        revoke: vi.fn(),
        supersede,
      },
      identityService: { addCommitment },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_200),
      tracer,
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-revoked-supersession",
      ...defaultCorrectiveTurnContext,
      userMessage: "Actually, keep Alice's trip tasks separate.",
      persistedUserEntryId: createStreamEntryId(),
      recentHistory: [],
      audienceEntityId: null,
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    await service.persistCommitment({
      commitment: result.commitment,
      supersession: result.commitmentSupersession,
      turnId: "turn-revoked-supersession",
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
    });

    expect(supersede).not.toHaveBeenCalled();
    expect(addCommitment.mock.calls[0]?.[0]).not.toHaveProperty("skipDirectiveFamilyMerge");
    expect(tracer.emit).toHaveBeenCalledWith("extraction.commitments.rejected", {
      turnId: "turn-revoked-supersession",
      session_id: DEFAULT_SESSION_ID,
      supersededId,
      validationStatus: "rejected",
      reason: "commitment_not_active",
    });
  });

  it("retires an eligible active commitment with online corrective-preference provenance", async () => {
    const retiredId = createCommitmentId();
    const target = commitmentFixture({ id: retiredId });
    const emit = vi.fn();
    const tracer = { enabled: true, includePayloads: false, emit };
    const llm = new FakeLLMClient({
      responses: [
        retireCommitmentResponse({
          commitmentId: retiredId,
          reason: "The temporary audience boundary is resolved.",
        }),
      ],
    });
    const { service, addCommitment, revoke } = createRetirementService({
      activeCommitments: [target],
      getCommitment: target,
      tracer,
    });

    const result = await extractAndPersistRetirement({
      service,
      llm,
      turnId: "turn-valid-retirement",
    });

    expect(result.commitment).toBeNull();
    expect(result.commitmentSupersession).toBeNull();
    expect(result.commitmentRetirement).toMatchObject({
      retiredId,
      allowedActiveCommitmentIds: [retiredId],
      reason: "The temporary audience boundary is resolved.",
      confidence: 0.91,
    });
    expect(addCommitment).not.toHaveBeenCalled();
    expect(revoke).toHaveBeenCalledWith(
      retiredId,
      "The temporary audience boundary is resolved.",
      {
        kind: "online",
        process: "corrective-preference-extractor",
      },
      undefined,
      { expectedVersion: target.record_version },
    );
    expect(emit).toHaveBeenCalledWith("extraction.commitments.transitioned", {
      turnId: "turn-valid-retirement",
      session_id: DEFAULT_SESSION_ID,
      retiredId,
      validationStatus: "accepted",
      reason: "retired_by_corrective_preference",
    });
  });

  it("rejects a retirement id outside the active visible allowed list", async () => {
    const retiredId = createCommitmentId();
    const target = commitmentFixture({ id: retiredId });
    const emit = vi.fn();
    const llm = new FakeLLMClient({
      responses: [retireCommitmentResponse({ commitmentId: retiredId })],
    });
    const { service, revoke } = createRetirementService({
      activeCommitments: [],
      getCommitment: target,
      tracer: { enabled: true, includePayloads: false, emit },
    });

    await extractAndPersistRetirement({
      service,
      llm,
      turnId: "turn-retirement-out-of-allowed-set",
    });

    expect(revoke).not.toHaveBeenCalled();
    expect(emit).toHaveBeenCalledWith("extraction.commitments.rejected", {
      turnId: "turn-retirement-out-of-allowed-set",
      session_id: DEFAULT_SESSION_ID,
      retiredId,
      validationStatus: "rejected",
      reason: "not_in_allowed_active_commitments",
    });
  });

  it("traces a stale retirement target when revoke detects a version conflict", async () => {
    const retiredId = createCommitmentId();
    const target = commitmentFixture({ id: retiredId, record_version: 7 });
    const emit = vi.fn();
    const revoke = vi.fn<CommitmentRepository["revoke"]>(() => {
      throw new IdentityCasMismatchError({
        recordType: "commitment",
        recordId: retiredId,
        expectedVersion: target.record_version ?? -1,
      });
    });
    const llm = new FakeLLMClient({
      responses: [retireCommitmentResponse({ commitmentId: retiredId })],
    });
    const { service } = createRetirementService({
      activeCommitments: [target],
      getCommitment: target,
      revoke,
      tracer: { enabled: true, includePayloads: false, emit },
    });

    await extractAndPersistRetirement({
      service,
      llm,
      turnId: "turn-retirement-version-conflict",
    });

    expect(revoke).toHaveBeenCalledWith(
      retiredId,
      "The model judged the active commitment no longer applicable.",
      {
        kind: "online",
        process: "corrective-preference-extractor",
      },
      undefined,
      { expectedVersion: 7 },
    );
    expect(emit).toHaveBeenCalledWith("extraction.commitments.rejected", {
      turnId: "turn-retirement-version-conflict",
      session_id: DEFAULT_SESSION_ID,
      retiredId,
      validationStatus: "rejected",
      reason: "commitment_version_conflict",
    });
  });

  it("rejects a retirement id that no longer exists at persistence", async () => {
    const retiredId = createCommitmentId();
    const visibleTarget = commitmentFixture({ id: retiredId });
    const emit = vi.fn();
    const llm = new FakeLLMClient({
      responses: [retireCommitmentResponse({ commitmentId: retiredId })],
    });
    const { service, revoke } = createRetirementService({
      activeCommitments: [visibleTarget],
      getCommitment: null,
      tracer: { enabled: true, includePayloads: false, emit },
    });

    await extractAndPersistRetirement({
      service,
      llm,
      turnId: "turn-retirement-unknown",
    });

    expect(revoke).not.toHaveBeenCalled();
    expect(emit).toHaveBeenCalledWith("extraction.commitments.rejected", {
      turnId: "turn-retirement-unknown",
      session_id: DEFAULT_SESSION_ID,
      retiredId,
      validationStatus: "rejected",
      reason: "unknown_commitment_id",
    });
  });

  it("rejects a visible retirement target that is no longer active at persistence", async () => {
    const retiredId = createCommitmentId();
    const visibleTarget = commitmentFixture({ id: retiredId });
    const revokedTarget = commitmentFixture({
      id: retiredId,
      revoked_at: 2_100,
      revoked_reason: "test revocation",
      revoke_provenance: { kind: "manual" },
    });
    const emit = vi.fn();
    const llm = new FakeLLMClient({
      responses: [retireCommitmentResponse({ commitmentId: retiredId })],
    });
    const { service, revoke } = createRetirementService({
      activeCommitments: [visibleTarget],
      getCommitment: revokedTarget,
      tracer: { enabled: true, includePayloads: false, emit },
      clock: new FixedClock(2_200),
    });

    await extractAndPersistRetirement({
      service,
      llm,
      turnId: "turn-retirement-inactive",
    });

    expect(revoke).not.toHaveBeenCalled();
    expect(emit).toHaveBeenCalledWith("extraction.commitments.rejected", {
      turnId: "turn-retirement-inactive",
      session_id: DEFAULT_SESSION_ID,
      retiredId,
      validationStatus: "rejected",
      reason: "commitment_not_active",
    });
  });

  it.each(["privacy", "safety", "explicit_no_disclosure", "internal_tool_hygiene"] as const)(
    "rejects retirement of critical %s commitments",
    async (criticalDomain) => {
      const retiredId = createCommitmentId();
      const target = commitmentFixture({
        id: retiredId,
        type: "boundary",
        kind: "boundary",
        enforcement_class: "critical",
        critical_domain: criticalDomain,
      });
      const emit = vi.fn();
      const llm = new FakeLLMClient({
        responses: [retireCommitmentResponse({ commitmentId: retiredId })],
      });
      const { service, revoke } = createRetirementService({
        activeCommitments: [target],
        getCommitment: target,
        tracer: { enabled: true, includePayloads: false, emit },
      });

      await extractAndPersistRetirement({
        service,
        llm,
        turnId: `turn-retirement-ineligible-${criticalDomain}`,
      });

      expect(revoke).not.toHaveBeenCalled();
      expect(emit).toHaveBeenCalledWith("extraction.commitments.rejected", {
        turnId: `turn-retirement-ineligible-${criticalDomain}`,
        session_id: DEFAULT_SESSION_ID,
        retiredId,
        validationStatus: "rejected",
        reason: "retirement_not_eligible",
      });
    },
  );

  it("rejects retirement of critical commitments with no explicit critical domain", async () => {
    const retiredId = createCommitmentId();
    const target = commitmentFixture({
      id: retiredId,
      type: "boundary",
      kind: "audience_rule",
      enforcement_class: "critical",
      critical_domain: null,
    });
    const emit = vi.fn();
    const llm = new FakeLLMClient({
      responses: [retireCommitmentResponse({ commitmentId: retiredId })],
    });
    const { service, revoke } = createRetirementService({
      activeCommitments: [target],
      getCommitment: target,
      tracer: { enabled: true, includePayloads: false, emit },
    });

    await extractAndPersistRetirement({
      service,
      llm,
      turnId: "turn-retirement-critical-null-domain",
    });

    expect(revoke).not.toHaveBeenCalled();
    expect(emit).toHaveBeenCalledWith("extraction.commitments.rejected", {
      turnId: "turn-retirement-critical-null-domain",
      session_id: DEFAULT_SESSION_ID,
      retiredId,
      validationStatus: "rejected",
      reason: "retirement_not_eligible",
    });
  });

  it("allows retirement of critical audience-scope commitments", async () => {
    const retiredId = createCommitmentId();
    const target = commitmentFixture({
      id: retiredId,
      type: "boundary",
      kind: "audience_rule",
      enforcement_class: "critical",
      critical_domain: "audience_scope",
    });
    const llm = new FakeLLMClient({
      responses: [retireCommitmentResponse({ commitmentId: retiredId })],
    });
    const { service, revoke } = createRetirementService({
      activeCommitments: [target],
      getCommitment: target,
    });

    await extractAndPersistRetirement({
      service,
      llm,
      turnId: "turn-retirement-audience-scope",
    });

    expect(revoke).toHaveBeenCalledWith(
      retiredId,
      "The model judged the active commitment no longer applicable.",
      {
        kind: "online",
        process: "corrective-preference-extractor",
      },
      undefined,
      { expectedVersion: target.record_version },
    );
  });
});

describe("CorrectivePreferenceTurnService cross-audience scoping", () => {
  function makeService(tracer?: TurnTracer) {
    const addCommitment = vi.fn();
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
        revoke: vi.fn(),
        supersede: vi.fn(),
      },
      identityService: { addCommitment },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer: tracer ?? { enabled: false, includePayloads: false, emit: vi.fn() },
    });

    return { service, addCommitment };
  }

  const audienceRuleResponse = (appliesToAudienceEntityId: string | null) =>
    correctivePreferenceResponse({
      type: "rule",
      kind: "audience_rule",
      directive: "In the crew channel, proactively flag deploy and incident risks.",
      directiveFamily: "crew_proactive_risk",
      appliesToAudienceEntityId,
    });

  function turnInput(input: {
    audienceEntityId: ReturnType<typeof createEntityId>;
    isUserTurn?: boolean;
    currentSenderEntityId?: ReturnType<typeof createEntityId> | null;
    currentSenderBorgRole?: "creator" | null;
    sessionAudienceRole?: "participant" | "operator";
    crossAudienceTargeting: {
      allowed: boolean;
      candidateAudiences: readonly {
        entity_id: ReturnType<typeof createEntityId>;
        label: string;
      }[];
    };
    appliesToAudienceEntityId: string | null;
  }) {
    return {
      llmClient: new FakeLLMClient({
        responses: [audienceRuleResponse(input.appliesToAudienceEntityId)],
      }),
      turnId: "turn-cross-audience",
      isUserTurn: input.isUserTurn ?? defaultCorrectiveTurnContext.isUserTurn,
      userMessage: "In the Project Crew channel, proactively flag deploy risks.",
      persistedUserEntryId: createStreamEntryId(),
      recentHistory: [],
      audienceEntityId: input.audienceEntityId,
      committedByEntityId: input.audienceEntityId,
      currentSenderEntityId:
        input.currentSenderEntityId ?? defaultCorrectiveTurnContext.currentSenderEntityId,
      currentSenderBorgRole:
        input.currentSenderBorgRole ?? defaultCorrectiveTurnContext.currentSenderBorgRole,
      sessionAudienceRole:
        input.sessionAudienceRole ?? defaultCorrectiveTurnContext.sessionAudienceRole,
      speakerDisplayName: "Tom",
      crossAudienceTargeting: input.crossAudienceTargeting,
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    };
  }

  it("scopes a commitment to another audience when authorized and the target is a candidate", async () => {
    const operatorAudience = createEntityId();
    const groupAudience = createEntityId();
    const emit = vi.fn();
    const { service } = makeService({ enabled: true, includePayloads: true, emit });

    const result = await service.extractAndApply(
      turnInput({
        audienceEntityId: operatorAudience,
        appliesToAudienceEntityId: groupAudience,
        crossAudienceTargeting: {
          allowed: true,
          candidateAudiences: [{ entity_id: groupAudience, label: "Project Crew" }],
        },
      }),
    );

    expect(result.commitment?.restricted_audience).toBe(groupAudience);
    expect(emit).toHaveBeenCalledWith(
      "corrective_preference.cross_audience_scope",
      expect.objectContaining({
        validationStatus: "accepted",
        reason: "cross_audience_scope_applied",
      }),
    );
  });

  it("falls back to the current audience when the target is not a supplied candidate", async () => {
    const operatorAudience = createEntityId();
    const groupAudience = createEntityId();
    const unlistedAudience = createEntityId();
    const { service } = makeService();

    const result = await service.extractAndApply(
      turnInput({
        audienceEntityId: operatorAudience,
        appliesToAudienceEntityId: unlistedAudience,
        crossAudienceTargeting: {
          allowed: true,
          candidateAudiences: [{ entity_id: groupAudience, label: "Project Crew" }],
        },
      }),
    );

    expect(result.commitment?.restricted_audience).toBe(operatorAudience);
  });

  it("ignores a cross-audience target when the turn is not authorized to cross-target", async () => {
    const participantAudience = createEntityId();
    const groupAudience = createEntityId();
    const { service } = makeService();

    const result = await service.extractAndApply(
      turnInput({
        audienceEntityId: participantAudience,
        // A hallucinated target on a non-creator-in-operator turn: the extraction
        // phase passes no candidates (allowed:false), and the service ignores it.
        appliesToAudienceEntityId: groupAudience,
        crossAudienceTargeting: { allowed: false, candidateAudiences: [] },
      }),
    );

    expect(result.commitment?.restricted_audience).toBe(participantAudience);
  });

  it("keeps the current audience when no cross-audience target is emitted", async () => {
    const operatorAudience = createEntityId();
    const groupAudience = createEntityId();
    const { service } = makeService();

    const result = await service.extractAndApply(
      turnInput({
        audienceEntityId: operatorAudience,
        appliesToAudienceEntityId: null,
        crossAudienceTargeting: {
          allowed: true,
          candidateAudiences: [{ entity_id: groupAudience, label: "Project Crew" }],
        },
      }),
    );

    expect(result.commitment?.restricted_audience).toBe(operatorAudience);
  });

  it("defers creator/operator cross-audience candidates to creator directives", async () => {
    const operatorAudience = createEntityId();
    const groupAudience = createEntityId();
    const creatorId = createEntityId();
    const emit = vi.fn();
    const { service, addCommitment } = makeService({
      enabled: true,
      includePayloads: false,
      emit,
    });

    const result = await service.extractAndApply(
      turnInput({
        audienceEntityId: operatorAudience,
        currentSenderEntityId: creatorId,
        currentSenderBorgRole: "creator",
        sessionAudienceRole: "operator",
        appliesToAudienceEntityId: groupAudience,
        crossAudienceTargeting: {
          allowed: true,
          candidateAudiences: [{ entity_id: groupAudience, label: "Project Crew" }],
        },
      }),
    );

    expect(result.commitment).toBeNull();
    expect(result.commitmentSupersession).toBeNull();

    await service.persistCommitment({
      commitment: result.commitment,
      supersession: result.commitmentSupersession,
      turnId: "turn-cross-audience",
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
    });

    expect(addCommitment).not.toHaveBeenCalled();
    expect(emit).toHaveBeenCalledWith(
      "corrective_preference.cross_audience_creator_deferred",
      expect.objectContaining({
        validationStatus: "deferred",
        reason: "creator_operator_cross_audience_deferred_to_creator_directive",
        requested_audience_entity_id: groupAudience,
        current_sender_entity_id: creatorId,
      }),
    );
  });

  it("suppresses creator/operator cross-audience candidates even when the target is unlisted (intentional lossy edge)", async () => {
    // Documents the safe/loss-tolerant edge of the authority partition: the
    // creator/operator deferral fires on ANY non-null cross-audience target,
    // BEFORE resolveCorrectiveRestrictedAudience would validate it and fall
    // back to the current audience (see the non-creator case above, which DOES
    // fall back). So a creator's unlisted/invalid cross-audience target is
    // suppressed -- NOT misfiled as a current-audience commitment. The
    // directive band owns operator cross-audience policy; a malformed target is
    // a safe miss, and the deferral trace records the requested target.
    const operatorAudience = createEntityId();
    const groupAudience = createEntityId();
    const unlistedAudience = createEntityId();
    const creatorId = createEntityId();
    const emit = vi.fn();
    const { service, addCommitment } = makeService({
      enabled: true,
      includePayloads: false,
      emit,
    });

    const result = await service.extractAndApply(
      turnInput({
        audienceEntityId: operatorAudience,
        currentSenderEntityId: creatorId,
        currentSenderBorgRole: "creator",
        sessionAudienceRole: "operator",
        appliesToAudienceEntityId: unlistedAudience,
        crossAudienceTargeting: {
          allowed: true,
          candidateAudiences: [{ entity_id: groupAudience, label: "Project Crew" }],
        },
      }),
    );

    expect(result.commitment).toBeNull();
    expect(result.commitmentSupersession).toBeNull();

    await service.persistCommitment({
      commitment: result.commitment,
      supersession: result.commitmentSupersession,
      turnId: "turn-cross-audience",
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
    });

    expect(addCommitment).not.toHaveBeenCalled();
    expect(emit).toHaveBeenCalledWith(
      "corrective_preference.cross_audience_creator_deferred",
      expect.objectContaining({
        validationStatus: "deferred",
        reason: "creator_operator_cross_audience_deferred_to_creator_directive",
        requested_audience_entity_id: unlistedAudience,
        current_sender_entity_id: creatorId,
      }),
    );
  });

  it("keeps non-creator cross-audience candidates in commitments", async () => {
    const operatorAudience = createEntityId();
    const groupAudience = createEntityId();
    const speakerId = createEntityId();
    const { service } = makeService();

    const result = await service.extractAndApply(
      turnInput({
        audienceEntityId: operatorAudience,
        currentSenderEntityId: speakerId,
        currentSenderBorgRole: null,
        sessionAudienceRole: "operator",
        appliesToAudienceEntityId: groupAudience,
        crossAudienceTargeting: {
          allowed: true,
          candidateAudiences: [{ entity_id: groupAudience, label: "Project Crew" }],
        },
      }),
    );

    expect(result.commitment).toMatchObject({
      kind: "audience_rule",
      restricted_audience: groupAudience,
    });
  });

  it("keeps creator/operator within-audience candidates in commitments", async () => {
    const operatorAudience = createEntityId();
    const groupAudience = createEntityId();
    const creatorId = createEntityId();
    const { service } = makeService();

    const result = await service.extractAndApply(
      turnInput({
        audienceEntityId: operatorAudience,
        currentSenderEntityId: creatorId,
        currentSenderBorgRole: "creator",
        sessionAudienceRole: "operator",
        appliesToAudienceEntityId: null,
        crossAudienceTargeting: {
          allowed: true,
          candidateAudiences: [{ entity_id: groupAudience, label: "Project Crew" }],
        },
      }),
    );

    expect(result.commitment).toMatchObject({
      kind: "audience_rule",
      restricted_audience: operatorAudience,
    });
  });

  it("keeps creator cross-audience candidates outside operator sessions in commitments", async () => {
    const participantAudience = createEntityId();
    const groupAudience = createEntityId();
    const creatorId = createEntityId();
    const { service } = makeService();

    const result = await service.extractAndApply(
      turnInput({
        audienceEntityId: participantAudience,
        currentSenderEntityId: creatorId,
        currentSenderBorgRole: "creator",
        sessionAudienceRole: "participant",
        appliesToAudienceEntityId: groupAudience,
        crossAudienceTargeting: { allowed: false, candidateAudiences: [] },
      }),
    );

    expect(result.commitment).toMatchObject({
      kind: "audience_rule",
      restricted_audience: participantAudience,
    });
  });
});
