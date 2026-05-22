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
import {
  DEFAULT_SESSION_ID,
  createCommitmentId,
  createEntityId,
  createRelationalSlotId,
  createStreamEntryId,
  type CommitmentId,
} from "../../util/ids.js";
import { CorrectivePreferenceTurnService } from "./corrective-preference-service.js";

type AddCommitmentInput = Parameters<IdentityService["addCommitment"]>[0];

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
    relationshipEvidenceRelationalSlotIds?: string[];
    relationshipEvidenceStreamEntryIds?: string[];
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
          relationship_evidence_relational_slot_ids:
            input.relationshipEvidenceRelationalSlotIds ?? [],
          relationship_evidence_stream_entry_ids: input.relationshipEvidenceStreamEntryIds ?? [],
          slot_negations: [],
        },
      },
    ],
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
      directive:
        "Do not disclose the deployment freeze discussion to the vendor channel.",
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

  it("skips corrective candidates with ungrounded protected relationship labels", async () => {
    const userEntryId = createStreamEntryId();
    const tracer = { enabled: true, includePayloads: false, emit: vi.fn() };
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          directive: "Use the parent constraint for future care-planning replies.",
          directiveFamily: "care_planning_parent_constraint",
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
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
        reason: "relationship_label_ungrounded",
        protected_relationship_labels: ["parent"],
        relationship_evidence_relational_slot_ids: [],
        relationship_evidence_stream_entry_ids: [],
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
          relationshipEvidenceStreamEntryIds: [assistantEntryId],
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
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
        protected_relationship_labels: ["parent"],
        relationship_evidence_stream_entry_ids: [assistantEntryId],
        rejected_relationship_evidence_stream_entry_ids: [
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
          relationshipEvidenceStreamEntryIds: [outsideEntryId],
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
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
        protected_relationship_labels: ["parent"],
        relationship_evidence_stream_entry_ids: [outsideEntryId],
        rejected_relationship_evidence_stream_entry_ids: [
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
          relationshipEvidenceStreamEntryIds: [userEntryId],
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
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
          relationshipEvidenceRelationalSlotIds: [slotId],
        }),
      ],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: {
        get: () => null,
        getApplicable: () => [],
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
      onHookFailure: vi.fn(),
    });

    expect(supersede).not.toHaveBeenCalled();
    expect(addCommitment.mock.calls[0]?.[0]).not.toHaveProperty("skipDirectiveFamilyMerge");
    expect(tracer.emit).toHaveBeenCalledWith("extraction.commitments.rejected", {
      turnId: "turn-invalid-supersession",
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
      onHookFailure: vi.fn(),
    });

    expect(supersede).not.toHaveBeenCalled();
    expect(addCommitment.mock.calls[0]?.[0]).not.toHaveProperty("skipDirectiveFamilyMerge");
    expect(tracer.emit).toHaveBeenCalledWith("extraction.commitments.rejected", {
      turnId: "turn-revoked-supersession",
      supersededId,
      validationStatus: "rejected",
      reason: "commitment_not_active",
    });
  });
});
