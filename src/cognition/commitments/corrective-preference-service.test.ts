import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { commitmentSchema, type CommitmentRecord } from "../../memory/commitments/index.js";
import type { IdentityService } from "../../memory/identity/index.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createCommitmentId,
  createEntityId,
  createStreamEntryId,
  type CommitmentId,
} from "../../util/ids.js";
import { CorrectivePreferenceTurnService } from "./corrective-preference-service.js";

type AddCommitmentInput = Parameters<IdentityService["addCommitment"]>[0];

function correctivePreferenceResponse(
  input: { supersedesCommitmentId?: CommitmentId | null } = {},
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
          type: "preference",
          kind: "participant_preference",
          directive: "Keep Alice's trip tasks separate from the group channel.",
          directive_family: "separate_trip_tasks",
          closure_pressure_relevance: "neutral",
          priority: 8,
          reason: "The current speaker made a durable correction.",
          confidence: 0.91,
          supersedes_commitment_id: input.supersedesCommitmentId ?? null,
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
