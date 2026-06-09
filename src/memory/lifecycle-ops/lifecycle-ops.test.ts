import { describe, expect, it, vi } from "vitest";

import type { CommitmentRecord } from "../commitments/types.js";
import type { SharedStateEntry } from "../shared-state/types.js";
import type { OpenQuestion } from "../self/open-questions.js";
import type { GoalRecord } from "../self/types.js";
import type { LifecycleTraceData, LifecycleTraceEventName, LifecycleTracer } from "./types.js";
import {
  canonicalizeActionWithSharedStateEntry,
  canonicalizeCommitmentWithSharedStateEntry,
  canonicalizeGoalWithSharedStateEntry,
  canonicalizeOpenQuestionWithSharedStateEntry,
  completeAction,
  markSemanticContradicted,
  markSemanticSuperseded,
  resolveOpenQuestionWithEvidence,
  resolveOpenQuestionThroughIdentityService,
  supersedeCommitment,
} from "./index.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import {
  createActionId,
  createCommitmentId,
  createEntityId,
  createGoalId,
  createOpenQuestionId,
  createSemanticNodeId,
  createSharedStateEntryId,
  createStreamEntryId,
} from "../../util/ids.js";

function lockedEntry(overrides: Partial<SharedStateEntry> = {}): SharedStateEntry {
  const streamEntryId = createStreamEntryId();

  return {
    id: overrides.id ?? createSharedStateEntryId(),
    audience_entity_id: overrides.audience_entity_id ?? createEntityId(),
    state_key: overrides.state_key ?? "decision.project",
    kind: "locked",
    text: overrides.text ?? "Canonical project state",
    owner_entity_id: overrides.owner_entity_id ?? null,
    provenance_stream_entry_ids: overrides.provenance_stream_entry_ids ?? [streamEntryId],
    last_updated_stream_entry_ids: overrides.last_updated_stream_entry_ids ?? [streamEntryId],
    created_at: overrides.created_at ?? 1_000,
    last_updated_at: overrides.last_updated_at ?? 1_000,
    last_updated_turn_global: overrides.last_updated_turn_global ?? null,
    superseded_by_id: overrides.superseded_by_id ?? null,
    rank: overrides.rank ?? 0,
    canonicalizes: overrides.canonicalizes ?? {
      goal_ids: [],
      commitment_ids: [],
      action_ids: [],
      open_question_ids: [],
    },
  };
}

function traceRecorder(): LifecycleTracer & {
  events: Array<{ event: LifecycleTraceEventName; data: LifecycleTraceData }>;
} {
  const events: Array<{ event: LifecycleTraceEventName; data: LifecycleTraceData }> = [];

  return {
    enabled: true,
    events,
    emit: vi.fn((event: LifecycleTraceEventName, data: LifecycleTraceData) => {
      events.push({ event, data });
    }),
  };
}

function casError(recordType: string, recordId: string): IdentityCasMismatchError {
  return new IdentityCasMismatchError({
    recordType,
    recordId,
    expectedVersion: 2,
  });
}

describe("lifecycle ops", () => {
  it("canonicalizes goals by marking them done with an artifact back-ref", () => {
    const entry = lockedEntry();
    const goalId = createGoalId();
    const updateStatus = vi.fn();

    const result = canonicalizeGoalWithSharedStateEntry({
      goalId,
      entry,
      repository: {
        get: vi.fn(() => ({ id: goalId, status: "active" }) as GoalRecord),
        updateStatus,
      },
    });

    expect(result.status).toBe("success");
    expect(updateStatus).toHaveBeenCalledWith(
      goalId,
      "done",
      { kind: "online", process: "decision_artifact_reconciliation" },
      { canonicalizedByArtifactEntryId: entry.id },
    );
  });

  it("canonicalizes actions by completing them without side effects", () => {
    const entry = lockedEntry();
    const actionId = createActionId();
    const update = vi.fn();

    const result = canonicalizeActionWithSharedStateEntry({
      actionId,
      entry,
      repository: {
        get: vi.fn(() => ({ id: actionId, state: "scheduled" }) as never),
        update,
      },
    });

    expect(result.status).toBe("success");
    expect(update).toHaveBeenCalledWith(
      actionId,
      {
        state: "completed",
        canonicalized_by_artifact_entry_id: entry.id,
      },
      { skipSideEffects: true },
    );
  });

  it("canonicalizes eligible commitments by revoking them with the artifact reason", () => {
    const entry = lockedEntry();
    const commitmentId = createCommitmentId();
    const commitment = {
      id: commitmentId,
      type: "promise",
      revoked_at: null,
      expired_at: null,
      expires_at: null,
      superseded_by: null,
    } as CommitmentRecord;
    const revoke = vi.fn(() => commitment);

    const result = canonicalizeCommitmentWithSharedStateEntry({
      commitmentId,
      entry,
      nowMs: 2_000,
      repository: {
        get: vi.fn(() => commitment),
        revoke,
      },
    });

    expect(result.status).toBe("success");
    expect(revoke).toHaveBeenCalledWith(
      commitmentId,
      `canonicalized_by_artifact_entry_id=${entry.id}`,
      { kind: "online", process: "decision_artifact_reconciliation" },
      undefined,
      { canonicalizedByArtifactEntryId: entry.id },
    );
  });

  it("skips non-canonicalizable commitment types without revoking", () => {
    const commitmentId = createCommitmentId();
    const commitment = {
      id: commitmentId,
      type: "preference",
      revoked_at: null,
      expired_at: null,
      expires_at: null,
      superseded_by: null,
    } as CommitmentRecord;
    const revoke = vi.fn();

    const result = canonicalizeCommitmentWithSharedStateEntry({
      commitmentId,
      entry: lockedEntry(),
      nowMs: 2_000,
      repository: {
        get: vi.fn(() => commitment),
        revoke,
      },
    });

    expect(result).toMatchObject({
      status: "no_op",
      reason: "non_canonicalizable_commitment_type",
    });
    expect(revoke).not.toHaveBeenCalled();
  });

  it("canonicalizes open questions by resolving with artifact stream evidence", () => {
    const entry = lockedEntry();
    const openQuestionId = createOpenQuestionId();
    const resolved = { id: openQuestionId, status: "resolved" } as OpenQuestion;
    const resolve = vi.fn(() => resolved);

    const result = canonicalizeOpenQuestionWithSharedStateEntry({
      openQuestionId,
      entry,
      repository: {
        get: vi.fn(() => ({ id: openQuestionId, status: "open" }) as OpenQuestion),
        resolve,
      },
    });

    expect(result.status).toBe("success");
    expect(resolve).toHaveBeenCalledWith(
      openQuestionId,
      {
        resolution_evidence_episode_ids: undefined,
        resolution_evidence_stream_entry_ids: entry.last_updated_stream_entry_ids,
        resolution_note: `resolved_by_artifact_entry_id=${entry.id}`,
      },
      { resolvedByArtifactEntryId: entry.id },
    );
  });

  it("returns conflicts for CAS failures", () => {
    const goalId = createGoalId();
    const error = new IdentityCasMismatchError({
      recordType: "goal",
      recordId: goalId,
      expectedVersion: 2,
    });

    const result = canonicalizeGoalWithSharedStateEntry({
      goalId,
      entry: lockedEntry(),
      repository: {
        get: vi.fn(() => ({ id: goalId, status: "active" }) as GoalRecord),
        updateStatus: vi.fn(() => {
          throw error;
        }),
      },
    });

    expect(result).toEqual({
      status: "conflict",
      error,
    });
  });

  it("supersedes commitments through the repository", () => {
    const commitmentId = createCommitmentId();
    const replacementCommitmentId = createCommitmentId();
    const commitment = {
      id: commitmentId,
      superseded_by: replacementCommitmentId,
    } as CommitmentRecord;
    const supersede = vi.fn(() => commitment);

    const result = supersedeCommitment({
      commitmentId,
      replacementCommitmentId,
      repository: {
        supersede,
      },
    });

    expect(result.status).toBe("success");
    expect(supersede).toHaveBeenCalledWith(commitmentId, replacementCommitmentId);
  });

  it("emits semantic status transition traces from semantic revision operations", async () => {
    const tracer = traceRecorder();
    const nodeId = createSemanticNodeId();
    const correctedBy = createStreamEntryId();

    const result = await markSemanticSuperseded({
      nodeId,
      correctedBy,
      supersededAt: 3_000,
      repository: {
        markSuperseded: vi.fn(async () => ({
          id: nodeId,
          fromStatus: "active" as const,
          toStatus: "superseded" as const,
          correctedBy,
          supersededAt: 3_000,
        })),
      },
      tracer,
      turnId: "turn_1",
      traceSource: "decision_artifact_semantic_revision",
    });

    expect(result.status).toBe("success");
    expect(tracer.events).toEqual([
      {
        event: "semantic_node.status.transitioned",
        data: {
          turnId: "turn_1",
          nodeId,
          fromStatus: "active",
          toStatus: "superseded",
          correctedBy,
          source: "decision_artifact_semantic_revision",
        },
      },
    ]);
  });

  it("resolves open questions through identity service and traces applied transitions", () => {
    const tracer = traceRecorder();
    const openQuestionId = createOpenQuestionId();
    const record = { id: openQuestionId, status: "resolved" } as OpenQuestion;
    const resolveOpenQuestion = vi.fn(() => ({
      status: "applied" as const,
      record,
    }));

    const result = resolveOpenQuestionThroughIdentityService({
      openQuestionId,
      identityService: {
        resolveOpenQuestion,
      },
      resolution: {
        resolution_evidence_stream_entry_ids: [createStreamEntryId()],
        resolution_note: "answered",
      },
      provenance: {
        kind: "online",
        process: "test",
      },
      tracer,
      turnId: "turn_1",
      traceSourcePath: "online_reflection",
      traceDecisionReason: "evidence_accepted",
    });

    expect(result.status).toBe("success");
    expect(tracer.events).toHaveLength(1);
    expect(tracer.events[0]).toMatchObject({
      event: "open_question_resolution.transitioned",
      data: {
        turnId: "turn_1",
        oq_id: openQuestionId,
        source_path: "online_reflection",
        decision: "resolved",
        decision_reason: "evidence_accepted",
      },
    });
  });

  it("returns missing for canonicalize operations when get exists and returns null", () => {
    const entry = lockedEntry();
    const goalId = createGoalId();
    const actionId = createActionId();
    const openQuestionId = createOpenQuestionId();
    const commitmentId = createCommitmentId();
    const updateStatus = vi.fn();
    const update = vi.fn();
    const resolve = vi.fn();
    const revoke = vi.fn();

    expect(
      canonicalizeGoalWithSharedStateEntry({
        goalId,
        entry,
        repository: {
          get: vi.fn(() => null),
          updateStatus,
        },
      }),
    ).toMatchObject({ status: "no_op", reason: "missing" });
    expect(updateStatus).not.toHaveBeenCalled();

    expect(
      canonicalizeActionWithSharedStateEntry({
        actionId,
        entry,
        repository: {
          get: vi.fn(() => null),
          update,
        },
      }),
    ).toMatchObject({ status: "no_op", reason: "missing" });
    expect(update).not.toHaveBeenCalled();

    expect(
      canonicalizeCommitmentWithSharedStateEntry({
        commitmentId,
        entry,
        nowMs: 2_000,
        repository: {
          get: vi.fn(() => null),
          revoke,
        },
      }),
    ).toMatchObject({ status: "no_op", reason: "missing" });
    expect(revoke).not.toHaveBeenCalled();

    expect(
      canonicalizeOpenQuestionWithSharedStateEntry({
        openQuestionId,
        entry,
        repository: {
          get: vi.fn(() => null),
          resolve,
        },
      }),
    ).toMatchObject({ status: "no_op", reason: "missing" });
    expect(resolve).not.toHaveBeenCalled();
  });

  it("still attempts canonicalize transitions when repository fakes omit get", () => {
    const entry = lockedEntry();
    const goalId = createGoalId();
    const actionId = createActionId();
    const openQuestionId = createOpenQuestionId();
    const updateStatus = vi.fn();
    const update = vi.fn();
    const resolve = vi.fn(() => ({ id: openQuestionId, status: "resolved" }) as OpenQuestion);

    expect(
      canonicalizeGoalWithSharedStateEntry({
        goalId,
        entry,
        repository: {
          updateStatus,
        },
      }).status,
    ).toBe("success");
    expect(updateStatus).toHaveBeenCalledOnce();

    expect(
      canonicalizeActionWithSharedStateEntry({
        actionId,
        entry,
        repository: {
          update,
        },
      }).status,
    ).toBe("success");
    expect(update).toHaveBeenCalledOnce();

    expect(
      canonicalizeOpenQuestionWithSharedStateEntry({
        openQuestionId,
        entry,
        repository: {
          resolve,
        },
      }).status,
    ).toBe("success");
    expect(resolve).toHaveBeenCalledOnce();
  });

  it("completes actions directly with and without side-effect skipping", () => {
    const firstActionId = createActionId();
    const secondActionId = createActionId();
    const firstUpdate = vi.fn();
    const secondUpdate = vi.fn();

    expect(
      completeAction({
        actionId: firstActionId,
        repository: {
          get: vi.fn(() => ({ id: firstActionId, state: "scheduled" }) as never),
          update: firstUpdate,
        },
        skipSideEffects: true,
      }).status,
    ).toBe("success");
    expect(firstUpdate).toHaveBeenCalledWith(
      firstActionId,
      { state: "completed" },
      { skipSideEffects: true },
    );

    expect(
      completeAction({
        actionId: secondActionId,
        repository: {
          get: vi.fn(() => ({ id: secondActionId, state: "scheduled" }) as never),
          update: secondUpdate,
        },
      }).status,
    ).toBe("success");
    expect(secondUpdate).toHaveBeenCalledWith(
      secondActionId,
      { state: "completed" },
      { skipSideEffects: undefined },
    );
  });

  it("returns missing when completing an action whose get lookup is empty", () => {
    const actionId = createActionId();
    const update = vi.fn();

    expect(
      completeAction({
        actionId,
        repository: {
          get: vi.fn(() => null),
          update,
        },
      }),
    ).toMatchObject({ status: "no_op", reason: "missing" });
    expect(update).not.toHaveBeenCalled();
  });

  it("resolves open questions with evidence through the repository", () => {
    const openQuestionId = createOpenQuestionId();
    const streamEntryId = createStreamEntryId();
    const resolved = { id: openQuestionId, status: "resolved" } as OpenQuestion;
    const resolve = vi.fn(() => resolved);

    const result = resolveOpenQuestionWithEvidence({
      openQuestionId,
      repository: {
        get: vi.fn(() => ({ id: openQuestionId, status: "open" }) as OpenQuestion),
        resolve,
      },
      resolutionEvidenceStreamEntryIds: [streamEntryId],
      resolutionNote: "answered",
    });

    expect(result.status).toBe("success");
    expect(resolve).toHaveBeenCalledWith(
      openQuestionId,
      {
        resolution_evidence_episode_ids: undefined,
        resolution_evidence_stream_entry_ids: [streamEntryId],
        resolution_note: "answered",
      },
      { resolvedByArtifactEntryId: undefined },
    );
  });

  it("returns missing when resolving an open question whose get lookup is empty", () => {
    const openQuestionId = createOpenQuestionId();
    const resolve = vi.fn();

    expect(
      resolveOpenQuestionWithEvidence({
        openQuestionId,
        repository: {
          get: vi.fn(() => null),
          resolve,
        },
        resolutionEvidenceStreamEntryIds: [createStreamEntryId()],
        resolutionNote: "answered",
      }),
    ).toMatchObject({ status: "no_op", reason: "missing" });
    expect(resolve).not.toHaveBeenCalled();
  });

  it("returns requires_review when identity open-question resolution is gated", () => {
    const openQuestionId = createOpenQuestionId();
    const current = { id: openQuestionId, status: "open" } as OpenQuestion;

    const result = resolveOpenQuestionThroughIdentityService({
      openQuestionId,
      identityService: {
        resolveOpenQuestion: vi.fn(() => ({
          status: "requires_review" as const,
          current,
        })),
      },
      resolution: {
        resolution_evidence_stream_entry_ids: [createStreamEntryId()],
        resolution_note: "answered",
      },
      provenance: {
        kind: "online",
        process: "test",
      },
    });

    expect(result).toMatchObject({
      status: "no_op",
      reason: "requires_review",
    });
  });

  it("returns conflicts from direct lifecycle operations", () => {
    const actionId = createActionId();
    const commitmentId = createCommitmentId();
    const replacementCommitmentId = createCommitmentId();
    const openQuestionId = createOpenQuestionId();
    const entry = lockedEntry();
    const actionError = casError("action", actionId);
    const commitmentError = casError("commitment", commitmentId);
    const openQuestionError = casError("open_question", openQuestionId);

    expect(
      completeAction({
        actionId,
        repository: {
          get: vi.fn(() => ({ id: actionId, state: "scheduled" }) as never),
          update: vi.fn(() => {
            throw actionError;
          }),
        },
      }),
    ).toEqual({ status: "conflict", error: actionError });

    expect(
      canonicalizeActionWithSharedStateEntry({
        actionId,
        entry,
        repository: {
          get: vi.fn(() => ({ id: actionId, state: "scheduled" }) as never),
          update: vi.fn(() => {
            throw actionError;
          }),
        },
      }),
    ).toEqual({ status: "conflict", error: actionError });

    expect(
      canonicalizeCommitmentWithSharedStateEntry({
        commitmentId,
        entry,
        nowMs: 2_000,
        repository: {
          get: vi.fn(
            () =>
              ({
                id: commitmentId,
                type: "promise",
                revoked_at: null,
                expired_at: null,
                expires_at: null,
                superseded_by: null,
              }) as CommitmentRecord,
          ),
          revoke: vi.fn(() => {
            throw commitmentError;
          }),
        },
      }),
    ).toEqual({ status: "conflict", error: commitmentError });

    expect(
      resolveOpenQuestionWithEvidence({
        openQuestionId,
        repository: {
          get: vi.fn(() => ({ id: openQuestionId, status: "open" }) as OpenQuestion),
          resolve: vi.fn(() => {
            throw openQuestionError;
          }),
        },
        resolutionEvidenceStreamEntryIds: [createStreamEntryId()],
        resolutionNote: "answered",
      }),
    ).toEqual({ status: "conflict", error: openQuestionError });

    expect(
      canonicalizeOpenQuestionWithSharedStateEntry({
        openQuestionId,
        entry,
        repository: {
          get: vi.fn(() => ({ id: openQuestionId, status: "open" }) as OpenQuestion),
          resolve: vi.fn(() => {
            throw openQuestionError;
          }),
        },
      }),
    ).toEqual({ status: "conflict", error: openQuestionError });

    expect(
      supersedeCommitment({
        commitmentId,
        replacementCommitmentId,
        repository: {
          supersede: vi.fn(() => {
            throw commitmentError;
          }),
        },
      }),
    ).toEqual({ status: "conflict", error: commitmentError });
  });

  it("returns conflicts from identity open-question resolution", () => {
    const openQuestionId = createOpenQuestionId();
    const error = casError("open_question", openQuestionId);

    expect(
      resolveOpenQuestionThroughIdentityService({
        openQuestionId,
        identityService: {
          resolveOpenQuestion: vi.fn(() => {
            throw error;
          }),
        },
        resolution: {
          resolution_evidence_stream_entry_ids: [createStreamEntryId()],
          resolution_note: "answered",
        },
        provenance: {
          kind: "online",
          process: "test",
        },
      }),
    ).toEqual({ status: "conflict", error });
  });

  it("returns missing when superseding an unknown commitment", () => {
    const commitmentId = createCommitmentId();

    expect(
      supersedeCommitment({
        commitmentId,
        replacementCommitmentId: createCommitmentId(),
        repository: {
          supersede: vi.fn(() => null),
        },
      }),
    ).toMatchObject({ status: "no_op", reason: "missing" });
  });

  it("marks semantic nodes contradicted and traces the supplied source", async () => {
    const tracer = traceRecorder();
    const nodeId = createSemanticNodeId();
    const correctedBy = createStreamEntryId();

    const result = await markSemanticContradicted({
      nodeId,
      correctedBy,
      supersededAt: 4_000,
      repository: {
        markContradicted: vi.fn(async () => ({
          id: nodeId,
          fromStatus: "active" as const,
          toStatus: "contradicted" as const,
          correctedBy,
          supersededAt: 4_000,
        })),
      },
      tracer,
      turnId: "turn_2",
      traceSource: "review_resolver",
    });

    expect(result.status).toBe("success");
    expect(tracer.events).toEqual([
      {
        event: "semantic_node.status.transitioned",
        data: {
          turnId: "turn_2",
          nodeId,
          fromStatus: "active",
          toStatus: "contradicted",
          correctedBy,
          source: "review_resolver",
        },
      },
    ]);
  });

  it("returns missing for semantic status operations when repository returns null", async () => {
    const nodeId = createSemanticNodeId();

    await expect(
      markSemanticSuperseded({
        nodeId,
        correctedBy: createStreamEntryId(),
        supersededAt: 3_000,
        repository: {
          markSuperseded: vi.fn(async () => null),
        },
        traceSource: "decision_artifact_semantic_revision",
      }),
    ).resolves.toMatchObject({ status: "no_op", reason: "missing" });

    await expect(
      markSemanticContradicted({
        nodeId,
        correctedBy: createStreamEntryId(),
        supersededAt: 4_000,
        repository: {
          markContradicted: vi.fn(async () => null),
        },
        traceSource: "belief_reviser",
      }),
    ).resolves.toMatchObject({ status: "no_op", reason: "missing" });
  });
});
