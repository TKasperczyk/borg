import type { ActionId, CommitmentId, GoalId, OpenQuestionId } from "../../util/ids.js";
import type { ActionRecord } from "../actions/types.js";
import type { SharedStateEntry } from "../shared-state/types.js";
import type { Provenance } from "../common/provenance.js";
import type { GoalRecord } from "../self/types.js";
import { isTerminalGoalStatus } from "../self/types.js";
export { isTerminalGoalStatus } from "../self/types.js";
import type { GoalsRepository } from "../self/goals-repository.js";
import type { CommitmentRecord, CommitmentType } from "../commitments/types.js";
import type { CommitmentRepository } from "../commitments/repository.js";
import type { OpenQuestion, OpenQuestionsRepository } from "../self/open-questions.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import {
  SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPES,
  type SharedStateCommitmentCanonicalizationType,
} from "./commitment-types.js";
import type { ActionRepository } from "../actions/repository.js";
import { sharedStateMemoryDisclosureLabel } from "../common/disclosure-serializers.js";
import { resolveOpenQuestionWithEvidence } from "./resolve.js";
import type { LifecycleOperationResult, LifecycleTracer } from "./types.js";

export const SHARED_STATE_RECONCILIATION_PROVENANCE = {
  kind: "online",
  process: "shared_state_reconciliation",
} as const satisfies Provenance;

const SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPE_SET = new Set<CommitmentType>(
  SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPES,
);

export function isTerminalCommitment(commitment: CommitmentRecord, nowMs: number): boolean {
  return (
    commitment.revoked_at !== null ||
    commitment.expired_at !== null ||
    (commitment.expires_at !== null && commitment.expires_at <= nowMs) ||
    commitment.superseded_by !== null
  );
}

export function isSharedStateArtifactCanonicalizableCommitmentType(
  type: CommitmentRecord["type"],
): type is SharedStateCommitmentCanonicalizationType {
  return SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPE_SET.has(type);
}

export function isTerminalOpenQuestionStatus(status: string): boolean {
  return status === "resolved" || status === "abandoned";
}

export type CanonicalizeGoalRepository = Pick<GoalsRepository, "updateStatus"> &
  Partial<Pick<GoalsRepository, "get">>;

export type CanonicalizeCommitmentRepository = Pick<CommitmentRepository, "get" | "revoke">;

export type CanonicalizeOpenQuestionRepository = Pick<OpenQuestionsRepository, "resolve"> &
  Partial<Pick<OpenQuestionsRepository, "get">>;

export function canonicalizeGoalWithSharedStateEntry(input: {
  goalId: GoalId;
  entry: SharedStateEntry;
  repository: CanonicalizeGoalRepository;
  provenance?: Provenance;
  tracer?: LifecycleTracer;
  turnId?: string;
}): LifecycleOperationResult<{ goalId: GoalId; previous: GoalRecord | null }> {
  const previous = input.repository.get?.(input.goalId);

  if (input.repository.get !== undefined && previous == null) {
    return {
      status: "no_op",
      reason: "missing",
      value: {
        goalId: input.goalId,
        previous: null,
      },
    };
  }

  if (previous !== undefined && previous !== null && isTerminalGoalStatus(previous.status)) {
    return {
      status: "no_op",
      reason: "terminal",
      value: {
        goalId: input.goalId,
        previous,
      },
    };
  }

  try {
    input.repository.updateStatus(
      input.goalId,
      "done",
      input.provenance ?? SHARED_STATE_RECONCILIATION_PROVENANCE,
      {
        canonicalizedByArtifactEntryId: input.entry.id,
      },
    );
  } catch (error) {
    if (error instanceof IdentityCasMismatchError) {
      return {
        status: "conflict",
        error,
      };
    }

    throw error;
  }

  if (input.tracer?.enabled === true && input.turnId !== undefined) {
    input.tracer.emit("extraction.goals.transitioned", {
      turnId: input.turnId,
      goalId: input.goalId,
      reason: "canonicalized_by_shared_state",
      artifact_entry_id: input.entry.id,
    });
  }

  return {
    status: "success",
    value: {
      goalId: input.goalId,
      previous: previous ?? null,
    },
  };
}

export function canonicalizeActionWithSharedStateEntry(input: {
  actionId: ActionId;
  entry: SharedStateEntry;
  repository: Partial<Pick<ActionRepository, "get" | "appendExtractionUpdate">>;
  nowMs?: number;
  turnCounter?: number | null;
  tracer?: LifecycleTracer;
  turnId?: string;
}): LifecycleOperationResult<{ actionId: ActionId; previous: ActionRecord | null }> {
  const previous = input.repository.get?.(input.actionId) ?? null;
  const selection = input.entry.canonicalizes.action_selections?.find(
    (choice) => choice.action_id === input.actionId,
  );
  const reject = (
    reason: string,
  ): LifecycleOperationResult<{ actionId: ActionId; previous: ActionRecord | null }> => {
    if (input.tracer?.enabled && input.turnId !== undefined)
      input.tracer.emit("action_state.continuation_rejected", {
        turnId: input.turnId,
        action_id: input.actionId,
        artifact_entry_id: input.entry.id,
        reason,
      });
    return { status: "no_op", reason, value: { actionId: input.actionId, previous } };
  };
  if (previous === null) return reject("missing");
  if (input.entry.owner_entity_id === null) return reject("actor_mismatch");
  if (input.repository.appendExtractionUpdate === undefined)
    return reject("repository_unavailable");
  const nowMs = input.nowMs ?? input.entry.last_updated_at;
  let result;
  try {
    result = input.repository.appendExtractionUpdate(
      input.actionId,
      {
        ...previous,
        actor:
          input.entry.owner_entity_id === selection?.self_entity_id
            ? "borg"
            : input.entry.owner_entity_id,
        description: input.entry.text,
        state: "completed",
        updated_at: nowMs,
        provenance_episode_ids: [],
        provenance_stream_entry_ids: [
          ...new Set([
            ...input.entry.provenance_stream_entry_ids,
            ...input.entry.last_updated_stream_entry_ids,
          ]),
        ],
        disclosure_label: sharedStateMemoryDisclosureLabel(input.entry),
        last_referenced_at_ms: nowMs,
        last_referenced_turn_counter: input.turnCounter ?? null,
        last_referenced_turn_global: input.turnCounter ?? null,
        selection_audit: {
          turn_id: input.turnId ?? null,
          candidate_index: 0,
          candidate_ids: selection?.candidate_ids ?? [],
          candidate_omitted_count: selection?.candidate_omitted_count ?? 0,
          chosen_action_id: input.actionId,
          rejection_reason: null,
          supplied_goal_ids: [],
          chosen_goal_id: null,
          goal_rejection_reason: null,
          shared_state_entry_id: input.entry.id,
        },
      },
      input.turnId ?? null,
      { skipSideEffects: true, canonicalizedByArtifactEntryId: input.entry.id },
    );
  } catch (error) {
    if (error instanceof IdentityCasMismatchError) return { status: "conflict", error };
    throw error;
  }
  if (result.status === "rejected") return reject(result.reason);
  if (result.status === "replayed")
    return {
      status: "no_op",
      reason: "already_applied",
      value: { actionId: input.actionId, previous },
    };
  if (input.tracer?.enabled && input.turnId !== undefined)
    input.tracer.emit("action_state.transitioned", {
      turnId: input.turnId,
      action_id: input.actionId,
      terminal_state: "completed",
      source: "shared_state_reconciliation",
      artifact_entry_id: input.entry.id,
    });
  return { status: "success", value: { actionId: input.actionId, previous } };
}

export function canonicalizeCommitmentWithSharedStateEntry(input: {
  commitmentId: CommitmentId;
  entry: SharedStateEntry;
  repository: CanonicalizeCommitmentRepository;
  nowMs: number;
  provenance?: Provenance;
  tracer?: LifecycleTracer;
  turnId?: string;
}): LifecycleOperationResult<{ commitment: CommitmentRecord | null }> {
  const commitment = input.repository.get(input.commitmentId);

  if (commitment === null) {
    return {
      status: "no_op",
      reason: "missing",
      value: {
        commitment: null,
      },
    };
  }

  if (commitment !== null && isTerminalCommitment(commitment, input.nowMs)) {
    return {
      status: "no_op",
      reason: "terminal",
      value: {
        commitment,
      },
    };
  }

  if (!isSharedStateArtifactCanonicalizableCommitmentType(commitment.type)) {
    return {
      status: "no_op",
      reason: "non_canonicalizable_commitment_type",
      value: {
        commitment,
      },
    };
  }

  let retired: CommitmentRecord | null;

  try {
    retired = input.repository.revoke(
      input.commitmentId,
      `canonicalized_by_artifact_entry_id=${input.entry.id}`,
      input.provenance ?? SHARED_STATE_RECONCILIATION_PROVENANCE,
      undefined,
      {
        canonicalizedByArtifactEntryId: input.entry.id,
      },
    );
  } catch (error) {
    if (error instanceof IdentityCasMismatchError) {
      return {
        status: "conflict",
        error,
      };
    }

    throw error;
  }

  if (retired === null) {
    return {
      status: "no_op",
      reason: "missing",
      value: {
        commitment: null,
      },
    };
  }

  if (input.tracer?.enabled === true && input.turnId !== undefined) {
    input.tracer.emit("extraction.commitments.transitioned", {
      turnId: input.turnId,
      supersededId: input.commitmentId,
      newId: input.entry.id,
      validationStatus: "accepted",
      reason: "canonicalized_by_shared_state",
    });
  }

  return {
    status: "success",
    value: {
      commitment: retired,
    },
  };
}

export function canonicalizeOpenQuestionWithSharedStateEntry(input: {
  openQuestionId: OpenQuestionId;
  entry: SharedStateEntry;
  repository: CanonicalizeOpenQuestionRepository;
  tracer?: LifecycleTracer;
  turnId?: string;
}): LifecycleOperationResult<{ question: OpenQuestion | null }> {
  const openQuestion = input.repository.get?.(input.openQuestionId);

  if (input.repository.get !== undefined && openQuestion == null) {
    return {
      status: "no_op",
      reason: "missing",
      value: {
        question: null,
      },
    };
  }

  if (
    openQuestion !== undefined &&
    openQuestion !== null &&
    isTerminalOpenQuestionStatus(openQuestion.status)
  ) {
    return {
      status: "no_op",
      reason: "terminal",
      value: {
        question: openQuestion,
      },
    };
  }

  return resolveOpenQuestionWithEvidence({
    openQuestionId: input.openQuestionId,
    repository: input.repository,
    resolutionEvidenceStreamEntryIds: input.entry.last_updated_stream_entry_ids,
    resolutionNote: `resolved_by_artifact_entry_id=${input.entry.id}`,
    resolvedByArtifactEntryId: input.entry.id,
    tracer: input.tracer,
    turnId: input.turnId,
    traceSourcePath: "shared_state_reconciliation",
    traceDecisionReason: "canonicalized_by_shared_state",
  });
}
