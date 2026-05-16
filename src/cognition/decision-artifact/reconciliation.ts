import type { ActionRepository } from "../../memory/actions/index.js";
import type {
  CommitmentRecord,
  CommitmentRepository,
  CommitmentType,
} from "../../memory/commitments/index.js";
import type {
  DecisionArtifact,
  DecisionArtifactCanonicalizes,
  DecisionArtifactEntry,
  DecisionArtifactSourceTrustRejectionReason,
  DecisionArtifactSourceTrustValidator,
} from "../../memory/decision-artifacts/index.js";
import type { GoalsRepository, OpenQuestionsRepository } from "../../memory/self/index.js";
import type { ActionId, CommitmentId, GoalId, OpenQuestionId } from "../../util/ids.js";
import type { Provenance } from "../../memory/common/provenance.js";
import type { DroppedCanonicalizeId } from "./compiler.js";
import { toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";
import {
  DECISION_ARTIFACT_COMMITMENT_CANONICALIZATION_TYPES,
  type DecisionArtifactCommitmentCanonicalizationType,
} from "./commitment-canonicalization.js";

const RECONCILIATION_PROVENANCE = {
  kind: "online",
  process: "decision_artifact_reconciliation",
} as const satisfies Provenance;

const DECISION_ARTIFACT_COMMITMENT_CANONICALIZATION_TYPE_SET = new Set<CommitmentType>(
  DECISION_ARTIFACT_COMMITMENT_CANONICALIZATION_TYPES,
);

export type DecisionArtifactReconciliationRepositories = {
  goalsRepository?: Pick<GoalsRepository, "updateStatus"> & Partial<Pick<GoalsRepository, "get">>;
  commitmentRepository?: Pick<CommitmentRepository, "get" | "revoke">;
  actionRepository?: Pick<ActionRepository, "update"> & Partial<Pick<ActionRepository, "get">>;
  openQuestionsRepository?: Pick<OpenQuestionsRepository, "resolve"> &
    Partial<Pick<OpenQuestionsRepository, "get">>;
};

export type DecisionArtifactReconciliationLookupRepositories = {
  goalsRepository?: Partial<Pick<GoalsRepository, "get">>;
  commitmentRepository?: Partial<Pick<CommitmentRepository, "get">>;
  actionRepository?: Partial<Pick<ActionRepository, "get">>;
  openQuestionsRepository?: Partial<Pick<OpenQuestionsRepository, "get">>;
};

export type DecisionArtifactUnsettledReconciliationSummary = {
  active_locked_canonicalizing_entry_count: number;
  referenced_goal_count: number;
  referenced_commitment_count: number;
  referenced_action_count: number;
  referenced_open_question_count: number;
  unsettled_goal_count: number;
  unsettled_commitment_count: number;
  unsettled_action_count: number;
  unsettled_open_question_count: number;
  unsettled_total_count: number;
};

export type DecisionArtifactUnsettledReconciliation = {
  summary: DecisionArtifactUnsettledReconciliationSummary;
  entries: DecisionArtifactEntry[];
};

export type DecisionArtifactReconciliationError = {
  channel: "goal" | "commitment" | "action" | "open_question";
  id: string;
  artifactEntryId: string;
  message: string;
};

export type DecisionArtifactSkippedCommitmentCanonicalization = {
  channel: "commitment";
  id: string;
  artifactEntryId: string;
  reason: "non_canonicalizable_commitment_type";
  commitmentType: CommitmentType;
};

export type DecisionArtifactReconciliationResult = {
  goals_retired: number;
  commitments_retired: number;
  actions_retired: number;
  open_questions_retired: number;
  goals_canonicalized_attempted: number;
  goals_canonicalized_succeeded: number;
  goals_canonicalized_skipped: number;
  commitments_revoked_attempted: number;
  commitments_revoked_succeeded: number;
  commitments_revoked_skipped: number;
  actions_completed_attempted: number;
  actions_completed_succeeded: number;
  actions_completed_skipped: number;
  open_questions_resolved_attempted: number;
  open_questions_resolved_succeeded: number;
  open_questions_resolved_skipped: number;
  unknown_ids: readonly DroppedCanonicalizeId[];
  skipped_commitments: DecisionArtifactSkippedCommitmentCanonicalization[];
  errors: DecisionArtifactReconciliationError[];
};

export type ReconcileDecisionArtifactCanonicalizationsInput = {
  entries: readonly DecisionArtifactEntry[];
  repositories?: DecisionArtifactReconciliationRepositories;
  unknownIds?: readonly DroppedCanonicalizeId[];
  nowMs?: number;
  sourceTrustValidator?: DecisionArtifactSourceTrustValidator;
  tracer?: TurnTracer;
  turnId?: string;
};

type ContaminatedDecisionArtifactSource = {
  streamEntryId: string;
  reason: DecisionArtifactSourceTrustRejectionReason | "unknown";
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function hasCanonicalizedIds(entry: DecisionArtifactEntry): boolean {
  return (
    entry.canonicalizes.goal_ids.length > 0 ||
    entry.canonicalizes.commitment_ids.length > 0 ||
    entry.canonicalizes.action_ids.length > 0 ||
    entry.canonicalizes.open_question_ids.length > 0
  );
}

function emptyCanonicalizes(): DecisionArtifactCanonicalizes {
  return {
    goal_ids: [],
    commitment_ids: [],
    action_ids: [],
    open_question_ids: [],
  };
}

function activeLockedEntries(
  entries: readonly DecisionArtifactEntry[],
): readonly DecisionArtifactEntry[] {
  return entries.filter(
    (entry) =>
      entry.kind === "locked" && entry.superseded_by_id === null && hasCanonicalizedIds(entry),
  );
}

function contaminatedDecisionArtifactSources(
  entry: DecisionArtifactEntry,
  validator: DecisionArtifactSourceTrustValidator | undefined,
): ContaminatedDecisionArtifactSource[] {
  if (validator === undefined) {
    return [];
  }

  const sources = new Set([
    ...entry.provenance_stream_entry_ids,
    ...entry.last_updated_stream_entry_ids,
  ]);
  const contaminated: ContaminatedDecisionArtifactSource[] = [];

  for (const streamEntryId of sources) {
    const trust = validator(streamEntryId);

    if (trust.allowed) {
      continue;
    }

    contaminated.push({
      streamEntryId,
      reason: trust.reason ?? "unknown",
    });
  }

  return contaminated;
}

function recordCanonicalizationSkipsForEntry(
  result: DecisionArtifactReconciliationResult,
  entry: DecisionArtifactEntry,
): void {
  const goalCount = entry.canonicalizes.goal_ids.length;
  const commitmentCount = entry.canonicalizes.commitment_ids.length;
  const actionCount = entry.canonicalizes.action_ids.length;
  const openQuestionCount = entry.canonicalizes.open_question_ids.length;

  result.goals_canonicalized_attempted += goalCount;
  result.goals_canonicalized_skipped += goalCount;
  result.commitments_revoked_attempted += commitmentCount;
  result.commitments_revoked_skipped += commitmentCount;
  result.actions_completed_attempted += actionCount;
  result.actions_completed_skipped += actionCount;
  result.open_questions_resolved_attempted += openQuestionCount;
  result.open_questions_resolved_skipped += openQuestionCount;
}

function traceContaminatedDecisionArtifactEntrySkip(input: {
  tracer?: TurnTracer;
  turnId?: string;
  entry: DecisionArtifactEntry;
  contaminatedSources: readonly ContaminatedDecisionArtifactSource[];
}): void {
  if (input.tracer?.enabled !== true || input.turnId === undefined) {
    return;
  }

  const quarantinedSourceCount = input.contaminatedSources.filter(
    (source) => source.reason === "quarantined",
  ).length;
  const inactiveSourceCount = input.contaminatedSources.filter(
    (source) => source.reason !== "quarantined",
  ).length;

  input.tracer.emit("decision_artifact_reconciliation_skipped_contaminated_entry", {
    turnId: input.turnId,
    artifact_entry_id: input.entry.id,
    kind: input.entry.kind,
    contaminated_source_id_count: input.contaminatedSources.length,
    quarantined_source_id_count: quarantinedSourceCount,
    inactive_source_id_count: inactiveSourceCount,
    contaminated_sources: toTraceJsonValue(input.contaminatedSources),
    canonicalizes: toTraceJsonValue(input.entry.canonicalizes),
  });
}

function isTerminalGoalStatus(status: string): boolean {
  return status === "done" || status === "abandoned" || status === "superseded";
}

function isTerminalCommitment(
  commitment: NonNullable<ReturnType<CommitmentRepository["get"]>>,
  nowMs: number,
): boolean {
  return (
    commitment.revoked_at !== null ||
    commitment.expired_at !== null ||
    (commitment.expires_at !== null && commitment.expires_at <= nowMs) ||
    commitment.superseded_by !== null
  );
}

function isDecisionArtifactCanonicalizableCommitmentType(
  type: CommitmentRecord["type"],
): type is DecisionArtifactCommitmentCanonicalizationType {
  return DECISION_ARTIFACT_COMMITMENT_CANONICALIZATION_TYPE_SET.has(type);
}

function isTerminalActionState(state: string): boolean {
  return state === "completed" || state === "not_done" || state === "superseded";
}

function isTerminalOpenQuestionStatus(status: string): boolean {
  return status === "resolved" || status === "abandoned";
}

function retryEntry(
  entriesById: Map<DecisionArtifactEntry["id"], DecisionArtifactEntry>,
  entry: DecisionArtifactEntry,
): DecisionArtifactEntry {
  const existing = entriesById.get(entry.id);

  if (existing !== undefined) {
    return existing;
  }

  const next = {
    ...entry,
    canonicalizes: emptyCanonicalizes(),
  };

  entriesById.set(entry.id, next);
  return next;
}

function recordUnknownIdSkips(
  result: DecisionArtifactReconciliationResult,
  unknownIds: readonly DroppedCanonicalizeId[],
): void {
  for (const unknownId of unknownIds) {
    if (unknownId.channel === "goal") {
      result.goals_canonicalized_attempted += 1;
      result.goals_canonicalized_skipped += 1;
    } else if (unknownId.channel === "commitment") {
      result.commitments_revoked_attempted += 1;
      result.commitments_revoked_skipped += 1;
    } else if (unknownId.channel === "action") {
      result.actions_completed_attempted += 1;
      result.actions_completed_skipped += 1;
    } else {
      result.open_questions_resolved_attempted += 1;
      result.open_questions_resolved_skipped += 1;
    }
  }
}

export function findUnsettledDecisionArtifactReconciliation(input: {
  previousArtifact: DecisionArtifact | null | undefined;
  repositories?: DecisionArtifactReconciliationLookupRepositories;
  nowMs?: number;
}): DecisionArtifactUnsettledReconciliation | null {
  const entries = activeLockedEntries(input.previousArtifact?.entries ?? []);

  if (entries.length === 0) {
    return null;
  }

  const nowMs = input.nowMs ?? Date.now();
  const goalsRepository = input.repositories?.goalsRepository;
  const commitmentRepository = input.repositories?.commitmentRepository;
  const actionRepository = input.repositories?.actionRepository;
  const openQuestionsRepository = input.repositories?.openQuestionsRepository;
  const goalIds = new Set(entries.flatMap((entry) => entry.canonicalizes.goal_ids));
  const commitmentIds = new Set(entries.flatMap((entry) => entry.canonicalizes.commitment_ids));
  const actionIds = new Set(entries.flatMap((entry) => entry.canonicalizes.action_ids));
  const openQuestionIds = new Set(
    entries.flatMap((entry) => entry.canonicalizes.open_question_ids),
  );
  const retryEntriesById = new Map<DecisionArtifactEntry["id"], DecisionArtifactEntry>();
  let unsettledGoalCount = 0;
  let unsettledCommitmentCount = 0;
  let unsettledActionCount = 0;
  let unsettledOpenQuestionCount = 0;

  for (const entry of entries) {
    for (const goalId of entry.canonicalizes.goal_ids) {
      const goal = goalsRepository?.get?.(goalId) ?? null;

      if (goal !== null && !isTerminalGoalStatus(goal.status)) {
        retryEntry(retryEntriesById, entry).canonicalizes.goal_ids.push(goalId);
        unsettledGoalCount += 1;
      }
    }

    for (const commitmentId of entry.canonicalizes.commitment_ids) {
      const commitment = commitmentRepository?.get?.(commitmentId) ?? null;

      if (
        commitment !== null &&
        !isTerminalCommitment(commitment, nowMs) &&
        isDecisionArtifactCanonicalizableCommitmentType(commitment.type)
      ) {
        retryEntry(retryEntriesById, entry).canonicalizes.commitment_ids.push(commitmentId);
        unsettledCommitmentCount += 1;
      }
    }

    for (const actionId of entry.canonicalizes.action_ids) {
      const action = actionRepository?.get?.(actionId) ?? null;

      if (action !== null && !isTerminalActionState(action.state)) {
        retryEntry(retryEntriesById, entry).canonicalizes.action_ids.push(actionId);
        unsettledActionCount += 1;
      }
    }

    for (const openQuestionId of entry.canonicalizes.open_question_ids) {
      const openQuestion = openQuestionsRepository?.get?.(openQuestionId) ?? null;

      if (openQuestion !== null && !isTerminalOpenQuestionStatus(openQuestion.status)) {
        retryEntry(retryEntriesById, entry).canonicalizes.open_question_ids.push(openQuestionId);
        unsettledOpenQuestionCount += 1;
      }
    }
  }

  const unsettledTotalCount =
    unsettledGoalCount +
    unsettledCommitmentCount +
    unsettledActionCount +
    unsettledOpenQuestionCount;

  if (unsettledTotalCount === 0) {
    return null;
  }

  return {
    summary: {
      active_locked_canonicalizing_entry_count: entries.length,
      referenced_goal_count: goalIds.size,
      referenced_commitment_count: commitmentIds.size,
      referenced_action_count: actionIds.size,
      referenced_open_question_count: openQuestionIds.size,
      unsettled_goal_count: unsettledGoalCount,
      unsettled_commitment_count: unsettledCommitmentCount,
      unsettled_action_count: unsettledActionCount,
      unsettled_open_question_count: unsettledOpenQuestionCount,
      unsettled_total_count: unsettledTotalCount,
    },
    entries: [...retryEntriesById.values()],
  };
}

export function reconcileDecisionArtifactCanonicalizations(
  input: ReconcileDecisionArtifactCanonicalizationsInput,
): DecisionArtifactReconciliationResult {
  const result: DecisionArtifactReconciliationResult = {
    goals_retired: 0,
    commitments_retired: 0,
    actions_retired: 0,
    open_questions_retired: 0,
    goals_canonicalized_attempted: 0,
    goals_canonicalized_succeeded: 0,
    goals_canonicalized_skipped: 0,
    commitments_revoked_attempted: 0,
    commitments_revoked_succeeded: 0,
    commitments_revoked_skipped: 0,
    actions_completed_attempted: 0,
    actions_completed_succeeded: 0,
    actions_completed_skipped: 0,
    open_questions_resolved_attempted: 0,
    open_questions_resolved_succeeded: 0,
    open_questions_resolved_skipped: 0,
    unknown_ids: input.unknownIds ?? [],
    skipped_commitments: [],
    errors: [],
  };
  recordUnknownIdSkips(result, result.unknown_ids);
  const nowMs = input.nowMs ?? Date.now();
  const entries = activeLockedEntries(input.entries);
  const goalsRepository = input.repositories?.goalsRepository;
  const commitmentRepository = input.repositories?.commitmentRepository;
  const actionRepository = input.repositories?.actionRepository;
  const openQuestionsRepository = input.repositories?.openQuestionsRepository;
  const retiredGoals = new Set<GoalId>();
  const retiredCommitments = new Set<CommitmentId>();
  const retiredActions = new Set<ActionId>();
  const retiredOpenQuestions = new Set<OpenQuestionId>();

  for (const entry of entries) {
    const contaminatedSources = contaminatedDecisionArtifactSources(
      entry,
      input.sourceTrustValidator,
    );

    if (contaminatedSources.length > 0) {
      recordCanonicalizationSkipsForEntry(result, entry);
      traceContaminatedDecisionArtifactEntrySkip({
        tracer: input.tracer,
        turnId: input.turnId,
        entry,
        contaminatedSources,
      });
      continue;
    }

    for (const goalId of entry.canonicalizes.goal_ids) {
      result.goals_canonicalized_attempted += 1;

      if (retiredGoals.has(goalId)) {
        result.goals_canonicalized_skipped += 1;
        continue;
      }

      if (goalsRepository === undefined) {
        result.goals_canonicalized_skipped += 1;
        continue;
      }

      try {
        const goal = goalsRepository.get?.(goalId) ?? null;

        if (goal !== null && isTerminalGoalStatus(goal.status)) {
          result.goals_canonicalized_skipped += 1;
          continue;
        }

        goalsRepository.updateStatus(goalId, "done", RECONCILIATION_PROVENANCE, {
          canonicalizedByArtifactEntryId: entry.id,
        });
        retiredGoals.add(goalId);
        result.goals_retired += 1;
        result.goals_canonicalized_succeeded += 1;
      } catch (error) {
        result.errors.push({
          channel: "goal",
          id: goalId,
          artifactEntryId: entry.id,
          message: errorMessage(error),
        });
      }
    }

    for (const commitmentId of entry.canonicalizes.commitment_ids) {
      result.commitments_revoked_attempted += 1;

      if (retiredCommitments.has(commitmentId)) {
        result.commitments_revoked_skipped += 1;
        continue;
      }

      if (commitmentRepository === undefined) {
        result.commitments_revoked_skipped += 1;
        continue;
      }

      try {
        const commitment = commitmentRepository.get(commitmentId);

        if (commitment !== null && isTerminalCommitment(commitment, nowMs)) {
          result.commitments_revoked_skipped += 1;
          continue;
        }

        if (
          commitment !== null &&
          !isDecisionArtifactCanonicalizableCommitmentType(commitment.type)
        ) {
          result.commitments_revoked_skipped += 1;
          result.skipped_commitments.push({
            channel: "commitment",
            id: commitmentId,
            artifactEntryId: entry.id,
            reason: "non_canonicalizable_commitment_type",
            commitmentType: commitment.type,
          });
          continue;
        }

        const retired = commitmentRepository.revoke(
          commitmentId,
          `canonicalized_by_artifact_entry_id=${entry.id}`,
          RECONCILIATION_PROVENANCE,
          undefined,
          {
            canonicalizedByArtifactEntryId: entry.id,
          },
        );

        if (retired === null) {
          result.commitments_revoked_skipped += 1;
          result.errors.push({
            channel: "commitment",
            id: commitmentId,
            artifactEntryId: entry.id,
            message: `Unknown commitment id: ${commitmentId}`,
          });
          continue;
        }

        retiredCommitments.add(commitmentId);
        result.commitments_retired += 1;
        result.commitments_revoked_succeeded += 1;
      } catch (error) {
        result.errors.push({
          channel: "commitment",
          id: commitmentId,
          artifactEntryId: entry.id,
          message: errorMessage(error),
        });
      }
    }

    for (const actionId of entry.canonicalizes.action_ids) {
      result.actions_completed_attempted += 1;

      if (retiredActions.has(actionId)) {
        result.actions_completed_skipped += 1;
        continue;
      }

      if (actionRepository === undefined) {
        result.actions_completed_skipped += 1;
        continue;
      }

      try {
        const action = actionRepository.get?.(actionId) ?? null;

        if (action !== null && isTerminalActionState(action.state)) {
          result.actions_completed_skipped += 1;
          continue;
        }

        actionRepository.update(
          actionId,
          {
            state: "completed",
            canonicalized_by_artifact_entry_id: entry.id,
          },
          {
            skipSideEffects: true,
          },
        );
        retiredActions.add(actionId);
        result.actions_retired += 1;
        result.actions_completed_succeeded += 1;
      } catch (error) {
        result.errors.push({
          channel: "action",
          id: actionId,
          artifactEntryId: entry.id,
          message: errorMessage(error),
        });
      }
    }

    for (const openQuestionId of entry.canonicalizes.open_question_ids) {
      result.open_questions_resolved_attempted += 1;

      if (retiredOpenQuestions.has(openQuestionId)) {
        result.open_questions_resolved_skipped += 1;
        continue;
      }

      if (openQuestionsRepository === undefined) {
        result.open_questions_resolved_skipped += 1;
        continue;
      }

      try {
        const openQuestion = openQuestionsRepository.get?.(openQuestionId) ?? null;

        if (openQuestion !== null && isTerminalOpenQuestionStatus(openQuestion.status)) {
          result.open_questions_resolved_skipped += 1;
          continue;
        }

        openQuestionsRepository.resolve(
          openQuestionId,
          {
            resolution_evidence_stream_entry_ids: entry.last_updated_stream_entry_ids,
            resolution_note: `resolved_by_artifact_entry_id=${entry.id}`,
          },
          {
            resolvedByArtifactEntryId: entry.id,
          },
        );
        retiredOpenQuestions.add(openQuestionId);
        result.open_questions_retired += 1;
        result.open_questions_resolved_succeeded += 1;
      } catch (error) {
        result.errors.push({
          channel: "open_question",
          id: openQuestionId,
          artifactEntryId: entry.id,
          message: errorMessage(error),
        });
      }
    }
  }

  return result;
}
