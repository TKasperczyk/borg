import type { ActionRepository } from "../../memory/actions/index.js";
import type {
  CommitmentRecord,
  CommitmentRepository,
  CommitmentType,
} from "../../memory/commitments/index.js";
import type {
  SharedStateArtifact,
  SharedStateCanonicalizes,
  SharedStateEntry,
  SharedStateOperation,
  SharedStateSourceTrustRejectionReason,
  SharedStateSourceTrustValidator,
} from "../../memory/decision-artifacts/index.js";
import type { EpisodicRepository } from "../../memory/episodic/index.js";
import type { SemanticNodeRepository } from "../../memory/semantic/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import type { LLMClient } from "../../llm/index.js";
import type { GoalsRepository, OpenQuestionsRepository } from "../../memory/self/index.js";
import type { ActionId, CommitmentId, GoalId, OpenQuestionId } from "../../util/ids.js";
export { SHARED_STATE_RECONCILIATION_PROVENANCE as RECONCILIATION_PROVENANCE } from "../../memory/lifecycle-ops/index.js";
import type { DroppedCanonicalizeId } from "./schema.js";
import type { SemanticRevisionVerdictCache } from "./semantic-revision-cache.js";
import { toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";

export type SharedStateReconciliationRepositories = {
  goalsRepository?: Pick<GoalsRepository, "updateStatus"> & Partial<Pick<GoalsRepository, "get">>;
  commitmentRepository?: Pick<CommitmentRepository, "get" | "revoke">;
  actionRepository?: Pick<ActionRepository, "update"> & Partial<Pick<ActionRepository, "get">>;
  openQuestionsRepository?: Pick<OpenQuestionsRepository, "resolve"> &
    Partial<Pick<OpenQuestionsRepository, "get">>;
};

export type SharedStateSemanticBeliefRevisionDependencies = {
  semanticNodeRepository: Pick<
    SemanticNodeRepository,
    "searchByVector" | "markSuperseded" | "markContradicted"
  >;
  episodicRepository: Pick<EpisodicRepository, "getMany">;
  embeddingClient: EmbeddingClient;
  llmClient: LLMClient;
  model: string;
  candidateLimit?: number;
  minSimilarity?: number;
  verdictCache?: SemanticRevisionVerdictCache;
};

export type SharedStateReconciliationLookupRepositories = {
  goalsRepository?: Partial<Pick<GoalsRepository, "get">>;
  commitmentRepository?: Partial<Pick<CommitmentRepository, "get">>;
  actionRepository?: Partial<Pick<ActionRepository, "get">>;
  openQuestionsRepository?: Partial<Pick<OpenQuestionsRepository, "get">>;
};

export type SharedStateUnsettledReconciliationSummary = {
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

export type SharedStateUnsettledReconciliation = {
  summary: SharedStateUnsettledReconciliationSummary;
  entries: SharedStateEntry[];
};

export type SharedStateReconciliationError = {
  channel: "goal" | "commitment" | "action" | "open_question";
  id: string;
  artifactEntryId: string;
  message: string;
};

export type SharedStateSkippedCommitmentCanonicalization = {
  channel: "commitment";
  id: string;
  artifactEntryId: string;
  reason: "non_canonicalizable_commitment_type";
  commitmentType: CommitmentType;
};

export type SharedStateReconciliationResult = {
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
  actions_closed_by_borg_self_performance: number;
  open_questions_resolved_attempted: number;
  open_questions_resolved_succeeded: number;
  open_questions_resolved_skipped: number;
  semantic_nodes_reviewed_attempted: number;
  semantic_nodes_marked_superseded: number;
  semantic_nodes_marked_contradicted: number;
  semantic_nodes_skipped: number;
  unknown_ids: readonly DroppedCanonicalizeId[];
  skipped_commitments: SharedStateSkippedCommitmentCanonicalization[];
  errors: SharedStateReconciliationError[];
};

export type ReconcileSharedStateCanonicalizationsInput = {
  entries: readonly SharedStateEntry[];
  repositories?: SharedStateReconciliationRepositories;
  unknownIds?: readonly DroppedCanonicalizeId[];
  nowMs?: number;
  turnCounter?: number | null;
  sourceTrustValidator?: SharedStateSourceTrustValidator;
  tracer?: TurnTracer;
  turnId?: string;
};

export type ReconcileSemanticBeliefRevisionInput = {
  artifact: SharedStateArtifact | null;
  operations: readonly SharedStateOperation[];
  dependencies?: SharedStateSemanticBeliefRevisionDependencies;
  nowMs?: number;
  sourceTrustValidator?: SharedStateSourceTrustValidator;
  tracer?: TurnTracer;
  turnId?: string;
  turnCounter?: number;
};

export type ContaminatedSharedStateArtifactSource = {
  streamEntryId: string;
  reason: SharedStateSourceTrustRejectionReason | "unknown";
};

type SemanticRevisionSkipReason =
  | "unknown_candidate"
  | "duplicate_verdict"
  | "node_missing"
  | "mark_failed"
  | "verdict_omitted";

type SemanticRevisionSkipCounts = Partial<Record<SemanticRevisionSkipReason, number>>;

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function hasCanonicalizedIds(entry: SharedStateEntry): boolean {
  return (
    entry.canonicalizes.goal_ids.length > 0 ||
    entry.canonicalizes.commitment_ids.length > 0 ||
    entry.canonicalizes.action_ids.length > 0 ||
    entry.canonicalizes.open_question_ids.length > 0
  );
}

export function emptyCanonicalizes(): SharedStateCanonicalizes {
  return {
    goal_ids: [],
    commitment_ids: [],
    action_ids: [],
    open_question_ids: [],
  };
}

export function activeLockedEntries(
  entries: readonly SharedStateEntry[],
): readonly SharedStateEntry[] {
  return entries.filter(
    (entry) =>
      entry.kind === "locked" && entry.superseded_by_id === null && hasCanonicalizedIds(entry),
  );
}

export function contaminatedSharedStateArtifactSources(
  entry: SharedStateEntry,
  validator: SharedStateSourceTrustValidator | undefined,
): ContaminatedSharedStateArtifactSource[] {
  if (validator === undefined) {
    return [];
  }

  const sources = new Set([
    ...entry.provenance_stream_entry_ids,
    ...entry.last_updated_stream_entry_ids,
  ]);
  const contaminated: ContaminatedSharedStateArtifactSource[] = [];

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

export function recordCanonicalizationSkipsForEntry(
  result: SharedStateReconciliationResult,
  entry: SharedStateEntry,
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

export function traceContaminatedSharedStateEntrySkip(input: {
  tracer?: TurnTracer;
  turnId?: string;
  entry: SharedStateEntry;
  contaminatedSources: readonly ContaminatedSharedStateArtifactSource[];
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

  input.tracer.emit("shared_state.reconcile.skipped", {
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

export function mergeSemanticBeliefRevisionResult(
  result: SharedStateReconciliationResult,
  semanticResult: Pick<
    SharedStateReconciliationResult,
    | "semantic_nodes_reviewed_attempted"
    | "semantic_nodes_marked_superseded"
    | "semantic_nodes_marked_contradicted"
    | "semantic_nodes_skipped"
  >,
): void {
  result.semantic_nodes_reviewed_attempted += semanticResult.semantic_nodes_reviewed_attempted;
  result.semantic_nodes_marked_superseded += semanticResult.semantic_nodes_marked_superseded;
  result.semantic_nodes_marked_contradicted += semanticResult.semantic_nodes_marked_contradicted;
  result.semantic_nodes_skipped += semanticResult.semantic_nodes_skipped;
}

export function recordUnknownIdSkips(
  result: SharedStateReconciliationResult,
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
