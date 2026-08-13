import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  ACTION_CANDIDATE_CLASSIFICATIONS,
  ACTION_STATES,
  COMMITMENT_CRITICAL_DOMAINS,
  COMMITMENT_ENFORCEMENT_CLASSES,
  COMMITMENT_KINDS,
  COMMITMENT_TYPES,
  RELATIONAL_SLOT_STATES,
  REVIEW_KINDS,
  ManualClock,
  createSessionId,
  type ActionRecord,
  type Borg,
  type CommitmentCriticalDomain,
  type CommitmentKind,
  type CommitmentType,
  type SessionId,
} from "../src/index.js";
import { CLASSIFICATION_DOWNGRADE_REASONS } from "../src/cognition/commitments/classification-normalizer.js";
import type { EmbeddingClient } from "../src/embeddings/index.js";
import { ActionRepository } from "../src/memory/actions/index.js";
import { actionMigrations } from "../src/memory/actions/migrations.js";
import { CommitmentRepository, commitmentMigrations } from "../src/memory/commitments/index.js";
import {
  IdentityEventRepository,
  IdentityService,
  identityMigrations,
} from "../src/memory/identity/index.js";
import {
  RelationalSlotRepository,
  relationalSlotMigrations,
} from "../src/memory/relational-slots/index.js";
import {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
  selfMigrations,
} from "../src/memory/self/index.js";
import { semanticMigrations } from "../src/memory/semantic/index.js";
import { ReviewQueueRepository } from "../src/memory/review-queue/index.js";
import { WorkingMemoryStore } from "../src/memory/working/index.js";
import { composeMigrations, openDatabase } from "../src/storage/sqlite/index.js";
import { ABORTED_TURN_EVENT, type StreamEntry } from "../src/stream/index.js";
import {
  createActionId,
  createSharedStateEntryId,
  createEntityId,
  createStreamEntryId,
} from "../src/util/ids.js";

import { MetricsCapture } from "./metrics.js";
import type { MetricsRow } from "./types.js";

const tempDirs: string[] = [];
const OPEN_QUESTION_OPEN_STATUS = "open";
const OPEN_QUESTION_RESOLVED_STATUS = "resolved";
const TURN_METRICS_KEY_ORDER = [
  "event",
  "ts",
  "turn_counter",
  "turnId",
  "transport_chat_attempts",
  "episode_count",
  "semantic_node_count",
  "semantic_node_count_by_status",
  "semantic_edge_count",
  "semantic_nodes_added_since_last_check",
  "semantic_edges_added_since_last_check",
  "semantic_nodes_rejected_ungrounded_claim_count",
  "semantic_nodes_rejected_ungrounded_claim_total",
  "semantic_nodes_rejected_ungrounded_claim_by_label_family",
  "shared_state_operations_rejected_ungrounded_claim_total",
  "shared_state_operations_rejected_ungrounded_claim_by_label_family",
  "commitment_candidates_rejected_ungrounded_claim_total",
  "commitment_candidates_rejected_ungrounded_claim_by_label_family",
  "open_question_count",
  "active_goal_count",
  "generation_suppression_count",
  "mood_valence",
  "mood_arousal",
  "retrieval_latency_ms",
  "deliberation_latency_ms",
  "borg_input_tokens",
  "borg_output_tokens",
  "open_question_resolved_count",
  "open_questions_by_source",
  "open_questions_by_status_age",
  "open_questions_resolved_this_run",
  "open_questions_rendered_to_finalizer_this_turn",
  "open_questions_promoted_from_review_items",
  "action_record_count_total",
  "action_record_count_by_state",
  "action_record_count_committed_to_do",
  "action_record_count_canonicalized",
  "action_record_count_active",
  "borg_owned_active_actions",
  "participant_owned_active_actions",
  "group_owned_active_actions",
  "prompt_salient_actions_total",
  "borg_owned_salient_active_actions",
  "participant_owned_salient_active_actions",
  "dormant_actions_total",
  "dormant_not_archive_eligible_count",
  "dormant_archive_eligible_count",
  "archive_oldest_inactive_turns",
  "archive_inactive_turn_distribution",
  "archive_archivable_count",
  "archive_skipped_borg_owned",
  "archive_skipped_due_date",
  "archive_skipped_below_threshold",
  "archive_skipped_other",
  "archive_oldest_archivable_inactive_turns",
  "stale_actions_omitted_from_prompt",
  "actions_per_turn",
  "salient_actions_per_turn",
  "action_retirement_ratio",
  "borg_owned_action_count",
  "stale_action_count",
  "action_record_creation_source_per_turn",
  "action_record_creation_count_this_turn",
  "action_candidate_classifications_per_turn",
  "action_candidate_rejected_classification",
  "action_persistence_dedup_skipped_embedding",
  "action_persistence_dedup_degraded",
  "actions_closed_by_terminal_emission",
  "actions_closed_by_borg_self_performance",
  "actions_expired_at_session_close",
  "actions_rejected_capability",
  "actions_canonicalized",
  "actions_completed_via_canonicalization",
  "actions_dormant_count",
  "actions_archived_count",
  "recent_completed_action_count",
  "commitment_count_active",
  "commitment_count_active_by_kind",
  "commitments_by_enforcement_class",
  "critical_commitments_by_kind_type_domain",
  "commitments_advisory_count",
  "commitments_critical_count",
  "commitments_critical_classification_downgraded_total",
  "commitments_critical_classification_downgraded_by_reason",
  "commitments_critical_classification_downgraded_by_kind_type_from_domain",
  "commitment_count_superseded",
  "commitment_count_revoked",
  "commitment_count_expired",
  "commitment_count_canonicalized",
  "commitment_regeneration_attempted_count",
  "commitment_regeneration_succeeded_count",
  "commitment_regeneration_failed_count",
  "commitment_regeneration_attempted_total",
  "commitment_regeneration_succeeded_total",
  "commitment_regeneration_failed_total",
  "commitment_guard_advisory_violations_total",
  "commitment_guard_advisory_violations_by_class",
  "pending_action_count",
  "pending_action_merge_count",
  "relational_slot_count_by_state",
  "review_queue_open_count_by_type",
  "review_resolver_attempted",
  "review_resolver_accepted",
  "review_resolver_dismissed",
  "review_resolver_rejected",
  "review_resolver_needs_manual",
  "review_queue_enqueued_this_turn",
  "review_queue_resolved_this_turn",
  "review_queue_drain_rate",
  "frame_anomaly_classifier_calls",
  "frame_anomaly_classified_normal_count",
  "frame_anomaly_actual_anomaly_count",
  "frame_anomaly_degraded_count",
  "frame_anomaly_degraded_fallback_match_count",
  "quarantined_user_entry_count",
  "early_extractors_skipped_frame_anomaly_count",
  "goal_promotion_salvaged_promotions",
  "goal_promotion_skipped_promotions",
  "goal_promotion_initial_step_downgraded",
  "goal_promotion_dedup_skipped_extractor_signal",
  "goal_promotion_dedup_skipped_embedding",
  "goal_promotion_dedup_degraded",
  "goal_promotion_classifications_per_turn",
  "goal_promotion_rejected_classification",
  "goal_promotion_cap_rejections",
  "shared_state_semantic_revisions_attempted",
  "shared_state_semantic_revisions_completed_succeeded",
  "shared_state_semantic_nodes_marked_superseded",
  "shared_state_semantic_nodes_marked_contradicted",
  "shared_state_semantic_revision_cache_hits",
  "shared_state_semantic_revision_cache_size",
  "embedding_cache_pending_overflow_total",
  "ledger_reverse_scan_entries_total",
  "ledger_reverse_scan_bytes_total",
  "ledger_reverse_scan_entry_cap_hit_total",
  "ledger_reverse_scan_byte_cap_hit_total",
  "ledger_image_refs_considered_total",
  "ledger_image_refs_attached_total",
  "ledger_image_refs_omitted_budget_total",
  "ledger_image_bytes_attached_total",
  "ledger_image_refs_omitted_inactive_total",
  "semantic_revision_error_count",
  "semantic_revision_skipped_due_to_error",
  "semantic_revision_error_total_by_reason",
  "semantic_revision_calls_total",
  "semantic_revision_candidates_reviewed_total",
  "semantic_revision_superseded_total",
  "semantic_revision_contradicted_total",
  "semantic_revision_degraded_total",
  "semantic_revision_skipped_over_cap_total",
  "overseer_due_on_suppressed_turn",
  "closure_loop_completed_count",
  "closure_loop_degraded_count",
  "closure_response_audit_failed_open_total",
  "closure_pressure_mixed_observed_total",
  "closure_pressure_closure_only_observed_total",
  "closure_pressure_closure_only_suppressed_total",
  "closure_pressure_mixed_passed_no_active_preference_total",
  "closure_pressure_mixed_by_span_kind",
  "corrective_preference_completed_count",
  "corrective_preference_degraded_count",
  "extractor_max_tokens_stop_count",
  "extractor_max_tokens_total_by_label",
  "extractor_degraded_total_by_label",
  "shared_state_compiler_max_tokens_total",
  "shared_state_compiler_degraded_total",
  "shared_state_compiler_repair_attempted_total",
  "shared_state_compiler_repair_succeeded_total",
  "shared_state_compiler_repair_failed_total",
  "shared_state_compiler_repair_failed_by_rejection_reason",
  "shared_state_update_checked_for_empty_total",
  "shared_state_empty_update_attempted_total",
  "shared_state_empty_update_dropped_total",
  "shared_state_empty_update_drop_rate",
  "shared_state_empty_update_repaired_total",
  "capability_overclaim_count",
  "capability_ambiguity_count",
  "capability_boundary_refusal_count",
  "shared_state_at_cap_turns",
  "shared_state_compile_evaluated_turns",
  "shared_state_omitted_recent_entries",
  "shared_state_omitted_recent_entries_total_across_compiles",
  "shared_state_omitted_live_recent_operational",
  "shared_state_omitted_live_recent_operational_total_across_compiles",
  "shared_state_omitted_live_recent_operational_final_compile",
  "shared_state_omitted_live_recent_low_salience",
  "shared_state_omitted_live_recent_low_salience_total_across_compiles",
  "shared_state_omitted_live_recent_low_salience_final_compile",
  "shared_state_omitted_live_old",
  "shared_state_omitted_live_old_total_across_compiles",
  "shared_state_omitted_live_old_final_compile",
  "shared_state_omitted_live_unknown_age",
  "shared_state_omitted_live_unknown_age_total_across_compiles",
  "shared_state_omitted_live_unknown_age_final_compile",
  "shared_state_omitted_locked",
  "shared_state_omitted_locked_total_across_compiles",
  "shared_state_omitted_locked_final_compile",
  "shared_state_omitted_locked_recent_total_across_compiles",
  "shared_state_omitted_locked_recent_final_compile",
  "shared_state_omitted_locked_old_total_across_compiles",
  "shared_state_omitted_locked_old_final_compile",
  "shared_state_omitted_locked_unknown_age_total_across_compiles",
  "shared_state_omitted_locked_unknown_age_final_compile",
  "shared_state_omitted_locked_with_active_critical_commitment_total_across_compiles",
  "shared_state_omitted_locked_with_active_critical_commitment_final_compile",
  "shared_state_omitted_locked_with_operational_canonicalizer_total_across_compiles",
  "shared_state_omitted_locked_with_operational_canonicalizer_final_compile",
  "shared_state_omitted_locked_indexed_only_total_across_compiles",
  "shared_state_omitted_locked_indexed_only_final_compile",
  "shared_state_omitted_pending",
  "shared_state_omitted_pending_total_across_compiles",
  "shared_state_omitted_pending_final_compile",
  "shared_state_omitted_low_salience_live",
  "shared_state_omitted_low_salience_live_final_compile",
  "shared_state_omitted_dormant_live",
  "shared_state_omitted_dormant_live_final_compile",
  "shared_state_active_low_salience_live",
  "shared_state_active_low_salience_live_final_compile",
  "shared_state_active_dormant_live",
  "shared_state_active_dormant_live_final_compile",
  "shared_state_demoted_live_to_low_salience_total",
  "shared_state_demoted_low_salience_to_dormant_total",
  "shared_state_lifecycle_aging_demotable_total",
  "shared_state_lifecycle_aging_demotable_final_compile",
  "shared_state_lifecycle_aging_demoted_total",
  "shared_state_lifecycle_aging_demoted_final_compile",
  "shared_state_lifecycle_aging_blocked_by_current_turn_update_total",
  "shared_state_lifecycle_aging_blocked_by_current_turn_update_final_compile",
  "shared_state_lifecycle_aging_blocked_by_patch_touch_total",
  "shared_state_lifecycle_aging_blocked_by_patch_touch_final_compile",
  "shared_state_lifecycle_aging_blocked_by_ledger_overlap_total",
  "shared_state_lifecycle_aging_blocked_by_ledger_overlap_final_compile",
  "shared_state_lifecycle_aging_blocked_by_recent_retrieval_total",
  "shared_state_lifecycle_aging_blocked_by_recent_retrieval_final_compile",
  "shared_state_lifecycle_aging_blocked_by_active_canonicalizer_critical_total",
  "shared_state_lifecycle_aging_blocked_by_active_canonicalizer_critical_final_compile",
  "shared_state_lifecycle_aging_blocked_by_active_canonicalizer_operational_total",
  "shared_state_lifecycle_aging_blocked_by_active_canonicalizer_operational_final_compile",
  "shared_state_lifecycle_aging_blocked_by_hard_total",
  "shared_state_lifecycle_aging_blocked_by_hard_final_compile",
  "shared_state_lifecycle_aging_blocked_by_soft_total",
  "shared_state_lifecycle_aging_blocked_by_soft_final_compile",
  "shared_state_lifecycle_aging_unknown_age_total",
  "shared_state_lifecycle_aging_unknown_age_final_compile",
  "shared_state_lifecycle_aging_blocked_by_multiple_reasons_total",
  "shared_state_lifecycle_aging_blocked_by_multiple_reasons_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_demotable_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_demotable_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_demoted_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_demoted_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_current_turn_update_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_current_turn_update_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_patch_touch_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_patch_touch_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_ledger_overlap_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_ledger_overlap_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_recent_retrieval_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_recent_retrieval_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_critical_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_critical_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_operational_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_operational_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_hard_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_hard_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_soft_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_soft_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_unknown_age_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_unknown_age_final_compile",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_multiple_reasons_total",
  "shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_multiple_reasons_final_compile",
  "shared_state_reactivated_low_salience_live_total",
  "shared_state_reactivated_dormant_live_total",
  "shared_state_at_cap_but_all_keys_indexed_compiles_total",
  "shared_state_at_cap_with_operational_omission_compiles_total",
  "shared_state_at_cap_with_cap_rejection_compiles_total",
  "shared_state_all_active_keys_indexed",
  "shared_state_live_entry_starvation",
  "shared_state_newest_entries_reserved",
  "shared_state_live_starvation_with_reserved",
  "shared_state_live_starvation_ever",
  "shared_state_live_starvation_final",
  "shared_state_compiler_operations_total_by_kind",
  "shared_state_add_to_update_ratio",
  "shared_state_entries_by_key",
  "shared_state_add_to_update_ratio_by_key",
  "shared_state_top_keys_by_entry_count",
  "shared_state_add_rejected_cap_exceeded_total",
  "shared_state_new_keys_per_compile",
  "shared_state_new_keys_per_turn",
  "shared_state_keys_with_single_entry_only",
  "shared_state_similar_key_cluster_count",
  "shared_state_add_rejected_near_duplicate_state_key_total",
  "shared_state_add_rejected_missing_new_key_reason_total",
  "session_reentry_card_rendered_total",
  "session_reentry_card_rendered_by_audience",
  "session_reentry_first_turn_with_existing_state_total",
  "session_reentry_first_turn_blank_audience_total",
  "simulator_persona_failures",
  "borg_hard_aborted_turns",
  "borg_intentional_suppressions",
  "borg_intentional_suppressions_by_reason",
  "finalizer_no_output_by_category",
  "finalizer_no_output_primary_by_reason",
  "finalizer_no_output_flags_by_flag",
  "finalizer_no_output_flags_by_primary_reason",
  "finalizer_no_output_when_borg_addressed_with_state_delta_total",
  "finalizer_no_output_closure_with_open_question_total",
  "borg_aborted_turns",
] as const;

class SameVectorEmbeddingClient implements EmbeddingClient {
  async embed(): Promise<Float32Array> {
    return Float32Array.from([1, 0]);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map(() => Float32Array.from([1, 0]));
  }
}

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "borg-simulator-metrics-"));
  tempDirs.push(dir);
  return dir;
}

function zeroCounts<K extends string>(keys: readonly K[]): Record<K, number> {
  return Object.fromEntries(keys.map((key) => [key, 0])) as Record<K, number>;
}

function zeroCriticalCommitmentsByKindTypeDomain(): Record<
  CommitmentKind,
  Record<CommitmentType, Record<CommitmentCriticalDomain, number>>
> {
  return Object.fromEntries(
    COMMITMENT_KINDS.map((kind) => [
      kind,
      Object.fromEntries(
        COMMITMENT_TYPES.map((type) => [type, zeroCounts(COMMITMENT_CRITICAL_DOMAINS)]),
      ),
    ]),
  ) as Record<CommitmentKind, Record<CommitmentType, Record<CommitmentCriticalDomain, number>>>;
}

function makeAction(overrides: Partial<ActionRecord> = {}): ActionRecord {
  const nowMs = overrides.created_at ?? 1_000;

  return {
    id: overrides.id ?? createActionId(),
    description: overrides.description ?? "Review metrics fixture",
    actor: overrides.actor ?? "borg",
    audience_entity_id: overrides.audience_entity_id ?? null,
    goal_id: overrides.goal_id ?? null,
    open_question_id: overrides.open_question_id ?? null,
    state: overrides.state ?? "committed_to_do",
    confidence: overrides.confidence ?? 0.8,
    provenance_episode_ids: overrides.provenance_episode_ids ?? [],
    provenance_stream_entry_ids: overrides.provenance_stream_entry_ids ?? [createStreamEntryId()],
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
    considering_at: overrides.considering_at ?? null,
    committed_at: overrides.committed_at ?? null,
    scheduled_at: overrides.scheduled_at ?? null,
    completed_at: overrides.completed_at ?? null,
    not_done_at: overrides.not_done_at ?? null,
    expired_at: overrides.expired_at ?? null,
    archived_at: overrides.archived_at ?? null,
    unknown_at: overrides.unknown_at ?? null,
    canonicalized_by_artifact_entry_id: overrides.canonicalized_by_artifact_entry_id ?? null,
    session_scope: overrides.session_scope ?? null,
    session_anchor_id: overrides.session_anchor_id ?? null,
    last_referenced_at_ms: overrides.last_referenced_at_ms ?? nowMs,
    last_referenced_turn_counter: overrides.last_referenced_turn_counter ?? null,
    last_referenced_turn_global:
      overrides.last_referenced_turn_global ?? overrides.last_referenced_turn_counter ?? null,
  };
}

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

function fakeBorg(
  counts: {
    semanticNodes?: number;
    semanticEdges?: number;
    activeGoals?: number;
    suppressedSessions?: readonly SessionId[];
    streamEntriesBySession?: ReadonlyMap<SessionId, readonly StreamEntry[]>;
  } = {},
  observed: { moodSessions?: SessionId[]; tailSessions?: SessionId[] } = {},
): Borg {
  const semanticNodeCount = counts.semanticNodes ?? 1;
  const semanticEdgeCount = counts.semanticEdges ?? 2;
  const activeGoalCount = counts.activeGoals ?? 2;
  const suppressedSessions = new Set(counts.suppressedSessions ?? []);
  const streamEntriesBySession = counts.streamEntriesBySession ?? new Map();

  return {
    mood: {
      current: (sessionId: SessionId) => {
        observed.moodSessions?.push(sessionId);
        return { valence: -0.2, arousal: 0.4 };
      },
    },
    episodic: {
      list: async () => ({ items: [{ id: "episode_1" }, { id: "episode_2" }] }),
    },
    semantic: {
      nodes: {
        list: async () =>
          Array.from({ length: semanticNodeCount }, (_, index) => ({
            id: `node_${index}`,
            status: "active",
          })),
      },
      edges: {
        list: () =>
          Array.from({ length: semanticEdgeCount }, (_, index) => ({ id: `edge_${index}` })),
      },
    },
    actions: {
      count: () => 0,
      countByState: () => zeroCounts(ACTION_STATES),
      countCanonicalized: () => 0,
      countActive: () => 0,
      getCreationCountsBySource: () => ({
        extractor: 0,
        reflector: 0,
        api: 0,
        tool: 0,
        unknown: 0,
      }),
      countCompletedSince: () => 0,
      latestCompletedAt: () => null,
      listCompletedIds: () => [],
      list: () => [],
    },
    self: {
      openQuestions: {
        list: () => [{ id: "question_1" }],
      },
      goals: {
        list: () =>
          Array.from({ length: activeGoalCount }, (_, index) => ({
            id: `goal_${index}`,
            children: [],
          })),
      },
    },
    commitments: {
      list: () => [],
      countActive: () => 0,
      countActiveByKind: () => zeroCounts(COMMITMENT_KINDS),
      countActiveByEnforcementClass: () => zeroCounts(COMMITMENT_ENFORCEMENT_CLASSES),
      countSuperseded: () => 0,
      countRevoked: () => 0,
      countExpired: () => 0,
      countCanonicalized: () => 0,
    },
    relationalSlots: {
      countByState: () => zeroCounts(RELATIONAL_SLOT_STATES),
    },
    review: {
      list: () => [],
    },
    identity: {
      listEvents: () => [],
    },
    workmem: {
      load: () => ({ pending_actions: [] }),
      getPendingActionMergeCount: () => 0,
    },
    stream: {
      tail: (_limit: number, options?: { session?: SessionId }) => {
        if (options?.session !== undefined) {
          observed.tailSessions?.push(options.session);
        }

        if (options?.session !== undefined && streamEntriesBySession.has(options.session)) {
          return [...(streamEntriesBySession.get(options.session) ?? [])];
        }

        return options?.session !== undefined && suppressedSessions.has(options.session)
          ? [{ kind: "agent_suppressed" }]
          : [];
      },
    },
  } as unknown as Borg;
}

function fakeBorgWithActions(actions: readonly ActionRecord[]): Borg {
  const activeStates = new Set(["considering", "committed_to_do", "scheduled", "unknown"]);
  const countByState = () => {
    const counts = zeroCounts(ACTION_STATES);

    for (const action of actions) {
      counts[action.state] += 1;
    }

    return counts;
  };

  return {
    ...fakeBorg(),
    actions: {
      count: () => actions.length,
      countByState,
      countCanonicalized: () => 0,
      countActive: () => actions.filter((action) => activeStates.has(action.state)).length,
      getCreationCountsBySource: () => ({
        extractor: 0,
        reflector: 0,
        api: 0,
        tool: 0,
        unknown: 0,
      }),
      countCompletedSince: () => 0,
      latestCompletedAt: () => null,
      listCompletedIds: () => [],
      list: (filter: { states?: readonly ActionRecord["state"][] } = {}) =>
        actions.filter(
          (action) => filter.states === undefined || filter.states.includes(action.state),
        ),
      findSimilarDescriptionPairs: async () => [],
    },
  } as unknown as Borg;
}

function createIdentityHarness(db: ReturnType<typeof openDatabase>, clock: ManualClock) {
  const identityEvents = new IdentityEventRepository({ db, clock });
  const valuesRepository = new ValuesRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const goalsRepository = new GoalsRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const traitsRepository = new TraitsRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const autobiographicalRepository = new AutobiographicalRepository({ db, clock });
  const growthMarkersRepository = new GrowthMarkersRepository({ db, clock });
  const openQuestionsRepository = new OpenQuestionsRepository({ db, clock });
  const commitmentRepository = new CommitmentRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const identity = new IdentityService({
    valuesRepository,
    goalsRepository,
    traitsRepository,
    autobiographicalRepository,
    growthMarkersRepository,
    openQuestionsRepository,
    commitmentRepository,
    identityEventRepository: identityEvents,
  });

  return {
    identity,
    openQuestionsRepository,
  };
}

describe("MetricsCapture", () => {
  it("captures Borg state, trace latency, and token usage to JSONL", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");

    writeFileSync(
      tracePath,
      [
        { ts: 100, turnId: "turn-1", event: "retrieval.started" },
        { ts: 125, turnId: "turn-1", event: "retrieval.completed" },
        { ts: 130, turnId: "turn-1", event: "llm_call.started" },
        {
          ts: 131,
          turnId: "turn-1",
          event: "llm_call.completed",
          label: "closure_loop_classifier",
          stopReason: "max_tokens",
        },
        {
          ts: 132,
          turnId: "turn-1",
          event: "closure_loop.degraded",
          reason: "missing_tool_call",
        },
        {
          ts: 133,
          turnId: "turn-1",
          event: "llm_call.completed",
          label: "corrective_preference_extractor",
          stop_reason: "max_tokens",
        },
        {
          ts: 134,
          turnId: "turn-1",
          event: "extraction.commitments.degraded",
          reason: "invalid_payload",
        },
        {
          ts: 136,
          turnId: "turn-1",
          event: "llm_call.completed",
          label: "pending_action_judge",
          stopReason: "max_tokens",
        },
        {
          ts: 137,
          turnId: "turn-1",
          event: "llm_call.completed",
          label: "procedural-context",
          stopReason: "max_tokens",
        },
        {
          ts: 138,
          turnId: "turn-1",
          event: "llm_call.completed",
          label: "not_registered_extractor",
          stopReason: "max_tokens",
        },
        {
          ts: 139,
          turnId: "turn-1",
          event: "shared_state.semantic_revision.degraded",
          reason: "semantic vector search failed",
          skipped_due_to_error: 1,
        },
        {
          ts: 140,
          turnId: "semantic_extractor",
          event: "semantic_insert.skipped",
          kind: "node",
          reason: "relationship_claim_ungrounded",
          relationship_claim_label_families: ["kinship"],
        },
        {
          ts: 141,
          turnId: "turn-1",
          event: "shared_state.compile.claim_ungrounded",
          relationship_claim_label_families: ["intimate_partner"],
        },
        {
          ts: 142,
          turnId: "turn-1",
          event: "corrective_preference.candidate_rejected_ungrounded",
          relationship_claim_label_families: ["kinship"],
        },
        {
          ts: 190,
          turnId: "turn-1",
          event: "llm_call.completed",
          usage: { inputTokens: 11, outputTokens: 7 },
        },
        {
          ts: 135,
          turnId: "other-turn",
          event: "llm_call.completed",
          label: "goal_promotion_extractor",
          stopReason: "max_tokens",
        },
        {
          ts: 136,
          turnId: "other-turn",
          event: "llm_call.completed",
          label: "shared_state_semantic_revision",
          stopReason: "tool_use",
        },
        {
          ts: 137,
          turnId: "other-turn",
          event: "semantic_revision.completed",
          artifact_entry_id: "dart_metrics_previous",
          candidates_enumerated: 6,
          superseded_count: 1,
          contradicted_count: 2,
        },
        {
          ts: 138,
          turnId: "other-turn",
          event: "semantic_revision.degraded",
          artifact_entry_id: "dart_metrics_over_cap",
          reason: "skipped_over_cap",
        },
        {
          ts: 139,
          turnId: "other-turn",
          event: "commitment_guard.regeneration_requested",
        },
        {
          ts: 140,
          turnId: "other-turn",
          event: "commitment_guard.regeneration_failed",
        },
        {
          ts: 189,
          turnId: "other-turn",
          event: "llm_call.completed",
          label: "decision_artifact_semantic_revision",
          stopReason: "tool_use",
        },
        {
          ts: 191,
          turnId: "turn-1",
          event: "semantic_revision.completed",
          artifact_entry_id: "dart_metrics_completed",
          candidates_enumerated: 4,
          superseded_count: 2,
          contradicted_count: 1,
        },
        {
          ts: 192,
          turnId: "turn-1",
          event: "semantic_revision.degraded",
          artifact_entry_id: "dart_metrics_degraded",
          reason: "judge unavailable",
        },
        {
          ts: 193,
          turnId: "turn-1",
          event: "semantic_revision.degraded",
          artifact_entry_id: "dart_metrics_completed",
          reason: "mark failed after partial apply",
        },
        {
          ts: 194,
          turnId: "turn-1",
          event: "semantic_revision.cache.completed",
          artifact_entry_id: "dart_metrics_completed",
          candidate_node_id: "node_metrics_cached",
          cached_verdict: "keep",
          age_turns: 1,
        },
        {
          ts: 195,
          turnId: "turn-1",
          event: "commitment_guard.regeneration_requested",
        },
        {
          ts: 196,
          turnId: "turn-1",
          event: "commitment_guard.regeneration_succeeded",
        },
        {
          ts: 197,
          turnId: "turn-1",
          event: "commitment_guard.advisory_violation_observed",
          violationCount: 2,
          commitmentEnforcementClasses: ["advisory"],
        },
        {
          ts: 198,
          turnId: "turn-1",
          event: "closure_response_guard.completed",
          mode: "enforce",
          verdict: "passed",
          wouldHaveVerdict: "suppressed",
          reason: "mixed_closure_observed",
          response_shape: "mixed",
          spans: [
            {
              text: "Closing line.",
              kind: "aphoristic_valediction",
            },
          ],
        },
        {
          ts: 199,
          turnId: "turn-1",
          event: "closure_response_guard.completed",
          mode: "enforce",
          verdict: "suppressed",
          reason: "closure_pressure_only",
          response_shape: "closure_only",
          spans: [
            {
              text: "Goodnight.",
              kind: "imperative_closer",
            },
          ],
        },
        {
          ts: 200,
          turnId: "turn-1",
          event: "closure_response_guard.completed",
          mode: "shadow",
          verdict: "passed",
          reason: "no_active_closure_preference",
          response_shape: "mixed",
          spans: [
            {
              text: "Noted.",
              kind: "quotable_closing_tail",
            },
          ],
        },
        {
          ts: 201,
          turnId: "turn-1",
          event: "closure_response_guard.completed",
          mode: "enforce",
          verdict: "passed",
          reason: "closure_response_audit_failed_open",
          response_shape: null,
        },
        {
          ts: 202,
          turnId: "turn-1",
          event: "closure_response_guard.completed",
          mode: "enforce",
          verdict: "passed",
          wouldHaveVerdict: "suppressed",
          reason: "closure_only_observed",
          response_shape: "closure_only",
        },
        {
          ts: 203,
          turnId: "turn-1",
          event: "post_generation.rejected",
          reason: "finalizer_no_output",
          primary_no_output_reason: "closure",
          no_output_categories: ["closure", "with_open_question"],
          structural_no_output_flags: ["with_open_question", "open_question_rendered"],
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const sessionId = createSessionId();
    const otherSessionId = createSessionId();
    const observed: { moodSessions: SessionId[]; tailSessions: SessionId[] } = {
      moodSessions: [],
      tailSessions: [],
    };
    const capture = new MetricsCapture(metricsPath, {
      tracePath,
      semanticRevisionVerdictCacheSize: () => 17,
    });
    const row = await capture.capture(
      fakeBorg({ suppressedSessions: [otherSessionId] }, observed),
      "turn-1",
      3,
      {
        sessionId,
        sessionIds: [sessionId, otherSessionId],
        transportChatAttempts: 2,
      },
    );
    const written = JSON.parse(readFileSync(metricsPath, "utf8").trim()) as MetricsRow;

    expect(row.turn_counter).toBe(3);
    expect(row.event).toBe("turn_metrics");
    expect(row.transport_chat_attempts).toBe(2);
    expect(row.episode_count).toBe(2);
    expect(row.semantic_node_count).toBe(1);
    expect(row.semantic_node_count_by_status).toEqual({
      active: 1,
      superseded: 0,
      contradicted: 0,
      quarantined: 0,
    });
    expect(row.semantic_edge_count).toBe(2);
    expect(row.semantic_nodes_added_since_last_check).toBe(0);
    expect(row.semantic_edges_added_since_last_check).toBe(0);
    expect(row.semantic_nodes_rejected_ungrounded_claim_count).toBe(1);
    expect(row.semantic_nodes_rejected_ungrounded_claim_total).toBe(1);
    expect(row.semantic_nodes_rejected_ungrounded_claim_by_label_family).toEqual({
      kinship: 1,
    });
    expect(row.shared_state_operations_rejected_ungrounded_claim_total).toBe(1);
    expect(row.shared_state_operations_rejected_ungrounded_claim_by_label_family).toEqual({
      intimate_partner: 1,
    });
    expect(row.commitment_candidates_rejected_ungrounded_claim_total).toBe(1);
    expect(row.commitment_candidates_rejected_ungrounded_claim_by_label_family).toEqual({
      kinship: 1,
    });
    expect(row.open_question_count).toBe(1);
    expect(row.active_goal_count).toBe(2);
    expect(row.generation_suppression_count).toBe(1);
    expect(row.retrieval_latency_ms).toBe(25);
    expect(row.deliberation_latency_ms).toBe(60);
    expect(row.borg_input_tokens).toBe(11);
    expect(row.borg_output_tokens).toBe(7);
    expect(row.closure_loop_completed_count).toBe(1);
    expect(row.closure_loop_degraded_count).toBe(1);
    expect(row.closure_response_audit_failed_open_total).toBe(1);
    expect(row.closure_pressure_mixed_observed_total).toBe(1);
    expect(row.closure_pressure_closure_only_observed_total).toBe(1);
    expect(row.closure_pressure_closure_only_suppressed_total).toBe(1);
    expect(row.closure_pressure_mixed_passed_no_active_preference_total).toBe(1);
    expect(row.closure_pressure_mixed_by_span_kind).toEqual({
      aphoristic_valediction: 1,
      quotable_closing_tail: 1,
    });
    expect(row.corrective_preference_completed_count).toBe(1);
    expect(row.corrective_preference_degraded_count).toBe(1);
    expect(row.extractor_max_tokens_stop_count).toBe(4);
    expect(row.extractor_max_tokens_total_by_label).toEqual({
      closure_loop_classifier: 1,
      corrective_preference_extractor: 1,
      goal_promotion_extractor: 1,
      pending_action_judge: 1,
      "procedural-context": 1,
    });
    expect(row.extractor_degraded_total_by_label).toEqual({
      closure_loop_classifier: 1,
      corrective_preference_extractor: 1,
    });
    expect(row.capability_overclaim_count).toBe(0);
    expect(row.capability_ambiguity_count).toBe(0);
    expect(row.capability_boundary_refusal_count).toBe(0);
    expect(row.shared_state_semantic_revisions_attempted).toBe(2);
    expect(row.shared_state_semantic_revisions_completed_succeeded).toBe(1);
    expect(row.shared_state_semantic_nodes_marked_superseded).toBe(2);
    expect(row.shared_state_semantic_nodes_marked_contradicted).toBe(1);
    expect(row.shared_state_semantic_revision_cache_hits).toBe(1);
    expect(row.shared_state_semantic_revision_cache_size).toBe(17);
    expect(row.semantic_revision_error_count).toBe(1);
    expect(row.semantic_revision_skipped_due_to_error).toBe(1);
    expect(row.semantic_revision_error_total_by_reason).toEqual({
      "semantic vector search failed": 1,
    });
    expect(row.commitment_regeneration_attempted_count).toBe(1);
    expect(row.commitment_regeneration_succeeded_count).toBe(1);
    expect(row.commitment_regeneration_failed_count).toBe(0);
    expect(row.commitment_regeneration_attempted_total).toBe(2);
    expect(row.commitment_regeneration_succeeded_total).toBe(1);
    expect(row.commitment_regeneration_failed_total).toBe(1);
    expect(row.commitment_guard_advisory_violations_total).toBe(2);
    expect(row.commitment_guard_advisory_violations_by_class).toEqual({
      critical: 0,
      advisory: 2,
    });
    expect(row.semantic_revision_calls_total).toBe(2);
    expect(row.semantic_revision_candidates_reviewed_total).toBe(10);
    expect(row.semantic_revision_superseded_total).toBe(3);
    expect(row.semantic_revision_contradicted_total).toBe(3);
    expect(row.semantic_revision_degraded_total).toBe(4);
    expect(row.semantic_revision_skipped_over_cap_total).toBe(1);
    expect(row.finalizer_no_output_by_category).toEqual({
      closure: 1,
      with_open_question: 1,
    });
    expect(row.finalizer_no_output_primary_by_reason).toEqual({
      closure: 1,
    });
    expect(row.finalizer_no_output_flags_by_flag).toEqual({
      open_question_rendered: 1,
      with_open_question: 1,
    });
    expect(row.finalizer_no_output_flags_by_primary_reason).toEqual({
      closure: {
        open_question_rendered: 1,
        with_open_question: 1,
      },
    });
    expect(row.finalizer_no_output_when_borg_addressed_with_state_delta_total).toBe(0);
    expect(row.finalizer_no_output_closure_with_open_question_total).toBe(1);
    expect(observed.moodSessions).toEqual([sessionId]);
    expect(observed.tailSessions).toEqual([sessionId, otherSessionId, sessionId, otherSessionId]);
    expect(written).toEqual(row);
  });

  it("derives no-output primary and structural flags from compatibility categories", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-addressed-state",
          event: "post_generation.rejected",
          reason: "finalizer_no_output",
          no_output_categories: ["when_borg_addressed", "with_state_delta"],
        },
        {
          ts: 101,
          turnId: "turn-other",
          event: "post_generation.rejected",
          reason: "finalizer_no_output",
          no_output_categories: [],
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-other",
      2,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row.finalizer_no_output_primary_by_reason).toEqual({
      other: 1,
      when_borg_addressed: 1,
    });
    expect(row.finalizer_no_output_flags_by_flag).toEqual({
      borg_directly_addressed: 1,
      current_turn_state_delta: 1,
      with_state_delta: 1,
    });
    expect(row.finalizer_no_output_flags_by_primary_reason).toEqual({
      when_borg_addressed: {
        borg_directly_addressed: 1,
        current_turn_state_delta: 1,
        with_state_delta: 1,
      },
    });
    expect(row.finalizer_no_output_when_borg_addressed_with_state_delta_total).toBe(1);
    expect(row.finalizer_no_output_closure_with_open_question_total).toBe(0);
  });

  it("writes turn metric keys in v21 order with new fields appended", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    await new MetricsCapture(metricsPath).capture(fakeBorg(), "turn-ordered", 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    const written = JSON.parse(readFileSync(metricsPath, "utf8").trim()) as MetricsRow;

    expect(Object.keys(written)).toEqual([...TURN_METRICS_KEY_ORDER]);
  });

  it("captures embedding cache pending-overflow totals", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    const row = await new MetricsCapture(metricsPath, {
      embeddingCacheStats: () => ({ pending_overflow: 3 }),
    }).capture(fakeBorg(), "turn-embedding-cache-stats", 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    expect(row.embedding_cache_pending_overflow_total).toBe(3);
  });

  it("captures cumulative evidence ledger reverse-scan telemetry from trace events", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        JSON.stringify({
          ts: 1,
          turnId: "turn-ledger-scan-1",
          event: "evidence_ledger.reverse_scan",
          ledger_reverse_scan_entries: 1024,
          ledger_reverse_scan_bytes: 4096,
          ledger_reverse_scan_entry_cap_hit: true,
          ledger_reverse_scan_byte_cap_hit: false,
        }),
        JSON.stringify({
          ts: 2,
          turnId: "turn-ledger-scan-2",
          event: "evidence_ledger.reverse_scan",
          ledger_reverse_scan_entries: 12,
          ledger_reverse_scan_bytes: 8_388_608,
          ledger_reverse_scan_entry_cap_hit: false,
          ledger_reverse_scan_byte_cap_hit: true,
        }),
        JSON.stringify({
          ts: 3,
          turnId: "turn-ledger-scan-2",
          event: "evidence_ledger.image_attach",
          considered_count: 3,
          attached_count: 1,
          omitted_budget_count: 1,
          omitted_inactive_count: 1,
          bytes_attached: 2048,
        }),
      ].join("\n"),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-ledger-scan-2",
      2,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row.ledger_reverse_scan_entries_total).toBe(1036);
    expect(row.ledger_reverse_scan_bytes_total).toBe(8_392_704);
    expect(row.ledger_reverse_scan_entry_cap_hit_total).toBe(1);
    expect(row.ledger_reverse_scan_byte_cap_hit_total).toBe(1);
    expect(row.ledger_image_refs_considered_total).toBe(3);
    expect(row.ledger_image_refs_attached_total).toBe(1);
    expect(row.ledger_image_refs_omitted_budget_total).toBe(1);
    expect(row.ledger_image_refs_omitted_inactive_total).toBe(1);
    expect(row.ledger_image_bytes_attached_total).toBe(2048);
  });

  it("captures session re-entry continuity counters from trace events", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const firstAudience = createEntityId();
    const secondAudience = createEntityId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-reentry-1",
          event: "session_reentry.continuity.evaluated",
          status: "rendered",
          audience_entity_id: firstAudience,
        },
        {
          ts: 101,
          turnId: "turn-reentry-1",
          event: "session_reentry.continuity.rendered",
          status: "rendered",
          audience_entity_id: firstAudience,
        },
        {
          ts: 102,
          turnId: "turn-reentry-2",
          event: "session_reentry.continuity.evaluated",
          status: "blank_audience",
          audience_entity_id: secondAudience,
        },
        {
          ts: 103,
          turnId: "turn-reentry-3",
          event: "session_reentry.continuity.evaluated",
          status: "rendered",
          audience_entity_id: firstAudience,
        },
        {
          ts: 104,
          turnId: "turn-reentry-3",
          event: "session_reentry.continuity.rendered",
          status: "rendered",
          audience_entity_id: firstAudience,
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-reentry-3",
      3,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row.session_reentry_card_rendered_total).toBe(2);
    expect(row.session_reentry_card_rendered_by_audience).toEqual({
      [firstAudience]: 2,
    });
    expect(row.session_reentry_first_turn_with_existing_state_total).toBe(2);
    expect(row.session_reentry_first_turn_blank_audience_total).toBe(1);
  });

  it("carries cumulative extractor health totals onto a clean final turn", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-earlier",
          event: "llm_call.completed",
          label: "closure_loop_classifier",
          stopReason: "max_tokens",
        },
        {
          ts: 101,
          turnId: "turn-earlier",
          event: "closure_loop.degraded",
          label: "closure_loop_classifier",
          stopReason: "max_tokens",
          reason: "missing_tool_call",
        },
        {
          ts: 102,
          turnId: "turn-earlier",
          event: "llm_call.completed",
          label: "corrective_preference_extractor",
          stopReason: "max_tokens",
        },
        {
          ts: 103,
          turnId: "turn-earlier",
          event: "extraction.commitments.degraded",
          label: "corrective_preference_extractor",
          stopReason: "max_tokens",
          reason: "invalid_payload",
        },
        {
          ts: 200,
          turnId: "turn-final-clean",
          event: "llm_call.completed",
          label: "closure_loop_classifier",
          stopReason: "tool_use",
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const capture = new MetricsCapture(metricsPath, { tracePath });
    const row = await capture.capture(fakeBorg(), "turn-final-clean", 2, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    expect(row.extractor_max_tokens_stop_count).toBe(0);
    expect(row.closure_loop_degraded_count).toBe(0);
    expect(row.corrective_preference_degraded_count).toBe(0);
    expect(row.extractor_max_tokens_total_by_label).toEqual({
      closure_loop_classifier: 1,
      corrective_preference_extractor: 1,
    });
    expect(row.extractor_degraded_total_by_label).toEqual({
      closure_loop_classifier: 1,
      corrective_preference_extractor: 1,
    });
  });

  it("does not count simulator health warning labels as extractor degradations", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-warning",
          event: "simulator_health.degraded",
          artifact: "simulator",
          warning_kind: "extractor_max_tokens_high",
          label: "closure_loop_classifier",
          threshold: 15,
          observed_value: 16,
        },
        {
          ts: 200,
          turnId: "turn-clean",
          event: "llm_call.completed",
          label: "closure_loop_classifier",
          stopReason: "tool_use",
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const capture = new MetricsCapture(metricsPath, { tracePath });
    const row = await capture.capture(fakeBorg(), "turn-clean", 2, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    expect(row.extractor_degraded_total_by_label).toEqual({});
    expect(row.closure_loop_degraded_count).toBe(0);
  });

  it("captures shared-state compiler health and operation bias totals", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-compiler-1",
          event: "llm_call.completed",
          label: "shared_state_compiler",
          stopReason: "max_tokens",
        },
        {
          ts: 101,
          turnId: "turn-compiler-1",
          event: "shared_state.compile.degraded",
          reason: "missing_tool_call",
        },
        {
          ts: 101.1,
          turnId: "turn-compiler-1",
          event: "shared_state.compile.repair_attempted",
        },
        {
          ts: 101.2,
          turnId: "turn-compiler-1",
          event: "shared_state.compile.repair_succeeded",
        },
        {
          ts: 101.3,
          turnId: "turn-compiler-failed",
          event: "shared_state.compile.repair_failed",
        },
        {
          ts: 102,
          turnId: "turn-compiler-1",
          event: "shared_state.compile.completed",
          applied: true,
          operation_counts_by_kind: {
            add: 4,
            update: 1,
            supersede: 1,
            prune: 0,
          },
          operation_counts_by_state_key: {
            "plan.attendees": {
              add: 3,
              update: 1,
              supersede: 0,
              prune: 0,
            },
            "decision.architecture": {
              add: 1,
              update: 0,
              supersede: 1,
              prune: 0,
            },
          },
          shared_state_entries_by_key: {
            "plan.attendees": 2,
            "decision.architecture": 1,
          },
          shared_state_top_keys_by_entry_count: {
            "plan.attendees": 2,
            "decision.architecture": 1,
          },
          new_state_key_count: 2,
          keys_with_single_entry_only: 1,
          similar_key_cluster_count: 0,
          update_checked_for_empty_count: 3,
          empty_update_attempted_count: 3,
          empty_update_dropped_count: 2,
          empty_update_repaired_count: 0,
        },
        {
          ts: 102.5,
          turnId: "turn-compiler-1",
          event: "shared_state.compile.add_rejected_cap_exceeded",
          state_key: "plan.attendees",
        },
        {
          ts: 102.6,
          turnId: "turn-compiler-1",
          event: "shared_state.compile.add_rejected_near_duplicate_state_key",
          state_key: "observation.nora.video_call_repeated_question_reconfirm",
          similar_state_keys: ["observation.nora.video_call_repeated_question"],
        },
        {
          ts: 102.7,
          turnId: "turn-compiler-1",
          event: "shared_state.compile.add_rejected_missing_new_key_reason",
          state_key: "decision.architecture.api_boundary",
        },
        {
          ts: 102.8,
          turnId: "turn-compiler-1",
          event: "shared_state.compile.empty_update_dropped",
          operation_index: 2,
          operation_id: "ssa_empty_1",
          state_key: "decision.architecture",
          field_presence: {
            kind: false,
            text: false,
            owner_entity_id: false,
            canonicalizes: false,
          },
        },
        {
          ts: 102.9,
          turnId: "turn-compiler-1",
          event: "shared_state.compile.empty_update_dropped",
          operation_index: 4,
          operation_id: "ssa_empty_2",
          state_key: "plan.attendees",
          field_presence: {
            kind: false,
            text: true,
            owner_entity_id: false,
            canonicalizes: true,
          },
        },
        {
          ts: 103,
          turnId: "turn-compiler-failed",
          event: "shared_state.compile.completed",
          applied: false,
          rejectedCount: 3,
          rejectionReasons: [
            "missing_new_key_reason",
            "relationship_claim_ungrounded",
            "relationship_claim_ungrounded",
          ],
          operation_counts_by_kind: {
            add: 9,
            update: 0,
            supersede: 0,
            prune: 0,
          },
          new_state_key_count: 0,
          keys_with_single_entry_only: 0,
          similar_key_cluster_count: 0,
        },
        {
          ts: 200,
          turnId: "turn-compiler-2",
          event: "llm_call.completed",
          label: "decision_artifact_compiler",
          stopReason: "tool_use",
        },
        {
          ts: 201,
          turnId: "turn-compiler-2",
          event: "shared_state.compile.completed",
          applied: true,
          operation_counts_by_kind: {
            add: 1,
            update: 0,
            supersede: 0,
            prune: 0,
          },
          operation_counts_by_state_key: {
            "plan.attendees": {
              add: 1,
              update: 0,
              supersede: 0,
              prune: 0,
            },
          },
          shared_state_entries_by_key: {
            "plan.attendees": 3,
            "decision.architecture": 1,
          },
          shared_state_top_keys_by_entry_count: {
            "plan.attendees": 3,
            "decision.architecture": 1,
          },
          new_state_key_count: 1,
          keys_with_single_entry_only: 1,
          similar_key_cluster_count: 0,
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const capture = new MetricsCapture(metricsPath, { tracePath });
    const row = await capture.capture(fakeBorg(), "turn-compiler-2", 2, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    expect(row.shared_state_compiler_max_tokens_total).toBe(1);
    expect(row.shared_state_compiler_degraded_total).toBe(1);
    expect(row.shared_state_compiler_repair_attempted_total).toBe(1);
    expect(row.shared_state_compiler_repair_succeeded_total).toBe(1);
    expect(row.shared_state_compiler_repair_failed_total).toBe(1);
    expect(row.shared_state_compiler_repair_failed_by_rejection_reason).toEqual({
      missing_new_key_reason: 1,
      relationship_claim_ungrounded: 2,
    });
    expect(row.shared_state_update_checked_for_empty_total).toBe(3);
    expect(row.shared_state_empty_update_attempted_total).toBe(3);
    expect(row.shared_state_empty_update_dropped_total).toBe(2);
    expect(row.shared_state_empty_update_drop_rate).toBe(2 / 3);
    expect(row.shared_state_empty_update_repaired_total).toBe(0);
    expect(row.shared_state_compiler_operations_total_by_kind).toEqual({
      add: 5,
      update: 1,
      supersede: 1,
      prune: 0,
    });
    expect(row.shared_state_add_to_update_ratio).toBe(2.5);
    expect(row.shared_state_entries_by_key).toEqual({
      "plan.attendees": 3,
      "decision.architecture": 1,
    });
    expect(row.shared_state_add_to_update_ratio_by_key).toEqual({
      "decision.architecture": 1,
      "plan.attendees": 4,
    });
    expect(row.shared_state_top_keys_by_entry_count).toEqual({
      "plan.attendees": 3,
      "decision.architecture": 1,
    });
    expect(row.shared_state_add_rejected_cap_exceeded_total).toBe(1);
    expect(row.shared_state_new_keys_per_compile).toEqual({
      "0": 1,
      "1": 1,
      "2": 1,
    });
    expect(row.shared_state_new_keys_per_turn).toBe(1);
    expect(row.shared_state_keys_with_single_entry_only).toBe(1);
    expect(row.shared_state_similar_key_cluster_count).toBe(0);
    expect(row.shared_state_add_rejected_near_duplicate_state_key_total).toBe(1);
    expect(row.shared_state_add_rejected_missing_new_key_reason_total).toBe(1);
    expect(capture.listHealthWarnings()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "shared_state_compiler_max_tokens_high" }),
        expect.objectContaining({ kind: "shared_state_compiler_add_dominant" }),
      ]),
    );
  });

  it("reports zero empty-update drop rate when no updates were checked", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      JSON.stringify({
        ts: 100,
        turnId: "turn-empty-update-zero",
        event: "shared_state.compile.completed",
        applied: true,
        update_checked_for_empty_count: 0,
        empty_update_dropped_count: 0,
      }),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-empty-update-zero",
      1,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row.shared_state_update_checked_for_empty_total).toBe(0);
    expect(row.shared_state_empty_update_attempted_total).toBe(0);
    expect(row.shared_state_empty_update_dropped_total).toBe(0);
    expect(row.shared_state_empty_update_drop_rate).toBe(0);
  });

  it("counts action candidate classification and embedding-dedup traces", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-action",
          event: "extraction.actions.completed",
          classification_counts: {
            concrete_action: 2,
            conversational_acknowledgment: 1,
            decision_or_preference: 0,
            already_represented: 0,
            outside_borg_capability: 3,
            none: 0,
            invalid_classification: 1,
          },
        },
        {
          ts: 101,
          turnId: "turn-action",
          event: "extraction.actions.rejected",
          classification: "conversational_acknowledgment",
          reason: "non_concrete_classification",
        },
        {
          ts: 102,
          turnId: "turn-action",
          event: "extraction.actions.rejected",
          classification: "concrete_action",
          reason: "embedding_dedup",
        },
        {
          ts: 103,
          turnId: "turn-action",
          event: "action_persistence.dedup.skipped",
          reason: "embedding_dedup",
        },
        {
          ts: 104,
          turnId: "turn-action",
          event: "action_persistence.dedup.degraded",
          reason: "candidate_embedding_failed",
        },
        {
          ts: 105,
          turnId: "turn-action",
          event: "action_state.transitioned",
          action_id: "act_terminal_close",
        },
        {
          ts: 106,
          turnId: "turn-action",
          event: "action_state.borg_self_performance.completed",
          action_id: "act_self_performed",
        },
        {
          ts: 200,
          turnId: "other-turn",
          event: "action_persistence.dedup.skipped",
          reason: "embedding_dedup",
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-action",
      1,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row).toMatchObject({
      action_candidate_classifications_per_turn: {
        ...zeroCounts(ACTION_CANDIDATE_CLASSIFICATIONS),
        concrete_action: 2,
        conversational_acknowledgment: 1,
        outside_borg_capability: 3,
        invalid_classification: 1,
      },
      action_candidate_rejected_classification: 1,
      action_persistence_dedup_skipped_embedding: 1,
      action_persistence_dedup_degraded: 1,
      actions_closed_by_terminal_emission: 1,
      actions_closed_by_borg_self_performance: 1,
      actions_rejected_capability: 3,
    });
  });

  it("counts shared-state action canonicalization lifecycle traces", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-shared-actions",
          event: "shared_state.reconcile.completed",
          mode: "primary",
          actions_retired: 2,
          actions_completed_succeeded: 2,
        },
        {
          ts: 101,
          turnId: "turn-shared-actions",
          event: "decision_artifact_reconcile.completed",
          mode: "retry_only",
          outcome_counts: {
            actions_retired: 1,
            actions_completed_succeeded: 1,
          },
        },
        {
          ts: 200,
          turnId: "other-turn",
          event: "shared_state.reconcile.completed",
          actions_retired: 4,
          actions_completed_succeeded: 4,
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-shared-actions",
      1,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row.actions_canonicalized).toBe(3);
    expect(row.actions_completed_via_canonicalization).toBe(3);
  });

  it("captures shared-state cap pressure from compile traces", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-shared-cap-1",
          event: "decision_artifact_compile.completed",
          artifact_active_entry_count: 40,
          artifact_max_active_entries: 40,
          artifact_omitted_entry_count: 3,
          omitted_live_recent_operational: 1,
          omitted_live_recent_low_salience: 2,
          omitted_live_old: 3,
          omitted_live_unknown_age: 2,
          omitted_locked: 4,
          omitted_locked_recent_final_compile: 1,
          omitted_locked_old_final_compile: 2,
          omitted_locked_unknown_age_final_compile: 1,
          omitted_locked_with_active_critical_commitment_final_compile: 1,
          omitted_locked_with_operational_canonicalizer_final_compile: 2,
          omitted_locked_indexed_only_final_compile: 4,
          omitted_pending: 5,
          omitted_low_salience_live: 6,
          omitted_dormant_live: 7,
          all_active_keys_indexed: true,
          newest_entries_reserved: 2,
          add_rejected_cap_exceeded_count: 1,
          active_by_kind: {
            low_salience_live: 1,
            dormant_live: 0,
          },
          rendered_by_kind: {
            locked: 14,
            live: 8,
            pending: 2,
            invalidated: 0,
            tentative: 0,
          },
          omitted_by_kind: {
            locked: 0,
            live: 2,
            pending: 1,
            invalidated: 0,
            tentative: 0,
          },
          live_starvation_with_reserved: true,
        },
        {
          ts: 101,
          turnId: "turn-shared-cap-2",
          event: "shared_state.compile.completed",
          artifact_active_entry_count: 39,
          artifact_max_active_entries: 40,
          artifact_omitted_entry_count: 0,
          omitted_live_recent_operational: 6,
          omitted_live_recent_low_salience: 7,
          omitted_live_old: 8,
          omitted_live_unknown_age: 3,
          omitted_locked: 9,
          omitted_locked_recent_final_compile: 2,
          omitted_locked_old_final_compile: 3,
          omitted_locked_unknown_age_final_compile: 4,
          omitted_locked_with_active_critical_commitment_final_compile: 0,
          omitted_locked_with_operational_canonicalizer_final_compile: 1,
          omitted_locked_indexed_only_final_compile: 9,
          omitted_pending: 10,
          omitted_low_salience_live: 11,
          omitted_dormant_live: 12,
          all_active_keys_indexed: false,
          newest_entries_reserved: 1,
          active_by_kind: {
            low_salience_live: 2,
            dormant_live: 1,
          },
          lifecycle_aging_blocker_counts_live_to_low_salience: {
            demotable_count: 5,
            demoted_count: 1,
            blocked_by_current_turn_update: 2,
            blocked_by_patch_touch: 3,
            blocked_by_ledger_overlap: 4,
            blocked_by_recent_retrieval: 5,
            blocked_by_active_canonicalizer_critical: 6,
            blocked_by_active_canonicalizer_operational: 7,
            blocked_by_hard_total: 9,
            blocked_by_soft_total: 10,
            unknown_age_count: 7,
            blocked_by_multiple_reasons: 8,
          },
          lifecycle_aging_blocker_counts_low_salience_to_dormant: {
            demotable_count: 2,
            demoted_count: 1,
            blocked_by_current_turn_update: 1,
            blocked_by_patch_touch: 2,
            blocked_by_ledger_overlap: 3,
            blocked_by_recent_retrieval: 4,
            blocked_by_active_canonicalizer_critical: 5,
            blocked_by_active_canonicalizer_operational: 6,
            blocked_by_hard_total: 7,
            blocked_by_soft_total: 8,
            unknown_age_count: 6,
            blocked_by_multiple_reasons: 7,
          },
          rendered_by_kind: {},
          omitted_by_kind: {},
        },
        {
          ts: 102,
          turnId: "turn-shared-cap-3",
          event: "shared_state.compile.completed",
          artifact_active_entry_count: 40,
          artifact_max_active_entries: 40,
          artifact_omitted_entry_count: 3,
          omitted_live_recent_operational: 11,
          omitted_live_recent_low_salience: 13,
          omitted_live_old: 17,
          omitted_live_unknown_age: 5,
          omitted_locked: 19,
          omitted_locked_recent_final_compile: 5,
          omitted_locked_old_final_compile: 10,
          omitted_locked_unknown_age_final_compile: 4,
          omitted_locked_with_active_critical_commitment_final_compile: 2,
          omitted_locked_with_operational_canonicalizer_final_compile: 3,
          omitted_locked_indexed_only_final_compile: 19,
          omitted_pending: 23,
          omitted_low_salience_live: 29,
          omitted_dormant_live: 31,
          all_active_keys_indexed: true,
          newest_entries_reserved: 3,
          active_by_kind: {
            low_salience_live: 3,
            dormant_live: 2,
          },
          lifecycle_aging_blocker_counts_live_to_low_salience: {
            demotable_count: 11,
            demoted_count: 0,
            blocked_by_current_turn_update: 13,
            blocked_by_patch_touch: 17,
            blocked_by_ledger_overlap: 19,
            blocked_by_recent_retrieval: 23,
            blocked_by_active_canonicalizer_critical: 29,
            blocked_by_active_canonicalizer_operational: 31,
            blocked_by_hard_total: 30,
            blocked_by_soft_total: 32,
            unknown_age_count: 31,
            blocked_by_multiple_reasons: 37,
          },
          lifecycle_aging_blocker_counts_low_salience_to_dormant: {
            demotable_count: 3,
            demoted_count: 0,
            blocked_by_current_turn_update: 4,
            blocked_by_patch_touch: 5,
            blocked_by_ledger_overlap: 6,
            blocked_by_recent_retrieval: 7,
            blocked_by_active_canonicalizer_critical: 8,
            blocked_by_active_canonicalizer_operational: 9,
            blocked_by_hard_total: 10,
            blocked_by_soft_total: 11,
            unknown_age_count: 9,
            blocked_by_multiple_reasons: 10,
          },
          rendered_by_kind: {
            locked: 10,
          },
          omitted_by_kind: {
            invalidated: 3,
          },
        },
        {
          ts: 103,
          turnId: "turn-shared-cap-3",
          event: "shared_state.lifecycle.demoted",
          from_kind: "live",
          to_kind: "low_salience_live",
        },
        {
          ts: 104,
          turnId: "turn-shared-cap-3",
          event: "shared_state.lifecycle.demoted",
          from_kind: "low_salience_live",
          to_kind: "dormant_live",
        },
        {
          ts: 105,
          turnId: "turn-shared-cap-3",
          event: "shared_state.lifecycle.reactivated",
          from_kind: "low_salience_live",
          to_kind: "live",
        },
        {
          ts: 106,
          turnId: "turn-shared-cap-3",
          event: "shared_state.lifecycle.reactivated",
          from_kind: "dormant_live",
          to_kind: "live",
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-shared-cap-3",
      4,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row.shared_state_at_cap_turns).toBe(2);
    expect(row.shared_state_compile_evaluated_turns).toBe(3);
    expect(row.shared_state_omitted_recent_entries).toBe(5);
    expect(row.shared_state_omitted_recent_entries_total_across_compiles).toBe(5);
    expect(row.shared_state_omitted_live_recent_operational).toBe(18);
    expect(row.shared_state_omitted_live_recent_operational_total_across_compiles).toBe(18);
    expect(row.shared_state_omitted_live_recent_operational_final_compile).toBe(11);
    expect(row.shared_state_omitted_live_recent_low_salience).toBe(22);
    expect(row.shared_state_omitted_live_recent_low_salience_total_across_compiles).toBe(22);
    expect(row.shared_state_omitted_live_recent_low_salience_final_compile).toBe(13);
    expect(row.shared_state_omitted_live_old).toBe(28);
    expect(row.shared_state_omitted_live_old_total_across_compiles).toBe(28);
    expect(row.shared_state_omitted_live_old_final_compile).toBe(17);
    expect(row.shared_state_omitted_live_unknown_age).toBe(10);
    expect(row.shared_state_omitted_live_unknown_age_total_across_compiles).toBe(10);
    expect(row.shared_state_omitted_live_unknown_age_final_compile).toBe(5);
    expect(row.shared_state_omitted_locked).toBe(32);
    expect(row.shared_state_omitted_locked_total_across_compiles).toBe(32);
    expect(row.shared_state_omitted_locked_final_compile).toBe(19);
    expect(row.shared_state_omitted_locked_recent_total_across_compiles).toBe(8);
    expect(row.shared_state_omitted_locked_recent_final_compile).toBe(5);
    expect(row.shared_state_omitted_locked_old_total_across_compiles).toBe(15);
    expect(row.shared_state_omitted_locked_old_final_compile).toBe(10);
    expect(row.shared_state_omitted_locked_unknown_age_total_across_compiles).toBe(9);
    expect(row.shared_state_omitted_locked_unknown_age_final_compile).toBe(4);
    expect(
      row.shared_state_omitted_locked_with_active_critical_commitment_total_across_compiles,
    ).toBe(3);
    expect(row.shared_state_omitted_locked_with_active_critical_commitment_final_compile).toBe(2);
    expect(
      row.shared_state_omitted_locked_with_operational_canonicalizer_total_across_compiles,
    ).toBe(6);
    expect(row.shared_state_omitted_locked_with_operational_canonicalizer_final_compile).toBe(3);
    expect(row.shared_state_omitted_locked_indexed_only_total_across_compiles).toBe(32);
    expect(row.shared_state_omitted_locked_indexed_only_final_compile).toBe(19);
    expect(row.shared_state_omitted_pending).toBe(38);
    expect(row.shared_state_omitted_pending_total_across_compiles).toBe(38);
    expect(row.shared_state_omitted_pending_final_compile).toBe(23);
    expect(row.shared_state_omitted_low_salience_live).toBe(29);
    expect(row.shared_state_omitted_low_salience_live_final_compile).toBe(29);
    expect(row.shared_state_omitted_dormant_live).toBe(31);
    expect(row.shared_state_omitted_dormant_live_final_compile).toBe(31);
    expect(row.shared_state_active_low_salience_live).toBe(3);
    expect(row.shared_state_active_low_salience_live_final_compile).toBe(3);
    expect(row.shared_state_active_dormant_live).toBe(2);
    expect(row.shared_state_active_dormant_live_final_compile).toBe(2);
    expect(row.shared_state_demoted_live_to_low_salience_total).toBe(1);
    expect(row.shared_state_demoted_low_salience_to_dormant_total).toBe(1);
    expect(row.shared_state_lifecycle_aging_demotable_total).toBe(16);
    expect(row.shared_state_lifecycle_aging_demotable_final_compile).toBe(11);
    expect(row.shared_state_lifecycle_aging_demoted_total).toBe(1);
    expect(row.shared_state_lifecycle_aging_demoted_final_compile).toBe(0);
    expect(row.shared_state_lifecycle_aging_blocked_by_current_turn_update_total).toBe(15);
    expect(row.shared_state_lifecycle_aging_blocked_by_current_turn_update_final_compile).toBe(13);
    expect(row.shared_state_lifecycle_aging_blocked_by_patch_touch_total).toBe(20);
    expect(row.shared_state_lifecycle_aging_blocked_by_patch_touch_final_compile).toBe(17);
    expect(row.shared_state_lifecycle_aging_blocked_by_ledger_overlap_total).toBe(23);
    expect(row.shared_state_lifecycle_aging_blocked_by_ledger_overlap_final_compile).toBe(19);
    expect(row.shared_state_lifecycle_aging_blocked_by_recent_retrieval_total).toBe(28);
    expect(row.shared_state_lifecycle_aging_blocked_by_recent_retrieval_final_compile).toBe(23);
    expect(row.shared_state_lifecycle_aging_blocked_by_active_canonicalizer_critical_total).toBe(
      35,
    );
    expect(
      row.shared_state_lifecycle_aging_blocked_by_active_canonicalizer_critical_final_compile,
    ).toBe(29);
    expect(row.shared_state_lifecycle_aging_blocked_by_active_canonicalizer_operational_total).toBe(
      38,
    );
    expect(
      row.shared_state_lifecycle_aging_blocked_by_active_canonicalizer_operational_final_compile,
    ).toBe(31);
    expect(row.shared_state_lifecycle_aging_blocked_by_hard_total).toBe(39);
    expect(row.shared_state_lifecycle_aging_blocked_by_hard_final_compile).toBe(30);
    expect(row.shared_state_lifecycle_aging_blocked_by_soft_total).toBe(42);
    expect(row.shared_state_lifecycle_aging_blocked_by_soft_final_compile).toBe(32);
    expect(row.shared_state_lifecycle_aging_unknown_age_total).toBe(38);
    expect(row.shared_state_lifecycle_aging_unknown_age_final_compile).toBe(31);
    expect(row.shared_state_lifecycle_aging_blocked_by_multiple_reasons_total).toBe(45);
    expect(row.shared_state_lifecycle_aging_blocked_by_multiple_reasons_final_compile).toBe(37);
    expect(row.shared_state_lifecycle_aging_low_salience_to_dormant_demotable_total).toBe(5);
    expect(row.shared_state_lifecycle_aging_low_salience_to_dormant_demotable_final_compile).toBe(
      3,
    );
    expect(row.shared_state_lifecycle_aging_low_salience_to_dormant_demoted_total).toBe(1);
    expect(row.shared_state_lifecycle_aging_low_salience_to_dormant_demoted_final_compile).toBe(0);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_current_turn_update_total,
    ).toBe(5);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_current_turn_update_final_compile,
    ).toBe(4);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_patch_touch_total,
    ).toBe(7);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_patch_touch_final_compile,
    ).toBe(5);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_ledger_overlap_total,
    ).toBe(9);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_ledger_overlap_final_compile,
    ).toBe(6);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_recent_retrieval_total,
    ).toBe(11);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_recent_retrieval_final_compile,
    ).toBe(7);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_critical_total,
    ).toBe(13);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_critical_final_compile,
    ).toBe(8);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_operational_total,
    ).toBe(15);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_operational_final_compile,
    ).toBe(9);
    expect(row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_hard_total).toBe(17);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_hard_final_compile,
    ).toBe(10);
    expect(row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_soft_total).toBe(19);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_soft_final_compile,
    ).toBe(11);
    expect(row.shared_state_lifecycle_aging_low_salience_to_dormant_unknown_age_total).toBe(15);
    expect(row.shared_state_lifecycle_aging_low_salience_to_dormant_unknown_age_final_compile).toBe(
      9,
    );
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_multiple_reasons_total,
    ).toBe(17);
    expect(
      row.shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_multiple_reasons_final_compile,
    ).toBe(10);
    expect(row.shared_state_reactivated_low_salience_live_total).toBe(1);
    expect(row.shared_state_reactivated_dormant_live_total).toBe(1);
    expect(row.shared_state_at_cap_but_all_keys_indexed_compiles_total).toBe(2);
    expect(row.shared_state_at_cap_with_operational_omission_compiles_total).toBe(2);
    expect(row.shared_state_at_cap_with_cap_rejection_compiles_total).toBe(1);
    expect(row.shared_state_all_active_keys_indexed).toBe(false);
    expect(row.shared_state_live_entry_starvation).toBe(true);
    expect(row.shared_state_newest_entries_reserved).toBe(6);
    expect(row.shared_state_live_starvation_with_reserved).toBe(true);
    expect(row.shared_state_live_starvation_ever).toBe(true);
    expect(row.shared_state_live_starvation_final).toBe(false);
  });

  it("reports persistent shared-state starvation on the latest compile", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-shared-recovered",
          event: "shared_state.compile.completed",
          rendered_by_kind: {
            locked: 12,
          },
          omitted_by_kind: {
            live: 0,
          },
          live_starvation_with_reserved: false,
        },
        {
          ts: 101,
          turnId: "turn-shared-persistent",
          event: "shared_state.compile.completed",
          rendered_by_kind: {
            locked: 12,
          },
          omitted_by_kind: {
            live: 2,
          },
          live_starvation_with_reserved: true,
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-shared-persistent",
      4,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row.shared_state_live_starvation_ever).toBe(true);
    expect(row.shared_state_live_starvation_final).toBe(true);
  });

  it("splits active actions by Borg, participant, and group actor ownership", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const audience = createEntityId();
    const participant = createEntityId();
    const actions = [
      makeAction({
        actor: "borg",
        audience_entity_id: audience,
        state: "committed_to_do",
      }),
      makeAction({
        actor: "user",
        audience_entity_id: audience,
        state: "considering",
      }),
      makeAction({
        actor: participant,
        audience_entity_id: audience,
        state: "scheduled",
      }),
      makeAction({
        actor: audience,
        audience_entity_id: audience,
        state: "committed_to_do",
      }),
      makeAction({
        actor: "borg",
        audience_entity_id: audience,
        state: "completed",
        completed_at: 1_000,
      }),
    ];
    const activeStates = new Set(["considering", "committed_to_do", "scheduled", "unknown"]);
    const countByState = () => {
      const counts = zeroCounts(ACTION_STATES);

      for (const action of actions) {
        counts[action.state] += 1;
      }

      return counts;
    };
    const borg = {
      ...fakeBorg(),
      actions: {
        count: () => actions.length,
        countByState,
        countCanonicalized: () => 0,
        countActive: () => actions.filter((action) => activeStates.has(action.state)).length,
        getCreationCountsBySource: () => ({
          extractor: 0,
          reflector: 0,
          api: 0,
          tool: 0,
          unknown: 0,
        }),
        countCompletedSince: () => 0,
        latestCompletedAt: () => null,
        listCompletedIds: () => [],
        list: (filter: { states?: readonly ActionRecord["state"][] } = {}) =>
          actions.filter(
            (action) => filter.states === undefined || filter.states.includes(action.state),
          ),
      },
    } as unknown as Borg;

    const row = await new MetricsCapture(metricsPath).capture(borg, "turn-actor-split", 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    expect(row.action_record_count_active).toBe(4);
    expect(row.borg_owned_active_actions).toBe(1);
    expect(row.participant_owned_active_actions).toBe(2);
    expect(row.group_owned_active_actions).toBe(1);
    expect(row.prompt_salient_actions_total).toBe(2);
    expect(row.borg_owned_salient_active_actions).toBe(1);
    expect(row.participant_owned_salient_active_actions).toBe(0);
    expect(row.actions_per_turn).toBe(5);
    expect(row.salient_actions_per_turn).toBe(2);
    expect(row.action_retirement_ratio).toBe(0.2);
    expect(row.borg_owned_action_count).toBe(2);
    expect(row.stale_action_count).toBe(0);
  });

  it("captures prompt-salient action counts and stale prompt omissions", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const audience = createEntityId();
    const actions = [
      makeAction({
        actor: "borg",
        audience_entity_id: audience,
        state: "committed_to_do",
        last_referenced_turn_counter: 2,
        last_referenced_turn_global: 20,
      }),
      makeAction({
        actor: "user",
        audience_entity_id: audience,
        state: "committed_to_do",
        last_referenced_turn_counter: 1,
        last_referenced_turn_global: 19,
      }),
      ...Array.from({ length: 7 }, (_, index) =>
        makeAction({
          actor: "user",
          audience_entity_id: audience,
          description: `Stale participant task ${index}`,
          state: "committed_to_do",
          created_at: 1_000 - index,
          updated_at: 1_000 - index,
          // A row can retain its legacy session stamp alongside the dedicated
          // lifecycle-global stamp. Metrics must prefer the global field.
          last_referenced_turn_counter: 20,
          last_referenced_turn_global: 0,
        }),
      ),
      makeAction({
        actor: "borg",
        audience_entity_id: audience,
        state: "expired",
        expired_at: 1_000,
      }),
    ];
    const activeStates = new Set(["considering", "committed_to_do", "scheduled", "unknown"]);
    const countByState = () => {
      const counts = zeroCounts(ACTION_STATES);

      for (const action of actions) {
        counts[action.state] += 1;
      }

      return counts;
    };
    const borg = {
      ...fakeBorg(),
      actions: {
        count: () => actions.length,
        countByState,
        countCanonicalized: () => 0,
        countActive: () => actions.filter((action) => activeStates.has(action.state)).length,
        getCreationCountsBySource: () => ({
          extractor: 0,
          reflector: 0,
          api: 0,
          tool: 0,
          unknown: 0,
        }),
        countCompletedSince: () => 0,
        latestCompletedAt: () => null,
        listCompletedIds: () => [],
        list: (filter: { states?: readonly ActionRecord["state"][] } = {}) =>
          actions.filter(
            (action) => filter.states === undefined || filter.states.includes(action.state),
          ),
        findSimilarDescriptionPairs: async () => [],
      },
    } as unknown as Borg;

    const row = await new MetricsCapture(metricsPath).capture(borg, "turn-salience", 20, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    expect(row.prompt_salient_actions_total).toBe(2);
    expect(row.borg_owned_salient_active_actions).toBe(1);
    expect(row.participant_owned_salient_active_actions).toBe(1);
    expect(row.stale_actions_omitted_from_prompt).toBe(2);
    expect(row.dormant_actions_total).toBe(7);
    expect(row.stale_action_count).toBe(8);
    expect(row.salient_actions_per_turn).toBe(0.1);
    expect(row.action_retirement_ratio).toBe(0.1);
  });

  it("captures dormant/archive eligibility and inactive turn distributions", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const audience = createEntityId();
    const actions = [
      makeAction({
        actor: "user",
        audience_entity_id: audience,
        state: "committed_to_do",
        last_referenced_turn_counter: 40,
      }),
      makeAction({
        actor: "user",
        audience_entity_id: audience,
        state: "committed_to_do",
        last_referenced_turn_counter: 25,
      }),
      makeAction({
        actor: "user",
        audience_entity_id: audience,
        state: "considering",
        last_referenced_turn_counter: 21,
      }),
      makeAction({
        actor: "borg",
        audience_entity_id: audience,
        state: "committed_to_do",
        last_referenced_turn_counter: 20,
      }),
      makeAction({
        actor: "user",
        audience_entity_id: audience,
        state: "scheduled",
        scheduled_at: 1_000,
        last_referenced_turn_counter: 10,
      }),
      makeAction({
        actor: "user",
        audience_entity_id: audience,
        state: "committed_to_do",
        last_referenced_turn_counter: 5,
      }),
      makeAction({
        actor: "user",
        audience_entity_id: audience,
        state: "completed",
        completed_at: 1_000,
        last_referenced_turn_counter: 0,
      }),
      makeAction({
        actor: "user",
        audience_entity_id: audience,
        state: "unknown",
        last_referenced_turn_counter: null,
      }),
    ];

    writeFileSync(
      tracePath,
      `${JSON.stringify({
        event: "action_archive_scan.completed",
        turnId: "turn-archive-visibility",
        scanned_count: 6,
        eligible_count: 1,
        archived_count: 1,
        skipped_by_reason: {
          below_inactive_threshold: 3,
          borg_owned: 1,
          scheduled_or_due: 1,
        },
        oldest_inactive_turns: 35,
        oldest_eligible_inactive_turns: 35,
        archive_after_turns: 20,
      })}\n`,
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorgWithActions(actions),
      "turn-archive-visibility",
      40,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row.dormant_not_archive_eligible_count).toBe(3);
    expect(row.dormant_archive_eligible_count).toBe(1);
    expect(row.archive_archivable_count).toBe(1);
    expect(row.archive_skipped_borg_owned).toBe(1);
    expect(row.archive_skipped_due_date).toBe(1);
    expect(row.archive_skipped_below_threshold).toBe(3);
    expect(row.archive_skipped_other).toBe(0);
    expect(row.archive_oldest_inactive_turns).toBe(35);
    expect(row.archive_oldest_archivable_inactive_turns).toBe(35);
    expect(row.archive_inactive_turn_distribution).toEqual({
      "0-15": 1,
      "15-20": 2,
      "20-30": 0,
      "30+": 1,
    });
  });

  it("counts goal-promotion salvage and initial-step downgrade traces", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-goal",
          event: "extraction.goals.completed",
          salvaged_promotion_count: 2,
          skipped_promotion_count: 1,
          classification_counts: {
            durable_borg_goal: 2,
            one_off: 1,
            not_borg_responsibility: 0,
            impossible_for_borg_without_capability: 0,
            already_represented: 0,
            none: 0,
            invalid_classification: 1,
          },
        },
        {
          ts: 100.5,
          turnId: "turn-goal",
          event: "extraction.goals.rejected",
          classification: "one_off",
          reason: "non_durable_classification",
        },
        {
          ts: 100.6,
          turnId: "turn-goal",
          event: "extraction.goals.rejected",
          classification: "durable_borg_goal",
          reason: "cap_exceeded",
        },
        {
          ts: 101,
          turnId: "turn-goal",
          event: "extraction.goals.transitioned",
          reason: "wait_without_due_at",
        },
        {
          ts: 102,
          turnId: "turn-goal",
          event: "extraction.goals.skipped",
          reason: "extractor_signal",
        },
        {
          ts: 103,
          turnId: "turn-goal",
          event: "extraction.goals.skipped",
          reason: "embedding",
        },
        {
          ts: 104,
          turnId: "turn-goal",
          event: "extraction.goals.dedup.degraded",
          reason: "candidate_embedding_failed",
        },
        {
          ts: 200,
          turnId: "other-turn",
          event: "extraction.goals.completed",
          salvaged_promotion_count: 1,
          skipped_promotion_count: 1,
          classification_counts: {
            durable_borg_goal: 1,
            one_off: 0,
            not_borg_responsibility: 0,
            impossible_for_borg_without_capability: 0,
            already_represented: 0,
            none: 0,
            invalid_classification: 0,
          },
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-goal",
      1,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row).toMatchObject({
      goal_promotion_salvaged_promotions: 2,
      goal_promotion_skipped_promotions: 1,
      goal_promotion_initial_step_downgraded: 1,
      goal_promotion_dedup_skipped_extractor_signal: 1,
      goal_promotion_dedup_skipped_embedding: 1,
      goal_promotion_dedup_degraded: 1,
      goal_promotion_classifications_per_turn: {
        durable_borg_goal: 2,
        one_off: 1,
        not_borg_responsibility: 0,
        impossible_for_borg_without_capability: 0,
        already_represented: 0,
        none: 0,
        invalid_classification: 1,
      },
      goal_promotion_rejected_classification: 1,
      goal_promotion_cap_rejections: 1,
    });
  });

  it("emits simulator health warning traces when active goals are high", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg({ activeGoals: 26 }),
      "turn-health-high",
      5,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );
    const trace = readFileSync(tracePath, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);

    expect(row.active_goal_count).toBe(26);
    expect(trace).toContainEqual(
      expect.objectContaining({
        event: "simulator_health.degraded",
        warning_kind: "active_goals_high",
        turn_counter: 5,
        threshold: 25,
        observed_value: 26,
      }),
    );
  });

  it("emits active-goals-high warnings only on rising edges", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const capture = new MetricsCapture(metricsPath, { tracePath });

    await capture.capture(fakeBorg({ activeGoals: 26 }), "turn-warning-rise-1", 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    await capture.capture(fakeBorg({ activeGoals: 27 }), "turn-warning-still-high", 2, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    await capture.capture(fakeBorg({ activeGoals: 24 }), "turn-warning-cleared", 3, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    await capture.capture(fakeBorg({ activeGoals: 28 }), "turn-warning-rise-2", 4, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    const warningEvents = readFileSync(tracePath, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>)
      .filter(
        (record) =>
          record.event === "simulator_health.degraded" &&
          record.warning_kind === "active_goals_high",
      );

    expect(warningEvents).toEqual([
      expect.objectContaining({
        turnId: "turn-warning-rise-1",
        observed_value: 26,
      }),
      expect.objectContaining({
        turnId: "turn-warning-rise-2",
        observed_value: 28,
      }),
    ]);
    expect(capture.listHealthWarnings().map((warning) => warning.turnId)).toEqual([
      "turn-warning-rise-1",
      "turn-warning-rise-2",
    ]);
  });

  it("emits simulator health warning traces when active goal growth is high after turn twenty", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const capture = new MetricsCapture(metricsPath, { tracePath });

    for (let turn = 21; turn <= 30; turn += 1) {
      await capture.capture(fakeBorg({ activeGoals: turn - 20 }), `turn-growth-${turn}`, turn, {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      });
    }

    const trace = readFileSync(tracePath, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);

    expect(trace).toContainEqual(
      expect.objectContaining({
        event: "simulator_health.degraded",
        warning_kind: "active_goals_growth_high",
        turn_counter: 30,
        threshold: 0.5,
        observed_value: 1,
        window_start_turn: 21,
        window_turns: 9,
      }),
    );
  });

  it("counts frame-anomaly classifier, fallback, and durable quarantine markers", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const anomalyTurnId = "turn-frame-anomaly";
    const degradedTurnId = "turn-frame-degraded";
    const normalTurnId = "turn-frame-normal";
    const quarantinedUserEntryId = createStreamEntryId();
    const streamEntriesBySession = new Map<SessionId, StreamEntry[]>([
      [
        sessionId,
        [
          {
            id: createStreamEntryId(),
            timestamp: 1,
            kind: "internal_event",
            content: {
              event: "quarantined_user_entry",
              turn_id: anomalyTurnId,
              source_stream_entry_id: quarantinedUserEntryId,
              kind: "frame_assignment_claim",
            },
            turn_id: anomalyTurnId,
            session_id: sessionId,
            compressed: false,
            sender_entity_id: null,
            reply_target_entity_id: null,
          },
        ],
      ],
    ]);

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: anomalyTurnId,
          event: "llm_call.started",
          label: "frame_anomaly_classifier",
        },
        {
          ts: 101,
          turnId: anomalyTurnId,
          event: "frame_anomaly.completed",
          status: "ok",
          kind: "frame_assignment_claim",
        },
        {
          ts: 200,
          turnId: degradedTurnId,
          event: "llm_call.started",
          label: "frame_anomaly_classifier",
        },
        {
          ts: 201,
          turnId: degradedTurnId,
          event: "frame_anomaly.completed",
          status: "degraded",
          reason: "invalid_payload",
        },
        {
          ts: 202,
          turnId: degradedTurnId,
          event: "frame_anomaly.fallback.completed",
          pattern: "i'm claude",
          kind: "assistant_self_claim_in_user_role",
          matched: true,
        },
        {
          ts: 300,
          turnId: normalTurnId,
          event: "llm_call.started",
          label: "frame_anomaly_classifier",
        },
        {
          ts: 301,
          turnId: normalTurnId,
          event: "frame_anomaly.completed",
          status: "ok",
          kind: "normal",
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const capture = new MetricsCapture(metricsPath, { tracePath });
    const borg = fakeBorg({ streamEntriesBySession });
    const anomaly = await capture.capture(borg, anomalyTurnId, 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    const degraded = await capture.capture(borg, degradedTurnId, 2, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    const normal = await capture.capture(borg, normalTurnId, 3, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    expect(anomaly).toMatchObject({
      frame_anomaly_classifier_calls: 1,
      frame_anomaly_classified_normal_count: 0,
      frame_anomaly_actual_anomaly_count: 1,
      frame_anomaly_degraded_count: 0,
      frame_anomaly_degraded_fallback_match_count: 0,
      quarantined_user_entry_count: 1,
      early_extractors_skipped_frame_anomaly_count: 1,
    });
    expect(degraded).toMatchObject({
      frame_anomaly_classifier_calls: 1,
      frame_anomaly_classified_normal_count: 0,
      frame_anomaly_actual_anomaly_count: 0,
      frame_anomaly_degraded_count: 1,
      frame_anomaly_degraded_fallback_match_count: 1,
      quarantined_user_entry_count: 0,
      early_extractors_skipped_frame_anomaly_count: 1,
    });
    expect(normal).toMatchObject({
      frame_anomaly_classifier_calls: 1,
      frame_anomaly_classified_normal_count: 1,
      frame_anomaly_actual_anomaly_count: 0,
      frame_anomaly_degraded_count: 0,
      frame_anomaly_degraded_fallback_match_count: 0,
      quarantined_user_entry_count: 0,
      early_extractors_skipped_frame_anomaly_count: 0,
    });
  });

  it("records semantic graph growth since the previous capture", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const capture = new MetricsCapture(metricsPath);
    const sessionId = createSessionId();

    await capture.capture(fakeBorg({ semanticNodes: 1, semanticEdges: 2 }), "turn-1", 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    const row = await capture.capture(
      fakeBorg({ semanticNodes: 4, semanticEdges: 5 }),
      "turn-2",
      2,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row.semantic_nodes_added_since_last_check).toBe(3);
    expect(row.semantic_edges_added_since_last_check).toBe(3);
  });

  it("counts backdated completed actions as newly completed by id", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const db = openDatabase(join(dir, "borg.db"), {
      migrations: actionMigrations,
    });
    const clock = new ManualClock(1_000);
    const capture = new MetricsCapture(metricsPath);
    const sessionId = createSessionId();
    const actions = new ActionRepository({ db, clock });
    const borg = {
      ...fakeBorg(),
      actions,
    } as unknown as Borg;

    try {
      actions.add(
        makeAction({
          description: "First completed action",
          state: "completed",
          created_at: 100,
          updated_at: 100,
          completed_at: 100,
        }),
      );

      const firstRow = await capture.capture(borg, "turn-complete-first", 1, {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      });

      actions.add(
        makeAction({
          description: "Backdated completed action",
          state: "completed",
          created_at: 99,
          updated_at: 101,
          completed_at: 99,
        }),
      );

      const secondRow = await capture.capture(borg, "turn-complete-second", 2, {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      });

      expect(firstRow.recent_completed_action_count).toBe(1);
      expect(secondRow.recent_completed_action_count).toBe(1);
    } finally {
      db.close();
    }
  });

  it("counts open questions resolved through the identity update path", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const tracePath = join(dir, "trace.jsonl");
    const db = openDatabase(join(dir, "borg.db"), {
      migrations: composeMigrations(selfMigrations, commitmentMigrations, identityMigrations),
    });
    const clock = new ManualClock(1_000);
    const sessionId = createSessionId();
    const { identity, openQuestionsRepository } = createIdentityHarness(db, clock);
    const provenance = { kind: "manual" } as const;

    try {
      const question = identity.addOpenQuestion({
        question: "Which metrics path resolves this?",
        urgency: 0.7,
        related_episode_ids: [],
        related_semantic_node_ids: [],
        provenance,
        source: "user",
      });
      identity.addOpenQuestion({
        question: "Does the review item need follow-up?",
        urgency: 0.6,
        related_episode_ids: [],
        related_semantic_node_ids: [],
        provenance,
        source: "overseer",
      });
      const result = identity.updateOpenQuestion(
        question.id,
        {
          status: OPEN_QUESTION_RESOLVED_STATUS,
          resolution_evidence_stream_entry_ids: [createStreamEntryId()],
          resolution_note: "The metrics update path resolved it.",
          resolved_at: clock.now(),
        },
        provenance,
        {
          throughReview: true,
        },
      );
      const borg = {
        ...fakeBorg(),
        identity: {
          listEvents: (...args: Parameters<IdentityService["listEvents"]>) =>
            identity.listEvents(...args),
        },
        self: {
          openQuestions: {
            list: (...args: Parameters<OpenQuestionsRepository["list"]>) =>
              openQuestionsRepository.list(...args),
          },
          goals: {
            list: () => [],
          },
        },
      } as unknown as Borg;
      writeFileSync(
        tracePath,
        `${JSON.stringify({
          ts: 1_001,
          turnId: "turn-oq-update",
          event: "evidence_ledger.completed",
          entry_counts: {
            open_questions: 2,
          },
        })}\n`,
      );

      const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
        borg,
        "turn-oq-update",
        1,
        {
          sessionId,
          sessionIds: [sessionId],
          transportChatAttempts: 1,
        },
      );

      expect(result.status).toBe("applied");
      expect(row.open_question_resolved_count).toBe(1);
      expect(row.open_question_count).toBe(1);
      expect(row.open_questions_by_source).toMatchObject({
        user_question: 1,
        review_promoted: 1,
      });
      expect(row.open_questions_by_status_age).toMatchObject({
        "open:<3_turns": 1,
        "resolved:<3_turns": 1,
      });
      expect(row.open_questions_resolved_this_run).toBe(1);
      expect(row.open_questions_rendered_to_finalizer_this_turn).toBe(2);
      expect(row.open_questions_promoted_from_review_items).toBe(1);
    } finally {
      db.close();
    }
  });

  it("captures simulator metrics for action, commitment, working-memory, relational slot, review, and open-question bands", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const db = openDatabase(join(dir, "borg.db"), {
      migrations: composeMigrations(
        actionMigrations,
        commitmentMigrations,
        relationalSlotMigrations,
        semanticMigrations,
        selfMigrations,
        identityMigrations,
      ),
    });
    const clock = new ManualClock(1_000);
    const sessionId = createSessionId();

    try {
      const actions = new ActionRepository({ db, clock });
      actions.add(
        makeAction({
          state: "considering",
          considering_at: 1_000,
        }),
      );
      actions.add(
        makeAction({
          description: "Send the metrics report",
          state: "completed",
          created_at: 1_100,
          updated_at: 1_100,
          completed_at: 1_100,
        }),
      );
      actions.add(
        makeAction({
          description: "Close the sprint notes",
          state: "completed",
          created_at: 1_200,
          updated_at: 1_200,
          completed_at: 1_200,
          canonicalized_by_artifact_entry_id: createSharedStateEntryId(),
        }),
      );

      const commitments = new CommitmentRepository({ db, clock });
      const activeCommitment = commitments.add({
        type: "promise",
        directiveFamily: "metrics active one",
        directive: "Keep the metrics visible.",
        priority: 5,
        provenance: { kind: "manual" },
      });
      commitments.add({
        type: "rule",
        kind: "audience_rule",
        directiveFamily: "metrics active two",
        directive: "Prefer count-only reads.",
        priority: 4,
        provenance: { kind: "manual" },
      });
      const supersededCommitment = commitments.add({
        type: "preference",
        directiveFamily: "metrics superseded",
        directive: "Use the older metrics wording.",
        priority: 3,
        provenance: { kind: "manual" },
      });
      const revokedCommitment = commitments.add({
        type: "promise",
        directiveFamily: "metrics revoked",
        directive: "Retire the old metric commitment.",
        priority: 3,
        provenance: { kind: "manual" },
      });
      commitments.add({
        type: "promise",
        directiveFamily: "metrics expired",
        directive: "Expire the old metric commitment.",
        priority: 3,
        provenance: { kind: "manual" },
        createdAt: 500,
        expiresAt: 900,
      });
      const canonicalizedCommitment = commitments.add({
        type: "promise",
        directiveFamily: "metrics canonicalized",
        directive: "Canonicalize the old metric commitment.",
        priority: 3,
        provenance: { kind: "manual" },
      });
      commitments.supersede(supersededCommitment.id, activeCommitment.id);
      commitments.revoke(revokedCommitment.id, "metrics test revocation", { kind: "manual" });
      commitments.revoke(
        canonicalizedCommitment.id,
        "metrics test canonicalization",
        { kind: "manual" },
        undefined,
        {
          canonicalizedByArtifactEntryId: createSharedStateEntryId(),
        },
      );

      const workingMemoryStore = new WorkingMemoryStore({ dataDir: dir, clock });
      const embeddingClient = new SameVectorEmbeddingClient();
      await workingMemoryStore.addPendingAction({
        sessionId,
        action: {
          description: "Follow up on metric output",
          next_action: "inspect the metrics JSONL row",
        },
        embeddingClient,
      });
      await workingMemoryStore.addPendingAction({
        sessionId,
        action: {
          description: "Check the simulator metrics artifact",
          next_action: "review the metrics JSONL output",
        },
        embeddingClient,
      });

      const relationalSlots = new RelationalSlotRepository({ db, clock });
      relationalSlots.applyAssertion({
        subject_entity_id: createEntityId(),
        slot_key: "partner.name",
        asserted_value: "Ari",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      const contestedSubject = createEntityId();
      relationalSlots.applyAssertion({
        subject_entity_id: contestedSubject,
        slot_key: "partner.name",
        asserted_value: "Bo",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      relationalSlots.applyAssertion({
        subject_entity_id: contestedSubject,
        slot_key: "partner.name",
        asserted_value: "Cam",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      const quarantinedSubject = createEntityId();
      relationalSlots.applyAssertion({
        subject_entity_id: quarantinedSubject,
        slot_key: "partner.name",
        asserted_value: "Dee",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      relationalSlots.applyAssertion({
        subject_entity_id: quarantinedSubject,
        slot_key: "partner.name",
        asserted_value: "Eli",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      relationalSlots.applyAssertion({
        subject_entity_id: quarantinedSubject,
        slot_key: "partner.name",
        asserted_value: "Finn",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      const revokedSlot = relationalSlots.applyAssertion({
        subject_entity_id: createEntityId(),
        slot_key: "partner.name",
        asserted_value: "Grey",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      relationalSlots.setState(revokedSlot.slot.id, "revoked");

      const reviewQueue = new ReviewQueueRepository({ db, clock });
      reviewQueue.enqueue({
        kind: "new_insight",
        refs: {},
        reason: "First metrics fixture.",
      });
      reviewQueue.enqueue({
        kind: "new_insight",
        refs: {},
        reason: "Second metrics fixture.",
      });
      reviewQueue.enqueue({
        kind: "contradiction",
        refs: {},
        reason: "Contradiction metrics fixture.",
      });

      const identityEvents = new IdentityEventRepository({ db, clock });
      identityEvents.record({
        record_type: "open_question",
        record_id: "open_question_metrics_1",
        action: "resolve",
        old_value: {
          status: OPEN_QUESTION_OPEN_STATUS,
        },
        new_value: {
          status: OPEN_QUESTION_RESOLVED_STATUS,
        },
        provenance: { kind: "manual" },
      });

      const borg = {
        ...fakeBorg(),
        actions,
        commitments: {
          list: (options = {}) => commitments.list(options),
          countActive: () => commitments.countActive(),
          countActiveByKind: () => commitments.countActiveByKind(),
          countActiveByEnforcementClass: () => commitments.countActiveByEnforcementClass(),
          countSuperseded: () => commitments.countSuperseded(),
          countRevoked: () => commitments.countRevoked(),
          countExpired: () => commitments.countExpired(),
          countCanonicalized: () => commitments.countCanonicalized(),
        },
        relationalSlots: {
          countByState: () => relationalSlots.countByState(),
        },
        review: {
          list: (options = {}) => reviewQueue.list(options),
        },
        identity: {
          listEvents: (...args: Parameters<IdentityEventRepository["list"]>) =>
            identityEvents.list(...args),
        },
        workmem: {
          load: (id = sessionId) => workingMemoryStore.load(id),
          getPendingActionMergeCount: () => workingMemoryStore.getPendingActionMergeCount(),
        },
      } as unknown as Borg;
      const row = await new MetricsCapture(metricsPath).capture(borg, "turn-memory-bands", 1, {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      });

      expect(row.action_record_count_total).toBe(3);
      expect(row.action_record_count_by_state).toEqual({
        ...zeroCounts(ACTION_STATES),
        considering: 1,
        completed: 2,
      });
      expect(row.action_record_count_committed_to_do).toBe(0);
      expect(row.action_record_count_canonicalized).toBe(1);
      expect(row.action_record_count_active).toBe(1);
      expect(row.borg_owned_active_actions).toBe(1);
      expect(row.participant_owned_active_actions).toBe(0);
      expect(row.group_owned_active_actions).toBe(0);
      expect(row.action_record_creation_source_per_turn).toEqual({
        extractor: 0,
        reflector: 0,
        api: 0,
        tool: 0,
        unknown: 3,
      });
      expect(row.action_record_creation_count_this_turn).toBe(3);
      expect(row.recent_completed_action_count).toBe(2);
      expect(row.commitment_count_active).toBe(2);
      expect(row.commitment_count_active_by_kind).toEqual({
        ...zeroCounts(COMMITMENT_KINDS),
        assistant_commitment: 1,
        audience_rule: 1,
      });
      expect(row.commitments_by_enforcement_class).toEqual({
        ...zeroCounts(COMMITMENT_ENFORCEMENT_CLASSES),
        advisory: 1,
        critical: 1,
      });
      const criticalCommitmentBreakdown = zeroCriticalCommitmentsByKindTypeDomain();
      criticalCommitmentBreakdown.audience_rule.rule.audience_scope = 1;
      expect(row.critical_commitments_by_kind_type_domain).toEqual(criticalCommitmentBreakdown);
      expect(row.commitments_advisory_count).toBe(1);
      expect(row.commitments_critical_count).toBe(1);
      expect(row.commitments_critical_classification_downgraded_total).toBe(0);
      expect(row.commitments_critical_classification_downgraded_by_reason).toEqual(
        zeroCounts(CLASSIFICATION_DOWNGRADE_REASONS),
      );
      expect(row.commitments_critical_classification_downgraded_by_kind_type_from_domain).toEqual(
        {},
      );
      expect(row.commitment_count_superseded).toBe(1);
      expect(row.commitment_count_revoked).toBe(2);
      expect(row.commitment_count_expired).toBe(1);
      expect(row.commitment_count_canonicalized).toBe(1);
      expect(row.pending_action_count).toBe(1);
      expect(row.pending_action_merge_count).toBe(1);
      expect(row.relational_slot_count_by_state).toEqual({
        ...zeroCounts(RELATIONAL_SLOT_STATES),
        established: 1,
        contested: 1,
        quarantined: 1,
        revoked: 1,
      });
      expect(row.review_queue_open_count_by_type).toEqual({
        ...zeroCounts(REVIEW_KINDS),
        contradiction: 1,
        new_insight: 2,
      });
      expect(row.open_question_resolved_count).toBe(1);
    } finally {
      db.close();
    }
  });

  it("counts commitment classification downgrade trace events", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-downgrade-1",
          event: "commitment_classification.downgraded",
          original_enforcement_class: "critical",
          original_critical_domain: "internal_tool_hygiene",
          new_enforcement_class: "advisory",
          new_critical_domain: null,
          reason: "preference_with_internal_tool_hygiene",
          kind: "participant_preference",
          type: "preference",
          directive_family: "surface_durable_decisions",
        },
        {
          ts: 101,
          turnId: "turn-downgrade-2",
          event: "commitment_classification.downgraded",
          original_enforcement_class: "critical",
          original_critical_domain: "explicit_no_disclosure",
          new_enforcement_class: "advisory",
          new_critical_domain: null,
          reason: "explicit_no_disclosure_without_boundary_type",
          kind: "participant_preference",
          type: "preference",
          directive_family: "owner_decides_wording",
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-downgrade-2",
      2,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row.commitments_critical_classification_downgraded_total).toBe(2);
    expect(row.commitments_critical_classification_downgraded_by_reason).toEqual({
      ...zeroCounts(CLASSIFICATION_DOWNGRADE_REASONS),
      preference_with_internal_tool_hygiene: 1,
      explicit_no_disclosure_without_boundary_type: 1,
    });
    expect(row.commitments_critical_classification_downgraded_by_kind_type_from_domain).toEqual({
      "participant_preference/preference/explicit_no_disclosure": 1,
      "participant_preference/preference/internal_tool_hygiene": 1,
    });
  });

  it("emits checkpoint duplicate-pressure traces without merging action records", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const first = makeAction({
      description: "Review Atlas rollout",
      state: "committed_to_do",
    });
    const second = makeAction({
      description: "Check Atlas deployment",
      state: "scheduled",
    });
    const third = makeAction({
      description: "Draft billing follow-up",
      state: "considering",
    });
    const records = [first, second, third];
    const actions = {
      count: () => records.length,
      countByState: () => ({
        ...zeroCounts(ACTION_STATES),
        considering: 1,
        committed_to_do: 1,
        scheduled: 1,
      }),
      countCanonicalized: () => 0,
      countActive: () => records.length,
      getCreationCountsBySource: () => ({
        extractor: 0,
        reflector: 0,
        api: 0,
        tool: 0,
        unknown: 0,
      }),
      countCompletedSince: () => 0,
      latestCompletedAt: () => null,
      listCompletedIds: () => [],
      list: () => records,
      findSimilarDescriptionPairs: async () => [
        {
          leftId: first.id,
          rightId: second.id,
          similarity: 0.9,
        },
      ],
    };
    const borg = {
      ...fakeBorg(),
      actions,
    } as unknown as Borg;

    await new MetricsCapture(metricsPath, { tracePath }).capture(borg, "turn-duplicate", 10, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    const trace = readFileSync(tracePath, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);

    expect(records).toHaveLength(3);
    expect(trace).toContainEqual(
      expect.objectContaining({
        event: "action_duplicate_pressure.completed",
        turnId: "turn-duplicate",
        cluster_count: 1,
        max_cluster_size: 2,
        total_actions_in_clusters: 2,
        threshold_used: 0.85,
      }),
    );
  });

  it("captures aborted turns with a failure reason", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const capture = new MetricsCapture(metricsPath);
    const sessionId = createSessionId();
    const failureReason = "transport failed";

    const row = await capture.captureAborted(fakeBorg(), 4, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 3,
      failureReason,
    });

    expect(row.event).toBe("aborted_turn");
    expect(row.turn_counter).toBe(4);
    expect(row.transport_chat_attempts).toBe(3);
    expect(row.failure_reason).toBe(failureReason);
  });

  it("excludes aborted suppressions from generation_suppression_count", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const capture = new MetricsCapture(metricsPath);
    const sessionId = createSessionId();
    const activeTurnId = "turn-active-suppression";
    const abortedTurnId = "turn-aborted-suppression";
    const streamEntriesBySession = new Map<SessionId, StreamEntry[]>([
      [
        sessionId,
        [
          {
            id: createStreamEntryId(),
            timestamp: 1,
            kind: "agent_suppressed",
            content: { reason: "generation_gate" },
            turn_id: activeTurnId,
            session_id: sessionId,
            compressed: false,
            sender_entity_id: null,
            reply_target_entity_id: null,
          },
          {
            id: createStreamEntryId(),
            timestamp: 2,
            kind: "agent_suppressed",
            content: { reason: "generation_gate" },
            turn_id: abortedTurnId,
            session_id: sessionId,
            compressed: false,
            sender_entity_id: null,
            reply_target_entity_id: null,
          },
          {
            id: createStreamEntryId(),
            timestamp: 3,
            kind: "internal_event",
            content: {
              event: ABORTED_TURN_EVENT,
              turn_id: abortedTurnId,
              reason: "turn failed",
            },
            turn_id: abortedTurnId,
            turn_status: "aborted",
            session_id: sessionId,
            compressed: false,
            sender_entity_id: null,
            reply_target_entity_id: null,
          },
        ],
      ],
    ]);
    const borg = fakeBorg({ streamEntriesBySession });

    const completed = await capture.capture(borg, activeTurnId, 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    const aborted = await capture.captureAborted(borg, 2, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
      failureReason: "turn failed",
      turnId: abortedTurnId,
    });

    expect(completed.generation_suppression_count).toBe(1);
    expect(aborted.generation_suppression_count).toBe(1);
  });
});
