import { appendFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ACTION_CANDIDATE_CLASSIFICATIONS,
  ACTION_STATES,
  Borg,
  COMMITMENT_ENFORCEMENT_CLASSES,
  COMMITMENT_KINDS,
  RELATIONAL_SLOT_STATES,
  type EntityId,
  type GenerationSuppressionReason,
  type SessionId,
} from "../src/index.js";
import { createEntityId, createSessionId } from "../src/util/ids.js";
import { FakeLLMClient } from "../src/llm/test-support/fake-client.js";
import {
  MaintenanceScheduler,
  type MaintenanceCadence,
  type MaintenanceTickResult,
} from "../src/offline/scheduler.js";
import { BorgTransport, type ChatWithBorgResult } from "../assessor/borg-transport.js";
import { readTraceEvents } from "../assessor/trace-reader.js";
import { createSimulatorScenario, formatSimulatorReport, runSimulation } from "./runner.js";
import { capabilityFindingMetrics, simulatorHealthWarningsForRows } from "./health-warnings.js";
import {
  validateOverseerVerdict,
  type FindingCarryoverCache,
  type OverseerAuditContext,
} from "./overseer.js";
import type { PersonaSession, PriorBorgTurn } from "./persona.js";
import { tomPersona } from "./personas/tom.js";
import type {
  MetricsRow,
  OverseerVerdict,
  RawOverseerVerdict,
  SimulatorHealthWarningKind,
} from "./types.js";

const tempDirs: string[] = [];

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "borg-simulator-runner-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  vi.restoreAllMocks();

  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

function spyMaintenanceTick() {
  return vi
    .spyOn(MaintenanceScheduler.prototype, "tick")
    .mockImplementation(async (cadence): Promise<MaintenanceTickResult> => {
      return {
        status: "ok",
        cadence,
        ts: Date.now(),
        processes: [],
        result: null,
      };
    });
}

function zeroCounts<K extends string>(keys: readonly K[]): Record<K, number> {
  return Object.fromEntries(keys.map((key) => [key, 0])) as Record<K, number>;
}

function fakeSimulatorBorg(): Borg {
  return {
    mood: {
      current: () => ({ valence: 0, arousal: 0 }),
    },
    episodic: {
      list: async () => ({ items: [] }),
    },
    semantic: {
      nodes: {
        list: async () => [],
      },
      edges: {
        list: () => [],
      },
    },
    self: {
      values: {
        list: () => [],
      },
      openQuestions: {
        list: () => [],
      },
      goals: {
        list: () => [],
      },
      traits: {
        list: () => [],
      },
      autobiographical: {
        currentPeriod: () => null,
        listPeriods: () => [],
      },
      growthMarkers: {
        list: () => [],
      },
    },
    actions: {
      count: () => 0,
      countByState: () => zeroCounts(ACTION_STATES),
      countCompletedSince: () => 0,
      latestCompletedAt: () => null,
      listCompletedIds: () => [],
      list: () => [],
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
    entities: {
      resolve: () => createEntityId(),
    },
    relationalSlots: {
      countByState: () => zeroCounts(RELATIONAL_SLOT_STATES),
    },
    identity: {
      listEvents: () => [],
    },
    skills: {
      list: () => [],
    },
    workmem: {
      load: () => ({ pending_actions: [] }),
      getPendingActionMergeCount: () => 0,
    },
    stream: {
      tail: () => [],
    },
    maintenance: {
      scheduler: {
        tick: async (cadence: string) => ({
          status: "ok",
          cadence,
          ts: Date.now(),
          processes: [],
          result: null,
        }),
      },
    },
    review: {
      list: () => [],
    },
    audit: {
      list: () => [],
    },
    close: async () => undefined,
  } as unknown as Borg;
}

function fakePersonaSession(messages: readonly string[]): {
  session: PersonaSession;
  prepareNextTurn: ReturnType<typeof vi.fn>;
  commit: ReturnType<typeof vi.fn>;
  rollback: ReturnType<typeof vi.fn>;
  startNewSession: ReturnType<typeof vi.fn>;
} {
  let index = 0;
  const prepareNextTurn = vi.fn(async (priorBorgTurn: PriorBorgTurn) => {
    const messageIndex = priorBorgTurn.retry === "persona_role_bleed" ? index + 1 : index;
    const message = messages[messageIndex] ?? messages.at(-1) ?? "persona turn";
    return {
      kind: "mock",
      message,
      history: null,
      mockIndex: index,
    };
  });
  const commit = vi.fn(() => {
    index += 1;
  });
  const rollback = vi.fn();
  const startNewSession = vi.fn();

  return {
    session: {
      prepareNextTurn,
      commit,
      rollback,
      startNewSession,
    } as unknown as PersonaSession,
    prepareNextTurn,
    commit,
    rollback,
    startNewSession,
  };
}

function mockTransportLifecycle() {
  const entityIds = new Map<string, EntityId>();
  vi.spyOn(BorgTransport.prototype, "open").mockResolvedValue(undefined);
  vi.spyOn(BorgTransport.prototype, "close").mockResolvedValue(undefined);
  vi.spyOn(BorgTransport.prototype, "getBorg").mockReturnValue(fakeSimulatorBorg());
  const resolveEntitySpy = vi
    .spyOn(BorgTransport.prototype, "resolveEntity")
    .mockImplementation((name) => {
      const existing = entityIds.get(name);

      if (existing !== undefined) {
        return existing;
      }

      const entityId = createEntityId();
      entityIds.set(name, entityId);
      return entityId;
    });

  return { entityIds, resolveEntitySpy };
}

function chatResult(input: {
  response: string;
  emitted: boolean;
  turnId: string;
  sessionId: SessionId;
  suppressionReason?: GenerationSuppressionReason;
  observedReason?: string;
}): ChatWithBorgResult {
  return {
    response: input.response,
    emitted: input.emitted,
    emission:
      input.observedReason !== undefined
        ? {
            kind: "observed",
            reason: input.observedReason,
          }
        : input.emitted
          ? ({
              kind: "message",
              content: input.response,
              agentMessageId: `strm_${input.turnId}`,
            } as ChatWithBorgResult["emission"])
          : {
              kind: "suppressed",
              reason: input.suppressionReason ?? "finalizer_no_output",
            },
    turnId: input.turnId,
    sessionId: input.sessionId,
    usage: {
      input_tokens: 0,
      output_tokens: 0,
    },
    moodAfter: {
      valence: 0,
      arousal: 0,
    },
    toolCalls: [],
  };
}

function healthyOverseerVerdict(
  turnCounter: number,
  observations: string[],
  recommendation = "Continue.",
): OverseerVerdict {
  const raw_verdict = {
    status: "healthy" as const,
    observations,
    recommendation,
    findings: [],
  };

  return {
    ts: Date.now(),
    turn_counter: turnCounter,
    ...raw_verdict,
    rejected_findings: [],
    raw_verdict,
  };
}

function emptyOverseerAuditContext(turnCounter: number): OverseerAuditContext {
  return {
    window: {
      from_turn: Math.max(1, turnCounter - 9),
      to_turn: turnCounter,
    },
    chronology_rule: "Stream ts is authoritative for tests.",
    assistant_emitted: [],
    user_messages: [],
    recent_user_statements: [],
    prompt_visible_memory: {
      summary: "Test memory.",
      note: "Test prompt-visible memory.",
    },
    snapshot_state: {
      markdown: "Test memory.",
      note: "Test snapshot state.",
    },
    metrics_window: [],
  };
}

function validateRunnerOverseerVerdict(input: {
  turnCounter: number;
  rawVerdict: RawOverseerVerdict;
  carryoverCache: FindingCarryoverCache;
}): OverseerVerdict {
  const validated = validateOverseerVerdict(
    input.rawVerdict,
    emptyOverseerAuditContext(input.turnCounter),
    input.carryoverCache,
  );

  return {
    ts: Date.now(),
    turn_counter: input.turnCounter,
    status: validated.status,
    observations: input.rawVerdict.observations,
    recommendation: input.rawVerdict.recommendation,
    findings: validated.findings,
    rejected_findings: validated.rejected_findings,
    raw_verdict: input.rawVerdict,
  };
}

function metricsRow(turnCounter: number): MetricsRow {
  return {
    event: "turn_metrics",
    ts: turnCounter,
    turn_counter: turnCounter,
    turnId: `turn-${turnCounter}`,
    transport_chat_attempts: 1,
    episode_count: 0,
    semantic_node_count: 0,
    semantic_node_count_by_status: {
      active: 0,
      superseded: 0,
      contradicted: 0,
      quarantined: 0,
    },
    semantic_edge_count: 0,
    semantic_nodes_added_since_last_check: 0,
    semantic_edges_added_since_last_check: 0,
    semantic_nodes_rejected_ungrounded_claim_count: 0,
    semantic_nodes_rejected_ungrounded_claim_total: 0,
    semantic_nodes_rejected_ungrounded_claim_by_label_family: {},
    shared_state_operations_rejected_ungrounded_claim_total: 0,
    shared_state_operations_rejected_ungrounded_claim_by_label_family: {},
    commitment_candidates_rejected_ungrounded_claim_total: 0,
    commitment_candidates_rejected_ungrounded_claim_by_label_family: {},
    open_question_count: 0,
    active_goal_count: 0,
    generation_suppression_count: 0,
    mood_valence: 0,
    mood_arousal: 0,
    retrieval_latency_ms: null,
    deliberation_latency_ms: null,
    borg_input_tokens: 0,
    borg_output_tokens: 0,
    open_question_resolved_count: 0,
    open_questions_by_source: {},
    open_questions_by_status_age: {},
    open_questions_resolved_this_run: 0,
    open_questions_rendered_to_finalizer_this_turn: 0,
    open_questions_promoted_from_review_items: 0,
    action_record_count_total: 0,
    action_record_count_by_state: zeroCounts(ACTION_STATES),
    action_record_count_committed_to_do: 0,
    action_record_count_canonicalized: 0,
    action_record_count_active: 0,
    borg_owned_active_actions: 0,
    participant_owned_active_actions: 0,
    group_owned_active_actions: 0,
    prompt_salient_actions_total: 0,
    borg_owned_salient_active_actions: 0,
    participant_owned_salient_active_actions: 0,
    dormant_actions_total: 0,
    dormant_not_archive_eligible_count: 0,
    dormant_archive_eligible_count: 0,
    archive_oldest_inactive_turns: 0,
    archive_inactive_turn_distribution: {
      "0-15": 0,
      "15-20": 0,
      "20-30": 0,
      "30+": 0,
    },
    archive_archivable_count: 0,
    archive_skipped_borg_owned: 0,
    archive_skipped_due_date: 0,
    archive_skipped_below_threshold: 0,
    archive_skipped_other: 0,
    archive_oldest_archivable_inactive_turns: 0,
    stale_actions_omitted_from_prompt: 0,
    actions_per_turn: 0,
    salient_actions_per_turn: 0,
    action_retirement_ratio: 0,
    borg_owned_action_count: 0,
    stale_action_count: 0,
    action_record_creation_source_per_turn: {
      extractor: 0,
      reflector: 0,
      api: 0,
      unknown: 0,
    },
    action_record_creation_count_this_turn: 0,
    action_candidate_classifications_per_turn: {
      ...zeroCounts(ACTION_CANDIDATE_CLASSIFICATIONS),
      invalid_classification: 0,
    },
    action_candidate_rejected_classification: 0,
    action_persistence_dedup_skipped_embedding: 0,
    action_persistence_dedup_degraded: 0,
    actions_closed_by_terminal_emission: 0,
    actions_closed_by_borg_self_performance: 0,
    actions_expired_at_session_close: 0,
    actions_rejected_capability: 0,
    actions_canonicalized: 0,
    actions_completed_via_canonicalization: 0,
    actions_dormant_count: 0,
    actions_archived_count: 0,
    recent_completed_action_count: 0,
    commitment_count_active: 0,
    commitment_count_active_by_kind: {
      assistant_commitment: 0,
      audience_rule: 0,
      participant_preference: 0,
      boundary: 0,
      process_norm: 0,
    },
    commitments_by_enforcement_class: zeroCounts(COMMITMENT_ENFORCEMENT_CLASSES),
    critical_commitments_by_kind_type_domain:
      {} as MetricsRow["critical_commitments_by_kind_type_domain"],
    commitments_advisory_count: 0,
    commitments_critical_count: 0,
    commitments_critical_classification_downgraded_total: 0,
    commitments_critical_classification_downgraded_by_reason:
      {} as MetricsRow["commitments_critical_classification_downgraded_by_reason"],
    commitments_critical_classification_downgraded_by_kind_type_from_domain: {},
    commitment_count_superseded: 0,
    commitment_count_revoked: 0,
    commitment_count_expired: 0,
    commitment_count_canonicalized: 0,
    commitment_regeneration_attempted_count: 0,
    commitment_regeneration_succeeded_count: 0,
    commitment_regeneration_failed_count: 0,
    commitment_regeneration_attempted_total: 0,
    commitment_regeneration_succeeded_total: 0,
    commitment_regeneration_failed_total: 0,
    commitment_guard_advisory_violations_total: 0,
    commitment_guard_advisory_violations_by_class: zeroCounts(COMMITMENT_ENFORCEMENT_CLASSES),
    pending_action_count: 0,
    pending_action_merge_count: 0,
    relational_slot_count_by_state: zeroCounts(RELATIONAL_SLOT_STATES),
    review_queue_open_count_by_type: {
      contradiction: 0,
      duplicate: 0,
      new_insight: 0,
      misattribution: 0,
      temporal_drift: 0,
      identity_inconsistency: 0,
      correction: 0,
      belief_revision: 0,
      skill_split: 0,
      creator_directive_reconciliation: 0,
      commitment_reconciliation: 0,
    },
    frame_anomaly_classifier_calls: 0,
    frame_anomaly_classified_normal_count: 0,
    frame_anomaly_actual_anomaly_count: 0,
    frame_anomaly_degraded_count: 0,
    frame_anomaly_degraded_fallback_match_count: 0,
    quarantined_user_entry_count: 0,
    early_extractors_skipped_frame_anomaly_count: 0,
    goal_promotion_salvaged_promotions: 0,
    goal_promotion_skipped_promotions: 0,
    goal_promotion_initial_step_downgraded: 0,
    goal_promotion_dedup_skipped_extractor_signal: 0,
    goal_promotion_dedup_skipped_embedding: 0,
    goal_promotion_dedup_degraded: 0,
    goal_promotion_classifications_per_turn: {
      durable_borg_goal: 0,
      one_off: 0,
      not_borg_responsibility: 0,
      impossible_for_borg_without_capability: 0,
      already_represented: 0,
      none: 0,
      invalid_classification: 0,
    },
    goal_promotion_rejected_classification: 0,
    goal_promotion_cap_rejections: 0,
    shared_state_semantic_revisions_attempted: 0,
    shared_state_semantic_revisions_completed_succeeded: 0,
    shared_state_semantic_nodes_marked_superseded: 0,
    shared_state_semantic_nodes_marked_contradicted: 0,
    shared_state_semantic_revision_cache_hits: 0,
    shared_state_semantic_revision_cache_size: 0,
    embedding_cache_pending_overflow_total: 0,
    ledger_reverse_scan_entries_total: 0,
    ledger_reverse_scan_bytes_total: 0,
    ledger_reverse_scan_entry_cap_hit_total: 0,
    ledger_reverse_scan_byte_cap_hit_total: 0,
    ledger_image_refs_considered_total: 0,
    ledger_image_refs_attached_total: 0,
    ledger_image_refs_omitted_budget_total: 0,
    ledger_image_bytes_attached_total: 0,
    ledger_image_refs_omitted_inactive_total: 0,
    semantic_revision_error_count: 0,
    semantic_revision_skipped_due_to_error: 0,
    semantic_revision_error_total_by_reason: {},
    semantic_revision_calls_total: 0,
    semantic_revision_candidates_reviewed_total: 0,
    semantic_revision_superseded_total: 0,
    semantic_revision_contradicted_total: 0,
    semantic_revision_degraded_total: 0,
    semantic_revision_skipped_over_cap_total: 0,
    overseer_due_on_suppressed_turn: false,
    closure_loop_completed_count: 0,
    closure_loop_degraded_count: 0,
    closure_response_audit_failed_open_total: 0,
    closure_pressure_mixed_observed_total: 0,
    closure_pressure_closure_only_observed_total: 0,
    closure_pressure_closure_only_suppressed_total: 0,
    closure_pressure_mixed_passed_no_active_preference_total: 0,
    closure_pressure_mixed_by_span_kind: {},
    corrective_preference_completed_count: 0,
    corrective_preference_degraded_count: 0,
    extractor_max_tokens_stop_count: 0,
    extractor_max_tokens_total_by_label: {},
    extractor_degraded_total_by_label: {},
    shared_state_compiler_max_tokens_total: 0,
    shared_state_compiler_degraded_total: 0,
    shared_state_compiler_repair_attempted_total: 0,
    shared_state_compiler_repair_succeeded_total: 0,
    shared_state_compiler_repair_failed_total: 0,
    shared_state_compiler_repair_failed_by_rejection_reason: {},
    shared_state_update_checked_for_empty_total: 0,
    shared_state_empty_update_attempted_total: 0,
    shared_state_empty_update_dropped_total: 0,
    shared_state_empty_update_drop_rate: 0,
    shared_state_empty_update_repaired_total: 0,
    capability_overclaim_count: 0,
    capability_ambiguity_count: 0,
    capability_boundary_refusal_count: 0,
    shared_state_at_cap_turns: 0,
    shared_state_compile_evaluated_turns: 0,
    shared_state_omitted_recent_entries: 0,
    shared_state_omitted_recent_entries_total_across_compiles: 0,
    shared_state_omitted_live_recent_operational: 0,
    shared_state_omitted_live_recent_operational_total_across_compiles: 0,
    shared_state_omitted_live_recent_operational_final_compile: 0,
    shared_state_omitted_live_recent_low_salience: 0,
    shared_state_omitted_live_recent_low_salience_total_across_compiles: 0,
    shared_state_omitted_live_recent_low_salience_final_compile: 0,
    shared_state_omitted_live_old: 0,
    shared_state_omitted_live_old_total_across_compiles: 0,
    shared_state_omitted_live_old_final_compile: 0,
    shared_state_omitted_live_unknown_age: 0,
    shared_state_omitted_live_unknown_age_total_across_compiles: 0,
    shared_state_omitted_live_unknown_age_final_compile: 0,
    shared_state_omitted_locked: 0,
    shared_state_omitted_locked_total_across_compiles: 0,
    shared_state_omitted_locked_final_compile: 0,
    shared_state_omitted_locked_recent_total_across_compiles: 0,
    shared_state_omitted_locked_recent_final_compile: 0,
    shared_state_omitted_locked_old_total_across_compiles: 0,
    shared_state_omitted_locked_old_final_compile: 0,
    shared_state_omitted_locked_unknown_age_total_across_compiles: 0,
    shared_state_omitted_locked_unknown_age_final_compile: 0,
    shared_state_omitted_locked_with_active_critical_commitment_total_across_compiles: 0,
    shared_state_omitted_locked_with_active_critical_commitment_final_compile: 0,
    shared_state_omitted_locked_with_operational_canonicalizer_total_across_compiles: 0,
    shared_state_omitted_locked_with_operational_canonicalizer_final_compile: 0,
    shared_state_omitted_locked_indexed_only_total_across_compiles: 0,
    shared_state_omitted_locked_indexed_only_final_compile: 0,
    shared_state_omitted_pending: 0,
    shared_state_omitted_pending_total_across_compiles: 0,
    shared_state_omitted_pending_final_compile: 0,
    shared_state_omitted_low_salience_live: 0,
    shared_state_omitted_low_salience_live_final_compile: 0,
    shared_state_omitted_dormant_live: 0,
    shared_state_omitted_dormant_live_final_compile: 0,
    shared_state_active_low_salience_live: 0,
    shared_state_active_low_salience_live_final_compile: 0,
    shared_state_active_dormant_live: 0,
    shared_state_active_dormant_live_final_compile: 0,
    shared_state_demoted_live_to_low_salience_total: 0,
    shared_state_demoted_low_salience_to_dormant_total: 0,
    shared_state_lifecycle_aging_demotable_total: 0,
    shared_state_lifecycle_aging_demotable_final_compile: 0,
    shared_state_lifecycle_aging_demoted_total: 0,
    shared_state_lifecycle_aging_demoted_final_compile: 0,
    shared_state_lifecycle_aging_blocked_by_current_turn_update_total: 0,
    shared_state_lifecycle_aging_blocked_by_current_turn_update_final_compile: 0,
    shared_state_lifecycle_aging_blocked_by_patch_touch_total: 0,
    shared_state_lifecycle_aging_blocked_by_patch_touch_final_compile: 0,
    shared_state_lifecycle_aging_blocked_by_ledger_overlap_total: 0,
    shared_state_lifecycle_aging_blocked_by_ledger_overlap_final_compile: 0,
    shared_state_lifecycle_aging_blocked_by_recent_retrieval_total: 0,
    shared_state_lifecycle_aging_blocked_by_recent_retrieval_final_compile: 0,
    shared_state_lifecycle_aging_blocked_by_active_canonicalizer_critical_total: 0,
    shared_state_lifecycle_aging_blocked_by_active_canonicalizer_critical_final_compile: 0,
    shared_state_lifecycle_aging_blocked_by_active_canonicalizer_operational_total: 0,
    shared_state_lifecycle_aging_blocked_by_active_canonicalizer_operational_final_compile: 0,
    shared_state_lifecycle_aging_blocked_by_hard_total: 0,
    shared_state_lifecycle_aging_blocked_by_hard_final_compile: 0,
    shared_state_lifecycle_aging_blocked_by_soft_total: 0,
    shared_state_lifecycle_aging_blocked_by_soft_final_compile: 0,
    shared_state_lifecycle_aging_unknown_age_total: 0,
    shared_state_lifecycle_aging_unknown_age_final_compile: 0,
    shared_state_lifecycle_aging_blocked_by_multiple_reasons_total: 0,
    shared_state_lifecycle_aging_blocked_by_multiple_reasons_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_demotable_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_demotable_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_demoted_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_demoted_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_current_turn_update_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_current_turn_update_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_patch_touch_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_patch_touch_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_ledger_overlap_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_ledger_overlap_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_recent_retrieval_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_recent_retrieval_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_critical_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_critical_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_operational_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_active_canonicalizer_operational_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_hard_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_hard_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_soft_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_soft_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_unknown_age_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_unknown_age_final_compile: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_multiple_reasons_total: 0,
    shared_state_lifecycle_aging_low_salience_to_dormant_blocked_by_multiple_reasons_final_compile: 0,
    shared_state_reactivated_low_salience_live_total: 0,
    shared_state_reactivated_dormant_live_total: 0,
    shared_state_at_cap_but_all_keys_indexed_compiles_total: 0,
    shared_state_at_cap_with_operational_omission_compiles_total: 0,
    shared_state_at_cap_with_cap_rejection_compiles_total: 0,
    shared_state_all_active_keys_indexed: true,
    shared_state_live_entry_starvation: false,
    shared_state_newest_entries_reserved: 0,
    shared_state_live_starvation_with_reserved: false,
    shared_state_live_starvation_ever: false,
    shared_state_live_starvation_final: false,
    shared_state_compiler_operations_total_by_kind: {
      add: 0,
      update: 0,
      supersede: 0,
      prune: 0,
    },
    shared_state_add_to_update_ratio: 0,
    shared_state_entries_by_key: {},
    shared_state_add_to_update_ratio_by_key: {},
    shared_state_top_keys_by_entry_count: {},
    shared_state_add_rejected_cap_exceeded_total: 0,
    shared_state_new_keys_per_compile: {},
    shared_state_new_keys_per_turn: 0,
    shared_state_keys_with_single_entry_only: 0,
    shared_state_similar_key_cluster_count: 0,
    shared_state_add_rejected_near_duplicate_state_key_total: 0,
    shared_state_add_rejected_missing_new_key_reason_total: 0,
    session_reentry_card_rendered_total: 0,
    session_reentry_card_rendered_by_audience: {},
    session_reentry_first_turn_with_existing_state_total: 0,
    session_reentry_first_turn_blank_audience_total: 0,
    simulator_persona_failures: 0,
    borg_hard_aborted_turns: 0,
    borg_intentional_suppressions: 0,
    borg_intentional_suppressions_by_reason: {},
    finalizer_no_output_by_category: {},
    finalizer_no_output_primary_by_reason: {},
    finalizer_no_output_flags_by_flag: {},
    finalizer_no_output_flags_by_primary_reason: {},
    finalizer_no_output_when_borg_addressed_with_state_delta_total: 0,
    finalizer_no_output_closure_with_open_question_total: 0,
    borg_aborted_turns: 0,
  };
}

function legacyOnlySemanticRevisionMetrics(row: MetricsRow): MetricsRow {
  const legacyRow: Record<string, unknown> = { ...row };

  legacyRow.decision_artifact_semantic_revisions_attempted =
    row.shared_state_semantic_revisions_attempted;
  legacyRow.decision_artifact_semantic_revisions_completed_succeeded =
    row.shared_state_semantic_revisions_completed_succeeded;
  legacyRow.decision_artifact_semantic_nodes_marked_superseded =
    row.shared_state_semantic_nodes_marked_superseded;
  legacyRow.decision_artifact_semantic_nodes_marked_contradicted =
    row.shared_state_semantic_nodes_marked_contradicted;
  legacyRow.decision_artifact_semantic_revision_cache_hits =
    row.shared_state_semantic_revision_cache_hits;
  legacyRow.decision_artifact_semantic_revision_cache_size =
    row.shared_state_semantic_revision_cache_size;

  delete legacyRow.shared_state_semantic_revisions_attempted;
  delete legacyRow.shared_state_semantic_revisions_completed_succeeded;
  delete legacyRow.shared_state_semantic_nodes_marked_superseded;
  delete legacyRow.shared_state_semantic_nodes_marked_contradicted;
  delete legacyRow.shared_state_semantic_revision_cache_hits;
  delete legacyRow.shared_state_semantic_revision_cache_size;

  return legacyRow as MetricsRow;
}

describe("SimulatorRunner", () => {
  it("formats carryover findings in a separate audit subsection", () => {
    const raw_verdict: RawOverseerVerdict = {
      status: "concerning",
      observations: ["Repeated prior incident."],
      recommendation: "Do not double count.",
      findings: [],
    };
    const report = formatSimulatorReport({
      runId: "sim-runner-format-carryover-test",
      persona: tomPersona.key,
      personas: [tomPersona.key],
      audience: "Tom",
      totalTurns: 20,
      resultState: "completed",
      sessions: [],
      suppressionEvents: [],
      overseerCheckpoints: [
        {
          ts: Date.now(),
          turn_counter: 20,
          status: "healthy",
          observations: raw_verdict.observations,
          recommendation: raw_verdict.recommendation,
          findings: [
            {
              category: "H",
              claim_status: "grounded",
              source_kind: "emitted_output",
              status_impact: "none",
              assistant_stream_entry_id: "strm_carryover_report",
              assistant_ts: 12,
              evidence_summary: "Prior hedged unsupported claim.",
              carryover_demoted: true,
              carryover_original_status_impact: "concerning",
              carryover_cached_status_impact: "concerning",
              carryover_cached_stream_entry_id: "strm_carryover_report",
              carryover_cached_at_turn: 10,
            },
          ],
          rejected_findings: [],
          raw_verdict,
        },
      ],
      turnFailures: [],
      finalMetrics: metricsRow(20),
      durationMs: 1,
    });

    expect(report).toContain("Carryover from earlier checkpoints");
    expect(report).toContain("original_impact=concerning");
    expect(report).toContain("carryover from turn 10");
    expect(report).not.toContain("Validated findings");
  });

  it("formats simulator health warnings in the run report", () => {
    const report = formatSimulatorReport({
      runId: "sim-runner-health-warning-test",
      persona: tomPersona.key,
      personas: [tomPersona.key],
      audience: "Tom",
      totalTurns: 30,
      resultState: "completed",
      sessions: [],
      suppressionEvents: [],
      overseerCheckpoints: [],
      healthWarnings: [
        {
          kind: "active_goals_growth_high",
          turn_counter: 30,
          turnId: "turn-30",
          threshold: 0.5,
          observed_value: 0.75,
          window_start_turn: 21,
          window_turns: 9,
        },
      ],
      turnFailures: [],
      finalMetrics: metricsRow(30),
      durationMs: 1,
    });

    expect(report).toContain("## Health Warnings");
    expect(report).toContain("active_goals_growth_high");
    expect(report).toContain("observed=0.75");
    expect(report).toContain("threshold=0.5");
  });

  it("formats simulator persona failures separately from Borg suppressions", () => {
    const report = formatSimulatorReport({
      runId: "sim-runner-failure-separation-report-test",
      persona: tomPersona.key,
      personas: [tomPersona.key],
      audience: "Tom",
      totalTurns: 3,
      resultState: "completed",
      sessions: [],
      suppressionEvents: [],
      overseerCheckpoints: [],
      healthWarnings: [],
      turnFailures: [],
      simulatorPersonaFailures: [
        {
          turn: 1,
          error: "persona_role_bleed: assistant_self_claim",
          attempts: 0,
        },
      ],
      borgBehavioralSuppressions: [
        {
          sessionIndex: 0,
          sessionId: createSessionId(),
          turn: 2,
          reason: "finalizer_no_output",
          sessionContinued: false,
        },
      ],
      finalMetrics: {
        ...metricsRow(3),
        simulator_persona_failures: 1,
        borg_intentional_suppressions: 1,
        borg_intentional_suppressions_by_reason: {
          finalizer_no_output: 1,
        },
        finalizer_no_output_by_category: {
          closure: 1,
        },
        finalizer_no_output_primary_by_reason: {
          closure: 1,
        },
      },
      durationMs: 1,
    });

    expect(report).toContain("Run completion: completed");
    expect(report).toContain("Simulator validity: partial (1 persona failure)");
    expect(report).toContain("Borg turn result: completed");
    expect(report).not.toContain("Run result:");
    expect(report).not.toContain("Result state:");
    expect(report).toContain(
      "Simulator aborts: persona failures 1, hard aborts 0, intentional suppressions 1 (by reason: finalizer_no_output=1)",
    );
    expect(report).toContain("Finalizer no-output (1 total):");
    expect(report).toContain("Primary by reason: closure=1");
    expect(report).toContain("Compatibility categories: closure=1");
    expect(report).toContain("## Simulator Persona Failures");
    expect(report).toContain("persona_role_bleed: assistant_self_claim");
    expect(report).toContain("## Borg Behavioral Suppressions");
    expect(report).toContain("finalizer_no_output; session ended");
    expect(report).not.toContain("## Borg Turn Failures");
  });

  it("formats capability and extractor health counters in final metrics", () => {
    const report = formatSimulatorReport({
      runId: "sim-runner-observability-counter-test",
      persona: tomPersona.key,
      personas: [tomPersona.key],
      audience: "Tom",
      totalTurns: 30,
      resultState: "completed",
      sessions: [],
      suppressionEvents: [],
      overseerCheckpoints: [],
      healthWarnings: [],
      turnFailures: [],
      finalMetrics: {
        ...metricsRow(30),
        capability_overclaim_count: 1,
        capability_ambiguity_count: 2,
        capability_boundary_refusal_count: 3,
        dormant_not_archive_eligible_count: 3,
        dormant_archive_eligible_count: 1,
        archive_oldest_inactive_turns: 24,
        archive_inactive_turn_distribution: {
          "0-15": 4,
          "15-20": 3,
          "20-30": 1,
          "30+": 0,
        },
        archive_archivable_count: 1,
        archive_skipped_borg_owned: 2,
        archive_skipped_due_date: 1,
        archive_skipped_below_threshold: 3,
        archive_skipped_other: 0,
        archive_oldest_archivable_inactive_turns: 24,
        shared_state_live_starvation_ever: true,
        shared_state_live_starvation_final: false,
        shared_state_at_cap_turns: 5,
        shared_state_compile_evaluated_turns: 20,
        shared_state_omitted_recent_entries_total_across_compiles: 7,
        shared_state_omitted_live_recent_operational_total_across_compiles: 4,
        shared_state_omitted_live_recent_operational_final_compile: 1,
        shared_state_omitted_live_recent_low_salience_total_across_compiles: 6,
        shared_state_omitted_live_recent_low_salience_final_compile: 2,
        shared_state_omitted_live_old_total_across_compiles: 9,
        shared_state_omitted_live_old_final_compile: 3,
        shared_state_omitted_live_unknown_age_total_across_compiles: 0,
        shared_state_omitted_live_unknown_age_final_compile: 0,
        shared_state_omitted_locked_total_across_compiles: 10,
        shared_state_omitted_locked_final_compile: 4,
        shared_state_omitted_pending_total_across_compiles: 12,
        shared_state_omitted_pending_final_compile: 5,
        shared_state_omitted_locked_recent_total_across_compiles: 3,
        shared_state_omitted_locked_recent_final_compile: 1,
        shared_state_omitted_locked_old_total_across_compiles: 7,
        shared_state_omitted_locked_old_final_compile: 3,
        shared_state_omitted_locked_unknown_age_total_across_compiles: 2,
        shared_state_omitted_locked_unknown_age_final_compile: 0,
        shared_state_omitted_locked_with_active_critical_commitment_total_across_compiles: 1,
        shared_state_omitted_locked_with_active_critical_commitment_final_compile: 0,
        shared_state_omitted_locked_with_operational_canonicalizer_total_across_compiles: 2,
        shared_state_omitted_locked_with_operational_canonicalizer_final_compile: 1,
        shared_state_omitted_locked_indexed_only_total_across_compiles: 10,
        shared_state_omitted_locked_indexed_only_final_compile: 4,
        shared_state_at_cap_but_all_keys_indexed_compiles_total: 5,
        shared_state_at_cap_with_operational_omission_compiles_total: 2,
        shared_state_at_cap_with_cap_rejection_compiles_total: 1,
        shared_state_lifecycle_aging_demotable_final_compile: 24,
        shared_state_lifecycle_aging_unknown_age_final_compile: 2,
        shared_state_lifecycle_aging_demoted_total: 1,
        shared_state_lifecycle_aging_blocked_by_current_turn_update_total: 12,
        shared_state_lifecycle_aging_blocked_by_ledger_overlap_total: 34,
        shared_state_lifecycle_aging_blocked_by_recent_retrieval_total: 120,
        shared_state_lifecycle_aging_blocked_by_active_canonicalizer_critical_total: 8,
        shared_state_lifecycle_aging_blocked_by_active_canonicalizer_operational_total: 6,
        shared_state_lifecycle_aging_blocked_by_hard_total: 54,
        shared_state_lifecycle_aging_blocked_by_soft_total: 126,
        shared_state_lifecycle_aging_blocked_by_multiple_reasons_total: 410,
        shared_state_lifecycle_aging_low_salience_to_dormant_demotable_final_compile: 0,
        shared_state_lifecycle_aging_low_salience_to_dormant_demoted_total: 0,
        closure_loop_completed_count: 9,
        closure_loop_degraded_count: 1,
        closure_response_audit_failed_open_total: 1,
        closure_pressure_mixed_observed_total: 3,
        closure_pressure_closure_only_observed_total: 2,
        closure_pressure_closure_only_suppressed_total: 1,
        closure_pressure_mixed_passed_no_active_preference_total: 2,
        closure_pressure_mixed_by_span_kind: {
          aphoristic_valediction: 2,
          imperative_closer: 1,
        },
        corrective_preference_completed_count: 8,
        corrective_preference_degraded_count: 2,
        extractor_max_tokens_stop_count: 4,
        extractor_max_tokens_total_by_label: {
          closure_loop_classifier: 46,
          corrective_preference_extractor: 10,
        },
        extractor_degraded_total_by_label: {
          closure_loop_classifier: 26,
          corrective_preference_extractor: 1,
        },
        commitment_candidates_rejected_ungrounded_claim_total: 1,
        commitment_candidates_rejected_ungrounded_claim_by_label_family: {
          kinship: 1,
        },
        shared_state_compiler_max_tokens_total: 1,
        shared_state_compiler_degraded_total: 2,
        shared_state_compiler_repair_attempted_total: 3,
        shared_state_compiler_repair_succeeded_total: 2,
        shared_state_compiler_repair_failed_total: 1,
        shared_state_compiler_repair_failed_by_rejection_reason: {
          missing_new_key_reason: 1,
          relationship_claim_ungrounded: 2,
        },
        shared_state_update_checked_for_empty_total: 4,
        shared_state_empty_update_attempted_total: 4,
        shared_state_empty_update_dropped_total: 4,
        shared_state_empty_update_drop_rate: 1,
        shared_state_empty_update_repaired_total: 0,
        shared_state_compiler_operations_total_by_kind: {
          add: 5,
          update: 1,
          supersede: 1,
          prune: 2,
        },
        shared_state_add_to_update_ratio: 2.5,
        shared_state_entries_by_key: {
          "plan.attendees": 3,
          "decision.architecture": 1,
        },
        shared_state_add_to_update_ratio_by_key: {
          "plan.attendees": 4,
          "decision.architecture": 1,
        },
        shared_state_top_keys_by_entry_count: {
          "plan.attendees": 3,
          "decision.architecture": 1,
        },
        shared_state_add_rejected_cap_exceeded_total: 1,
        shared_state_operations_rejected_ungrounded_claim_total: 2,
        shared_state_operations_rejected_ungrounded_claim_by_label_family: {
          intimate_partner: 2,
        },
        semantic_revision_calls_total: 6,
        semantic_revision_candidates_reviewed_total: 31,
        semantic_revision_superseded_total: 4,
        semantic_revision_contradicted_total: 2,
        semantic_revision_degraded_total: 3,
        semantic_revision_skipped_over_cap_total: 1,
        commitment_regeneration_attempted_total: 2,
        commitment_regeneration_succeeded_total: 1,
        commitment_regeneration_failed_total: 1,
      },
      durationMs: 1,
    });

    expect(report).toContain("Capability audit: overclaims 1, ambiguities 2, boundary refusals 3");
    expect(report).toContain("Commitment regeneration: attempted 2, succeeded 1, failed 1");
    expect(report).toContain(
      "Action archive visibility: archivable 1, skipped Borg-owned 2, skipped due-date 1, skipped below threshold 3, skipped other 0, oldest archivable 24 turns, inactive buckets 0-15=4, 15-20=3, 20-30=1, 30+=0",
    );
    expect(report).toContain("live starvation ever true, live starvation final false");
    expect(report).toContain("Shared-state omission (final compile / cumulative):");
    expect(report).toContain("recent operational: 1 / 4");
    expect(report).toContain("locked: 4 / 10");
    expect(report).toContain("Shared-state locked omission severity (final compile / cumulative):");
    expect(report).toContain("with operational canonicalizer: 1 / 2");
    expect(report).toContain("Shared-state at-cap severity:");
    expect(report).toContain("at cap with cap rejection: 1");
    expect(report).toContain("Lifecycle aging (live -> low_salience):");
    expect(report).toContain("demotable: 24 (final compile)");
    expect(report).toContain("unknown age: 2 (final compile)");
    expect(report).toContain(
      "blocked by: current_turn=12, patch_touch=0, ledger_overlap=34, recent_retrieval=120, active_canonicalizer_critical=8, active_canonicalizer_operational=6, hard=54, soft=126, multiple=410",
    );
    expect(report).toContain("Lifecycle aging (low_salience -> dormant):");
    expect(report).toContain(
      "Closure pressure: audit failed open 1, mixed observed 3, closure-only observed 2, closure-only suppressed 1, mixed/no-active-preference 2, mixed span kinds aphoristic_valediction=2, imperative_closer=1",
    );
    expect(report).toContain(
      "Extractor health: closure loop degraded 1/9, corrective preference degraded 2/8, max-token stops 4",
    );
    expect(report).toContain("## Cumulative Extractor Health");
    expect(report).toContain(
      "Max-token stops by label: closure_loop_classifier=46, corrective_preference_extractor=10",
    );
    expect(report).toContain(
      "Degraded by label: closure_loop_classifier=26, corrective_preference_extractor=1",
    );
    expect(report).toContain(
      "Commitment gate rejections: ungrounded relationship claims 1 total (kinship=1)",
    );
    expect(report).toContain("## Cumulative Compiler Health");
    expect(report).toContain("Shared-state compiler max-token stops: 1");
    expect(report).toContain("Shared-state compiler degraded events: 2");
    expect(report).toContain("Shared-state compiler repair: attempted 3, succeeded 2, failed 1");
    expect(report).toContain(
      "Shared-state repair failures by reason: missing_new_key_reason=1, relationship_claim_ungrounded=2",
    );
    expect(report).toContain("Shared-state empty-update:");
    expect(report).toContain("update operations checked: 4");
    expect(report).toContain("empty updates dropped: 4");
    expect(report).toContain("drop rate: 100.0%");
    expect(report).toContain(
      "Shared-state compiler operations by kind: add=5, prune=2, supersede=1, update=1",
    );
    expect(report).toContain("Shared-state compiler add/update ratio: 2.50");
    expect(report).toContain("Shared-state add rejected by per-key cap: 1");
    expect(report).toContain(
      "Shared-state gate rejections: ungrounded relationship claims 2 total (intimate_partner=2)",
    );
    expect(report).toContain("## Cumulative Semantic Revision Health");
    expect(report).toContain("Revision LLM calls: 6");
    expect(report).toContain("Candidates reviewed: 31");
    expect(report).toContain("Nodes superseded: 4");
    expect(report).toContain("Nodes contradicted: 2");
    expect(report).toContain("Degraded events: 3");
    expect(report).toContain("Skipped over cap: 1");
  });

  it("formats overseer checkpoint statuses as labelled fields", () => {
    const report = formatSimulatorReport({
      runId: "sim-runner-status-label-test",
      persona: tomPersona.key,
      personas: [tomPersona.key],
      audience: "Tom",
      totalTurns: 10,
      resultState: "completed",
      sessions: [],
      suppressionEvents: [],
      overseerCheckpoints: [
        {
          ts: Date.now(),
          turn_counter: 10,
          status: "concerning",
          observations: ["Capability overclaim found."],
          recommendation: "Healthy with one concerning note.",
          findings: [
            {
              category: "K",
              claim_status: "unsupported",
              source_kind: "emitted_output",
              status_impact: "concerning",
              assistant_stream_entry_id: "strm_capability_report",
              assistant_ts: 12,
              quoted_emitted_span: "I'll monitor p95",
              evidence_summary: "Borg promised external monitoring.",
            },
            {
              category: "I",
              claim_status: "grounded",
              source_kind: "snapshot_memory",
              status_impact: "none",
              evidence_summary: "Instrumentation is stable.",
            },
          ],
          rejected_findings: [],
          raw_verdict: {
            status: "healthy",
            observations: ["Capability overclaim found."],
            recommendation: "Healthy with one concerning note.",
            findings: [],
          },
        },
      ],
      turnFailures: [],
      finalMetrics: metricsRow(10),
      durationMs: 1,
    });

    expect(report).toContain("- Turn 10:");
    expect(report).toContain("Raw status: healthy");
    expect(report).toContain("Recommendation: Healthy with one concerning note.");
    expect(report).toContain("Run worst behavioral status: healthy");
    expect(report).toContain("Run worst capability status: concerning");
    expect(report).toContain("Final checkpoint status: concerning");
    expect(report).toContain("Final checkpoint active findings: 1");
    expect(report).toContain("Validated checkpoint concerns by turn:");
    expect(report).toContain(
      "- Turn 10: capability (K unsupported: Borg promised external monitoring.)",
    );
    expect(report).toContain("Behavioral status: healthy");
    expect(report).toContain("Substrate status: healthy");
    expect(report).toContain("Capability status: concerning");
    expect(report).toContain("Worst status: concerning");
    expect(report).toContain("Open concerns:");
    expect(report).not.toContain("Turn 10: concerning --");
  });

  it("summarizes worst run status across checkpoints above final checkpoint status", () => {
    const report = formatSimulatorReport({
      runId: "sim-runner-worst-status-summary-test",
      persona: tomPersona.key,
      personas: [tomPersona.key],
      audience: "Tom",
      totalTurns: 20,
      resultState: "completed",
      sessions: [],
      suppressionEvents: [],
      overseerCheckpoints: [
        {
          ts: Date.now(),
          turn_counter: 10,
          status: "concerning",
          observations: ["A behavioral concern appeared."],
          recommendation: "Inspect.",
          findings: [
            {
              category: "H",
              claim_status: "unsupported",
              source_kind: "emitted_output",
              status_impact: "concerning",
              evidence_summary: "Borg made an unsupported memory claim mid-run.",
            },
          ],
          rejected_findings: [],
          raw_verdict: {
            status: "concerning",
            observations: ["A behavioral concern appeared."],
            recommendation: "Inspect.",
            findings: [],
          },
        },
        {
          ts: Date.now(),
          turn_counter: 20,
          status: "healthy",
          observations: ["Final window was healthy."],
          recommendation: "No action.",
          findings: [],
          rejected_findings: [],
          raw_verdict: {
            status: "healthy",
            observations: ["Final window was healthy."],
            recommendation: "No action.",
            findings: [],
          },
        },
      ],
      healthWarnings: [],
      turnFailures: [],
      finalMetrics: metricsRow(20),
      durationMs: 1,
    });

    expect(report).toContain("Run worst behavioral status: concerning");
    expect(report).toContain("Run worst substrate status: healthy");
    expect(report).toContain("Run worst capability status: healthy");
    expect(report).toContain("Final checkpoint status: healthy");
    expect(report).toContain("Final checkpoint active findings: 0");
    expect(report).toContain(
      "- Turn 10: behavioral (H unsupported: Borg made an unsupported memory claim mid-run.)",
    );
  });

  it("uses the validated checkpoint status for the final checkpoint summary", () => {
    const report = formatSimulatorReport({
      runId: "sim-runner-final-validated-status-test",
      persona: tomPersona.key,
      personas: [tomPersona.key],
      audience: "Tom",
      totalTurns: 10,
      resultState: "completed",
      sessions: [],
      suppressionEvents: [],
      overseerCheckpoints: [
        {
          ts: Date.now(),
          turn_counter: 10,
          status: "concerning",
          observations: ["Overseer call cap reached before a structured verdict."],
          recommendation: "Rerun the checkpoint.",
          findings: [],
          rejected_findings: [],
          raw_verdict: {
            status: "healthy",
            observations: ["Overseer call cap reached before a structured verdict."],
            recommendation: "Rerun the checkpoint.",
            findings: [],
          },
        },
      ],
      healthWarnings: [],
      turnFailures: [],
      finalMetrics: metricsRow(10),
      durationMs: 1,
    });

    expect(report).toContain("Final checkpoint status: concerning");
    expect(report).toContain("Final checkpoint active findings: 0");
    expect(report).toContain(
      "- Turn 10: concerning (Overseer call cap reached before a structured verdict.)",
    );
  });

  it("downgrades behavioral status for hard aborts but not intentional suppressions", () => {
    const intentionalOnly = formatSimulatorReport({
      runId: "sim-runner-intentional-suppression-status-test",
      persona: tomPersona.key,
      personas: [tomPersona.key],
      audience: "Tom",
      totalTurns: 1,
      resultState: "completed",
      sessions: [],
      suppressionEvents: [],
      overseerCheckpoints: [],
      healthWarnings: [],
      turnFailures: [],
      finalMetrics: {
        ...metricsRow(1),
        borg_intentional_suppressions: 1,
        borg_intentional_suppressions_by_reason: {
          finalizer_no_output: 1,
        },
      },
      durationMs: 1,
    });
    const hardAbort = formatSimulatorReport({
      runId: "sim-runner-hard-abort-status-test",
      persona: tomPersona.key,
      personas: [tomPersona.key],
      audience: "Tom",
      totalTurns: 1,
      resultState: "completed",
      sessions: [],
      suppressionEvents: [],
      overseerCheckpoints: [],
      healthWarnings: [],
      turnFailures: [],
      finalMetrics: {
        ...metricsRow(1),
        borg_hard_aborted_turns: 1,
        borg_aborted_turns: 1,
      },
      durationMs: 1,
    });

    expect(intentionalOnly).toContain("Run worst behavioral status: healthy");
    expect(hardAbort).toContain("Run worst behavioral status: concerning");
  });

  it("detects expanded simulator health warnings with conservative thresholds", () => {
    const capabilityCheckpoint = (findings: OverseerVerdict["findings"]): OverseerVerdict => ({
      ts: Date.now(),
      turn_counter: 20,
      status: "concerning",
      observations: ["Capability overclaim found."],
      recommendation: "Inspect.",
      findings,
      rejected_findings: [],
      raw_verdict: {
        status: "concerning",
        observations: ["Capability overclaim found."],
        recommendation: "Inspect.",
        findings: [],
      },
    });
    const capabilityFinding = (
      claimStatus: "grounded" | "unsupported" | "contradicted" | "unclear",
      index = 0,
      statusImpact: "none" | "concerning" | "failing" = claimStatus === "grounded"
        ? "none"
        : "concerning",
    ): OverseerVerdict["findings"][number] => ({
      category: "K",
      claim_status: claimStatus,
      source_kind: claimStatus === "grounded" ? "snapshot_memory" : "emitted_output",
      status_impact: statusImpact,
      assistant_stream_entry_id:
        claimStatus === "grounded" ? undefined : `strm_capability_warning_${index}`,
      quoted_emitted_span: claimStatus === "grounded" ? undefined : "I'll monitor p95",
      evidence_summary:
        claimStatus === "grounded"
          ? "Borg correctly refused an unwired capability."
          : "Borg capability claim needs audit.",
    });
    const capabilityOverclaimCheckpoint = capabilityCheckpoint([capabilityFinding("unsupported")]);
    const cases: Array<{
      name: string;
      rows: MetricsRow[];
      expectedKinds: SimulatorHealthWarningKind[];
      scenarioKey?: string;
      overseerCheckpoints?: OverseerVerdict[];
    }> = [
      {
        name: "active actions final high fires",
        rows: [{ ...metricsRow(12), action_record_count_active: 31 }],
        expectedKinds: ["active_actions_final_high"],
      },
      {
        name: "committed actions final high fires",
        rows: [{ ...metricsRow(12), action_record_count_committed_to_do: 19 }],
        expectedKinds: ["committed_to_do_actions_final_high"],
      },
      {
        name: "committed actions at threshold does not fire",
        rows: [{ ...metricsRow(12), action_record_count_committed_to_do: 18 }],
        expectedKinds: [],
      },
      {
        name: "actions per turn high fires",
        rows: [{ ...metricsRow(12), actions_per_turn: 2.01 }],
        expectedKinds: ["actions_per_turn_high"],
      },
      {
        name: "actions per turn at threshold does not fire",
        rows: [{ ...metricsRow(12), actions_per_turn: 2 }],
        expectedKinds: [],
      },
      {
        name: "salient actions per turn high fires",
        rows: [{ ...metricsRow(12), salient_actions_per_turn: 0.81 }],
        expectedKinds: ["salient_actions_per_turn_high"],
      },
      {
        name: "salient actions per turn at threshold does not fire",
        rows: [{ ...metricsRow(12), salient_actions_per_turn: 0.8 }],
        expectedKinds: [],
      },
      {
        name: "low action retirement ratio fires",
        rows: [
          {
            ...metricsRow(12),
            action_record_count_total: 10,
            action_retirement_ratio: 0.29,
          },
        ],
        expectedKinds: ["action_retirement_ratio_low"],
      },
      {
        name: "low action retirement ratio waits for enough actions",
        rows: [
          {
            ...metricsRow(12),
            action_record_count_total: 9,
            action_retirement_ratio: 0,
          },
        ],
        expectedKinds: [],
      },
      {
        name: "low action canonicalization rate fires",
        rows: [
          {
            ...metricsRow(12),
            action_record_count_total: 30,
            action_record_count_canonicalized: 0,
            action_retirement_ratio: 0.3,
          },
        ],
        expectedKinds: ["action_canonicalization_rate_low"],
      },
      {
        name: "zero action denominator does not fire",
        rows: [
          {
            ...metricsRow(12),
            action_record_count_total: 0,
            action_record_count_canonicalized: 0,
          },
        ],
        expectedKinds: [],
      },
      {
        name: "retrieval latency max high fires",
        rows: [{ ...metricsRow(12), retrieval_latency_ms: 30_001 }],
        expectedKinds: ["retrieval_latency_max_high"],
      },
      {
        name: "retrieval latency at threshold does not fire",
        rows: [{ ...metricsRow(12), retrieval_latency_ms: 30_000 }],
        expectedKinds: [],
      },
      {
        name: "deliberation latency max high fires",
        rows: [{ ...metricsRow(12), deliberation_latency_ms: 120_001 }],
        expectedKinds: ["deliberation_latency_max_high"],
      },
      {
        name: "deliberation latency at threshold does not fire",
        rows: [{ ...metricsRow(12), deliberation_latency_ms: 120_000 }],
        expectedKinds: [],
      },
      {
        name: "semantic revision LLM calls high fires",
        rows: [
          {
            ...metricsRow(12),
            shared_state_semantic_revisions_attempted: 41,
            shared_state_semantic_nodes_marked_superseded: 41,
          },
        ],
        expectedKinds: ["semantic_revision_llm_calls_high"],
      },
      {
        name: "semantic revision LLM calls at threshold does not fire",
        rows: [
          {
            ...metricsRow(12),
            shared_state_semantic_revisions_attempted: 40,
            shared_state_semantic_nodes_marked_superseded: 40,
          },
        ],
        expectedKinds: [],
      },
      {
        name: "semantic revision transition yield low fires",
        rows: [
          {
            ...metricsRow(12),
            shared_state_semantic_revisions_attempted: 6,
            shared_state_semantic_nodes_marked_superseded: 1,
          },
        ],
        expectedKinds: ["semantic_revision_transition_yield_low"],
      },
      {
        name: "zero semantic revision denominator does not fire",
        rows: [
          {
            ...metricsRow(12),
            shared_state_semantic_revisions_attempted: 0,
            shared_state_semantic_nodes_marked_superseded: 0,
            shared_state_semantic_nodes_marked_contradicted: 0,
          },
        ],
        expectedKinds: [],
      },
      {
        name: "classifier degraded rate high fires",
        rows: [
          {
            ...metricsRow(12),
            frame_anomaly_classifier_calls: 10,
            frame_anomaly_degraded_count: 3,
          },
        ],
        expectedKinds: ["classifier_degraded_rate_high"],
      },
      {
        name: "zero classifier denominator does not fire",
        rows: [
          {
            ...metricsRow(12),
            frame_anomaly_classifier_calls: 0,
            frame_anomaly_degraded_count: 0,
          },
        ],
        expectedKinds: [],
      },
      {
        name: "closure loop degraded rate high fires",
        rows: [
          {
            ...metricsRow(12),
            closure_loop_completed_count: 10,
            closure_loop_degraded_count: 2,
          },
        ],
        expectedKinds: ["closure_loop_degraded_rate_high"],
      },
      {
        name: "closure loop degraded rate at threshold does not fire",
        rows: [
          {
            ...metricsRow(12),
            closure_loop_completed_count: 10,
            closure_loop_degraded_count: 1,
          },
        ],
        expectedKinds: [],
      },
      {
        name: "closure loop degraded-only outage fires at full rate",
        rows: [
          {
            ...metricsRow(12),
            closure_loop_completed_count: 0,
            closure_loop_degraded_count: 2,
          },
        ],
        expectedKinds: ["closure_loop_degraded_rate_high"],
      },
      {
        name: "corrective preference degraded rate high fires",
        rows: [
          {
            ...metricsRow(12),
            corrective_preference_completed_count: 10,
            corrective_preference_degraded_count: 2,
          },
        ],
        expectedKinds: ["corrective_preference_degraded_rate_high"],
      },
      {
        name: "corrective preference degraded-only outage fires at full rate",
        rows: [
          {
            ...metricsRow(12),
            corrective_preference_completed_count: 0,
            corrective_preference_degraded_count: 2,
          },
        ],
        expectedKinds: ["corrective_preference_degraded_rate_high"],
      },
      {
        name: "extractor max tokens high fires for registered canonical labels",
        rows: [
          {
            ...metricsRow(12),
            extractor_max_tokens_total_by_label: {
              pending_action_judge: 1,
            },
          },
        ],
        expectedKinds: ["extractor_max_tokens_high"],
      },
      {
        name: "extractor max tokens severe fires for repeated label stops",
        rows: [
          {
            ...metricsRow(12),
            extractor_max_tokens_total_by_label: {
              closure_loop_classifier: 3,
            },
          },
        ],
        expectedKinds: ["extractor_max_tokens_high", "extractor_max_tokens_severe"],
      },
      {
        name: "shared-state compiler max tokens fires on any stop",
        rows: [
          {
            ...metricsRow(12),
            shared_state_compiler_max_tokens_total: 1,
          },
        ],
        expectedKinds: ["shared_state_compiler_max_tokens_high"],
      },
      {
        name: "dormant archive-eligible active actions fire",
        rows: [
          {
            ...metricsRow(12),
            dormant_archive_eligible_count: 1,
            archive_archivable_count: 1,
          },
        ],
        expectedKinds: ["dormant_archive_eligible_count_high"],
      },
      {
        name: "semantic revision degraded high fires at threshold",
        rows: [
          {
            ...metricsRow(12),
            semantic_revision_error_count: 3,
          },
        ],
        expectedKinds: ["semantic_revision_degraded_high"],
      },
      {
        name: "semantic revision degraded high does not fire below threshold",
        rows: [
          {
            ...metricsRow(12),
            semantic_revision_error_count: 2,
          },
        ],
        expectedKinds: [],
      },
      {
        name: "capability overclaim count high fires",
        rows: [metricsRow(12)],
        overseerCheckpoints: [capabilityOverclaimCheckpoint],
        expectedKinds: ["capability_overclaim_count_high"],
      },
      {
        name: "capability contradicted count high fires",
        rows: [metricsRow(12)],
        overseerCheckpoints: [capabilityCheckpoint([capabilityFinding("contradicted")])],
        expectedKinds: ["capability_overclaim_count_high"],
      },
      {
        name: "capability unclear count high fires only at three",
        rows: [metricsRow(12)],
        overseerCheckpoints: [
          capabilityCheckpoint([
            capabilityFinding("unclear", 1),
            capabilityFinding("unclear", 2),
            capabilityFinding("unclear", 3),
          ]),
        ],
        expectedKinds: ["capability_ambiguity_count_high"],
      },
      {
        name: "capability unclear count below threshold does not fire",
        rows: [metricsRow(12)],
        overseerCheckpoints: [
          capabilityCheckpoint([capabilityFinding("unclear", 1), capabilityFinding("unclear", 2)]),
        ],
        expectedKinds: [],
      },
      {
        name: "capability grounded refusal count does not warn",
        rows: [metricsRow(12)],
        overseerCheckpoints: [capabilityCheckpoint([capabilityFinding("grounded")])],
        expectedKinds: [],
      },
      {
        name: "capability overclaim carryover demotion does not fire",
        rows: [metricsRow(12)],
        overseerCheckpoints: [
          {
            ...capabilityOverclaimCheckpoint,
            findings: [
              {
                ...capabilityOverclaimCheckpoint.findings[0]!,
                status_impact: "none",
                carryover_demoted: true,
              },
            ],
          },
        ],
        expectedKinds: [],
      },
      {
        name: "review queue backlog high fires",
        rows: [
          {
            ...metricsRow(12),
            review_queue_open_count_by_type: {
              ...metricsRow(12).review_queue_open_count_by_type,
              contradiction: 51,
            },
          },
        ],
        expectedKinds: ["review_queue_backlog_high"],
      },
      {
        name: "review queue backlog at threshold does not fire",
        rows: [
          {
            ...metricsRow(12),
            review_queue_open_count_by_type: {
              ...metricsRow(12).review_queue_open_count_by_type,
              contradiction: 50,
            },
          },
        ],
        expectedKinds: [],
      },
      {
        name: "shared-state cap saturation high fires against evaluated compiles",
        rows: [
          {
            ...metricsRow(100),
            shared_state_at_cap_turns: 40,
            shared_state_compile_evaluated_turns: 50,
          },
        ],
        expectedKinds: ["shared_state_cap_saturation_high"],
      },
      {
        name: "shared-state cap saturation ignores skipped compile turns",
        rows: [
          {
            ...metricsRow(100),
            shared_state_at_cap_turns: 10,
            shared_state_compile_evaluated_turns: 50,
          },
        ],
        expectedKinds: [],
      },
      {
        name: "shared-state cap saturation at evaluated threshold does not fire",
        rows: [
          {
            ...metricsRow(100),
            shared_state_at_cap_turns: 25,
            shared_state_compile_evaluated_turns: 50,
          },
        ],
        expectedKinds: [],
      },
      {
        name: "shared-state starvation recovered still fires ever warning",
        rows: [
          {
            ...metricsRow(100),
            shared_state_live_starvation_ever: true,
            shared_state_live_starvation_final: false,
          },
        ],
        expectedKinds: ["shared_state_starvation_high"],
      },
      {
        name: "shared-state starvation persistent fires both warnings",
        rows: [
          {
            ...metricsRow(100),
            shared_state_live_starvation_ever: true,
            shared_state_live_starvation_final: true,
          },
        ],
        expectedKinds: ["shared_state_starvation_high", "shared_state_starvation_persistent"],
      },
      {
        name: "shared-state compiler add dominance fires above threshold",
        rows: [
          {
            ...metricsRow(100),
            shared_state_add_to_update_ratio: 2.01,
          },
        ],
        expectedKinds: ["shared_state_compiler_add_dominant"],
      },
      {
        name: "shared-state compiler add dominance does not fire at threshold",
        rows: [
          {
            ...metricsRow(100),
            shared_state_add_to_update_ratio: 2,
          },
        ],
        expectedKinds: [],
      },
    ];

    for (const testCase of cases) {
      const warnings = simulatorHealthWarningsForRows(testCase.rows, {
        scenarioKey: testCase.scenarioKey,
        overseerCheckpoints: testCase.overseerCheckpoints,
      });

      expect(
        warnings.map((warning) => warning.kind),
        testCase.name,
      ).toEqual(testCase.expectedKinds);
    }
  });

  it("warns on the highest extractor max-token label", () => {
    const oneLabelHighWarnings = simulatorHealthWarningsForRows([
      {
        ...metricsRow(12),
        extractor_max_tokens_total_by_label: {
          closure_loop_classifier: 2,
          corrective_preference_extractor: 1,
        },
      },
    ]).filter((warning) => warning.kind === "extractor_max_tokens_high");

    expect(oneLabelHighWarnings).toEqual([
      expect.objectContaining({
        kind: "extractor_max_tokens_high",
        label: "closure_loop_classifier",
        observed_value: 2,
      }),
    ]);
  });

  it("normalizes legacy-only semantic revision metric fields for direct health warnings", () => {
    const newFieldRow = {
      ...metricsRow(12),
      shared_state_semantic_revisions_attempted: 41,
      shared_state_semantic_revisions_completed_succeeded: 1,
      shared_state_semantic_nodes_marked_superseded: 1,
      shared_state_semantic_nodes_marked_contradicted: 0,
      shared_state_semantic_revision_cache_hits: 2,
      shared_state_semantic_revision_cache_size: 3,
    } satisfies MetricsRow;
    const legacyFieldRow = legacyOnlySemanticRevisionMetrics(newFieldRow);

    expect(legacyFieldRow).not.toHaveProperty("shared_state_semantic_revisions_attempted");
    expect(legacyFieldRow).toHaveProperty("decision_artifact_semantic_revisions_attempted", 41);
    expect(simulatorHealthWarningsForRows([legacyFieldRow])).toEqual(
      simulatorHealthWarningsForRows([newFieldRow]),
    );
    expect(simulatorHealthWarningsForRows([legacyFieldRow]).map((warning) => warning.kind)).toEqual([
      "semantic_revision_llm_calls_high",
      "semantic_revision_transition_yield_low",
    ]);
  });

  it("summarizes capability findings by claim-status severity", () => {
    const checkpoint: OverseerVerdict = {
      ts: Date.now(),
      turn_counter: 10,
      status: "concerning",
      observations: ["Mixed capability findings."],
      recommendation: "Inspect.",
      findings: [
        {
          category: "K",
          claim_status: "unsupported",
          source_kind: "emitted_output",
          status_impact: "concerning",
          assistant_stream_entry_id: "strm_capability_unsupported",
          quoted_emitted_span: "I'll monitor p95",
          evidence_summary: "Unsupported capability claim.",
        },
        {
          category: "K",
          claim_status: "contradicted",
          source_kind: "emitted_output",
          status_impact: "failing",
          assistant_stream_entry_id: "strm_capability_contradicted",
          quoted_emitted_span: "I'll send it tomorrow",
          evidence_summary: "Contradicted capability claim.",
        },
        {
          category: "K",
          claim_status: "unclear",
          source_kind: "emitted_output",
          status_impact: "concerning",
          assistant_stream_entry_id: "strm_capability_unclear",
          quoted_emitted_span: "I'll surface it",
          evidence_summary: "Ambiguous capability phrasing.",
        },
        {
          category: "K",
          claim_status: "grounded",
          source_kind: "snapshot_memory",
          status_impact: "none",
          evidence_summary: "Borg refused the unwired capability.",
        },
        {
          category: "K",
          claim_status: "unsupported",
          source_kind: "emitted_output",
          status_impact: "none",
          carryover_demoted: true,
          evidence_summary: "Prior finding already counted.",
        },
      ],
      rejected_findings: [],
      raw_verdict: {
        status: "concerning",
        observations: ["Mixed capability findings."],
        recommendation: "Inspect.",
        findings: [],
      },
    };

    expect(capabilityFindingMetrics([checkpoint])).toEqual({
      capability_overclaim_count: 2,
      capability_ambiguity_count: 1,
      capability_boundary_refusal_count: 1,
    });
  });

  it("builds emission baseline config overrides", () => {
    const scenario = createSimulatorScenario(tomPersona, 100, {
      emissionBaseline: true,
    });

    expect(scenario.borgConfigOverrides).toEqual({
      generation: {
        evidenceLedger: { enabled: true },
        postGenerationGuards: {
          commitment: { mode: "enforce" },
          closurePressure: { mode: "enforce" },
        },
      },
    });
  });

  it("keeps simulator config overrides unset by default", () => {
    const scenario = createSimulatorScenario(tomPersona, 100);

    expect(scenario.borgConfigOverrides).toBeUndefined();
  });

  it("enables evidence ledger in Borg config for multi-persona runs without emission baseline", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const alice = {
      key: "alice-test",
      displayName: "Alice",
      systemPrompt: "Speak as Alice.",
    };
    const bob = {
      key: "bob-test",
      displayName: "Bob",
      systemPrompt: "Speak as Bob.",
    };
    const aliceSession = fakePersonaSession(["alice first"]);
    const bobSession = fakePersonaSession(["bob first"]);
    const openSpy = vi.spyOn(Borg, "open").mockResolvedValue(fakeSimulatorBorg());
    vi.spyOn(BorgTransport.prototype, "resolveEntity").mockReturnValue(createEntityId());
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) =>
      chatResult({
        response: "Borg replied.",
        emitted: true,
        turnId: "turn-multi-ledger",
        sessionId: options.sessionId as SessionId,
      }),
    );

    await runSimulation({
      runId: "sim-runner-multi-persona-ledger-test",
      persona: alice,
      personas: [alice, bob],
      personaSessions: [aliceSession.session, bobSession.session],
      totalTurns: 1,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(openSpy.mock.calls[0]?.[0]?.config?.generation.evidenceLedger.enabled).toBe(true);
  });

  it("drafts one persona turn while retrying transient transport failures", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const draftMessage = "stable persona draft";
    const borgResponse = "Borg replied.";
    const failureMessage = "transient transport failure";
    const persona = fakePersonaSession([draftMessage]);
    const seenMessages: string[] = [];
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (message, options = {}) => {
      seenMessages.push(message);

      if (seenMessages.length < 3) {
        throw new Error(failureMessage);
      }

      return chatResult({
        response: borgResponse,
        emitted: true,
        turnId: "turn-retry-success",
        sessionId: options.sessionId as SessionId,
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-retry-draft-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 1,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });
    const [metricsRow] = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line) as { transport_chat_attempts: number });

    expect(seenMessages).toEqual([draftMessage, draftMessage, draftMessage]);
    expect(persona.prepareNextTurn).toHaveBeenCalledOnce();
    expect(persona.prepareNextTurn).toHaveBeenCalledWith({ kind: "new_session" });
    expect(persona.commit).toHaveBeenCalledOnce();
    expect(persona.commit.mock.calls[0]?.[0]).toMatchObject({ message: draftMessage });
    expect(persona.commit.mock.calls[0]?.[1]).toBe(borgResponse);
    expect(persona.rollback).not.toHaveBeenCalled();
    expect(report.turnFailures).toEqual([]);
    expect(metricsRow?.transport_chat_attempts).toBe(3);
  });

  it("records aborted attempt metrics rows before a retry succeeds", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const tracePath = join(dir, "trace.jsonl");
    const draftMessage = "stable persona draft";
    const failureMessage = "transient transport failure";
    const persona = fakePersonaSession([draftMessage]);
    let attempts = 0;
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      attempts += 1;

      if (attempts <= 2) {
        appendFileSync(
          tracePath,
          `${JSON.stringify({
            ts: attempts,
            turnId: `turn-aborted-${attempts}`,
            event: "turn.rejected",
          })}\n`,
        );
        throw new Error(failureMessage);
      }

      return chatResult({
        response: "Borg recovered.",
        emitted: true,
        turnId: "turn-success",
        sessionId: options.sessionId as SessionId,
      });
    });

    await runSimulation({
      runId: "sim-runner-aborted-attempt-metrics-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 1,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath,
      mock: true,
    });
    const metricsRows = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map(
        (line) =>
          JSON.parse(line) as {
            event: string;
            turnId: string;
            transport_chat_attempts: number;
          },
      );

    expect(metricsRows).toMatchObject([
      {
        event: "aborted_attempt",
        turnId: "turn-aborted-1",
        transport_chat_attempts: 1,
      },
      {
        event: "aborted_attempt",
        turnId: "turn-aborted-2",
        transport_chat_attempts: 2,
      },
      {
        event: "turn_metrics",
        turnId: "turn-success",
        transport_chat_attempts: 3,
      },
    ]);
  });

  it("rolls back a persona draft and records aborted metrics after exhausted retries", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const draftMessage = "rollback persona draft";
    const borgResponse = "Recovered Borg reply.";
    const failureMessage = "exhausted transport failure";
    const persona = fakePersonaSession([draftMessage]);
    const seenMessages: string[] = [];
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (message, options = {}) => {
      seenMessages.push(message);

      if (seenMessages.length <= 3) {
        throw new Error(failureMessage);
      }

      return chatResult({
        response: borgResponse,
        emitted: true,
        turnId: "turn-after-abort",
        sessionId: options.sessionId as SessionId,
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-retry-rollback-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 2,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });
    const metricsRows = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map(
        (line) =>
          JSON.parse(line) as {
            event: string;
            transport_chat_attempts: number;
            failure_reason?: string;
            borg_hard_aborted_turns?: number;
            borg_aborted_turns?: number;
          },
      );

    expect(seenMessages).toEqual([draftMessage, draftMessage, draftMessage, draftMessage]);
    expect(persona.prepareNextTurn).toHaveBeenCalledTimes(2);
    expect(persona.prepareNextTurn.mock.calls.map(([previous]) => previous)).toEqual([
      { kind: "new_session" },
      { kind: "new_session" },
    ]);
    expect(persona.rollback).toHaveBeenCalledOnce();
    expect(persona.commit).toHaveBeenCalledOnce();
    expect(report.turnFailures).toEqual([
      {
        turn: 1,
        attempts: 3,
        error: failureMessage,
      },
    ]);
    expect(metricsRows).toMatchObject([
      {
        event: "aborted_turn",
        transport_chat_attempts: 3,
        failure_reason: failureMessage,
        borg_hard_aborted_turns: 1,
        borg_aborted_turns: 1,
      },
      {
        event: "turn_metrics",
        transport_chat_attempts: 1,
        borg_hard_aborted_turns: 1,
        borg_aborted_turns: 1,
      },
    ]);
  });

  it("runs a 20-turn mock simulation with overseer checkpoints and metrics", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const auditWindows: Array<[number | undefined, number]> = [];
    spyMaintenanceTick();
    const report = await runSimulation({
      runId: "sim-runner-test",
      persona: tomPersona,
      totalTurns: 20,
      checkEvery: 10,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
      overseerRunner: async ({ auditWindowStartTurn, turnCounter }) => {
        auditWindows.push([auditWindowStartTurn, turnCounter]);
        return healthyOverseerVerdict(turnCounter, ["Mock overseer saw no degradation."]);
      },
    });
    const metricsRows = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line) as { turn_counter: number });

    expect(report.totalTurns).toBe(20);
    expect(Object.hasOwn(report, "probes")).toBe(false);
    expect(report.overseerCheckpoints).toHaveLength(2);
    expect(auditWindows).toEqual([
      [1, 10],
      [11, 20],
    ]);
    expect(metricsRows).toHaveLength(20);
    expect(metricsRows.at(-1)?.turn_counter).toBe(20);
  });

  it("writes aggregate capability metrics into the final JSONL row", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    spyMaintenanceTick();

    const report = await runSimulation({
      runId: "sim-runner-capability-final-metrics-test",
      persona: tomPersona,
      totalTurns: 1,
      checkEvery: 1,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
      overseerRunner: async ({ turnCounter }) => {
        const finding = {
          category: "K" as const,
          claim_status: "unclear" as const,
          source_kind: "emitted_output" as const,
          status_impact: "concerning" as const,
          assistant_stream_entry_id: "strm_capability_final_metrics",
          quoted_emitted_span: "I'll monitor that.",
          evidence_summary: "Capability boundary was ambiguous.",
        };
        const raw_verdict = {
          status: "concerning" as const,
          observations: ["Capability audit found one ambiguity."],
          recommendation: "Inspect the boundary wording.",
          findings: [finding],
        };

        return {
          ts: Date.now(),
          turn_counter: turnCounter,
          status: raw_verdict.status,
          observations: raw_verdict.observations,
          recommendation: raw_verdict.recommendation,
          findings: [finding],
          rejected_findings: [],
          raw_verdict,
        };
      },
    });
    const finalJsonlRow = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line) as MetricsRow)
      .at(-1);

    expect(report.finalMetrics.capability_ambiguity_count).toBe(1);
    expect(finalJsonlRow).toMatchObject({
      capability_overclaim_count: report.finalMetrics.capability_overclaim_count,
      capability_ambiguity_count: report.finalMetrics.capability_ambiguity_count,
      capability_boundary_refusal_count: report.finalMetrics.capability_boundary_refusal_count,
    });
  });

  it("carries overseer finding dedup state across checkpoints", async () => {
    const dir = tempDir();
    const streamId = "strm_runner_carryover";
    spyMaintenanceTick();

    const report = await runSimulation({
      runId: "sim-runner-carryover-dedup-test",
      persona: tomPersona,
      totalTurns: 20,
      checkEvery: 10,
      metricsPath: join(dir, "metrics.jsonl"),
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
      overseerRunner: async ({ turnCounter, carryoverCache }) => {
        if (carryoverCache === undefined) {
          throw new Error("missing carryover cache");
        }

        return validateRunnerOverseerVerdict({
          turnCounter,
          carryoverCache,
          rawVerdict: {
            status: "concerning",
            observations: ["Same incident was surfaced."],
            recommendation: "Dedup repeated finding.",
            findings: [
              {
                category: "H",
                claim_status: "grounded",
                source_kind: "emitted_output",
                status_impact: "concerning",
                assistant_stream_entry_id: streamId,
                assistant_ts: turnCounter,
                evidence_summary: "Borg hedged around a specific unsupported claim.",
              },
            ],
          },
        });
      },
    });

    expect(report.overseerCheckpoints).toHaveLength(2);
    expect(report.overseerCheckpoints[0]?.status).toBe("concerning");
    expect(report.overseerCheckpoints[1]?.status).toBe("healthy");
    expect(report.overseerCheckpoints[1]?.findings[0]).toMatchObject({
      status_impact: "none",
      carryover_demoted: true,
      carryover_original_status_impact: "concerning",
      carryover_cached_status_impact: "concerning",
      carryover_cached_stream_entry_id: streamId,
      carryover_cached_at_turn: 10,
    });
  });

  it("keeps higher-impact overseer findings as cross-checkpoint escalations", async () => {
    const dir = tempDir();
    const streamId = "strm_runner_escalation";
    spyMaintenanceTick();

    const report = await runSimulation({
      runId: "sim-runner-carryover-escalation-test",
      persona: tomPersona,
      totalTurns: 20,
      checkEvery: 10,
      metricsPath: join(dir, "metrics.jsonl"),
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
      overseerRunner: async ({ turnCounter, carryoverCache }) => {
        if (carryoverCache === undefined) {
          throw new Error("missing carryover cache");
        }

        const statusImpact = turnCounter === 10 ? "concerning" : "failing";

        return validateRunnerOverseerVerdict({
          turnCounter,
          carryoverCache,
          rawVerdict: {
            status: statusImpact,
            observations: ["Same incident escalated."],
            recommendation: "Treat higher impact as real escalation.",
            findings: [
              {
                category: "H",
                claim_status: "grounded",
                source_kind: "emitted_output",
                status_impact: statusImpact,
                assistant_stream_entry_id: streamId,
                assistant_ts: turnCounter,
                evidence_summary: "Borg hedged around a specific unsupported claim.",
              },
            ],
          },
        });
      },
    });

    expect(report.overseerCheckpoints).toHaveLength(2);
    expect(report.overseerCheckpoints[0]?.status).toBe("concerning");
    expect(report.overseerCheckpoints[1]?.status).toBe("failing");
    expect(report.overseerCheckpoints[1]?.findings[0]).toMatchObject({
      status_impact: "failing",
    });
    expect(report.overseerCheckpoints[1]?.findings[0]?.carryover_demoted).toBeUndefined();
  });

  it("runs a final overseer checkpoint for trailing partial audit windows", async () => {
    spyMaintenanceTick();

    for (const input of [
      {
        totalTurns: 15,
        checkEvery: 10,
        expectedWindows: [
          [1, 10],
          [11, 15],
        ],
      },
      {
        totalTurns: 5,
        checkEvery: 10,
        expectedWindows: [[1, 5]],
      },
    ] as const) {
      const dir = tempDir();
      const auditWindows: Array<[number | undefined, number]> = [];

      const report = await runSimulation({
        runId: `sim-runner-final-overseer-${input.totalTurns}`,
        persona: tomPersona,
        totalTurns: input.totalTurns,
        checkEvery: input.checkEvery,
        metricsPath: join(dir, "metrics.jsonl"),
        dataDir: join(dir, "data"),
        tracePath: join(dir, "trace.jsonl"),
        mock: true,
        overseerRunner: async ({ auditWindowStartTurn, turnCounter }) => {
          auditWindows.push([auditWindowStartTurn, turnCounter]);
          return healthyOverseerVerdict(turnCounter, ["Mock overseer saw no degradation."]);
        },
      });

      expect(report.overseerCheckpoints).toHaveLength(input.expectedWindows.length);
      expect(auditWindows).toEqual(input.expectedWindows);
    }
  });

  it("runs overseer checkpoints on suppressed turns before continuing", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const persona = fakePersonaSession(["checkpoint suppression"]);
    const overseerTurns: number[] = [];
    mockTransportLifecycle();
    spyMaintenanceTick();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) =>
      chatResult({
        response: "",
        emitted: false,
        turnId: "turn-suppressed-checkpoint",
        sessionId: options.sessionId as SessionId,
        suppressionReason: "commitment_revision_failed",
      }),
    );

    const report = await runSimulation({
      runId: "sim-runner-suppressed-overseer-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 1,
      checkEvery: 1,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
      overseerRunner: async ({ turnCounter }) => {
        overseerTurns.push(turnCounter);
        return healthyOverseerVerdict(turnCounter, ["Suppression checkpoint inspected."]);
      },
    });
    const [metricsRow] = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map(
        (line) =>
          JSON.parse(line) as {
            turn_counter: number;
            overseer_due_on_suppressed_turn: boolean;
          },
      );

    expect(report.overseerCheckpoints).toHaveLength(1);
    expect(overseerTurns).toEqual([1]);
    expect(metricsRow).toMatchObject({
      turn_counter: 1,
      overseer_due_on_suppressed_turn: true,
    });
  });

  it("records checkpoint due on suppressed turns even if the overseer throws", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const persona = fakePersonaSession(["checkpoint suppression"]);
    mockTransportLifecycle();
    spyMaintenanceTick();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) =>
      chatResult({
        response: "",
        emitted: false,
        turnId: "turn-suppressed-overseer-error",
        sessionId: options.sessionId as SessionId,
        suppressionReason: "commitment_revision_failed",
      }),
    );

    await expect(
      runSimulation({
        runId: "sim-runner-suppressed-overseer-error-test",
        persona: tomPersona,
        personaSession: persona.session,
        totalTurns: 1,
        checkEvery: 1,
        metricsPath,
        dataDir: join(dir, "data"),
        tracePath: join(dir, "trace.jsonl"),
        mock: true,
        overseerRunner: async () => {
          throw new Error("overseer failed");
        },
      }),
    ).rejects.toThrow("overseer failed");

    const [metricsRow] = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map(
        (line) =>
          JSON.parse(line) as {
            turn_counter: number;
            overseer_due_on_suppressed_turn: boolean;
          },
      );

    expect(metricsRow).toMatchObject({
      turn_counter: 1,
      overseer_due_on_suppressed_turn: true,
    });
  });

  it("captures session-close action expirations on the session-ending turn metrics row", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const tracePath = join(dir, "trace.jsonl");
    const persona = fakePersonaSession(["natural silence"]);
    const borg = {
      ...fakeSimulatorBorg(),
      endSession: vi.fn((sessionId: SessionId) => {
        appendFileSync(
          tracePath,
          `${JSON.stringify({
            ts: Date.now(),
            turnId: `session_end:${sessionId}`,
            event: "action_session_scope.expired",
            actions_expired_at_session_close: 2,
          })}\n`,
        );
      }),
    } as unknown as Borg & { endSession: ReturnType<typeof vi.fn> };

    vi.spyOn(BorgTransport.prototype, "open").mockResolvedValue(undefined);
    vi.spyOn(BorgTransport.prototype, "close").mockResolvedValue(undefined);
    vi.spyOn(BorgTransport.prototype, "getBorg").mockReturnValue(borg);
    vi.spyOn(BorgTransport.prototype, "resolveEntity").mockReturnValue(createEntityId());
    spyMaintenanceTick();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) =>
      chatResult({
        response: "",
        emitted: false,
        turnId: "turn-session-ending-expiration",
        sessionId: options.sessionId as SessionId,
        suppressionReason: "finalizer_no_output",
      }),
    );

    await runSimulation({
      runId: "sim-runner-session-expiration-metrics-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 1,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath,
      mock: true,
    });

    const [metricsRow] = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map(
        (line) =>
          JSON.parse(line) as {
            actions_expired_at_session_close: number;
          },
      );

    expect(borg.endSession).toHaveBeenCalledOnce();
    expect(metricsRow?.actions_expired_at_session_close).toBe(2);
  });

  it("runs periodic maintenance ticks on cadence in mock mode", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const tickSpy = spyMaintenanceTick();

    await runSimulation({
      runId: "sim-runner-maintenance-test",
      persona: tomPersona,
      totalTurns: 20,
      checkEvery: 999,
      maintenanceEvery: 10,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(tickSpy).toHaveBeenCalledTimes(2);
    expect(tickSpy.mock.calls.map(([cadence]) => cadence)).toEqual(["light", "light"]);
  });

  it("reports final metrics after the final maintenance tick", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const borg = fakeSimulatorBorg();
    let semanticNodeCount = 1;
    let chatCalls = 0;

    vi.spyOn(BorgTransport.prototype, "open").mockResolvedValue(undefined);
    vi.spyOn(BorgTransport.prototype, "close").mockResolvedValue(undefined);
    vi.spyOn(BorgTransport.prototype, "getBorg").mockReturnValue(borg);
    vi.spyOn(BorgTransport.prototype, "resolveEntity").mockReturnValue(createEntityId());
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      chatCalls += 1;

      return chatResult({
        response: "Borg replied.",
        emitted: true,
        turnId: `turn-${chatCalls}`,
        sessionId: options.sessionId as SessionId,
      });
    });
    vi.spyOn(borg.semantic.nodes, "list").mockImplementation(async () =>
      Array.from({ length: semanticNodeCount }, () => ({}) as never),
    );
    vi.spyOn(borg.maintenance.scheduler, "tick").mockImplementation(
      async (cadence: MaintenanceCadence): Promise<MaintenanceTickResult> => {
        semanticNodeCount += 1;

        return {
          status: "ok",
          cadence,
          ts: Date.now(),
          processes: [],
          result: null,
        };
      },
    );

    const report = await runSimulation({
      runId: "sim-runner-final-maintenance-metrics-test",
      persona: tomPersona,
      totalTurns: 2,
      checkEvery: 999,
      maintenanceEvery: 2,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });
    const metricsRows = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map(
        (line) =>
          JSON.parse(line) as {
            semantic_node_count: number;
            semantic_nodes_added_since_last_check: number;
          },
      );

    expect(report.finalMetrics.semantic_node_count).toBe(2);
    expect(report.finalMetrics.semantic_nodes_added_since_last_check).toBe(1);
    expect(metricsRows.at(-1)).toMatchObject({
      semantic_node_count: 2,
      semantic_nodes_added_since_last_check: 1,
    });
  });

  it("passes the persona display name as the stable Borg audience", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const chatSpy = vi.spyOn(BorgTransport.prototype, "chat");
    spyMaintenanceTick();

    await runSimulation({
      runId: "sim-runner-audience-test",
      persona: tomPersona,
      totalTurns: 2,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(chatSpy.mock.calls.map(([, options]) => options?.audience)).toEqual(["Tom", "Tom"]);
  });

  it("round-robins multi-persona channel turns with sender ids and a group audience", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const alice = {
      key: "alice-test",
      displayName: "Alice",
      systemPrompt: "Speak as Alice.",
    };
    const bob = {
      key: "bob-test",
      displayName: "Bob",
      systemPrompt: "Speak as Bob.",
    };
    const aliceSession = fakePersonaSession(["alice first", "alice second"]);
    const bobSession = fakePersonaSession(["bob first"]);
    const { entityIds, resolveEntitySpy } = mockTransportLifecycle();
    let chatCalls = 0;
    const chatSpy = vi
      .spyOn(BorgTransport.prototype, "chat")
      .mockImplementation(async (_message, options = {}) => {
        chatCalls += 1;

        return chatResult({
          response: `Borg reply ${chatCalls}`,
          emitted: true,
          turnId: `turn-group-${chatCalls}`,
          sessionId: options.sessionId as SessionId,
        });
      });

    const report = await runSimulation({
      runId: "sim-runner-multi-persona-test",
      persona: alice,
      personas: [alice, bob],
      personaSessions: [aliceSession.session, bobSession.session],
      channelName: "Planning Channel",
      totalTurns: 3,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(chatSpy.mock.calls.map(([message]) => message)).toEqual([
      "alice first",
      "bob first",
      "alice second",
    ]);
    expect(chatSpy.mock.calls.map(([, options]) => options?.audience)).toEqual([
      "Planning Channel",
      "Planning Channel",
      "Planning Channel",
    ]);
    expect(chatSpy.mock.calls.map(([, options]) => options?.senderEntityId)).toEqual([
      entityIds.get("Alice"),
      entityIds.get("Bob"),
      entityIds.get("Alice"),
    ]);
    expect(resolveEntitySpy).toHaveBeenCalledWith("Planning Channel", {
      kind: "group",
      provenance: "transport_audience_label",
    });
    expect(report.personas).toEqual(["alice-test", "bob-test"]);
    expect(report.audience).toBe("Planning Channel");
  });

  it("keeps observed multi-persona turns in session and threads peer transcript", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const alice = {
      key: "alice-test",
      displayName: "Alice",
      systemPrompt: "Speak as Alice.",
    };
    const bob = {
      key: "bob-test",
      displayName: "Bob",
      systemPrompt: "Speak as Bob.",
    };
    const aliceSession = fakePersonaSession(["alice first", "alice second", "alice third"]);
    const bobSession = fakePersonaSession(["bob first", "bob second"]);
    const chatSessionIds: SessionId[] = [];
    mockTransportLifecycle();
    spyMaintenanceTick();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      const callNumber = chatSessionIds.length + 1;
      const sessionId = options.sessionId as SessionId;
      chatSessionIds.push(sessionId);

      if (callNumber === 2 || callNumber === 4) {
        return chatResult({
          response: `Borg answer ${callNumber}`,
          emitted: true,
          turnId: `turn-peer-${callNumber}`,
          sessionId,
        });
      }

      return chatResult({
        response: "[borg observation: The participants are coordinating directly.]",
        emitted: false,
        observedReason: "The participants are coordinating directly.",
        turnId: `turn-peer-${callNumber}`,
        sessionId,
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-observe-peer-transcript-test",
      persona: alice,
      personas: [alice, bob],
      personaSessions: [aliceSession.session, bobSession.session],
      channelName: "Planning Channel",
      totalTurns: 5,
      checkEvery: 999,
      maxSessions: 3,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(new Set(chatSessionIds).size).toBe(1);
    expect(aliceSession.startNewSession).not.toHaveBeenCalled();
    expect(bobSession.startNewSession).not.toHaveBeenCalled();
    expect(report.sessions).toEqual([
      {
        sessionIndex: 0,
        sessionId: chatSessionIds[0],
        startedAtTurn: 1,
        endedAtTurn: 5,
        endReason: "run_complete",
      },
    ]);
    expect(report.suppressionEvents).toEqual([]);
    expect(aliceSession.prepareNextTurn.mock.calls.map(([previous]) => previous)).toEqual([
      { kind: "new_session" },
      {
        kind: "normal",
        text: "Borg answer 2",
        channelTranscript: [
          { speaker_display_name: "Bob", text: "bob first" },
          { speaker_display_name: "Borg", text: "Borg answer 2" },
        ],
      },
      {
        kind: "normal",
        text: "Borg answer 4",
        channelTranscript: [
          { speaker_display_name: "Bob", text: "bob second" },
          { speaker_display_name: "Borg", text: "Borg answer 4" },
        ],
      },
    ]);
    expect(bobSession.prepareNextTurn.mock.calls.map(([previous]) => previous)).toEqual([
      {
        kind: "new_session",
        channelTranscript: [{ speaker_display_name: "Alice", text: "alice first" }],
      },
      {
        kind: "normal",
        text: "Borg answer 2",
        channelTranscript: [
          { speaker_display_name: "Borg", text: "Borg answer 2" },
          { speaker_display_name: "Alice", text: "alice second" },
        ],
      },
    ]);
    expect(
      JSON.stringify([
        ...aliceSession.prepareNextTurn.mock.calls,
        ...bobSession.prepareNextTurn.mock.calls,
      ]),
    ).not.toContain("[borg observation:");
  });

  it("passes the persona display name as Borg defaultUser when opening", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const openSpy = vi.spyOn(Borg, "open").mockResolvedValue(fakeSimulatorBorg());
    vi.spyOn(BorgTransport.prototype, "resolveEntity").mockReturnValue(createEntityId());
    vi.spyOn(BorgTransport.prototype, "chat").mockResolvedValue({
      response: "Mock response",
      emitted: true,
      emission: undefined as never,
      turnId: "turn-default-user",
      usage: {
        input_tokens: 0,
        output_tokens: 0,
      },
      moodAfter: {
        valence: 0,
        arousal: 0,
      },
      toolCalls: [],
    });

    await runSimulation({
      runId: "sim-runner-default-user-test",
      persona: tomPersona,
      totalTurns: 1,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(openSpy.mock.calls[0]?.[0]?.config?.defaultUser).toBe("Tom");
  });

  it("omits Borg defaultUser when opening multi-persona group runs", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const alice = {
      key: "alice-test",
      displayName: "Alice",
      systemPrompt: "Speak as Alice.",
    };
    const bob = {
      key: "bob-test",
      displayName: "Bob",
      systemPrompt: "Speak as Bob.",
    };
    const aliceSession = fakePersonaSession(["alice first"]);
    const bobSession = fakePersonaSession(["bob first"]);
    const openSpy = vi.spyOn(Borg, "open").mockResolvedValue(fakeSimulatorBorg());
    vi.spyOn(BorgTransport.prototype, "resolveEntity").mockReturnValue(createEntityId());
    vi.spyOn(BorgTransport.prototype, "chat").mockResolvedValue({
      response: "Mock response",
      emitted: true,
      emission: undefined as never,
      turnId: "turn-group-default-user",
      usage: {
        input_tokens: 0,
        output_tokens: 0,
      },
      moodAfter: {
        valence: 0,
        arousal: 0,
      },
      toolCalls: [],
    });

    await runSimulation({
      runId: "sim-runner-group-default-user-test",
      persona: alice,
      personas: [alice, bob],
      personaSessions: [aliceSession.session, bobSession.session],
      channelName: "Planning Channel",
      totalTurns: 1,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(openSpy.mock.calls[0]?.[0]?.config?.defaultUser).toBeUndefined();
  });

  it("passes normal prior Borg output to the next persona turn", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const persona = fakePersonaSession(["first persona turn", "second persona turn"]);
    let chatCalls = 0;
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      chatCalls += 1;

      return chatResult({
        response: chatCalls === 1 ? "First Borg reply." : "Second Borg reply.",
        emitted: true,
        turnId: `turn-normal-${chatCalls}`,
        sessionId: options.sessionId as SessionId,
      });
    });

    await runSimulation({
      runId: "sim-runner-normal-prior-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 2,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(persona.prepareNextTurn.mock.calls.map(([previous]) => previous)).toEqual([
      { kind: "new_session" },
      { kind: "normal", text: "First Borg reply." },
    ]);
  });

  it("passes distinct session IDs after finalizer_no_output rotation and records them", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const chatSessionIds: SessionId[] = [];
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      const sessionId = options.sessionId as SessionId;
      chatSessionIds.push(sessionId);
      const emitted = chatSessionIds.length > 1;

      return chatResult({
        response: emitted ? "Second session response" : "",
        emitted,
        turnId: `turn-${chatSessionIds.length}`,
        sessionId,
        suppressionReason: "finalizer_no_output",
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-session-id-test",
      persona: tomPersona,
      totalTurns: 2,
      checkEvery: 999,
      maxSessions: 3,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(chatSessionIds).toHaveLength(2);
    expect(chatSessionIds[0]).toMatch(/^sess_[a-z0-9]{16}$/);
    expect(chatSessionIds[1]).toMatch(/^sess_[a-z0-9]{16}$/);
    expect(chatSessionIds[0]).not.toBe(chatSessionIds[1]);
    expect(report.sessions).toEqual([
      {
        sessionIndex: 0,
        sessionId: chatSessionIds[0],
        startedAtTurn: 1,
        endedAtTurn: 1,
        endReason: "suppression",
        suppressionReason: "finalizer_no_output",
      },
      {
        sessionIndex: 1,
        sessionId: chatSessionIds[1],
        startedAtTurn: 2,
        endedAtTurn: 2,
        endReason: "run_complete",
      },
    ]);
  });

  it("continues the same session after a guard-driven suppression", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const persona = fakePersonaSession(["first persona turn", "second persona turn"]);
    const chatSessionIds: SessionId[] = [];
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      const sessionId = options.sessionId as SessionId;
      chatSessionIds.push(sessionId);
      const emitted = chatSessionIds.length > 1;

      return chatResult({
        response: emitted ? "Borg replied." : "",
        emitted,
        turnId: `turn-${chatSessionIds.length}`,
        sessionId,
        suppressionReason: "commitment_revision_failed",
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-guard-suppression-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 2,
      checkEvery: 999,
      maxSessions: 3,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(chatSessionIds).toHaveLength(2);
    expect(chatSessionIds[0]).toBe(chatSessionIds[1]);
    expect(persona.prepareNextTurn).toHaveBeenCalledTimes(2);
    expect(persona.prepareNextTurn.mock.calls.map(([previous]) => previous)).toEqual([
      { kind: "new_session" },
      { kind: "continued_suppression", reason: "commitment_revision_failed" },
    ]);
    expect(persona.startNewSession).not.toHaveBeenCalled();
    expect(report.resultState).toBe("completed");
    expect(report.sessions).toHaveLength(1);
    expect(report.sessions[0]).toMatchObject({
      sessionIndex: 0,
      sessionId: chatSessionIds[0],
      startedAtTurn: 1,
      endedAtTurn: 2,
      endReason: "run_complete",
    });
    expect(report.suppressionEvents).toEqual([
      {
        sessionIndex: 0,
        sessionId: chatSessionIds[0],
        turn: 1,
        reason: "commitment_revision_failed",
      },
    ]);
    expect(report.borgBehavioralSuppressions).toEqual([
      {
        sessionIndex: 0,
        sessionId: chatSessionIds[0],
        turn: 1,
        reason: "commitment_revision_failed",
        sessionContinued: true,
      },
    ]);
    expect(report.finalMetrics.borg_hard_aborted_turns).toBe(0);
    expect(report.finalMetrics.borg_aborted_turns).toBe(0);
    expect(report.finalMetrics.borg_intentional_suppressions).toBe(1);
    expect(report.finalMetrics.borg_intentional_suppressions_by_reason).toEqual({
      commitment_revision_failed: 1,
    });
    expect(report.finalMetrics.simulator_persona_failures).toBe(0);
  });

  it("uses post-generation rejection trace reason for intentional suppression metrics", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const tracePath = join(dir, "trace.jsonl");
    const persona = fakePersonaSession(["trace reason turn"]);
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      appendFileSync(
        tracePath,
        `${JSON.stringify({
          ts: Date.now(),
          turnId: "turn-trace-suppression",
          event: "post_generation.rejected",
          reason: "active_discourse_stop",
        })}\n`,
      );

      return chatResult({
        response: "",
        emitted: false,
        turnId: "turn-trace-suppression",
        sessionId: options.sessionId as SessionId,
        suppressionReason: "generation_gate",
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-trace-suppression-reason-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 1,
      checkEvery: 999,
      maxSessions: 3,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath,
      mock: true,
    });

    expect(report.finalMetrics.borg_intentional_suppressions).toBe(1);
    expect(report.finalMetrics.borg_intentional_suppressions_by_reason).toEqual({
      active_discourse_stop: 1,
    });
  });

  it("rejects a persona role-bleed draft, regenerates, and records a trace artifact", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const tracePath = join(dir, "trace.jsonl");
    const rejectedDraft = "I'm Claude. I was playing Tom inside the fiction.";
    const persona = fakePersonaSession([
      rejectedDraft,
      "Are you still there? I was asking because this still feels tangled.",
    ]);
    const seenMessages: string[] = [];
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (message, options = {}) => {
      seenMessages.push(message);

      return chatResult({
        response: "Borg replied.",
        emitted: true,
        turnId: "turn-role-bleed-recovered",
        sessionId: options.sessionId as SessionId,
      });
    });

    await runSimulation({
      runId: "sim-runner-persona-role-bleed-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 1,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath,
      mock: true,
    });
    const roleBleedEvents = readTraceEvents(tracePath).filter(
      (event) => event.event === "persona.role_bleed.rejected",
    );

    expect(seenMessages).toEqual([
      "Are you still there? I was asking because this still feels tangled.",
    ]);
    expect(persona.rollback).toHaveBeenCalledOnce();
    expect(persona.prepareNextTurn.mock.calls.map(([previous]) => previous)).toEqual([
      { kind: "new_session" },
      { kind: "new_session", retry: "persona_role_bleed" },
    ]);
    expect(roleBleedEvents).toMatchObject([
      {
        event: "persona.role_bleed.rejected",
        artifact: "simulator",
        prior_kind: "new_session",
        matched_patterns: ["i'm claude", "i was playing tom", "inside the fiction"],
        rejected_preview: rejectedDraft,
        action: "regenerated",
      },
    ]);
  });

  it("reports persona draft failures before Borg runs as simulator persona failures", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const tracePath = join(dir, "trace.jsonl");
    let calls = 0;
    const prepareNextTurn = vi.fn(async () => {
      calls += 1;

      if (calls === 1) {
        throw new Error("Persona LLM produced malformed content");
      }

      return {
        kind: "mock",
        message: "Can we return to the design doc?",
        history: null,
        mockIndex: null,
      };
    });
    const commit = vi.fn();
    const rollback = vi.fn();
    const startNewSession = vi.fn();
    const seenMessages: string[] = [];
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (message, options = {}) => {
      seenMessages.push(message);

      return chatResult({
        response: "Borg replied.",
        emitted: true,
        turnId: "turn-after-persona-failure",
        sessionId: options.sessionId as SessionId,
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-persona-malformed-failure-test",
      persona: tomPersona,
      personaSession: {
        prepareNextTurn,
        commit,
        rollback,
        startNewSession,
      } as unknown as PersonaSession,
      totalTurns: 2,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath,
      mock: true,
    });

    expect(seenMessages).toEqual(["Can we return to the design doc?"]);
    expect(report.turnFailures).toEqual([]);
    expect(report.simulatorPersonaFailures).toEqual([
      {
        turn: 1,
        error: "persona_malformed: Persona LLM produced malformed content",
        attempts: 0,
      },
    ]);
    expect(report.finalMetrics.simulator_persona_failures).toBe(1);
    expect(report.finalMetrics.borg_aborted_turns).toBe(0);
  });

  it("aborts a turn when the role-bleed retry also bleeds", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const tracePath = join(dir, "trace.jsonl");
    const firstBleedDraft = `I'm Claude. ${"x".repeat(600)}`;
    const secondBleedDraft = "I have been playing Tom for this exchange.";
    const cleanDraft = "Can we talk about the design doc again?";
    const drafts = [firstBleedDraft, secondBleedDraft, cleanDraft];
    let draftIndex = 0;
    const prepareNextTurn = vi.fn(async (_priorBorgTurn: PriorBorgTurn) => {
      const message = drafts[draftIndex] ?? cleanDraft;
      draftIndex += 1;
      return {
        kind: "mock",
        message,
        history: null,
        mockIndex: null,
      };
    });
    const commit = vi.fn();
    const rollback = vi.fn();
    const startNewSession = vi.fn();
    const seenMessages: string[] = [];
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (message, options = {}) => {
      seenMessages.push(message);

      return chatResult({
        response: "Borg replied.",
        emitted: true,
        turnId: "turn-after-role-bleed-abort",
        sessionId: options.sessionId as SessionId,
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-persona-role-bleed-abort-test",
      persona: tomPersona,
      personaSession: {
        prepareNextTurn,
        commit,
        rollback,
        startNewSession,
      } as unknown as PersonaSession,
      totalTurns: 2,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath,
      mock: true,
    });
    const roleBleedEvents = readTraceEvents(tracePath).filter(
      (event) => event.event === "persona.role_bleed.rejected",
    );

    expect(seenMessages).toEqual([cleanDraft]);
    expect(rollback).toHaveBeenCalledTimes(2);
    expect(prepareNextTurn.mock.calls.map(([previous]) => previous)).toEqual([
      { kind: "new_session" },
      { kind: "new_session", retry: "persona_role_bleed" },
      { kind: "new_session" },
    ]);
    expect(report.turnFailures).toEqual([]);
    expect(report.simulatorPersonaFailures).toEqual([
      {
        turn: 1,
        error: "persona_role_bleed: i have been playing tom",
        attempts: 0,
      },
    ]);
    expect(report.finalMetrics.simulator_persona_failures).toBe(1);
    expect(report.finalMetrics.borg_aborted_turns).toBe(0);
    expect(roleBleedEvents).toMatchObject([
      {
        event: "persona.role_bleed.rejected",
        matched_patterns: ["i'm claude"],
        rejected_preview: firstBleedDraft.slice(0, 500),
        attempt: 1,
        action: "regenerated",
      },
      {
        event: "persona.role_bleed.rejected",
        matched_patterns: ["i have been playing tom"],
        rejected_preview: secondBleedDraft,
        attempt: 2,
        action: "aborted",
      },
    ]);
    expect((roleBleedEvents[0] as { rejected_preview?: string }).rejected_preview).toHaveLength(
      500,
    );
  });

  it("starts a new session after finalizer_no_output suppression", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const persona = fakePersonaSession(["first persona turn", "second persona turn"]);
    const chatSessionIds: SessionId[] = [];
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      const sessionId = options.sessionId as SessionId;
      chatSessionIds.push(sessionId);
      const emitted = chatSessionIds.length > 1;

      return chatResult({
        response: emitted ? "Borg replied." : "",
        emitted,
        turnId: `turn-${chatSessionIds.length}`,
        sessionId,
        suppressionReason: "finalizer_no_output",
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-no-output-suppression-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 2,
      checkEvery: 999,
      maxSessions: 3,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(chatSessionIds).toHaveLength(2);
    expect(chatSessionIds[0]).not.toBe(chatSessionIds[1]);
    expect(persona.prepareNextTurn).toHaveBeenCalledTimes(2);
    expect(persona.prepareNextTurn.mock.calls.map(([previous]) => previous)).toEqual([
      { kind: "new_session" },
      {
        kind: "new_session",
        gapContext: "It's the next evening. You're back on the couch after dinner.",
      },
    ]);
    expect(persona.startNewSession).toHaveBeenCalledTimes(1);
    expect(persona.startNewSession.mock.calls[0]).toEqual([]);
    expect(report.sessions).toHaveLength(2);
    expect(report.sessions[0]).toMatchObject({
      sessionIndex: 0,
      sessionId: chatSessionIds[0],
      startedAtTurn: 1,
      endedAtTurn: 1,
      endReason: "suppression",
      suppressionReason: "finalizer_no_output",
    });
    expect(report.sessions[1]).toMatchObject({
      sessionIndex: 1,
      sessionId: chatSessionIds[1],
      startedAtTurn: 2,
      endedAtTurn: 2,
      endReason: "run_complete",
    });
    expect(report.suppressionEvents).toEqual([]);
    expect(report.borgBehavioralSuppressions).toEqual([
      {
        sessionIndex: 0,
        sessionId: chatSessionIds[0],
        turn: 1,
        reason: "finalizer_no_output",
        sessionContinued: false,
      },
    ]);
    expect(report.finalMetrics.borg_hard_aborted_turns).toBe(0);
    expect(report.finalMetrics.borg_aborted_turns).toBe(0);
    expect(report.finalMetrics.borg_intentional_suppressions).toBe(1);
    expect(report.finalMetrics.borg_intentional_suppressions_by_reason).toEqual({
      finalizer_no_output: 1,
    });
  });

  it("ends simulator sessions before starting the next session", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const persona = fakePersonaSession(["first persona turn", "second persona turn"]);
    const chatSessionIds: SessionId[] = [];
    const endSession = vi.fn();
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "getBorg").mockReturnValue({
      ...fakeSimulatorBorg(),
      endSession,
    } as unknown as Borg);
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      const sessionId = options.sessionId as SessionId;
      chatSessionIds.push(sessionId);
      const emitted = chatSessionIds.length > 1;

      return chatResult({
        response: emitted ? "Borg replied." : "",
        emitted,
        turnId: `turn-end-session-${chatSessionIds.length}`,
        sessionId,
        suppressionReason: "finalizer_no_output",
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-end-session-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 2,
      checkEvery: 999,
      maxSessions: 3,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(report.sessions).toHaveLength(2);
    expect(endSession).toHaveBeenCalledTimes(2);
    expect(endSession.mock.calls[0]?.[0]).toBe(chatSessionIds[0]);
    expect(endSession.mock.calls[0]?.[1]).toMatchObject({
      nextSessionId: chatSessionIds[1],
    });
    expect(endSession.mock.calls[1]).toEqual([chatSessionIds[1], {}]);
    expect(persona.startNewSession).toHaveBeenCalledTimes(1);
    expect(endSession.mock.invocationCallOrder[0]).toBeLessThan(
      persona.startNewSession.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER,
    );
  });

  it("rotates sessions when Borg suppresses a turn and stops at maxSessions", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    spyMaintenanceTick();
    const report = await runSimulation({
      runId: "sim-runner-suppression-test",
      persona: tomPersona,
      totalTurns: 5,
      checkEvery: 999,
      maxSessions: 1,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "toolu_plan",
                name: "EmitTurnPlan",
                input: {
                  uncertainty: "",
                  verification_steps: [],
                  tensions: [],
                  voice_note: "",
                  intents: [],
                },
              },
            ],
          },
          {
            text: "",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "toolu_emit_no_output",
                name: "EmitNoOutput",
                input: { reason: "No assistant message is needed." },
              },
            ],
          },
        ],
      }),
    });
    const metricsRows = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line) as { turn_counter: number });

    expect(report.resultState).toBe("max_sessions_reached");
    expect(report.sessions).toHaveLength(1);
    expect(report.sessions[0]).toMatchObject({
      sessionIndex: 0,
      startedAtTurn: 1,
      endedAtTurn: 1,
      endReason: "suppression",
    });
    expect(metricsRows).toHaveLength(1);
    expect(metricsRows[0]?.turn_counter).toBe(1);
  });
});
