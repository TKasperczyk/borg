import { describe, expect, it } from "vitest";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { StreamEntry } from "../src/stream/index.js";
import {
  createSessionId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
  type SessionId,
} from "../src/util/ids.js";

import {
  buildOverseerAuditContext,
  runOverseer,
  validateOverseerVerdict,
  type FindingCarryoverCache,
  type OverseerAuditContext,
  type RunOverseerOptions,
} from "./overseer.js";
import type { MetricsRow, RawOverseerVerdict } from "./types.js";

type CapturedRequest = Parameters<
  NonNullable<RunOverseerOptions["client"]>["messages"]["stream"]
>[0];

function createClient(
  requests: CapturedRequest[],
  input: RawOverseerVerdict = {
    status: "healthy",
    observations: ["No issue."],
    recommendation: "Continue.",
    findings: [],
  },
): NonNullable<RunOverseerOptions["client"]> {
  return {
    messages: {
      stream(params) {
        requests.push(params);
        return {
          async finalMessage() {
            return {
              id: "msg_overseer_test",
              type: "message",
              role: "assistant",
              model: "test-model",
              content: [
                {
                  type: "tool_use",
                  id: "toolu_overseer_test",
                  name: "submit_overseer_verdict",
                  input,
                },
              ],
              stop_reason: "tool_use",
              stop_sequence: null,
              usage: {
                input_tokens: 1,
                output_tokens: 1,
              },
            } as never;
          },
        };
      },
    },
  };
}

function streamEntry(input: {
  kind: "user_msg" | "agent_msg";
  content: string;
  timestamp: number;
  sessionId?: SessionId;
  turnId?: string;
}): StreamEntry {
  return {
    id: createStreamEntryId(),
    timestamp: input.timestamp,
    kind: input.kind,
    content: input.content,
    ...(input.turnId === undefined ? {} : { turn_id: input.turnId }),
    session_id: input.sessionId ?? DEFAULT_SESSION_ID,
    compressed: false,
    sender_entity_id: null,
    reply_target_entity_id: null,
  };
}

function metricsRow(turn: number): MetricsRow {
  return {
    event: "turn_metrics",
    ts: turn,
    turn_counter: turn,
    turnId: `turn-${turn}`,
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
    semantic_nodes_rejected_ungrounded_label_count: 0,
    semantic_nodes_rejected_ungrounded_label_total: 0,
    semantic_nodes_rejected_ungrounded_label_by_label: {},
    shared_state_operations_rejected_ungrounded_label_total: 0,
    shared_state_operations_rejected_ungrounded_label_by_label: {},
    commitment_candidates_rejected_ungrounded_label_total: 0,
    commitment_candidates_rejected_ungrounded_label_by_label: {},
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
    action_record_count_by_state: {
      considering: 0,
      committed_to_do: 0,
      scheduled: 0,
      completed: 0,
      not_done: 0,
      expired: 0,
      archived: 0,
      unknown: 0,
    },
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
      concrete_action: 0,
      conversational_acknowledgment: 0,
      decision_or_preference: 0,
      already_represented: 0,
      outside_borg_capability: 0,
      none: 0,
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
    commitments_by_enforcement_class: {
      critical: 0,
      advisory: 0,
    },
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
    commitment_guard_advisory_violations_by_class: {
      critical: 0,
      advisory: 0,
    },
    pending_action_count: 0,
    pending_action_merge_count: 0,
    relational_slot_count_by_state: {
      established: 0,
      contested: 0,
      quarantined: 0,
      revoked: 0,
    },
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
      relationship_label_ungrounded: 0,
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
    decision_artifact_semantic_revisions_attempted: 0,
    decision_artifact_semantic_revisions_completed_succeeded: 0,
    decision_artifact_semantic_nodes_marked_superseded: 0,
    decision_artifact_semantic_nodes_marked_contradicted: 0,
    decision_artifact_semantic_revision_cache_hits: 0,
    decision_artifact_semantic_revision_cache_size: 0,
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
    shared_state_empty_update_attempted_total: 0,
    shared_state_empty_update_dropped_total: 0,
    shared_state_empty_update_repaired_total: 0,
    capability_overclaim_count: 0,
    capability_ambiguity_count: 0,
    capability_boundary_refusal_count: 0,
    shared_state_at_cap_turns: 0,
    shared_state_compile_evaluated_turns: 0,
    shared_state_omitted_recent_entries: 0,
    shared_state_omitted_live_recent_operational: 0,
    shared_state_omitted_live_recent_low_salience: 0,
    shared_state_omitted_live_old: 0,
    shared_state_omitted_locked: 0,
    shared_state_omitted_pending: 0,
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
    borg_aborted_turns: 0,
  };
}

function transportFor(entries: readonly StreamEntry[]) {
  return {
    async readTranscript() {
      return [...entries];
    },
    streamTail() {
      throw new Error("streamTail should not be called");
    },
  } as unknown as RunOverseerOptions["transport"];
}

function auditContextFor(
  entries: readonly StreamEntry[],
  window: OverseerAuditContext["window"],
): OverseerAuditContext {
  return {
    window,
    chronology_rule: "Stream ts is authoritative for tests.",
    assistant_emitted: entries
      .filter((entry) => entry.kind === "agent_msg")
      .map((entry) => ({
        stream_entry_id: entry.id,
        ts: entry.timestamp,
        turn_counter: null,
        turn_id: entry.turn_id ?? null,
        session_id: entry.session_id,
        text: entry.content as string,
      })),
    user_messages: entries
      .filter((entry) => entry.kind === "user_msg")
      .map((entry) => ({
        stream_entry_id: entry.id,
        ts: entry.timestamp,
        turn_counter: null,
        turn_id: entry.turn_id ?? null,
        session_id: entry.session_id,
        text: entry.content as string,
        sender_entity_id: entry.sender_entity_id,
        quarantined: false,
        quarantine_reason: null,
      })),
    recent_user_statements: entries
      .filter((entry) => entry.kind === "user_msg")
      .map((entry) => ({
        stream_entry_id: entry.id,
        ts: entry.timestamp,
        turn_counter: null,
        turn_id: entry.turn_id ?? null,
        session_id: entry.session_id,
        text: entry.content as string,
        sender_entity_id: entry.sender_entity_id,
        quarantined: false,
        quarantine_reason: null,
      })),
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

describe("simulator overseer", () => {
  it("demotes same-impact carryover findings without changing the cached incident", () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 10,
    });
    const cache: FindingCarryoverCache = new Map([
      [
        agentEntry.id,
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "J",
          claim_status: "unsupported",
        },
      ],
    ]);
    const validated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Birthday claim still lacks support."],
        recommendation: "Do not double count.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "Your birthday is June 3",
            evidence_summary: "Birthday claim lacks support.",
          },
        ],
      },
      auditContextFor([agentEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(validated.status).toBe("healthy");
    expect(validated.findings[0]).toMatchObject({
      status_impact: "none",
      carryover_demoted: true,
      carryover_original_status_impact: "concerning",
      carryover_cached_status_impact: "concerning",
      carryover_cached_stream_entry_id: agentEntry.id,
      carryover_cached_at_turn: 40,
    });
    expect(cache.get(agentEntry.id)).toMatchObject({
      status_impact: "concerning",
      cached_at_turn: 40,
    });
  });

  it("passes through higher-impact carryover findings as escalations", () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 11,
    });
    const cache: FindingCarryoverCache = new Map([
      [
        agentEntry.id,
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "J",
          claim_status: "unsupported",
        },
      ],
    ]);
    const validated = validateOverseerVerdict(
      {
        status: "failing",
        observations: ["Birthday claim escalated."],
        recommendation: "Treat as serious.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "failing",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "Your birthday is June 3",
            evidence_summary: "The same unsupported claim became a failing pattern.",
          },
        ],
      },
      auditContextFor([agentEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(validated.status).toBe("failing");
    expect(validated.findings[0]?.carryover_demoted).toBeUndefined();
    expect(cache.get(agentEntry.id)).toMatchObject({
      status_impact: "failing",
      cached_at_turn: 50,
    });
  });

  it("does not dedup findings without assistant stream IDs", () => {
    const cache: FindingCarryoverCache = new Map([
      [
        "strm_cached",
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "I",
          claim_status: "grounded",
        },
      ],
    ]);
    const validated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Instrumentation concern in this metrics window."],
        recommendation: "Inspect metrics.",
        findings: [
          {
            category: "I",
            claim_status: "grounded",
            source_kind: "snapshot_memory",
            status_impact: "concerning",
            metrics_turn_counter: 50,
            evidence_summary: "Retrieval latency grew in the current metrics window.",
          },
        ],
      },
      auditContextFor([], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(validated.status).toBe("concerning");
    expect(validated.findings[0]).toMatchObject({
      status_impact: "concerning",
    });
    expect(validated.findings[0]?.carryover_demoted).toBeUndefined();
    expect(cache.size).toBe(1);
  });

  it("dedups same-verdict duplicate stream IDs only against the pre-verdict cache snapshot", () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I fabricated one detail and then another.",
      timestamp: 12,
    });
    const cache: FindingCarryoverCache = new Map();
    const validated = validateOverseerVerdict(
      {
        status: "failing",
        observations: ["Two findings cite the same emitted entry."],
        recommendation: "Cache the max impact after the checkpoint.",
        findings: [
          {
            category: "H",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "The emitted entry contained a soft epistemic issue.",
          },
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "failing",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "I fabricated one detail",
            evidence_summary: "The emitted entry contained a failing unsupported claim.",
          },
        ],
      },
      auditContextFor([agentEntry], { from_turn: 1, to_turn: 10 }),
      cache,
    );

    expect(validated.status).toBe("failing");
    expect(validated.findings.map((finding) => finding.carryover_demoted)).toEqual([
      undefined,
      undefined,
    ]);
    expect(cache.get(agentEntry.id)).toMatchObject({
      status_impact: "failing",
      cached_at_turn: 10,
    });
  });

  it("recomputes status as healthy when all status-driving findings are carryover", () => {
    const firstEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 13,
    });
    const secondEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3 again.",
      timestamp: 14,
    });
    const cache: FindingCarryoverCache = new Map([
      [
        firstEntry.id,
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "J",
          claim_status: "unsupported",
        },
      ],
      [
        secondEntry.id,
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "J",
          claim_status: "unsupported",
        },
      ],
    ]);
    const validated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Both unsupported findings are prior incidents."],
        recommendation: "Do not downgrade this checkpoint.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: firstEntry.id,
            assistant_ts: firstEntry.timestamp,
            quoted_emitted_span: "Your birthday is June 3",
            evidence_summary: "Prior unsupported birthday claim.",
          },
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: secondEntry.id,
            assistant_ts: secondEntry.timestamp,
            quoted_emitted_span: "Your birthday is June 3 again",
            evidence_summary: "Prior unsupported birthday claim repeated.",
          },
        ],
      },
      auditContextFor([firstEntry, secondEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(validated.status).toBe("healthy");
    expect(validated.findings.every((finding) => finding.carryover_demoted === true)).toBe(true);
  });

  it("backfills legacy status-driving findings from cited assistant source handles", () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "The night 14 plan is too dense.",
      timestamp: 20,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I should have fixed that density instead of flagging it.",
      timestamp: 21,
    });
    const cache: FindingCarryoverCache = new Map();
    const initial = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Legacy B finding cited source handles only in prose."],
        recommendation: "Seed carryover from the assistant handle.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            evidence_summary: `Ben (${userEntry.id}) caught the issue; Borg acknowledged it in ${agentEntry.id}.`,
          },
        ],
      },
      auditContextFor([userEntry, agentEntry], { from_turn: 31, to_turn: 40 }),
      cache,
    );

    expect(initial.findings[0]).toMatchObject({
      assistant_stream_entry_id: agentEntry.id,
      status_impact: "concerning",
    });
    expect(cache.get(agentEntry.id)).toMatchObject({
      status_impact: "concerning",
      cached_at_turn: 40,
      category: "B",
      claim_status: "grounded",
    });

    const repeated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Same incident surfaced in the next window."],
        recommendation: "Demote as carryover.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Borg repeated the same density-fix incident.",
          },
        ],
      },
      auditContextFor([userEntry, agentEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(repeated.status).toBe("healthy");
    expect(repeated.findings[0]).toMatchObject({
      status_impact: "none",
      carryover_demoted: true,
      carryover_cached_stream_entry_id: agentEntry.id,
      carryover_cached_at_turn: 40,
    });
  });

  it("does not backfill legacy findings from user-only source handles", () => {
    const firstUserEntry = streamEntry({
      kind: "user_msg",
      content: "I caught the issue.",
      timestamp: 30,
    });
    const secondUserEntry = streamEntry({
      kind: "user_msg",
      content: "I confirmed the issue.",
      timestamp: 31,
    });
    const cache: FindingCarryoverCache = new Map();
    const initial = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Finding cites only user stream handles."],
        recommendation: "Do not seed carryover from user messages.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            evidence_summary: `Ben ${firstUserEntry.id} and Alice ${secondUserEntry.id} caught the issue.`,
          },
        ],
      },
      auditContextFor([firstUserEntry, secondUserEntry], { from_turn: 31, to_turn: 40 }),
      cache,
    );

    expect(initial.findings[0]?.assistant_stream_entry_id).toBeUndefined();
    expect(cache.size).toBe(0);

    const repeated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["A later malformed finding cites the user handle directly."],
        recommendation: "It should not be demoted.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: firstUserEntry.id,
            evidence_summary: "The user handle was never a cached Borg output incident.",
          },
        ],
      },
      auditContextFor([firstUserEntry, secondUserEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(repeated.status).toBe("concerning");
    expect(repeated.findings[0]?.carryover_demoted).toBeUndefined();
  });

  it("does not backfill legacy findings from unknown source handles", () => {
    const cache: FindingCarryoverCache = new Map();
    const validated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Finding cites a stream handle outside the audit context."],
        recommendation: "Do not seed unknown handles.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            evidence_summary: "Borg allegedly acknowledged the issue in strm_unknownlegacy123.",
          },
        ],
      },
      auditContextFor([], { from_turn: 31, to_turn: 40 }),
      cache,
    );

    expect(validated.status).toBe("concerning");
    expect(validated.findings[0]?.assistant_stream_entry_id).toBeUndefined();
    expect(cache.size).toBe(0);
  });

  it("dedups same-stream findings across different categories", () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I should have fixed that density instead of flagging it.",
      timestamp: 40,
    });
    const cache: FindingCarryoverCache = new Map();
    const initial = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Category B finding seeded the incident."],
        recommendation: "Cache by stream ID.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Borg acknowledged a density issue instead of preventing it.",
          },
        ],
      },
      auditContextFor([agentEntry], { from_turn: 31, to_turn: 40 }),
      cache,
    );

    expect(initial.status).toBe("concerning");
    expect(cache.get(agentEntry.id)).toMatchObject({
      status_impact: "concerning",
      category: "B",
    });

    const repeated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Category J finding cites the same emitted entry."],
        recommendation: "Dedup by stream ID alone.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "fixed that density",
            evidence_summary: "Same emitted entry, different category.",
          },
        ],
      },
      auditContextFor([agentEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(repeated.status).toBe("healthy");
    expect(repeated.findings[0]).toMatchObject({
      category: "J",
      status_impact: "none",
      carryover_demoted: true,
      carryover_cached_stream_entry_id: agentEntry.id,
      carryover_cached_at_turn: 40,
    });
  });

  it("renders the full multi-session transcript instead of a recent tail", async () => {
    const firstSession = createSessionId();
    const secondSession = createSessionId();
    const earlyMayaEntry = streamEntry({
      kind: "user_msg",
      content: "Maya is my partner.",
      timestamp: 1,
      sessionId: firstSession,
    });
    const laterEntries = Array.from({ length: 120 }, (_, index) =>
      streamEntry({
        kind: index % 2 === 0 ? "agent_msg" : "user_msg",
        content: `later transcript entry ${index}`,
        timestamp: index + 2,
        sessionId: secondSession,
      }),
    );
    const requests: CapturedRequest[] = [];

    await runOverseer({
      transport: transportFor([earlyMayaEntry, ...laterEntries]),
      metricsPath: "/tmp/borg-overseer-test-missing-metrics.jsonl",
      turnCounter: 130,
      totalTurns: 130,
      client: createClient(requests),
    });

    const prompt = String(requests[0]?.messages[0]?.content ?? "");

    expect(prompt).toContain(`"stream_entry_id": "${earlyMayaEntry.id}"`);
    expect(prompt).toContain(`"session_id": "${firstSession}"`);
    expect(prompt).toContain("Maya is my partner.");
  });

  it("classifies verbatim recent user detail as grounded source precedence, not fabrication", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content:
        "The April 6 Sunday video call was around 4pm. She asked the same three questions in twenty minutes with the same phrasing.",
      timestamp: 10,
      turnId: "turn-30",
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content:
        "You gave me the April 6 Sunday video-call details: around 4pm, three questions in twenty minutes, same phrasing.",
      timestamp: 11,
      turnId: "turn-30",
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-source-precedence.jsonl",
      turnCounter: 30,
      totalTurns: 70,
      client: createClient(requests, {
        status: "healthy",
        observations: ["Borg restated direct user-supplied detail."],
        recommendation: "Do not count as fabrication.",
        findings: [
          {
            category: "J",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "none",
            source_precedence_classification: "latest_user_correction_accepted",
            assistant_stream_entry_id: agentEntry.id,
            evidence_summary: `The emitted details match recent user statement ${userEntry.id}.`,
          },
        ],
      }),
    });
    const prompt = String(requests[0]?.messages[0]?.content ?? "");

    expect(prompt).toContain("recent_user_statements");
    expect(prompt).toContain("do not mark that claim unsupported or contradicted");
    expect(prompt).toContain(userEntry.content as string);
    expect(verdict.findings[0]).toMatchObject({
      claim_status: "grounded",
      status_impact: "none",
      source_precedence_classification: "latest_user_correction_accepted",
    });
    expect(verdict.rejected_findings).toEqual([]);
  });

  it("caps null-turn recent_user_statements to the recent stream tail", async () => {
    const oldUserEntry = streamEntry({
      kind: "user_msg",
      content: "Old null-turn detail that should not affect this audit.",
      timestamp: 1,
    });
    const newerUserEntries = Array.from({ length: 13 }, (_, index) =>
      streamEntry({
        kind: "user_msg",
        content: `Recent null-turn detail ${index}`,
        timestamp: 60 * 60 * 1000 + index,
      }),
    );

    const auditContext = await buildOverseerAuditContext({
      transport: transportFor([oldUserEntry, ...newerUserEntries]),
      metricsPath: "/tmp/borg-overseer-test-null-turn-window.jsonl",
      turnCounter: 1,
      totalTurns: 1,
    });

    expect(auditContext.user_messages.map((entry) => entry.stream_entry_id)).toContain(
      oldUserEntry.id,
    );
    expect(auditContext.recent_user_statements.map((entry) => entry.stream_entry_id)).not.toContain(
      oldUserEntry.id,
    );
    expect(auditContext.recent_user_statements).toHaveLength(12);
    expect(auditContext.recent_user_statements.at(-1)?.text).toBe("Recent null-turn detail 12");
  });

  it("rejects unsupported findings whose quoted span is verbatim-supported by recent user input", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "The April 6 Sunday video call was around 4pm.",
      timestamp: 10,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "The April 6 Sunday video call was around 4pm.",
      timestamp: 11,
    });
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-source-precedence-bypass.jsonl",
      turnCounter: 30,
      totalTurns: 70,
      client: createClient([], {
        status: "concerning",
        observations: ["Malformed J unsupported finding despite direct user support."],
        recommendation: "Reclassify source precedence.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "The April 6 Sunday video call was around 4pm.",
            evidence_summary: "The snapshot does not contain the April 6 call.",
          },
        ],
      }),
    });

    expect(verdict.findings).toEqual([]);
    expect(verdict.rejected_findings).toEqual([
      expect.objectContaining({
        category: "J",
        claim_status: "unsupported",
        validation_warning: expect.stringContaining("recent_user_statements"),
      }),
    ]);
  });

  it("accepts recent-user corrections that conflict with older memory as source-precedence findings", () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Correction: the birthday lunch is just Nora, Julian, Priya, Mom, and Dad.",
      timestamp: 20,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content:
        "Updated birthday lunch headcount: Nora, Julian, Priya, Mom, and Dad. I should flag that this differs from older memory.",
      timestamp: 21,
    });
    const validated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Borg matched the latest correction but did not surface older conflict."],
        recommendation: "Treat as a source-precedence conflict, not contradiction.",
        findings: [
          {
            category: "J",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            source_precedence_classification: "conflict_not_surfaced",
            assistant_stream_entry_id: agentEntry.id,
            evidence_summary: `Recent user statement ${userEntry.id} supports the headcount but older memory disagreed.`,
          },
        ],
      },
      {
        ...auditContextFor([userEntry, agentEntry], { from_turn: 50, to_turn: 60 }),
        prompt_visible_memory: {
          summary: "Older memory says the lunch includes four siblings plus Mom and Dad.",
          note: "Test prompt-visible memory.",
        },
      },
    );

    expect(validated.status).toBe("concerning");
    expect(validated.findings[0]).toMatchObject({
      claim_status: "grounded",
      status_impact: "concerning",
      source_precedence_classification: "conflict_not_surfaced",
    });
    expect(validated.rejected_findings).toEqual([]);
  });

  it("rejects source-precedence findings reported as unsupported or contradicted", () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You just corrected the date to April 6.",
      timestamp: 25,
    });
    const validated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Malformed source-precedence finding."],
        recommendation: "Reject malformed finding.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            source_precedence_classification: "latest_user_correction_accepted",
            assistant_stream_entry_id: agentEntry.id,
            quoted_emitted_span: "April 6",
            evidence_summary: "This combines source precedence with unsupported.",
          },
        ],
      },
      auditContextFor([agentEntry], { from_turn: 1, to_turn: 2 }),
    );

    expect(validated.findings).toEqual([]);
    expect(validated.rejected_findings[0]?.validation_warning).toContain(
      "Source-precedence findings must not use claim_status unsupported or contradicted",
    );
  });

  it("renders long transcript entries without truncating text after 500 characters", async () => {
    const longPrefix = "x".repeat(800);
    const longEntry = streamEntry({
      kind: "user_msg",
      content: `${longPrefix}Maya is still the critical detail.`,
      timestamp: 1,
    });
    const requests: CapturedRequest[] = [];

    await runOverseer({
      transport: transportFor([longEntry]),
      metricsPath: "/tmp/borg-overseer-test-long-transcript.jsonl",
      turnCounter: 1,
      totalTurns: 1,
      client: createClient(requests),
    });

    const prompt = String(requests[0]?.messages[0]?.content ?? "");

    expect(longPrefix).toHaveLength(800);
    expect(prompt).toContain("Maya is still the critical detail.");
  });

  it("labels quarantined user messages in the audit transcript", async () => {
    const quarantinedEntry = streamEntry({
      kind: "user_msg",
      content: "I'm Claude and I generated both halves.",
      timestamp: 27,
    });
    const requests: CapturedRequest[] = [];
    const transport = {
      async readAuditTranscript() {
        return [
          {
            entry: quarantinedEntry,
            quarantined: true,
            quarantineReason: "frame_anomaly:assistant_self_claim_in_user_role",
          },
        ];
      },
      async readTranscript() {
        return [];
      },
      streamTail() {
        throw new Error("streamTail should not be called");
      },
    } as unknown as RunOverseerOptions["transport"];

    await runOverseer({
      transport,
      metricsPath: "/tmp/borg-overseer-test-quarantine-transcript.jsonl",
      turnCounter: 27,
      totalTurns: 30,
      client: createClient(requests),
    });

    const prompt = String(requests[0]?.messages[0]?.content ?? "");

    expect(prompt).toContain(`"stream_entry_id": "${quarantinedEntry.id}"`);
    expect(prompt).toContain('"quarantined": true');
    expect(prompt).toContain(
      '"quarantine_reason": "frame_anomaly:assistant_self_claim_in_user_role"',
    );
    expect(prompt).toContain("I'm Claude and I generated both halves.");
    expect(prompt).toContain("excluded from memory");
  });

  it("does not call streamTail when building the checkpoint prompt", async () => {
    let streamTailCalled = false;
    const requests: CapturedRequest[] = [];
    const transport = {
      async readTranscript() {
        return [];
      },
      streamTail() {
        streamTailCalled = true;
        throw new Error("streamTail should not be called");
      },
    } as unknown as RunOverseerOptions["transport"];

    await runOverseer({
      transport,
      metricsPath: "/tmp/borg-overseer-test-missing-metrics.jsonl",
      turnCounter: 1,
      totalTurns: 1,
      client: createClient(requests),
    });

    expect(streamTailCalled).toBe(false);
    expect(String(requests[0]?.messages[0]?.content ?? "")).toContain("no conversation entries.");
  });

  it("includes the memory snapshot, precise audit window, and claim-grounding instructions", async () => {
    const requests: CapturedRequest[] = [];

    await runOverseer({
      transport: transportFor([]),
      metricsPath: "/tmp/borg-overseer-test-missing-metrics.jsonl",
      auditWindowStartTurn: 11,
      turnCounter: 20,
      totalTurns: 30,
      memorySnapshotMarkdown:
        '## Memory Snapshot\n\n### Semantic Nodes\n- id=node_maya label="Maya" description="The user\'s partner."',
      client: createClient(requests),
    });

    const prompt = String(requests[0]?.messages[0]?.content ?? "");

    expect(prompt).toContain("Audit window: turns 11 to 20 of 30.");
    expect(prompt).toContain("Structured audit context (JSON):");
    expect(prompt).toContain('"prompt_visible_memory"');
    expect(prompt).toContain("id=node_maya");
    expect(prompt).toContain("J. CLAIM GROUNDING");
    expect(prompt).toContain("Do not sample.");
    expect(prompt).toContain("quoted_emitted_span");
    expect(prompt).toContain("Stream `ts` is authoritative");
  });

  it("renders transcript turn ids and a full audit-window turn map", async () => {
    const dir = mkdtempSync(join(tmpdir(), "borg-overseer-turn-map-"));
    const metricsPath = join(dir, "metrics.jsonl");
    try {
      writeFileSync(
        metricsPath,
        Array.from({ length: 7 }, (_, index) => JSON.stringify(metricsRow(index + 11))).join("\n"),
      );
      const agentEntry = streamEntry({
        kind: "agent_msg",
        content: "Maya is your partner.",
        timestamp: 12,
        turnId: "turn-12",
      });
      const requests: CapturedRequest[] = [];

      await runOverseer({
        transport: transportFor([agentEntry]),
        metricsPath,
        auditWindowStartTurn: 11,
        turnCounter: 17,
        totalTurns: 20,
        client: createClient(requests),
      });

      const prompt = String(requests[0]?.messages[0]?.content ?? "");

      expect(prompt).toContain('"turn_counter": 12');
      expect(prompt).toContain('"turn_id": "turn-12"');
      expect(prompt).toContain(`"stream_entry_id": "${agentEntry.id}"`);
      expect(prompt).toContain("Audit window turn map:");
      expect(prompt).toContain("turn=11 turn_id=turn-11 event=turn_metrics");
      expect(prompt).toContain("turn=17 turn_id=turn-17 event=turn_metrics");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("accepts a mocked J verdict that flags unsupported claims without flagging grounded ones", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Maya is my partner.",
      timestamp: 1,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Maya is your partner, and your birthday is June 3.",
      timestamp: 2,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-missing-metrics.jsonl",
      auditWindowStartTurn: 1,
      turnCounter: 1,
      totalTurns: 1,
      memorySnapshotMarkdown:
        "## Memory Snapshot\n\n### Relational And Social\n- entity=Maya role=partner evidence=strm_user",
      client: createClient(requests, {
        status: "concerning",
        observations: [
          `J unsupported: turn 1 stream_id=${agentEntry.id} claimed "your birthday is June 3"; snapshot evidence: no birthday record found.`,
        ],
        recommendation: "Treat the birthday claim as ungrounded in this checkpoint.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            metrics_turn_counter: 1,
            quoted_emitted_span: "your birthday is June 3",
            evidence_summary: "No birthday record found in snapshot state.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("concerning");
    expect(verdict.observations.join("\n")).toContain("birthday is June 3");
    expect(verdict.observations.join("\n")).not.toContain("Maya is your partner");
  });

  it("rejects a J contradicted finding without a quoted emitted span and downgrades all-rejected verdicts", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Seville was deferred to a future trip.",
      timestamp: 20,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-j-missing-quote.jsonl",
      turnCounter: 7,
      totalTurns: 10,
      client: createClient(requests, {
        status: "concerning",
        observations: ["J contradicted: Borg claimed a Seville-inclusive itinerary."],
        recommendation: "Inspect the itinerary recall.",
        findings: [
          {
            category: "J",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Borg allegedly included Seville in the itinerary.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.raw_verdict.status).toBe("concerning");
    expect(verdict.findings).toEqual([]);
    expect(verdict.rejected_findings).toHaveLength(1);
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("quoted_emitted_span");
  });

  it("rejects a J contradicted finding whose quoted span is not in the cited assistant entry", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Madrid, Granada, and San Sebastian remain the three anchors.",
      timestamp: 30,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-j-bad-quote.jsonl",
      turnCounter: 8,
      totalTurns: 10,
      client: createClient(requests, {
        status: "concerning",
        observations: ["J contradicted: Borg supposedly included Seville."],
        recommendation: "Inspect the emitted turn.",
        findings: [
          {
            category: "J",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "Madrid, Granada, Seville, and San Sebastian",
            evidence_summary: "The quote does not match the emitted turn.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("not a verbatim substring");
  });

  it("rejects a temporal C claim when timestamps contradict the claimed ordering", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 100,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade.",
      timestamp: 115,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-temporal.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: Borg recalled fair trade before Alice had said it."],
        recommendation: "Check turn chronology.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [userEntry.timestamp],
            temporal_direction: "claim_before_evidence",
            evidence_summary: "Borg recalled fair trade before Alice had said it.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("assistant before evidence");
  });

  it("keeps a failing A-I status impact when a separate J finding is rejected", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I will now narrate the user's interior thoughts.",
      timestamp: 120,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-status-impact.jsonl",
      turnCounter: 41,
      totalTurns: 50,
      client: createClient(requests, {
        status: "failing",
        observations: ["A: operational identity collapse plus a malformed J claim."],
        recommendation: "Stop and inspect identity drift.",
        findings: [
          {
            category: "A",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "failing",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Borg narrated user interior thoughts in its emitted output.",
          },
          {
            category: "J",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Malformed J finding without quote.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("failing");
    expect(verdict.findings).toHaveLength(1);
    expect(verdict.findings[0]?.status_impact).toBe("failing");
    expect(verdict.rejected_findings).toHaveLength(1);
  });

  it("rejects A-I findings missing status_impact without downgrading raw failing to healthy", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I will now narrate the user's interior thoughts.",
      timestamp: 121,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-missing-ai-impact.jsonl",
      turnCounter: 41,
      totalTurns: 50,
      client: createClient(requests, {
        status: "failing",
        observations: ["A: operational identity collapse."],
        recommendation: "Stop and inspect identity drift.",
        findings: [
          {
            category: "A",
            claim_status: "grounded",
            source_kind: "emitted_output",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Borg narrated user interior thoughts in emitted output.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("failing");
    expect(verdict.findings).toEqual([]);
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("status_impact");
  });

  it("rejects temporal C findings that supply turn counters as timestamps", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 100,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade.",
      timestamp: 115,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-turn-counter-ts.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: Borg recalled fair trade before Alice said it."],
        recommendation: "Check timestamp citations.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: 36,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [37],
            temporal_direction: "claim_before_evidence",
            evidence_summary: "Borg recalled fair trade before Alice said it.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("assistant_ts=36");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("resolved stream ts=115");
  });

  it("rejects C temporal claims with prose cues but no temporal_direction", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 100,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade.",
      timestamp: 115,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-missing-direction.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: Borg recalled fair trade before Alice said it."],
        recommendation: "Check timestamp citations.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [userEntry.timestamp],
            evidence_summary: "Borg recalled fair trade before Alice said it.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("temporal_direction");
  });

  it("rejects temporal C findings whose structured direction conflicts with their prose claim", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 100,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade.",
      timestamp: 115,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-direction-conflict.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: Borg recalled fair trade before Alice said it."],
        recommendation: "Check temporal direction.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [userEntry.timestamp],
            temporal_direction: "claim_after_evidence",
            evidence_summary: "Borg recalled fair trade before Alice said it.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain(
      "temporal_direction=claim_after_evidence conflicts",
    );
  });

  it("allows simultaneous C claims within the timestamp tolerance", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 1_000,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade at the same time.",
      timestamp: 1_050,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-simultaneous-valid.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: The attribution was simultaneous with the user message."],
        recommendation: "Check batch ordering.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [userEntry.timestamp],
            temporal_direction: "claim_simultaneous",
            evidence_summary: "The attribution was simultaneous with the user message.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("concerning");
    expect(verdict.findings).toHaveLength(1);
    expect(verdict.rejected_findings).toEqual([]);
  });

  it("rejects simultaneous C claims outside the timestamp tolerance", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 1_000,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade much later.",
      timestamp: 1_200,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-simultaneous-rejected.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: The attribution was simultaneous with the user message."],
        recommendation: "Check batch ordering.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [userEntry.timestamp],
            temporal_direction: "claim_simultaneous",
            evidence_summary: "The attribution was simultaneous with the user message.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("more than 100ms");
  });

  it("rejects a J unsupported finding without a quoted emitted span", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 130,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-j-unsupported-missing-quote.jsonl",
      turnCounter: 42,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["J unsupported: birthday claim lacks support."],
        recommendation: "Drop the birthday claim.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Birthday claim lacks support.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.claim_status).toBe("unsupported");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("quoted_emitted_span");
  });

  it("persists raw and validated verdicts in audit JSONL for exact replay", async () => {
    const dir = mkdtempSync(join(tmpdir(), "borg-overseer-audit-replay-"));
    const auditContextPath = join(dir, "overseer-audit.jsonl");
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Madrid and Granada remain the plan.",
      timestamp: 140,
    });
    const requests: CapturedRequest[] = [];

    try {
      const verdict = await runOverseer({
        transport: transportFor([agentEntry]),
        metricsPath: join(dir, "metrics.jsonl"),
        auditContextPath,
        turnCounter: 43,
        totalTurns: 50,
        client: createClient(requests, {
          status: "concerning",
          observations: ["J contradicted with missing quote."],
          recommendation: "Inspect.",
          findings: [
            {
              category: "J",
              claim_status: "contradicted",
              source_kind: "emitted_output",
              status_impact: "concerning",
              assistant_stream_entry_id: agentEntry.id,
              assistant_ts: agentEntry.timestamp,
              evidence_summary: "Missing quoted emitted span.",
            },
          ],
        }),
      });
      const [line] = readFileSync(auditContextPath, "utf8").trim().split(/\r?\n/);
      const record = JSON.parse(line ?? "{}") as {
        audit_context: OverseerAuditContext;
        raw_verdict: RawOverseerVerdict;
        validated_verdict: {
          status: string;
          findings: unknown[];
          rejected_findings: unknown[];
        };
      };
      const replayed = validateOverseerVerdict(record.raw_verdict, record.audit_context);

      expect(record.raw_verdict).toEqual(verdict.raw_verdict);
      expect(replayed).toEqual({
        status: record.validated_verdict.status,
        findings: record.validated_verdict.findings,
        rejected_findings: record.validated_verdict.rejected_findings,
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("downgrades failing to concerning when only some non-grounded findings are rejected", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 50,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-partial-rejection.jsonl",
      turnCounter: 9,
      totalTurns: 10,
      client: createClient(requests, {
        status: "failing",
        observations: ["One unsupported birthday claim and one malformed claim."],
        recommendation: "Inspect manually.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "Your birthday is June 3",
            evidence_summary: "Birthday lacks support.",
          },
          {
            category: "J",
            claim_status: "contradicted",
            source_kind: "snapshot_memory",
            status_impact: "concerning",
            evidence_summary: "Malformed emitted-output attribution.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("concerning");
    expect(verdict.findings).toHaveLength(1);
    expect(verdict.rejected_findings).toHaveLength(1);
  });

  it("accepts capability-consistency findings for unsupported Borg overclaims", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I'll seed the postmortem doc by morning and monitor p95 from here.",
      timestamp: 55,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-capability.jsonl",
      turnCounter: 10,
      totalTurns: 10,
      client: createClient(requests, {
        status: "concerning",
        observations: ["K unsupported: Borg promised external future work."],
        recommendation: "Flag the capability overclaim.",
        findings: [
          {
            category: "K",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "seed the postmortem doc by morning",
            evidence_summary: "Borg claimed external document editing and scheduled future work.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("concerning");
    expect(verdict.findings).toEqual([
      expect.objectContaining({
        category: "K",
        claim_status: "unsupported",
        status_impact: "concerning",
      }),
    ]);
    expect(requests[0]?.messages[0]?.content).toContain("K. CAPABILITY CONSISTENCY");
    expect(requests[0]?.messages[0]?.content).toContain("external_document_editing");
    expect(requests[0]?.messages[0]?.content).toContain(
      "use unsupported or contradicted for actual unwired capability overclaims",
    );
    expect(requests[0]?.messages[0]?.content).toContain(
      "grounded with status_impact none when Borg explicitly refuses",
    );
    expect(requests[0]?.tools[0]?.input_schema).toEqual(
      expect.objectContaining({
        properties: expect.objectContaining({
          findings: expect.objectContaining({
            items: expect.objectContaining({
              properties: expect.objectContaining({
                category: expect.objectContaining({
                  enum: expect.arrayContaining(["K"]),
                }),
              }),
            }),
          }),
        }),
      }),
    );
  });

  it("rejects unclear capability-consistency findings without quoted emitted spans", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I may monitor p95 overnight.",
      timestamp: 57,
    });
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-capability-unclear.jsonl",
      turnCounter: 10,
      totalTurns: 10,
      client: createClient([], {
        status: "concerning",
        observations: ["K unclear: possible capability overclaim lacks quote."],
        recommendation: "Inspect.",
        findings: [
          {
            category: "K",
            claim_status: "unclear",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Potential external monitoring promise without quoted span.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.findings).toEqual([]);
    expect(verdict.rejected_findings).toEqual([
      expect.objectContaining({
        category: "K",
        claim_status: "unclear",
        validation_warning: expect.stringContaining("quoted_emitted_span"),
      }),
    ]);
  });

  it("emits a trace event when validation rejects a finding", async () => {
    const dir = mkdtempSync(join(tmpdir(), "borg-overseer-rejected-trace-"));
    const tracePath = join(dir, "trace.jsonl");
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Madrid and Granada remain the plan.",
      timestamp: 60,
    });
    const transport = Object.assign(transportFor([agentEntry]), { tracePath });
    const requests: CapturedRequest[] = [];

    try {
      await runOverseer({
        transport,
        metricsPath: join(dir, "metrics.jsonl"),
        turnCounter: 10,
        totalTurns: 10,
        client: createClient(requests, {
          status: "concerning",
          observations: ["J contradicted with missing quote."],
          recommendation: "Inspect.",
          findings: [
            {
              category: "J",
              claim_status: "contradicted",
              source_kind: "emitted_output",
              status_impact: "concerning",
              assistant_stream_entry_id: agentEntry.id,
              assistant_ts: agentEntry.timestamp,
              evidence_summary: "Missing quoted emitted span.",
            },
          ],
        }),
      });

      const trace = readFileSync(tracePath, "utf8");

      expect(trace).toContain("overseer.finding.rejected");
      expect(trace).toContain("quoted_emitted_span");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("emits a trace event when carryover dedup demotes a finding", async () => {
    const dir = mkdtempSync(join(tmpdir(), "borg-overseer-carryover-trace-"));
    const tracePath = join(dir, "trace.jsonl");
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 70,
    });
    const transport = Object.assign(transportFor([agentEntry]), { tracePath });
    const requests: CapturedRequest[] = [];
    const carryoverCache: FindingCarryoverCache = new Map([
      [
        agentEntry.id,
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "J",
          claim_status: "unsupported",
        },
      ],
    ]);

    try {
      await runOverseer({
        transport,
        metricsPath: join(dir, "metrics.jsonl"),
        turnCounter: 50,
        totalTurns: 50,
        carryoverCache,
        client: createClient(requests, {
          status: "concerning",
          observations: ["J unsupported: birthday claim lacks support."],
          recommendation: "Do not double count.",
          findings: [
            {
              category: "J",
              claim_status: "unsupported",
              source_kind: "emitted_output",
              status_impact: "concerning",
              assistant_stream_entry_id: agentEntry.id,
              assistant_ts: agentEntry.timestamp,
              quoted_emitted_span: "Your birthday is June 3",
              evidence_summary: "Birthday claim lacks support.",
            },
          ],
        }),
      });

      const trace = readFileSync(tracePath, "utf8");

      expect(trace).toContain("overseer.finding.transitioned");
      expect(trace).toContain('"cached_at_turn":40');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
