import type { TurnTraceEventName } from "./tracer.js";

export const TRACE_EVENT_TAXONOMY = {
  perception: [
    "perception.started",
    "perception.completed",
    "perception.classifier.degraded",
    "recency.completed",
    "participant_scan.skipped",
    "frame_anomaly.degraded",
    "frame_anomaly.degraded_fail_open",
    "frame_anomaly.completed",
    "frame_anomaly.transitioned",
    "closure_loop.degraded",
    "closure_loop.transitioned",
  ],
  working_memory: ["working_memory.degraded"],
  executive_focus: [],
  retrieval: [
    "retrieval.started",
    "retrieval.completed",
    "retrieval.degraded",
    "recall_expansion.completed",
    "citation_resolution.degraded",
    "evidence_ledger.completed",
    "evidence_ledger.compaction.completed",
    "shared_state.compile.skipped",
    "shared_state.compile.transitioned",
    "shared_state.compile.completed",
    "shared_state.compile.degraded",
    "shared_state.compile.add_rejected_cap_exceeded",
    "shared_state.compile.label_ungrounded",
    "shared_state.compile.repair_attempted",
    "shared_state.compile.repair_succeeded",
    "shared_state.compile.repair_failed",
    "shared_state.lifecycle.degraded",
    "shared_state.reconcile.completed",
    "shared_state.reconcile.skipped",
    "shared_state.canonicalization.completed",
    "shared_state.semantic_revision.degraded",
    "semantic_revision.completed",
    "semantic_revision.cache.completed",
    "semantic_revision.degraded",
  ],
  deliberation: [
    "llm_call.started",
    "llm_call.completed",
    "deliberation.planner_ledger.completed",
    "deliberation.contradiction_routing.completed",
    "deliberation.contradiction_routing.transitioned",
    "deliberation.path.completed",
    "deliberation.path.transitioned",
    "deliberation.plan.completed",
    "deliberation.planner.degraded",
    "deliberation.plan_persistence.completed",
    "deliberation.plan_persistence.skipped",
    "finalizer.completed",
  ],
  tools: ["tool_call.started", "tool_call.completed"],
  commitments: [
    "commitment_guard.shadow_observation",
    "commitment_guard.enforce_suppression",
    "commitment_guard.enforce_rewrite",
    "commitment_guard.regeneration_requested",
    "commitment_guard.regeneration_succeeded",
    "commitment_guard.regeneration_failed",
    "commitment_check.completed",
    "closure_response_guard.completed",
    "closure_pressure_audit.degraded",
    "internal_identifier_guard.completed",
    "post_generation.rejected",
  ],
  extraction: [
    "extraction.commitments.degraded",
    "extraction.commitments.rejected",
    "extraction.commitments.transitioned",
    "corrective_preference.candidate_rejected_ungrounded",
    "extraction.actions.rejected",
    "extraction.actions.completed",
    "extraction.actions.degraded",
    "action_persistence.dedup.skipped",
    "action_persistence.dedup.degraded",
    "action_state.transitioned",
    "action_state.borg_self_performance.completed",
    "action_state.archived",
    "action_session_scope.expired",
    "action_session_scope.rolled_over",
    "action_archive.completed",
    "action_archive_scan.completed",
    "action_inactivity_scan.completed",
    "action_duplicate_pressure.completed",
    "extraction.goals.degraded",
    "extraction.goals.completed",
    "extraction.goals.rejected",
    "extraction.goals.transitioned",
    "extraction.goals.skipped",
    "extraction.goals.dedup.degraded",
  ],
  discourse: ["discourse_state.transitioned", "discourse_state.rejected"],
  reflection: [
    "reflection.completed",
    "open_question_resolution.started",
    "open_question_resolution.degraded",
    "open_question_resolution.transitioned",
    "open_question_resolution.rejected",
    "reflector.intent_update.completed",
    "reflector.intent_update.rejected",
    "reflector.candidate.completed",
  ],
  review: [
    "review_resolver.started",
    "review_resolver.decision.completed",
    "review_resolver.completed",
    "review_resolver.degraded",
    "review_queue.completed",
  ],
  ingestion: [
    "semantic_extractor.started",
    "semantic_extractor.degraded",
    "semantic_insert.skipped",
    "semantic_node.status.transitioned",
  ],
  offline: ["offline_process.completed"],
  maintenance: ["maintenance_snapshot.completed"],
  session: ["session.completed", "turn.rejected"],
} as const satisfies Record<string, readonly TurnTraceEventName[]>;

export type TraceTaxonomyPhase = keyof typeof TRACE_EVENT_TAXONOMY;
export type TracePhaseWithOther = TraceTaxonomyPhase | "other";

export const TRACE_TAXONOMY_PHASES = Object.keys(TRACE_EVENT_TAXONOMY) as TraceTaxonomyPhase[];
export const TRACE_TAXONOMY_PHASES_WITH_OTHER: readonly TracePhaseWithOther[] = [
  ...TRACE_TAXONOMY_PHASES,
  "other",
];

export const TRACE_EVENT_DEPRECATION_ALIASES = {
  decision_artifact_compile: "shared_state.compile.completed",
  "decision_artifact_compile.completed": "shared_state.compile.completed",
  "decision_artifact_compile.degraded": "shared_state.compile.degraded",
  "decision_artifact_compile.skipped": "shared_state.compile.skipped",
  "decision_artifact_compile.transitioned": "shared_state.compile.transitioned",
  "decision_artifact_lifecycle.degraded": "shared_state.lifecycle.degraded",
  "decision_artifact_reconcile.completed": "shared_state.reconcile.completed",
  "decision_artifact_reconcile.skipped": "shared_state.reconcile.skipped",
  "decision_artifact_canonicalization.completed": "shared_state.canonicalization.completed",
} as const satisfies Partial<Record<string, TurnTraceEventName>>;

const TRACE_EVENT_PHASES = new Map<string, TraceTaxonomyPhase>();

for (const phase of TRACE_TAXONOMY_PHASES) {
  for (const event of TRACE_EVENT_TAXONOMY[phase]) {
    TRACE_EVENT_PHASES.set(event, phase);
  }
}

for (const [alias, preferred] of Object.entries(TRACE_EVENT_DEPRECATION_ALIASES)) {
  const phase = TRACE_EVENT_PHASES.get(preferred);

  if (phase !== undefined) {
    TRACE_EVENT_PHASES.set(alias, phase);
  }
}

export function canonicalTraceEventName(event: string): string {
  const aliases: Partial<Record<string, string>> = TRACE_EVENT_DEPRECATION_ALIASES;

  return aliases[event] ?? event;
}

export function phaseForTraceEventName(event: string): TracePhaseWithOther {
  return TRACE_EVENT_PHASES.get(canonicalTraceEventName(event)) ?? "other";
}
