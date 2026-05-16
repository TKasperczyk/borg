import type { EvidenceLedgerSectionId } from "./types.js";

export function normalizePositiveInteger(value: number | undefined, fallback: number): number {
  return value === undefined || !Number.isFinite(value) || value <= 0
    ? fallback
    : Math.floor(value);
}

export function allSectionIds(): EvidenceLedgerSectionId[] {
  return [
    "current_user_message",
    "current_session_transcript",
    "current_session_attribution_sidebar",
    "attribution_matrix",
    "commitments_and_constraints",
    "closure_discourse_state",
    "contradictions_quarantines",
    "action_states",
    "group_channel_memory",
    "relational_slots",
    "retrieved_raw_stream_evidence",
    "retrieved_memory_evidence",
    "episodes",
    "semantic_graph",
    "open_questions",
    "prior_session_memory",
  ];
}

export function emptySectionCountRecord(): Record<EvidenceLedgerSectionId, number> {
  return Object.fromEntries(allSectionIds().map((sectionId) => [sectionId, 0])) as Record<
    EvidenceLedgerSectionId,
    number
  >;
}
