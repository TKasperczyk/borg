import type { SessionId, StreamCursor, StreamResponseTo } from "./types.js";

/**
 * An exact set, not the contiguous stream prefix ending at a response. The
 * source IDs are authoritative even when unanswered arrivals interleave with
 * generation. terminalCursor identifies the response/terminal marker that set
 * the edge; through_cursor_inclusive identifies the last answered input.
 */
export type AnsweredStreamWindow = {
  responseTo: StreamResponseTo;
  terminalCursor: StreamCursor;
};

export type AnsweredWindowEvidence = {
  session_id: SessionId;
  observed_at: number;
  state: "recorded" | "no_answered_edge" | "basis_records_unavailable";
  basis: null | {
    turn_id: string | null;
    response_entry_id: string;
    response_at: number;
    response_kind: string;
    last_answered_entry_id: string;
    last_answered_at: number;
    answered_entry_count: number;
  };
  outside: {
    state:
      | "arrived_after_edge"
      | "outside_answered_set"
      | "none"
      | "no_answered_edge"
      | "unavailable";
    arrived_after_edge: number | null;
    unselected_within_window: number | null;
    before_window: number | null;
    without_edge: number | null;
  };
};

export function renderAnsweredWindowEvidence(evidence: AnsweredWindowEvidence): string {
  return `Answered-window edge (session=${evidence.session_id}, observed_at=${evidence.observed_at}): ${JSON.stringify(evidence)}. Basis is the persisted terminal response_to stamp, not the ingestion watermark. Outside counts partition recorded user_msg arrivals in this session at this read, excluding the exact answered source IDs: after the last answered entry (including arrivals during generation), unselected within this response's span, and before its from cursor. none means all three counts are zero; no_answered_edge means no stamped terminal response exists, not that nothing arrived. Scope and state labels only; these counts neither gate recall nor decide whether to respond.`;
}
