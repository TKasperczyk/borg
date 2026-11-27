import type { OpenQuestion } from "./types.js";

export type OpenQuestionRefusalFields = Pick<
  OpenQuestion,
  | "consecutive_refusals"
  | "last_rumination_attempt_outcome"
  | "last_rumination_attempt_at"
  | "refusal_backoff_until"
>;

// The key the stored fields travel under on the ledger row and on the dormant-question wake. One
// key, so a surface that has to cut for size keeps the state whole or drops it whole, and never
// shows a count without the deadline that belongs beside it. (A tool response that returns the raw
// record still carries the same four fields flat, as that record has them.)
export const OPEN_QUESTION_REFUSAL_STATE_KEY = "refusal_state";

// True while the row holds anything about refused offline calls: a count above zero, or a backoff
// deadline, expired or not. The write that records a completed call clears both together, but a
// restored snapshot can hold either without the other, and a deadline the scheduler acts on must
// not be the one that goes unshown.
export function hasOpenQuestionRefusalState(question: OpenQuestionRefusalFields): boolean {
  return question.consecutive_refusals > 0 || question.refusal_backoff_until !== null;
}

// The stored fields under the store's own names and with the store's own values: nothing derived,
// so nothing here can go stale between the read and the reader, and the fields read the same as in
// a tool response or an identity-event excerpt that carries the raw record. Each surface gives the
// two timestamps the readable form its other timestamps get. Spread into a row like
// openQuestionReviewFields: a question with no refusal state contributes no key at all.
export function openQuestionRefusalState(question: OpenQuestionRefusalFields): {
  [OPEN_QUESTION_REFUSAL_STATE_KEY]?: OpenQuestionRefusalFields;
} {
  if (!hasOpenQuestionRefusalState(question)) {
    return {};
  }

  return {
    [OPEN_QUESTION_REFUSAL_STATE_KEY]: {
      consecutive_refusals: question.consecutive_refusals,
      last_rumination_attempt_outcome: question.last_rumination_attempt_outcome,
      last_rumination_attempt_at: question.last_rumination_attempt_at,
      refusal_backoff_until: question.refusal_backoff_until,
    },
  };
}

// Said once per surface, not once per row: the fields are short, and a page that carries several
// refused questions should spend its budget on the questions. Each sentence is scoped to what the
// code does: an attempt counts once it is recorded, which a dry run does not do; the count is
// cleared by more than a completed call (resolving or reopening a question clears it too); the
// deadline bounds resolution calls only (a held question can still be merged as a duplicate); and
// the stale-dismissal guard is read when the pass plans, not when it writes.
export const OPEN_QUESTION_REFUSAL_STATE_NOTE =
  "consecutive_refusals on a question counts my offline pass's resolution calls on it recorded as refused by the provider since the count was last cleared; a completed call, once recorded, clears it. A refused call writes no rumination note and moves neither last_ruminated_at nor unresolved_rumination_ticks. While a refusal_backoff_until deadline lies ahead the pass schedules no resolution call on that question, and while the count is above zero the pass does not plan to dismiss or abandon it as stale.";
