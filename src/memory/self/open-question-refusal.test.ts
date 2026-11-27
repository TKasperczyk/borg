import { describe, expect, it } from "vitest";

import {
  OPEN_QUESTION_REFUSAL_STATE_KEY,
  OPEN_QUESTION_REFUSAL_STATE_NOTE,
  hasOpenQuestionRefusalState,
  openQuestionRefusalState,
} from "./open-question-refusal.js";

const NOW_MS = Date.parse("2026-09-02T12:00:00.000Z");
const HOUR_MS = 60 * 60 * 1_000;
const NO_REFUSAL_STATE = {
  consecutive_refusals: 0,
  last_rumination_attempt_outcome: null,
  last_rumination_attempt_at: null,
  refusal_backoff_until: null,
} as const;

describe("open question refusal state", () => {
  it("finds refusal state in a count or a deadline, and in nothing else", () => {
    expect(hasOpenQuestionRefusalState(NO_REFUSAL_STATE)).toBe(false);
    // A completed attempt is not refusal state: the write that records it zeroes the count.
    expect(
      hasOpenQuestionRefusalState({
        ...NO_REFUSAL_STATE,
        last_rumination_attempt_outcome: "completed",
        last_rumination_attempt_at: NOW_MS,
      }),
    ).toBe(false);
    expect(hasOpenQuestionRefusalState({ ...NO_REFUSAL_STATE, consecutive_refusals: 1 })).toBe(
      true,
    );
    // A deadline is state for as long as it is stored, whether or not it has passed.
    expect(
      hasOpenQuestionRefusalState({ ...NO_REFUSAL_STATE, refusal_backoff_until: NOW_MS - 1 }),
    ).toBe(true);
  });

  it("carries exactly the stored fields under one key, and no key for a row without state", () => {
    const stored = {
      consecutive_refusals: 3,
      last_rumination_attempt_outcome: "refused",
      last_rumination_attempt_at: NOW_MS - 2 * HOUR_MS,
      refusal_backoff_until: NOW_MS + HOUR_MS,
    } as const;

    expect(openQuestionRefusalState(NO_REFUSAL_STATE)).toEqual({});
    // Nothing is derived and nothing else on the record rides along.
    expect(
      openQuestionRefusalState({ ...stored, question: "not a refusal field" } as typeof stored),
    ).toEqual({ [OPEN_QUESTION_REFUSAL_STATE_KEY]: stored });
    expect(OPEN_QUESTION_REFUSAL_STATE_KEY).toBe("refusal_state");
  });

  it("shows pairings production does not write as they are stored", () => {
    // A restored snapshot can hold a deadline without a count. The scheduler acts on the deadline.
    const deadlineOnly = { ...NO_REFUSAL_STATE, refusal_backoff_until: NOW_MS + HOUR_MS };
    expect(openQuestionRefusalState(deadlineOnly)).toEqual({ refusal_state: deadlineOnly });
    // A count beside a completed attempt keeps that attempt's own label.
    const countBesideCompleted = {
      ...NO_REFUSAL_STATE,
      consecutive_refusals: 1,
      last_rumination_attempt_outcome: "completed",
      last_rumination_attempt_at: NOW_MS - HOUR_MS,
    } as const;
    expect(openQuestionRefusalState(countBesideCompleted)).toEqual({
      refusal_state: countBesideCompleted,
    });
  });

  it("names in its note every field the note explains", () => {
    for (const field of [
      "consecutive_refusals",
      "refusal_backoff_until",
      "last_ruminated_at",
      "unresolved_rumination_ticks",
    ]) {
      expect(OPEN_QUESTION_REFUSAL_STATE_NOTE).toContain(field);
    }
  });
});
