import { describe, expect, it } from "vitest";

import type { OpenQuestion } from "../../memory/self/index.js";
import type { OpenQuestionId } from "../../util/ids.js";
import { openQuestionStateMetadata } from "./open-question-handles.js";

const NOW_MS = Date.parse("2026-09-02T12:00:00.000Z");
const DAY_MS = 24 * 60 * 60 * 1_000;

function openQuestion(overrides: Partial<OpenQuestion> = {}): OpenQuestion {
  return {
    id: "oq_test0000000000000" as OpenQuestionId,
    question: "Does the page carry the counter the dismissal reads?",
    urgency: 0.5,
    status: "open",
    goal_id: null,
    audience_entity_id: null,
    related_episode_ids: [],
    related_semantic_node_ids: [],
    provenance: null,
    source: "reflection",
    created_at: Date.parse("2026-08-20T12:00:00.000Z"),
    last_touched: Date.parse("2026-08-20T12:00:00.000Z"),
    resolution_evidence_episode_ids: [],
    resolution_evidence_stream_entry_ids: [],
    resolution_note: null,
    resolved_at: null,
    abandoned_reason: null,
    abandoned_at: null,
    unresolved_rumination_ticks: 0,
    last_ruminated_at: null,
    consecutive_refusals: 0,
    last_rumination_attempt_outcome: null,
    last_rumination_attempt_at: null,
    refusal_backoff_until: null,
    ...overrides,
  };
}

describe("openQuestionStateMetadata", () => {
  it("renders the rumination counter and the threshold it feeds on an open question", () => {
    const metadata = openQuestionStateMetadata(
      openQuestion({
        unresolved_rumination_ticks: 6,
        last_ruminated_at: Date.parse("2026-09-01T20:40:39.311Z"),
      }),
      NOW_MS,
      4,
    );

    expect(metadata).toMatchObject({
      unresolved_rumination_ticks: 6,
      last_ruminated_at: "2026-09-01T20:40:39.311Z",
      dismissal_threshold_ticks: 4,
    });
    expect(metadata?.last_ruminated_relative_age).toEqual(expect.any(String));
  });

  it("distinguishes a question the loop has never selected from one it has stopped selecting", () => {
    const neverSelected = openQuestionStateMetadata(openQuestion(), NOW_MS, 4);
    const stoppedBeingSelected = openQuestionStateMetadata(
      openQuestion({
        unresolved_rumination_ticks: 3,
        last_ruminated_at: Date.parse("2026-08-25T20:40:39.311Z"),
      }),
      NOW_MS,
      4,
    );

    // Both have a zero-ish look on the page; only last_ruminated_at separates them, so the
    // key has to be present and null rather than omitted when the loop never ran.
    expect(neverSelected).toMatchObject({
      unresolved_rumination_ticks: 0,
      last_ruminated_at: null,
    });
    expect(Object.keys(neverSelected ?? {})).toContain("last_ruminated_at");
    expect(neverSelected).not.toHaveProperty("last_ruminated_relative_age");
    expect(stoppedBeingSelected).toMatchObject({ last_ruminated_at: "2026-08-25T20:40:39.311Z" });
  });

  it("spends the explanatory sentence only where the count is live", () => {
    const inert = openQuestionStateMetadata(openQuestion(), NOW_MS, 4);
    const live = openQuestionStateMetadata(
      openQuestion({ unresolved_rumination_ticks: 1 }),
      NOW_MS,
      4,
    );

    expect(inert).not.toHaveProperty("unresolved_rumination_ticks_note");
    expect(live?.unresolved_rumination_ticks_note).toContain("does not close it");
    // The count alone never closes a question -- the citation condition is the rest of
    // the predicate, and a page that printed the count without them would read as a countdown.
    expect(live?.unresolved_rumination_ticks_note).toContain(
      "no cited episode created after the question",
    );
    expect(live?.unresolved_rumination_ticks_note).toContain(
      "action and conversational progress are not measured",
    );
    // A count past the threshold on an open row is the ordinary case, not a puzzle: the pass that
    // raised it called the model, and that path returns before shouldDismissStaleNoTraction is
    // reached. Naming only the two conjuncts invites reading an over-threshold row as evidence
    // that one of them fired, when the predicate was never consulted on that pass at all.
    expect(live?.unresolved_rumination_ticks_note).toContain("is not evaluated at all on a pass");
    // stampRuminationForRun's note-less branch writes last_ruminated_at alone, so a sentence that
    // calls this a count of passes invites reading a gap between the count and the notes as a lost
    // note.
    expect(live?.unresolved_rumination_ticks_note).toContain("recorded no note");
    // A completed dormant-question wake (commitFired) also advances it, and writes no note, so a
    // row can carry ticks with no notes at all; a scope of note-bearing passes alone is false there.
    expect(live?.unresolved_rumination_ticks_note).toContain("completed dormant-question wakes");
    // A note-bearing pass stamps at or just after its own note, and a wake or a note-less pass
    // stamps with no note, so a stamp later than the newest note does not say which one wrote it.
    expect(live?.unresolved_rumination_ticks_note).not.toContain("later than the newest note");
  });

  it("names the third state: zero notes beside a live stamp", () => {
    const neverSelected = openQuestionStateMetadata(openQuestion(), NOW_MS, 4);
    const selectedAndSilent = openQuestionStateMetadata(
      openQuestion({
        unresolved_rumination_ticks: 0,
        last_ruminated_at: Date.parse("2026-09-01T20:40:39.311Z"),
      }),
      NOW_MS,
      4,
    );

    // Suppressing the sentence at zero is right only for the null-stamp zero. Zero beside a live
    // stamp is a pass that produced nothing, and it is the one zero a reader cannot tell apart
    // from never-selected -- so it is exactly where the omitted sentence was load-bearing.
    expect(neverSelected).not.toHaveProperty("unresolved_rumination_ticks_note");
    expect(selectedAndSilent?.unresolved_rumination_ticks_note).toContain("produced no note");
    expect(selectedAndSilent?.unresolved_rumination_ticks_note).toContain(
      "not a question the loop has never reached",
    );
  });

  it("carries the store's note count beside the ticks, in both directions of divergence", () => {
    // A merge hands a duplicate's notes to the primary without its ticks, so the two quantities
    // separate. Neither direction is recoverable from the page while only the counter is on it.
    const ticksAboveNotes = openQuestionStateMetadata(
      openQuestion({ unresolved_rumination_ticks: 4, last_ruminated_at: NOW_MS }),
      NOW_MS,
      4,
      2,
    );
    const notesAboveTicks = openQuestionStateMetadata(
      openQuestion({ unresolved_rumination_ticks: 2, last_ruminated_at: NOW_MS }),
      NOW_MS,
      4,
      4,
    );

    expect(ticksAboveNotes).toMatchObject({
      unresolved_rumination_ticks: 4,
      recorded_rumination_notes: 2,
    });
    expect(notesAboveTicks).toMatchObject({
      unresolved_rumination_ticks: 2,
      recorded_rumination_notes: 4,
    });
    // The ticks label must not name itself as the note count once the note count is beside it.
    expect(ticksAboveNotes?.unresolved_rumination_ticks_note).toContain(
      "not recorded_rumination_notes beside it",
    );
  });

  it("suppresses the note count only where the row carries no rumination signal at all", () => {
    const silent = openQuestionStateMetadata(openQuestion(), NOW_MS, 4, 0);
    // Zero ticks and a null stamp with notes on the row is the merge case, and it is the one the
    // reader cannot reconstruct: the counter reads the same as a question never worked.
    const mergedOntoUnworked = openQuestionStateMetadata(openQuestion(), NOW_MS, 4, 3);

    expect(silent).not.toHaveProperty("recorded_rumination_notes");
    expect(mergedOntoUnworked).toMatchObject({
      unresolved_rumination_ticks: 0,
      recorded_rumination_notes: 3,
    });
  });

  it("shows a refused offline call that the stamp and the tick count do not record", () => {
    const refusedAt = Date.parse("2026-09-01T13:44:33.000Z");
    const neverPicked = openQuestionStateMetadata(openQuestion(), NOW_MS, 4);
    const refusedOnce = openQuestionStateMetadata(
      openQuestion({
        consecutive_refusals: 1,
        last_rumination_attempt_outcome: "refused",
        last_rumination_attempt_at: refusedAt,
      }),
      NOW_MS,
      4,
    );

    // A refused call moves neither the stamp nor the tick count, so this one object is the whole
    // difference between this row and one the offline pass never picked. It holds the store's own
    // names and values, in the form the row's other timestamps take.
    expect(neverPicked).toBeDefined();
    expect(neverPicked).not.toHaveProperty("refusal_state");
    expect(refusedOnce).toEqual({
      ...neverPicked,
      refusal_state: {
        consecutive_refusals: 1,
        last_rumination_attempt_outcome: "refused",
        last_rumination_attempt_at: "2026-09-01T13:44:33.000Z",
        last_rumination_attempt_relative_age: "22h ago",
        refusal_backoff_until: null,
      },
    });
  });

  it("shows a backoff deadline with the side of now it lies on", () => {
    const held = openQuestion({
      consecutive_refusals: 3,
      last_rumination_attempt_outcome: "refused",
      last_rumination_attempt_at: NOW_MS - 2 * DAY_MS,
      refusal_backoff_until: NOW_MS + 5 * DAY_MS,
    });
    const stateOf = (question: OpenQuestion, nowMs?: number) =>
      openQuestionStateMetadata(question, nowMs, 4)?.refusal_state;

    expect(stateOf(held, NOW_MS)).toEqual({
      consecutive_refusals: 3,
      last_rumination_attempt_outcome: "refused",
      last_rumination_attempt_at: new Date(NOW_MS - 2 * DAY_MS).toISOString(),
      last_rumination_attempt_relative_age: "2d ago",
      refusal_backoff_until: new Date(NOW_MS + 5 * DAY_MS).toISOString(),
      refusal_backoff_until_relative: "in 5d",
    });
    // A deadline that has passed stays on the row with the count that set it, and reads as past.
    expect(stateOf({ ...held, refusal_backoff_until: NOW_MS - 3 * DAY_MS }, NOW_MS)).toMatchObject({
      consecutive_refusals: 3,
      refusal_backoff_until: new Date(NOW_MS - 3 * DAY_MS).toISOString(),
      refusal_backoff_until_relative: "3d ago",
    });
    // The offline pass holds a question only while the deadline is later than its clock, so a
    // deadline equal to now already reads as past.
    expect(stateOf({ ...held, refusal_backoff_until: NOW_MS }, NOW_MS)).toMatchObject({
      refusal_backoff_until_relative: "~0s ago",
    });

    // Without a clock nothing is said about now; the stored values are still shown.
    expect(stateOf(held)).toEqual({
      consecutive_refusals: 3,
      last_rumination_attempt_outcome: "refused",
      last_rumination_attempt_at: new Date(NOW_MS - 2 * DAY_MS).toISOString(),
      refusal_backoff_until: new Date(NOW_MS + 5 * DAY_MS).toISOString(),
    });
  });

  it("renders no refusal state on a row that holds none", () => {
    // A completed call zeroes the count; the attempt it leaves behind is not a refusal.
    const open = openQuestionStateMetadata(
      openQuestion({
        last_rumination_attempt_outcome: "completed",
        last_rumination_attempt_at: NOW_MS - DAY_MS,
        last_ruminated_at: NOW_MS - DAY_MS,
        unresolved_rumination_ticks: 2,
      }),
      NOW_MS,
      4,
    );
    // Closed rows keep their own branches; refusal state left on one is not rendered there.
    const resolved = openQuestionStateMetadata(
      openQuestion({
        status: "resolved",
        resolution_note: "Settled.",
        resolved_at: NOW_MS,
        consecutive_refusals: 2,
        last_rumination_attempt_outcome: "refused",
        last_rumination_attempt_at: NOW_MS - DAY_MS,
      }),
      NOW_MS,
      4,
    );
    const abandoned = openQuestionStateMetadata(
      openQuestion({
        status: "abandoned",
        abandoned_reason: "No longer needed.",
        abandoned_at: NOW_MS,
        consecutive_refusals: 3,
        last_rumination_attempt_outcome: "refused",
        last_rumination_attempt_at: NOW_MS - DAY_MS,
        refusal_backoff_until: NOW_MS + DAY_MS,
      }),
      NOW_MS,
      4,
    );

    // Each row still renders; it is only the refusal state that is absent, under any of its names.
    expect(open).toMatchObject({ unresolved_rumination_ticks: 2 });
    expect(resolved).toMatchObject({ resolution_note: "Settled." });
    expect(abandoned).toMatchObject({ abandoned_reason: "No longer needed." });
    for (const metadata of [open, resolved, abandoned]) {
      for (const key of [
        "refusal_state",
        "consecutive_refusals",
        "last_rumination_attempt_outcome",
        "last_rumination_attempt_at",
        "refusal_backoff_until",
      ]) {
        expect(metadata).not.toHaveProperty(key);
      }
    }
  });

  it("shows stored refusal state that production does not write, as it is stored", () => {
    // A restored snapshot can hold a deadline without a count. The scheduler acts on the deadline,
    // so the row shows it.
    expect(
      openQuestionStateMetadata(openQuestion({ refusal_backoff_until: NOW_MS + DAY_MS }), NOW_MS, 4)
        ?.refusal_state,
    ).toEqual({
      consecutive_refusals: 0,
      last_rumination_attempt_outcome: null,
      last_rumination_attempt_at: null,
      refusal_backoff_until: new Date(NOW_MS + DAY_MS).toISOString(),
      refusal_backoff_until_relative: "in 1d",
    });
    // A count beside a completed attempt keeps that attempt's own label.
    expect(
      openQuestionStateMetadata(
        openQuestion({
          consecutive_refusals: 1,
          last_rumination_attempt_outcome: "completed",
          last_rumination_attempt_at: NOW_MS - 3 * DAY_MS,
        }),
        NOW_MS,
        4,
      )?.refusal_state,
    ).toEqual({
      consecutive_refusals: 1,
      last_rumination_attempt_outcome: "completed",
      last_rumination_attempt_at: new Date(NOW_MS - 3 * DAY_MS).toISOString(),
      last_rumination_attempt_relative_age: "3d ago",
      refusal_backoff_until: null,
    });
  });

  it("shows a deadline Date cannot represent as stored, without failing the row", () => {
    // toISOString() throws outside Date's range, and null in the deadline's place would say there
    // is no hold while the scheduler is honouring one.
    expect(
      openQuestionStateMetadata(
        openQuestion({ consecutive_refusals: 3, refusal_backoff_until: Number.MAX_VALUE }),
        NOW_MS,
        4,
      )?.refusal_state,
    ).toEqual({
      consecutive_refusals: 3,
      last_rumination_attempt_outcome: null,
      last_rumination_attempt_at: null,
      refusal_backoff_until: Number.MAX_VALUE,
    });
  });

  it("renders retained abandonment evidence and explains legacy cleared counters", () => {
    const abandoned = openQuestionStateMetadata(
      openQuestion({
        status: "abandoned",
        abandoned_reason: "stale",
        abandoned_at: NOW_MS,
        unresolved_rumination_ticks: 4,
        last_ruminated_at: NOW_MS - 1_000,
      }),
      NOW_MS,
      4,
      18,
    );

    expect(abandoned).toMatchObject({
      recorded_rumination_notes: 18,
      unresolved_rumination_ticks: 4,
      last_ruminated_at: new Date(NOW_MS - 1_000).toISOString(),
    });
    expect(abandoned?.rumination_record_note).toContain("Legacy closures may have cleared them");
    expect(abandoned?.rumination_record_note).toContain(
      "not that action or conversational progress was assessed",
    );
  });

  it("omits the threshold rather than inventing one when it was not supplied", () => {
    const metadata = openQuestionStateMetadata(
      openQuestion({ unresolved_rumination_ticks: 2 }),
      NOW_MS,
    );

    expect(metadata).toMatchObject({ unresolved_rumination_ticks: 2 });
    expect(metadata).not.toHaveProperty("dismissal_threshold_ticks");
  });

  it("leaves resolved and abandoned metadata alone", () => {
    const resolved = openQuestionStateMetadata(
      openQuestion({
        status: "resolved",
        resolution_note: "settled",
        resolved_at: NOW_MS,
        unresolved_rumination_ticks: 0,
      }),
      NOW_MS,
      4,
    );

    expect(resolved).not.toHaveProperty("unresolved_rumination_ticks");
    expect(resolved).toMatchObject({ resolution_note: "settled" });
  });
});
