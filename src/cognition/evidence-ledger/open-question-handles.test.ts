import { describe, expect, it } from "vitest";

import type { OpenQuestion } from "../../memory/self/index.js";
import type { OpenQuestionId } from "../../util/ids.js";
import { openQuestionStateMetadata } from "./open-question-handles.js";

const NOW_MS = Date.parse("2026-09-02T12:00:00.000Z");

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
    // The store advances this only when a note is recorded (stampRuminationForRun's note-less
    // branch writes last_ruminated_at alone), so a sentence that calls it a count of passes
    // invites reading a gap between the count and the notes as a lost note.
    expect(live?.unresolved_rumination_ticks_note).toContain("recorded no note");
    // Both fields are on the object, so the one inference the pair supports belongs beside them:
    // a stamp later than the newest note is a pass that recorded nothing, and the page should not
    // leave a reader to discover that by drawing the notes and comparing dates by hand.
    expect(live?.unresolved_rumination_ticks_note).toContain(
      "later than the newest note tool.openQuestions.ruminations returns",
    );
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
