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
    // The count alone never closes a question -- the two structural conditions are the rest of
    // the predicate, and a page that printed the count without them would read as a countdown.
    expect(live?.unresolved_rumination_ticks_note).toContain(
      "no episode created after the question citing it",
    );
    expect(live?.unresolved_rumination_ticks_note).toContain("no active action against it");
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
