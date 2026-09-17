import { describe, expect, it } from "vitest";

import { DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET } from "../../cognition/deliberation/constants.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { OpenQuestionsRepository, selfMigrations } from "../../memory/self/index.js";
import { FixedClock } from "../../util/clock.js";
import { createEpisodeId, createMaintenanceRunId } from "../../util/ids.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";

import { createOpenQuestionsRuminationsTool } from "./open-questions-ruminations.js";

const manualProvenance = { kind: "manual" } as const;
const toolContext = { turnId: "turn_test" } as never;

function openRepository(): {
  db: ReturnType<typeof openDatabase>;
  repository: OpenQuestionsRepository;
} {
  const db = openDatabase(":memory:", { migrations: selfMigrations });

  return { db, repository: new OpenQuestionsRepository({ db, clock: new FixedClock(10_000) }) };
}

function toolFor(repository: OpenQuestionsRepository) {
  return createOpenQuestionsRuminationsTool({
    listRuminations: (input) => repository.listRuminationsInRange(input),
    getOpenQuestion: (id) => repository.get(id),
    countRuminations: (id) => repository.countRuminations(id),
    clock: new FixedClock(100_000),
  });
}

describe("tool.openQuestions.ruminations", () => {
  it("reaches notes written against a question that has since closed", async () => {
    const { db, repository } = openRepository();

    try {
      const resolved = repository.add({
        question: "Did the earlier pass settle this?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });
      const abandoned = repository.add({
        question: "Was this worth carrying?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });
      repository.recordRumination({
        open_question_id: resolved.id,
        note: "The pass kept it open on one unresolved tension.",
        source_process: "offline_ruminator",
        provenance: manualProvenance,
        created_at: 11_000,
      });
      repository.recordRumination({
        open_question_id: abandoned.id,
        note: "Nothing has cited this since it was written.",
        source_process: "offline_ruminator",
        provenance: manualProvenance,
        created_at: 12_000,
      });
      repository.resolve(resolved.id, {
        resolution_note: "Settled.",
        resolution_evidence_episode_ids: [createEpisodeId()],
        resolution_evidence_stream_entry_ids: [],
      });
      repository.abandon(abandoned.id, "stale_no_traction");

      const output = await toolFor(repository).invoke(
        { since: new Date(0).toISOString(), until: new Date(20_000).toISOString() },
        toolContext,
      );

      // The whole point of the reader: closing a question does not take its ruminations with it,
      // and the row says which way the question went so the note is legible without a second lookup.
      expect(output.ruminations.map((row) => [row.question_status, row.note])).toEqual([
        ["abandoned", "Nothing has cited this since it was written."],
        ["resolved", "The pass kept it open on one unresolved tension."],
      ]);
      expect(output.has_more).toBe(false);
      expect(output.ruminations.every((row) => row.disclosure.length > 0)).toBe(true);
    } finally {
      db.close();
    }
  });

  it("returns nothing for a pass that stamped the run without writing a note", async () => {
    const { db, repository } = openRepository();

    try {
      const question = repository.add({
        question: "Did the loop reach this one at all?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });
      const stamped = repository.stampRuminationForRun({
        open_question_id: question.id,
        source_run_id: createMaintenanceRunId(),
        next_unresolved_rumination_ticks: 1,
        rumination: null,
      });

      const output = await toolFor(repository).invoke(
        { since: new Date(0).toISOString(), until: new Date(20_000).toISOString() },
        toolContext,
      );

      // The pass reached the question -- it stamped the run and moved last_ruminated_at -- and this
      // reader still returns nothing. The tick counter does not separate the two states either, so
      // an empty range here is a count of notes and not a count of passes, which is what the
      // description now says rather than leaving the reader to infer silence from the emptiness.
      expect(stamped.stamped).toBe(true);
      expect(stamped.question.last_ruminated_at).toBe(10_000);
      expect(stamped.question.unresolved_rumination_ticks).toBe(0);
      expect(output.ruminations).toEqual([]);
      expect(output.has_more).toBe(false);
    } finally {
      db.close();
    }
  });

  it("filters by created-at range and by a single question id", async () => {
    const { db, repository } = openRepository();

    try {
      const first = repository.add({
        question: "First question?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });
      const second = repository.add({
        question: "Second question?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });

      for (const [question, note, createdAt] of [
        [first, "first-old", 11_000],
        [first, "first-new", 13_000],
        [second, "second-mid", 12_000],
      ] as const) {
        repository.recordRumination({
          open_question_id: question.id,
          note,
          source_process: "offline_ruminator",
          provenance: manualProvenance,
          created_at: createdAt,
        });
      }

      const tool = toolFor(repository);
      const windowed = await tool.invoke(
        { since: new Date(11_500).toISOString(), until: new Date(20_000).toISOString() },
        toolContext,
      );
      const scoped = await tool.invoke(
        {
          since: new Date(0).toISOString(),
          until: new Date(20_000).toISOString(),
          open_question_id: first.id,
        },
        toolContext,
      );

      expect(windowed.ruminations.map((row) => row.note)).toEqual(["first-new", "second-mid"]);
      expect(scoped.ruminations.map((row) => row.note)).toEqual(["first-new", "first-old"]);
    } finally {
      db.close();
    }
  });

  it("reports has_more when the range holds more than the requested limit", async () => {
    const { db, repository } = openRepository();

    try {
      const question = repository.add({
        question: "How deep does this go?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });

      for (let index = 0; index < 3; index += 1) {
        repository.recordRumination({
          open_question_id: question.id,
          note: `pass-${index}`,
          source_process: "offline_ruminator",
          provenance: manualProvenance,
          created_at: 11_000 + index,
        });
      }

      const output = await toolFor(repository).invoke(
        { since: new Date(0).toISOString(), until: new Date(20_000).toISOString(), limit: 2 },
        toolContext,
      );

      expect(output.ruminations.map((row) => row.note)).toEqual(["pass-2", "pass-1"]);
      expect(output.has_more).toBe(true);
      // has_more says the page was cut; it never says by how much. The per-row count is the
      // record's size, so two delivered out of three still report three.
      expect(output.ruminations.map((row) => row.question_recorded_rumination_notes)).toEqual([
        3, 3,
      ]);
    } finally {
      db.close();
    }
  });

  it.each([50, 51])("looks ahead at the maximum page size with %i stored notes", async (count) => {
    const { db, repository } = openRepository();
    try {
      const question = repository.add({
        question: "Why?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });
      for (let index = 0; index < count; index += 1) {
        repository.recordRumination({
          open_question_id: question.id,
          note: `pass-${index}`,
          source_process: "offline_ruminator",
          provenance: manualProvenance,
          created_at: 11_000 + index,
        });
      }
      const tool = toolFor(repository);
      const output = await tool.invoke(
        { since: new Date(0).toISOString(), until: new Date(20_000).toISOString(), limit: 50 },
        toolContext,
      );

      expect(output.ruminations).toHaveLength(50);
      expect(output.ruminations.every((row) => row.payload_status === "exact")).toBe(true);
      expect(
        output.ruminations.every((row) => row.question_recorded_rumination_notes === count),
      ).toBe(true);
      expect(output.has_more).toBe(count > 50);
      expect(tool.outputSchema.safeParse(output).success).toBe(true);
    } finally {
      db.close();
    }
  });

  it("bounds the first-row fallback even when retained fields exceed the budget", async () => {
    const { db, repository } = openRepository();
    try {
      const question = repository.add({
        question: "What remains readable?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });
      const hugeField = "界".repeat(DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET * 5);
      const connectedQuestion = repository.add({
        question: "How does this connect?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });
      const stored = repository.recordRumination({
        open_question_id: question.id,
        note: "A short note.",
        tensions: [hugeField],
        connected_open_question_ids: [connectedQuestion.id],
        source_process: hugeField,
        provenance: manualProvenance,
        created_at: 11_000,
      });
      expect(stored.connected_open_question_ids).toEqual([connectedQuestion.id]);
      const tool = toolFor(repository);
      const output = await tool.invoke(
        { since: new Date(0).toISOString(), until: new Date(20_000).toISOString() },
        toolContext,
      );

      expect(output.ruminations).toEqual([
        expect.objectContaining({
          question: null,
          note: null,
          tensions: [],
          connected_open_question_ids: [],
          source_process: null,
          question_recorded_rumination_notes: 1,
          payload_status: "check_not_completed_budget",
          payload_included_chars: 0,
          disclosure_label: expect.objectContaining({ disclosure_class: "unknown" }),
        }),
      ]);
      expect(output.ruminations[0]!.payload_total_chars).toBeGreaterThan(hugeField.length);
      expect(output.has_more).toBe(false);
      expect(estimatePromptTokens(JSON.stringify(output))).toBeLessThanOrEqual(
        DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET,
      );
      expect(tool.outputSchema.safeParse(output).success).toBe(true);
    } finally {
      db.close();
    }
  });

  it("counts a question's whole record even when the range excludes most of it", async () => {
    const { db, repository } = openRepository();

    try {
      const question = repository.add({
        question: "How much of this am I holding?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });

      for (let index = 0; index < 5; index += 1) {
        repository.recordRumination({
          open_question_id: question.id,
          note: `pass-${index}`,
          source_process: "offline_ruminator",
          provenance: manualProvenance,
          created_at: 11_000 + index,
        });
      }

      const output = await toolFor(repository).invoke(
        { since: new Date(11_003).toISOString(), until: new Date(20_000).toISOString() },
        toolContext,
      );

      // A range cut leaves has_more false, so nothing on the page marks the tail at all. Without
      // the count, two notes read as the whole record rather than as the newest two of five.
      expect(output.ruminations.map((row) => row.note)).toEqual(["pass-4", "pass-3"]);
      expect(output.has_more).toBe(false);
      expect(output.ruminations.map((row) => row.question_recorded_rumination_notes)).toEqual([
        5, 5,
      ]);
    } finally {
      db.close();
    }
  });

  it("names how the loop abandons a question, since these notes are that reasoning's only record", () => {
    const { db, repository } = openRepository();

    try {
      const { description } = toolFor(repository);

      expect(description).toContain("still-open passes reach the no-traction threshold");
      expect(description).toContain("no episode created after it cites it");
      // 27716714 dropped hasActiveAssociatedAction from shouldDismissStaleNoTraction
      // (src/offline/ruminator/index.ts), leaving two conjuncts. A description naming a third
      // makes a still-open row past the threshold readable as an active action holding it open.
      expect(description).not.toContain("no action against it is active");
      // The conjuncts alone read as a test every pass applies. Both call sites reach
      // shouldDismissStaleNoTraction only where the model was not consulted for this question:
      // planResolution's two null returns are both upstream of onLlmVisit, and the sweep over the
      // remaining questions skips the LLM window outright. Without the gate, a still-open row
      // sitting past the threshold has to be read as one of the other conjuncts having fired.
      expect(description).toContain("evaluated only on a pass that did not call the model");
    } finally {
      db.close();
    }
  });

  it("rejects an inverted range", async () => {
    const { db, repository } = openRepository();

    try {
      expect(() =>
        toolFor(repository).inputSchema.parse({
          since: new Date(20_000).toISOString(),
          until: new Date(10_000).toISOString(),
        }),
      ).toThrow();
    } finally {
      db.close();
    }
  });
});
