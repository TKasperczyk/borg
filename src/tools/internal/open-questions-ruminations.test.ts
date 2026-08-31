import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { OpenQuestionsRepository, selfMigrations } from "../../memory/self/index.js";
import { FixedClock } from "../../util/clock.js";
import { createEpisodeId } from "../../util/ids.js";

import { createOpenQuestionsRuminationsTool } from "./open-questions-ruminations.js";

const manualProvenance = { kind: "manual" } as const;
const toolContext = { turnId: "turn_test" } as never;

function openRepository(): { db: ReturnType<typeof openDatabase>; repository: OpenQuestionsRepository } {
  const db = openDatabase(":memory:", { migrations: selfMigrations });

  return { db, repository: new OpenQuestionsRepository({ db, clock: new FixedClock(10_000) }) };
}

function toolFor(repository: OpenQuestionsRepository) {
  return createOpenQuestionsRuminationsTool({
    listRuminations: (input) => repository.listRuminationsInRange(input),
    getOpenQuestion: (id) => repository.get(id),
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
      expect(
        output.ruminations.map((row) => [row.question_status, row.note]),
      ).toEqual([
        ["abandoned", "Nothing has cited this since it was written."],
        ["resolved", "The pass kept it open on one unresolved tension."],
      ]);
      expect(output.has_more).toBe(false);
      expect(output.ruminations.every((row) => row.disclosure.length > 0)).toBe(true);
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
