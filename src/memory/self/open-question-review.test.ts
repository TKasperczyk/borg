import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { openDatabase } from "../../storage/sqlite/index.js";
import { parseOpenQuestionId } from "../../util/ids.js";
import { REVIEW_QUEUE_BASELINE_SQL } from "../review-queue/migrations.js";
import { ReviewQueueRepository } from "../review-queue/review-queue.js";
import { createReviewLookup } from "../review-queue/review-lookup.js";
import { selfMigrations } from "./migrations.js";
import { OpenQuestionsRepository } from "./open-questions.js";
import { openQuestionSchema } from "./types.js";
import { openQuestionReviewFields, renderOpenQuestionReview } from "./open-question-review.js";
import { renderQuestionSurfaces, REVIEW_TEST_NOW } from "./open-question-review-test-support.js";
import fixture from "./__fixtures__/open-question-review-dev.json";

const root = fileURLToPath(new URL("../../../", import.meta.url)).replace(/\/$/, "");
const question = openQuestionSchema.parse(fixture.question);

describe("review lifecycle labels", () => {
  it.each([null, "dismiss", "accept"] as const)(
    "renders %s on every question surface without closing the question",
    async (resolution) => {
      const db = openDatabase(":memory:", { migrations: selfMigrations });
      db.exec(REVIEW_QUEUE_BASELINE_SQL);
      try {
        const kind = resolution === "accept" ? "identity_inconsistency" : "contradiction";
        const queue = new ReviewQueueRepository({
          db,
          clock: { now: () => REVIEW_TEST_NOW - 100_000 },
        });
        const review = queue.enqueue({
          kind,
          refs: { synthetic_payload: "x".repeat(100_000) },
          reason: "unbounded-reason".repeat(10_000),
        });
        const repository = new OpenQuestionsRepository({
          db,
          reviewLookup: createReviewLookup(db),
          clock: { now: () => REVIEW_TEST_NOW - 99_000 },
        });
        const added = repository.add({
          id: question.id,
          question: question.question,
          urgency: 0.5,
          source: kind === "contradiction" ? "contradiction" : "overseer",
          provenance: { kind: "manual" },
          review_id: review.id,
        });
        expect(added.review).toEqual({ id: review.id, kind, status: "open" });
        const storedBefore = db.prepare("SELECT * FROM open_questions").get();
        if (resolution !== null)
          db.prepare("UPDATE review_queue SET resolution = ?, resolved_at = ? WHERE id = ?").run(
            resolution,
            REVIEW_TEST_NOW,
            review.id,
          );
        const current = repository.get(added.id)!;
        const expected =
          resolution === null
            ? { id: review.id, kind, status: "open" }
            : {
                id: review.id,
                kind,
                status: "resolved",
                resolution,
                resolved_at: new Date(REVIEW_TEST_NOW).toISOString(),
              };
        expect(current.review).toEqual(expected);
        expect(repository.list({ status: "open" })[0]?.review).toEqual(expected);
        expect(current.status).toBe("open");
        expect(db.prepare("SELECT * FROM open_questions").get()).toEqual(storedBefore);
        expect(JSON.stringify(openQuestionReviewFields(current)).length).toBeLessThan(256);
        const surfaces = await renderQuestionSurfaces(root, [current]);
        for (const [name, text] of Object.entries(surfaces)) {
          // This compact slice does not admit open questions in dev or in this change.
          if (name === "planner_ledger") {
            expect(text).toBe(fixture.outputs.planner_ledger);
            continue;
          }
          expect(text, name).toContain("review");
          expect(text, name).toContain(kind);
          expect(text, name).not.toContain("synthetic_payload");
          expect(text, name).not.toContain("unbounded-reason");
          if (resolution !== null) {
            expect(text, name).toContain(resolution);
            expect(text, name).toContain(new Date(REVIEW_TEST_NOW).toISOString());
          }
        }
      } finally {
        db.close();
      }
    },
  );

  it("labels a missing review without losing its question or silently declaring it resolved", () => {
    const db = openDatabase(":memory:", { migrations: selfMigrations });
    db.exec(REVIEW_QUEUE_BASELINE_SQL);
    try {
      const repository = new OpenQuestionsRepository({ db, reviewLookup: createReviewLookup(db) });
      const row = repository.add({
        question: "Synthetic missing review?",
        urgency: 0.5,
        source: "contradiction",
        provenance: { kind: "manual" },
        review_id: 42,
      });
      expect(row.review).toEqual({ id: 42, kind: null, status: "missing" });
      expect(repository.get(row.id)?.status).toBe("open");
    } finally {
      db.close();
    }
    expect(renderOpenQuestionReview({ review_id: 42 })).toBe("");
  });

  it("keeps review-linked questions usable and unlabeled without an injected lookup or queue table", async () => {
    const db = openDatabase(":memory:", { migrations: selfMigrations });
    try {
      const repository = new OpenQuestionsRepository({ db });
      const added = repository.add({
        id: question.id,
        question: question.question,
        urgency: 0.5,
        source: "contradiction",
        provenance: { kind: "manual" },
        review_id: 42,
      });
      const current = repository.get(added.id)!;
      for (const row of [added, current, ...repository.list(), ...repository.listAllOpen()]) {
        expect(row.review_id).toBe(42);
        expect(row).not.toHaveProperty("review");
        expect(openQuestionReviewFields(row)).toEqual({});
        expect(renderOpenQuestionReview(row)).toBe("");
      }
      const { review_id: _, ...unlinked } = current;
      expect(await renderQuestionSurfaces(root, [current])).toEqual(
        await renderQuestionSurfaces(root, [unlinked]),
      );
    } finally {
      db.close();
    }
  });

  it("keeps dormant and planner row caps and complete bounded labels with long questions", async () => {
    const questions = Array.from({ length: 25 }, (_, index) => ({
      ...question,
      id: parseOpenQuestionId(`oq_${String(index).padStart(16, "0")}`),
      question: `Synthetic ${index} ${"原文".repeat(20_000)}`,
      review_id: index + 1,
      review: {
        id: index + 1,
        kind: "contradiction" as const,
        status: "resolved" as const,
        resolution: "dismiss" as const,
        resolved_at: new Date(REVIEW_TEST_NOW).toISOString(),
      },
    }));
    const surfaces = await renderQuestionSurfaces(root, questions);
    expect(surfaces.planner!.match(/<open_loop_row /g)?.length).toBe(20);
    expect(surfaces.planner).toContain("dismiss");
    expect(surfaces.planner).toContain(new Date(REVIEW_TEST_NOW).toISOString());
    expect(surfaces.wake!.match(/open_question_id/g)?.length).toBeLessThanOrEqual(3);
    expect(surfaces.wake!.length).toBeLessThanOrEqual(32_000);
    expect(surfaces.wake).toContain("dismiss");
    expect(surfaces.wake).toContain(new Date(REVIEW_TEST_NOW).toISOString());
  });
});

describe("complete pages versus dev cc7877c4 (no linked review)", () => {
  let populated: Record<string, string>;
  let empty: Record<string, string>;
  beforeAll(async () => {
    populated = await renderQuestionSurfaces(root, [question]);
    empty = await renderQuestionSurfaces(root, []);
  });
  // Frozen from dev source using synthetic inputs. No stripping or normalization.
  for (const [name, expected] of Object.entries(fixture.outputs)) {
    it(`${name} with an unlinked question is byte-identical`, () => {
      expect(Buffer.from(populated[name]!)).toEqual(Buffer.from(expected));
    });
  }
  for (const [name, expected] of Object.entries(fixture.emptyOutputs)) {
    it(`${name} with no questions is byte-identical`, () => {
      expect(Buffer.from(empty[name]!)).toEqual(Buffer.from(expected));
    });
  }
});
