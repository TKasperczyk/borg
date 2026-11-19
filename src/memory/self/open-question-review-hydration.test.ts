import { afterEach, describe, expect, it, vi } from "vitest";
import type { LanceDbTable } from "../../storage/lancedb/index.js";
import { openDatabase, SqliteStatement } from "../../storage/sqlite/index.js";
import { createGoalId, createSemanticNodeId } from "../../util/ids.js";
import { REVIEW_QUEUE_BASELINE_SQL } from "../review-queue/migrations.js";
import { ReviewQueueRepository } from "../review-queue/review-queue.js";
import { selfMigrations } from "./migrations.js";
import { OpenQuestionsRepository } from "./open-questions.js";

function fixture(linked = true) {
  const db = openDatabase(":memory:", { migrations: selfMigrations });
  db.exec(REVIEW_QUEUE_BASELINE_SQL);
  const queue = new ReviewQueueRepository({ db });
  const reviews = ["contradiction", "identity_inconsistency"].map((kind) =>
    queue.enqueue({
      kind: kind as "contradiction" | "identity_inconsistency",
      refs: {},
      reason: "Synthetic review",
    }),
  );
  const repository = new OpenQuestionsRepository({ db });
  const goalId = createGoalId();
  const nodeId = createSemanticNodeId();
  const questions = Array.from({ length: 25 }, (_, index) =>
    repository.add({
      question: `Synthetic hydration question ${index}?`,
      urgency: index / 25,
      source: "contradiction",
      review_id: linked ? reviews[index % 2]!.id : undefined,
      goal_id: goalId,
      related_semantic_node_ids: [nodeId],
    }),
  );
  return { db, repository, reviews, questions, goalId, nodeId };
}

afterEach(() => vi.restoreAllMocks());

describe("batched open-question review hydration", () => {
  it.each(["list", "listAllOpen", "listByGoal", "findByHandles"] as const)(
    "%s reads 25 questions and their distinct reviews in two queries",
    (method) => {
      const { db, repository, reviews, goalId, nodeId } = fixture();
      try {
        const reads = vi.spyOn(SqliteStatement.prototype, "all");
        const get = vi.spyOn(SqliteStatement.prototype, "get");
        const prepare = vi.spyOn(db, "prepare");
        const questions =
          method === "listByGoal"
            ? repository.listByGoal({ goalId })
            : method === "findByHandles"
              ? repository.findByHandles({ semanticNodeIds: [nodeId] })
              : repository[method]();
        expect(questions).toHaveLength(25);
        expect(reads).toHaveBeenCalledTimes(2);
        expect(get).not.toHaveBeenCalled();
        expect(prepare.mock.calls[1]?.[0]).toContain("WHERE id IN");
        expect(JSON.parse(String(reads.mock.calls[1]?.[0])).sort()).toEqual(
          reviews.map((r) => r.id).sort(),
        );
        for (const question of questions) {
          const review = reviews.find((r) => r.id === question.review_id)!;
          expect(question.review).toEqual({ id: review.id, kind: review.kind, status: "open" });
        }
      } finally {
        db.close();
      }
    },
  );

  it("batches vector-result hydration and preserves vector order", async () => {
    const { db, questions } = fixture();
    try {
      const vectorRows = [...questions].reverse().map((q) => ({ id: q.id, _distance: 0.1 }));
      const repository = new OpenQuestionsRepository({
        db,
        table: { search: async () => vectorRows } as unknown as LanceDbTable,
      });
      const reads = vi.spyOn(SqliteStatement.prototype, "all");
      const get = vi.spyOn(SqliteStatement.prototype, "get");
      const result = await repository.searchByVector(new Float32Array([1, 0]), { limit: 25 });
      expect(result.map(({ question }) => question.id)).toEqual(vectorRows.map((r) => r.id));
      expect(result.every(({ question }) => question.review?.status === "open")).toBe(true);
      expect(reads).toHaveBeenCalledTimes(2);
      expect(get).not.toHaveBeenCalled();
    } finally {
      db.close();
    }
  });

  it.each(["dismiss", "accept"] as const)(
    "sees %s on the next collection read without changing question state",
    (resolution) => {
      const { db, repository, reviews } = fixture();
      try {
        const before = repository.list();
        const at = Date.UTC(2026, 8, 2);
        db.prepare("UPDATE review_queue SET resolution = ?, resolved_at = ? WHERE id = ?").run(
          resolution,
          at,
          reviews[0]!.id,
        );
        const after = repository.list();
        expect(before.every((q) => q.review?.status === "open")).toBe(true);
        expect(after.filter((q) => q.review_id === reviews[0]!.id).map((q) => q.review)).toEqual(
          Array(13).fill({
            id: reviews[0]!.id,
            kind: reviews[0]!.kind,
            status: "resolved",
            resolution,
            resolved_at: new Date(at).toISOString(),
          }),
        );
        expect(after.map(({ review, ...q }) => q)).toEqual(before.map(({ review, ...q }) => q));
      } finally {
        db.close();
      }
    },
  );

  it("does not query reviews for unlinked or empty collections", () => {
    const { db, repository } = fixture(false);
    try {
      const reads = vi.spyOn(SqliteStatement.prototype, "all");
      expect(repository.list()).toHaveLength(25);
      expect(reads).toHaveBeenCalledTimes(1);
      reads.mockClear();
      expect(repository.list({ status: "resolved" })).toEqual([]);
      expect(reads).toHaveBeenCalledTimes(1);
    } finally {
      db.close();
    }
  });

  it("keeps missing review ids labeled within a mixed collection", () => {
    const { db, repository, questions } = fixture();
    try {
      db.prepare("UPDATE open_questions SET review_id = ? WHERE id = ?").run(
        999_001,
        questions[0]!.id,
      );
      db.prepare("UPDATE open_questions SET review_id = NULL WHERE id = ?").run(questions[1]!.id);
      const reads = vi.spyOn(SqliteStatement.prototype, "all");
      const rows = repository.list();
      expect(reads).toHaveBeenCalledTimes(2);
      expect(rows.find((q) => q.id === questions[0]!.id)?.review).toEqual({
        id: 999_001,
        kind: null,
        status: "missing",
      });
      expect(rows.find((q) => q.id === questions[1]!.id)).not.toHaveProperty("review");
    } finally {
      db.close();
    }
  });
});
