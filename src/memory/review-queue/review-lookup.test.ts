import { afterEach, describe, expect, it, vi } from "vitest";
import { openDatabase, SqliteStatement } from "../../storage/sqlite/index.js";
import { REVIEW_QUEUE_BASELINE_SQL } from "./migrations.js";
import { createReviewLookup } from "./review-lookup.js";
import { ReviewQueueRepository } from "./review-queue.js";

afterEach(() => vi.restoreAllMocks());

describe("review lifecycle lookup", () => {
  it("reads distinct ids in one primary-key IN query and omits missing ids and unbounded payloads", () => {
    const db = openDatabase(":memory:");
    db.exec(REVIEW_QUEUE_BASELINE_SQL);
    try {
      const queue = new ReviewQueueRepository({ db });
      const first = queue.enqueue({
        kind: "contradiction",
        refs: { synthetic: "x".repeat(10_000) },
        reason: "r".repeat(10_000),
      });
      const second = queue.enqueue({
        kind: "identity_inconsistency",
        refs: {},
        reason: "Synthetic review",
      });
      const lookup = createReviewLookup(db);
      const prepare = vi.spyOn(db, "prepare");
      const all = vi.spyOn(SqliteStatement.prototype, "all");
      const result = lookup.getByIds([first.id, second.id, first.id, 999_001]);
      expect(result).toEqual([
        { id: first.id, kind: "contradiction", status: "open" },
        { id: second.id, kind: "identity_inconsistency", status: "open" },
      ]);
      expect(prepare).toHaveBeenCalledTimes(1);
      expect(all).toHaveBeenCalledTimes(1);
      const sql = prepare.mock.calls[0]![0];
      expect(sql).toContain("WHERE id IN");
      expect(JSON.parse(String(all.mock.calls[0]![0]))).toEqual([first.id, second.id, 999_001]);
      const plan = db
        .prepare(`EXPLAIN QUERY PLAN ${sql}`)
        .all(JSON.stringify([first.id, second.id]));
      expect(plan.some((row) => String(row.detail).includes("USING INTEGER PRIMARY KEY"))).toBe(
        true,
      );
    } finally {
      db.close();
    }
  });

  it.each([null, "dismiss", "accept"] as const)(
    "returns fresh resolved %s facts with an ISO timestamp",
    (resolution) => {
      const db = openDatabase(":memory:");
      db.exec(REVIEW_QUEUE_BASELINE_SQL);
      try {
        const item = new ReviewQueueRepository({ db }).enqueue({
          kind: "contradiction",
          refs: {},
          reason: "Synthetic review",
        });
        const lookup = createReviewLookup(db);
        expect(lookup.getByIds([item.id])).toEqual([
          { id: item.id, kind: item.kind, status: "open" },
        ]);
        const at = Date.UTC(2026, 8, 2);
        db.prepare("UPDATE review_queue SET resolution = ?, resolved_at = ? WHERE id = ?").run(
          resolution,
          at,
          item.id,
        );
        expect(lookup.getByIds([item.id])).toEqual([
          {
            id: item.id,
            kind: item.kind,
            status: "resolved",
            resolution,
            resolved_at: new Date(at).toISOString(),
          },
        ]);
      } finally {
        db.close();
      }
    },
  );

  it("does not query for an empty id collection", () => {
    const db = openDatabase(":memory:");
    try {
      const prepare = vi.spyOn(db, "prepare");
      expect(createReviewLookup(db).getByIds([])).toEqual([]);
      expect(prepare).not.toHaveBeenCalled();
    } finally {
      db.close();
    }
  });
});
