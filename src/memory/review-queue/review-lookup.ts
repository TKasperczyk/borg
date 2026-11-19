import type { z } from "zod";
import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { reviewResolutionSchema, storedReviewKindSchema } from "./labels.js";

export type ReviewLookupRow = {
  id: number;
  kind: z.infer<typeof storedReviewKindSchema>;
} & (
  | { status: "open" }
  | {
      status: "resolved";
      resolution: z.infer<typeof reviewResolutionSchema> | null;
      resolved_at: string;
    }
);

export type ReviewLookup = {
  getByIds(ids: readonly number[]): readonly ReviewLookupRow[];
};

// The queue owns its storage; consumers receive only bounded lifecycle facts.
export function createReviewLookup(db: SqliteDatabase): ReviewLookup {
  return {
    getByIds(ids) {
      const distinctIds = [...new Set(ids)];
      if (distinctIds.length === 0) return [];
      // JSON keeps the primary-key IN lookup below SQLite's bind-variable limit
      // even for an unbounded question collection.
      const rows = db
        .prepare(
          "SELECT id, kind, resolved_at, resolution FROM review_queue WHERE id IN (SELECT value FROM json_each(?))",
        )
        .all(JSON.stringify(distinctIds));
      return rows.map((row): ReviewLookupRow => {
        const common = { id: Number(row.id), kind: storedReviewKindSchema.parse(row.kind) };
        return row.resolved_at == null
          ? { ...common, status: "open" }
          : {
              ...common,
              status: "resolved",
              resolution: reviewResolutionSchema.nullable().parse(row.resolution),
              resolved_at: new Date(Number(row.resolved_at)).toISOString(),
            };
      });
    },
  };
}
