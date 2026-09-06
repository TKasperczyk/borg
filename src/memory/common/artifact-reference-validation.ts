import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import type { ArtifactReference } from "./artifact-reference.js";

/** The same tagged record handles used by reflection, resolved across turns. */
export function artifactReferenceExists(
  db: SqliteDatabase,
  ref: ArtifactReference,
  nowMs: number,
): boolean {
  switch (ref.kind) {
    case "journal_entry":
      return (
        db
          .prepare(
            "SELECT 1 FROM train_of_thought_journal_entries WHERE id = ? AND created_at <= ?",
          )
          .get(ref.id, nowMs) !== undefined
      );
    case "created_open_question":
      return (
        db
          .prepare("SELECT 1 FROM open_questions WHERE id = ? AND created_at <= ?")
          .get(ref.id, nowMs) !== undefined
      );
    case "resolved_open_question":
      return (
        db
          .prepare(
            "SELECT 1 FROM open_questions WHERE id = ? AND status = 'resolved' AND resolved_at <= ?",
          )
          .get(ref.id, nowMs) !== undefined
      );
    case "scheduled_wake":
      return (
        db
          .prepare("SELECT 1 FROM scheduled_wakes WHERE id = ? AND created_at <= ?")
          .get(ref.id, nowMs) !== undefined
      );
    case "executive_step_outcome":
      return (
        db
          .prepare(
            "SELECT 1 FROM executive_steps WHERE id = ? AND status IN ('done', 'blocked', 'abandoned') AND updated_at <= ?",
          )
          .get(ref.id, nowMs) !== undefined
      );
    case "delivered_outbound_post":
      return (
        db
          .prepare(
            "SELECT 1 FROM stream_entry_index WHERE entry_id = ? AND kind = 'agent_msg' AND active = 1 AND receipt_pending = 0 AND timestamp <= ?",
          )
          .get(ref.id, nowMs) !== undefined
      );
    case "stream_entry":
      return (
        db
          .prepare(
            "SELECT 1 FROM stream_entry_index WHERE entry_id = ? AND active = 1 AND receipt_pending = 0 AND timestamp <= ?",
          )
          .get(ref.id, nowMs) !== undefined
      );
  }
}
