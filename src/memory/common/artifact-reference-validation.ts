import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import type { ArtifactReference } from "./artifact-reference.js";
import { deliveredOutboundPostArtifactOutputSchema } from "./artifact-reference.js";
import { readStreamEntryAtOffset } from "../../stream/entry-lookup.js";
import type { SessionId } from "../../util/ids.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromMetadata,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "./disclosure-label.js";
import { openQuestionMemoryDisclosureLabel } from "./disclosure-serializers.js";

/** The same tagged record handles used by reflection, resolved across turns. */
export function artifactReferenceExists(
  db: SqliteDatabase,
  ref: ArtifactReference,
  nowMs: number,
  dataDir?: string,
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
      return recordedDeliveredPostExists(db, ref.id, nowMs, dataDir);
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

function recordedDeliveredPostExists(
  db: SqliteDatabase,
  messageId: string,
  nowMs: number,
  dataDir?: string,
): boolean {
  if (dataDir === undefined) return false;
  const message = db
    .prepare(
      "SELECT timestamp FROM stream_entry_index WHERE entry_id = ? AND kind = 'agent_msg' AND active = 1 AND receipt_pending = 0 AND timestamp <= ?",
    )
    .get(messageId, nowMs) as { timestamp: number } | undefined;
  if (message === undefined) return false;
  // The message is appended before transport. Only a successful tool result's
  // recorded delivered outcome supplies the same basis that reflection uses.
  const rows = db
    .prepare(
      "SELECT entry_id, session_id, byte_offset FROM stream_entry_index WHERE kind = 'tool_result' AND active = 1 AND receipt_pending = 0 AND timestamp >= ? AND timestamp <= ?",
    )
    .all(message.timestamp, nowMs) as {
    entry_id: string;
    session_id: SessionId;
    byte_offset: number;
  }[];
  return rows.some((row) => {
    const entry = readStreamEntryAtOffset({
      dataDir,
      sessionId: row.session_id,
      byteOffset: row.byte_offset,
    });
    if (
      entry?.id !== row.entry_id ||
      entry.content === null ||
      typeof entry.content !== "object" ||
      Array.isArray(entry.content)
    )
      return false;
    const content = entry.content as Record<string, unknown>;
    if (content.ok !== true) return false;
    const output = deliveredOutboundPostArtifactOutputSchema.safeParse(content.output);
    return output.success && output.data.outbound.delivery_outcome.agent_message_id === messageId;
  });
}

/** Read stored source policies; an artifact's existence never implies public disclosure. */
export function artifactReferenceDisclosureLabel(
  db: SqliteDatabase,
  ref: ArtifactReference,
): MemoryDisclosureLabel {
  if (ref.kind === "journal_entry") return selfPrivateMemoryDisclosureLabel();
  if (ref.kind === "created_open_question" || ref.kind === "resolved_open_question") {
    const row = db
      .prepare(
        "SELECT audience_entity_id, disclosure_label, resolution_disclosure_label FROM open_questions WHERE id = ?",
      )
      .get(ref.id) as
      | {
          audience_entity_id: import("../../util/ids.js").EntityId | null;
          disclosure_label: string | null;
          resolution_disclosure_label: string | null;
        }
      | undefined;
    if (row === undefined) return unknownMemoryDisclosureLabel();
    const labels = [
      openQuestionMemoryDisclosureLabel({
        audience_entity_id: row.audience_entity_id,
        disclosure_label:
          row.disclosure_label === null
            ? undefined
            : memoryDisclosureLabelFromMetadata(JSON.parse(row.disclosure_label)),
      }),
    ];
    if (ref.kind === "resolved_open_question")
      labels.push(
        row.resolution_disclosure_label === null
          ? unknownMemoryDisclosureLabel()
          : (memoryDisclosureLabelFromMetadata(JSON.parse(row.resolution_disclosure_label)) ??
              unknownMemoryDisclosureLabel()),
      );
    return combineMemoryDisclosureLabels(labels);
  }
  return unknownMemoryDisclosureLabel();
}
