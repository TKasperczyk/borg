import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { StorageError } from "../../util/errors.js";
import { operatorAttentionPromptRow } from "./disclosure.js";
import {
  OPERATOR_ATTENTION_RECENT_LIMIT,
  operatorAttentionRecordSchema,
  type OperatorAttentionIndex,
  type OperatorAttentionRecord,
} from "./types.js";

export class OperatorAttentionRepository {
  constructor(private readonly options: { db: SqliteDatabase }) {}

  /** First filing wins. A resend, including an existence-only backfill, cannot rewrite it. */
  record(input: OperatorAttentionRecord): { inserted: boolean } {
    const record = operatorAttentionRecordSchema.parse(input);
    const result = this.options.db
      .prepare(
        `
      INSERT INTO operator_attention_records (record_key, filed_at, filer_entity_id, subject)
      VALUES (?, ?, ?, ?)
      ON CONFLICT (record_key) DO NOTHING
    `,
      )
      .run(record.record_key, record.filed_at, record.filer_entity_id, record.subject);
    return { inserted: result.changes > 0 };
  }

  snapshot(): OperatorAttentionIndex {
    const total = Number(
      this.options.db.prepare("SELECT COUNT(*) AS total FROM operator_attention_records").get()
        ?.total ?? 0,
    );
    const records = this.options.db
      .prepare(
        `
      SELECT record_key, filed_at, filer_entity_id, subject
      FROM operator_attention_records
      ORDER BY filed_at DESC, record_key DESC
      LIMIT ?
    `,
      )
      .all(OPERATOR_ATTENTION_RECENT_LIMIT)
      .map((row) => {
        const parsed = operatorAttentionRecordSchema.safeParse(row);
        if (!parsed.success) {
          throw new StorageError("Operator attention index row failed validation", {
            code: "OPERATOR_ATTENTION_ROW_INVALID",
            cause: parsed.error,
          });
        }
        return operatorAttentionPromptRow(parsed.data);
      });
    return { total, records };
  }
}
