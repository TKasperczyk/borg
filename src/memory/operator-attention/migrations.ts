import type { Migration } from "../../storage/sqlite/index.js";

export const operatorAttentionMigrations = [
  {
    id: 1,
    name: "operator_attention_index",
    up: `
      CREATE TABLE operator_attention_records (
        record_key TEXT PRIMARY KEY,
        filed_at INTEGER NOT NULL,
        filer_entity_id TEXT NOT NULL,
        subject TEXT
      );
      CREATE INDEX idx_operator_attention_latest
        ON operator_attention_records (filed_at DESC, record_key DESC);
    `,
  },
] as const satisfies readonly Migration[];
