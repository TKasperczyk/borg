import type { Migration } from "../../storage/sqlite/index.js";

export const relationalSlotMigrations = [
  {
    id: 1,
    name: "relational_slots_initial_schema",
    up: `
      CREATE TABLE IF NOT EXISTS relational_slots (
        id TEXT PRIMARY KEY,
        subject_entity_id TEXT NOT NULL,
        slot_key TEXT NOT NULL,
        value TEXT NOT NULL,
        state TEXT NOT NULL CHECK (
          state IN ('established', 'contested', 'quarantined', 'revoked')
        ),
        evidence_stream_entry_ids TEXT NOT NULL,
        contradicted_by_stream_entry_ids TEXT NOT NULL,
        alternate_values TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        UNIQUE(subject_entity_id, slot_key)
      );

      CREATE INDEX IF NOT EXISTS relational_slots_subject_slot_idx
        ON relational_slots(subject_entity_id, slot_key);
      CREATE INDEX IF NOT EXISTS relational_slots_state_idx
        ON relational_slots(state, updated_at DESC);
    `,
  },
] as const satisfies readonly Migration[];
