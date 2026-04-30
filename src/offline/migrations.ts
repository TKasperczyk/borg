import type { Migration } from "../storage/sqlite/index.js";

export const offlineMigrations = [
  {
    id: 1,
    name: "offline_initial_schema",
    up: (db) => {
      db.exec(`
        CREATE TABLE maintenance_audit (
          id INTEGER PRIMARY KEY,
          run_id TEXT NOT NULL,
          process TEXT NOT NULL,
          action TEXT NOT NULL,
          targets TEXT NOT NULL,
          reversal TEXT NOT NULL,
          applied_at INTEGER NOT NULL,
          reverted_at INTEGER,
          reverted_by TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_maintenance_audit_run_id
          ON maintenance_audit (run_id);
        CREATE INDEX IF NOT EXISTS idx_maintenance_audit_process
          ON maintenance_audit (process);
        CREATE INDEX IF NOT EXISTS idx_maintenance_audit_reverted
          ON maintenance_audit (reverted_at);
      `);
    },
  },
] as const satisfies readonly Migration[];
