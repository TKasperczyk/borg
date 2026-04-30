import type { Migration } from "../../storage/sqlite/index.js";
import { tableHasColumn } from "../../storage/sqlite/migrations-utils.js";

export const commitmentMigrations: Migration[] = [
  {
    id: 140,
    name: "entities_and_commitments",
    up: `
      CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY,
        canonical_name TEXT NOT NULL,
        aliases TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS entities_name_idx
        ON entities(canonical_name);

      CREATE TABLE IF NOT EXISTS commitments (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        directive TEXT NOT NULL,
        priority INTEGER NOT NULL,
        made_to_entity TEXT NULL,
        restricted_audience TEXT NULL,
        about_entity TEXT NULL,
        source_episode_ids TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        expires_at INTEGER NULL,
        revoked_at INTEGER NULL,
        superseded_by TEXT NULL
      );

      CREATE INDEX IF NOT EXISTS commitments_audience_idx
        ON commitments(restricted_audience);
      CREATE INDEX IF NOT EXISTS commitments_about_idx
        ON commitments(about_entity);
    `,
  },
  {
    id: 210,
    name: "add-commitment-provenance",
    up: (db) => {
      if (!tableHasColumn(db, "commitments", "provenance_kind")) {
        db.exec("ALTER TABLE commitments ADD COLUMN provenance_kind TEXT");
      }
      if (!tableHasColumn(db, "commitments", "provenance_episode_ids")) {
        db.exec("ALTER TABLE commitments ADD COLUMN provenance_episode_ids TEXT");
      }
      if (!tableHasColumn(db, "commitments", "provenance_process")) {
        db.exec("ALTER TABLE commitments ADD COLUMN provenance_process TEXT");
      }
    },
  },
  {
    id: 211,
    name: "add-commitment-lifecycle-columns",
    up: (db) => {
      if (!tableHasColumn(db, "commitments", "expired_at")) {
        db.exec("ALTER TABLE commitments ADD COLUMN expired_at INTEGER");
      }
      if (!tableHasColumn(db, "commitments", "revoked_reason")) {
        db.exec("ALTER TABLE commitments ADD COLUMN revoked_reason TEXT");
      }
      if (!tableHasColumn(db, "commitments", "revoke_provenance_kind")) {
        db.exec("ALTER TABLE commitments ADD COLUMN revoke_provenance_kind TEXT");
      }
      if (!tableHasColumn(db, "commitments", "revoke_provenance_episode_ids")) {
        db.exec("ALTER TABLE commitments ADD COLUMN revoke_provenance_episode_ids TEXT");
      }
      if (!tableHasColumn(db, "commitments", "revoke_provenance_process")) {
        db.exec("ALTER TABLE commitments ADD COLUMN revoke_provenance_process TEXT");
      }

      db.exec(`
        UPDATE commitments
        SET expired_at = expires_at
        WHERE expired_at IS NULL
          AND expires_at IS NOT NULL
          AND revoked_at IS NULL
          AND superseded_by IS NULL
      `);
    },
  },
];
