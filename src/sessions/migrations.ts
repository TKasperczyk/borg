import type { Migration } from "../storage/sqlite/index.js";

export const sessionMigrations = [
  {
    id: 1,
    name: "session_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE sessions (
          session_id TEXT PRIMARY KEY,
          -- source_type is an open routing/label key (connectors register their own); shape is
          -- validated at the app layer (sessionSourceTypeSchema slug), not enumerated here.
          source_type TEXT NOT NULL,
          source_external_id TEXT,
          source_url TEXT,
          label TEXT NOT NULL,
          audience_label TEXT NOT NULL,
          audience_entity_id TEXT,
          conversation_kind TEXT NOT NULL CHECK (
            conversation_kind IN ('dm', 'channel', 'thread', 'demo')
          ),
          created_at INTEGER NOT NULL,
          last_activity_at INTEGER NOT NULL,
          last_turn_id TEXT,
          message_count INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL CHECK (status IN ('active', 'idle', 'archived')),
          privacy_level TEXT NOT NULL DEFAULT 'payload_off' CHECK (
            privacy_level IN ('payload_off', 'payload_on')
          )
        , participation_policy TEXT NOT NULL DEFAULT 'active' CHECK (
            participation_policy IN ('active', 'paused', 'observing', 'muted')
          ), audience_role TEXT NOT NULL DEFAULT 'participant' CHECK (
            audience_role IN ('participant', 'operator')
          ));
        CREATE INDEX idx_sessions_last_activity
          ON sessions (last_activity_at);
        CREATE INDEX idx_sessions_source_type_last_activity
          ON sessions (source_type, last_activity_at);
      `);
    },
  },
  {
    // Open `source_type` from a closed CHECK to any string (connectors register their own;
    // shape is validated at the app layer). Existing DBs created under the original baseline
    // still carry the closed CHECK, so rebuild the table in place when it is detected. No-op on
    // fresh DBs (their baseline already omits the CHECK). Nothing FK-references `sessions`.
    id: 2,
    name: "session_source_type_open",
    up: (db) => {
      const row = db
        .prepare("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'sessions'")
        .get() as { sql?: string } | undefined;
      if (row?.sql === undefined || !/source_type\s+IN\s*\(/i.test(row.sql)) {
        return;
      }
      db.exec(`
        CREATE TABLE sessions_rebuilt (
          session_id TEXT PRIMARY KEY,
          source_type TEXT NOT NULL,
          source_external_id TEXT,
          source_url TEXT,
          label TEXT NOT NULL,
          audience_label TEXT NOT NULL,
          audience_entity_id TEXT,
          conversation_kind TEXT NOT NULL CHECK (
            conversation_kind IN ('dm', 'channel', 'thread', 'demo')
          ),
          created_at INTEGER NOT NULL,
          last_activity_at INTEGER NOT NULL,
          last_turn_id TEXT,
          message_count INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL CHECK (status IN ('active', 'idle', 'archived')),
          privacy_level TEXT NOT NULL DEFAULT 'payload_off' CHECK (
            privacy_level IN ('payload_off', 'payload_on')
          ),
          participation_policy TEXT NOT NULL DEFAULT 'active' CHECK (
            participation_policy IN ('active', 'paused', 'observing', 'muted')
          ),
          audience_role TEXT NOT NULL DEFAULT 'participant' CHECK (
            audience_role IN ('participant', 'operator')
          )
        );
        INSERT INTO sessions_rebuilt SELECT * FROM sessions;
        DROP TABLE sessions;
        ALTER TABLE sessions_rebuilt RENAME TO sessions;
        CREATE INDEX idx_sessions_last_activity
          ON sessions (last_activity_at);
        CREATE INDEX idx_sessions_source_type_last_activity
          ON sessions (source_type, last_activity_at);
      `);
    },
  },
] as const satisfies readonly Migration[];
