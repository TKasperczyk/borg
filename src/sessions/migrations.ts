import type { Migration } from "../storage/sqlite/index.js";

export const sessionMigrations = [
  {
    id: 1,
    name: "sessions_initial_schema",
    up: (db) => {
      db.exec(`
        CREATE TABLE sessions (
          session_id TEXT PRIMARY KEY,
          source_type TEXT NOT NULL CHECK (
            source_type IN ('demo', 'slack', 'discord', 'imessage', 'autonomy')
          ),
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
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_last_activity
          ON sessions (last_activity_at);

        CREATE INDEX IF NOT EXISTS idx_sessions_source_type_last_activity
          ON sessions (source_type, last_activity_at);
      `);
    },
  },
] as const satisfies readonly Migration[];
