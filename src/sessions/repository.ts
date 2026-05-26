import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import type { SessionId } from "../util/ids.js";

import {
  sessionEnsureInputSchema,
  sessionIdSchema,
  sessionListOptionsSchema,
  sessionParticipationPolicySchema,
  sessionRecordSchema,
  sessionTouchUpdateSchema,
  type SessionEnsureInput,
  type SessionListOptions,
  type SessionParticipationPolicy,
  type SessionRecord,
  type SessionTouchUpdate,
} from "./types.js";

const DEFAULT_SESSION_LIST_LIMIT = 100;
const MAX_SESSION_LIST_LIMIT = 1000;

type SessionRow = {
  session_id: string;
  source_type: string;
  source_external_id: string | null;
  source_url: string | null;
  label: string;
  audience_label: string;
  audience_entity_id: string | null;
  conversation_kind: string;
  created_at: number;
  last_activity_at: number;
  last_turn_id: string | null;
  message_count: number;
  status: string;
  privacy_level: string;
  participation_policy: string;
};

export type SessionsRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function mapSessionRow(row: SessionRow): SessionRecord {
  const parsed = sessionRecordSchema.safeParse(row);

  if (!parsed.success) {
    throw new StorageError("Session row failed validation", {
      cause: parsed.error,
      code: "SESSION_ROW_INVALID",
    });
  }

  return parsed.data;
}

function boundedLimit(limit: number | undefined): number {
  return Math.min(Math.max(1, limit ?? DEFAULT_SESSION_LIST_LIMIT), MAX_SESSION_LIST_LIMIT);
}

export class SessionsRepository {
  private readonly clock: Clock;

  constructor(private readonly options: SessionsRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  ensure(input: SessionEnsureInput): SessionRecord {
    const parsed = sessionEnsureInputSchema.parse(input);
    const nowMs = this.clock.now();
    const createdAt = parsed.created_at ?? nowMs;
    const lastActivityAt = parsed.last_activity_at ?? createdAt;

    this.db
      .prepare(
        `
          INSERT INTO sessions (
            session_id, source_type, source_external_id, source_url, label, audience_label,
            audience_entity_id, conversation_kind, created_at, last_activity_at, last_turn_id,
            message_count, status, privacy_level
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
          ON CONFLICT(session_id) DO UPDATE SET
            source_type = excluded.source_type,
            source_external_id = excluded.source_external_id,
            source_url = excluded.source_url,
            label = excluded.label,
            audience_label = excluded.audience_label,
            audience_entity_id = excluded.audience_entity_id,
            conversation_kind = excluded.conversation_kind,
            last_activity_at = MAX(sessions.last_activity_at, excluded.last_activity_at),
            last_turn_id = COALESCE(excluded.last_turn_id, sessions.last_turn_id),
            status = excluded.status,
            privacy_level = excluded.privacy_level
        `,
      )
      .run(
        parsed.session_id,
        parsed.source_type,
        parsed.source_external_id ?? null,
        parsed.source_url ?? null,
        parsed.label,
        parsed.audience_label,
        parsed.audience_entity_id ?? null,
        parsed.conversation_kind,
        createdAt,
        lastActivityAt,
        parsed.last_turn_id ?? null,
        parsed.status ?? "active",
        parsed.privacy_level ?? "payload_off",
      );

    const record = this.get(parsed.session_id);
    if (record === null) {
      throw new StorageError(`Session ${parsed.session_id} was not stored`, {
        code: "SESSION_ENSURE_FAILED",
      });
    }

    return record;
  }

  touch(sessionId: SessionId, update?: SessionTouchUpdate): SessionRecord | null {
    const parsedSessionId = sessionIdSchema.parse(sessionId);
    const parsedUpdate = update === undefined ? undefined : sessionTouchUpdateSchema.parse(update);
    const at = parsedUpdate?.at ?? this.clock.now();
    const delta = parsedUpdate?.messageCountDelta ?? 1;
    const hasLastTurnId = parsedUpdate !== undefined && "lastTurnId" in parsedUpdate;

    this.db
      .prepare(
        `
          UPDATE sessions
          SET
            last_activity_at = ?,
            last_turn_id = CASE WHEN ? = 1 THEN ? ELSE last_turn_id END,
            message_count = message_count + ?
          WHERE session_id = ?
        `,
      )
      .run(at, hasLastTurnId ? 1 : 0, parsedUpdate?.lastTurnId ?? null, delta, parsedSessionId);

    return this.get(parsedSessionId);
  }

  setParticipationPolicy(
    sessionId: SessionId,
    policy: SessionParticipationPolicy,
    options?: { now?: number },
  ): SessionRecord | null {
    const parsedSessionId = sessionIdSchema.parse(sessionId);
    const parsedPolicy = sessionParticipationPolicySchema.parse(policy);
    void options;

    this.db
      .prepare(
        `
          UPDATE sessions
          SET participation_policy = ?
          WHERE session_id = ?
        `,
      )
      .run(parsedPolicy, parsedSessionId);

    return this.get(parsedSessionId);
  }

  get(sessionId: SessionId): SessionRecord | null {
    const parsedSessionId = sessionIdSchema.parse(sessionId);
    const row = this.db
      .prepare(
        `
          SELECT
            session_id, source_type, source_external_id, source_url, label, audience_label,
            audience_entity_id, conversation_kind, created_at, last_activity_at, last_turn_id,
            message_count, status, privacy_level, participation_policy
          FROM sessions
          WHERE session_id = ?
        `,
      )
      .get(parsedSessionId) as SessionRow | undefined;

    return row === undefined ? null : mapSessionRow(row);
  }

  list(options?: SessionListOptions): SessionRecord[] {
    const parsed = options === undefined ? undefined : sessionListOptionsSchema.parse(options);
    const filters: string[] = [];
    const values: unknown[] = [];

    if (parsed?.activeSince !== undefined) {
      filters.push("last_activity_at >= ?");
      values.push(parsed.activeSince);
    }

    if (parsed?.sourceType !== undefined) {
      filters.push("source_type = ?");
      values.push(parsed.sourceType);
    }

    values.push(boundedLimit(parsed?.limit));

    const where = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const rows = this.db
      .prepare(
        `
          SELECT
            session_id, source_type, source_external_id, source_url, label, audience_label,
            audience_entity_id, conversation_kind, created_at, last_activity_at, last_turn_id,
            message_count, status, privacy_level, participation_policy
          FROM sessions
          ${where}
          ORDER BY last_activity_at DESC, session_id ASC
          LIMIT ?
        `,
      )
      .all(...values) as SessionRow[];

    return rows.map(mapSessionRow);
  }
}
