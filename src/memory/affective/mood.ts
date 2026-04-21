import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { parseEpisodeId, type EpisodeId, type SessionId } from "../../util/ids.js";

import {
  moodHistoryEntrySchema,
  moodStateSchema,
  type MoodHistoryEntry,
  type MoodState,
} from "./types.js";

const HOUR_MS = 60 * 60 * 1_000;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function parseStringArray(value: string, label: string): string[] {
  try {
    const parsed = JSON.parse(value) as unknown;

    if (!Array.isArray(parsed)) {
      throw new TypeError(`${label} must be an array`);
    }

    return parsed.filter((item): item is string => typeof item === "string");
  } catch (error) {
    throw new StorageError(`Failed to parse ${label}`, {
      cause: error,
      code: "MOOD_ROW_INVALID",
    });
  }
}

function decayFactor(nowMs: number, updatedAt: number, halfLifeHours: number): number {
  const elapsed = Math.max(0, nowMs - updatedAt);
  return Math.pow(0.5, elapsed / (halfLifeHours * HOUR_MS));
}

function mapMoodStateRow(row: Record<string, unknown>, nowMs: number): MoodState {
  const base = moodStateSchema.parse({
    session_id: row.session_id,
    valence: Number(row.valence),
    arousal: Number(row.arousal),
    updated_at: Number(row.updated_at),
    half_life_hours: Number(row.half_life_hours),
    recent_triggers: parseStringArray(String(row.recent_triggers ?? "[]"), "recent_triggers"),
  });
  const factor = decayFactor(nowMs, base.updated_at, base.half_life_hours);

  return {
    ...base,
    valence: clamp(base.valence * factor, -1, 1),
    arousal: clamp(base.arousal * factor, 0, 1),
  };
}

function mapMoodHistoryRow(row: Record<string, unknown>): MoodHistoryEntry {
  const parsed = moodHistoryEntrySchema.safeParse({
    id: Number(row.id),
    session_id: row.session_id,
    ts: Number(row.ts),
    valence: Number(row.valence),
    arousal: Number(row.arousal),
    trigger_episode_id:
      row.trigger_episode_id === null || row.trigger_episode_id === undefined
        ? null
        : parseEpisodeId(String(row.trigger_episode_id)),
    trigger_reason:
      row.trigger_reason === null || row.trigger_reason === undefined
        ? null
        : String(row.trigger_reason),
  });

  if (!parsed.success) {
    throw new StorageError("Mood history row failed validation", {
      cause: parsed.error,
      code: "MOOD_ROW_INVALID",
    });
  }

  return parsed.data;
}

export type MoodRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  defaultHalfLifeHours?: number;
  incomingWeight?: number;
};

export class MoodRepository {
  private readonly clock: Clock;
  private readonly defaultHalfLifeHours: number;
  private readonly incomingWeight: number;

  constructor(private readonly options: MoodRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.defaultHalfLifeHours = options.defaultHalfLifeHours ?? 24;
    this.incomingWeight = clamp(options.incomingWeight ?? 0.3, 0, 1);
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  current(sessionId: SessionId): MoodState {
    const row = this.db.prepare("SELECT * FROM mood_state WHERE session_id = ?").get(sessionId) as
      | Record<string, unknown>
      | undefined;

    if (row === undefined) {
      return moodStateSchema.parse({
        session_id: sessionId,
        valence: 0,
        arousal: 0,
        updated_at: this.clock.now(),
        half_life_hours: this.defaultHalfLifeHours,
        recent_triggers: [],
      });
    }

    return mapMoodStateRow(row, this.clock.now());
  }

  listStates(): MoodState[] {
    const nowMs = this.clock.now();
    return this.listStoredStates().map((state) => ({
      ...state,
      valence: clamp(
        state.valence * decayFactor(nowMs, state.updated_at, state.half_life_hours),
        -1,
        1,
      ),
      arousal: clamp(
        state.arousal * decayFactor(nowMs, state.updated_at, state.half_life_hours),
        0,
        1,
      ),
    }));
  }

  listStoredStates(): MoodState[] {
    const rows = this.db
      .prepare("SELECT * FROM mood_state ORDER BY updated_at DESC, session_id ASC")
      .all() as Record<string, unknown>[];

    return rows.map((row) =>
      moodStateSchema.parse({
        session_id: row.session_id,
        valence: Number(row.valence),
        arousal: Number(row.arousal),
        updated_at: Number(row.updated_at),
        half_life_hours: Number(row.half_life_hours),
        recent_triggers: parseStringArray(String(row.recent_triggers ?? "[]"), "recent_triggers"),
      }),
    );
  }

  update(
    sessionId: SessionId,
    input: {
      valence: number;
      arousal: number;
      trigger_episode_id?: EpisodeId;
      reason?: string;
    },
  ): MoodState {
    const nowMs = this.clock.now();
    const current = this.current(sessionId);
    const nextValence = clamp(
      current.valence * (1 - this.incomingWeight) +
        clamp(input.valence, -1, 1) * this.incomingWeight,
      -1,
      1,
    );
    const nextArousal = clamp(
      current.arousal * (1 - this.incomingWeight) +
        clamp(input.arousal, 0, 1) * this.incomingWeight,
      0,
      1,
    );
    const nextTriggers = [...current.recent_triggers, input.reason ?? "unspecified"].slice(-10);
    const next = moodStateSchema.parse({
      session_id: sessionId,
      valence: nextValence,
      arousal: nextArousal,
      updated_at: nowMs,
      half_life_hours: current.half_life_hours || this.defaultHalfLifeHours,
      recent_triggers: nextTriggers,
    });

    this.db.transaction(() => {
      this.db
        .prepare(
          `
            INSERT INTO mood_history (
              session_id, ts, valence, arousal, trigger_episode_id, trigger_reason
            ) VALUES (?, ?, ?, ?, ?, ?)
          `,
        )
        .run(
          sessionId,
          nowMs,
          clamp(input.valence, -1, 1),
          clamp(input.arousal, 0, 1),
          input.trigger_episode_id ?? null,
          input.reason ?? null,
        );
      this.db
        .prepare(
          `
            INSERT INTO mood_state (
              session_id, valence, arousal, updated_at, half_life_hours, recent_triggers
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (session_id) DO UPDATE SET
              valence = excluded.valence,
              arousal = excluded.arousal,
              updated_at = excluded.updated_at,
              half_life_hours = excluded.half_life_hours,
              recent_triggers = excluded.recent_triggers
          `,
        )
        .run(
          next.session_id,
          next.valence,
          next.arousal,
          next.updated_at,
          next.half_life_hours,
          serializeJsonValue(next.recent_triggers),
        );
    })();

    return next;
  }

  history(
    sessionId: SessionId,
    options: {
      fromTs?: number;
      toTs?: number;
      limit?: number;
    } = {},
  ): MoodHistoryEntry[] {
    const filters = ["session_id = ?"];
    const values: unknown[] = [sessionId];

    if (options.fromTs !== undefined) {
      filters.push("ts >= ?");
      values.push(options.fromTs);
    }

    if (options.toTs !== undefined) {
      filters.push("ts <= ?");
      values.push(options.toTs);
    }

    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM mood_history
          WHERE ${filters.join(" AND ")}
          ORDER BY ts DESC, id DESC
          LIMIT ?
        `,
      )
      .all(...values, options.limit ?? 50) as Record<string, unknown>[];

    return rows.map((row) => mapMoodHistoryRow(row));
  }

  historyBefore(thresholdTs: number): MoodHistoryEntry[] {
    const rows = this.db
      .prepare("SELECT * FROM mood_history WHERE ts < ? ORDER BY ts ASC, id ASC")
      .all(thresholdTs) as Record<string, unknown>[];

    return rows.map((row) => mapMoodHistoryRow(row));
  }

  trimHistory(retentionDays: number, nowMs = this.clock.now()): MoodHistoryEntry[] {
    const threshold = nowMs - retentionDays * 24 * HOUR_MS;
    const removed = this.db
      .prepare("SELECT * FROM mood_history WHERE ts < ? ORDER BY ts ASC, id ASC")
      .all(threshold) as Record<string, unknown>[];
    this.db.prepare("DELETE FROM mood_history WHERE ts < ?").run(threshold);
    return removed.map((row) => mapMoodHistoryRow(row));
  }

  restoreHistory(entries: readonly MoodHistoryEntry[]): void {
    const insert = this.db.prepare(
      `
        INSERT OR IGNORE INTO mood_history (
          id, session_id, ts, valence, arousal, trigger_episode_id, trigger_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
      `,
    );

    for (const entry of entries) {
      insert.run(
        entry.id,
        entry.session_id,
        entry.ts,
        entry.valence,
        entry.arousal,
        entry.trigger_episode_id,
        entry.trigger_reason,
      );
    }
  }

  saveState(state: MoodState): MoodState {
    const parsed = moodStateSchema.parse(state);

    this.db
      .prepare(
        `
          INSERT INTO mood_state (
            session_id, valence, arousal, updated_at, half_life_hours, recent_triggers
          ) VALUES (?, ?, ?, ?, ?, ?)
          ON CONFLICT (session_id) DO UPDATE SET
            valence = excluded.valence,
            arousal = excluded.arousal,
            updated_at = excluded.updated_at,
            half_life_hours = excluded.half_life_hours,
            recent_triggers = excluded.recent_triggers
        `,
      )
      .run(
        parsed.session_id,
        parsed.valence,
        parsed.arousal,
        parsed.updated_at,
        parsed.half_life_hours,
        serializeJsonValue(parsed.recent_triggers),
      );

    return parsed;
  }
}
