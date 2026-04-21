import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import type { CommitmentRepository } from "../commitments/index.js";
import type { EntityId } from "../../util/ids.js";

import {
  socialProfileSchema,
  socialSentimentPointSchema,
  type SocialProfile,
  type SocialSentimentPoint,
} from "./types.js";

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function parseSentimentHistory(value: string): SocialSentimentPoint[] {
  let parsed: unknown;

  try {
    parsed = JSON.parse(value) as unknown;
  } catch (error) {
    throw new StorageError("Failed to parse social sentiment history", {
      cause: error,
      code: "SOCIAL_ROW_INVALID",
    });
  }

  const result = socialSentimentPointSchema.array().safeParse(parsed);

  if (!result.success) {
    throw new StorageError("Invalid social sentiment history", {
      cause: result.error,
      code: "SOCIAL_ROW_INVALID",
    });
  }

  return result.data;
}

function mapProfileRow(row: Record<string, unknown>): SocialProfile {
  const parsed = socialProfileSchema.safeParse({
    entity_id: row.entity_id,
    trust: Number(row.trust),
    attachment: Number(row.attachment),
    communication_style:
      row.communication_style === null || row.communication_style === undefined
        ? null
        : String(row.communication_style),
    shared_history_summary:
      row.shared_history_summary === null || row.shared_history_summary === undefined
        ? null
        : String(row.shared_history_summary),
    last_interaction_at:
      row.last_interaction_at === null || row.last_interaction_at === undefined
        ? null
        : Number(row.last_interaction_at),
    interaction_count: Number(row.interaction_count),
    commitment_count: Number(row.commitment_count),
    sentiment_history: parseSentimentHistory(String(row.sentiment_history ?? "[]")),
    notes: row.notes === null || row.notes === undefined ? null : String(row.notes),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  });

  if (!parsed.success) {
    throw new StorageError("Social profile row failed validation", {
      cause: parsed.error,
      code: "SOCIAL_ROW_INVALID",
    });
  }

  return parsed.data;
}

export type SocialRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

export class SocialRepository {
  private readonly clock: Clock;

  constructor(private readonly options: SocialRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private writeProfile(profile: SocialProfile): SocialProfile {
    const parsed = socialProfileSchema.parse(profile);

    this.db
      .prepare(
        `
          INSERT INTO social_profiles (
            entity_id, trust, attachment, communication_style, shared_history_summary,
            last_interaction_at, interaction_count, commitment_count, sentiment_history, notes,
            created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (entity_id) DO UPDATE SET
            trust = excluded.trust,
            attachment = excluded.attachment,
            communication_style = excluded.communication_style,
            shared_history_summary = excluded.shared_history_summary,
            last_interaction_at = excluded.last_interaction_at,
            interaction_count = excluded.interaction_count,
            commitment_count = excluded.commitment_count,
            sentiment_history = excluded.sentiment_history,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        `,
      )
      .run(
        parsed.entity_id,
        parsed.trust,
        parsed.attachment,
        parsed.communication_style,
        parsed.shared_history_summary,
        parsed.last_interaction_at,
        parsed.interaction_count,
        parsed.commitment_count,
        serializeJsonValue(parsed.sentiment_history),
        parsed.notes,
        parsed.created_at,
        parsed.updated_at,
      );

    return parsed;
  }

  upsertProfile(entityId: EntityId): SocialProfile {
    const existing = this.getProfile(entityId);

    if (existing !== null) {
      return existing;
    }

    const nowMs = this.clock.now();
    return this.writeProfile({
      entity_id: entityId,
      trust: 0.5,
      attachment: 0,
      communication_style: null,
      shared_history_summary: null,
      last_interaction_at: null,
      interaction_count: 0,
      commitment_count: 0,
      sentiment_history: [],
      notes: null,
      created_at: nowMs,
      updated_at: nowMs,
    });
  }

  getProfile(entityId: EntityId): SocialProfile | null {
    const row = this.db
      .prepare("SELECT * FROM social_profiles WHERE entity_id = ?")
      .get(entityId) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapProfileRow(row);
  }

  list(limit = 100): SocialProfile[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM social_profiles
          ORDER BY updated_at DESC, created_at DESC
          LIMIT ?
        `,
      )
      .all(limit) as Record<string, unknown>[];

    return rows.map((row) => mapProfileRow(row));
  }

  recordInteraction(
    entityId: EntityId,
    input: {
      episode_id?: string;
      valence?: number;
      now?: number;
    },
  ): SocialProfile {
    const existing = this.upsertProfile(entityId);
    const nowMs = input.now ?? this.clock.now();
    const sentimentHistory =
      input.valence === undefined
        ? existing.sentiment_history
        : [
            ...existing.sentiment_history,
            socialSentimentPointSchema.parse({
              ts: nowMs,
              valence: clamp(input.valence, -1, 1),
            }),
          ].slice(-50);

    return this.writeProfile({
      ...existing,
      last_interaction_at: nowMs,
      interaction_count: existing.interaction_count + 1,
      sentiment_history: sentimentHistory,
      updated_at: nowMs,
    });
  }

  adjustTrust(entityId: EntityId, delta: number): SocialProfile {
    const existing = this.upsertProfile(entityId);

    return this.writeProfile({
      ...existing,
      trust: clamp(existing.trust + delta, 0, 1),
      updated_at: this.clock.now(),
    });
  }

  recomputeCommitmentCount(
    entityId: EntityId,
    commitmentRepository: CommitmentRepository,
  ): SocialProfile {
    const existing = this.upsertProfile(entityId);
    const count = commitmentRepository
      .list({ activeOnly: true })
      .filter(
        (commitment) =>
          commitment.made_to_entity === entityId ||
          commitment.restricted_audience === entityId ||
          commitment.about_entity === entityId,
      ).length;

    return this.writeProfile({
      ...existing,
      commitment_count: count,
      updated_at: this.clock.now(),
    });
  }

  restoreProfile(profile: SocialProfile): SocialProfile {
    return this.writeProfile(profile);
  }
}
