import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { ProvenanceError, StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import type { CommitmentRepository } from "../commitments/index.js";
import type { EntityId } from "../../util/ids.js";
import { parseStoredProvenance, provenanceSchema, toStoredProvenance, type Provenance } from "../common/provenance.js";

import {
  socialEventSchema,
  socialEventKindSchema,
  socialProfileSchema,
  socialSentimentPointSchema,
  type SocialEvent,
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

function mapEventRow(row: Record<string, unknown>): SocialEvent {
  return socialEventSchema.parse({
    id: Number(row.id),
    entity_id: row.entity_id,
    ts: Number(row.ts),
    kind: row.kind,
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_process: row.provenance_process,
    }),
    trust_delta: Number(row.trust_delta),
    attachment_delta: Number(row.attachment_delta),
    interaction_delta: Number(row.interaction_delta),
    valence: row.valence === null || row.valence === undefined ? null : Number(row.valence),
  });
}

function requireProvenance(provenance: Provenance | undefined, label: string): Provenance {
  if (provenance === undefined) {
    throw new ProvenanceError(`${label} requires provenance`, {
      code: "PROVENANCE_REQUIRED",
    });
  }

  return provenanceSchema.parse(provenance);
}

export type SocialRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

export type SocialInteractionRecord = {
  interaction_id: number;
  profile: SocialProfile;
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
      provenance: Provenance;
      valence?: number;
      now?: number;
    },
  ): SocialProfile {
    return this.recordInteractionWithId(entityId, input).profile;
  }

  recordInteractionWithId(
    entityId: EntityId,
    input: {
      provenance: Provenance;
      valence?: number;
      now?: number;
    },
  ): SocialInteractionRecord {
    const existing = this.upsertProfile(entityId);
    const nowMs = input.now ?? this.clock.now();
    const provenance = requireProvenance(input.provenance, "Social interaction");
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
    const storedProvenance = toStoredProvenance(provenance);

    const insertResult = this.db
      .prepare(
        `
          INSERT INTO social_events (
            entity_id, ts, kind, provenance_kind, provenance_episode_ids, provenance_process,
            trust_delta, attachment_delta, interaction_delta, valence
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        entityId,
        nowMs,
        socialEventKindSchema.parse("interaction"),
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        0,
        0,
        1,
        input.valence === undefined ? null : clamp(input.valence, -1, 1),
      );

    return {
      interaction_id: Number(insertResult.lastInsertRowid),
      profile: this.writeProfile({
        ...existing,
        last_interaction_at: nowMs,
        interaction_count: existing.interaction_count + 1,
        sentiment_history: sentimentHistory,
        updated_at: nowMs,
      }),
    };
  }

  attachSentiment(
    interactionId: number,
    input: {
      valence: number;
      now?: number;
    },
  ): SocialProfile {
    const row = this.db
      .prepare(
        `
          SELECT entity_id, ts
          FROM social_events
          WHERE id = ? AND kind = 'interaction'
        `,
      )
      .get(interactionId) as Record<string, unknown> | undefined;

    if (row === undefined) {
      throw new StorageError(`Missing interaction event ${interactionId}`, {
        code: "SOCIAL_ROW_INVALID",
      });
    }

    const entityId = String(row.entity_id) as EntityId;
    const interactionTs = Number(row.ts);
    const existing = this.upsertProfile(entityId);
    const nowMs = input.now ?? this.clock.now();
    const valence = clamp(input.valence, -1, 1);
    const sentimentHistory = [
      ...existing.sentiment_history,
      socialSentimentPointSchema.parse({
        ts: interactionTs,
        valence,
      }),
    ].slice(-50);

    this.db
      .prepare(
        `
          UPDATE social_events
          SET valence = ?
          WHERE id = ?
        `,
      )
      .run(valence, interactionId);

    return this.writeProfile({
      ...existing,
      sentiment_history: sentimentHistory,
      updated_at: nowMs,
    });
  }

  adjustTrust(entityId: EntityId, delta: number, provenance: Provenance): SocialProfile {
    const existing = this.upsertProfile(entityId);
    const parsedProvenance = requireProvenance(provenance, "Social trust adjustment");
    const storedProvenance = toStoredProvenance(parsedProvenance);
    const nowMs = this.clock.now();

    this.db
      .prepare(
        `
          INSERT INTO social_events (
            entity_id, ts, kind, provenance_kind, provenance_episode_ids, provenance_process,
            trust_delta, attachment_delta, interaction_delta, valence
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        entityId,
        nowMs,
        socialEventKindSchema.parse("trust_adjustment"),
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        delta,
        0,
        0,
        null,
      );

    return this.writeProfile({
      ...existing,
      trust: clamp(existing.trust + delta, 0, 1),
      updated_at: nowMs,
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

  listEvents(entityId?: EntityId): SocialEvent[] {
    const rows =
      entityId === undefined
        ? (this.db
            .prepare(
              `
                SELECT *
                FROM social_events
                ORDER BY ts DESC, id DESC
              `,
            )
            .all() as Record<string, unknown>[])
        : (this.db
            .prepare(
              `
                SELECT *
                FROM social_events
                WHERE entity_id = ?
                ORDER BY ts DESC, id DESC
              `,
            )
            .all(entityId) as Record<string, unknown>[]);

    return rows.map((row) => mapEventRow(row));
  }
}
