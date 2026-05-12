import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { ProvenanceError, StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import type { CommitmentRepository } from "../commitments/index.js";
import type { EntityId } from "../../util/ids.js";
import {
  assertIdentityCasUpdated,
  expectedRecordVersion,
  nextRecordVersion,
} from "../common/cas.js";
import {
  parseStoredProvenance,
  provenanceSchema,
  toStoredProvenance,
  type Provenance,
} from "../common/provenance.js";

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
    record_version: Number(row.record_version ?? 1),
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

  private requireProfile(entityId: EntityId): SocialProfile {
    const profile = this.getProfile(entityId);

    if (profile === null) {
      throw new StorageError(`Missing social profile for ${entityId}`, {
        code: "SOCIAL_ROW_INVALID",
      });
    }

    return profile;
  }

  private assertAtomicProfileUpdated(result: { changes: number }, entityId: EntityId): void {
    if (result.changes > 0) {
      return;
    }

    throw new StorageError(`Missing social profile for ${entityId}`, {
      code: "SOCIAL_ROW_INVALID",
    });
  }

  private insertProfileIfMissing(profile: SocialProfile): SocialProfile {
    const parsed = socialProfileSchema.parse(profile);

    this.db
      .prepare(
        `
          INSERT OR IGNORE INTO social_profiles (
            entity_id, record_version, trust, attachment, communication_style, shared_history_summary,
            last_interaction_at, interaction_count, commitment_count, sentiment_history, notes,
            created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        parsed.entity_id,
        parsed.record_version ?? 1,
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

    return this.requireProfile(parsed.entity_id);
  }

  private writeProfile(profile: SocialProfile): SocialProfile {
    const parsed = socialProfileSchema.parse(profile);
    const expectedVersion = expectedRecordVersion(parsed);
    const result = this.db
      .prepare(
        `
          UPDATE social_profiles
          SET trust = ?,
              attachment = ?,
              communication_style = ?,
              shared_history_summary = ?,
              last_interaction_at = ?,
              interaction_count = ?,
              commitment_count = ?,
              sentiment_history = ?,
              notes = ?,
              created_at = ?,
              updated_at = ?,
              record_version = record_version + 1
          WHERE entity_id = ? AND record_version = ?
        `,
      )
      .run(
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
        parsed.entity_id,
        expectedVersion,
      );

    assertIdentityCasUpdated({
      result,
      recordType: "social_profile",
      recordId: parsed.entity_id,
      expectedVersion,
    });

    return {
      ...parsed,
      record_version: nextRecordVersion(expectedVersion),
    };
  }

  private sentimentHistoryFromEvents(entityId: EntityId): SocialSentimentPoint[] {
    const rows = this.db
      .prepare(
        `
          SELECT ts, valence
          FROM (
            SELECT id, ts, valence
            FROM social_events
            WHERE entity_id = ? AND kind = 'interaction' AND valence IS NOT NULL
            ORDER BY ts DESC, id DESC
            LIMIT 50
          )
          ORDER BY ts ASC, id ASC
        `,
      )
      .all(entityId) as Record<string, unknown>[];

    return rows.map((row) =>
      socialSentimentPointSchema.parse({
        ts: Number(row.ts),
        valence: Number(row.valence),
      }),
    );
  }

  private refreshSentimentHistory(
    entityId: EntityId,
    timestamp: number,
    options: { bumpRecordVersion: boolean },
  ): void {
    const sentimentHistory = this.sentimentHistoryFromEvents(entityId);
    const result = this.db
      .prepare(
        options.bumpRecordVersion
          ? `
              UPDATE social_profiles
              SET sentiment_history = ?,
                  updated_at = ?,
                  record_version = record_version + 1
              WHERE entity_id = ?
            `
          : `
              UPDATE social_profiles
              SET sentiment_history = ?,
                  updated_at = ?
              WHERE entity_id = ?
            `,
      )
      .run(serializeJsonValue(sentimentHistory), timestamp, entityId);

    this.assertAtomicProfileUpdated(result, entityId);
  }

  private applyInteractionAggregate(entityId: EntityId, timestamp: number): void {
    const result = this.db
      .prepare(
        `
          UPDATE social_profiles
          SET last_interaction_at = CASE
                WHEN last_interaction_at IS NULL OR last_interaction_at < ? THEN ?
                ELSE last_interaction_at
              END,
              interaction_count = interaction_count + 1,
              updated_at = CASE
                WHEN updated_at < ? THEN ?
                ELSE updated_at
              END,
              record_version = record_version + 1
          WHERE entity_id = ?
        `,
      )
      .run(timestamp, timestamp, timestamp, timestamp, entityId);

    this.assertAtomicProfileUpdated(result, entityId);
  }

  upsertProfile(entityId: EntityId): SocialProfile {
    const existing = this.getProfile(entityId);

    if (existing !== null) {
      return existing;
    }

    const nowMs = this.clock.now();
    return this.insertProfileIfMissing({
      entity_id: entityId,
      record_version: 1,
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
    this.upsertProfile(entityId);
    const nowMs = input.now ?? this.clock.now();
    const provenance = requireProvenance(input.provenance, "Social interaction");
    const valence = input.valence === undefined ? null : clamp(input.valence, -1, 1);
    const storedProvenance = toStoredProvenance(provenance);

    const writeInteraction = this.db.transaction(() => {
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
          valence,
        );

      this.applyInteractionAggregate(entityId, nowMs);

      if (valence !== null) {
        this.refreshSentimentHistory(entityId, nowMs, { bumpRecordVersion: false });
      }

      return Number(insertResult.lastInsertRowid);
    });
    const interactionId = writeInteraction();

    return {
      interaction_id: interactionId,
      profile: this.requireProfile(entityId),
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
          SELECT entity_id
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
    this.upsertProfile(entityId);
    const nowMs = input.now ?? this.clock.now();
    const valence = clamp(input.valence, -1, 1);

    const attach = this.db.transaction(() => {
      this.db
        .prepare(
          `
            UPDATE social_events
            SET valence = ?
            WHERE id = ?
          `,
        )
        .run(valence, interactionId);

      this.refreshSentimentHistory(entityId, nowMs, { bumpRecordVersion: true });
    });
    attach();

    return this.requireProfile(entityId);
  }

  adjustTrust(entityId: EntityId, delta: number, provenance: Provenance): SocialProfile {
    this.upsertProfile(entityId);
    const parsedProvenance = requireProvenance(provenance, "Social trust adjustment");
    const storedProvenance = toStoredProvenance(parsedProvenance);
    const nowMs = this.clock.now();

    const adjust = this.db.transaction(() => {
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

      const result = this.db
        .prepare(
          `
            UPDATE social_profiles
            SET trust = CASE
                  WHEN trust + ? < 0 THEN 0
                  WHEN trust + ? > 1 THEN 1
                  ELSE trust + ?
                END,
                updated_at = ?,
                record_version = record_version + 1
            WHERE entity_id = ?
          `,
        )
        .run(delta, delta, delta, nowMs, entityId);

      this.assertAtomicProfileUpdated(result, entityId);
    });
    adjust();

    return this.requireProfile(entityId);
  }

  recomputeCommitmentCount(
    entityId: EntityId,
    commitmentRepository: CommitmentRepository,
  ): SocialProfile {
    this.upsertProfile(entityId);
    const count = commitmentRepository
      .list({ activeOnly: true })
      .filter(
        (commitment) =>
          commitment.made_to_entity === entityId ||
          commitment.restricted_audience === entityId ||
          commitment.about_entity === entityId,
      ).length;

    const result = this.db
      .prepare(
        `
          UPDATE social_profiles
          SET commitment_count = ?,
              updated_at = ?,
              record_version = record_version + 1
          WHERE entity_id = ?
        `,
      )
      .run(count, this.clock.now(), entityId);

    this.assertAtomicProfileUpdated(result, entityId);

    return this.requireProfile(entityId);
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
