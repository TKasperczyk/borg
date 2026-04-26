import { z } from "zod";

import { type SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { createValueId, type EpisodeId, type ValueId } from "../../util/ids.js";
import {
  isEpisodeProvenance,
  parseStoredProvenance,
  toStoredProvenance,
  type Provenance,
} from "../common/provenance.js";
import { type IdentityEventRepository } from "../identity/repository.js";

import {
  clamp,
  computeConfidence,
  EVIDENCE_EPISODE_LIMIT,
  summarizeEvidence,
} from "./shared/evidence.js";
import { recordIdentityEvent } from "./shared/identity-events.js";
import {
  getPromotionMetadataFromEvents,
  resolveValueInitialState,
  VALUE_PROMOTION_THRESHOLD,
  type PromotionMetadata,
} from "./shared/promotion.js";
import { requireProvenance } from "./shared/provenance.js";
import { mapValueRow } from "./shared/sql-mapping.js";
import { valuePatchSchema, valueSchema, type ValueRecord } from "./types.js";

export type ValuesRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  identityEventRepository?: IdentityEventRepository;
};

export type ValueReinforcementEvent = {
  id: number;
  value_id: ValueId;
  ts: number;
  provenance: Provenance;
};

export type ValueContradictionEvent = {
  id: number;
  value_id: ValueId;
  ts: number;
  provenance: Provenance;
  weight: number;
};

export class ValuesRepository {
  private readonly clock: Clock;

  constructor(private readonly options: ValuesRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private get identityEventRepository(): IdentityEventRepository | undefined {
    return this.options.identityEventRepository;
  }

  private getById(valueId: ValueId): Record<string, unknown> | undefined {
    return this.db
      .prepare(
        `
          SELECT
            id, label, description, priority, created_at, last_affirmed, state, established_at,
            confidence, last_tested_at, last_contradicted_at, support_count, contradiction_count,
            evidence_episode_ids, provenance_kind, provenance_episode_ids, provenance_process
          FROM "values"
          WHERE id = ?
        `,
      )
      .get(valueId) as Record<string, unknown> | undefined;
  }

  private insertReinforcementEvent(
    valueId: ValueId,
    provenance: Provenance,
    timestamp: number,
  ): ValueReinforcementEvent {
    const storedProvenance = toStoredProvenance(provenance);
    const result = this.db
      .prepare(
        `
          INSERT INTO value_reinforcement_events (
            value_id, ts, provenance_kind, provenance_episode_ids, provenance_process
          ) VALUES (?, ?, ?, ?, ?)
        `,
      )
      .run(
        valueId,
        timestamp,
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
      );

    return {
      id: Number(result.lastInsertRowid),
      value_id: valueId,
      ts: timestamp,
      provenance,
    };
  }

  private insertContradictionEvent(
    valueId: ValueId,
    provenance: Provenance,
    weight: number,
    timestamp: number,
  ): ValueContradictionEvent {
    const storedProvenance = toStoredProvenance(provenance);
    const result = this.db
      .prepare(
        `
          INSERT INTO value_contradiction_events (
            value_id, ts, weight, provenance_kind, provenance_episode_ids, provenance_process
          ) VALUES (?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        valueId,
        timestamp,
        weight,
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
      );

    return {
      id: Number(result.lastInsertRowid),
      value_id: valueId,
      ts: timestamp,
      provenance,
      weight,
    };
  }

  private getPromotionMetadata(valueId: ValueId): PromotionMetadata {
    return getPromotionMetadataFromEvents(
      this.listReinforcementEvents(valueId),
      VALUE_PROMOTION_THRESHOLD,
    );
  }

  get(valueId: ValueId): ValueRecord | null {
    const row = this.getById(valueId);
    return row === undefined ? null : mapValueRow(row);
  }

  add(input: {
    id?: ValueId;
    label: string;
    description: string;
    priority: number;
    provenance: Provenance;
    createdAt?: number;
    lastAffirmed?: number | null;
  }): ValueRecord {
    const provenance = requireProvenance(input.provenance, "Value");
    const createdAt = input.createdAt ?? this.clock.now();
    const initialState = resolveValueInitialState(provenance, createdAt);
    const initialEvidenceEpisodeIds = isEpisodeProvenance(provenance)
      ? [...new Set(provenance.episode_ids)].slice(0, EVIDENCE_EPISODE_LIMIT)
      : [];
    const value = valueSchema.parse({
      id: input.id ?? createValueId(),
      label: input.label,
      description: input.description,
      priority: input.priority,
      created_at: createdAt,
      last_affirmed: input.lastAffirmed ?? null,
      state: initialState.state,
      established_at: initialState.established_at,
      confidence: computeConfidence(isEpisodeProvenance(provenance) ? 1 : 0, 0),
      last_tested_at: isEpisodeProvenance(provenance) ? createdAt : null,
      last_contradicted_at: null,
      support_count: isEpisodeProvenance(provenance) ? 1 : 0,
      contradiction_count: 0,
      evidence_episode_ids: initialEvidenceEpisodeIds,
      provenance,
    });
    const storedProvenance = toStoredProvenance(value.provenance);

    this.db
      .prepare(
        `
          INSERT INTO "values" (
            id, label, description, priority, created_at, last_affirmed, provenance_kind,
            provenance_episode_ids, provenance_process, state, established_at, confidence,
            last_tested_at, last_contradicted_at, support_count, contradiction_count, evidence_episode_ids
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        value.id,
        value.label,
        value.description,
        value.priority,
        value.created_at,
        value.last_affirmed,
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        value.state,
        value.established_at,
        value.confidence,
        value.last_tested_at,
        value.last_contradicted_at,
        value.support_count,
        value.contradiction_count,
        JSON.stringify(value.evidence_episode_ids),
      );

    if (isEpisodeProvenance(provenance)) {
      this.insertReinforcementEvent(value.id, provenance, createdAt);

      for (const episodeId of new Set(provenance.episode_ids)) {
        this.db
          .prepare(
            `
              INSERT OR IGNORE INTO value_sources (value_id, episode_id)
              VALUES (?, ?)
            `,
          )
          .run(value.id, episodeId);
      }
    }

    recordIdentityEvent(this.identityEventRepository, {
      record_type: "value",
      record_id: value.id,
      action: "create",
      old_value: null,
      new_value: value,
      provenance: value.provenance,
    });

    return value;
  }

  list(): ValueRecord[] {
    return (
      this.db
        .prepare(
          `
            SELECT
              id, label, description, priority, created_at, last_affirmed, state, established_at,
              confidence, last_tested_at, last_contradicted_at, support_count, contradiction_count,
              evidence_episode_ids, provenance_kind, provenance_episode_ids, provenance_process
            FROM "values"
            ORDER BY priority DESC, created_at ASC
          `,
        )
        .all() as Record<string, unknown>[]
    ).map((row) => mapValueRow(row));
  }

  affirm(valueId: ValueId, timestamp = this.clock.now()): void {
    const result = this.db
      .prepare('UPDATE "values" SET last_affirmed = ? WHERE id = ?')
      .run(timestamp, valueId);

    if (result.changes === 0) {
      throw new StorageError(`Unknown value id: ${valueId}`, {
        code: "VALUE_NOT_FOUND",
      });
    }
  }

  remove(valueId: ValueId): boolean {
    const result = this.db.prepare('DELETE FROM "values" WHERE id = ?').run(valueId);
    return result.changes > 0;
  }

  reinforce(valueId: ValueId, provenance: Provenance, timestamp = this.clock.now()): ValueRecord {
    const current = this.get(valueId);

    if (current === null) {
      throw new StorageError(`Unknown value id: ${valueId}`, {
        code: "VALUE_NOT_FOUND",
      });
    }

    const parsedProvenance = requireProvenance(provenance, "Value reinforcement");
    this.insertReinforcementEvent(valueId, parsedProvenance, timestamp);

    if (isEpisodeProvenance(parsedProvenance)) {
      for (const episodeId of new Set(parsedProvenance.episode_ids)) {
        this.db
          .prepare(
            `
              INSERT OR IGNORE INTO value_sources (value_id, episode_id)
              VALUES (?, ?)
            `,
          )
          .run(valueId, episodeId);
      }
    }

    const promotion = this.getPromotionMetadata(valueId);
    const evidence = summarizeEvidence(
      this.listReinforcementEvents(valueId),
      this.listContradictionEvents(valueId),
    );
    const isPromoted = current.state !== "established" && promotion.state === "established";
    const nextState = current.state === "established" ? current.state : promotion.state;
    const nextProvenance: Provenance =
      isPromoted && promotion.promotionProvenance !== null
        ? promotion.promotionProvenance
        : current.provenance.kind === "episodes" && isEpisodeProvenance(parsedProvenance)
          ? {
              kind: "episodes",
              episode_ids: [
                ...new Set([...current.provenance.episode_ids, ...parsedProvenance.episode_ids]),
              ],
            }
          : isEpisodeProvenance(parsedProvenance)
            ? parsedProvenance
            : current.provenance;
    const next = valueSchema.parse({
      ...current,
      provenance: nextProvenance,
      state: nextState,
      established_at:
        current.state === "established" ? current.established_at : promotion.established_at,
      confidence: computeConfidence(evidence.supportCount, evidence.contradictionCount),
      last_tested_at: evidence.lastTestedAt,
      last_contradicted_at: evidence.lastContradictedAt,
      support_count: evidence.supportCount,
      contradiction_count: evidence.contradictionCount,
      evidence_episode_ids: evidence.evidenceEpisodeIds,
    });
    const storedProvenance = toStoredProvenance(next.provenance);

    this.db
      .prepare(
        `
          UPDATE "values"
          SET provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?, state = ?, established_at = ?,
              confidence = ?, last_tested_at = ?, last_contradicted_at = ?, support_count = ?,
              contradiction_count = ?, evidence_episode_ids = ?
          WHERE id = ?
        `,
      )
      .run(
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        next.state,
        next.established_at,
        next.confidence,
        next.last_tested_at,
        next.last_contradicted_at,
        next.support_count,
        next.contradiction_count,
        JSON.stringify(next.evidence_episode_ids),
        valueId,
      );

    if (current.state !== "established" && next.state === "established") {
      recordIdentityEvent(this.identityEventRepository, {
        record_type: "value",
        record_id: valueId,
        action: "promote",
        old_value: current,
        new_value: next,
        provenance: parsedProvenance,
      });
    }

    return next;
  }

  recordContradiction(input: {
    valueId: ValueId;
    provenance: Provenance;
    weight?: number;
    timestamp?: number;
  }): ValueRecord {
    const current = this.get(input.valueId);

    if (current === null) {
      throw new StorageError(`Unknown value id: ${input.valueId}`, {
        code: "VALUE_NOT_FOUND",
      });
    }

    const timestamp = input.timestamp ?? this.clock.now();
    const provenance = requireProvenance(input.provenance, "Value contradiction");
    const weight = Number.isFinite(input.weight) && input.weight !== undefined ? input.weight : 1;
    const normalizedWeight = clamp(weight, 0, Number.POSITIVE_INFINITY);
    this.insertContradictionEvent(input.valueId, provenance, normalizedWeight, timestamp);

    const evidence = summarizeEvidence(
      this.listReinforcementEvents(input.valueId),
      this.listContradictionEvents(input.valueId),
    );
    const next = valueSchema.parse({
      ...current,
      confidence: computeConfidence(evidence.supportCount, evidence.contradictionCount),
      last_tested_at: evidence.lastTestedAt,
      last_contradicted_at: evidence.lastContradictedAt,
      support_count: evidence.supportCount,
      contradiction_count: evidence.contradictionCount,
      evidence_episode_ids: evidence.evidenceEpisodeIds,
    });

    this.db
      .prepare(
        `
          UPDATE "values"
          SET confidence = ?, last_tested_at = ?, last_contradicted_at = ?, support_count = ?,
              contradiction_count = ?, evidence_episode_ids = ?
          WHERE id = ?
        `,
      )
      .run(
        next.confidence,
        next.last_tested_at,
        next.last_contradicted_at,
        next.support_count,
        next.contradiction_count,
        JSON.stringify(next.evidence_episode_ids),
        input.valueId,
      );

    recordIdentityEvent(this.identityEventRepository, {
      record_type: "value",
      record_id: input.valueId,
      action: "contradict",
      old_value: current,
      new_value: next,
      provenance,
      ts: timestamp,
    });

    return next;
  }

  bindToEpisode(valueId: ValueId, episodeId: EpisodeId): void {
    void this.reinforce(valueId, {
      kind: "episodes",
      episode_ids: [episodeId],
    });
  }

  /**
   * @internal Prefer IdentityService.updateValue() so established records cannot
   * bypass review gating.
   */
  update(
    valueId: ValueId,
    patch: z.infer<typeof valuePatchSchema>,
    provenance: Provenance,
    options: {
      reason?: string | null;
      reviewItemId?: number | null;
      overwriteWithoutReview?: boolean;
    } = {},
  ): ValueRecord {
    const current = this.get(valueId);

    if (current === null) {
      throw new StorageError(`Unknown value id: ${valueId}`, {
        code: "VALUE_NOT_FOUND",
      });
    }

    const parsedPatch = valuePatchSchema.parse(patch);
    const parsedProvenance = requireProvenance(provenance, "Value update");
    const next = valueSchema.parse({
      ...current,
      ...parsedPatch,
      provenance: parsedPatch.provenance ?? current.provenance,
    });
    const storedProvenance = toStoredProvenance(next.provenance);

    this.db
      .prepare(
        `
          UPDATE "values"
          SET label = ?, description = ?, priority = ?, last_affirmed = ?, state = ?, established_at = ?,
              confidence = ?, last_tested_at = ?, last_contradicted_at = ?, support_count = ?,
              contradiction_count = ?, evidence_episode_ids = ?, provenance_kind = ?,
              provenance_episode_ids = ?, provenance_process = ?
          WHERE id = ?
        `,
      )
      .run(
        next.label,
        next.description,
        next.priority,
        next.last_affirmed,
        next.state,
        next.established_at,
        next.confidence,
        next.last_tested_at,
        next.last_contradicted_at,
        next.support_count,
        next.contradiction_count,
        JSON.stringify(next.evidence_episode_ids),
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        valueId,
      );

    recordIdentityEvent(this.identityEventRepository, {
      record_type: "value",
      record_id: valueId,
      action:
        options.reviewItemId === null || options.reviewItemId === undefined
          ? "update"
          : "correction_apply",
      old_value: current,
      new_value: next,
      reason: options.reason ?? null,
      provenance: parsedProvenance,
      review_item_id: options.reviewItemId ?? null,
      overwrite_without_review: options.overwriteWithoutReview === true,
    });

    return next;
  }

  listReinforcementEvents(valueId: ValueId): ValueReinforcementEvent[] {
    return (
      this.db
        .prepare(
          `
            SELECT id, value_id, ts, provenance_kind, provenance_episode_ids, provenance_process
            FROM value_reinforcement_events
            WHERE value_id = ?
            ORDER BY ts ASC, id ASC
          `,
        )
        .all(valueId) as Record<string, unknown>[]
    ).map((row) => ({
      id: Number(row.id),
      value_id: row.value_id as ValueId,
      ts: Number(row.ts),
      provenance: parseStoredProvenance({
        provenance_kind: row.provenance_kind,
        provenance_episode_ids: row.provenance_episode_ids,
        provenance_process: row.provenance_process,
      }),
    }));
  }

  listContradictionEvents(valueId: ValueId): ValueContradictionEvent[] {
    return (
      this.db
        .prepare(
          `
            SELECT id, value_id, ts, weight, provenance_kind, provenance_episode_ids, provenance_process
            FROM value_contradiction_events
            WHERE value_id = ?
            ORDER BY ts ASC, id ASC
          `,
        )
        .all(valueId) as Record<string, unknown>[]
    ).map((row) => ({
      id: Number(row.id),
      value_id: row.value_id as ValueId,
      ts: Number(row.ts),
      weight: Number(row.weight),
      provenance: parseStoredProvenance({
        provenance_kind: row.provenance_kind,
        provenance_episode_ids: row.provenance_episode_ids,
        provenance_process: row.provenance_process,
      }),
    }));
  }
}
