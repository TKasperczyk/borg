import { z } from "zod";

import { type SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { createTraitId, type TraitId } from "../../util/ids.js";
import {
  parseStoredProvenance,
  toStoredProvenance,
  type Provenance,
} from "../common/provenance.js";
import { type IdentityEventRepository } from "../identity/repository.js";

import { clamp, computeConfidence, summarizeEvidence } from "./shared/evidence.js";
import { recordIdentityEvent } from "./shared/identity-events.js";
import {
  getPromotionMetadataFromEvents,
  TRAIT_PROMOTION_THRESHOLD,
  type PromotionMetadata,
} from "./shared/promotion.js";
import { requireProvenance } from "./shared/provenance.js";
import { mapTraitRow } from "./shared/sql-mapping.js";
import { traitPatchSchema, traitSchema, type TraitRecord } from "./types.js";

export type TraitsRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  identityEventRepository?: IdentityEventRepository;
};

export type TraitReinforcementEvent = {
  id: number;
  trait_id: TraitId;
  delta: number;
  ts: number;
  provenance: Provenance;
};

export type TraitContradictionEvent = {
  id: number;
  trait_id: TraitId;
  ts: number;
  provenance: Provenance;
  weight: number;
};

export class TraitsRepository {
  private readonly clock: Clock;

  constructor(private readonly options: TraitsRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private get identityEventRepository(): IdentityEventRepository | undefined {
    return this.options.identityEventRepository;
  }

  private getById(traitId: TraitId): Record<string, unknown> | undefined {
    return this.db
      .prepare(
        `
          SELECT id, label, strength, last_reinforced, last_decayed, state, established_at,
                 confidence, last_tested_at, last_contradicted_at, support_count, contradiction_count,
                 evidence_episode_ids, provenance_kind, provenance_episode_ids, provenance_process
          FROM traits
          WHERE id = ?
        `,
      )
      .get(traitId) as Record<string, unknown> | undefined;
  }

  private getByLabel(label: string): Record<string, unknown> | undefined {
    return this.db
      .prepare(
        `
          SELECT id, label, strength, last_reinforced, last_decayed, state, established_at,
                 confidence, last_tested_at, last_contradicted_at, support_count, contradiction_count,
                 evidence_episode_ids, provenance_kind, provenance_episode_ids, provenance_process
          FROM traits
          WHERE label = ?
        `,
      )
      .get(label) as Record<string, unknown> | undefined;
  }

  private getPromotionMetadata(traitId: TraitId): PromotionMetadata {
    return getPromotionMetadataFromEvents(
      this.listReinforcementEvents(traitId),
      TRAIT_PROMOTION_THRESHOLD,
    );
  }

  private insertContradictionEvent(
    traitId: TraitId,
    provenance: Provenance,
    weight: number,
    timestamp: number,
  ): TraitContradictionEvent {
    const storedProvenance = toStoredProvenance(provenance);
    const result = this.db
      .prepare(
        `
          INSERT INTO trait_contradiction_events (
            trait_id, ts, weight, provenance_kind, provenance_episode_ids, provenance_process
          ) VALUES (?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        traitId,
        timestamp,
        weight,
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
      );

    return {
      id: Number(result.lastInsertRowid),
      trait_id: traitId,
      ts: timestamp,
      provenance,
      weight,
    };
  }

  get(traitId: TraitId): TraitRecord | null {
    const row = this.getById(traitId);
    return row === undefined ? null : mapTraitRow(row);
  }

  reinforce(input: {
    label: string;
    delta: number;
    provenance: Provenance;
    timestamp?: number;
  }): TraitRecord {
    const timestamp = input.timestamp ?? this.clock.now();
    const provenance = requireProvenance(input.provenance, "Trait reinforcement");
    const existing = this.getByLabel(input.label);
    const current = existing === undefined ? null : mapTraitRow(existing);
    const nextStrength = clamp(
      (existing === undefined ? 0 : Number(existing.strength)) + input.delta,
      0,
      1,
    );
    const traitId = existing === undefined ? createTraitId() : current!.id;
    const aggregateProvenance = existing === undefined ? provenance : current!.provenance;
    const currentEvidenceEpisodeIds = current === null ? [] : current.evidence_episode_ids;
    const currentState =
      current === null
        ? {
            state: "candidate" as const,
            established_at: null,
          }
        : {
            state: current.state,
            established_at: current.established_at,
          };
    const storedAggregateProvenance = toStoredProvenance(aggregateProvenance);
    const storedEventProvenance = toStoredProvenance(provenance);

    const insertOrUpdate = this.db.prepare(
      `
        INSERT INTO traits (
          id, label, strength, last_reinforced, last_decayed, provenance_kind,
          provenance_episode_ids, provenance_process, state, established_at, confidence,
          last_tested_at, last_contradicted_at, support_count, contradiction_count, evidence_episode_ids
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (label) DO UPDATE SET
          strength = excluded.strength,
          last_reinforced = excluded.last_reinforced,
          last_decayed = excluded.last_decayed
      `,
    );
    const insertEvent = this.db.prepare(
      `
        INSERT INTO trait_reinforcement_events (
          trait_id, delta, ts, provenance_kind, provenance_episode_ids, provenance_process
        ) VALUES (?, ?, ?, ?, ?, ?)
      `,
    );
    const updatePromotion = this.db.prepare(
      `
        UPDATE traits
        SET state = ?, established_at = ?,
            provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?, confidence = ?,
            last_tested_at = ?, last_contradicted_at = ?, support_count = ?, contradiction_count = ?,
            evidence_episode_ids = ?
        WHERE id = ?
      `,
    );
    const apply = this.db.transaction(() => {
      insertOrUpdate.run(
        traitId,
        input.label,
        nextStrength,
        timestamp,
        existing?.last_decayed ?? null,
        storedAggregateProvenance.provenance_kind,
        storedAggregateProvenance.provenance_episode_ids,
        storedAggregateProvenance.provenance_process,
        currentState.state,
        currentState.established_at,
        current === null ? computeConfidence(0, 0) : current.confidence,
        current?.last_tested_at ?? null,
        current?.last_contradicted_at ?? null,
        current?.support_count ?? 0,
        current?.contradiction_count ?? 0,
        JSON.stringify(currentEvidenceEpisodeIds),
      );
      insertEvent.run(
        traitId,
        input.delta,
        timestamp,
        storedEventProvenance.provenance_kind,
        storedEventProvenance.provenance_episode_ids,
        storedEventProvenance.provenance_process,
      );
      const promotion = this.getPromotionMetadata(traitId);
      const evidence = summarizeEvidence(
        this.listReinforcementEvents(traitId),
        this.listContradictionEvents(traitId),
      );
      const nextState = currentState.state === "established" ? currentState.state : promotion.state;
      const finalProvenance =
        currentState.state !== "established" && promotion.state === "established"
          ? (promotion.promotionProvenance ?? aggregateProvenance)
          : aggregateProvenance;
      const storedFinalProvenance = toStoredProvenance(finalProvenance);
      updatePromotion.run(
        nextState,
        currentState.state === "established"
          ? currentState.established_at
          : promotion.established_at,
        storedFinalProvenance.provenance_kind,
        storedFinalProvenance.provenance_episode_ids,
        storedFinalProvenance.provenance_process,
        computeConfidence(evidence.supportCount, evidence.contradictionCount),
        evidence.lastTestedAt,
        evidence.lastContradictedAt,
        evidence.supportCount,
        evidence.contradictionCount,
        JSON.stringify(evidence.evidenceEpisodeIds),
        traitId,
      );
    });
    apply();

    const next = this.get(traitId);

    if (next === null) {
      throw new StorageError(`Failed to reload trait after reinforcement: ${traitId}`, {
        code: "TRAIT_NOT_FOUND",
      });
    }

    if (current === null) {
      recordIdentityEvent(this.identityEventRepository, {
        record_type: "trait",
        record_id: traitId,
        action: "create",
        old_value: null,
        new_value: next,
        provenance,
        ts: timestamp,
      });
    }

    if (current !== null && current.state !== "established" && next.state === "established") {
      recordIdentityEvent(this.identityEventRepository, {
        record_type: "trait",
        record_id: traitId,
        action: "promote",
        old_value: current,
        new_value: next,
        provenance,
        ts: timestamp,
      });
    }

    return next;
  }

  recordContradiction(input: {
    label: string;
    provenance: Provenance;
    weight?: number;
    timestamp?: number;
  }): TraitRecord {
    const existing = this.getByLabel(input.label);

    if (existing === undefined) {
      throw new StorageError(`Unknown trait label: ${input.label}`, {
        code: "TRAIT_NOT_FOUND",
      });
    }

    const current = mapTraitRow(existing);
    const timestamp = input.timestamp ?? this.clock.now();
    const provenance = requireProvenance(input.provenance, "Trait contradiction");
    const weight = Number.isFinite(input.weight) && input.weight !== undefined ? input.weight : 1;
    this.insertContradictionEvent(
      current.id,
      provenance,
      clamp(weight, 0, Number.POSITIVE_INFINITY),
      timestamp,
    );

    const evidence = summarizeEvidence(
      this.listReinforcementEvents(current.id),
      this.listContradictionEvents(current.id),
    );
    const next = traitSchema.parse({
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
          UPDATE traits
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
        current.id,
      );

    recordIdentityEvent(this.identityEventRepository, {
      record_type: "trait",
      record_id: current.id,
      action: "contradict",
      old_value: current,
      new_value: next,
      provenance,
      ts: timestamp,
    });

    return next;
  }

  decay(
    halfLifeHours: number,
    nowMs = this.clock.now(),
    options: {
      traitIds?: TraitId[];
    } = {},
  ): TraitRecord[] {
    if (!Number.isFinite(halfLifeHours) || halfLifeHours <= 0) {
      throw new StorageError("Trait half-life must be positive", {
        code: "TRAIT_DECAY_INVALID",
      });
    }

    const traitIds = options.traitIds ?? null;
    const placeholders = traitIds?.map(() => "?").join(", ") ?? "";
    const rows = this.db
      .prepare(
        `
          SELECT id, label, strength, last_reinforced, last_decayed, provenance_kind,
                 provenance_episode_ids, provenance_process, state, established_at, confidence,
                 last_tested_at, last_contradicted_at, support_count, contradiction_count,
                 evidence_episode_ids
          FROM traits
          ${traitIds === null || traitIds.length === 0 ? "" : `WHERE id IN (${placeholders})`}
        `,
      )
      .all(...(traitIds ?? [])) as Record<string, unknown>[];
    const update = this.db.prepare(
      "UPDATE traits SET strength = ?, last_decayed = ? WHERE label = ?",
    );
    const records: TraitRecord[] = [];

    for (const row of rows) {
      const current = mapTraitRow(row);
      const lastTouched = Math.max(current.last_reinforced, current.last_decayed ?? 0);
      const elapsedHours = Math.max(0, nowMs - lastTouched) / 3_600_000;
      const nextStrength = clamp(
        current.strength * Math.pow(0.5, elapsedHours / halfLifeHours),
        0,
        1,
      );

      const next = traitSchema.parse({
        ...current,
        strength: nextStrength,
        last_decayed: nowMs,
      });

      update.run(next.strength, nowMs, next.label);
      records.push(next);

      if (Math.abs(next.strength - current.strength) > Number.EPSILON) {
        recordIdentityEvent(this.identityEventRepository, {
          record_type: "trait",
          record_id: current.id,
          action: "decay",
          old_value: current,
          new_value: next,
          provenance: current.provenance,
          ts: nowMs,
        });
      }
    }

    return records;
  }

  cull(threshold: number): number {
    const result = this.db
      .prepare("DELETE FROM traits WHERE strength < ?")
      .run(clamp(threshold, 0, 1));
    return result.changes;
  }

  /**
   * @internal Prefer IdentityService.updateTrait() so episode-backed established
   * records cannot bypass review gating.
   */
  update(
    traitId: TraitId,
    patch: z.infer<typeof traitPatchSchema>,
    provenance: Provenance,
    options: {
      reason?: string | null;
      reviewItemId?: number | null;
      overwriteWithoutReview?: boolean;
    } = {},
  ): TraitRecord {
    const current = this.get(traitId);

    if (current === null) {
      throw new StorageError(`Unknown trait id: ${traitId}`, {
        code: "TRAIT_NOT_FOUND",
      });
    }

    const parsedPatch = traitPatchSchema.parse(patch);
    const parsedProvenance = requireProvenance(provenance, "Trait update");
    const next = traitSchema.parse({
      ...current,
      ...parsedPatch,
      provenance: parsedPatch.provenance ?? current.provenance,
    });
    const storedProvenance = toStoredProvenance(next.provenance);

    this.db
      .prepare(
        `
          UPDATE traits
          SET label = ?, strength = ?, last_reinforced = ?, last_decayed = ?, state = ?, established_at = ?,
              confidence = ?, last_tested_at = ?, last_contradicted_at = ?, support_count = ?,
              contradiction_count = ?, evidence_episode_ids = ?, provenance_kind = ?,
              provenance_episode_ids = ?, provenance_process = ?
          WHERE id = ?
        `,
      )
      .run(
        next.label,
        next.strength,
        next.last_reinforced,
        next.last_decayed,
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
        traitId,
      );

    recordIdentityEvent(this.identityEventRepository, {
      record_type: "trait",
      record_id: traitId,
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

  remove(traitId: TraitId): boolean {
    const result = this.db.prepare("DELETE FROM traits WHERE id = ?").run(traitId);
    return result.changes > 0;
  }

  list(): TraitRecord[] {
    return (
      this.db
        .prepare(
          `
            SELECT id, label, strength, last_reinforced, last_decayed, provenance_kind,
                   provenance_episode_ids, provenance_process, state, established_at, confidence,
                   last_tested_at, last_contradicted_at, support_count, contradiction_count,
                   evidence_episode_ids
            FROM traits
            ORDER BY strength DESC, label ASC
          `,
        )
        .all() as Record<string, unknown>[]
    ).map((row) => mapTraitRow(row));
  }

  listReinforcementEvents(traitId: TraitId): TraitReinforcementEvent[] {
    return (
      this.db
        .prepare(
          `
            SELECT id, trait_id, delta, ts, provenance_kind, provenance_episode_ids, provenance_process
            FROM trait_reinforcement_events
            WHERE trait_id = ?
            ORDER BY ts ASC, id ASC
          `,
        )
        .all(traitId) as Record<string, unknown>[]
    ).map((row) => ({
      id: Number(row.id),
      trait_id: row.trait_id as TraitId,
      delta: Number(row.delta),
      ts: Number(row.ts),
      provenance: parseStoredProvenance({
        provenance_kind: row.provenance_kind,
        provenance_episode_ids: row.provenance_episode_ids,
        provenance_process: row.provenance_process,
      }),
    }));
  }

  listContradictionEvents(traitId: TraitId): TraitContradictionEvent[] {
    return (
      this.db
        .prepare(
          `
            SELECT id, trait_id, ts, weight, provenance_kind, provenance_episode_ids, provenance_process
            FROM trait_contradiction_events
            WHERE trait_id = ?
            ORDER BY ts ASC, id ASC
          `,
        )
        .all(traitId) as Record<string, unknown>[]
    ).map((row) => ({
      id: Number(row.id),
      trait_id: row.trait_id as TraitId,
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
