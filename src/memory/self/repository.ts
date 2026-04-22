import { SystemClock, type Clock } from "../../util/clock.js";
import { ProvenanceError, StorageError } from "../../util/errors.js";
import {
  createGoalId,
  createTraitId,
  createValueId,
  type EpisodeId,
  type GoalId,
  type TraitId,
  type ValueId,
} from "../../util/ids.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import {
  parseStoredProvenance,
  provenanceSchema,
  toStoredProvenance,
  type Provenance,
} from "../common/provenance.js";

import {
  goalSchema,
  goalStatusSchema,
  traitSchema,
  valueSchema,
  type GoalRecord,
  type GoalStatus,
  type GoalTreeNode,
  type TraitRecord,
  type ValueRecord,
} from "./types.js";

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function mapGoalRow(row: Record<string, unknown>): GoalRecord {
  return goalSchema.parse({
    id: row.id,
    description: row.description,
    priority: Number(row.priority),
    parent_goal_id:
      row.parent_goal_id === null || row.parent_goal_id === undefined
        ? null
        : String(row.parent_goal_id),
    status: row.status,
    progress_notes:
      row.progress_notes === null || row.progress_notes === undefined
        ? null
        : String(row.progress_notes),
    created_at: Number(row.created_at),
    target_at: row.target_at === null || row.target_at === undefined ? null : Number(row.target_at),
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_process: row.provenance_process,
    }),
  });
}

function mapValueRow(row: Record<string, unknown>): ValueRecord {
  return valueSchema.parse({
    id: row.id,
    label: row.label,
    description: row.description,
    priority: Number(row.priority),
    created_at: Number(row.created_at),
    last_affirmed:
      row.last_affirmed === null || row.last_affirmed === undefined ? null : Number(row.last_affirmed),
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_process: row.provenance_process,
    }),
  });
}

function mapTraitRow(row: Record<string, unknown>): TraitRecord {
  return traitSchema.parse({
    id: row.id,
    label: row.label,
    strength: Number(row.strength),
    last_reinforced: Number(row.last_reinforced),
    last_decayed:
      row.last_decayed === null || row.last_decayed === undefined ? null : Number(row.last_decayed),
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_process: row.provenance_process,
    }),
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

export type ValuesRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

export class ValuesRepository {
  private readonly clock: Clock;

  constructor(private readonly options: ValuesRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
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
    const value = valueSchema.parse({
      id: input.id ?? createValueId(),
      label: input.label,
      description: input.description,
      priority: input.priority,
      created_at: input.createdAt ?? this.clock.now(),
      last_affirmed: input.lastAffirmed ?? null,
      provenance,
    });
    const storedProvenance = toStoredProvenance(value.provenance);

    this.db
      .prepare(
        `
          INSERT INTO "values" (
            id, label, description, priority, created_at, last_affirmed, provenance_kind,
            provenance_episode_ids, provenance_process
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
      );
    return value;
  }

  list(): ValueRecord[] {
    return (
      this.db
        .prepare(
          `
            SELECT
              id, label, description, priority, created_at, last_affirmed, provenance_kind,
              provenance_episode_ids, provenance_process
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

  bindToEpisode(valueId: ValueId, episodeId: EpisodeId): void {
    const current = this.db
      .prepare(
        `
          SELECT
            id, label, description, priority, created_at, last_affirmed, provenance_kind,
            provenance_episode_ids, provenance_process
          FROM "values"
          WHERE id = ?
        `,
      )
      .get(valueId) as Record<string, unknown> | undefined;

    if (current === undefined) {
      throw new StorageError(`Unknown value id: ${valueId}`, {
        code: "VALUE_NOT_FOUND",
      });
    }
    const parsed = mapValueRow(current);
    const provenance: Provenance =
      parsed.provenance.kind === "episodes"
        ? {
            kind: "episodes",
            episode_ids: [...new Set([...parsed.provenance.episode_ids, episodeId])],
          }
        : ({
            kind: "episodes",
            episode_ids: [episodeId],
          } satisfies Provenance);
    const storedProvenance = toStoredProvenance(provenance);

    this.db
      .prepare(
        `
          UPDATE "values"
          SET provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?
          WHERE id = ?
        `,
      )
      .run(
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        valueId,
      );
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

export type GoalsRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

export class GoalsRepository {
  private readonly clock: Clock;

  constructor(private readonly options: GoalsRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  add(input: {
    id?: GoalId;
    description: string;
    priority: number;
    parentId?: GoalId | null;
    status?: GoalStatus;
    progressNotes?: string | null;
    provenance: Provenance;
    createdAt?: number;
    targetAt?: number | null;
  }): GoalRecord {
    const parentGoalId = input.parentId ?? null;

    if (parentGoalId !== null) {
      const parentExists =
        this.db.prepare("SELECT 1 FROM goals WHERE id = ?").get(parentGoalId) !== undefined;

      if (!parentExists) {
        throw new StorageError(`Parent goal does not exist: ${parentGoalId}`, {
          code: "GOAL_PARENT_MISSING",
        });
      }
    }
    const provenance = requireProvenance(input.provenance, "Goal");

    const goal = goalSchema.parse({
      id: input.id ?? createGoalId(),
      description: input.description,
      priority: input.priority,
      parent_goal_id: parentGoalId,
      status: input.status ?? "active",
      progress_notes: input.progressNotes ?? null,
      created_at: input.createdAt ?? this.clock.now(),
      target_at: input.targetAt ?? null,
      provenance,
    });
    const storedProvenance = toStoredProvenance(goal.provenance);

    this.db
      .prepare(
        `
          INSERT INTO goals (
            id, description, priority, parent_goal_id, status, progress_notes, created_at, target_at,
            provenance_kind, provenance_episode_ids, provenance_process
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        goal.id,
        goal.description,
        goal.priority,
        goal.parent_goal_id,
        goal.status,
        goal.progress_notes,
        goal.created_at,
        goal.target_at,
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
      );
    return goal;
  }

  list(options: { status?: GoalStatus } = {}): GoalTreeNode[] {
    if (options.status !== undefined) {
      goalStatusSchema.parse(options.status);
    }

    const rows = (
      options.status === undefined
        ? this.db
            .prepare(
              `
                SELECT id, description, priority, parent_goal_id, status, progress_notes, created_at, target_at
                    , provenance_kind, provenance_episode_ids, provenance_process
                FROM goals
                ORDER BY priority DESC, created_at ASC
              `,
            )
            .all()
        : this.db
            .prepare(
              `
                SELECT id, description, priority, parent_goal_id, status, progress_notes, created_at, target_at
                    , provenance_kind, provenance_episode_ids, provenance_process
                FROM goals
                WHERE status = ?
                ORDER BY priority DESC, created_at ASC
              `,
            )
            .all(options.status)
    ) as Record<string, unknown>[];

    const nodes: GoalTreeNode[] = rows.map((row) => ({
      ...mapGoalRow(row),
      children: [],
    }));
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const roots: GoalTreeNode[] = [];

    for (const node of nodes) {
      if (node.parent_goal_id !== null) {
        const parent = byId.get(node.parent_goal_id);

        if (parent !== undefined) {
          parent.children.push(node);
          continue;
        }
      }

      roots.push(node);
    }

    return roots;
  }

  updateStatus(goalId: GoalId, status: GoalStatus, provenance: Provenance): void {
    const parsedStatus = goalStatusSchema.parse(status);
    const storedProvenance = toStoredProvenance(requireProvenance(provenance, "Goal status update"));
    const result = this.db
      .prepare(
        `
          UPDATE goals
          SET status = ?, provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?
          WHERE id = ?
        `,
      )
      .run(
        parsedStatus,
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        goalId,
      );

    if (result.changes === 0) {
      throw new StorageError(`Unknown goal id: ${goalId}`, {
        code: "GOAL_NOT_FOUND",
      });
    }
  }

  updateProgress(goalId: GoalId, progressNotes: string, provenance: Provenance): void {
    const storedProvenance = toStoredProvenance(requireProvenance(provenance, "Goal progress update"));
    const result = this.db
      .prepare(
        `
          UPDATE goals
          SET progress_notes = ?, provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?
          WHERE id = ?
        `,
      )
      .run(
        progressNotes,
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        goalId,
      );

    if (result.changes === 0) {
      throw new StorageError(`Unknown goal id: ${goalId}`, {
        code: "GOAL_NOT_FOUND",
      });
    }
  }

  remove(goalId: GoalId): boolean {
    const reparent = this.db
      .prepare("UPDATE goals SET parent_goal_id = NULL WHERE parent_goal_id = ?")
      .run(goalId);
    void reparent;
    const result = this.db.prepare("DELETE FROM goals WHERE id = ?").run(goalId);
    return result.changes > 0;
  }
}

export type TraitsRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

export type TraitReinforcementEvent = {
  id: number;
  trait_id: TraitId;
  delta: number;
  ts: number;
  provenance: Provenance;
};

export class TraitsRepository {
  private readonly clock: Clock;

  constructor(private readonly options: TraitsRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  reinforce(input: {
    label: string;
    delta: number;
    provenance: Provenance;
    timestamp?: number;
  }): TraitRecord {
    const timestamp = input.timestamp ?? this.clock.now();
    const provenance = requireProvenance(input.provenance, "Trait reinforcement");
    const existing = this.db
      .prepare(
        `
          SELECT id, label, strength, last_reinforced, last_decayed, provenance_kind,
                 provenance_episode_ids, provenance_process
          FROM traits
          WHERE label = ?
        `,
      )
      .get(input.label) as Record<string, unknown> | undefined;
    const nextStrength = clamp(
      (existing === undefined ? 0 : Number(existing.strength)) + input.delta,
      0,
      1,
    );
    const traitId = existing === undefined ? createTraitId() : mapTraitRow(existing).id;
    const aggregateProvenance =
      existing === undefined ? provenance : mapTraitRow(existing).provenance;
    const storedAggregateProvenance = toStoredProvenance(aggregateProvenance);
    const storedEventProvenance = toStoredProvenance(provenance);

    this.db
      .prepare(
        `
          INSERT INTO traits (
            id, label, strength, last_reinforced, last_decayed, provenance_kind,
            provenance_episode_ids, provenance_process
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (label) DO UPDATE SET
            strength = excluded.strength,
            last_reinforced = excluded.last_reinforced,
            last_decayed = excluded.last_decayed
        `,
      )
      .run(
        traitId,
        input.label,
        nextStrength,
        timestamp,
        existing?.last_decayed ?? null,
        storedAggregateProvenance.provenance_kind,
        storedAggregateProvenance.provenance_episode_ids,
        storedAggregateProvenance.provenance_process,
      );
    this.db
      .prepare(
        `
          INSERT INTO trait_reinforcement_events (
            trait_id, delta, ts, provenance_kind, provenance_episode_ids, provenance_process
          ) VALUES (?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        traitId,
        input.delta,
        timestamp,
        storedEventProvenance.provenance_kind,
        storedEventProvenance.provenance_episode_ids,
        storedEventProvenance.provenance_process,
      );

    return traitSchema.parse({
      id: traitId,
      label: input.label,
      strength: nextStrength,
      last_reinforced: timestamp,
      last_decayed:
        existing?.last_decayed === null || existing?.last_decayed === undefined
          ? null
          : Number(existing.last_decayed),
      provenance: aggregateProvenance,
    });
  }

  decay(halfLifeHours: number, nowMs = this.clock.now()): TraitRecord[] {
    if (!Number.isFinite(halfLifeHours) || halfLifeHours <= 0) {
      throw new StorageError("Trait half-life must be positive", {
        code: "TRAIT_DECAY_INVALID",
      });
    }

    const rows = this.db
      .prepare(
        `
          SELECT id, label, strength, last_reinforced, last_decayed, provenance_kind,
                 provenance_episode_ids, provenance_process
          FROM traits
        `,
      )
      .all() as Record<string, unknown>[];
    const update = this.db.prepare(
      "UPDATE traits SET strength = ?, last_decayed = ? WHERE label = ?",
    );
    const records: TraitRecord[] = [];

    for (const row of rows) {
      const lastTouched = Math.max(
        Number(row.last_reinforced),
        row.last_decayed === null || row.last_decayed === undefined ? 0 : Number(row.last_decayed),
      );
      const elapsedHours = Math.max(0, nowMs - lastTouched) / 3_600_000;
      const nextStrength = clamp(
        Number(row.strength) * Math.pow(0.5, elapsedHours / halfLifeHours),
        0,
        1,
      );

      update.run(nextStrength, nowMs, row.label);
      records.push(
        traitSchema.parse({
          id: row.id,
          label: row.label,
          strength: nextStrength,
          last_reinforced: Number(row.last_reinforced),
          last_decayed: nowMs,
          provenance: parseStoredProvenance({
            provenance_kind: row.provenance_kind,
            provenance_episode_ids: row.provenance_episode_ids,
            provenance_process: row.provenance_process,
          }),
        }),
      );
    }

    return records;
  }

  cull(threshold: number): number {
    const result = this.db
      .prepare("DELETE FROM traits WHERE strength < ?")
      .run(clamp(threshold, 0, 1));
    return result.changes;
  }

  list(): TraitRecord[] {
    return (
      this.db
        .prepare(
          `
            SELECT id, label, strength, last_reinforced, last_decayed, provenance_kind,
                   provenance_episode_ids, provenance_process
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
}
