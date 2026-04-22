import { z } from "zod";

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
  isEpisodeProvenance,
  parseStoredProvenance,
  provenanceSchema,
  toStoredProvenance,
  type Provenance,
} from "../common/provenance.js";
import { IdentityEventRepository } from "../identity/repository.js";

import {
  goalSchema,
  goalPatchSchema,
  goalStatusSchema,
  traitSchema,
  traitPatchSchema,
  valueSchema,
  valuePatchSchema,
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
    state: row.state,
    established_at:
      row.established_at === null || row.established_at === undefined
        ? null
        : Number(row.established_at),
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
    state: row.state,
    established_at:
      row.established_at === null || row.established_at === undefined
        ? null
        : Number(row.established_at),
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

const VALUE_PROMOTION_THRESHOLD = 3;
const TRAIT_PROMOTION_THRESHOLD = 5;
const PROMOTION_PROVENANCE_EPISODE_LIMIT = 3;

type PromotionMetadata = Pick<ValueRecord, "state" | "established_at"> & {
  promotionProvenance: Provenance | null;
};

function getPromotionMetadataFromEvents<T extends { ts: number; provenance: Provenance }>(
  events: T[],
  threshold: number,
): PromotionMetadata {
  const distinctEpisodeIds = new Set<EpisodeId>();
  const latestEpisodeTs = new Map<EpisodeId, number>();
  let establishedAt: number | null = null;

  for (const event of events) {
    if (!isEpisodeProvenance(event.provenance)) {
      continue;
    }

    for (const episodeId of event.provenance.episode_ids) {
      distinctEpisodeIds.add(episodeId);
      latestEpisodeTs.set(episodeId, event.ts);
    }

    if (distinctEpisodeIds.size >= threshold) {
      establishedAt = event.ts;
      break;
    }
  }

  if (establishedAt === null) {
    return {
      state: "candidate",
      established_at: null,
      promotionProvenance: null,
    };
  }

  const promotionEpisodeIds = [...latestEpisodeTs.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, PROMOTION_PROVENANCE_EPISODE_LIMIT)
    .map(([episodeId]) => episodeId);

  return {
    state: "established",
    established_at: establishedAt,
    promotionProvenance: {
      kind: "episodes",
      episode_ids: promotionEpisodeIds,
    },
  };
}

function resolveValueInitialState(
  provenance: Provenance,
  timestamp: number,
): Pick<ValueRecord, "state" | "established_at"> {
  switch (provenance.kind) {
    case "manual":
    case "system":
      return {
        state: "established",
        established_at: timestamp,
      };
    case "episodes":
      return {
        state:
          new Set(provenance.episode_ids).size >= VALUE_PROMOTION_THRESHOLD
            ? "established"
            : "candidate",
        established_at:
          new Set(provenance.episode_ids).size >= VALUE_PROMOTION_THRESHOLD ? timestamp : null,
      };
    case "offline":
      return {
        state: "candidate",
        established_at: null,
      };
  }
}

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
            provenance_kind, provenance_episode_ids, provenance_process
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

  private getPromotionMetadata(
    valueId: ValueId,
  ): PromotionMetadata {
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
    const value = valueSchema.parse({
      id: input.id ?? createValueId(),
      label: input.label,
      description: input.description,
      priority: input.priority,
      created_at: createdAt,
      last_affirmed: input.lastAffirmed ?? null,
      state: initialState.state,
      established_at: initialState.established_at,
      provenance,
    });
    const storedProvenance = toStoredProvenance(value.provenance);

    this.db
      .prepare(
        `
          INSERT INTO "values" (
            id, label, description, priority, created_at, last_affirmed, provenance_kind,
            provenance_episode_ids, provenance_process, state, established_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    this.identityEventRepository?.record({
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
              provenance_kind, provenance_episode_ids, provenance_process
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

  reinforce(
    valueId: ValueId,
    provenance: Provenance,
    timestamp = this.clock.now(),
  ): ValueRecord {
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
    const isPromoted = current.state !== "established" && promotion.state === "established";
    const nextProvenance: Provenance =
      isPromoted && promotion.promotionProvenance !== null
        ? promotion.promotionProvenance
        : current.provenance.kind === "episodes" && isEpisodeProvenance(parsedProvenance)
          ? {
              kind: "episodes",
              episode_ids: [...new Set([...current.provenance.episode_ids, ...parsedProvenance.episode_ids])],
            }
          : isEpisodeProvenance(parsedProvenance)
            ? parsedProvenance
            : current.provenance;
    const next = valueSchema.parse({
      ...current,
      provenance: nextProvenance,
      state: current.state === "established" ? current.state : promotion.state,
      established_at:
        current.state === "established" ? current.established_at : promotion.established_at,
    });
    const storedProvenance = toStoredProvenance(next.provenance);

    this.db
      .prepare(
        `
          UPDATE "values"
          SET provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?, state = ?, established_at = ?
          WHERE id = ?
        `,
      )
      .run(
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        next.state,
        next.established_at,
        valueId,
      );

    if (current.state !== "established" && next.state === "established") {
      this.identityEventRepository?.record({
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

  bindToEpisode(valueId: ValueId, episodeId: EpisodeId): void {
    void this.reinforce(valueId, {
      kind: "episodes",
      episode_ids: [episodeId],
    });
  }

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
              provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?
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
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        valueId,
      );

    this.identityEventRepository?.record({
      record_type: "value",
      record_id: valueId,
      action: options.reviewItemId === null || options.reviewItemId === undefined ? "update" : "correction_apply",
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
}

export type GoalsRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  identityEventRepository?: IdentityEventRepository;
};

export class GoalsRepository {
  private readonly clock: Clock;

  constructor(private readonly options: GoalsRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private get identityEventRepository(): IdentityEventRepository | undefined {
    return this.options.identityEventRepository;
  }

  get(goalId: GoalId): GoalRecord | null {
    const row = this.db
      .prepare(
        `
          SELECT id, description, priority, parent_goal_id, status, progress_notes, created_at, target_at
              , provenance_kind, provenance_episode_ids, provenance_process
          FROM goals
          WHERE id = ?
        `,
      )
      .get(goalId) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapGoalRow(row);
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
    this.identityEventRepository?.record({
      record_type: "goal",
      record_id: goal.id,
      action: "create",
      old_value: null,
      new_value: goal,
      provenance: goal.provenance,
    });
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
    const current = this.get(goalId);

    if (current === null) {
      throw new StorageError(`Unknown goal id: ${goalId}`, {
        code: "GOAL_NOT_FOUND",
      });
    }

    const parsedStatus = goalStatusSchema.parse(status);
    const parsedProvenance = requireProvenance(provenance, "Goal status update");
    const storedProvenance = toStoredProvenance(parsedProvenance);
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

    this.identityEventRepository?.record({
      record_type: "goal",
      record_id: goalId,
      action: "update",
      old_value: current,
      new_value: {
        ...current,
        status: parsedStatus,
        provenance: parsedProvenance,
      },
      provenance: parsedProvenance,
    });
  }

  updateProgress(goalId: GoalId, progressNotes: string, provenance: Provenance): void {
    const current = this.get(goalId);

    if (current === null) {
      throw new StorageError(`Unknown goal id: ${goalId}`, {
        code: "GOAL_NOT_FOUND",
      });
    }

    const parsedProvenance = requireProvenance(provenance, "Goal progress update");
    const storedProvenance = toStoredProvenance(parsedProvenance);
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

    this.identityEventRepository?.record({
      record_type: "goal",
      record_id: goalId,
      action: "update",
      old_value: current,
      new_value: {
        ...current,
        progress_notes: progressNotes,
        provenance: parsedProvenance,
      },
      provenance: parsedProvenance,
    });
  }

  update(
    goalId: GoalId,
    patch: z.infer<typeof goalPatchSchema>,
    provenance: Provenance,
    options: {
      reason?: string | null;
      reviewItemId?: number | null;
      overwriteWithoutReview?: boolean;
    } = {},
  ): GoalRecord {
    const current = this.get(goalId);

    if (current === null) {
      throw new StorageError(`Unknown goal id: ${goalId}`, {
        code: "GOAL_NOT_FOUND",
      });
    }

    const parsedPatch = goalPatchSchema.parse(patch);
    const parsedProvenance = requireProvenance(provenance, "Goal update");
    const next = goalSchema.parse({
      ...current,
      ...parsedPatch,
      provenance: parsedPatch.provenance ?? current.provenance,
    });
    const storedProvenance = toStoredProvenance(next.provenance);

    this.db
      .prepare(
        `
          UPDATE goals
          SET description = ?, priority = ?, parent_goal_id = ?, status = ?, progress_notes = ?, target_at = ?,
              provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?
          WHERE id = ?
        `,
      )
      .run(
        next.description,
        next.priority,
        next.parent_goal_id,
        next.status,
        next.progress_notes,
        next.target_at,
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        goalId,
      );

    this.identityEventRepository?.record({
      record_type: "goal",
      record_id: goalId,
      action: options.reviewItemId === null || options.reviewItemId === undefined ? "update" : "correction_apply",
      old_value: current,
      new_value: next,
      reason: options.reason ?? null,
      provenance: parsedProvenance,
      review_item_id: options.reviewItemId ?? null,
      overwrite_without_review: options.overwriteWithoutReview === true,
    });

    return next;
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
  identityEventRepository?: IdentityEventRepository;
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

  private get identityEventRepository(): IdentityEventRepository | undefined {
    return this.options.identityEventRepository;
  }

  private getById(traitId: TraitId): Record<string, unknown> | undefined {
    return this.db
      .prepare(
        `
          SELECT id, label, strength, last_reinforced, last_decayed, state, established_at,
                 provenance_kind, provenance_episode_ids, provenance_process
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
                 provenance_kind, provenance_episode_ids, provenance_process
          FROM traits
          WHERE label = ?
        `,
      )
      .get(label) as Record<string, unknown> | undefined;
  }

  private getPromotionMetadata(
    traitId: TraitId,
  ): PromotionMetadata {
    return getPromotionMetadataFromEvents(
      this.listReinforcementEvents(traitId),
      TRAIT_PROMOTION_THRESHOLD,
    );
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
          provenance_episode_ids, provenance_process, state, established_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?
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
      const finalProvenance =
        currentState.state !== "established" && promotion.state === "established"
          ? (promotion.promotionProvenance ?? aggregateProvenance)
          : aggregateProvenance;
      const storedFinalProvenance = toStoredProvenance(finalProvenance);
      updatePromotion.run(
        currentState.state === "established" ? currentState.state : promotion.state,
        currentState.state === "established" ? currentState.established_at : promotion.established_at,
        storedFinalProvenance.provenance_kind,
        storedFinalProvenance.provenance_episode_ids,
        storedFinalProvenance.provenance_process,
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
      this.identityEventRepository?.record({
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
      this.identityEventRepository?.record({
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
                 provenance_episode_ids, provenance_process, state, established_at
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
          state: row.state,
          established_at:
            row.established_at === null || row.established_at === undefined
              ? null
              : Number(row.established_at),
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
              provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?
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
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        traitId,
      );

    this.identityEventRepository?.record({
      record_type: "trait",
      record_id: traitId,
      action: options.reviewItemId === null || options.reviewItemId === undefined ? "update" : "correction_apply",
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
                   provenance_episode_ids, provenance_process, state, established_at
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
