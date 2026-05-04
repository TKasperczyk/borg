import { z } from "zod";

import type { ExecutiveStepsRepository } from "../../executive/index.js";
import { type SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { createGoalId, type EntityId, type GoalId, type StreamEntryId } from "../../util/ids.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { toStoredProvenance, type Provenance } from "../common/provenance.js";
import { type IdentityEventRepository } from "../identity/repository.js";

import { recordIdentityEvent } from "./shared/identity-events.js";
import { requireProvenance } from "./shared/provenance.js";
import { mapGoalRow } from "./shared/sql-mapping.js";
import {
  goalAudienceEntityIdSchema,
  goalPatchSchema,
  goalSchema,
  goalStatusSchema,
  type GoalRecord,
  type GoalStatus,
  type GoalTreeNode,
} from "./types.js";

export type GoalsRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  identityEventRepository?: IdentityEventRepository;
  executiveStepsRepository?: Pick<ExecutiveStepsRepository, "abandonOpenStepsForGoal">;
};

export type GoalListOptions = {
  status?: GoalStatus;
  visibleToAudienceEntityId?: EntityId | null;
};

const GOAL_SELECT_COLUMNS = `
  id, description, priority, parent_goal_id, status, progress_notes, last_progress_ts,
  created_at, target_at, audience_entity_id, source_stream_entry_ids,
  provenance_kind, provenance_episode_ids, provenance_process
`;

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

  private runGoalWrite<T>(callback: () => T): T {
    if (this.db.raw.inTransaction) {
      return callback();
    }

    this.db.exec("BEGIN IMMEDIATE");

    try {
      const result = callback();
      this.db.exec("COMMIT");
      return result;
    } catch (error) {
      try {
        this.db.exec("ROLLBACK");
      } catch {
        // Preserve the original failure.
      }

      throw error;
    }
  }

  private abandonOpenStepsWhenClosingGoal(current: GoalRecord, nextStatus: GoalStatus): void {
    if (current.status !== "active" || nextStatus === "active") {
      return;
    }

    this.options.executiveStepsRepository?.abandonOpenStepsForGoal(current.id, "goal_closed");
  }

  get(goalId: GoalId): GoalRecord | null {
    const row = this.db
      .prepare(
        `
          SELECT ${GOAL_SELECT_COLUMNS}
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
    audienceEntityId?: EntityId | null;
    sourceStreamEntryIds?: readonly StreamEntryId[];
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
    const createdAt = input.createdAt ?? this.clock.now();
    const progressNotes = input.progressNotes ?? null;

    const goal = goalSchema.parse({
      id: input.id ?? createGoalId(),
      description: input.description,
      priority: input.priority,
      parent_goal_id: parentGoalId,
      status: input.status ?? "active",
      progress_notes: progressNotes,
      last_progress_ts:
        progressNotes === null || progressNotes.trim().length === 0 ? null : createdAt,
      created_at: createdAt,
      target_at: input.targetAt ?? null,
      audience_entity_id: input.audienceEntityId ?? null,
      ...(input.sourceStreamEntryIds === undefined || input.sourceStreamEntryIds.length === 0
        ? {}
        : { source_stream_entry_ids: [...input.sourceStreamEntryIds] }),
      provenance,
    });
    const storedProvenance = toStoredProvenance(goal.provenance);

    return this.runGoalWrite(() => {
      this.db
        .prepare(
          `
            INSERT INTO goals (
              id, description, priority, parent_goal_id, status, progress_notes, last_progress_ts,
              created_at, target_at, audience_entity_id, source_stream_entry_ids,
              provenance_kind, provenance_episode_ids, provenance_process
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          `,
        )
        .run(
          goal.id,
          goal.description,
          goal.priority,
          goal.parent_goal_id,
          goal.status,
          goal.progress_notes,
          goal.last_progress_ts,
          goal.created_at,
          goal.target_at,
          goal.audience_entity_id,
          goal.source_stream_entry_ids === undefined
            ? null
            : serializeJsonValue(goal.source_stream_entry_ids),
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
        );
      recordIdentityEvent(this.identityEventRepository, {
        record_type: "goal",
        record_id: goal.id,
        action: "create",
        old_value: null,
        new_value: goal,
        provenance: goal.provenance,
      });
      return goal;
    });
  }

  list(options: GoalListOptions = {}): GoalTreeNode[] {
    const filters: string[] = [];
    const values: unknown[] = [];

    if (options.status !== undefined) {
      filters.push("status = ?");
      values.push(goalStatusSchema.parse(options.status));
    }

    if (options.visibleToAudienceEntityId !== undefined) {
      if (options.visibleToAudienceEntityId === null) {
        filters.push("audience_entity_id IS NULL");
      } else {
        filters.push("(audience_entity_id IS NULL OR audience_entity_id = ?)");
        values.push(goalAudienceEntityIdSchema.parse(options.visibleToAudienceEntityId));
      }
    }

    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const rows = this.db
      .prepare(
        `
          SELECT ${GOAL_SELECT_COLUMNS}
          FROM goals
          ${whereClause}
          ORDER BY priority DESC, created_at ASC
        `,
      )
      .all(...values) as Record<string, unknown>[];

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

    this.runGoalWrite(() => {
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

      this.abandonOpenStepsWhenClosingGoal(current, parsedStatus);

      recordIdentityEvent(this.identityEventRepository, {
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
    const nowMs = this.clock.now();

    this.runGoalWrite(() => {
      const result = this.db
        .prepare(
          `
            UPDATE goals
            SET progress_notes = ?, last_progress_ts = ?, provenance_kind = ?, provenance_episode_ids = ?,
                provenance_process = ?
            WHERE id = ?
          `,
        )
        .run(
          progressNotes,
          nowMs,
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

      recordIdentityEvent(this.identityEventRepository, {
        record_type: "goal",
        record_id: goalId,
        action: "update_progress",
        old_value: current,
        new_value: {
          ...current,
          progress_notes: progressNotes,
          last_progress_ts: nowMs,
          provenance: parsedProvenance,
        },
        provenance: parsedProvenance,
      });
    });
  }

  /**
   * @internal Prefer IdentityService.updateGoal() so established records cannot
   * bypass review gating.
   */
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
    const nextProgressNotes =
      parsedPatch.progress_notes === undefined
        ? current.progress_notes
        : parsedPatch.progress_notes;
    const progressChanged = nextProgressNotes !== current.progress_notes;
    const nextLastProgressTs = !progressChanged ? current.last_progress_ts : this.clock.now();
    const next = goalSchema.parse({
      ...current,
      ...parsedPatch,
      progress_notes: nextProgressNotes,
      last_progress_ts: nextLastProgressTs,
      provenance: parsedPatch.provenance ?? current.provenance,
    });
    const storedProvenance = toStoredProvenance(next.provenance);

    this.runGoalWrite(() => {
      this.db
        .prepare(
          `
            UPDATE goals
            SET description = ?, priority = ?, parent_goal_id = ?, status = ?, progress_notes = ?,
                last_progress_ts = ?, target_at = ?, audience_entity_id = ?, source_stream_entry_ids = ?,
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
          next.last_progress_ts,
          next.target_at,
          next.audience_entity_id,
          next.source_stream_entry_ids === undefined
            ? null
            : serializeJsonValue(next.source_stream_entry_ids),
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
          goalId,
        );

      this.abandonOpenStepsWhenClosingGoal(current, next.status);

      recordIdentityEvent(this.identityEventRepository, {
        record_type: "goal",
        record_id: goalId,
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
    });

    return next;
  }

  restore(goal: GoalRecord): GoalRecord {
    const parsed = goalSchema.parse(goal);
    const storedProvenance = toStoredProvenance(parsed.provenance);

    this.runGoalWrite(() => {
      this.db
        .prepare(
          `
            UPDATE goals
            SET description = ?, priority = ?, parent_goal_id = ?, status = ?, progress_notes = ?,
                last_progress_ts = ?, created_at = ?, target_at = ?, audience_entity_id = ?,
                source_stream_entry_ids = ?, provenance_kind = ?, provenance_episode_ids = ?,
                provenance_process = ?
            WHERE id = ?
          `,
        )
        .run(
          parsed.description,
          parsed.priority,
          parsed.parent_goal_id,
          parsed.status,
          parsed.progress_notes,
          parsed.last_progress_ts,
          parsed.created_at,
          parsed.target_at,
          parsed.audience_entity_id,
          parsed.source_stream_entry_ids === undefined
            ? null
            : serializeJsonValue(parsed.source_stream_entry_ids),
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
          parsed.id,
        );
    });

    return parsed;
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
