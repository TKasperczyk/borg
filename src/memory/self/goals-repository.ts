import { z } from "zod";

import { type SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { createGoalId, type GoalId } from "../../util/ids.js";
import { toStoredProvenance, type Provenance } from "../common/provenance.js";
import { type IdentityEventRepository } from "../identity/repository.js";

import { recordIdentityEvent } from "./shared/identity-events.js";
import { requireProvenance } from "./shared/provenance.js";
import { mapGoalRow } from "./shared/sql-mapping.js";
import {
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
          SELECT id, description, priority, parent_goal_id, status, progress_notes, last_progress_ts,
                 created_at, target_at
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
      provenance,
    });
    const storedProvenance = toStoredProvenance(goal.provenance);

    this.db
      .prepare(
        `
          INSERT INTO goals (
            id, description, priority, parent_goal_id, status, progress_notes, last_progress_ts,
            created_at, target_at, provenance_kind, provenance_episode_ids, provenance_process
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                SELECT id, description, priority, parent_goal_id, status, progress_notes, last_progress_ts,
                       created_at, target_at, provenance_kind, provenance_episode_ids, provenance_process
                FROM goals
                ORDER BY priority DESC, created_at ASC
              `,
            )
            .all()
        : this.db
            .prepare(
              `
                SELECT id, description, priority, parent_goal_id, status, progress_notes, last_progress_ts,
                       created_at, target_at, provenance_kind, provenance_episode_ids, provenance_process
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
  }

  /**
   * @internal Prefer IdentityService.updateGoal() so episode-backed established
   * records cannot bypass review gating.
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

    this.db
      .prepare(
        `
          UPDATE goals
          SET description = ?, priority = ?, parent_goal_id = ?, status = ?, progress_notes = ?,
              last_progress_ts = ?, target_at = ?, provenance_kind = ?, provenance_episode_ids = ?,
              provenance_process = ?
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
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        goalId,
      );

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
