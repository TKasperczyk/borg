import { z } from "zod";

import type { ExecutiveStepsRepository } from "../../executive/index.js";
import { type SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import {
  createGoalId,
  type SharedStateEntryId,
  type EntityId,
  type GoalId,
  type StreamEntryId,
} from "../../util/ids.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { toStoredProvenance, type Provenance } from "../common/provenance.js";
import {
  assertIdentityCasUpdated,
  expectedRecordVersion,
  nextRecordVersion,
  type IdentityCasOptions,
} from "../common/cas.js";
import { type IdentityEventRepository } from "../identity/repository.js";

import { recordIdentityEvent } from "./shared/identity-events.js";
import { requireProvenance } from "./shared/provenance.js";
import { mapGoalRow } from "./shared/sql-mapping.js";
import {
  goalAudienceEntityIdSchema,
  goalOwnerEntityIdSchema,
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
  ownerEntityId?: EntityId | null;
};

export type GoalFollowupDueCandidate = {
  goal: GoalRecord;
  due_at: number;
};

export type GoalFollowupDueCandidateOptions = {
  lookaheadMs: number;
  staleMs: number;
  limit: number;
};

const GOAL_SELECT_COLUMNS = `
  id, record_version, description, terminal_condition, priority, parent_goal_id, status,
  progress_notes, last_progress_ts, created_at, target_at, audience_entity_id, owner_entity_id,
  source_stream_entry_ids, canonicalized_by_artifact_entry_id, provenance_kind,
  provenance_episode_ids, provenance_stream_entry_ids, provenance_process
`;

export type GoalStatusUpdateOptions = IdentityCasOptions & {
  canonicalizedByArtifactEntryId?: SharedStateEntryId | null;
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
    terminalCondition?: string | null;
    priority: number;
    parentId?: GoalId | null;
    status?: GoalStatus;
    progressNotes?: string | null;
    provenance: Provenance;
    createdAt?: number;
    targetAt?: number | null;
    audienceEntityId?: EntityId | null;
    ownerEntityId?: EntityId | null;
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
      record_version: 1,
      description: input.description,
      terminal_condition: input.terminalCondition ?? null,
      priority: input.priority,
      parent_goal_id: parentGoalId,
      status: input.status ?? "active",
      progress_notes: progressNotes,
      last_progress_ts:
        progressNotes === null || progressNotes.trim().length === 0 ? null : createdAt,
      created_at: createdAt,
      target_at: input.targetAt ?? null,
      audience_entity_id: input.audienceEntityId ?? null,
      owner_entity_id: input.ownerEntityId ?? null,
      canonicalized_by_artifact_entry_id: null,
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
              id, description, terminal_condition, priority, parent_goal_id, status, progress_notes,
              last_progress_ts, created_at, target_at, audience_entity_id, owner_entity_id,
              source_stream_entry_ids, canonicalized_by_artifact_entry_id, provenance_kind,
              provenance_episode_ids, provenance_stream_entry_ids, provenance_process
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          `,
        )
        .run(
          goal.id,
          goal.description,
          goal.terminal_condition,
          goal.priority,
          goal.parent_goal_id,
          goal.status,
          goal.progress_notes,
          goal.last_progress_ts,
          goal.created_at,
          goal.target_at,
          goal.audience_entity_id,
          goal.owner_entity_id,
          goal.source_stream_entry_ids === undefined
            ? null
            : serializeJsonValue(goal.source_stream_entry_ids),
          goal.canonicalized_by_artifact_entry_id,
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_stream_entry_ids ?? null,
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

    if (options.ownerEntityId !== undefined) {
      if (options.ownerEntityId === null) {
        filters.push("owner_entity_id IS NULL");
      } else {
        filters.push("owner_entity_id = ?");
        values.push(goalOwnerEntityIdSchema.parse(options.ownerEntityId));
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

  listActiveFollowupDueCandidatesReadOnly(
    options: GoalFollowupDueCandidateOptions,
  ): GoalFollowupDueCandidate[] {
    const limit = Number.isFinite(options.limit) ? Math.max(0, Math.floor(options.limit)) : 0;
    const rows = this.db
      .prepare(
        `
          SELECT ${GOAL_SELECT_COLUMNS},
            CASE
              WHEN target_at IS NULL THEN COALESCE(last_progress_ts, created_at) + ? + 1
              WHEN target_at - ? + 1 < COALESCE(last_progress_ts, created_at) + ? + 1
                THEN target_at - ? + 1
              ELSE COALESCE(last_progress_ts, created_at) + ? + 1
            END AS autonomy_due_at
          FROM goals
          WHERE status = 'active'
          ORDER BY autonomy_due_at ASC, priority DESC, created_at ASC, id ASC
          LIMIT ?
        `,
      )
      .all(
        options.staleMs,
        options.lookaheadMs,
        options.staleMs,
        options.lookaheadMs,
        options.staleMs,
        limit,
      ) as Record<string, unknown>[];

    return rows.map((row) => ({
      goal: mapGoalRow(row),
      due_at: Number(row.autonomy_due_at),
    }));
  }

  updateStatus(
    goalId: GoalId,
    status: GoalStatus,
    provenance: Provenance,
    options: GoalStatusUpdateOptions = {},
  ): void {
    const current = this.get(goalId);

    if (current === null) {
      throw new StorageError(`Unknown goal id: ${goalId}`, {
        code: "GOAL_NOT_FOUND",
      });
    }

    const parsedStatus = goalStatusSchema.parse(status);
    const expectedVersion = expectedRecordVersion(current, options);
    const parsedProvenance = requireProvenance(provenance, "Goal status update");
    const storedProvenance = toStoredProvenance(parsedProvenance);

    this.runGoalWrite(() => {
      const result = this.db
        .prepare(
          `
            UPDATE goals
            SET status = ?, provenance_kind = ?, provenance_episode_ids = ?,
                provenance_stream_entry_ids = ?, provenance_process = ?,
                canonicalized_by_artifact_entry_id = ?,
                record_version = record_version + 1
            WHERE id = ? AND record_version = ?
          `,
        )
        .run(
          parsedStatus,
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_stream_entry_ids ?? null,
          storedProvenance.provenance_process,
          options.canonicalizedByArtifactEntryId === undefined
            ? (current.canonicalized_by_artifact_entry_id ?? null)
            : options.canonicalizedByArtifactEntryId,
          goalId,
          expectedVersion,
        );

      if (result.changes === 0) {
        assertIdentityCasUpdated({
          result,
          recordType: "goal",
          recordId: goalId,
          expectedVersion,
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
          record_version: nextRecordVersion(expectedVersion),
          status: parsedStatus,
          canonicalized_by_artifact_entry_id:
            options.canonicalizedByArtifactEntryId === undefined
              ? (current.canonicalized_by_artifact_entry_id ?? null)
              : options.canonicalizedByArtifactEntryId,
          provenance: parsedProvenance,
        },
        provenance: parsedProvenance,
      });
    });
  }

  updateProgress(
    goalId: GoalId,
    progressNotes: string,
    provenance: Provenance,
    options: IdentityCasOptions = {},
  ): void {
    const current = this.get(goalId);

    if (current === null) {
      throw new StorageError(`Unknown goal id: ${goalId}`, {
        code: "GOAL_NOT_FOUND",
      });
    }

    const parsedProvenance = requireProvenance(provenance, "Goal progress update");
    const expectedVersion = expectedRecordVersion(current, options);
    const storedProvenance = toStoredProvenance(parsedProvenance);
    const nowMs = this.clock.now();

    this.runGoalWrite(() => {
      const result = this.db
        .prepare(
          `
            UPDATE goals
            SET progress_notes = ?, last_progress_ts = ?, provenance_kind = ?, provenance_episode_ids = ?,
                provenance_stream_entry_ids = ?, provenance_process = ?,
                record_version = record_version + 1
            WHERE id = ? AND record_version = ?
          `,
        )
        .run(
          progressNotes,
          nowMs,
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_stream_entry_ids ?? null,
          storedProvenance.provenance_process,
          goalId,
          expectedVersion,
        );

      if (result.changes === 0) {
        assertIdentityCasUpdated({
          result,
          recordType: "goal",
          recordId: goalId,
          expectedVersion,
        });
      }

      recordIdentityEvent(this.identityEventRepository, {
        record_type: "goal",
        record_id: goalId,
        action: "update_progress",
        old_value: current,
        new_value: {
          ...current,
          record_version: nextRecordVersion(expectedVersion),
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
      expectedVersion?: number;
    } = {},
  ): GoalRecord {
    const current = this.get(goalId);

    if (current === null) {
      throw new StorageError(`Unknown goal id: ${goalId}`, {
        code: "GOAL_NOT_FOUND",
      });
    }

    const parsedPatch = goalPatchSchema.parse(patch);
    const expectedVersion = expectedRecordVersion(current, options);
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
      record_version: nextRecordVersion(expectedVersion),
      progress_notes: nextProgressNotes,
      last_progress_ts: nextLastProgressTs,
      provenance: parsedPatch.provenance ?? current.provenance,
    });
    const storedProvenance = toStoredProvenance(next.provenance);

    this.runGoalWrite(() => {
      const result = this.db
        .prepare(
          `
            UPDATE goals
            SET description = ?, terminal_condition = ?, priority = ?, parent_goal_id = ?,
                status = ?, progress_notes = ?, last_progress_ts = ?, target_at = ?,
                audience_entity_id = ?, owner_entity_id = ?, source_stream_entry_ids = ?,
                canonicalized_by_artifact_entry_id = ?,
                provenance_kind = ?, provenance_episode_ids = ?, provenance_stream_entry_ids = ?,
                provenance_process = ?,
                record_version = record_version + 1
            WHERE id = ? AND record_version = ?
          `,
        )
        .run(
          next.description,
          next.terminal_condition,
          next.priority,
          next.parent_goal_id,
          next.status,
          next.progress_notes,
          next.last_progress_ts,
          next.target_at,
          next.audience_entity_id,
          next.owner_entity_id,
          next.source_stream_entry_ids === undefined
            ? null
            : serializeJsonValue(next.source_stream_entry_ids),
          next.canonicalized_by_artifact_entry_id ?? null,
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_stream_entry_ids ?? null,
          storedProvenance.provenance_process,
          goalId,
          expectedVersion,
        );

      assertIdentityCasUpdated({
        result,
        recordType: "goal",
        recordId: goalId,
        expectedVersion,
      });

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
            SET description = ?, terminal_condition = ?, priority = ?, parent_goal_id = ?,
                status = ?, progress_notes = ?, last_progress_ts = ?, created_at = ?,
                target_at = ?, audience_entity_id = ?, owner_entity_id = ?, source_stream_entry_ids = ?,
                canonicalized_by_artifact_entry_id = ?, provenance_kind = ?, provenance_episode_ids = ?,
                provenance_stream_entry_ids = ?, provenance_process = ?
            WHERE id = ?
          `,
        )
        .run(
          parsed.description,
          parsed.terminal_condition,
          parsed.priority,
          parsed.parent_goal_id,
          parsed.status,
          parsed.progress_notes,
          parsed.last_progress_ts,
          parsed.created_at,
          parsed.target_at,
          parsed.audience_entity_id,
          parsed.owner_entity_id,
          parsed.source_stream_entry_ids === undefined
            ? null
            : serializeJsonValue(parsed.source_stream_entry_ids),
          parsed.canonicalized_by_artifact_entry_id ?? null,
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_stream_entry_ids ?? null,
          storedProvenance.provenance_process,
          parsed.id,
        );
    });

    return parsed;
  }

  remove(goalId: GoalId, options: IdentityCasOptions = {}): boolean {
    const current = this.get(goalId);

    if (current === null) {
      if (options.expectedVersion !== undefined) {
        assertIdentityCasUpdated({
          result: { changes: 0 },
          recordType: "goal",
          recordId: goalId,
          expectedVersion: options.expectedVersion,
        });
      }

      return false;
    }

    const expectedVersion = expectedRecordVersion(current, options);

    return this.runGoalWrite(() => {
      const result = this.db
        .prepare("DELETE FROM goals WHERE id = ? AND record_version = ?")
        .run(goalId, expectedVersion);
      assertIdentityCasUpdated({
        result,
        recordType: "goal",
        recordId: goalId,
        expectedVersion,
      });

      const reparent = this.db
        .prepare("UPDATE goals SET parent_goal_id = NULL WHERE parent_goal_id = ?")
        .run(goalId);
      void reparent;

      return result.changes > 0;
    });
  }
}
