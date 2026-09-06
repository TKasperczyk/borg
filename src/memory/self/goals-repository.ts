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
import { artifactReferenceExists } from "../common/artifact-reference-validation.js";
import {
  currentGoalBlock,
  goalBlockInputSchema,
  type GoalBlockInput,
  type GoalBlockRecord,
} from "./goal-blocks.js";

import { recordIdentityEvent } from "./shared/identity-events.js";
import { requireProvenance } from "./shared/provenance.js";
import { mapGoalRow } from "./shared/sql-mapping.js";
import {
  goalAudienceEntityIdSchema,
  goalCounterpartyEntityIdSchema,
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
  statuses?: readonly GoalStatus[];
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
  progress_notes, last_progress_ts, created_at, target_at, block_history_json, audience_entity_id, owner_entity_id,
  counterparty_entity_id, source_stream_entry_ids, canonicalized_by_artifact_entry_id, provenance_kind,
  provenance_episode_ids, provenance_stream_entry_ids, provenance_process
`;

export type GoalStatusUpdateOptions = IdentityCasOptions & {
  canonicalizedByArtifactEntryId?: SharedStateEntryId | null;
};

export type GoalRemovalAuditContext = {
  reason: string;
  provenance: Provenance;
};

export type GoalRemovalOptions = IdentityCasOptions & {
  auditContext: GoalRemovalAuditContext | null;
};

export const GOAL_TURN_ROLLBACK_REASON =
  "turn rollback: reverted goal mutations from an aborted turn";

export type GoalRetirementResult =
  | {
      status: "applied";
      goal: GoalRecord;
    }
  | {
      status: "no_op";
      reason: "missing";
      goal: null;
    }
  | {
      status: "no_op";
      reason: "not_active";
      goal: GoalRecord;
    };

export class GoalsRepository {
  private readonly clock: Clock;
  private reconcilingBlocks = false;

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
    if (
      (current.status !== "active" && current.status !== "blocked") ||
      (nextStatus !== "done" && nextStatus !== "abandoned")
    ) {
      return;
    }

    const block = currentGoalBlock(current);
    if (block !== null) {
      const history = (current.block_history ?? []).map((entry) =>
        entry.unblocked_at === null
          ? {
              ...entry,
              unblocked_at: this.clock.now(),
              unblock_reason: `goal closed as ${nextStatus}`,
            }
          : entry,
      );
      this.db
        .prepare("UPDATE goals SET block_history_json = ? WHERE id = ?")
        .run(serializeJsonValue(history), current.id);
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
    counterpartyEntityId?: EntityId | null;
    sourceStreamEntryIds?: readonly StreamEntryId[];
  }): GoalRecord {
    if (input.status === "blocked") {
      throw new StorageError(
        "Use goals.block with a named blocker and an attempted-unavailable declaration",
        { code: "GOAL_BLOCKER_REQUIRED" },
      );
    }
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
      counterparty_entity_id:
        input.counterpartyEntityId === undefined
          ? null
          : goalCounterpartyEntityIdSchema.nullable().parse(input.counterpartyEntityId),
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
              counterparty_entity_id, source_stream_entry_ids, canonicalized_by_artifact_entry_id, provenance_kind,
              provenance_episode_ids, provenance_stream_entry_ids, provenance_process
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
          goal.counterparty_entity_id,
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
    if (options.statuses !== undefined) {
      filters.push(`status IN (${options.statuses.map(() => "?").join(",") || "NULL"})`);
      values.push(...options.statuses.map((status) => goalStatusSchema.parse(status)));
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
          WITH scheduling AS (
            SELECT *,
              COALESCE(last_progress_ts, created_at) + COALESCE((
                SELECT SUM(MAX(0, json_extract(value, '$.unblocked_at') - MAX(COALESCE(last_progress_ts, created_at), json_extract(value, '$.blocked_at'))))
                FROM json_each(block_history_json) WHERE json_extract(value, '$.unblocked_at') IS NOT NULL
              ), 0) AS scheduling_progress_anchor,
              target_at + COALESCE((
                SELECT SUM(MAX(0, json_extract(value, '$.unblocked_at') - json_extract(value, '$.blocked_at')))
                FROM json_each(block_history_json) WHERE json_extract(value, '$.unblocked_at') IS NOT NULL
              ), 0) AS scheduling_target_at
            FROM goals WHERE status = 'active'
          )
          SELECT ${GOAL_SELECT_COLUMNS},
            CASE
              WHEN scheduling_target_at IS NULL THEN scheduling_progress_anchor + ? + 1
              WHEN scheduling_target_at - ? + 1 < scheduling_progress_anchor + ? + 1
                THEN scheduling_target_at - ? + 1
              ELSE scheduling_progress_anchor + ? + 1
            END AS autonomy_due_at
          FROM scheduling
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

  retire(goalId: GoalId, reason: string, provenance: Provenance): GoalRetirementResult {
    return this.runGoalWrite(() => {
      const current = this.get(goalId);

      if (current === null) {
        return {
          status: "no_op",
          reason: "missing",
          goal: null,
        };
      }

      if (current.status !== "active") {
        return {
          status: "no_op",
          reason: "not_active",
          goal: current,
        };
      }

      const note = `[${this.clock.now()}] ${reason}`;
      const progressNotes =
        current.progress_notes === null ? note : `${current.progress_notes}\n${note}`;

      return {
        status: "applied",
        goal: this.update(
          goalId,
          {
            status: "abandoned",
            progress_notes: progressNotes,
            provenance,
          },
          provenance,
          {
            reason,
            expectedVersion: expectedRecordVersion(current),
          },
        ),
      };
    });
  }

  block(goalId: GoalId, input: GoalBlockInput, provenance: Provenance): GoalRecord {
    const parsed = goalBlockInputSchema.parse(input);
    return this.runGoalWrite(() => {
      const goal = this.get(goalId);
      if (goal?.status !== "active") {
        throw new StorageError("Only an active goal may be blocked", { code: "GOAL_NOT_ACTIVE" });
      }
      const blocker = parsed.blocker;
      if (
        blocker.kind === "goal" &&
        (blocker.goal_id === goalId || this.get(blocker.goal_id) === null)
      ) {
        throw new StorageError("Blocker must name another existing goal", {
          code: "GOAL_BLOCKER_INVALID",
        });
      }
      if (
        blocker.kind === "entity" &&
        this.db.prepare("SELECT 1 FROM entities WHERE id = ?").get(blocker.entity_id) === undefined
      ) {
        throw new StorageError("Blocker entity does not exist", { code: "GOAL_BLOCKER_INVALID" });
      }
      const nowMs = this.clock.now();
      if (
        parsed.attempt_evidence !== undefined &&
        !artifactReferenceExists(this.db, parsed.attempt_evidence, nowMs)
      ) {
        throw new StorageError(
          "Attempt evidence must name an existing artifact from this or an earlier turn",
          { code: "GOAL_ATTEMPT_EVIDENCE_INVALID" },
        );
      }
      this.writeBlockTransition(
        goal,
        "blocked",
        [
          ...(goal.block_history ?? []),
          {
            ...parsed,
            blocked_at: nowMs,
            unblocked_at: null,
            unblock_reason: null,
          },
        ],
        parsed.reason,
        provenance,
        "block",
      );
      this.reconcileBlocks();
      return this.get(goalId)!;
    });
  }

  unblock(
    goalId: GoalId,
    reason: string,
    provenance: Provenance,
    effectiveAt = this.clock.now(),
  ): GoalRecord {
    const parsedReason = z.string().trim().min(1).parse(reason);
    return this.runGoalWrite(() => {
      const goal = this.get(goalId);
      if (goal === null) throw new StorageError("Unknown goal", { code: "GOAL_NOT_FOUND" });
      if (goal.status !== "blocked") return goal;
      const history = (goal.block_history ?? []).map((block) =>
        block.unblocked_at === null
          ? {
              ...block,
              unblocked_at: Math.max(block.blocked_at, effectiveAt),
              unblock_reason: parsedReason,
            }
          : block,
      );
      return this.writeBlockTransition(
        goal,
        "active",
        history,
        parsedReason,
        provenance,
        "unblock",
      );
    });
  }

  private writeBlockTransition(
    goal: GoalRecord,
    status: GoalStatus,
    history: GoalBlockRecord[],
    reason: string,
    provenance: Provenance,
    action: string,
  ): GoalRecord {
    const parsedProvenance = requireProvenance(provenance, "Goal block transition");
    const expectedVersion = expectedRecordVersion(goal);
    const result = this.db
      .prepare(
        "UPDATE goals SET status = ?, block_history_json = ?, record_version = record_version + 1 WHERE id = ? AND record_version = ?",
      )
      .run(status, serializeJsonValue(history), goal.id, expectedVersion);
    assertIdentityCasUpdated({ result, recordType: "goal", recordId: goal.id, expectedVersion });
    const next = this.get(goal.id)!;
    recordIdentityEvent(this.identityEventRepository, {
      record_type: "goal",
      record_id: goal.id,
      action,
      old_value: goal,
      new_value: next,
      reason,
      provenance: parsedProvenance,
    });
    return next;
  }

  /** Replays durable structural facts on startup, stream append, and scheduler/turn entry. */
  reconcileBlocks(): void {
    if (this.reconcilingBlocks) return;
    this.reconcilingBlocks = true;
    try {
      this.runGoalWrite(() => {
        const rows = this.db
          .prepare(`SELECT ${GOAL_SELECT_COLUMNS} FROM goals WHERE status = 'blocked'`)
          .all() as Record<string, unknown>[];
        for (const row of rows) {
          const goal = mapGoalRow(row);
          const block = currentGoalBlock(goal);
          if (block === null) continue; // Historical unnamed blocks are repaired by the identity migration.
          const blocker = block.blocker;
          const nowMs = this.clock.now();
          if (blocker.kind === "until" && nowMs >= blocker.until) {
            this.unblock(
              goal.id,
              `until timestamp ${blocker.until} passed; observed at ${nowMs}`,
              { kind: "system" },
              blocker.until,
            );
          } else if (blocker.kind === "goal") {
            const dependency = this.get(blocker.goal_id);
            if (dependency?.status === "done" || dependency?.status === "abandoned") {
              const event =
                this.identityEventRepository === undefined
                  ? undefined
                  : (this.db
                      .prepare(
                        "SELECT id, ts FROM identity_events WHERE record_type = 'goal' AND record_id = ? AND json_extract(new_value_json, '$.status') IN ('done', 'abandoned') AND COALESCE(json_extract(old_value_json, '$.status'), '') NOT IN ('done', 'abandoned') ORDER BY id DESC LIMIT 1",
                      )
                      .get(dependency.id) as { id: number; ts: number } | undefined);
              this.unblock(
                goal.id,
                `blocker goal ${dependency.id} is ${dependency.status}; identity event ${event?.id ?? "unrecorded"} at ${event?.ts ?? nowMs}; observed at ${nowMs}`,
                { kind: "system" },
                event?.ts ?? nowMs,
              );
            }
          } else if (blocker.kind === "entity") {
            const entry = this.db
              .prepare(
                `SELECT entry_id, timestamp FROM stream_entry_index WHERE sender_entity_id = ? AND timestamp > ? AND timestamp <= ? AND active = 1 AND receipt_pending = 0 AND kind IN ('user_msg', 'user_image_attachment', 'agent_observed') ORDER BY timestamp, byte_offset LIMIT 1`,
              )
              .get(blocker.entity_id, block.blocked_at, nowMs) as
              | { entry_id: string; timestamp: number }
              | undefined;
            if (entry !== undefined) {
              this.unblock(
                goal.id,
                `inbound stream entry ${entry.entry_id} from entity ${blocker.entity_id} at ${entry.timestamp}; observed at ${nowMs}`,
                { kind: "system" },
                entry.timestamp,
              );
            }
          }
        }
      });
    } finally {
      this.reconcilingBlocks = false;
    }
  }

  private assertStatusTransition(current: GoalRecord, status: GoalStatus): void {
    if (status === "blocked" && current.status !== "blocked") {
      throw new StorageError("Use goals.block with a named blocker", {
        code: "GOAL_BLOCKER_REQUIRED",
      });
    }
    if (current.status === "blocked" && status === "active") {
      throw new StorageError("Use goals.unblock with a reason", {
        code: "GOAL_UNBLOCK_REASON_REQUIRED",
      });
    }
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
    this.assertStatusTransition(current, parsedStatus);
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
        new_value: this.get(goalId)!,
        provenance: parsedProvenance,
      });
      this.reconcileBlocks();
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
    this.assertStatusTransition(current, parsedPatch.status ?? current.status);
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
                audience_entity_id = ?, owner_entity_id = ?, counterparty_entity_id = ?,
                source_stream_entry_ids = ?,
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
          next.counterparty_entity_id,
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
        new_value: this.get(goalId)!,
        reason: options.reason ?? null,
        provenance: parsedProvenance,
        review_item_id: options.reviewItemId ?? null,
        overwrite_without_review: options.overwriteWithoutReview === true,
      });
      this.reconcileBlocks();
    });

    return this.get(goalId)!;
  }

  restore(goal: GoalRecord): GoalRecord {
    const parsed = goalSchema.parse(goal);
    const storedProvenance = toStoredProvenance(parsed.provenance);

    return this.runGoalWrite(() => {
      const current = this.get(parsed.id);
      this.db
        .prepare(
          `
            UPDATE goals
            SET description = ?, terminal_condition = ?, priority = ?, parent_goal_id = ?,
                status = ?, progress_notes = ?, last_progress_ts = ?, created_at = ?,
                target_at = ?, audience_entity_id = ?, owner_entity_id = ?,
                counterparty_entity_id = ?, source_stream_entry_ids = ?,
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
          parsed.counterparty_entity_id ?? null,
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

      this.db
        .prepare("UPDATE goals SET block_history_json = ? WHERE id = ?")
        .run(serializeJsonValue(parsed.block_history ?? []), parsed.id);
      const restored = this.get(parsed.id);

      if (current === null || restored === null) {
        return parsed;
      }

      recordIdentityEvent(this.identityEventRepository, {
        record_type: "goal",
        record_id: restored.id,
        action: "update",
        old_value: current,
        new_value: restored,
        reason: GOAL_TURN_ROLLBACK_REASON,
        provenance: restored.provenance,
      });

      return restored;
    });
  }

  remove(goalId: GoalId, options: GoalRemovalOptions): boolean {
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
    const auditContext =
      options.auditContext === null
        ? null
        : {
            reason: options.auditContext.reason,
            provenance: requireProvenance(
              options.auditContext.provenance,
              "Goal removal audit context",
            ),
          };

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

      if (auditContext !== null) {
        recordIdentityEvent(this.identityEventRepository, {
          record_type: "goal",
          record_id: goalId,
          action: "delete",
          old_value: current,
          new_value: null,
          reason: auditContext.reason,
          provenance: auditContext.provenance,
        });
      }

      return result.changes > 0;
    });
  }
}
