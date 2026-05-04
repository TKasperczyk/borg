import {
  parseStoredProvenance,
  provenanceSchema,
  toStoredProvenance,
} from "../memory/common/provenance.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { ProvenanceError, StorageError } from "../util/errors.js";
import { createExecutiveStepId, type ExecutiveStepId, type GoalId } from "../util/ids.js";

import {
  executiveStepKindSchema,
  executiveStepPatchSchema,
  executiveStepSchema,
  executiveStepStatusSchema,
  type ExecutiveStep,
  type ExecutiveStepKind,
  type ExecutiveStepPatch,
  type ExecutiveStepProvenance,
  type ExecutiveStepStatus,
} from "./types.js";

export type ExecutiveStepsRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

export type ExecutiveStepAddInput = {
  id?: ExecutiveStepId;
  goalId: GoalId;
  description: string;
  kind: ExecutiveStepKind;
  status?: ExecutiveStepStatus;
  dueAt?: number | null;
  lastAttemptTs?: number | null;
  provenance: ExecutiveStepProvenance;
  createdAt?: number;
};

export type ExecutiveStepAbandonReason = "goal_closed";

const OPEN_STATUSES = new Set<ExecutiveStepStatus>(["queued", "doing"]);
const MAX_OPEN_STEPS_PER_GOAL = 3;

const VALID_TRANSITIONS: Record<ExecutiveStepStatus, ReadonlySet<ExecutiveStepStatus>> = {
  queued: new Set(["queued", "doing", "abandoned"]),
  doing: new Set(["doing", "done", "blocked", "abandoned"]),
  blocked: new Set(["blocked", "doing", "abandoned"]),
  done: new Set(["done"]),
  abandoned: new Set(["abandoned"]),
};

function requireProvenance(
  provenance: ExecutiveStepProvenance | undefined,
): ExecutiveStepProvenance {
  if (provenance === undefined) {
    throw new ProvenanceError("Executive step requires provenance", {
      code: "PROVENANCE_REQUIRED",
    });
  }

  return provenanceSchema.parse(provenance);
}

function isOpenStatus(status: ExecutiveStepStatus): boolean {
  return OPEN_STATUSES.has(status);
}

function mapExecutiveStepRow(row: Record<string, unknown>): ExecutiveStep {
  return executiveStepSchema.parse({
    id: row.id,
    goal_id: row.goal_id,
    description: row.description,
    status: row.status,
    kind: row.kind,
    due_at: row.due_at === null || row.due_at === undefined ? null : Number(row.due_at),
    last_attempt_ts:
      row.last_attempt_ts === null || row.last_attempt_ts === undefined
        ? null
        : Number(row.last_attempt_ts),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_process: row.provenance_process,
    }),
  });
}

function assertTransition(current: ExecutiveStepStatus, next: ExecutiveStepStatus): void {
  if (VALID_TRANSITIONS[current].has(next)) {
    return;
  }

  throw new StorageError(`Invalid executive step status transition: ${current} -> ${next}`, {
    code: "EXECUTIVE_STEP_INVALID_TRANSITION",
  });
}

function assertWaitStepHasDueAt(kind: ExecutiveStepKind, dueAt: number | null): void {
  if (kind !== "wait" || dueAt !== null) {
    return;
  }

  throw new StorageError("Executive wait steps require due_at", {
    code: "EXECUTIVE_STEP_WAIT_REQUIRES_DUE_AT",
  });
}

export class ExecutiveStepsRepository {
  private readonly clock: Clock;

  constructor(private readonly options: ExecutiveStepsRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  add(input: ExecutiveStepAddInput): ExecutiveStep {
    const provenance = requireProvenance(input.provenance);
    const status = executiveStepStatusSchema.parse(input.status ?? "queued");
    const kind = executiveStepKindSchema.parse(input.kind);
    const nowMs = this.clock.now();
    const createdAt = input.createdAt ?? nowMs;

    const step = executiveStepSchema.parse({
      id: input.id ?? createExecutiveStepId(),
      goal_id: input.goalId,
      description: input.description,
      status,
      kind,
      due_at: input.dueAt ?? null,
      last_attempt_ts: input.lastAttemptTs ?? null,
      created_at: createdAt,
      updated_at: createdAt,
      provenance,
    });

    assertWaitStepHasDueAt(step.kind, step.due_at);

    const storedProvenance = toStoredProvenance(step.provenance);

    this.runImmediateTransaction(() => {
      if (isOpenStatus(status)) {
        this.assertOpenStepCapacity(step.goal_id);
      }

      this.db
        .prepare(
          `
            INSERT INTO executive_steps (
              id, goal_id, description, status, kind, due_at, last_attempt_ts, created_at,
              updated_at, provenance_kind, provenance_episode_ids, provenance_process
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          `,
        )
        .run(
          step.id,
          step.goal_id,
          step.description,
          step.status,
          step.kind,
          step.due_at,
          step.last_attempt_ts,
          step.created_at,
          step.updated_at,
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
        );
    });

    return step;
  }

  get(id: ExecutiveStepId): ExecutiveStep | null {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM executive_steps
          WHERE id = ?
        `,
      )
      .get(id) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapExecutiveStepRow(row);
  }

  list(goalId: GoalId): ExecutiveStep[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM executive_steps
          WHERE goal_id = ?
          ORDER BY created_at ASC, id ASC
        `,
      )
      .all(goalId) as Record<string, unknown>[];

    return rows.map((row) => mapExecutiveStepRow(row));
  }

  listOpen(goalId: GoalId): ExecutiveStep[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM executive_steps
          WHERE goal_id = ? AND status IN ('queued', 'doing')
          ORDER BY
            CASE status WHEN 'doing' THEN 0 ELSE 1 END ASC,
            due_at IS NULL ASC,
            due_at ASC,
            created_at ASC,
            id ASC
        `,
      )
      .all(goalId) as Record<string, unknown>[];

    return rows.map((row) => mapExecutiveStepRow(row));
  }

  topOpen(goalId: GoalId): ExecutiveStep | null {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM executive_steps
          WHERE goal_id = ? AND status IN ('queued', 'doing')
          ORDER BY
            CASE status WHEN 'doing' THEN 0 ELSE 1 END ASC,
            due_at IS NULL ASC,
            due_at ASC,
            created_at ASC,
            id ASC
          LIMIT 1
        `,
      )
      .get(goalId) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapExecutiveStepRow(row);
  }

  update(id: ExecutiveStepId, patch: ExecutiveStepPatch): ExecutiveStep {
    const parsedPatch = executiveStepPatchSchema.parse(patch);

    return this.runImmediateTransaction(() => {
      const current = this.get(id);

      if (current === null) {
        throw new StorageError(`Unknown executive step id: ${id}`, {
          code: "EXECUTIVE_STEP_NOT_FOUND",
        });
      }

      const nextStatus = parsedPatch.status ?? current.status;

      assertTransition(current.status, nextStatus);

      if (!isOpenStatus(current.status) && isOpenStatus(nextStatus)) {
        this.assertOpenStepCapacity(current.goal_id);
      }

      const next = executiveStepSchema.parse({
        ...current,
        ...parsedPatch,
        updated_at: this.clock.now(),
      });

      assertWaitStepHasDueAt(next.kind, next.due_at);

      const storedProvenance = toStoredProvenance(next.provenance);

      this.db
        .prepare(
          `
            UPDATE executive_steps
            SET description = ?, status = ?, kind = ?, due_at = ?, last_attempt_ts = ?,
                updated_at = ?, provenance_kind = ?, provenance_episode_ids = ?,
                provenance_process = ?
            WHERE id = ?
          `,
        )
        .run(
          next.description,
          next.status,
          next.kind,
          next.due_at,
          next.last_attempt_ts,
          next.updated_at,
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
          id,
        );

      return next;
    });
  }

  delete(id: ExecutiveStepId): boolean {
    const result = this.db.prepare("DELETE FROM executive_steps WHERE id = ?").run(id);

    return result.changes > 0;
  }

  restore(step: ExecutiveStep): ExecutiveStep {
    const parsed = executiveStepSchema.parse(step);
    const storedProvenance = toStoredProvenance(parsed.provenance);

    this.runImmediateTransaction(() => {
      this.db
        .prepare(
          `
            UPDATE executive_steps
            SET goal_id = ?, description = ?, status = ?, kind = ?, due_at = ?,
                last_attempt_ts = ?, created_at = ?, updated_at = ?, provenance_kind = ?,
                provenance_episode_ids = ?, provenance_process = ?
            WHERE id = ?
          `,
        )
        .run(
          parsed.goal_id,
          parsed.description,
          parsed.status,
          parsed.kind,
          parsed.due_at,
          parsed.last_attempt_ts,
          parsed.created_at,
          parsed.updated_at,
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
          parsed.id,
        );
    });

    return parsed;
  }

  abandonOpenStepsForGoal(goalId: GoalId, reason: ExecutiveStepAbandonReason): ExecutiveStep[] {
    return this.runImmediateTransaction(() => {
      const openSteps = this.listOpen(goalId);

      if (openSteps.length === 0) {
        return [];
      }

      const updatedAt = this.clock.now();
      const provenance = {
        kind: "offline" as const,
        process: reason,
      };
      const storedProvenance = toStoredProvenance(provenance);

      this.db
        .prepare(
          `
            UPDATE executive_steps
            SET status = 'abandoned', updated_at = ?, provenance_kind = ?,
                provenance_episode_ids = ?, provenance_process = ?
            WHERE goal_id = ? AND status IN ('queued', 'doing')
          `,
        )
        .run(
          updatedAt,
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
          goalId,
        );

      return openSteps.map((step) => ({
        ...step,
        status: "abandoned",
        updated_at: updatedAt,
        provenance,
      }));
    });
  }

  private countOpen(goalId: GoalId): number {
    const row = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM executive_steps
          WHERE goal_id = ? AND status IN ('queued', 'doing')
        `,
      )
      .get(goalId) as { count: number } | undefined;

    return Number(row?.count ?? 0);
  }

  private assertOpenStepCapacity(goalId: GoalId): void {
    if (this.countOpen(goalId) < MAX_OPEN_STEPS_PER_GOAL) {
      return;
    }

    throw new StorageError(
      `Goal ${goalId} already has ${MAX_OPEN_STEPS_PER_GOAL} open executive steps`,
      {
        code: "EXECUTIVE_STEP_OPEN_LIMIT",
      },
    );
  }

  private runImmediateTransaction<T>(callback: () => T): T {
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
}

export type { ExecutiveStep, ExecutiveStepKind, ExecutiveStepPatch, ExecutiveStepStatus };
