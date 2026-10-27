import { describe, expect, it, vi } from "vitest";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../../tracing/tracer.js";
import { TurnLifecycleTracker } from "./turn-lifecycle-tracker.js";
import {
  createWorkingMemory,
  pendingActionResolutionId,
  WorkingMemoryStore,
  type WorkingMemory,
} from "../../memory/working/index.js";
import {
  ActionRepository,
  actionMigrations,
  actionRecordSchema,
} from "../../memory/actions/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { SessionLock } from "../session-lock.js";
import { GOAL_TURN_ROLLBACK_REASON, type GoalRemovalOptions } from "../../memory/self/index.js";
import {
  createActionId,
  createExecutiveStepId,
  createGoalId,
  createSessionId,
  createStreamEntryId,
  type ActionId,
  type ExecutiveStepId,
  type GoalId,
} from "../../util/ids.js";

function makeTracker(input: {
  tracer?: TurnTracer;
  saveWorkingMemory?: WorkingMemoryStore["save"];
  loadWorkingMemory?: WorkingMemoryStore["load"];
  deleteAction?: (id: ActionId) => Promise<boolean>;
  deleteStep?: (id: ExecutiveStepId) => boolean;
  removeGoal?: (id: GoalId, options: GoalRemovalOptions) => boolean;
}) {
  return new TurnLifecycleTracker({
    workingMemoryStore: {
      recordPendingActionMerges: vi.fn(),
      save: input.saveWorkingMemory ?? vi.fn((workingMemory) => workingMemory),
      load: input.loadWorkingMemory ?? vi.fn((sessionId) => createWorkingMemory(sessionId, 100)),
    },
    actionRepository: {
      delete: input.deleteAction ?? vi.fn(async () => true),
    },
    executiveStepsRepository: {
      delete: input.deleteStep ?? vi.fn(() => true),
      restore: vi.fn(),
    },
    goalsRepository: {
      remove: input.removeGoal ?? vi.fn(() => true),
      restore: vi.fn(),
    },
    openQuestionsRepository: {
      delete: vi.fn(),
      restore: vi.fn(),
    },
    episodicRepository: {
      updateStats: vi.fn(),
    },
    relationalSlotRepository: {
      restore: vi.fn(),
    },
    tracer: input.tracer,
  });
}

describe("TurnLifecycleTracker", () => {
  it.each([
    "happy",
    "before_delete",
    "after_delete",
    "first_rollback_write",
    "before_insert",
    "before_commit",
    "happy_commit",
    "commit_cleanup",
    "commit_write_before_persist",
    "commit_write_after_persist",
  ])(
    "retains the intention until durable commit and recovers %s after restart",
    async (crashPoint) => {
      const dataDir = mkdtempSync(join(tmpdir(), "borg-local-rollback-"));
      const databasePath = join(dataDir, "actions.db");
      let db = openDatabase(databasePath, { migrations: actionMigrations });
      try {
        const sessionId = createSessionId();
        const sessionLock = new SessionLock({ dataDir });
        const actionRepository = new ActionRepository({ db });
        const emit = vi.fn();
        const store = new WorkingMemoryStore({ dataDir, actionRepository, tracer: { emit } });
        const original = store.save({
          ...createWorkingMemory(sessionId, 100),
          pending_actions: [
            { description: "確認を完了", next_action: null },
            { description: "earlier foreign completion", next_action: null },
          ],
        });
        const pending = original.pending_actions[0]!;
        const ownAction = pendingActionResolutionId({
          session_id: sessionId,
          pending_action_id: pending.pending_action_id!,
        });
        const foreignAction = pendingActionResolutionId({
          session_id: sessionId,
          pending_action_id: original.pending_actions[1]!.pending_action_id!,
        });
        for (const id of [ownAction, foreignAction]) {
          actionRepository.add(
            actionRecordSchema.parse({
              id,
              description: "確認を完了",
              actor: "borg",
              audience_entity_id: null,
              state: "completed",
              confidence: 1,
              provenance_episode_ids: [],
              provenance_stream_entry_ids: [createStreamEntryId()],
              created_at: 200,
              updated_at: 200,
              considering_at: null,
              committed_at: null,
              scheduled_at: null,
              completed_at: 200,
              not_done_at: null,
              unknown_at: null,
            }),
          );
        }
        const completion = actionRepository.get(ownAction)!;
        await actionRepository.delete(ownAction);
        const marker = { action_id: ownAction, turn_id: "aborted-wake" };
        const completingMarker = { ...marker, phase: "provisional" as const };
        const commits = ["happy_commit", "commit_cleanup", "commit_write_after_persist"].includes(
          crashPoint,
        );
        const lease = await sessionLock.acquire(sessionId);
        const saveWorkingMemory = vi.fn<WorkingMemoryStore["save"]>((...args) =>
          store.save(...args),
        );
        if (crashPoint === "after_delete" || crashPoint === "commit_cleanup") {
          saveWorkingMemory.mockImplementationOnce((...args) => store.save(...args));
          saveWorkingMemory.mockImplementationOnce(() => {
            throw new Error("crash before marker cleanup");
          });
        }
        if (crashPoint === "first_rollback_write")
          saveWorkingMemory.mockImplementationOnce(() => {
            throw new Error("first rollback write failed before persistence");
          });
        if (crashPoint === "commit_write_before_persist")
          saveWorkingMemory.mockImplementationOnce(() => {
            throw new Error("commit write failed before persistence");
          });
        if (crashPoint === "commit_write_after_persist")
          saveWorkingMemory.mockImplementationOnce((...args) => {
            store.save(...args);
            throw new Error("commit saved before callback failed");
          });
        const deleteAction = vi.fn(async (id: ActionId) => {
          expect(sessionLock.isHeld(sessionId)).toBe(true);
          expect(store.load(sessionId).pending_actions).toEqual([
            { ...pending, restored_from_action: marker },
          ]);
          if (crashPoint === "before_delete") throw new Error("crash before delete");
          return actionRepository.delete(id);
        });
        const tracker = makeTracker({
          saveWorkingMemory,
          loadWorkingMemory: store.load.bind(store),
          deleteAction,
        });
        tracker.captureInitialWorkingMemory(original);
        tracker.trackReflectionEffects({
          createdActionIds: crashPoint === "before_insert" ? [] : [ownAction],
          committedCrossSessionActionIds: [foreignAction],
          createdExecutiveStepIds: [],
          createdOpenQuestionIds: [],
          updatedExecutiveSteps: [],
          updatedGoals: [],
          retiredGoalIds: [],
          resolvedOpenQuestions: [],
          updatedEpisodeStats: [],
        });
        try {
          // The real local forward path must persist the marker before insertion.
          expect(
            await store.resolvePendingAction({
              sessionId,
              turnId: "aborted-wake",
              pendingActionId: pending.pending_action_id!,
              lockedSessionId: sessionId,
              sessionLock,
              persist: false,
              apply: async () => {
                expect(store.load(sessionId).pending_actions).toEqual([
                  { ...pending, completing_action: completingMarker },
                ]);
                expect(actionRepository.get(ownAction)).toBeNull();
                if (crashPoint === "before_insert")
                  throw new Error("crash after marker before insert");
                actionRepository.add(completion);
                return true;
              },
            }),
          ).toBe(crashPoint === "before_insert" ? "failed" : "applied");
          // Even a stale pre-reflection snapshot cannot remove the protected row.
          store.save({ ...original, pending_actions: [] });
          expect(store.load(sessionId).pending_actions).toEqual([
            { ...pending, completing_action: completingMarker },
          ]);
          if (commits) {
            tracker.commitTurnState({ turnId: "aborted-wake", sessionId });
            expect(
              await tracker.cleanupAbortedTurnState({ turnId: "aborted-wake", sessionId }),
            ).toEqual([]);
            expect(deleteAction).not.toHaveBeenCalled();
          } else if (crashPoint !== "before_commit" && crashPoint !== "before_insert") {
            if (crashPoint === "commit_write_before_persist") {
              expect(() => tracker.commitTurnState({ turnId: "aborted-wake", sessionId })).toThrow(
                "commit write failed",
              );
            }
            const failures = await tracker.cleanupAbortedTurnState({
              turnId: "aborted-wake",
              sessionId,
            });
            expect(failures).toHaveLength(
              ["happy", "commit_write_before_persist"].includes(crashPoint) ? 0 : 1,
            );
            if (crashPoint === "first_rollback_write") expect(deleteAction).not.toHaveBeenCalled();
            else expect(deleteAction).toHaveBeenCalledExactlyOnceWith(ownAction);
          }
          expect(deleteAction).not.toHaveBeenCalledWith(foreignAction);
          expect(actionRepository.get(foreignAction)).not.toBeNull();
          expect(actionRepository.get(ownAction) !== null).toBe(
            commits ||
              ["before_delete", "first_rollback_write", "before_commit"].includes(crashPoint),
          );
        } finally {
          await lease?.release();
        }
        db.close();
        db = openDatabase(databasePath, { migrations: actionMigrations });
        const reopenedActions = new ActionRepository({ db });
        const restarted = new WorkingMemoryStore({
          dataDir,
          actionRepository: reopenedActions,
          tracer: { emit },
        });
        const path = join(dataDir, "working", `${sessionId}.json`);
        const beforeProjection = readFileSync(path, "utf8");
        const projection = await restarted.projectPendingActions({
          current: createWorkingMemory(createSessionId(), 300),
          sessionLock,
          includeOtherSessions: true,
        });
        expect(projection.actions).toHaveLength(commits ? 0 : 1);
        if (!commits) {
          expect(projection.actions[0]).toMatchObject({
            session_id: sessionId,
            pending_action_id: pending.pending_action_id,
            disclosure_label: { disclosure_class: "self_private" },
          });
          expect(projection.actions[0]?.restored_from_action).toEqual(
            ["before_delete", "after_delete"].includes(crashPoint) ? marker : undefined,
          );
          if (["first_rollback_write", "before_insert", "before_commit"].includes(crashPoint)) {
            expect(projection.actions[0]).toMatchObject({
              completing_action: completingMarker,
              completion_status: "completion_pending",
            });
          }
        }
        expect(projection.completed_pending_cleanup).toHaveLength(
          crashPoint === "commit_cleanup" ? 1 : 0,
        );
        expect(readFileSync(path, "utf8")).toBe(beforeProjection);
        const recoveryLease = await sessionLock.acquire(sessionId);
        try {
          // The next conversational turn reuses its lease for locked migration
          // and rollback recovery before taking a snapshot or writing new state.
          await restarted.migratePendingActionIds(sessionId, sessionLock, {
            lockedSessionId: sessionId,
            turnId: "next-turn",
          });
          const latest = restarted.load(sessionId);
          expect(latest.pending_actions).toEqual(commits ? [] : [pending]);
          restarted.save({ ...latest, hot_entities: ["new origin state"] });
          expect(reopenedActions.get(ownAction) === null).toBe(!commits);
          expect(reopenedActions.get(foreignAction)).not.toBeNull();
        } finally {
          await recoveryLease?.release();
        }
        const apply = vi.fn(async () => {
          reopenedActions.add(completion);
          return true;
        });
        const resolution = {
          sessionId,
          pendingActionId: pending.pending_action_id!,
          lockedSessionId: createSessionId(),
          sessionLock,
          apply,
          turnId: "later-wake",
        };
        expect(await restarted.resolvePendingAction(resolution)).toBe("applied");
        expect(await restarted.resolvePendingAction(resolution)).toBe("applied");
        expect(apply).toHaveBeenCalledTimes(commits ? 0 : 1);
        expect(reopenedActions.list({ state: "completed" })).toHaveLength(2);
        expect(restarted.load(sessionId).pending_actions).toEqual([]);
        expect(emit).toHaveBeenCalledWith(
          "working_memory.pending_action_resolved",
          expect.objectContaining({
            action_id: ownAction,
            step: commits
              ? "completion_commit"
              : crashPoint === "before_insert"
                ? "completion_pending"
                : "rollback_restore",
            outcome: "saved",
          }),
        );
        expect(emit).toHaveBeenCalledWith(
          "working_memory.pending_action_resolved",
          expect.objectContaining({
            action_id: ownAction,
            step: commits ? "cleanup" : "rollback_cleanup",
            outcome: "saved",
          }),
        );
      } finally {
        db.close();
        rmSync(dataDir, { recursive: true, force: true });
      }
    },
  );

  it("supplies explicit rollback audit context when removing a turn-created goal", async () => {
    const goalId = createGoalId();
    const removeGoal = vi.fn<(id: GoalId, options: GoalRemovalOptions) => boolean>(() => true);
    const tracker = makeTracker({ removeGoal });
    tracker.trackCreatedGoalIds([goalId]);

    await tracker.cleanupAbortedTurnState({ turnId: "turn_goal_rollback" });

    expect(removeGoal).toHaveBeenCalledWith(goalId, {
      auditContext: {
        reason: GOAL_TURN_ROLLBACK_REASON,
        provenance: { kind: "system" },
      },
    });
  });

  it("traces incomplete abort cleanup without throwing away best-effort rollback", async () => {
    const events: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: (event, data) => events.push({ event, data }),
    };
    const actionId = createActionId();
    const stepId = createExecutiveStepId();
    const sessionId = createSessionId();
    const deleteStep = vi.fn<(id: ExecutiveStepId) => boolean>(() => true);
    const tracker = makeTracker({
      tracer,
      deleteAction: async () => {
        throw new Error("delete action failed");
      },
      deleteStep,
    });

    tracker.trackCreatedActionIds([actionId]);
    tracker.trackCreatedExecutiveStepIds([stepId]);

    await tracker.cleanupAbortedTurnState({
      turnId: "turn_cleanup_trace",
      sessionId,
    });

    expect(deleteStep).toHaveBeenCalledWith(stepId);
    expect(events).toEqual([
      {
        event: "turn.rollback_incomplete",
        data: {
          turnId: "turn_cleanup_trace",
          turn_id: "turn_cleanup_trace",
          session_id: sessionId,
          failure_count: 1,
          failures: [
            {
              operation: "delete_action",
              id: actionId,
              error: "Error: delete action failed",
            },
          ],
        },
      },
    ]);
  });

  it("captures working-memory restore failures and continues cleanup", async () => {
    const events: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: (event, data) => events.push({ event, data }),
    };
    const sessionId = createSessionId();
    const stepId = createExecutiveStepId();
    const deleteStep = vi.fn<(id: ExecutiveStepId) => boolean>(() => true);
    const deleteAction = vi.fn(async () => true);
    const tracker = makeTracker({
      tracer,
      saveWorkingMemory: () => {
        throw new Error("working memory save failed");
      },
      deleteAction,
      deleteStep,
    });

    tracker.captureInitialWorkingMemory(createWorkingMemory(sessionId, 100));
    tracker.trackCreatedActionIds([createActionId()]);
    tracker.trackCreatedExecutiveStepIds([stepId]);

    const failures = await tracker.cleanupAbortedTurnState({
      turnId: "turn_working_memory_cleanup",
      sessionId,
    });

    expect(deleteStep).toHaveBeenCalledWith(stepId);
    expect(deleteAction).not.toHaveBeenCalled();
    expect(failures).toEqual([
      {
        operation: "restore_working_memory",
        id: sessionId,
        error: "Error: working memory save failed",
      },
    ]);
    expect(events[0]?.data).toMatchObject({
      failure_count: 1,
      failures,
    });
  });

  it("does not let rollback-incomplete tracing mask cleanup completion", async () => {
    const actionId = createActionId();
    const tracker = makeTracker({
      tracer: {
        enabled: true,
        includePayloads: false,
        emit: () => {
          throw new Error("trace write failed");
        },
      },
      deleteAction: async () => {
        throw new Error("delete action failed");
      },
    });

    tracker.trackCreatedActionIds([actionId]);

    await expect(
      tracker.cleanupAbortedTurnState({
        turnId: "turn_trace_failure",
        sessionId: createSessionId(),
      }),
    ).resolves.toEqual([
      {
        operation: "delete_action",
        id: actionId,
        error: "Error: delete action failed",
      },
    ]);
  });
});
