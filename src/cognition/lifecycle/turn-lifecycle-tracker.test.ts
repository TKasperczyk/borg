import { describe, expect, it, vi } from "vitest";

import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../../tracing/tracer.js";
import { TurnLifecycleTracker } from "./turn-lifecycle-tracker.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import {
  createActionId,
  createExecutiveStepId,
  createSessionId,
  type ActionId,
  type ExecutiveStepId,
} from "../../util/ids.js";

function makeTracker(input: {
  tracer?: TurnTracer;
  saveWorkingMemory?: (workingMemory: WorkingMemory) => WorkingMemory;
  deleteAction?: (id: ActionId) => Promise<boolean>;
  deleteStep?: (id: ExecutiveStepId) => boolean;
}) {
  return new TurnLifecycleTracker({
    workingMemoryStore: {
      recordPendingActionMerges: vi.fn(),
      save: input.saveWorkingMemory ?? vi.fn((workingMemory) => workingMemory),
    },
    actionRepository: {
      delete: input.deleteAction ?? vi.fn(async () => true),
    },
    executiveStepsRepository: {
      delete: input.deleteStep ?? vi.fn(() => true),
      restore: vi.fn(),
    },
    goalsRepository: {
      remove: vi.fn(),
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
    const tracker = makeTracker({
      tracer,
      saveWorkingMemory: () => {
        throw new Error("working memory save failed");
      },
      deleteStep,
    });

    tracker.captureInitialWorkingMemory({ session_id: sessionId } as WorkingMemory);
    tracker.trackCreatedExecutiveStepIds([stepId]);

    const failures = await tracker.cleanupAbortedTurnState({
      turnId: "turn_working_memory_cleanup",
      sessionId,
    });

    expect(deleteStep).toHaveBeenCalledWith(stepId);
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
