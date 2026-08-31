import { describe, expect, it } from "vitest";

import { executiveMigrations, ExecutiveStepsRepository } from "../../executive/index.js";
import { GoalsRepository, selfMigrations } from "../../memory/self/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { createGoalId, DEFAULT_SESSION_ID } from "../../util/ids.js";
import { createGoalsRetireTool } from "./goals-retire.js";

const context = {
  sessionId: DEFAULT_SESSION_ID,
  origin: "autonomous" as const,
};

function createHarness() {
  const db = openDatabase(":memory:", {
    migrations: composeMigrations(selfMigrations, executiveMigrations),
  });
  const clock = new ManualClock(1_000);
  const executiveStepsRepository = new ExecutiveStepsRepository({ db, clock });
  const goalsRepository = new GoalsRepository({
    db,
    clock,
    executiveStepsRepository,
  });
  const tool = createGoalsRetireTool({ goalsRepository });

  return {
    clock,
    db,
    executiveStepsRepository,
    goalsRepository,
    tool,
  };
}

describe("tool.goals.retire", () => {
  it("retires an active goal, appends the reason, and abandons its open steps", async () => {
    const harness = createHarness();
    const description = "A".repeat(200);

    try {
      const goal = harness.goalsRepository.add({
        description,
        priority: 8,
        progressNotes: "The tracker was created.",
        provenance: { kind: "manual" },
      });
      const queued = harness.executiveStepsRepository.add({
        goalId: goal.id,
        description: "Queued work",
        kind: "think",
        provenance: { kind: "manual" },
      });
      const doing = harness.executiveStepsRepository.add({
        goalId: goal.id,
        description: "Ongoing work",
        kind: "act",
        status: "doing",
        provenance: { kind: "manual" },
      });
      const done = harness.executiveStepsRepository.add({
        goalId: goal.id,
        description: "Finished work",
        kind: "think",
        status: "done",
        provenance: { kind: "manual" },
      });
      harness.clock.advance(500);

      const result = await harness.tool.invoke(
        {
          goal_id: goal.id,
          reason: "The premise was answered by newer work.",
        },
        context,
      );

      expect(harness.tool.name).toBe("tool.goals.retire");
      expect(harness.tool.allowedOrigins).toEqual(["autonomous", "deliberator"]);
      expect(harness.tool.writeScope).toBe("write");
      expect(result).toMatchObject({
        status: "applied",
        goal: {
          id: goal.id,
          description: `${"A".repeat(157)}...`,
          status: "abandoned",
          disclosure: expect.stringContaining("disclosure_class=self_private"),
          disclosure_label: {
            disclosure_class: "self_private",
          },
        },
      });
      expect(harness.tool.outputSchema.safeParse(result).success).toBe(true);
      expect(harness.goalsRepository.get(goal.id)).toMatchObject({
        status: "abandoned",
        record_version: (goal.record_version ?? 1) + 1,
        progress_notes: "The tracker was created.\n[1500] The premise was answered by newer work.",
        last_progress_ts: 1_500,
        provenance: {
          kind: "online",
          process: "tool.goals.retire",
        },
      });
      expect(harness.executiveStepsRepository.get(queued.id)?.status).toBe("abandoned");
      expect(harness.executiveStepsRepository.get(doing.id)?.status).toBe("abandoned");
      expect(harness.executiveStepsRepository.get(done.id)?.status).toBe("done");
    } finally {
      harness.db.close();
    }
  });

  it("returns an absent no-op without throwing for an unknown goal", async () => {
    const harness = createHarness();

    try {
      const goalId = createGoalId();

      await expect(
        harness.tool.invoke(
          {
            goal_id: goalId,
            reason: "There is no tracker to retire.",
          },
          context,
        ),
      ).resolves.toMatchObject({
        status: "no_op",
        reason: "missing",
        goal: {
          id: goalId,
          description: null,
          status: "absent",
          disclosure: expect.stringContaining("disclosure_class=unknown"),
          disclosure_label: {
            disclosure_class: "unknown",
          },
        },
      });
    } finally {
      harness.db.close();
    }
  });

  it("returns the actual status and leaves an already-terminal goal unchanged", async () => {
    const harness = createHarness();

    try {
      const goal = harness.goalsRepository.add({
        description: "Already completed tracker",
        priority: 5,
        status: "done",
        progressNotes: "Completed earlier.",
        provenance: { kind: "manual" },
      });
      const before = harness.goalsRepository.get(goal.id);

      const result = await harness.tool.invoke(
        {
          goal_id: goal.id,
          reason: "This should be a no-op.",
        },
        context,
      );

      expect(result).toMatchObject({
        status: "no_op",
        reason: "not_active",
        goal: {
          id: goal.id,
          description: "Already completed tracker",
          status: "done",
          disclosure: expect.stringContaining("disclosure_class=self_private"),
          disclosure_label: {
            disclosure_class: "self_private",
          },
        },
      });
      expect(harness.goalsRepository.get(goal.id)).toEqual(before);
    } finally {
      harness.db.close();
    }
  });

  it("validates the goal id and non-empty reason at the input schema", () => {
    const harness = createHarness();

    try {
      const goalId = createGoalId();

      expect(
        harness.tool.inputSchema.safeParse({ goal_id: "not-a-goal-id", reason: "Retire it." })
          .success,
      ).toBe(false);
      expect(harness.tool.inputSchema.safeParse({ goal_id: goalId, reason: "" }).success).toBe(
        false,
      );
      expect(
        harness.tool.inputSchema.safeParse({ goal_id: goalId, reason: "Retire it.", extra: true })
          .success,
      ).toBe(false);
      expect(
        harness.tool.inputSchema.safeParse({ goal_id: goalId, reason: "Retire it." }).success,
      ).toBe(true);
    } finally {
      harness.db.close();
    }
  });
});
