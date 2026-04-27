import { describe, expect, it } from "vitest";

import { selfMigrations, GoalsRepository } from "../memory/self/index.js";
import { openDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";

import { executiveMigrations } from "./migrations.js";
import { ExecutiveStepsRepository } from "./steps-repository.js";

const manualProvenance = { kind: "manual" } as const;

function createHarness(start = 1_000) {
  const db = openDatabase(":memory:", {
    migrations: [...selfMigrations, ...executiveMigrations],
  });
  const clock = new ManualClock(start);
  const goals = new GoalsRepository({
    db,
    clock,
  });
  const steps = new ExecutiveStepsRepository({
    db,
    clock,
  });
  const goal = goals.add({
    description: "Ship durable goal steps",
    priority: 10,
    provenance: manualProvenance,
  });

  return {
    db,
    clock,
    goal,
    steps,
  };
}

describe("ExecutiveStepsRepository", () => {
  it("supports add/get/list/listOpen/update/delete CRUD", () => {
    const harness = createHarness();

    try {
      const step = harness.steps.add({
        goalId: harness.goal.id,
        description: "Sketch the repository",
        kind: "think",
        dueAt: 2_000,
        provenance: manualProvenance,
      });

      expect(step).toMatchObject({
        goal_id: harness.goal.id,
        description: "Sketch the repository",
        status: "queued",
        kind: "think",
        due_at: 2_000,
        last_attempt_ts: null,
        created_at: 1_000,
        updated_at: 1_000,
        provenance: manualProvenance,
      });
      expect(harness.steps.get(step.id)).toEqual(step);
      expect(harness.steps.list(harness.goal.id)).toEqual([step]);
      expect(harness.steps.listOpen(harness.goal.id)).toEqual([step]);

      harness.clock.advance(50);
      const updated = harness.steps.update(step.id, {
        description: "Implement the repository",
        status: "doing",
        kind: "act",
        due_at: null,
        last_attempt_ts: 1_050,
      });

      expect(updated).toMatchObject({
        description: "Implement the repository",
        status: "doing",
        kind: "act",
        due_at: null,
        last_attempt_ts: 1_050,
        created_at: 1_000,
        updated_at: 1_050,
      });
      expect(harness.steps.delete(step.id)).toBe(true);
      expect(harness.steps.get(step.id)).toBeNull();
      expect(harness.steps.list(harness.goal.id)).toEqual([]);
      expect(harness.steps.delete(step.id)).toBe(false);
    } finally {
      harness.db.close();
    }
  });

  it("enforces at most three open steps per goal", () => {
    const harness = createHarness();

    try {
      for (const description of ["One", "Two", "Three"]) {
        harness.steps.add({
          goalId: harness.goal.id,
          description,
          kind: "think",
          provenance: manualProvenance,
        });
      }

      expect(() =>
        harness.steps.add({
          goalId: harness.goal.id,
          description: "Four",
          kind: "think",
          provenance: manualProvenance,
        }),
      ).toThrow(StorageError);

      const closed = harness.steps.add({
        goalId: harness.goal.id,
        description: "Closed step does not count",
        kind: "think",
        status: "done",
        provenance: manualProvenance,
      });
      expect(closed.status).toBe("done");
    } finally {
      harness.db.close();
    }
  });

  it("enforces at most three open steps when reopening a blocked step", () => {
    const harness = createHarness();

    try {
      const blocked = harness.steps.add({
        goalId: harness.goal.id,
        description: "Blocked step",
        kind: "think",
        status: "blocked",
        provenance: manualProvenance,
      });
      for (const description of ["One", "Two", "Three"]) {
        harness.steps.add({
          goalId: harness.goal.id,
          description,
          kind: "think",
          provenance: manualProvenance,
        });
      }

      expect(() => harness.steps.update(blocked.id, { status: "doing" })).toThrow(StorageError);
      expect(harness.steps.get(blocked.id)?.status).toBe("blocked");
    } finally {
      harness.db.close();
    }
  });

  it("validates cheap status transitions", () => {
    const harness = createHarness();

    try {
      const step = harness.steps.add({
        goalId: harness.goal.id,
        description: "Queued step",
        kind: "think",
        provenance: manualProvenance,
      });

      expect(() => harness.steps.update(step.id, { status: "done" })).toThrow(StorageError);
      const doing = harness.steps.update(step.id, { status: "doing" });
      expect(doing.status).toBe("doing");
      expect(() => harness.steps.update(step.id, { status: "queued" })).toThrow(StorageError);
      expect(harness.steps.update(step.id, { status: "blocked" }).status).toBe("blocked");
      expect(harness.steps.update(step.id, { status: "doing" }).status).toBe("doing");
      expect(harness.steps.update(step.id, { status: "abandoned" }).status).toBe("abandoned");
      expect(() => harness.steps.update(step.id, { status: "doing" })).toThrow(StorageError);
    } finally {
      harness.db.close();
    }
  });

  it("orders topOpen by doing over queued, due date ascending, then created_at ascending", () => {
    const harness = createHarness();

    try {
      const queuedEarly = harness.steps.add({
        goalId: harness.goal.id,
        description: "Queued early due",
        kind: "research",
        dueAt: 2_000,
        provenance: manualProvenance,
      });
      harness.clock.advance(10);
      const queuedLate = harness.steps.add({
        goalId: harness.goal.id,
        description: "Queued late due",
        kind: "research",
        dueAt: 3_000,
        provenance: manualProvenance,
      });
      harness.clock.advance(10);
      const doingLate = harness.steps.add({
        goalId: harness.goal.id,
        description: "Doing late due",
        kind: "research",
        status: "doing",
        dueAt: 4_000,
        provenance: manualProvenance,
      });

      expect(harness.steps.topOpen(harness.goal.id)?.id).toBe(doingLate.id);
      harness.steps.update(doingLate.id, { status: "done" });
      expect(harness.steps.topOpen(harness.goal.id)?.id).toBe(queuedEarly.id);
      harness.steps.update(queuedEarly.id, { status: "doing" });
      harness.steps.update(queuedEarly.id, { status: "done" });
      expect(harness.steps.topOpen(harness.goal.id)?.id).toBe(queuedLate.id);
      harness.steps.update(queuedLate.id, { status: "doing" });
      harness.steps.update(queuedLate.id, { status: "done" });

      harness.clock.advance(10);
      const noDeadlineOlder = harness.steps.add({
        goalId: harness.goal.id,
        description: "No deadline older",
        kind: "wait",
        provenance: manualProvenance,
      });
      harness.clock.advance(10);
      const noDeadlineNewer = harness.steps.add({
        goalId: harness.goal.id,
        description: "No deadline newer",
        kind: "wait",
        provenance: manualProvenance,
      });

      expect(harness.steps.topOpen(harness.goal.id)?.id).toBe(noDeadlineOlder.id);
      expect(harness.steps.listOpen(harness.goal.id).map((step) => step.id)).toEqual([
        noDeadlineOlder.id,
        noDeadlineNewer.id,
      ]);
    } finally {
      harness.db.close();
    }
  });

  it("orders non-null due dates before older null due dates", () => {
    const harness = createHarness();

    try {
      const noDeadlineOlder = harness.steps.add({
        goalId: harness.goal.id,
        description: "No deadline older",
        kind: "wait",
        provenance: manualProvenance,
      });
      harness.clock.advance(10);
      const datedNewer = harness.steps.add({
        goalId: harness.goal.id,
        description: "Dated newer",
        kind: "wait",
        dueAt: 2_000,
        provenance: manualProvenance,
      });

      expect(harness.steps.topOpen(harness.goal.id)?.id).toBe(datedNewer.id);
      expect(harness.steps.listOpen(harness.goal.id).map((step) => step.id)).toEqual([
        datedNewer.id,
        noDeadlineOlder.id,
      ]);
    } finally {
      harness.db.close();
    }
  });

  it("returns null from topOpen when no open steps exist", () => {
    const harness = createHarness();

    try {
      expect(harness.steps.topOpen(harness.goal.id)).toBeNull();
      const step = harness.steps.add({
        goalId: harness.goal.id,
        description: "Closed only",
        kind: "think",
        status: "done",
        provenance: manualProvenance,
      });
      expect(harness.steps.topOpen(harness.goal.id)).toBeNull();
      expect(harness.steps.listOpen(harness.goal.id)).toEqual([]);
      expect(harness.steps.list(harness.goal.id)).toEqual([step]);
    } finally {
      harness.db.close();
    }
  });
});
