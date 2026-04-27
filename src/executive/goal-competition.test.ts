import { describe, expect, it } from "vitest";

import type { GoalRecord } from "../memory/self/index.js";
import { type GoalId } from "../util/ids.js";

import { selectExecutiveFocus } from "./goal-competition.js";

const nowMs = 1_000_000;
const dayMs = 24 * 60 * 60 * 1_000;

function goal(
  input: Omit<Partial<GoalRecord>, "id" | "description"> & { id: string; description: string },
): GoalRecord {
  return {
    id: input.id as GoalId,
    description: input.description,
    priority: input.priority ?? 1,
    parent_goal_id: input.parent_goal_id ?? null,
    status: input.status ?? "active",
    progress_notes: input.progress_notes ?? null,
    last_progress_ts: input.last_progress_ts ?? null,
    created_at: input.created_at ?? nowMs,
    target_at: input.target_at ?? null,
    provenance: input.provenance ?? {
      kind: "system",
    },
  };
}

function select(
  goals: readonly GoalRecord[],
  overrides: Partial<Parameters<typeof selectExecutiveFocus>[0]> = {},
) {
  return selectExecutiveFocus({
    goals,
    cognitionInput: "",
    perceptionEntities: [],
    nowMs,
    threshold: 0,
    deadlineLookaheadMs: 7 * dayMs,
    staleMs: 14 * dayMs,
    ...overrides,
  });
}

describe("selectExecutiveFocus", () => {
  it("returns empty focus for empty goal lists", () => {
    const focus = select([], {
      threshold: 0.45,
    });

    expect(focus.candidates).toEqual([]);
    expect(focus.selected_goal).toBeNull();
    expect(focus.selected_score).toBeNull();
  });

  it("ignores inactive goals", () => {
    const focus = select(
      [
        goal({
          id: "goal_aaaaaaaaaaaaaaaa",
          description: "Completed goal",
          priority: 10,
          status: "done",
          target_at: nowMs - dayMs,
          created_at: nowMs - 30 * dayMs,
        }),
      ],
      {
        threshold: 0,
      },
    );

    expect(focus.candidates).toEqual([]);
    expect(focus.selected_goal).toBeNull();
  });

  it("scores priority in isolation", () => {
    const focus = select([
      goal({ id: "goal_aaaaaaaaaaaaaaaa", description: "Low priority", priority: 1 }),
      goal({ id: "goal_bbbbbbbbbbbbbbbb", description: "High priority", priority: 10 }),
    ]);

    expect(focus.candidates[0]?.goal.description).toBe("High priority");
    expect(focus.candidates[0]?.components).toMatchObject({
      priority: 1,
      deadline_pressure: 0,
      context_fit: 0,
      progress_debt: 0,
    });
    expect(focus.candidates[0]?.score).toBeCloseTo(0.35);
  });

  it("scores deadline pressure in isolation", () => {
    const focus = select([
      goal({
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "Future deadline",
        priority: 1,
        target_at: nowMs + 7 * dayMs,
      }),
      goal({
        id: "goal_bbbbbbbbbbbbbbbb",
        description: "Overdue deadline",
        priority: 1,
        target_at: nowMs - dayMs,
      }),
    ]);

    expect(focus.candidates[0]?.goal.description).toBe("Overdue deadline");
    expect(focus.candidates[0]?.components.deadline_pressure).toBe(1);
    expect(focus.candidates[0]?.score).toBeCloseTo(0.65);
  });

  it("treats null target dates as no deadline pressure", () => {
    const focus = select([
      goal({
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "No deadline",
        priority: 1,
        target_at: null,
      }),
    ]);

    expect(focus.candidates[0]?.components.deadline_pressure).toBe(0);
  });

  it("scores context fit in isolation", () => {
    const focus = select(
      [
        goal({
          id: "goal_aaaaaaaaaaaaaaaa",
          description: "Apollo launch plan",
          priority: 1,
        }),
      ],
      {
        cognitionInput: "Work on the Apollo launch plan now.",
      },
    );

    expect(focus.candidates[0]?.components.context_fit).toBe(1);
    expect(focus.candidates[0]?.score).toBeCloseTo(0.55);
  });

  it("scores no context input as zero context fit", () => {
    const focus = select([
      goal({
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "Apollo launch plan",
        priority: 1,
      }),
    ]);

    expect(focus.candidates[0]?.components.context_fit).toBe(0);
  });

  it("scores progress debt in isolation", () => {
    const focus = select([
      goal({
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "Stale goal",
        priority: 1,
        created_at: nowMs - 30 * dayMs,
      }),
    ]);

    expect(focus.candidates[0]?.components.progress_debt).toBe(1);
    expect(focus.candidates[0]?.score).toBeCloseTo(0.5);
  });

  it("uses created_at as the progress-debt anchor when last_progress_ts is null", () => {
    const focus = select([
      goal({
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "Created long ago",
        priority: 1,
        last_progress_ts: null,
        created_at: nowMs - 30 * dayMs,
      }),
    ]);

    expect(focus.candidates[0]?.components.progress_debt).toBe(1);
  });

  it("ranks an overdue goal above a high-priority non-urgent goal when deadline and staleness compound", () => {
    const focus = select([
      goal({
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "High priority ambient goal",
        priority: 10,
      }),
      goal({
        id: "goal_bbbbbbbbbbbbbbbb",
        description: "Moderate priority overdue goal",
        priority: 4,
        created_at: nowMs - 30 * dayMs,
        target_at: nowMs - dayMs,
      }),
    ]);

    expect(focus.selected_goal?.description).toBe("Moderate priority overdue goal");
    expect(focus.selected_score?.score).toBeGreaterThan(0.35);
  });

  it("ranks a fresh high-priority goal above a stale low-priority goal", () => {
    const focus = select([
      goal({
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "Fresh high priority",
        priority: 10,
      }),
      goal({
        id: "goal_bbbbbbbbbbbbbbbb",
        description: "Stale low priority",
        priority: 1,
        created_at: nowMs - 30 * dayMs,
      }),
    ]);

    expect(focus.selected_goal?.description).toBe("Fresh high priority");
  });

  it("returns null focus when no goal clears the threshold", () => {
    const focus = select(
      [
        goal({
          id: "goal_aaaaaaaaaaaaaaaa",
          description: "Ambient priority only",
          priority: 10,
        }),
      ],
      {
        threshold: 0.45,
      },
    );

    expect(focus.candidates[0]?.score).toBeCloseTo(0.35);
    expect(focus.selected_goal).toBeNull();
    expect(focus.selected_score).toBeNull();
  });

  it("breaks exact score ties deterministically by target date, priority, creation time, then id", () => {
    const targetTie = select([
      goal({
        id: "goal_bbbbbbbbbbbbbbbb",
        description: "Later target",
        priority: 1,
        target_at: nowMs + 21 * dayMs,
      }),
      goal({
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "Earlier target",
        priority: 1,
        target_at: nowMs + 20 * dayMs,
      }),
    ]);
    const createdAtTie = select([
      goal({
        id: "goal_bbbbbbbbbbbbbbbb",
        description: "Newer same score",
        priority: 1,
        last_progress_ts: nowMs,
        created_at: nowMs - dayMs,
      }),
      goal({
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "Older same score",
        priority: 1,
        last_progress_ts: nowMs,
        created_at: nowMs - 2 * dayMs,
      }),
    ]);
    const idTie = select([
      goal({
        id: "goal_bbbbbbbbbbbbbbbb",
        description: "Beta id",
        priority: 1,
      }),
      goal({
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "Alpha id",
        priority: 1,
      }),
    ]);

    expect(targetTie.selected_goal?.description).toBe("Earlier target");
    expect(createdAtTie.selected_goal?.description).toBe("Older same score");
    expect(idTie.selected_goal?.description).toBe("Alpha id");
  });
});
