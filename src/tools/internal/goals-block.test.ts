import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createMigrations } from "../../borg/storage-setup.js";
import { selectExecutiveFocus, ExecutiveStepsRepository } from "../../executive/index.js";
import { EntityRepository } from "../../memory/commitments/index.js";
import { IdentityEventRepository } from "../../memory/identity/index.js";
import { GoalsRepository, currentGoalBlock, goalSchedulingTimes } from "../../memory/self/index.js";
import { TrainOfThoughtRepository } from "../../memory/train-of-thought/index.js";
import { listUnfinishedGoalsForCognition } from "../../cognition/self/active-goals.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { StreamEntryIndexRepository, StreamWriter } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import {
  createEntityId,
  createGoalId,
  createSessionId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
} from "../../util/ids.js";
import { createGoalsBlockTool, createGoalsUnblockTool } from "./goals-block.js";

const provenance = { kind: "manual" } as const;
const declaration = {
  attempt_status: "attempted_unavailable",
  reason: "試しましたが、まだ利用できません。",
} as const;
const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const cleanup of cleanups.splice(0)) cleanup();
});

function harness() {
  const dataDir = mkdtempSync(join(tmpdir(), "borg-goal-block-"));
  const db = openDatabase(":memory:", { migrations: createMigrations() });
  cleanups.push(() => {
    db.close();
    rmSync(dataDir, { recursive: true, force: true });
  });
  const clock = new ManualClock(1_000);
  const events = new IdentityEventRepository({ db, clock });
  const steps = new ExecutiveStepsRepository({ db, clock });
  const goals = new GoalsRepository({
    db,
    clock,
    identityEventRepository: events,
    executiveStepsRepository: steps,
  });
  const entities = new EntityRepository({ db, clock });
  const entryIndex = new StreamEntryIndexRepository({ db, dataDir });
  const writer = new StreamWriter({
    dataDir,
    clock,
    entryIndex,
    onAppend: () => goals.reconcileBlocks(),
  });
  const add = () =>
    goals.add({ description: "等待資料", priority: 9, targetAt: 10_000, provenance });
  return { dataDir, db, clock, events, steps, goals, entities, writer, entryIndex, add };
}

describe("named goal blocks", () => {
  it("rejects absent, multiple, malformed and unknown blockers and not-attempted declarations", () => {
    const h = harness();
    const goal = h.add();
    const tool = createGoalsBlockTool({ goalsRepository: h.goals });
    for (const input of [
      { goal_id: goal.id, ...declaration },
      { goal_id: goal.id, ...declaration, blocker: {} },
      {
        goal_id: goal.id,
        ...declaration,
        blocker: { kind: "until", until: 2_000, entity_id: createEntityId() },
      },
      { goal_id: goal.id, ...declaration, blocker: { kind: "until", until: NaN } },
      {
        goal_id: goal.id,
        ...declaration,
        blocker: { kind: "until", until: 2_000 },
        attempt_status: "not_attempted",
      },
      { goal_id: goal.id, ...declaration, blocker: { kind: "until", until: 2_000 }, reason: " " },
    ])
      expect(tool.inputSchema.safeParse(input).success).toBe(false);
    for (const blocker of [
      { kind: "goal", goal_id: createGoalId() },
      { kind: "goal", goal_id: goal.id },
      { kind: "entity", entity_id: createEntityId() },
    ] as const)
      expect(() => h.goals.block(goal.id, { ...declaration, blocker }, provenance)).toThrow();
    expect(() =>
      h.goals.add({ description: "x", priority: 1, status: "blocked", provenance }),
    ).toThrow();
    expect(() => h.goals.updateStatus(goal.id, "blocked", provenance)).toThrow();
    expect(() => h.goals.update(goal.id, { status: "blocked" }, provenance)).toThrow();
    expect(h.goals.get(goal.id)?.status).toBe("active");
  });

  it("validates artifacts across turns, rejects missing, future and aborted record handles", async () => {
    const h = harness();
    const selfId = h.entities.resolve("私", { kind: "self" });
    const journal = new TrainOfThoughtRepository({ db: h.db, clock: h.clock });
    const prior = journal.append({
      text: "試行記録",
      selfEntityId: selfId,
      sourceTurnId: "earlier",
    });
    h.clock.advance(100);
    const current = journal.append({
      text: "Tentative indisponible",
      selfEntityId: selfId,
      sourceTurnId: "current",
    });
    for (const record of [prior, current]) {
      const blocked = h.goals.block(
        h.add().id,
        {
          ...declaration,
          blocker: { kind: "until", until: 9_000 },
          attempt_evidence: { kind: "journal_entry", id: record.id },
        },
        provenance,
      );
      expect(currentGoalBlock(blocked)?.attempt_evidence?.id).toBe(record.id);
    }
    const goal = h.add();
    const future = journal.append({ text: "future", selfEntityId: selfId, now: 20_000 });
    const aborted = await h.writer.append({
      kind: "tool_result",
      turn_status: "aborted",
      content: {},
    });
    for (const ref of [
      { kind: "journal_entry", id: 999_999 },
      { kind: "journal_entry", id: future.id },
      { kind: "stream_entry", id: createStreamEntryId() },
      { kind: "stream_entry", id: aborted.id },
    ] as const)
      expect(() =>
        h.goals.block(
          goal.id,
          { ...declaration, blocker: { kind: "until", until: 9_000 }, attempt_evidence: ref },
          provenance,
        ),
      ).toThrow();
  });

  it.each(["done", "abandoned", "retired"] as const)(
    "unblocks when the blocker goal becomes %s, with the event basis",
    (status) => {
      const h = harness();
      const dependency = h.add();
      const goal = h.add();
      h.goals.block(
        goal.id,
        { ...declaration, blocker: { kind: "goal", goal_id: dependency.id } },
        provenance,
      );
      h.clock.advance(400);
      if (status === "retired") h.goals.retire(dependency.id, "完成", provenance);
      else h.goals.updateStatus(dependency.id, status, provenance);
      const unblocked = h.goals.get(goal.id)!;
      expect(unblocked.status).toBe("active");
      expect(unblocked.block_history?.[0]).toMatchObject({
        blocked_at: 1_000,
        unblocked_at: 1_400,
        reason: declaration.reason,
      });
      expect(h.events.list({ recordId: goal.id })[0]).toMatchObject({
        action: "unblock",
        reason: expect.stringContaining(`blocker goal ${dependency.id}`),
      });
      expect(h.events.list({ recordId: goal.id })[0]?.reason).toContain("identity event");
      h.goals.reconcileBlocks();
      expect(
        h.events.list({ recordId: goal.id }).filter((event) => event.action === "unblock"),
      ).toHaveLength(1);
    },
  );

  it("does not unblock when the blocker goal itself becomes blocked", () => {
    const h = harness();
    const dependency = h.add();
    const goal = h.add();
    h.goals.block(
      goal.id,
      { ...declaration, blocker: { kind: "goal", goal_id: dependency.id } },
      provenance,
    );
    h.goals.block(
      dependency.id,
      { ...declaration, blocker: { kind: "until", until: 9_000 } },
      provenance,
    );
    expect(h.goals.get(goal.id)?.status).toBe("blocked");
  });

  it("unblocks at the until boundary, pauses both clocks and preserves steps and recall", () => {
    const h = harness();
    const goal = h.add();
    const step = h.steps.add({
      goalId: goal.id,
      description: "待つ",
      kind: "wait",
      dueAt: 10_000,
      provenance,
    });
    h.clock.advance(1_000);
    h.goals.block(
      goal.id,
      { ...declaration, blocker: { kind: "until", until: 5_000 } },
      provenance,
    );
    const recalled = listUnfinishedGoalsForCognition(h.goals);
    expect(recalled.map((goal) => goal.id)).toContain(goal.id);
    const score = (goals: typeof recalled, nowMs: number) =>
      selectExecutiveFocus({
        goals,
        cognitionInput: "",
        nowMs,
        staleMs: 10_000,
        deadlineLookaheadMs: 10_000,
      });
    expect(score(recalled, 4_999).candidates).toEqual([]);
    expect(
      h.goals.listActiveFollowupDueCandidatesReadOnly({
        lookaheadMs: 10_000,
        staleMs: 10_000,
        limit: 10,
      }),
    ).toEqual([]);
    expect(h.steps.get(step.id)?.status).toBe("queued");
    h.clock.advance(2_999);
    h.goals.reconcileBlocks();
    expect(h.goals.get(goal.id)?.status).toBe("blocked");
    h.clock.advance(1);
    h.goals.reconcileBlocks();
    const active = h.goals.get(goal.id)!;
    expect(active.status).toBe("active");
    expect(active.last_progress_ts).toBeNull();
    expect(active.target_at).toBe(10_000);
    expect(goalSchedulingTimes(active)).toEqual({ progressAnchor: 4_000, targetAt: 13_000 });
    expect(score([active], 5_000).candidates[0]?.components).toMatchObject({
      progress_debt: 0.1,
      deadline_pressure: expect.closeTo(0.2),
    });
    expect(
      h.goals.listActiveFollowupDueCandidatesReadOnly({
        lookaheadMs: 1_000,
        staleMs: 10_000,
        limit: 10,
      })[0]?.due_at,
    ).toBe(12_001);
    expect(h.events.list({ recordId: goal.id })[0]?.reason).toContain(
      "until timestamp 5000 passed",
    );
  });

  it("unblocks only for a later inbound from the named entity, including a different session", async () => {
    const h = harness();
    const entity = h.entities.resolve("李", { kind: "person" });
    const other = h.entities.resolve("Mira", { kind: "person" });
    const goal = h.add();
    await h.writer.append({ kind: "user_msg", sender_entity_id: entity, content: "old" });
    h.goals.block(
      goal.id,
      { ...declaration, blocker: { kind: "entity", entity_id: entity } },
      provenance,
    );
    await h.writer.append({ kind: "user_msg", sender_entity_id: entity, content: "same instant" });
    h.clock.advance(1);
    await h.writer.append({ kind: "agent_msg", sender_entity_id: entity, content: "outbound" });
    await h.writer.append({
      kind: "user_msg",
      sender_entity_id: other,
      content: "inbound from someone else",
    });
    expect(h.goals.get(goal.id)?.status).toBe("blocked");
    const otherSession = new StreamWriter({
      dataDir: h.dataDir,
      sessionId: createSessionId(),
      clock: h.clock,
      entryIndex: h.entryIndex,
      onAppend: () => h.goals.reconcileBlocks(),
    });
    const arrival = await otherSession.append({
      kind: "agent_observed",
      sender_entity_id: entity,
      content: "戻りました",
    });
    expect(h.goals.get(goal.id)?.status).toBe("active");
    expect(h.events.list({ recordId: goal.id })[0]?.reason).toContain(
      `inbound stream entry ${arrival.id}`,
    );
  });

  it("commits the transition and audit atomically, and manual unblock requires a reason", async () => {
    const h = harness();
    const goal = h.add();
    const spy = vi.spyOn(h.events, "record").mockImplementationOnce(() => {
      throw new Error("audit failed");
    });
    expect(() =>
      h.goals.block(
        goal.id,
        { ...declaration, blocker: { kind: "until", until: 9_000 } },
        provenance,
      ),
    ).toThrow("audit failed");
    expect(h.goals.get(goal.id)?.status).toBe("active");
    spy.mockRestore();
    const blockTool = createGoalsBlockTool({ goalsRepository: h.goals });
    const unblockTool = createGoalsUnblockTool({ goalsRepository: h.goals });
    const context = { sessionId: DEFAULT_SESSION_ID, origin: "autonomous" as const };
    expect(blockTool.allowedOrigins).toEqual(["autonomous", "deliberator"]);
    const result = await blockTool.invoke(
      { goal_id: goal.id, ...declaration, blocker: { kind: "until", until: 9_000 } },
      context,
    );
    expect(blockTool.outputSchema.safeParse(result).success).toBe(true);
    expect(unblockTool.inputSchema.safeParse({ goal_id: goal.id, reason: "" }).success).toBe(false);
    const active = await unblockTool.invoke(
      { goal_id: goal.id, reason: "Reprise manuelle" },
      context,
    );
    expect(active.goal.status).toBe("active");
    expect(active.goal.block_history?.[0]?.unblock_reason).toBe("Reprise manuelle");
  });
});
