import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { goalAdmissionPayload } from "../../autonomy/triggers/goal-admission.js";

import { createMigrations } from "../../borg/storage-setup.js";
import { selectExecutiveFocus, ExecutiveStepsRepository } from "../../executive/index.js";
import { EntityRepository } from "../../memory/commitments/index.js";
import { IdentityEventRepository } from "../../memory/identity/index.js";
import {
  GoalsRepository,
  OpenQuestionsRepository,
  currentGoalBlock,
  goalSchedulingTimes,
  goalBlockStateFields,
} from "../../memory/self/index.js";
import { relationshipPrivateMemoryDisclosureLabel } from "../../memory/common/disclosure-label.js";
import { TrainOfThoughtRepository } from "../../memory/train-of-thought/index.js";
import { listUnfinishedGoalsForCognition } from "../../cognition/self/active-goals.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import {
  StreamEntryIndexRepository,
  StreamWriter,
  StreamWatermarkRepository,
} from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import {
  createEntityId,
  createGoalId,
  createSessionId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
  type GoalId,
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
    dataDir,
    clock,
    identityEventRepository: events,
    executiveStepsRepository: steps,
    watermarkRepository: new StreamWatermarkRepository({ db, clock }),
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

// Frozen from `git show 632981aa:src/memory/self/goal-blocks.ts` and its
// artifact-reference/disclosure-label dependencies. ID validators inline the
// base util/ids.ts grammar; TypeScript brands do not change stored JSON.
const baseEntityIdSchema = z.string().regex(/^ent_[a-z0-9]{16}$/);
const baseGoalIdSchema = z.string().regex(/^goal_[a-z0-9]{16}$/);
const baseOpenQuestionIdSchema = z.string().regex(/^oq_[a-z0-9]{16}$/);
const baseStreamEntryIdSchema = z.string().regex(/^strm_[a-z0-9]{16}$/);
const baseArtifactReferenceSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("journal_entry"), id: z.number().int().positive() }).strict(),
  z.object({ kind: z.literal("created_open_question"), id: baseOpenQuestionIdSchema }).strict(),
  z.object({ kind: z.literal("resolved_open_question"), id: baseOpenQuestionIdSchema }).strict(),
  z
    .object({ kind: z.literal("scheduled_wake"), id: z.string().regex(/^swake_[a-z0-9]{16}$/) })
    .strict(),
  z
    .object({
      kind: z.literal("executive_step_outcome"),
      id: z.string().regex(/^exstep_[a-z0-9]{16}$/),
    })
    .strict(),
  z.object({ kind: z.literal("delivered_outbound_post"), id: baseStreamEntryIdSchema }).strict(),
  z.object({ kind: z.literal("stream_entry"), id: baseStreamEntryIdSchema }).strict(),
]);
const baseDisclosureLabelSchema = z
  .object({
    disclosure_class: z.enum([
      "public",
      "relationship_private",
      "operator_private",
      "self_private",
      "sensitive",
      "unknown",
    ]),
    origin_audience_entity_ids: z.array(baseEntityIdSchema),
    private_to_entity_ids: z.array(baseEntityIdSchema),
    public_to_entity_ids: z.array(baseEntityIdSchema),
  })
  .overwrite((metadata) => ({
    disclosure_class: metadata.disclosure_class,
    origin_audience_entity_ids: [...new Set(metadata.origin_audience_entity_ids)],
    private_to_entity_ids: [...new Set(metadata.private_to_entity_ids)].sort(),
    public_to_entity_ids: [...new Set(metadata.public_to_entity_ids)].sort(),
  }));
const baseBlockerSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("goal"), goal_id: baseGoalIdSchema }).strict(),
  z.object({ kind: z.literal("entity"), entity_id: baseEntityIdSchema }).strict(),
  z
    .object({
      kind: z.literal("until"),
      until: z.number().int().nonnegative().max(8_640_000_000_000_000),
    })
    .strict(),
]);
const baseBlockInputSchema = z
  .object({
    blocker: baseBlockerSchema,
    attempt_status: z.literal("attempted_unavailable"),
    reason: z.string().trim().min(1),
    attempt_evidence: baseArtifactReferenceSchema.optional(),
  })
  .strict();
const baseBlockHistoryFields = {
  unblocked_at: z.number().finite().nullable(),
  unblock_reason: z.string().nullable(),
  disclosure_label: baseDisclosureLabelSchema.default(() => ({
    disclosure_class: "unknown" as const,
    origin_audience_entity_ids: [],
    private_to_entity_ids: [],
    public_to_entity_ids: [],
  })),
};
const baseBlockHistorySchema = z.union([
  baseBlockInputSchema.extend({
    ...baseBlockHistoryFields,
    blocked_at: z.number().finite(),
  }),
  z.object({
    ...baseBlockHistoryFields,
    blocker: z.object({ kind: z.literal("legacy_unknown") }).strict(),
    attempt_status: z.literal("not_recorded"),
    reason: z.string(),
    attempt_evidence: baseArtifactReferenceSchema.optional(),
    blocked_at: z.null(),
  }),
]);

// Frozen from `git show 632981aa:src/executive/goal-presentation.ts`.
// These are plain objects in the base: unknown fields are stripped, not rejected.
const baseExecutiveAcknowledgementSchema = z.object({
  progressAnchor: z.number(),
  inputGeneration: z.number().int().nonnegative(),
  turnId: z.string(),
});
const basePresentationStateSchema = z.object({
  version: z.literal(1),
  inputGeneration: z.number().int().nonnegative(),
  lastProgressVersion: z.number().int().nonnegative().nullable(),
  presented: baseExecutiveAcknowledgementSchema.nullable(),
});

function expectBaseBlockHistory(h: ReturnType<typeof harness>, goalId: GoalId): void {
  const row = h.db.prepare("SELECT block_history_json FROM goals WHERE id = ?").get(goalId) as {
    block_history_json: string;
  };
  const history = JSON.parse(row.block_history_json) as Record<string, unknown>[];
  expect(history.length).toBeGreaterThan(0);
  for (const block of history) {
    expect(baseBlockHistorySchema.parse(block)).toEqual(block);
    expect(baseBlockHistorySchema.safeParse({ ...block, unblock_basis: null }).success).toBe(false);
  }
}

function expectBaseWatermarks(
  h: ReturnType<typeof harness>,
  goalId: GoalId,
  expected: z.infer<typeof basePresentationStateSchema>,
  expectedExecutive: z.infer<typeof baseExecutiveAcknowledgementSchema> | null,
): void {
  const read = h.db.prepare(
    "SELECT metadata_json FROM stream_watermarks WHERE process_name = ? AND session_id = ?",
  );
  const state = read.get(`autonomy:goal-presentation:${goalId}`, DEFAULT_SESSION_ID) as {
    metadata_json: string;
  };
  const stored = JSON.parse(state.metadata_json);
  const parsed = basePresentationStateSchema.parse(stored);
  expect(stored).toHaveProperty("inputEvents");
  expect(parsed).not.toHaveProperty("inputEvents");
  expect(parsed).toEqual(expected);
  const executive = read.get(
    `autonomy:goal-presentation:executive:${goalId}`,
    DEFAULT_SESSION_ID,
  ) as { metadata_json: string } | undefined;
  if (expectedExecutive === null) expect(executive).toBeUndefined();
  else
    expect(baseExecutiveAcknowledgementSchema.parse(JSON.parse(executive!.metadata_json))).toEqual(
      expectedExecutive,
    );
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
      expectBaseBlockHistory(h, blocked.id);
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

  it.each(["done", "superseded", "abandoned", "retired"] as const)(
    "unblocks only when the blocker goal reaches terminal status %s, with the event basis",
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
      expect(h.goals.presentation!.admission(unblocked, "executive").new_inputs).toMatchObject([
        {
          kind: "block_ended",
          ended_by: "goal_terminal",
          goal_id: dependency.id,
          status: status === "retired" ? "abandoned" : status,
        },
      ]);
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

  it("keeps the dependency blocked because blocked is not a terminal status", () => {
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
    expect(h.goals.presentation!.admission(active, "executive").new_inputs).toMatchObject([
      { kind: "block_ended", ended_by: "until_elapsed", until: 5_000, blocked_at: 2_000 },
    ]);
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
    expect(
      h.goals.presentation!.admission(h.goals.get(goal.id)!, "executive").new_inputs,
    ).toMatchObject([
      { kind: "block_ended", ended_by: "entity_inbound", entity_id: entity, entry_id: arrival.id },
    ]);
    expect(h.events.list({ recordId: goal.id })[0]?.reason).toContain(
      `inbound stream entry ${arrival.id}`,
    );
  });

  it("persists each rationale's conservative source labels instead of inheriting the goal owner's label", async () => {
    const h = harness();
    const owner = h.entities.resolve("Alice");
    const sourceOwner = h.entities.resolve("ボブ");
    const question = new OpenQuestionsRepository({ db: h.db, clock: h.clock }).add({
      question: "私的な試行記録",
      urgency: 0.5,
      audience_entity_id: sourceOwner,
      disclosure_label: relationshipPrivateMemoryDisclosureLabel([sourceOwner]),
      source: "reflection",
      provenance,
    });
    const goal = h.goals.add({
      description: "Follow up",
      ownerEntityId: owner,
      priority: 1,
      provenance,
    });
    const tool = createGoalsBlockTool({ goalsRepository: h.goals });
    const result = await tool.invoke(
      {
        goal_id: goal.id,
        ...declaration,
        blocker: { kind: "until", until: 9_000 },
        attempt_evidence: { kind: "created_open_question", id: question.id },
      },
      { sessionId: DEFAULT_SESSION_ID, origin: "autonomous" },
    );
    const label = result.goal.block_history![0]!.disclosure_label;
    expect(label).toMatchObject({
      disclosure_class: "unknown",
      private_to_entity_ids: expect.arrayContaining([owner, sourceOwner]),
      public_to_entity_ids: [],
    });
    const stored = JSON.parse(
      (
        h.db.prepare("SELECT block_history_json FROM goals WHERE id = ?").get(goal.id) as {
          block_history_json: string;
        }
      ).block_history_json,
    );
    expect(stored[0].disclosure_label).toEqual(label);
    const reopened = new GoalsRepository({
      db: h.db,
      clock: h.clock,
      identityEventRepository: h.events,
    });
    expect(goalBlockStateFields(reopened.get(goal.id)!).block?.disclosure_label).toEqual(label);
    const unblocked = reopened.unblock(goal.id, "Autre raison privée", provenance);
    expect(unblocked.block_history![0]!.disclosure_label).toEqual(label);
    expect(listUnfinishedGoalsForCognition(reopened).map((row) => row.id)).toContain(goal.id);
  });

  it("pauses only the portion of a block after the current deadline's assignment in both TS and SQL", () => {
    const h = harness();
    const goal = h.add();
    h.goals.block(
      goal.id,
      { ...declaration, blocker: { kind: "until", until: 5_000 } },
      provenance,
    );
    h.clock.advance(4_000);
    h.goals.reconcileBlocks();
    const assignedLater = h.goals.update(goal.id, { target_at: 7_000 }, provenance);
    expect(assignedLater.target_assigned_at).toBe(5_000);
    expect(goalSchedulingTimes(assignedLater).targetAt).toBe(7_000);
    const dueAt = () =>
      h.goals.listActiveFollowupDueCandidatesReadOnly({
        lookaheadMs: 0,
        staleMs: 100_000,
        limit: 5,
      })[0]?.due_at;
    expect(dueAt()).toBe(7_001);
    h.clock.advance(1_000);
    h.goals.block(
      goal.id,
      { ...declaration, blocker: { kind: "until", until: 8_000 } },
      provenance,
    );
    h.clock.advance(1_000);
    h.goals.update(goal.id, { target_at: 9_000 }, provenance);
    h.clock.advance(1_000);
    h.goals.reconcileBlocks();
    expect(goalSchedulingTimes(h.goals.get(goal.id)!).targetAt).toBe(10_000);
    expect(dueAt()).toBe(10_001);
  });

  it("requires a recorded delivered outcome for delivered_outbound_post while keeping generic message evidence", async () => {
    const h = harness();
    const message = await h.writer.append({
      kind: "agent_msg",
      content: "composed before transport",
    });
    const input = {
      ...declaration,
      blocker: { kind: "until" as const, until: 9_000 },
      attempt_evidence: { kind: "delivered_outbound_post" as const, id: message.id },
    };
    const goal = h.add();
    expect(() => h.goals.block(goal.id, input, provenance)).toThrow();
    for (const state of ["transport_failed", "not_transportable"]) {
      await h.writer.append({
        kind: "tool_result",
        content: {
          ok: true,
          output: { outbound: { delivery_outcome: { state, agent_message_id: message.id } } },
        },
      });
      expect(() => h.goals.block(goal.id, input, provenance)).toThrow();
    }
    const delivered = {
      ok: true,
      output: {
        outbound: { delivery_outcome: { state: "delivered", agent_message_id: message.id } },
      },
    };
    await h.writer.append({ kind: "tool_result", turn_status: "aborted", content: delivered });
    expect(() => h.goals.block(goal.id, input, provenance)).toThrow();
    expect(
      h.goals.block(
        h.add().id,
        { ...input, attempt_evidence: { kind: "stream_entry", id: message.id } },
        provenance,
      ).status,
    ).toBe("blocked");
    await h.writer.append({ kind: "tool_result", content: delivered });
    expect(h.goals.block(goal.id, input, provenance).status).toBe("blocked");
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
    expect(blockTool.description).toContain(
      "After a lane has shown me a goal, that lane shows it again",
    );
    expect(blockTool.description).toContain("a linked input, a goal write, or the end of a block");
    expect(blockTool.description).toContain(
      "makes the goal eligible for me again when the timestamp passes, under the usual scheduling checks",
    );
    const unblockTool = createGoalsUnblockTool({ goalsRepository: h.goals });
    const context = { sessionId: DEFAULT_SESSION_ID, origin: "autonomous" as const };
    expect(blockTool.allowedOrigins).toEqual(["autonomous", "deliberator"]);
    const result = await blockTool.invoke(
      { goal_id: goal.id, ...declaration, blocker: { kind: "until", until: 9_000 } },
      context,
    );
    expect(blockTool.outputSchema.safeParse(result).success).toBe(true);
    expect(result.goal.block_history?.[0]).toMatchObject({
      unblocked_at: null,
      unblock_reason: null,
    });
    expectBaseBlockHistory(h, goal.id);
    expect(h.goals.presentation!.snapshot(goal).inputGeneration).toBe(0);
    expect(h.goals.get(goal.id)?.last_interaction_at).toBe(h.clock.now());
    expect(unblockTool.inputSchema.safeParse({ goal_id: goal.id, reason: "" }).success).toBe(false);
    const active = await unblockTool.invoke(
      { goal_id: goal.id, reason: "Reprise manuelle" },
      context,
    );
    expect(active.goal.status).toBe("active");
    expect(active.goal.block_history?.[0]?.unblock_reason).toBe("Reprise manuelle");
    expectBaseBlockHistory(h, goal.id);
    expect(
      h.goals.presentation!.admission(h.goals.get(goal.id)!, "executive").new_inputs,
    ).toMatchObject([{ generation: 1, kind: "block_ended", ended_by: "manual" }]);
  });

  it.each(["until_elapsed", "entity_inbound", "goal_terminal", "manual"] as const)(
    "keeps block and %s unblock writes readable by the frozen base schemas",
    async (endedBy) => {
      const h = harness();
      const goal = h.add();
      const dependency = h.add();
      const entity = h.entities.resolve("李", { kind: "person" });
      const store = h.goals.presentation!;
      const presented = { progressAnchor: goal.created_at, inputGeneration: 0, turnId: "before" };
      store.presented([store.snapshot(goal)], presented.turnId, new Set([goal.id]));
      h.goals.block(
        goal.id,
        {
          ...declaration,
          blocker:
            endedBy === "entity_inbound"
              ? { kind: "entity", entity_id: entity }
              : endedBy === "goal_terminal"
                ? { kind: "goal", goal_id: dependency.id }
                : { kind: "until", until: 2_000 },
        },
        provenance,
      );
      expectBaseBlockHistory(h, goal.id);
      expectBaseWatermarks(
        h,
        goal.id,
        {
          version: 1,
          inputGeneration: 0,
          lastProgressVersion: null,
          presented,
        },
        presented,
      );
      h.clock.advance(1_000);
      if (endedBy === "manual") h.goals.unblock(goal.id, "解決", provenance);
      else if (endedBy === "entity_inbound") {
        await h.writer.append({
          kind: "user_msg",
          sender_entity_id: entity,
          content: "戻りました",
        });
      } else if (endedBy === "goal_terminal")
        h.goals.updateStatus(dependency.id, "done", provenance);
      else h.goals.reconcileBlocks();
      const released = h.goals.get(goal.id)!;
      expect(released.status).toBe("active");
      expectBaseBlockHistory(h, goal.id);
      expectBaseWatermarks(
        h,
        goal.id,
        {
          version: 1,
          inputGeneration: 1,
          lastProgressVersion: 3,
          presented,
        },
        presented,
      );
      expect(h.goals.presentation!.admission(released, "executive").new_inputs).toMatchObject([
        { kind: "block_ended", ended_by: endedBy },
      ]);
    },
  );

  it("keeps linked input, goal write and executive presentation watermarks readable by the frozen base schemas", () => {
    const h = harness();
    const goal = h.add();
    const store = h.goals.presentation!;
    store.associateInbound(goal.id, createStreamEntryId());
    expectBaseWatermarks(
      h,
      goal.id,
      {
        version: 1,
        inputGeneration: 1,
        lastProgressVersion: null,
        presented: null,
      },
      null,
    );
    h.clock.advance(100);
    h.goals.updateProgress(goal.id, "進みました", provenance);
    expectBaseWatermarks(
      h,
      goal.id,
      {
        version: 1,
        inputGeneration: 2,
        lastProgressVersion: 2,
        presented: null,
      },
      null,
    );
    const progressed = h.goals.get(goal.id)!;
    const presented = { progressAnchor: h.clock.now(), inputGeneration: 2, turnId: "executive" };
    store.presented([store.snapshot(progressed)], presented.turnId, new Set([goal.id]));
    expectBaseWatermarks(
      h,
      goal.id,
      {
        version: 1,
        inputGeneration: 2,
        lastProgressVersion: 2,
        presented,
      },
      presented,
    );
  });

  it.each(["entity_inbound", "goal_terminal"] as const)(
    "labels flattened %s handles at least as restrictively as their blocker",
    async (endedBy) => {
      const h = harness();
      const blockerEntity = h.entities.resolve("李", { kind: "person" });
      const dependency = h.goals.add({
        description: "私的な依存",
        ownerEntityId: blockerEntity,
        priority: 1,
        provenance,
      });
      const goal = h.add();
      const blocked = h.goals.block(
        goal.id,
        {
          ...declaration,
          blocker:
            endedBy === "entity_inbound"
              ? { kind: "entity", entity_id: blockerEntity }
              : { kind: "goal", goal_id: dependency.id },
        },
        provenance,
      );
      const blockLabel = blocked.block_history![0]!.disclosure_label;
      h.clock.advance(100);
      const entry =
        endedBy === "entity_inbound"
          ? await h.writer.append({
              kind: "user_msg",
              sender_entity_id: blockerEntity,
              content: "戻りました",
            })
          : null;
      if (endedBy === "goal_terminal") h.goals.updateStatus(dependency.id, "done", provenance);
      const released = h.goals.get(goal.id)!;
      const event = goalAdmissionPayload(
        h.goals.presentation!.admission(released, "executive"),
        released,
      ).new_inputs[0]!;
      expect(event).toMatchObject({
        kind: "block_ended",
        ended_by: endedBy,
        disclosure: expect.any(String),
        ...(entry === null
          ? { goal_id: dependency.id, status: "done" }
          : { entity_id: blockerEntity, entry_id: entry.id }),
        disclosure_label: {
          disclosure_class: "unknown",
          private_to_entity_ids: [blockerEntity],
          public_to_entity_ids: [],
        },
      });
      expect(event.disclosure_label).toEqual(blockLabel);
      expect(event).not.toHaveProperty("basis");
    },
  );

  it.each(["done", "superseded", "abandoned"] as const)(
    "records closure as %s in block history without adding a block-ended input",
    (status) => {
      const h = harness();
      const store = h.goals.presentation!;
      for (const write of ["update_status", "update"] as const) {
        const goal = h.add();
        h.goals.block(
          goal.id,
          { ...declaration, blocker: { kind: "until", until: 9_000 } },
          provenance,
        );
        const presented = {
          progressAnchor: goal.created_at,
          inputGeneration: 0,
          turnId: `shown-${write}`,
        };
        store.presented([store.snapshot(goal)], presented.turnId, new Set([goal.id]));
        h.clock.advance(100);
        if (write === "update_status") h.goals.updateStatus(goal.id, status, provenance);
        else h.goals.update(goal.id, { status }, provenance);
        h.goals.reconcileBlocks();
        const closed = h.goals.get(goal.id)!;
        expect(closed.block_history?.[0]).toMatchObject({
          unblocked_at: h.clock.now(),
          unblock_reason: `goal closed as ${status}`,
        });
        expectBaseBlockHistory(h, goal.id);
        expectBaseWatermarks(
          h,
          goal.id,
          {
            version: 1,
            inputGeneration: 1,
            lastProgressVersion: 3,
            presented,
          },
          presented,
        );
        // Closing the history interval creates no event; only the normal goal
        // status write advances the generation, exactly as before.
        expect(store.admission(closed, "followup").new_inputs).toEqual([
          { generation: 1, kind: "goal_write", write, record_version: 3 },
        ]);
        expect(store.snapshot(closed).inputGeneration).toBe(1);
      }
    },
  );

  it("rolls back the input log with a failed unblock audit", () => {
    const h = harness();
    const goal = h.add();
    h.goals.block(
      goal.id,
      { ...declaration, blocker: { kind: "until", until: 9_000 } },
      provenance,
    );
    const before = h.goals.presentation!.admission(h.goals.get(goal.id)!, "executive");
    vi.spyOn(h.events, "record").mockImplementationOnce(() => {
      throw new Error("audit failed");
    });
    expect(() => h.goals.unblock(goal.id, "解決", provenance)).toThrow("audit failed");
    expect(h.goals.get(goal.id)?.status).toBe("blocked");
    expect(h.goals.presentation!.admission(h.goals.get(goal.id)!, "executive")).toEqual(before);
  });
});
