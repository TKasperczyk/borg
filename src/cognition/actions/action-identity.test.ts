import { afterEach, describe, expect, it, vi } from "vitest";
import { openDatabase } from "../../storage/sqlite/index.js";
import {
  ActionRepository,
  actionMigrations,
  ACTIVE_ACTION_STATES,
  type ActionRecord,
} from "../../memory/actions/index.js";
import { goalSchema } from "../../memory/self/index.js";
import { makeActionRecord } from "../../test-support/factories/memory.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { LLMCompleteResult } from "../../llm/index.js";
import {
  createActionId,
  createEntityId,
  createGoalId,
  createStreamEntryId,
} from "../../util/ids.js";
import { FixedClock } from "../../util/clock.js";
import { ActionStateExtractor } from "./action-state-extractor.js";
import {
  presentActionCandidates,
  ACTION_CANDIDATE_MAX_CHARS,
  ACTION_CANDIDATE_LIMIT,
  retrieveActionCandidates,
} from "./action-candidates.js";
import { actionMemoryDisclosureLabel } from "../../memory/common/disclosure-serializers.js";
import {
  renderActionThreadText,
  ACTION_UPDATE_HISTORY_LIMIT,
  ACTION_UPDATE_HISTORY_MAX_CHARS,
} from "../evidence-ledger/action-threads.js";

const close: Array<() => void> = [];
afterEach(() => {
  for (const cleanup of close.splice(0)) cleanup();
});

function fixture(onCompleted?: (record: ActionRecord, previous: ActionRecord | null) => void) {
  const db = openDatabase(":memory:", { migrations: actionMigrations });
  close.push(() => db.close());
  const repository = new ActionRepository({ db, clock: new FixedClock(2_000), onCompleted });
  const action = makeActionRecord({
    actor: createEntityId(),
    created_at: 1_000,
    updated_at: 1_000,
    goal_id: createGoalId(),
    description: "最初の約束",
  });
  repository.add(action);
  return { db, repository, action };
}

function result(candidate: Record<string, unknown>, cut = false): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 10,
    output_tokens: 20,
    stop_reason: cut ? "max_tokens" : "tool_use",
    tool_calls: [
      { id: "toolu_actions", name: "EmitActionStates", input: { action_states: [candidate] } },
    ],
  };
}

async function extract(
  f: ReturnType<typeof fixture>,
  options: {
    chosen?: ActionRecord["id"] | null;
    offered?: ActionRecord[];
    actor?: ActionRecord["actor"];
    state?: "scheduled" | "completed" | "considering";
    references?: ActionRecord["id"][];
    invalidSibling?: boolean;
    referenceOnly?: boolean;
    repaired?: boolean;
    nowMs?: number;
    turnCounter?: number;
    failure?: "cut" | "failed";
    goal?: ReturnType<typeof goalSchema.parse>;
    chosenGoal?: ReturnType<typeof createGoalId> | null;
  } = {},
) {
  const source = createStreamEntryId();
  const candidate = {
    classification: "concrete_action",
    description: "J’ai terminé cette étape, voici le résultat.",
    actor: options.actor ?? f.action.actor,
    state: options.state ?? "completed",
    matched_existing_action_id: options.chosen === undefined ? f.action.id : options.chosen,
    goal_id: options.chosenGoal ?? null,
    evidence_stream_entry_ids: [source],
    confidence: 0.9,
  };
  const response = result(candidate, options.failure === "cut");
  const valid = structuredClone(response);
  response.tool_calls![0]!.input = {
    action_states: [
      ...(options.referenceOnly ? [] : [candidate]),
      ...(options.invalidSibling ? [{ ...candidate, state: "invalid" }] : []),
    ],
    referenced_action_ids: options.references ?? [],
  };
  valid.tool_calls![0]!.input = {
    action_states: [candidate],
    referenced_action_ids: options.references ?? [],
  };
  const llm = new FakeLLMClient({ responses: [response, options.repaired ? valid : response] });
  if (options.failure === "failed")
    vi.spyOn(llm, "complete").mockImplementation(async (request) => {
      llm.requests.push(request);
      throw new Error("offline");
    });
  const events: Array<{ event: string; data: Record<string, unknown> }> = [];
  const extractor = new ActionStateExtractor({
    llmClient: llm,
    model: "mock",
    actionRepository: f.repository,
    clock: new FixedClock(options.nowMs ?? 2_000),
    turnId: "turn_selection",
    tracer: {
      enabled: true,
      includePayloads: true,
      emit: (event, data) => events.push({ event, data }),
    },
  });
  const rows = await extractor.extract({
    userMessage: "An opaque turn",
    currentUserStreamEntryId: source,
    recentHistory: [],
    audienceEntityId: createEntityId(),
    activeActionsForReference: options.offered ?? [f.action],
    actionCandidateTotal: 12,
    suppliedGoals: options.goal ? [options.goal] : [],
    goalId: f.action.goal_id,
    turnCounter: options.turnCounter ?? null,
  });
  return { rows, events, llm, source, candidate };
}

describe("model-selected action identity", () => {
  it.each(["scheduled", "completed"] as const)(
    "atomically keeps wording and selection audit for a %s continuation",
    async (state) => {
      const f = fixture();
      const before = f.repository.get(f.action.id)!;
      const { rows, source, candidate, llm, events } = await extract(f, { state });
      expect(
        events.filter((event) => event.event === "extraction.actions.selection_summary"),
      ).toEqual([
        {
          event: "extraction.actions.selection_summary",
          data: expect.objectContaining({
            candidates_shown: 1,
            candidates_omitted: 11,
            matched: 1,
            inserted: 0,
            rejected_by_reason: {},
          }),
        },
      ]);
      expect(rows).toEqual([]);
      expect(llm.requests).toHaveLength(1);
      expect(f.repository.count()).toBe(1);
      expect(f.repository.get(f.action.id)).toMatchObject({
        description: before.description,
        goal_id: before.goal_id,
        state,
      });
      const history = new ActionRepository({ db: f.db }).readUpdateHistory(f.action.id, 5);
      expect(history.total).toBe(1);
      expect(history.updates[0]).toMatchObject({
        text: candidate.description,
        from_state: before.state,
        to_state: state,
        turn_id: "turn_selection",
        timestamp: 2_000,
        source_stream_entry_ids: [source],
        selection_audit: {
          candidate_ids: [f.action.id],
          candidate_omitted_count: 11,
          chosen_action_id: f.action.id,
          rejection_reason: null,
        },
      });
      expect(actionMemoryDisclosureLabel(f.repository.get(f.action.id)!)).toEqual(
        expect.objectContaining({ disclosureClass: "unknown" }),
      );
    },
  );

  it("records none durably and keeps a separate assertion", async () => {
    const f = fixture();
    const before = f.repository.get(f.action.id);
    const { rows, events } = await extract(f, { chosen: null });
    expect(
      events.filter((event) => event.event === "extraction.actions.selection_summary"),
    ).toEqual([
      {
        event: "extraction.actions.selection_summary",
        data: expect.objectContaining({
          candidates_shown: 1,
          candidates_omitted: 11,
          matched: 0,
          inserted: 1,
          rejected_by_reason: {},
        }),
      },
    ]);
    expect(rows).toHaveLength(1);
    expect(new ActionRepository({ db: f.db }).get(rows[0]!.id)?.selection_audit).toMatchObject({
      candidate_ids: [f.action.id],
      candidate_omitted_count: 11,
      chosen_action_id: null,
      rejection_reason: null,
    });
    expect(f.repository.get(f.action.id)).toEqual(before);
  });

  it.each(["match", "none"] as const)(
    "reports committed %s writes when the completion hook throws",
    async (choice) => {
      const onCompleted = vi.fn(() => {
        throw new Error("question resolution failed");
      });
      const f = fixture(onCompleted);
      const { rows, events } = await extract(f, {
        chosen: choice === "match" ? f.action.id : null,
      });
      const committedId = choice === "match" ? f.action.id : rows[0]!.id;
      expect(onCompleted).toHaveBeenCalledOnce();
      expect(rows).toHaveLength(choice === "match" ? 0 : 1);
      expect(f.repository.get(committedId)?.state).toBe("completed");
      expect(f.repository.readUpdateHistory(f.action.id, 5).total).toBe(choice === "match" ? 1 : 0);
      expect(f.db.prepare("SELECT * FROM action_extraction_decisions").all()).toHaveLength(1);
      expect(events).toContainEqual({
        event: "action_state.completion_hook_failed",
        data: expect.objectContaining({
          action_id: committedId,
          error: "question resolution failed",
        }),
      });
      expect(events).toContainEqual({
        event: "extraction.actions.selection_summary",
        data: expect.objectContaining({
          matched: choice === "match" ? 1 : 0,
          inserted: choice === "none" ? 1 : 0,
        }),
      });
      expect(events).toContainEqual({
        event: "extraction.actions.completed",
        data: expect.objectContaining({ degraded: false }),
      });
      expect(events.some((event) => event.data.reason === "repository_failed")).toBe(false);
    },
  );

  it("renders durable history newest first with count and character omissions", async () => {
    const f = fixture();
    for (let index = 0; index < 7; index += 1) await extract(f, { state: "scheduled" });
    const history = f.repository.readUpdateHistory(f.action.id, ACTION_UPDATE_HISTORY_LIMIT);
    expect(history.total).toBe(7);
    expect(history.updates.map((update) => update.id)).toEqual([7, 6, 5, 4, 3]);
    const current = f.repository.get(f.action.id)!;
    const thread = {
      id: current.id,
      current,
      origin: current,
      records: [current],
      scope: "global" as const,
    };
    const text = renderActionThreadText(thread, undefined, history);
    expect(text).toContain("updates: total=7, shown=5, omitted=2; newest first");
    expect(text).toContain(history.updates[0]!.text);
    expect(text).toContain(current.description);
    const oversized = {
      ...history,
      updates: [{ ...history.updates[0]!, text: "x".repeat(ACTION_UPDATE_HISTORY_MAX_CHARS + 1) }],
    };
    expect(renderActionThreadText(thread, undefined, oversized)).toContain(
      "updates: total=7, shown=0, omitted=7",
    );
  });

  it.each(["not_offered", "terminal", "actor_mismatch", "missing"] as const)(
    "preserves a rejected %s target and writes a new row with the reason",
    async (reason) => {
      const f = fixture();
      let chosen = f.action.id;
      let actor = f.action.actor;
      if (reason === "not_offered") chosen = createActionId();
      if (reason === "terminal") f.repository.update(f.action.id, { state: "completed" });
      if (reason === "actor_mismatch") actor = createEntityId();
      if (reason === "missing")
        f.db.prepare("DELETE FROM action_records WHERE id = ?").run(f.action.id);
      const before = f.repository.get(f.action.id);
      const { rows, events } = await extract(f, { chosen, actor });
      expect(rows).toHaveLength(1);
      expect(
        events.filter((event) => event.event === "extraction.actions.selection_summary"),
      ).toEqual([
        {
          event: "extraction.actions.selection_summary",
          data: expect.objectContaining({
            candidates_shown: 1,
            candidates_omitted: 11,
            matched: 0,
            inserted: 1,
            rejected_by_reason: { [reason]: 1 },
          }),
        },
      ]);
      expect(rows[0]!.id).not.toBe(chosen);
      expect(f.repository.get(f.action.id)).toEqual(before);
      expect(f.repository.get(rows[0]!.id)?.selection_audit).toMatchObject({
        chosen_action_id: chosen,
        rejection_reason: reason,
      });
      expect(events).toContainEqual({
        event: "extraction.actions.selection_rejected",
        data: expect.objectContaining({ reason }),
      });
    },
  );

  it.each(["completed", "not_done", "declined", "expired", "archived"] as const)(
    "never reopens lifecycle-protected %s",
    async (state) => {
      const f = fixture();
      f.repository.update(f.action.id, { state });
      const { rows } = await extract(f, { state: "scheduled" });
      expect(rows[0]?.selection_audit?.rejection_reason).toBe("terminal");
      expect(f.repository.get(f.action.id)?.state).toBe(state);
      expect(f.repository.readUpdateHistory(f.action.id, 5).total).toBe(0);
    },
  );

  it.each(["cut", "failed"] as const)(
    "a %s call cannot touch rows or append partial assertions",
    async (failure) => {
      const f = fixture();
      const before = f.repository.list();
      const { rows, events } = await extract(f, { failure, references: [f.action.id] });
      expect(f.db.prepare("SELECT * FROM action_extraction_decisions").all()).toEqual([]);
      expect(
        events.filter((event) => event.event === "extraction.actions.selection_summary"),
      ).toHaveLength(1);
      expect(rows).toEqual([]);
      expect(f.repository.list()).toEqual(before);
      expect(f.repository.readUpdateHistory(f.action.id, 5).total).toBe(0);
    },
  );

  it("rolls back update text and state together if the action write fails", async () => {
    const f = fixture();
    const before = f.repository.get(f.action.id);
    f.db.exec(
      "CREATE TRIGGER reject_action_update BEFORE UPDATE ON action_records BEGIN SELECT RAISE(ABORT, 'write failed'); END;",
    );
    await extract(f);
    expect(f.repository.get(f.action.id)).toEqual(before);
    expect(f.repository.readUpdateHistory(f.action.id, 5).total).toBe(0);
  });

  it("retains the append-only audit when an action is explicitly deleted", async () => {
    const f = fixture();
    await extract(f);
    expect(await f.repository.delete(f.action.id)).toBe(true);
    expect(f.repository.get(f.action.id)).toBeNull();
    expect(f.repository.readUpdateHistory(f.action.id, 5).total).toBe(1);
  });

  it.each(["supplied", "unoffered", "none"] as const)(
    "sets a %s goal only from the model choice, never executive focus",
    async (choice) => {
      const f = fixture();
      const goal = goalSchema.parse({
        id: createGoalId(),
        description: "Another purpose",
        status: "active",
        priority: 0.5,
        parent_goal_id: null,
        progress_notes: null,
        last_progress_ts: null,
        created_at: 1,
        target_at: null,
        provenance: { kind: "manual" },
      });
      const chosenGoal = choice === "none" ? null : goal.id;
      const { rows } = await extract(f, {
        chosen: null,
        chosenGoal,
        goal: choice === "supplied" ? goal : undefined,
      });
      expect(rows[0]).toMatchObject({
        goal_id: choice === "supplied" ? goal.id : null,
        goal_link_basis: "extractor",
      });
      expect(rows[0]!.goal_id).not.toBe(f.action.goal_id);
      expect(rows[0]!.selection_audit?.goal_rejection_reason).toBe(
        choice === "unoffered" ? "not_offered" : null,
      );
    },
  );

  it("retrieves older other-audience candidates by similarity plus recent rows, without deciding identity", async () => {
    const f = fixture();
    for (let index = 0; index < 90; index += 1)
      f.repository.add(makeActionRecord({ created_at: 1_500, updated_at: 1_500 }));
    const find = vi.spyOn(f.repository, "findByDescription").mockResolvedValue([f.action]);
    const result = await retrieveActionCandidates({
      repository: f.repository,
      text: "Any language",
      audienceEntityId: createEntityId(),
      onSearchFailure: () => {},
    });
    const shown = presentActionCandidates(result.records, result.total);
    expect(find).toHaveBeenCalledOnce();
    expect(shown.shown).toContainEqual(expect.objectContaining({ id: f.action.id }));
    expect(shown.shown.length).toBeLessThanOrEqual(ACTION_CANDIDATE_LIMIT);
    expect(shown.omitted).toBe(91 - shown.shown.length);
    expect(
      shown.rows.every((row) => row.disclosure_label && row.created_at && row.updated_at),
    ).toBe(true);
  });

  it("bounds candidate characters and rejects an id omitted from the actual prompt", async () => {
    const f = fixture();
    const huge = { ...f.action, description: "任務".repeat(ACTION_CANDIDATE_MAX_CHARS) };
    const { rows, llm } = await extract(f, { offered: [huge] });
    const payload = JSON.parse(String(llm.requests[0]!.messages[0]!.content));
    expect(payload.active_actions_for_reference).toEqual([]);
    expect(payload.action_candidate_omitted_count).toBe(12);
    expect(rows[0]?.selection_audit).toMatchObject({
      candidate_ids: [],
      candidate_omitted_count: 12,
      rejection_reason: "not_offered",
    });
  });

  it("measures the bounded candidate block against the previous 80-row shape", () => {
    const rows = Array.from({ length: 80 }, () =>
      makeActionRecord({
        description: "Opaque action wording ".repeat(12),
        actor: "borg",
        audience_entity_id: createEntityId(),
      }),
    );
    const shown = presentActionCandidates(rows, 549);
    const block = JSON.stringify({
      active_actions_for_reference: shown.rows,
      action_candidate_omitted_count: shown.omitted,
    });
    // Old shape was identical except for the two stamps, and admitted all 80.
    const unbounded = presentActionCandidates(rows.slice(0, 1), 1).rows[0]!;
    const oldOne = { ...unbounded };
    delete oldOne.created_at;
    delete oldOne.updated_at;
    const oldChars = JSON.stringify({
      active_actions_for_reference: rows.map(() => oldOne),
    }).length;
    expect(block.length).toBeLessThanOrEqual(ACTION_CANDIDATE_MAX_CHARS);
    expect(shown.omitted).toBe(549 - shown.shown.length);
    console.info(
      `Action candidate fixture: old 80-row block=${oldChars} chars; new block=${block.length} chars; shown=${shown.shown.length}; omitted=${shown.omitted}; cap=${ACTION_CANDIDATE_MAX_CHARS}.`,
    );
  });

  it("prioritizes every active state over a completed flood in the 256-record draw", () => {
    const f = fixture();
    const open = ACTIVE_ACTION_STATES.map((state) =>
      makeActionRecord({ state, updated_at: 1_000, created_at: 1_000 }),
    );
    for (const record of open) f.repository.add(record);
    for (let index = 0; index < 270; index += 1)
      f.repository.add(
        makeActionRecord({ state: "completed", updated_at: 2_000, created_at: 2_000 }),
      );
    const draw = f.repository.list({
      recallAllAudiences: true,
      prioritizeActive: true,
      limit: 256,
    });
    expect(draw).toHaveLength(256);
    expect(
      draw.slice(0, open.length + 1).every((row) => ACTIVE_ACTION_STATES.includes(row.state)),
    ).toBe(true);
  });
  it.each([false, true])(
    "validates every item before persistence; repaired=%s",
    async (repaired) => {
      const f = fixture();
      const before = f.repository.get(f.action.id);
      const { llm, events } = await extract(f, {
        invalidSibling: true,
        repaired,
        references: [f.action.id],
      });
      expect(llm.requests).toHaveLength(2);
      expect(f.repository.readUpdateHistory(f.action.id, 5).total).toBe(repaired ? 1 : 0);
      if (!repaired) {
        expect(f.repository.get(f.action.id)).toEqual(before);
        expect(f.db.prepare("SELECT * FROM action_extraction_decisions").all()).toEqual([]);
        expect(events).toContainEqual({
          event: "extraction.actions.completed",
          data: expect.objectContaining({ fatal_reason: "invalid_payload", degraded: true }),
        });
      }
      expect(
        events.filter((event) => event.event === "extraction.actions.selection_summary"),
      ).toHaveLength(1);
    },
  );

  it("audits reference-only decisions and keeps their counters and timestamps monotonic", async () => {
    const f = fixture();
    const unoffered = makeActionRecord();
    f.repository.add(unoffered);
    for (const [nowMs, turnCounter] of [
      [3_000, 30],
      [2_000, 20],
    ])
      await extract(f, {
        referenceOnly: true,
        references: [f.action.id, unoffered.id],
        nowMs,
        turnCounter,
      });
    expect(f.repository.get(f.action.id)).toMatchObject({
      updated_at: 3_000,
      last_referenced_at_ms: 3_000,
      last_referenced_turn_counter: 30,
      last_referenced_turn_global: 30,
    });
    expect(f.repository.get(unoffered.id)).toMatchObject({ updated_at: unoffered.updated_at });
    expect(f.repository.readUpdateHistory(f.action.id, 5).total).toBe(0);
    const audits = f.db
      .prepare("SELECT audit FROM action_extraction_decisions ORDER BY id")
      .all()
      .map((row) => JSON.parse(String(row.audit)));
    expect(audits).toHaveLength(2);
    for (const audit of audits)
      expect(audit).toMatchObject({
        referenced_action_ids: [f.action.id, unoffered.id],
        applied_reference_ids: [f.action.id],
        ignored_reference_ids: [unoffered.id],
        decisions: [],
      });
  });

  it("rolls reference touches and their audit back with a failed history insert", async () => {
    const f = fixture();
    const before = f.repository.get(f.action.id);
    f.db.exec(
      "CREATE TRIGGER fail_history BEFORE INSERT ON action_record_updates BEGIN SELECT RAISE(ABORT, 'history failed'); END;",
    );
    await extract(f, { references: [f.action.id], turnCounter: 20 });
    expect(f.repository.get(f.action.id)).toEqual(before);
    expect(f.db.prepare("SELECT * FROM action_extraction_decisions").all()).toEqual([]);
  });

  it("rejects the later finisher after a competing terminal decision wins", async () => {
    const f = fixture();
    await extract(f, {
      state: "completed",
      nowMs: 3_000,
      turnCounter: 30,
      references: [f.action.id],
    });
    // Both calls see the same offered snapshot, but the second finishes after closure.
    const { rows } = await extract(f, {
      state: "scheduled",
      nowMs: 2_000,
      turnCounter: 20,
      references: [f.action.id],
    });
    expect(f.repository.get(f.action.id)).toMatchObject({
      state: "completed",
      description: f.action.description,
      last_referenced_at_ms: 3_000,
      last_referenced_turn_counter: 30,
      last_referenced_turn_global: 30,
    });
    expect(f.repository.readUpdateHistory(f.action.id, 5).total).toBe(1);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.selection_audit).toMatchObject({
      chosen_action_id: f.action.id,
      rejection_reason: "terminal",
    });
    expect(f.db.prepare("SELECT * FROM action_extraction_decisions").all()).toHaveLength(2);
  });

  it("keeps both out-of-order non-terminal updates with last-finisher state and monotonic references", async () => {
    const f = fixture();
    await extract(f, {
      state: "considering",
      nowMs: 3_000,
      turnCounter: 30,
      references: [f.action.id],
    });
    await extract(f, {
      state: "scheduled",
      nowMs: 2_000,
      turnCounter: 20,
      references: [f.action.id],
    });
    expect(f.repository.get(f.action.id)).toMatchObject({
      state: "scheduled",
      last_referenced_at_ms: 3_000,
      last_referenced_turn_counter: 30,
      last_referenced_turn_global: 30,
    });
    const history = f.repository.readUpdateHistory(f.action.id, 5);
    expect(history.total).toBe(2);
    expect(history.updates.map((update) => update.to_state).sort()).toEqual([
      "considering",
      "scheduled",
    ]);
    expect(
      history.updates.every(
        (update) => update.selection_audit.applied_reference_ids?.[0] === f.action.id,
      ),
    ).toBe(true);
    const calls = f.db
      .prepare("SELECT audit FROM action_extraction_decisions ORDER BY id")
      .all()
      .map((row) => JSON.parse(String(row.audit)));
    expect(calls).toHaveLength(2);
    expect(calls[1]).toMatchObject({
      referenced_action_ids: [f.action.id],
      applied_reference_ids: [f.action.id],
    });
  });

  it("reserves Borg and participant actors before candidate and 256-record truncation", async () => {
    const f = fixture();
    const speaker = createEntityId(),
      participant = createEntityId();
    const own = makeActionRecord({ actor: "borg", updated_at: 1_000 });
    const other = makeActionRecord({ actor: participant, updated_at: 1_000 });
    f.repository.add(own);
    f.repository.add(other);
    for (let i = 0; i < 300; i++)
      f.repository.add(
        makeActionRecord({ actor: speaker, audience_entity_id: speaker, updated_at: 2_000 }),
      );
    const retrieval = await retrieveActionCandidates({
      repository: f.repository,
      text: "opaque",
      audienceEntityId: speaker,
      speakerEntityId: speaker,
      participantEntityIds: [participant],
      onSearchFailure: () => {},
    });
    const shown = presentActionCandidates(retrieval.records, retrieval.total).shown;
    expect(shown.map((row) => row.id)).toContain(own.id);
    expect(shown.map((row) => row.id)).toContain(other.id);
    const draw = f.repository.list({
      recallAllAudiences: true,
      prioritizeActive: true,
      reserveActors: ["borg", speaker, participant],
      rankAudienceEntityIds: [speaker],
      rankActorEntityIds: [speaker],
      limit: 256,
    });
    expect(draw).toHaveLength(256);
    expect(draw.map((row) => row.id)).toContain(own.id);
    expect(draw.map((row) => row.id)).toContain(other.id);
  });
});
