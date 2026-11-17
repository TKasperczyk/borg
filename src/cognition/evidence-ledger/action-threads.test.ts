import { describe, expect, it, vi } from "vitest";

import {
  makeActionRecord,
  makeCompletedActionRecord,
} from "../../test-support/factories/memory.js";
import { DEFAULT_SESSION_ID, createEntityId, createGoalId } from "../../util/ids.js";
import {
  MEMORY_DISCLOSURE_INTERNAL_USE_NOTE,
  publicMemoryDisclosureLabel,
  selfPrivateMemoryDisclosureLabel,
} from "../../retrieval/index.js";
import type { ActionRecord } from "../../memory/actions/index.js";
import {
  actionSalienceClass,
  allocateActionThreadRenderSlots,
  buildActionThreads,
  renderActionThreadText,
  renderOlderActionThreadsSummary,
  type ActionThread,
  type ActionThreadWithSalience,
} from "./action-threads.js";

const DOMINANT_AUDIENCE = createEntityId();
const QUIET_AUDIENCE_A = createEntityId();
const QUIET_AUDIENCE_B = createEntityId();

it("renders declined as a terminal choice with terminal salience", () => {
  const record = makeActionRecord({
    state: "declined",
    declined_at: 2_000,
    last_referenced_turn_global: 10,
  });
  const thread: ActionThread = {
    id: record.id,
    records: [record],
    origin: record,
    current: record,
    scope: "current_session",
  };
  expect(renderActionThreadText(thread, undefined)).toContain("declined");
  expect(renderActionThreadText(thread, undefined)).toContain("choice to decline");
  expect(actionSalienceClass({ thread, currentTurnGlobal: 10 })).toBe("completed_recent");
  expect(actionSalienceClass({ thread, currentTurnGlobal: 1_000 })).toBeNull();
});

function makeCompletedThread(input: {
  audienceEntityId: ReturnType<typeof createEntityId>;
  updatedAt: number;
  lastReferencedTurnCounter?: number;
  lastReferencedTurnGlobal?: number;
}): ActionThread {
  const record = makeCompletedActionRecord({
    audience_entity_id: input.audienceEntityId,
    updated_at: input.updatedAt,
    last_referenced_turn_counter: input.lastReferencedTurnCounter ?? null,
    last_referenced_turn_global: input.lastReferencedTurnGlobal ?? null,
  });

  return {
    id: record.id,
    records: [record],
    origin: record,
    current: record,
    scope: "current_session",
  };
}

function makeThread(input: {
  audienceEntityId: ReturnType<typeof createEntityId>;
  updatedAt: number;
  salienceClass?: ActionThreadWithSalience["salienceClass"];
}): ActionThreadWithSalience {
  return {
    ...makeCompletedThread(input),
    salienceClass: input.salienceClass ?? "completed_recent",
  };
}

function classifyAtGlobalTurn(
  thread: ActionThread,
  currentTurnGlobal: number,
): ActionThreadWithSalience[] {
  const salienceClass = actionSalienceClass({ thread, currentTurnGlobal });

  return salienceClass === null ? [] : [{ ...thread, salienceClass }];
}

describe("actionSalienceClass", () => {
  it("compares the global reference stamp to the current global turn", () => {
    const stampedTurnGlobal = 4_800;
    const thread = makeCompletedThread({
      audienceEntityId: DOMINANT_AUDIENCE,
      updatedAt: 2_000,
      // The dedicated field remains global even when a legacy session stamp is present.
      lastReferencedTurnCounter: 67,
      lastReferencedTurnGlobal: stampedTurnGlobal,
    });

    expect(actionSalienceClass({ thread, currentTurnGlobal: stampedTurnGlobal + 4 })).toBeNull();
    expect(actionSalienceClass({ thread, currentTurnGlobal: stampedTurnGlobal + 2 })).toBe(
      "completed_recent",
    );
  });

  it("drops archived threads on state, not on the recency window", () => {
    const stampedTurnGlobal = 4_800;
    const archived = makeActionRecord({
      state: "archived",
      audience_entity_id: DOMINANT_AUDIENCE,
      updated_at: 2_000,
      last_referenced_turn_global: stampedTurnGlobal,
    });
    const archivedThread: ActionThread = {
      id: archived.id,
      records: [archived],
      origin: archived,
      current: archived,
      scope: "current_session",
    };

    // Same stamp, same current turn: the window admits the completed thread and the archived
    // thread still classifies null, so widening the window can never surface it.
    expect(
      actionSalienceClass({ thread: archivedThread, currentTurnGlobal: stampedTurnGlobal }),
    ).toBeNull();
    expect(
      actionSalienceClass({
        thread: makeCompletedThread({
          audienceEntityId: DOMINANT_AUDIENCE,
          updatedAt: 2_000,
          lastReferencedTurnGlobal: stampedTurnGlobal,
        }),
        currentTurnGlobal: stampedTurnGlobal,
      }),
    ).toBe("completed_recent");
  });
});

describe("allocateActionThreadRenderSlots", () => {
  // Once lifecycle recency uses one global clock, the genuinely recent completion
  // pool is thin enough that both allocation modes preserve every audience.
  it("needs no audience rescue when the global completion window binds", () => {
    const currentTurnGlobal = 2_000;
    const threads: ActionThreadWithSalience[] = [
      ...Array.from({ length: 10 }, (_unused, index) => {
        const referencedTurnGlobal = 2_000 - index;

        return classifyAtGlobalTurn(
          makeCompletedThread({
            audienceEntityId: DOMINANT_AUDIENCE,
            updatedAt: referencedTurnGlobal,
            lastReferencedTurnCounter: referencedTurnGlobal,
            lastReferencedTurnGlobal: referencedTurnGlobal,
          }),
          currentTurnGlobal,
        );
      }).flat(),
      ...classifyAtGlobalTurn(
        makeCompletedThread({
          audienceEntityId: QUIET_AUDIENCE_A,
          updatedAt: 1_000,
          lastReferencedTurnCounter: 1_999,
          lastReferencedTurnGlobal: 1_999,
        }),
        currentTurnGlobal,
      ),
      ...classifyAtGlobalTurn(
        makeCompletedThread({
          audienceEntityId: QUIET_AUDIENCE_B,
          updatedAt: 500,
          lastReferencedTurnCounter: 1_998,
          lastReferencedTurnGlobal: 1_998,
        }),
        currentTurnGlobal,
      ),
    ];
    const audiencesOf = (selected: readonly ActionThreadWithSalience[]): string[] => [
      ...new Set(selected.map((thread) => thread.current.audience_entity_id ?? "global")),
    ];

    const unreserved = allocateActionThreadRenderSlots({
      threads,
      limit: 8,
      salienceClassReservedSlots: 0,
      audienceReservedSlots: 0,
    });
    const reserved = allocateActionThreadRenderSlots({
      threads,
      limit: 8,
      salienceClassReservedSlots: 1,
      audienceReservedSlots: 1,
    });

    expect(unreserved).toHaveLength(6);
    expect(audiencesOf(unreserved).sort()).toEqual(
      [DOMINANT_AUDIENCE, QUIET_AUDIENCE_A, QUIET_AUDIENCE_B].sort(),
    );

    expect(reserved).toHaveLength(6);
    expect(audiencesOf(reserved).sort()).toEqual(
      [DOMINANT_AUDIENCE, QUIET_AUDIENCE_A, QUIET_AUDIENCE_B].sort(),
    );
    expect(reserved.map((thread) => thread.id).sort()).toEqual(
      unreserved.map((thread) => thread.id).sort(),
    );
    expect(
      reserved
        .filter((thread) => thread.current.audience_entity_id === DOMINANT_AUDIENCE)
        .map((thread) => thread.current.updated_at),
    ).toEqual([2_000, 1_999, 1_998, 1_997]);
  });

  // The salience reservation only bites once the higher-ranked classes can fill the
  // limit on their own; below that it re-picks threads the recency draw already had.
  it("only changes the selected set when earlier salience classes fill the limit", () => {
    const headHeavy: ActionThreadWithSalience[] = [
      ...Array.from({ length: 5 }, (_unused, index) =>
        makeThread({
          audienceEntityId: DOMINANT_AUDIENCE,
          updatedAt: 2_000 - index,
          salienceClass: "borg_memory_tracking_action",
        }),
      ),
      ...Array.from({ length: 3 }, (_unused, index) =>
        makeThread({ audienceEntityId: DOMINANT_AUDIENCE, updatedAt: 1_000 - index }),
      ),
    ];
    const headSaturated: ActionThreadWithSalience[] = [
      ...Array.from({ length: 12 }, (_unused, index) =>
        makeThread({
          audienceEntityId: DOMINANT_AUDIENCE,
          updatedAt: 2_000 - index,
          salienceClass: "borg_memory_tracking_action",
        }),
      ),
      makeThread({ audienceEntityId: DOMINANT_AUDIENCE, updatedAt: 1_000 }),
    ];
    const idsOf = (selected: readonly ActionThreadWithSalience[]): string[] =>
      [...selected.map((thread) => thread.id)].sort();
    const allocate = (
      threads: readonly ActionThreadWithSalience[],
      salienceClassReservedSlots: number,
    ): ActionThreadWithSalience[] =>
      allocateActionThreadRenderSlots({
        threads,
        limit: 6,
        salienceClassReservedSlots,
        audienceReservedSlots: 0,
      });

    expect(idsOf(allocate(headHeavy, 1))).toEqual(idsOf(allocate(headHeavy, 0)));

    const saturatedUnreserved = allocate(headSaturated, 0);
    const saturatedReserved = allocate(headSaturated, 1);

    expect(saturatedUnreserved.map((thread) => thread.salienceClass)).toEqual(
      Array.from({ length: 6 }, () => "borg_memory_tracking_action"),
    );
    expect(saturatedReserved.map((thread) => thread.salienceClass)).toEqual([
      ...Array.from({ length: 5 }, () => "borg_memory_tracking_action"),
      "completed_recent",
    ]);
  });
});

describe("renderActionThreadText", () => {
  async function build(records: ActionRecord[]) {
    const findSimilarDescriptionPairs = vi.fn(async () =>
      records
        .slice(1)
        .map((record) => ({ leftId: records[0]!.id, rightId: record.id, similarity: 1 })),
    );
    const threads = await buildActionThreads({
      records,
      repository: { list: () => records, findSimilarDescriptionPairs },
      resolver: {
        currentSessionId: DEFAULT_SESSION_ID,
        streamEntriesById: new Map(),
        streamOrderById: new Map(),
        episodeScopesById: new Map(),
        episodeSourceStreamIdsById: new Map(),
      },
    });
    expect(findSimilarDescriptionPairs).not.toHaveBeenCalled();
    return threads;
  }

  it("renders each separate row's own description and disposition", async () => {
    const open = makeActionRecord({
      description: "commit to filing",
      state: "committed_to_do",
      goal_id: createGoalId(),
    });
    const closed = makeActionRecord({
      description: "filed it",
      state: "completed",
      goal_id: open.goal_id,
    });
    const threads = await build([open, closed]);
    expect(threads).toHaveLength(2);
    expect(renderActionThreadText(threads[0]!, undefined)).toContain(`action: ${open.description}`);
    expect(renderActionThreadText(threads[1]!, undefined)).toContain(
      "current: completed (closed by outcome)",
    );
    expect(
      threads.every(
        (thread) => thread.records.length === 1 && thread.origin.id === thread.current.id,
      ),
    ).toBe(true);
  });

  it("keeps same-goal similar rows separate even when the oldest row was updated last", async () => {
    const goal = createGoalId();
    const older = makeActionRecord({
      goal_id: goal,
      description: "確認を続ける。",
      created_at: 1_000,
      updated_at: 3_000,
    });
    const newer = makeActionRecord({
      goal_id: goal,
      description: older.description,
      created_at: 2_000,
      updated_at: 2_000,
    });
    const threads = await build([newer, older]);
    expect(threads.map((thread) => thread.id)).toEqual([older.id, newer.id]);
    expect(
      threads.map((thread) => renderActionThreadText(thread, undefined)).join("\n"),
    ).not.toMatch(/originating_intent|current_intent|grouped|similarity/);
  });

  it("dates each separate row's state from its own stamp", async () => {
    const closed = makeActionRecord({
      state: "completed",
      created_at: 1_000,
      updated_at: 4_000,
      completed_at: 4_000,
    });
    const scheduled = makeActionRecord({
      state: "scheduled",
      created_at: 9_000,
      updated_at: 9_000,
      scheduled_at: 9_000,
    });
    const threads = await build([closed, scheduled]);
    expect(renderActionThreadText(threads[0]!, undefined)).toContain(
      `current: scheduled (open) since ${new Date(9_000).toISOString()}`,
    );
    expect(renderActionThreadText(threads[1]!, undefined)).toContain(
      `current: completed (closed by outcome) since ${new Date(4_000).toISOString()}`,
    );
  });

  it("names a single-record thread's stamp as a state-entry time rather than a bare at", async () => {
    const record = makeActionRecord({
      description: "requested the restart",
      state: "scheduled",
      created_at: 7_000,
      updated_at: 7_000,
      scheduled_at: 7_000,
    });
    const [thread] = await build([record]);
    const text = renderActionThreadText(thread!, undefined);
    expect(text).toContain(
      `records: 1, current: scheduled (open) since ${new Date(7_000).toISOString()}`,
    );
    expect(text).not.toContain("(open) at ");
    expect(text).toContain(`action: ${record.description}`);
  });

  it("keeps equal-created-at rows separate without inventing an origin", async () => {
    const first = makeActionRecord({ created_at: 1_000, updated_at: 1_000 });
    const second = makeActionRecord({ ...first, id: makeActionRecord().id });
    const threads = await build([first, second]);
    expect(threads).toHaveLength(2);
    for (const thread of threads)
      expect(renderActionThreadText(thread, undefined)).not.toContain("originating");
  });

  it("labels legacy goal links without changing them, and labels extractor-selected links", async () => {
    const legacy = makeActionRecord({ goal_id: createGoalId() });
    const [thread] = await build([legacy]);
    expect(renderActionThreadText(thread!, undefined)).toContain(
      "legacy turn-selected goal, not an action-specific link",
    );
    expect(legacy.goal_id).not.toBeNull();
    thread!.current = { ...legacy, goal_link_basis: "extractor" };
    expect(renderActionThreadText(thread!, undefined)).toContain(
      "goal_link_basis: extractor selected",
    );
  });
});

it("allocates open rows before terminal audience or salience reservations", () => {
  const threads: ActionThreadWithSalience[] = [
    "considering",
    "committed_to_do",
    "scheduled",
    "unknown",
    "completed",
    "declined",
  ].map((state, index) => {
    const current = makeActionRecord({
      state: state as ActionRecord["state"],
      audience_entity_id: createEntityId(),
      updated_at: index + 1,
    });
    return {
      id: current.id,
      current,
      origin: current,
      records: [current],
      scope: "global",
      salienceClass: index < 4 ? "participant_pending_stale" : "completed_recent",
    };
  });
  const selected = allocateActionThreadRenderSlots({
    threads,
    limit: 4,
    salienceClassReservedSlots: 1,
    audienceReservedSlots: 1,
    audienceOrder: [...threads].reverse().map((thread) => thread.current.audience_entity_id!),
  });
  expect(selected.map((thread) => thread.current.state).sort()).toEqual(
    ["considering", "committed_to_do", "scheduled", "unknown"].sort(),
  );
});

describe("renderOlderActionThreadsSummary", () => {
  const summaryInput = (overrides: {
    consideredRecordCount: number;
    sourceRecordLimit: number;
    sourceRecordTotal?: number | null;
  }) => ({
    groups: [
      {
        audienceScope: "global" as const,
        salienceClass: "completed_recent" as const,
        threads: [makeCompletedThread({ audienceEntityId: QUIET_AUDIENCE_A, updatedAt: 10 })],
        disclosureLabel: selfPrivateMemoryDisclosureLabel([QUIET_AUDIENCE_A]),
      },
    ],
    renderedThreadCount: 1,
    renderLimit: 12,
    threadsBuiltCount: 2,
    salienceDroppedThreadCount: 0,
    sourceRecordTotal: 8,
    ...overrides,
  });

  // The floor is a measurement, not an enumeration: it must move with the store rather than name
  // a condition. Two totals over one draw is what separates a live figure from a frozen token --
  // a reader holding a single page can only tell the difference if the field can differ.
  it("counts the records below the draw floor instead of naming the condition", () => {
    const saturated = renderOlderActionThreadsSummary(
      summaryInput({ consideredRecordCount: 256, sourceRecordLimit: 256, sourceRecordTotal: 2706 }),
    );
    const larger = renderOlderActionThreadsSummary(
      summaryInput({ consideredRecordCount: 256, sourceRecordLimit: 256, sourceRecordTotal: 2707 }),
    );
    const exhausted = renderOlderActionThreadsSummary(
      summaryInput({ consideredRecordCount: 4, sourceRecordLimit: 256, sourceRecordTotal: 4 }),
    );

    expect(saturated).toContain("records_below_draw_floor=2450");
    expect(larger).toContain("records_below_draw_floor=2451");
    expect(exhausted).toContain("records_below_draw_floor=0");
    expect(saturated).not.toContain("unknown_count");
  });

  // The floor is the stated difference of two printed totals, never an independent count of the
  // rows below it. Printing it bare would let a derived figure pose as its own witness.
  it("states the floor's derivation beside it", () => {
    const summary = renderOlderActionThreadsSummary(
      summaryInput({ consideredRecordCount: 256, sourceRecordLimit: 256, sourceRecordTotal: 2706 }),
    );

    expect(summary).toContain(
      "records_considered=256 records, source_record_limit=256, source_record_total=2706; " +
        "records_below_draw_floor is that total minus records_considered, not a separate count",
    );
  });

  it("prints the slot limit without the removed stale-participant cap", () => {
    const summary = renderOlderActionThreadsSummary({
      ...summaryInput({ consideredRecordCount: 256, sourceRecordLimit: 256 }),
      renderedThreadCount: 8,
      renderLimit: 12,
      threadsBuiltCount: 214,
      salienceDroppedThreadCount: 205,
    });

    expect(summary).toContain("threads_built=214 = rendered 8 + omitted 1 + dropped 205");
    expect(summary).toContain("render_slots=12;");
    expect(summary).not.toContain("stale_participant_threads_withheld");
  });

  // A repository that cannot count says so. Falling back to 0 would claim the draw saw everything.
  it("names an unavailable source total instead of implying an empty floor", () => {
    const summary = renderOlderActionThreadsSummary(
      summaryInput({ consideredRecordCount: 256, sourceRecordLimit: 256, sourceRecordTotal: null }),
    );

    expect(summary).toContain("records_below_draw_floor=unknown_count_source_total_unavailable");
    expect(summary).toContain("source_record_total=unavailable");
  });

  // The internal-use sentence is byte-identical on every non-public label, so it is stated once
  // for the section instead of once per group: this section truncates from the tail, and the
  // copies were spending the space the surviving group lines need.
  it("states the internal-use note once for the section, not per group", () => {
    const summary = renderOlderActionThreadsSummary(
      summaryInput({ consideredRecordCount: 4, sourceRecordLimit: 256 }),
    );
    const noteCount = summary.split(MEMORY_DISCLOSURE_INTERNAL_USE_NOTE).length - 1;
    const groupLine = summary.split("\n").find((line) => line.startsWith("- audience_scope="));

    expect(noteCount).toBe(1);
    expect(summary.indexOf(MEMORY_DISCLOSURE_INTERNAL_USE_NOTE)).toBeLessThan(
      summary.indexOf("- audience_scope="),
    );
    // What varies per group -- the class and the private-to binding -- stays on the group line.
    expect(groupLine).toContain(
      `disclosure_label=disclosure_class=self_private origin_audience=${QUIET_AUDIENCE_A} private-to=${QUIET_AUDIENCE_A} recent_samples=`,
    );
  });

  it("omits the note when no group is non-public", () => {
    const summary = renderOlderActionThreadsSummary({
      ...summaryInput({ consideredRecordCount: 4, sourceRecordLimit: 256 }),
      groups: [
        {
          audienceScope: "global" as const,
          salienceClass: "completed_recent" as const,
          threads: [makeCompletedThread({ audienceEntityId: QUIET_AUDIENCE_A, updatedAt: 10 })],
          disclosureLabel: publicMemoryDisclosureLabel(),
        },
      ],
    });

    expect(summary).not.toContain(MEMORY_DISCLOSURE_INTERNAL_USE_NOTE);
    expect(summary).toContain("disclosure_label=disclosure_class=public recent_samples=");
  });
});

it.each(["borg", "speaker"] as const)(
  "reserves open Borg and speaker slots when %s owns the newest rows",
  (dominant) => {
    const speaker = createEntityId();
    const actors =
      dominant === "borg" ? (["borg", speaker] as const) : ([speaker, "borg"] as const);
    const records = [
      ...Array.from({ length: 30 }, (_, i) =>
        makeActionRecord({ actor: actors[0], updated_at: 1_000 + i }),
      ),
      makeActionRecord({ actor: actors[1], updated_at: 1 }),
    ];
    const threads: ActionThreadWithSalience[] = records.map((current) => ({
      id: current.id,
      current,
      origin: current,
      records: [current],
      scope: "global",
      salienceClass: "participant_pending_stale",
    }));
    const selected = allocateActionThreadRenderSlots({
      threads,
      limit: 12,
      salienceClassReservedSlots: 1,
      audienceReservedSlots: 1,
      audienceOrder: [],
      reservedActors: ["borg", speaker],
    });
    expect(selected).toHaveLength(12);
    expect(new Set(selected.map((thread) => thread.current.actor))).toEqual(
      new Set(["borg", speaker]),
    );
  },
);
