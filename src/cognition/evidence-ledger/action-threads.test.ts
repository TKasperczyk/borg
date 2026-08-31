import { describe, expect, it } from "vitest";

import {
  makeActionRecord,
  makeCompletedActionRecord,
} from "../../test-support/factories/memory.js";
import { createEntityId } from "../../util/ids.js";
import {
  MEMORY_DISCLOSURE_INTERNAL_USE_NOTE,
  publicMemoryDisclosureLabel,
  selfPrivateMemoryDisclosureLabel,
} from "../../retrieval/index.js";
import {
  actionSalienceClass,
  allocateActionThreadRenderSlots,
  renderOlderActionThreadsSummary,
  type ActionThread,
  type ActionThreadWithSalience,
} from "./action-threads.js";

const DOMINANT_AUDIENCE = createEntityId();
const QUIET_AUDIENCE_A = createEntityId();
const QUIET_AUDIENCE_B = createEntityId();

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
    staleParticipantThreadsWithheldCount: 0,
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

  // `rendered` is the allocator's output over a pool, not the render limit. A stale-participant
  // cap withholds threads before the slot draw, so the count can fall below the limit without the
  // limit ever binding -- and with neither operand printed, the two cases looked identical.
  it("prints the slot limit and the pool withheld before it, so a short rendered count decomposes", () => {
    const summary = renderOlderActionThreadsSummary({
      ...summaryInput({ consideredRecordCount: 256, sourceRecordLimit: 256 }),
      renderedThreadCount: 8,
      renderLimit: 12,
      staleParticipantThreadsWithheldCount: 1,
      threadsBuiltCount: 214,
      salienceDroppedThreadCount: 205,
    });

    // The withheld thread is the omitted one: it stays inside the identity rather than opening a
    // fourth term, and rendered = 214 - 205 - 1 = 8 reads off the printed operands, below the
    // twelve slots without twelve ever having been the binding constraint.
    expect(summary).toContain("threads_built=214 = rendered 8 + omitted 1 + dropped 205");
    expect(summary).toContain(
      "render_slots=12, stale_participant_threads_withheld=1 of that omitted count, " +
        "held out before the slot draw, so rendered can sit below render_slots without " +
        "render_slots binding",
    );
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
