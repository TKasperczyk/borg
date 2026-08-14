import { describe, expect, it } from "vitest";

import {
  makeActionRecord,
  makeCompletedActionRecord,
} from "../../test-support/factories/memory.js";
import { createEntityId } from "../../util/ids.js";
import {
  actionSalienceClass,
  allocateActionThreadRenderSlots,
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
