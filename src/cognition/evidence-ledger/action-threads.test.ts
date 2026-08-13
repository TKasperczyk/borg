import { describe, expect, it } from "vitest";

import { makeCompletedActionRecord } from "../../test-support/factories/memory.js";
import { createEntityId } from "../../util/ids.js";
import {
  allocateActionThreadRenderSlots,
  type ActionThreadWithSalience,
} from "./action-threads.js";

const DOMINANT_AUDIENCE = createEntityId();
const QUIET_AUDIENCE_A = createEntityId();
const QUIET_AUDIENCE_B = createEntityId();

function makeThread(input: {
  audienceEntityId: ReturnType<typeof createEntityId>;
  updatedAt: number;
  salienceClass?: ActionThreadWithSalience["salienceClass"];
}): ActionThreadWithSalience {
  const record = makeCompletedActionRecord({
    audience_entity_id: input.audienceEntityId,
    updated_at: input.updatedAt,
  });

  return {
    id: record.id,
    records: [record],
    origin: record,
    current: record,
    scope: "current_session",
    salienceClass: input.salienceClass ?? "completed_recent",
  };
}

describe("allocateActionThreadRenderSlots", () => {
  // Production runs 1/1 (config default). Without the audience reservation a single
  // busy room's recency wins every slot in its class, so the render collapses onto
  // that one audience even when other audiences have qualifying threads.
  it("gives each audience a slot the plain recency draw would deny it", () => {
    const threads: ActionThreadWithSalience[] = [
      ...Array.from({ length: 10 }, (_unused, index) =>
        makeThread({ audienceEntityId: DOMINANT_AUDIENCE, updatedAt: 2_000 - index }),
      ),
      makeThread({ audienceEntityId: QUIET_AUDIENCE_A, updatedAt: 1_000 }),
      makeThread({ audienceEntityId: QUIET_AUDIENCE_B, updatedAt: 500 }),
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

    expect(unreserved).toHaveLength(8);
    expect(audiencesOf(unreserved)).toEqual([DOMINANT_AUDIENCE]);

    expect(reserved).toHaveLength(8);
    expect(audiencesOf(reserved).sort()).toEqual(
      [DOMINANT_AUDIENCE, QUIET_AUDIENCE_A, QUIET_AUDIENCE_B].sort(),
    );
    // The reservation costs the dominant audience its two oldest selected threads,
    // never its most recent ones.
    expect(
      reserved
        .filter((thread) => thread.current.audience_entity_id === DOMINANT_AUDIENCE)
        .map((thread) => thread.current.updated_at),
    ).toEqual([2_000, 1_999, 1_998, 1_997, 1_996, 1_995]);
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
