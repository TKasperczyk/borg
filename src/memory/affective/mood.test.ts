import { afterEach, describe, expect, it } from "vitest";

import { ManualClock } from "../../util/clock.js";
import { ProvenanceError } from "../../util/errors.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { createOfflineTestHarness } from "../../offline/test-support.js";

describe("MoodRepository", () => {
  const systemProvenance = { kind: "system" } as const;

  let harness: Awaited<ReturnType<typeof createOfflineTestHarness>> | undefined;

  afterEach(async () => {
    await harness?.cleanup();
    harness = undefined;
  });

  it("decays on read without mutating persisted state and appends history on update", async () => {
    const clock = new ManualClock(1_000_000);
    harness = await createOfflineTestHarness({
      clock,
      configOverrides: {
        affective: {
          moodHalfLifeHours: 1,
          incomingMoodWeight: 0.5,
          useLlmFallback: false,
          moodHistoryRetentionDays: 90,
        },
      } as never,
    });

    const initial = harness.moodRepository.update(DEFAULT_SESSION_ID, {
      valence: -0.8,
      arousal: 0.6,
      reason: "frustrated turn",
      provenance: systemProvenance,
    });

    expect(initial.valence).toBeCloseTo(-0.4, 3);
    clock.set(1_000_000 + 60 * 60 * 1_000);
    const decayed = harness.moodRepository.current(DEFAULT_SESSION_ID);
    const stored = harness.moodRepository.listStoredStates()[0];

    expect(decayed.valence).toBeCloseTo(stored?.valence ? stored.valence / 2 : 0, 2);
    expect(stored?.updated_at).toBe(1_000_000);
    expect(harness.moodRepository.history(DEFAULT_SESSION_ID)).toHaveLength(1);
  });

  it("rejects provenance-less mood updates", async () => {
    harness = await createOfflineTestHarness();

    expect(() =>
      harness!.moodRepository.update(DEFAULT_SESSION_ID, {
        valence: 0.1,
        arousal: 0.2,
        provenance: undefined as never,
      }),
    ).toThrow(ProvenanceError);
  });
});
