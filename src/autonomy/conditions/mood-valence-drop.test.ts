import { afterEach, describe, expect, it } from "vitest";

import { createOfflineTestHarness } from "../../offline/test-support.js";
import { StreamWatermarkRepository } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";

import { createMoodValenceDropCondition } from "./mood-valence-drop.js";

describe("mood valence drop condition", () => {
  let cleanup: (() => Promise<void>) | undefined;

  afterEach(async () => {
    await cleanup?.();
    cleanup = undefined;
  });

  it("requires the full window and does not re-fire within the activation period", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const condition = createMoodValenceDropCondition({
      moodRepository: harness.moodRepository,
      watermarkRepository,
      threshold: -0.5,
      windowN: 3,
      activationPeriodMs: 10_000,
      clock,
    });

    harness.moodRepository.update("default" as never, {
      valence: -0.8,
      arousal: 0.4,
      provenance: { kind: "system" },
    });
    clock.advance(1_000);
    harness.moodRepository.update("default" as never, {
      valence: -0.7,
      arousal: 0.3,
      provenance: { kind: "system" },
    });

    expect(await condition.scan()).toEqual([]);

    clock.advance(1_000);
    harness.moodRepository.update("default" as never, {
      valence: -0.9,
      arousal: 0.5,
      provenance: { kind: "system" },
    });

    const firstScan = await condition.scan();
    expect(firstScan).toHaveLength(1);
    expect(firstScan[0]?.payload.average_valence).toBeLessThan(-0.5);

    watermarkRepository.set(firstScan[0]!.watermarkProcessName, "default" as never, {
      lastTs: clock.now(),
      lastEntryId: "watermark",
    });
    expect(await condition.scan()).toEqual([]);

    clock.advance(10_001);
    const secondScan = await condition.scan();
    expect(secondScan).toHaveLength(1);
  });
});
