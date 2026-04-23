import { afterEach, describe, expect, it } from "vitest";

import { StreamWatermarkRepository } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { createOfflineTestHarness } from "../../offline/test-support.js";

import { createScheduledReflectionTrigger } from "./scheduled-reflection.js";

describe("scheduled reflection trigger", () => {
  let cleanup: (() => Promise<void>) | undefined;

  afterEach(async () => {
    await cleanup?.();
    cleanup = undefined;
  });

  it("fires on its cadence using watermarks", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });

    const firstScan = await trigger.scan();
    expect(firstScan).toHaveLength(1);

    watermarkRepository.set(firstScan[0]!.watermarkProcessName, "default", {
      lastTs: clock.now(),
      lastEntryId: null,
    });
    expect(await trigger.scan()).toEqual([]);

    clock.advance(10_001);
    expect(await trigger.scan()).toHaveLength(1);
  });
});
