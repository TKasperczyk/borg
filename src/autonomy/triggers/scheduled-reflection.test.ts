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
      intervalMs: 60_000,
      clock,
    });

    const firstScan = await trigger.scan();
    expect(firstScan).toHaveLength(1);
    expect(firstScan[0]?.id).toBe("scheduled-reflection:1000000");

    clock.advance(30_000);
    const sameWindowScan = await trigger.scan();
    expect(sameWindowScan).toHaveLength(1);
    expect(sameWindowScan[0]?.id).toBe(firstScan[0]?.id);

    watermarkRepository.set(firstScan[0]!.watermarkProcessName, "default", {
      lastTs: clock.now(),
      lastEntryId: null,
    });
    expect(await trigger.scan()).toEqual([]);

    clock.advance(60_001);
    const nextWindowScan = await trigger.scan();
    expect(nextWindowScan).toHaveLength(1);
    expect(nextWindowScan[0]?.id).not.toBe(firstScan[0]?.id);
  });
});
