import { afterEach, describe, expect, it } from "vitest";

import { StreamWatermarkRepository } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { createOfflineTestHarness } from "../../offline/test-support.js";

import { createCommitmentExpiringTrigger } from "./commitment-expiring.js";

describe("commitment expiring trigger", () => {
  let cleanup: (() => Promise<void>) | undefined;

  afterEach(async () => {
    await cleanup?.();
    cleanup = undefined;
  });

  it("finds commitments expiring inside the lookahead window and dedupes fired ones", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    const dueCommitment = harness.commitmentRepository.add({
      type: "promise",
      directiveFamily: "autonomy_design_review_response",
      directive: "Respond to the autonomy design review",
      priority: 8,
      provenance: { kind: "manual" },
      expiresAt: clock.now() + 10_000,
    });
    harness.commitmentRepository.add({
      type: "promise",
      directiveFamily: "far_future_commitment",
      directive: "Far future commitment",
      priority: 2,
      provenance: { kind: "manual" },
      expiresAt: clock.now() + 200_000,
    });

    const trigger = createCommitmentExpiringTrigger({
      commitmentRepository: harness.commitmentRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      clock,
    });

    const firstScan = await trigger.scan();
    expect(firstScan.map((event) => event.payload.commitment_id)).toEqual([dueCommitment.id]);

    watermarkRepository.set(firstScan[0]!.watermarkProcessName, "default" as never, {
      lastTs: clock.now(),
      lastEntryId: "watermark",
    });

    expect(await trigger.scan()).toEqual([]);
  });
});
