import { afterEach, describe, expect, it } from "vitest";

import { createOfflineTestHarness } from "../../offline/test-support.js";
import { StreamWatermarkRepository } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";

import { createCommitmentRevokedCondition } from "./commitment-revoked.js";

describe("commitment revoked condition", () => {
  let cleanup: (() => Promise<void>) | undefined;

  afterEach(async () => {
    await cleanup?.();
    cleanup = undefined;
  });

  it("fires once per revocation and ignores active commitments", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    const active = harness.commitmentRepository.add({
      type: "promise",
      directive: "Keep this active",
      priority: 1,
      provenance: { kind: "manual" },
    });
    const revoked = harness.commitmentRepository.add({
      type: "boundary",
      directive: "Stop oversharing",
      priority: 4,
      provenance: { kind: "manual" },
    });
    harness.commitmentRepository.revoke(
      revoked.id,
      "The premise changed",
      { kind: "manual" },
      clock.now(),
    );
    const condition = createCommitmentRevokedCondition({
      commitmentRepository: harness.commitmentRepository,
      watermarkRepository,
      clock,
    });

    const firstScan = await condition.scan();
    expect(firstScan).toHaveLength(1);
    expect(firstScan[0]?.payload).toMatchObject({
      commitment_id: revoked.id,
      directive: "Stop oversharing",
      reason: "The premise changed",
    });
    expect(firstScan[0]?.payload.commitment_id).not.toBe(active.id);

    watermarkRepository.set(firstScan[0]!.watermarkProcessName, "default", {
      lastTs: clock.now(),
      lastEntryId: null,
    });
    expect(await condition.scan()).toEqual([]);

    clock.advance(1_000);
    harness.commitmentRepository.revoke(
      revoked.id,
      "The context changed again",
      { kind: "manual" },
      clock.now(),
    );

    const secondScan = await condition.scan();
    expect(secondScan).toHaveLength(1);
    expect(secondScan[0]?.payload.reason).toBe("The context changed again");
  });
});
