import { afterEach, describe, expect, it } from "vitest";

import { StreamWatermarkRepository } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { createOfflineTestHarness } from "../../offline/test-support.js";
import { formatAutonomyTriggerContext } from "../../cognition/autonomy-trigger.js";

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

  it("renders expiring commitment disclosure labels in the autonomy payload", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const alex = harness.entityRepository.resolve("Alex");
    const dueCommitment = harness.commitmentRepository.add({
      type: "boundary",
      directiveFamily: "alex_private_boundary",
      directive: "Keep Alex planning details scoped to Alex",
      priority: 10,
      restrictedAudience: alex,
      provenance: { kind: "manual" },
      expiresAt: clock.now() + 10_000,
    });

    const trigger = createCommitmentExpiringTrigger({
      commitmentRepository: harness.commitmentRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      clock,
    });

    const events = await trigger.scan();
    const turn = trigger.buildTurn(events[0]!);
    const rendered = formatAutonomyTriggerContext(turn.autonomyTrigger!);

    expect(events[0]?.payload).toMatchObject({
      commitment_id: dueCommitment.id,
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [alex],
      },
    });
    expect(rendered).toContain("disclosure_label");
    expect(rendered).toContain("relationship_private");
    expect(rendered).toContain(alex);
  });
});
