import { afterEach, describe, expect, it } from "vitest";

import { createOfflineTestHarness } from "../../offline/test-support.js";
import { StreamWatermarkRepository } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { formatAutonomyTriggerContext } from "../../cognition/autonomy-trigger.js";

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
      directiveFamily: "keep_active",
      directive: "Keep this active",
      priority: 1,
      provenance: { kind: "manual" },
    });
    const revoked = harness.commitmentRepository.add({
      type: "boundary",
      directiveFamily: "stop_oversharing",
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
    expect(firstScan[0]?.stateTs).toBe(clock.now());
    expect(firstScan[0]?.payload).toMatchObject({
      commitment_id: revoked.id,
      directive: "Stop oversharing",
      reason: "The premise changed",
    });
    expect(firstScan[0]?.payload.commitment_id).not.toBe(active.id);

    watermarkRepository.set(firstScan[0]!.watermarkProcessName, "default" as never, {
      lastTs: clock.now(),
      lastEntryId: "watermark",
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

  it("renders revoked commitment disclosure labels in the autonomy payload", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const sam = harness.entityRepository.resolve("Sam");
    const revoked = harness.commitmentRepository.add({
      type: "boundary",
      directiveFamily: "sam_private_boundary",
      directive: "Keep Sam's planning details private",
      priority: 4,
      restrictedAudience: sam,
      provenance: { kind: "manual" },
    });
    harness.commitmentRepository.revoke(
      revoked.id,
      "The boundary was replaced",
      { kind: "manual" },
      clock.now(),
    );
    const condition = createCommitmentRevokedCondition({
      commitmentRepository: harness.commitmentRepository,
      watermarkRepository,
      clock,
    });

    const events = await condition.scan();
    const turn = condition.buildTurn(events[0]!);
    const rendered = formatAutonomyTriggerContext(turn.autonomyTrigger!);

    expect(events[0]?.payload).toMatchObject({
      commitment_id: revoked.id,
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [sam],
      },
    });
    expect(rendered).toContain("disclosure_label");
    expect(rendered).toContain("relationship_private");
    expect(rendered).toContain(sam);
  });
});
