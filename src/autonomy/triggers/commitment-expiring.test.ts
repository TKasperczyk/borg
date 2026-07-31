import { readFileSync } from "node:fs";
import { join } from "node:path";

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
    expect(firstScan[0]?.stateTs).toBe(dueCommitment.updated_at);

    watermarkRepository.set(firstScan[0]!.watermarkProcessName, "default" as never, {
      lastTs: clock.now(),
      lastEntryId: "watermark",
    });

    expect(await trigger.scan()).toEqual([]);
  });

  it("describes the next lookahead boundary for active expiring commitments", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    const commitment = harness.commitmentRepository.add({
      type: "promise",
      directiveFamily: "future_commitment",
      directive: "Handle the future commitment",
      priority: 4,
      provenance: { kind: "manual" },
      expiresAt: clock.now() + 120_000,
    });
    const trigger = createCommitmentExpiringTrigger({
      commitmentRepository: harness.commitmentRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      clock,
    });

    await expect(trigger.nextDueAt!()).resolves.toBe(commitment.expires_at! - 20_000 + 1);

    clock.advance(110_000);
    await expect(trigger.nextDueAt!()).resolves.toBe(clock.now());
  });

  it("retains overdue unresolved commitments until their state-tuple latch fires", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    const expired = harness.commitmentRepository.add({
      type: "promise",
      directiveFamily: "expired_commitment",
      directive: "This commitment expires before the read",
      priority: 4,
      provenance: { kind: "manual" },
      expiresAt: clock.now() + 10_000,
    });
    const future = harness.commitmentRepository.add({
      type: "promise",
      directiveFamily: "future_commitment",
      directive: "This commitment expires soon",
      priority: 4,
      provenance: { kind: "manual" },
      expiresAt: clock.now() + 120_000,
    });
    const trigger = createCommitmentExpiringTrigger({
      commitmentRepository: harness.commitmentRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      clock,
    });
    clock.advance(20_000);
    harness.db.pragma("wal_checkpoint(TRUNCATE)");
    const beforeBytes = readFileSync(join(harness.tempDir, "borg.db"));
    const beforeIdentityEvents = harness.identityEventRepository.list({
      recordType: "commitment",
      limit: 100,
    });

    await expect(trigger.nextDueAt!()).resolves.toBe(clock.now());
    const overdueEvents = await trigger.scan();
    expect(overdueEvents.map((event) => event.payload.commitment_id)).toEqual([expired.id]);

    const afterBytes = readFileSync(join(harness.tempDir, "borg.db"));
    const afterIdentityEvents = harness.identityEventRepository.list({
      recordType: "commitment",
      limit: 100,
    });
    expect(afterBytes.equals(beforeBytes)).toBe(true);
    expect(harness.commitmentRepository.get(expired.id)?.expired_at).toBeNull();
    expect(afterIdentityEvents).toEqual(beforeIdentityEvents);

    watermarkRepository.set(overdueEvents[0]!.watermarkProcessName, "default" as never, {
      lastTs: overdueEvents[0]!.sortTs,
      lastEntryId: overdueEvents[0]!.id,
    });
    await expect(trigger.scan()).resolves.toEqual([]);
    await expect(trigger.nextDueAt!()).resolves.toBe(future.expires_at! - 20_000 + 1);
  });

  it("uses expires_at mutation time as the fleet-freshness anchor", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const commitment = harness.commitmentRepository.add({
      type: "promise",
      directiveFamily: "fresh_expiration_patch",
      directive: "Surface a newly changed expiration.",
      priority: 5,
      provenance: { kind: "manual" },
      expiresAt: clock.now() + 200_000,
    });

    clock.advance(1);
    const patched = harness.commitmentRepository.update(
      commitment.id,
      { expires_at: clock.now() + 10_000 },
      { kind: "manual" },
    );
    const trigger = createCommitmentExpiringTrigger({
      commitmentRepository: harness.commitmentRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      clock,
    });

    const events = await trigger.scan();
    expect(events).toHaveLength(1);
    expect(events[0]?.stateTs).toBe(patched?.updated_at);
    expect(events[0]?.stateTs).toBeGreaterThan(commitment.updated_at ?? commitment.created_at);
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
