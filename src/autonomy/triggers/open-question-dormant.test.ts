import { afterEach, describe, expect, it } from "vitest";

import { StreamWatermarkRepository } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { createOfflineTestHarness } from "../../offline/test-support.js";
import { formatAutonomyTriggerContext } from "../../cognition/autonomy-trigger.js";

import { createOpenQuestionDormantTrigger } from "./open-question-dormant.js";

describe("open question dormant trigger", () => {
  let cleanup: (() => Promise<void>) | undefined;

  afterEach(async () => {
    await cleanup?.();
    cleanup = undefined;
  });

  it("finds dormant open questions", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    const dormant = harness.openQuestionsRepository.add({
      question: "What is the right autonomy cadence?",
      urgency: 0.5,
      provenance: { kind: "system" },
      source: "user",
      last_touched: clock.now() - 100_000,
    });
    harness.openQuestionsRepository.add({
      question: "Fresh question",
      urgency: 0.5,
      provenance: { kind: "system" },
      source: "user",
      last_touched: clock.now() - 1_000,
    });

    const trigger = createOpenQuestionDormantTrigger({
      openQuestionsRepository: harness.openQuestionsRepository,
      watermarkRepository,
      dormantMs: 50_000,
      clock,
    });

    const events = await trigger.scan();
    expect(events.map((event) => event.payload.open_question_id)).toEqual([dormant.id]);
    expect(events[0]?.stateTs).toBe(dormant.last_touched);
  });

  it("describes the next dormancy threshold without firing", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    harness.openQuestionsRepository.add({
      question: "When should this be revisited?",
      urgency: 0.5,
      provenance: { kind: "system" },
      source: "user",
      last_touched: clock.now() - 10_000,
    });
    const trigger = createOpenQuestionDormantTrigger({
      openQuestionsRepository: harness.openQuestionsRepository,
      watermarkRepository,
      dormantMs: 50_000,
      clock,
    });

    await expect(trigger.nextDueAt!()).resolves.toBe(clock.now() + 40_001);

    clock.advance(50_000);
    await expect(trigger.nextDueAt!()).resolves.toBe(clock.now());
  });

  it("renders dormant question disclosure labels in the autonomy payload", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const sam = harness.entityRepository.resolve("Sam");
    const dormant = harness.openQuestionsRepository.add({
      question: "What should I remember about Sam's planning thread?",
      urgency: 0.5,
      provenance: { kind: "system" },
      source: "user",
      audience_entity_id: sam,
      last_touched: clock.now() - 100_000,
    });

    const trigger = createOpenQuestionDormantTrigger({
      openQuestionsRepository: harness.openQuestionsRepository,
      watermarkRepository,
      dormantMs: 50_000,
      clock,
    });

    const events = await trigger.scan();
    const turn = trigger.buildTurn(events[0]!);
    const rendered = formatAutonomyTriggerContext(turn.autonomyTrigger!);

    expect(events[0]?.payload).toMatchObject({
      open_question_id: dormant.id,
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [sam],
      },
    });
    expect(rendered).toContain("disclosure_label");
    expect(rendered).toContain("relationship_private");
    expect(rendered).toContain(sam);
  });

  it("carries the offline rumination bookkeeping the wake itself does not produce", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const untouched = harness.openQuestionsRepository.add({
      question: "What has no offline pass reached?",
      urgency: 0.6,
      provenance: { kind: "system" },
      source: "user",
      last_touched: clock.now() - 100_000,
    });
    const ruminated = harness.openQuestionsRepository.add({
      question: "What has the offline loop worked?",
      urgency: 0.4,
      provenance: { kind: "system" },
      source: "user",
      last_touched: clock.now() - 100_000,
    });
    harness.openQuestionsRepository.markRuminated(ruminated.id, 2);

    const trigger = createOpenQuestionDormantTrigger({
      openQuestionsRepository: harness.openQuestionsRepository,
      watermarkRepository,
      dormantMs: 50_000,
      clock,
    });

    const events = await trigger.scan();
    const payloads = new Map(events.map((event) => [event.payload.open_question_id, event.payload]));

    // A question no offline pass has reached says so on its face rather than
    // arriving indistinguishable from one the loop has worked repeatedly.
    expect(payloads.get(untouched.id)).toMatchObject({
      unresolved_rumination_ticks: 0,
      last_ruminated_at: null,
    });
    expect(payloads.get(ruminated.id)).toMatchObject({
      unresolved_rumination_ticks: 2,
      last_ruminated_at: clock.now(),
    });
    // Rumination is not a touch, so it neither clears the dormancy nor moves the
    // stamp the event id latches on.
    expect(payloads.get(ruminated.id)?.last_touched).toBe(ruminated.last_touched);

    const rendered = formatAutonomyTriggerContext(
      trigger.buildTurn(events.find((event) => event.payload.open_question_id === untouched.id)!)
        .autonomyTrigger!,
    );

    expect(rendered).toContain('"last_ruminated_at": null');
    expect(rendered).toContain("no offline pass has been written against it since it was opened");
  });
});
