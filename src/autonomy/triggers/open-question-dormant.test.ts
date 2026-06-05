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
});
