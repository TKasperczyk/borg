import { afterEach, describe, expect, it } from "vitest";

import { createOfflineTestHarness } from "../../offline/test-support.js";
import { StreamWatermarkRepository } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";

import { createOpenQuestionUrgencyBumpCondition } from "./open-question-urgency-bump.js";

describe("open question urgency bump condition", () => {
  let cleanup: (() => Promise<void>) | undefined;

  afterEach(async () => {
    await cleanup?.();
    cleanup = undefined;
  });

  it("fires at threshold, re-fires after an urgency bump, and ignores resolved questions", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    const question = harness.openQuestionsRepository.add({
      question: "How should the wake loop escalate?",
      urgency: 0.9,
      provenance: { kind: "system" },
      source: "user",
    });
    const resolved = harness.openQuestionsRepository.add({
      question: "What already shipped?",
      urgency: 0.95,
      provenance: { kind: "system" },
      source: "user",
    });
    harness.openQuestionsRepository.resolve(resolved.id, {
      resolution_episode_id: "ep_aaaaaaaaaaaaaaaa" as never,
    });
    const condition = createOpenQuestionUrgencyBumpCondition({
      openQuestionsRepository: harness.openQuestionsRepository,
      watermarkRepository,
      threshold: 0.9,
      clock,
    });

    const firstScan = await condition.scan();
    expect(firstScan).toHaveLength(1);
    expect(firstScan[0]?.payload).toMatchObject({
      open_question_id: question.id,
      urgency: 0.9,
    });
    expect(firstScan[0]?.payload.open_question_id).not.toBe(resolved.id);

    watermarkRepository.set(firstScan[0]!.watermarkProcessName, "default", {
      lastTs: clock.now(),
      lastEntryId: null,
    });
    expect(await condition.scan()).toEqual([]);

    harness.openQuestionsRepository.bumpUrgency(question.id, 0.03);
    const secondScan = await condition.scan();
    expect(secondScan).toHaveLength(1);
    expect(secondScan[0]?.payload.urgency).toBeCloseTo(0.93, 2);
  });
});
