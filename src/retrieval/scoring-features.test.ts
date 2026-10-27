import { describe, expect, it, vi } from "vitest";
import { goalScoringText, buildSelfScoringFeatureSet } from "./scoring-features.js";
import { createOfflineTestHarness } from "../offline/test-support.js";
import { ManualClock } from "../util/clock.js";
import { StreamWatermarkRepository } from "../stream/watermark.js";
import { createExecutiveFocusDueTrigger } from "../autonomy/triggers/executive-focus-due.js";
import { createGoalFollowupDueTrigger } from "../autonomy/triggers/goal-followup-due.js";
import { TurnSelfContextBuilder } from "../cognition/self/turn-self-context.js";
import { NOOP_TRACER } from "../tracing/tracer.js";

describe("goal scoring byte budget", () => {
  it("retains the newest opaque notes in chronological order and hits the byte cap exactly", () => {
    expect(
      goalScoringText({ description: "goal", progress_notes: "old-middle-NEW" }, 15),
    ).toMatchObject({ text: "goal\nmiddle-NEW", renderedBytes: 15, elidedBytes: 4 });
  });
  it("never cuts through a UTF-8 character at either boundary", () => {
    expect(goalScoringText({ description: "目", progress_notes: "旧🙂新" }, 8).text).toBe("目\n新");
    expect(goalScoringText({ description: "目🙂標", progress_notes: "new" }, 6)).toMatchObject({
      text: "目",
      descriptionElidedBytes: 7,
      progressElidedBytes: 3,
    });
    expect(goalScoringText({ description: "d", progress_notes: "🙂新" }, 9).text).toBe("d\n🙂新");
  });
  it("uses only the safe description prefix when the description alone exceeds the budget", () => {
    expect(goalScoringText({ description: "abcdef", progress_notes: "recent" }, 4)).toMatchObject({
      text: "abcd",
      renderedBytes: 4,
      elidedBytes: 9,
    });
  });
  it("all three production callers use the same effective transport budget and report elision", async () => {
    const embeddingClient = {
      maxInputBytes: 20,
      embed: vi.fn(async () => new Float32Array([1, 0, 0, 0])),
      embedBatch: vi.fn(async (texts: readonly string[]) =>
        texts.map(() => new Float32Array([1, 0, 0, 0])),
      ),
    };
    const clock = new ManualClock(1_000);
    const h = await createOfflineTestHarness({ clock, embeddingClient });
    try {
      const goal = h.goalsRepository.add({
        description: "目的",
        priority: 10,
        progressNotes: "古い".repeat(20) + "LATEST🙂新",
        provenance: { kind: "manual" },
      });
      const expected = goalScoringText(goal, 20).text;
      clock.advance(100_000);
      const watermarks = new StreamWatermarkRepository({ db: h.db, clock });
      await createExecutiveFocusDueTrigger({
        enabled: true,
        goalsRepository: h.goalsRepository,
        executiveStepsRepository: h.executiveStepsRepository,
        episodicRepository: h.episodicRepository,
        embeddingClient,
        watermarkRepository: watermarks,
        threshold: 0,
        stalenessMs: 1,
        progressDebtStaleMs: 1,
        dueLeadMs: 0,
        wakeCooldownMs: 1,
        wakeCooldownBackoffMultiplier: 2,
        wakeCooldownMaxMs: 10,
        wakeEmptyDormancyCount: 3,
        deadlineLookaheadMs: 1,
        goalFollowupDue: { enabled: false, lookaheadMs: 1, staleMs: 1 },
        clock,
      }).scan();
      expect(embeddingClient.embedBatch.mock.calls[0]?.[0]).toEqual([expected]);
      embeddingClient.embedBatch.mockClear();
      await createGoalFollowupDueTrigger({
        goalsRepository: h.goalsRepository,
        watermarkRepository: watermarks,
        lookaheadMs: 1,
        staleMs: 1,
        clock,
        staleBackoff: { baseCooldownMs: 1, multiplier: 2, maxCooldownMs: 10, dormancyCount: 3 },
        respectStaleBackoff: true,
        executiveScoring: { embeddingClient, threshold: 0, deadlineLookaheadMs: 1, staleMs: 1 },
      }).scan();
      expect(embeddingClient.embedBatch.mock.calls[0]?.[0]).toEqual([expected]);
      embeddingClient.embedBatch.mockClear();
      await new TurnSelfContextBuilder({
        embeddingClient,
        goalsRepository: h.goalsRepository,
        valuesRepository: h.valuesRepository,
        traitsRepository: h.traitsRepository,
        executiveStepsRepository: h.executiveStepsRepository,
        clock,
        tracer: NOOP_TRACER,
        goalFocusThreshold: 0,
        goalFollowupLookaheadMs: 1,
        goalFollowupStaleMs: 1,
      }).build({
        turnId: "turn_scoring",
        cognitionInput: "",
        audienceEntityId: null,
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
          temporalCue: null,
        },
      });
      expect(embeddingClient.embedBatch.mock.calls[0]?.[0]).toEqual([expected]);
      const emit = vi.fn();
      await buildSelfScoringFeatureSet({
        embeddingClient,
        goals: [goal],
        activeValues: [],
        tracer: { emit },
      });
      expect(emit).toHaveBeenCalledWith(
        "retrieval.degraded",
        expect.objectContaining({
          goal_id: goal.id,
          max_input_bytes: 20,
          elidedBytes: goalScoringText(goal, 20).elidedBytes,
        }),
      );
    } finally {
      await h.cleanup();
    }
  });
});
