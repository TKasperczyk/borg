import { describe, expect, it } from "vitest";

import { shouldRenderRecentLivedExperience } from "./recent-lived-experience-gap.js";

describe("shouldRenderRecentLivedExperience", () => {
  it("renders when returning after the silence threshold with intervening other-session life", () => {
    // Last engaged this session at 1_000; intervening other-session activity at
    // 5_000; returning now at 10_000 (silence 9_000 >= threshold 1_000).
    expect(
      shouldRenderRecentLivedExperience({
        nowMs: 10_000,
        mostRecentOtherSessionActivityAt: 5_000,
        currentSessionPreviousTurnAt: 1_000,
        gapThresholdMs: 1_000,
      }),
    ).toBe(true);
  });

  it("does not render on rapid same-session back-and-forth below the threshold", () => {
    // Returned only 200ms after the last turn -> below a 1_000 threshold, even
    // though a tiny bit of other-session activity happened in between.
    expect(
      shouldRenderRecentLivedExperience({
        nowMs: 1_200,
        mostRecentOtherSessionActivityAt: 1_100,
        currentSessionPreviousTurnAt: 1_000,
        gapThresholdMs: 1_000,
      }),
    ).toBe(false);
  });

  it("thresholds on return-silence (now - last turn), not on when the other activity happened", () => {
    // Other-session activity clustered right after the last turn (1_050), then a
    // long return gap to now (10_000). Keying on activity-vs-last-turn would miss
    // this; the silence (9_000) clears the threshold, so it renders.
    expect(
      shouldRenderRecentLivedExperience({
        nowMs: 10_000,
        mostRecentOtherSessionActivityAt: 1_050,
        currentSessionPreviousTurnAt: 1_000,
        gapThresholdMs: 1_000,
      }),
    ).toBe(true);
  });

  it("does not render without any intervening other-session activity", () => {
    // Long silence, but the most-recent other-session activity predates the last
    // turn in this session -> nothing new happened elsewhere since.
    expect(
      shouldRenderRecentLivedExperience({
        nowMs: 10_000,
        mostRecentOtherSessionActivityAt: 500,
        currentSessionPreviousTurnAt: 1_000,
        gapThresholdMs: 1_000,
      }),
    ).toBe(false);
    expect(
      shouldRenderRecentLivedExperience({
        nowMs: 10_000,
        mostRecentOtherSessionActivityAt: null,
        currentSessionPreviousTurnAt: 1_000,
        gapThresholdMs: 1_000,
      }),
    ).toBe(false);
  });

  it("renders on the first turn of a session when there is a life elsewhere", () => {
    expect(
      shouldRenderRecentLivedExperience({
        nowMs: 10_000,
        mostRecentOtherSessionActivityAt: 5_000,
        currentSessionPreviousTurnAt: null,
        gapThresholdMs: 1_000,
      }),
    ).toBe(true);
  });
});
