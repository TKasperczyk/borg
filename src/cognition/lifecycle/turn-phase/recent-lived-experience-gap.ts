export type RecentLivedExperienceGapInput = {
  nowMs: number;
  mostRecentOtherSessionActivityAt: number | null;
  currentSessionPreviousTurnAt: number | null;
  gapThresholdMs: number;
};

export function shouldRenderRecentLivedExperience(input: RecentLivedExperienceGapInput): boolean {
  // No life elsewhere at all -> nothing intervening to surface.
  if (input.mostRecentOtherSessionActivityAt === null) {
    return false;
  }

  // First turn in this session: there is no return-silence to measure, but the
  // entity may have a life elsewhere worth surfacing on first contact.
  if (input.currentSessionPreviousTurnAt === null) {
    return true;
  }

  // Surface only when both hold:
  // (a) there is intervening other-session activity since the entity last engaged
  //     THIS session, and
  // (b) the entity is returning after a silence in this session of at least
  //     gapThresholdMs -- "how long have I been away from this audience" measured
  //     against now, not against when the other-session activity happened.
  const hasInterveningActivity =
    input.mostRecentOtherSessionActivityAt > input.currentSessionPreviousTurnAt;
  const returnedAfterSilence =
    input.nowMs - input.currentSessionPreviousTurnAt >= Math.max(0, input.gapThresholdMs);

  return hasInterveningActivity && returnedAfterSilence;
}
