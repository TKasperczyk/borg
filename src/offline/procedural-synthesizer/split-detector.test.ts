import { describe, expect, it } from "vitest";

import { createSkillId, type SkillId } from "../../util/ids.js";
import {
  deriveProceduralContextKey,
  type SkillContextStatsRecord,
  type SkillRecord,
} from "../../memory/procedural/index.js";
import { createSkillFixture } from "../test-support.js";
import { detectDivergentSkillSplits } from "./split-detector.js";

const NOW_MS = 10 * 24 * 60 * 60 * 1_000;
const TYPESCRIPT_DEBUG_CONTEXT_KEY = deriveProceduralContextKey({
  problem_kind: "code_debugging",
  domain_tags: ["typescript"],
  audience_scope: "self",
});
const ROADMAP_PLANNING_CONTEXT_KEY = deriveProceduralContextKey({
  problem_kind: "planning",
  domain_tags: ["roadmap"],
  audience_scope: "self",
});

function makeSkill(overrides: Partial<SkillRecord> = {}): SkillRecord {
  return createSkillFixture({
    id: overrides.id ?? createSkillId(),
    applies_when: "deployment rollback comparison",
    approach: "Compare the failing state with the last known-good state.",
    ...overrides,
  });
}

function makeStats(
  skillId: SkillId,
  contextKey: string,
  overrides: Partial<SkillContextStatsRecord> = {},
): SkillContextStatsRecord {
  return {
    skill_id: skillId,
    context_key: contextKey,
    alpha: 1,
    beta: 1,
    attempts: 0,
    successes: 0,
    failures: 0,
    last_used: null,
    last_successful: null,
    updated_at: 1_000,
    ...overrides,
  };
}

function detect(
  skill: SkillRecord,
  stats: readonly SkillContextStatsRecord[],
  overrides: Partial<Parameters<typeof detectDivergentSkillSplits>[0]> = {},
) {
  return detectDivergentSkillSplits({
    skills: [skill],
    contextStatsBySkillId: new Map([[skill.id, stats]]),
    nowMs: NOW_MS,
    minContextAttemptsForSplit: 5,
    minDivergenceForSplit: 0.3,
    splitCooldownDays: 7,
    splitClaimStaleSec: 1_800,
    maxSplitParseFailures: 3,
    ...overrides,
  });
}

describe("detectDivergentSkillSplits", () => {
  it("does not flag a skill with only one context bucket", () => {
    const skill = makeSkill();

    expect(
      detect(skill, [
        makeStats(skill.id, TYPESCRIPT_DEBUG_CONTEXT_KEY, {
          alpha: 6,
          beta: 1,
          attempts: 5,
        }),
      ]),
    ).toEqual([]);
  });

  it("does not flag similar posterior means", () => {
    const skill = makeSkill();

    expect(
      detect(skill, [
        makeStats(skill.id, TYPESCRIPT_DEBUG_CONTEXT_KEY, {
          alpha: 6,
          beta: 4,
          attempts: 8,
        }),
        makeStats(skill.id, ROADMAP_PLANNING_CONTEXT_KEY, {
          alpha: 5,
          beta: 5,
          attempts: 8,
        }),
      ]),
    ).toEqual([]);
  });

  it("does not flag divergent buckets below the attempt threshold", () => {
    const skill = makeSkill();

    expect(
      detect(skill, [
        makeStats(skill.id, TYPESCRIPT_DEBUG_CONTEXT_KEY, {
          alpha: 5,
          beta: 1,
          attempts: 4,
        }),
        makeStats(skill.id, ROADMAP_PLANNING_CONTEXT_KEY, {
          alpha: 1,
          beta: 5,
          attempts: 4,
        }),
      ]),
    ).toEqual([]);
  });

  it("flags divergent buckets with sufficient attempts", () => {
    const skill = makeSkill();
    const candidates = detect(skill, [
      makeStats(skill.id, TYPESCRIPT_DEBUG_CONTEXT_KEY, {
        alpha: 6,
        beta: 1,
        attempts: 5,
      }),
      makeStats(skill.id, ROADMAP_PLANNING_CONTEXT_KEY, {
        alpha: 1,
        beta: 6,
        attempts: 5,
      }),
    ]);

    expect(candidates).toHaveLength(1);
    expect(candidates[0]?.skill).toBe(skill);
    expect(candidates[0]?.divergence).toBeCloseTo(5 / 7);
  });

  it("does not flag a skill inside the split cooldown", () => {
    const skill = makeSkill({
      last_split_attempt_at: NOW_MS - 2 * 24 * 60 * 60 * 1_000,
    });

    expect(
      detect(skill, [
        makeStats(skill.id, TYPESCRIPT_DEBUG_CONTEXT_KEY, {
          alpha: 6,
          beta: 1,
          attempts: 5,
        }),
        makeStats(skill.id, ROADMAP_PLANNING_CONTEXT_KEY, {
          alpha: 1,
          beta: 6,
          attempts: 5,
        }),
      ]),
    ).toEqual([]);
  });

  it("does not flag a skill with a fresh split claim", () => {
    const skill = makeSkill({
      splitting_at: NOW_MS - 60_000,
    });

    expect(
      detect(skill, [
        makeStats(skill.id, TYPESCRIPT_DEBUG_CONTEXT_KEY, {
          alpha: 6,
          beta: 1,
          attempts: 5,
        }),
        makeStats(skill.id, ROADMAP_PLANNING_CONTEXT_KEY, {
          alpha: 1,
          beta: 6,
          attempts: 5,
        }),
      ]),
    ).toEqual([]);
  });

  it("allows a stale split claim to be reclaimed", () => {
    const skill = makeSkill({
      splitting_at: NOW_MS - 2 * 60 * 60 * 1_000,
    });

    expect(
      detect(skill, [
        makeStats(skill.id, TYPESCRIPT_DEBUG_CONTEXT_KEY, {
          alpha: 6,
          beta: 1,
          attempts: 5,
        }),
        makeStats(skill.id, ROADMAP_PLANNING_CONTEXT_KEY, {
          alpha: 1,
          beta: 6,
          attempts: 5,
        }),
      ]),
    ).toHaveLength(1);
  });

  it("suppresses skills that exceeded split parse failures", () => {
    const skill = makeSkill({
      split_failure_count: 3,
      last_split_error: "Malformed split tool output",
    });

    expect(
      detect(skill, [
        makeStats(skill.id, TYPESCRIPT_DEBUG_CONTEXT_KEY, {
          alpha: 6,
          beta: 1,
          attempts: 5,
        }),
        makeStats(skill.id, ROADMAP_PLANNING_CONTEXT_KEY, {
          alpha: 1,
          beta: 6,
          attempts: 5,
        }),
      ]),
    ).toEqual([]);
  });
});
