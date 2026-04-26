import { describe, expect, it } from "vitest";

import type { MoodHistoryEntry } from "../../../memory/affective/index.js";
import type {
  SkillRecord,
  SkillSelectionCandidate,
  SkillSelectionResult,
} from "../../../memory/procedural/index.js";
import { DEFAULT_SESSION_ID } from "../../../util/ids.js";
import { UNTRUSTED_DATA_PREAMBLE } from "../constants.js";
import type { DeliberationContext } from "../types.js";

import { buildBaseSystemPrompt } from "./system-prompt.js";

const NOW_MS = 1_700_000_000_000;
const PROMPT_OPTIONS = {
  retrievalContextBudget: 1_000,
  semanticContextBudget: 1_000,
};

function makeContext(overrides: Partial<DeliberationContext> = {}): DeliberationContext {
  return {
    sessionId: DEFAULT_SESSION_ID,
    userMessage: "Help me debug the rollout.",
    perception: {
      entities: ["rollout"],
      mode: "problem_solving",
      affectiveSignal: {
        valence: 0,
        arousal: 0,
      },
      temporalCue: null,
    },
    retrievalResult: [],
    workingMemory: {
      session_id: DEFAULT_SESSION_ID,
      turn_counter: 3,
      current_focus: "rollout",
      hot_entities: ["rollout"],
      pending_intents: [],
      pending_social_attribution: null,
      pending_trait_attribution: null,
      suppressed: [],
      mood: {
        valence: 0.9,
        arousal: 0.9,
        dominant_emotion: null,
      },
      last_selected_skill_id: null,
      last_selected_skill_turn: null,
      mode: "problem_solving",
      updated_at: NOW_MS,
    },
    selfSnapshot: {
      values: [],
      goals: [],
      traits: [],
    },
    ...overrides,
  };
}

function makeSkill(id: string, appliesWhen: string, approach: string): SkillRecord {
  return {
    id: id as SkillRecord["id"],
    applies_when: appliesWhen,
    approach,
    alpha: 4,
    beta: 3,
    attempts: 5,
    successes: 3,
    failures: 2,
    alternatives: [],
    source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as SkillRecord["source_episode_ids"][number]],
    last_used: null,
    last_successful: null,
    created_at: 0,
    updated_at: 0,
  };
}

function makeCandidate(
  skill: SkillRecord,
  sampledValue: number,
  mean: number,
  ci95: [number, number],
  similarity: number,
): SkillSelectionCandidate {
  return {
    skill,
    sampledValue,
    similarity,
    stats: {
      mean,
      ci_95: ci95,
    },
  };
}

function makeSelection(
  selected: SkillRecord,
  candidates: readonly SkillSelectionCandidate[],
): SkillSelectionResult {
  const selectedCandidate = candidates.find((candidate) => candidate.skill.id === selected.id);

  return {
    skill: selected,
    sampledValue: selectedCandidate?.sampledValue ?? 0,
    evaluatedCandidates: [...candidates],
  };
}

function makeMoodHistoryEntry(
  id: number,
  minutesAgo: number,
  valence: number,
  arousal: number,
  triggerReason: string | null,
): MoodHistoryEntry {
  return {
    id,
    session_id: DEFAULT_SESSION_ID,
    ts: NOW_MS - minutesAgo * 60_000,
    valence,
    arousal,
    trigger_reason: triggerReason,
    provenance: {
      kind: "system",
    },
  };
}

function extractBlock(prompt: string, tag: string): string {
  const openTag = `<${tag}>`;
  const closeTag = `</${tag}>`;
  const start = prompt.indexOf(openTag);
  const end = prompt.indexOf(closeTag);

  expect(start).toBeGreaterThanOrEqual(0);
  expect(end).toBeGreaterThan(start);

  return prompt.slice(start, end + closeTag.length);
}

describe("buildBaseSystemPrompt", () => {
  it("renders the selected skill first with up to two evaluated alternatives", () => {
    const tracePath = makeSkill(
      "skl_aaaaaaaaaaaaaaaa",
      "Trace the failing path",
      "Walk the smallest repro through logs.",
    );
    const focusedTest = makeSkill(
      "skl_bbbbbbbbbbbbbbbb",
      "Write a focused regression test",
      "Start with failing coverage before changing behavior.",
    );
    const compareRollout = makeSkill(
      "skl_cccccccccccccccc",
      "Compare previous rollout",
      "Diff the last known-good deployment.",
    );
    const broadRefactor = makeSkill(
      "skl_dddddddddddddddd",
      "Broad refactor",
      "Rewrite the deployment module.",
    );
    const selectedSkill = makeSelection(focusedTest, [
      makeCandidate(tracePath, 0.9, 0.5, [0.2, 0.8], 0.91),
      makeCandidate(focusedTest, 0.77, 0.55, [0.3, 0.8], 0.83),
      makeCandidate(compareRollout, 0.66, 0.7, [0.5, 0.9], 0.76),
      makeCandidate(broadRefactor, 0.6, 0.4, [0.1, 0.7], 0.71),
    ]);

    const prompt = buildBaseSystemPrompt(makeContext({ selectedSkill }), PROMPT_OPTIONS);
    const block = extractBlock(prompt, "borg_procedural_guidance");

    expect(block).toContain(
      "Skill candidates considered (winner first; activation_sample is a Thompson draw, not confidence):",
    );
    expect(block).toContain(
      "- winner: Write a focused regression test -- Start with failing coverage before changing behavior. (activation_sample=0.77 posterior_mean=0.55 ci95_width=0.50 similarity=0.83)",
    );
    expect(block).toContain(
      "- alternative: Trace the failing path -- Walk the smallest repro through logs. (activation_sample=0.90 posterior_mean=0.50 ci95_width=0.60 similarity=0.91)",
    );
    expect(block).toContain(
      "- alternative: Compare previous rollout -- Diff the last known-good deployment. (activation_sample=0.66 posterior_mean=0.70 ci95_width=0.40 similarity=0.76)",
    );
    expect(block).not.toContain("Broad refactor");
    expect(block).not.toContain("Success rate");
    expect(block.indexOf("- winner:")).toBeLessThan(block.indexOf("- alternative: Trace"));
  });

  it("renders an empty procedural placeholder when no candidates were evaluated", () => {
    // Same pattern as the empty-commitments fix: when problem_solving mode is
    // active but the procedural band has nothing to surface, render the channel
    // with an honest placeholder so the being can distinguish "no skills exist
    // yet" from "the channel doesn't exist".
    const selected = makeSkill(
      "skl_aaaaaaaaaaaaaaaa",
      "Trace the failing path",
      "Walk the smallest repro through logs.",
    );
    const prompt = buildBaseSystemPrompt(
      makeContext({
        selectedSkill: makeSelection(selected, []),
      }),
      PROMPT_OPTIONS,
    );

    expect(prompt).toContain("<borg_procedural_guidance>");
    expect(prompt).toContain(
      "No procedural skills matched this turn. Use tool.skills.list to inspect the registry.",
    );
    expect(prompt).not.toContain("tool.skills.add");
  });

  it("renders an empty procedural placeholder when no skill was selected at all", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        selectedSkill: null,
      }),
      PROMPT_OPTIONS,
    );

    expect(prompt).toContain("<borg_procedural_guidance>");
    expect(prompt).toContain("No procedural skills matched this turn.");
    expect(prompt).not.toContain("tool.skills.add");
  });

  it("omits procedural guidance outside problem-solving mode", () => {
    const selected = makeSkill(
      "skl_aaaaaaaaaaaaaaaa",
      "Trace the failing path",
      "Walk the smallest repro through logs.",
    );
    const prompt = buildBaseSystemPrompt(
      makeContext({
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
          },
          temporalCue: null,
        },
        selectedSkill: makeSelection(selected, [
          makeCandidate(selected, 0.82, 0.67, [0.4, 0.9], 0.9),
        ]),
      }),
      PROMPT_OPTIONS,
    );

    expect(prompt).not.toContain("<borg_procedural_guidance>");
  });

  it("renders a capped affective trajectory with relative ages and triggers", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        affectiveTrajectory: [
          makeMoodHistoryEntry(1, 2, -0.3, 0.4, "user expressed frustration"),
          makeMoodHistoryEntry(2, 14, 0, 0.1, "topic shift"),
          makeMoodHistoryEntry(3, 32, 0.2, 0.2, "problem-solving exchange"),
          makeMoodHistoryEntry(4, 67, -0.1, 0.5, null),
          makeMoodHistoryEntry(5, 130, 0.1, 0.2, "follow-up"),
          makeMoodHistoryEntry(6, 150, -0.4, 0.8, "sixth entry"),
        ],
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_affective_trajectory");

    expect(prompt.indexOf(UNTRUSTED_DATA_PREAMBLE)).toBeLessThan(
      prompt.indexOf("<borg_affective_trajectory>"),
    );
    expect(block).toContain(
      "Affective trajectory (newest first; current snapshot in working state):",
    );
    expect(block).toContain(
      '- 2m ago: valence=-0.30 arousal=0.40 trigger="user expressed frustration"',
    );
    expect(block).toContain('- 14m ago: valence=0.00 arousal=0.10 trigger="topic shift"');
    expect(block).toContain(
      '- 32m ago: valence=0.20 arousal=0.20 trigger="problem-solving exchange"',
    );
    expect(block).toContain("- 1h ago: valence=-0.10 arousal=0.50");
    expect(block).toContain('- 2h ago: valence=0.10 arousal=0.20 trigger="follow-up"');
    expect(block).not.toContain("sixth entry");
    expect(block).not.toContain("0.90");
  });

  it("omits affective trajectory when history is empty or undefined", () => {
    const emptyPrompt = buildBaseSystemPrompt(
      makeContext({
        affectiveTrajectory: [],
      }),
      PROMPT_OPTIONS,
    );
    const undefinedPrompt = buildBaseSystemPrompt(makeContext(), PROMPT_OPTIONS);

    expect(emptyPrompt).not.toContain("<borg_affective_trajectory>");
    expect(undefinedPrompt).not.toContain("<borg_affective_trajectory>");
  });
});
