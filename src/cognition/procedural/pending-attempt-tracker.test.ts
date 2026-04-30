import { describe, expect, it } from "vitest";

import type { ActionResult } from "../action/index.js";
import type { PerceptionResult } from "../types.js";
import {
  deriveProceduralContextKey,
  proceduralContextSchema,
  type SkillSelectionResult,
} from "../../memory/procedural/index.js";
import { createWorkingMemory, type PendingProceduralAttempt } from "../../memory/working/index.js";
import {
  DEFAULT_SESSION_ID,
  type EntityId,
  type SkillId,
  type StreamEntryId,
} from "../../util/ids.js";
import { PendingProceduralAttemptTracker } from "./pending-attempt-tracker.js";

const userEntryId = "strm_abcdefghijklmnop" as StreamEntryId;
const agentEntryId = "strm_bcdefghijklmnopa" as StreamEntryId;
const audienceEntityId = "ent_abcdefghijklmnop" as EntityId;
const skillId = "skl_abcdefghijklmnop" as SkillId;

function parseProceduralContext(input: Parameters<typeof deriveProceduralContextKey>[0]) {
  return proceduralContextSchema.parse({
    ...input,
    context_key: deriveProceduralContextKey(input),
  });
}

function makePerception(mode: PerceptionResult["mode"]): PerceptionResult {
  return {
    mode,
    entities: [],
    temporalCue: null,
    affectiveSignal: {
      valence: 0,
      arousal: 0,
      dominant_emotion: null,
    },
  };
}

function makeActionResult(response: string): ActionResult {
  return {
    response,
    tool_calls: [],
    intents: [],
    workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
  };
}

function makeAttempt(turnCounter: number): PendingProceduralAttempt {
  return {
    problem_text: `problem ${turnCounter}`,
    approach_summary: `approach ${turnCounter}`,
    selected_skill_id: null,
    source_stream_ids: [userEntryId],
    turn_counter: turnCounter,
    audience_entity_id: null,
  };
}

describe("PendingProceduralAttemptTracker", () => {
  it("carries live attempts, drops expired ones, and appends a user problem-solving attempt", () => {
    const selectedSkill: SkillSelectionResult = {
      skill: {
        id: skillId,
        applies_when: "Known approach applies.",
        approach: "Use the known approach.",
        status: "active",
        alpha: 1,
        beta: 1,
        attempts: 0,
        successes: 0,
        failures: 0,
        alternatives: [],
        superseded_by: [],
        superseded_at: null,
        splitting_at: null,
        split_failure_count: 0,
        last_split_error: null,
        requires_manual_review: false,
        source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
        last_used: null,
        last_successful: null,
        created_at: 0,
        updated_at: 0,
      },
      sampledValue: 0.5,
      evaluatedCandidates: [],
    };
    const reflectedWorkingMemory = {
      ...createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      turn_counter: 10,
      pending_procedural_attempts: [makeAttempt(1), makeAttempt(3)],
    };

    const result = new PendingProceduralAttemptTracker().update({
      isUserTurn: true,
      userMessage: "  Fix\n\nAtlas   now  ",
      perception: makePerception("problem_solving"),
      actionResult: makeActionResult("unused response"),
      selectedSkill,
      reflectedWorkingMemory,
      persistedUserEntryId: userEntryId,
      persistedAgentEntryId: agentEntryId,
      audienceEntityId,
    });

    expect(result).toEqual([
      makeAttempt(3),
      {
        problem_text: "Fix Atlas now",
        approach_summary: "Use the known approach.",
        selected_skill_id: skillId,
        source_stream_ids: [userEntryId, agentEntryId],
        turn_counter: 10,
        audience_entity_id: audienceEntityId,
      },
    ]);
  });

  it("does not append attempts for autonomous turns", () => {
    const reflectedWorkingMemory = {
      ...createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      turn_counter: 2,
      pending_procedural_attempts: [makeAttempt(1)],
    };

    const result = new PendingProceduralAttemptTracker().update({
      isUserTurn: false,
      userMessage: "Fix Atlas",
      perception: makePerception("problem_solving"),
      actionResult: makeActionResult("Use a plan."),
      selectedSkill: null,
      reflectedWorkingMemory,
      persistedUserEntryId: userEntryId,
      persistedAgentEntryId: agentEntryId,
      audienceEntityId: null,
    });

    expect(result).toEqual([makeAttempt(1)]);
  });

  it("persists derived procedural context on new attempts", () => {
    const proceduralContext = parseProceduralContext({
      problem_kind: "code_debugging",
      domain_tags: ["TypeScript"],
      audience_scope: "self",
    });
    const result = new PendingProceduralAttemptTracker().update({
      isUserTurn: true,
      userMessage: "Fix TypeScript",
      perception: makePerception("problem_solving"),
      actionResult: makeActionResult("Use a focused test."),
      selectedSkill: null,
      proceduralContext,
      reflectedWorkingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      persistedUserEntryId: userEntryId,
      persistedAgentEntryId: agentEntryId,
      audienceEntityId: null,
    });

    expect(result.at(-1)?.procedural_context).toEqual(proceduralContext);
  });

  it("caps pending attempts by dropping the oldest entries", () => {
    const reflectedWorkingMemory = {
      ...createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      turn_counter: 7,
      pending_procedural_attempts: [2, 3, 4, 5, 6].map(makeAttempt),
    };

    const result = new PendingProceduralAttemptTracker().update({
      isUserTurn: true,
      userMessage: "Fix Atlas",
      perception: makePerception("problem_solving"),
      actionResult: makeActionResult("   "),
      selectedSkill: null,
      reflectedWorkingMemory,
      persistedUserEntryId: userEntryId,
      persistedAgentEntryId: agentEntryId,
      audienceEntityId: null,
    });

    expect(result.map((attempt) => attempt.turn_counter)).toEqual([3, 4, 5, 6, 7]);
    expect(result.at(-1)?.approach_summary).toBe("No explicit approach stated.");
  });
});
