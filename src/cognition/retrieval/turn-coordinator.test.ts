import { describe, expect, it, vi } from "vitest";

import type { MoodHistoryEntry } from "../../memory/affective/index.js";
import type { ExecutiveFocus } from "../../executive/index.js";
import type { CommitmentRecord, EntityRecord } from "../../memory/commitments/index.js";
import type { ReviewQueueItem } from "../../memory/semantic/index.js";
import type { SkillSelectionResult } from "../../memory/procedural/index.js";
import type { SocialProfile } from "../../memory/social/index.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import type { RetrievedContext } from "../../retrieval/index.js";
import { ManualClock } from "../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  type CommitmentId,
  type EntityId,
  type GoalId,
  type ValueId,
} from "../../util/ids.js";
import { SuppressionSet } from "../attention/index.js";
import type { SelfSnapshot } from "../deliberation/deliberator.js";
import type { PerceptionResult } from "../types.js";
import { TurnRetrievalCoordinator } from "./turn-coordinator.js";

const audienceEntityId = "entity_alice" as EntityId;
const atlasEntityId = "entity_atlas" as EntityId;
const bobEntityId = "entity_bob" as EntityId;

function makeCommitment(id: string, priority: number, createdAt: number): CommitmentRecord {
  return {
    id: id as CommitmentId,
    type: "promise",
    directive: `directive ${id}`,
    priority,
    made_to_entity: null,
    restricted_audience: null,
    about_entity: null,
    provenance: {
      kind: "system",
    },
    created_at: createdAt,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
  };
}

function makeReviewItem(id: number, refs: Record<string, unknown>): ReviewQueueItem {
  return {
    id,
    kind: "correction",
    refs,
    reason: `correction ${id}`,
    created_at: id,
    resolved_at: null,
    resolution: null,
  };
}

function makePerception(mode: PerceptionResult["mode"]): PerceptionResult {
  return {
    mode,
    entities: ["Atlas", "Bob"],
    temporalCue: null,
    affectiveSignal: {
      valence: 0.1,
      arousal: 0.1,
      dominant_emotion: "curiosity",
    },
  };
}

function makeSelfSnapshot(): SelfSnapshot {
  return {
    values: [
      {
        id: "value_established" as ValueId,
        label: "Care",
        description: "Be careful",
        priority: 1,
        created_at: 100,
        last_affirmed: null,
        state: "established",
        established_at: 100,
        confidence: 0.8,
        last_tested_at: null,
        last_contradicted_at: null,
        support_count: 0,
        contradiction_count: 0,
        evidence_episode_ids: [],
        provenance: {
          kind: "system",
        },
      },
      {
        id: "value_candidate" as ValueId,
        label: "Speed",
        description: "Move quickly",
        priority: 5,
        created_at: 200,
        last_affirmed: null,
        state: "candidate",
        established_at: null,
        confidence: 0.4,
        last_tested_at: null,
        last_contradicted_at: null,
        support_count: 0,
        contradiction_count: 0,
        evidence_episode_ids: [],
        provenance: {
          kind: "system",
        },
      },
    ],
    goals: [
      {
        id: "goal_1" as GoalId,
        description: "Ship the sprint",
        priority: 1,
        parent_goal_id: null,
        status: "active",
        progress_notes: null,
        last_progress_ts: null,
        created_at: 100,
        target_at: null,
        provenance: {
          kind: "system",
        },
      },
    ],
    traits: [],
  };
}

function makeAudienceProfile(): SocialProfile {
  return {
    entity_id: audienceEntityId,
    trust: 0.7,
    attachment: 0.4,
    communication_style: null,
    shared_history_summary: null,
    last_interaction_at: null,
    interaction_count: 3,
    commitment_count: 0,
    sentiment_history: [],
    notes: null,
    created_at: 100,
    updated_at: 100,
  };
}

function makeRetrievedContext(): RetrievedContext {
  return {
    episodes: [],
    semantic: {},
    open_questions: [],
    contradiction_present: false,
    confidence: {},
  } as RetrievedContext;
}

describe("TurnRetrievalCoordinator", () => {
  it("builds retrieval context and preserves reRetrieve override precedence", async () => {
    const high = makeCommitment("cmt_high", 10, 200);
    const low = makeCommitment("cmt_low", 1, 100);
    const getApplicable = vi.fn(({ aboutEntity }: { aboutEntity?: EntityId | null }) =>
      aboutEntity === atlasEntityId || aboutEntity === bobEntityId ? [low, high] : [],
    );
    const pendingCorrections = [
      makeReviewItem(1, {}),
      makeReviewItem(2, { audience_entity_id: audienceEntityId }),
      makeReviewItem(3, { audience_entity_id: bobEntityId }),
    ];
    const currentMood = {
      session_id: DEFAULT_SESSION_ID,
      valence: 0.6,
      arousal: 0.1,
      updated_at: 1_900,
      half_life_hours: 24,
      recent_triggers: [],
    };
    const affectiveTrajectory: MoodHistoryEntry[] = [
      {
        id: 1,
        session_id: DEFAULT_SESSION_ID,
        ts: 900,
        valence: 0.2,
        arousal: 0.2,
        trigger_reason: null,
        provenance: {
          kind: "system",
        },
      },
    ];
    const retrieval = makeRetrievedContext();
    const searchWithContext = vi.fn(async () => retrieval);
    const search = vi.fn(async () => []);
    const selectedSkill: SkillSelectionResult = {
      skill: {
        id: "skl_aaaaaaaaaaaaaaaa" as never,
        applies_when: "Known fix applies.",
        approach: "Use the known fix.",
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
        source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
        last_used: null,
        last_successful: null,
        created_at: 0,
        updated_at: 0,
      },
      sampledValue: 0.5,
      evaluatedCandidates: [],
    };
    const select = vi.fn(async () => selectedSkill);
    const coordinator = new TurnRetrievalCoordinator({
      commitmentRepository: {
        getApplicable,
      },
      reviewQueueRepository: {
        list: vi.fn(() => pendingCorrections),
      },
      moodRepository: {
        current: vi.fn(() => currentMood),
        history: vi.fn(() => affectiveTrajectory),
      },
      retrievalPipeline: {
        searchWithContext,
        search,
      },
      skillSelector: {
        select,
      },
      clock: new ManualClock(2_000),
    });
    const workingMemory = {
      ...createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      mood: {
        valence: 0.1,
        arousal: 0.1,
        dominant_emotion: null,
      },
    };
    const suppressionSet = SuppressionSet.fromEntries([], 1);
    const audienceEntity: EntityRecord = {
      id: audienceEntityId,
      canonical_name: "Alice",
      aliases: ["Al"],
      created_at: 100,
    };

    const result = await coordinator.coordinate({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-1",
      userMessage: "Solve Atlas",
      cognitionInput: "Solve Atlas",
      inputAudience: "alice",
      isSelfAudience: false,
      audienceEntityId,
      audienceEntity,
      audienceProfile: makeAudienceProfile(),
      perception: makePerception("problem_solving"),
      workingMemory,
      selfSnapshot: makeSelfSnapshot(),
      suppressionSet,
      findEntityByName: (name) =>
        name === "Atlas" ? atlasEntityId : name === "Bob" ? bobEntityId : null,
    });

    expect(result.applicableCommitments).toEqual([high, low]);
    expect(result.pendingCorrections.map((item) => item.id)).toEqual([1, 2]);
    expect(result.affectiveTrajectory).toBe(affectiveTrajectory);
    expect(result.retrieval).toBe(retrieval);
    expect(result.selectedSkill).toBe(selectedSkill);
    expect(result.proceduralContext).toMatchObject({
      problem_kind: "other",
      domain_tags: ["atlas", "bob"],
      audience_scope: "known_other",
      context_key: "other:atlas,bob:known_other",
    });
    expect(select).toHaveBeenCalledWith("Solve Atlas Atlas Bob", {
      k: 5,
      proceduralContext: result.proceduralContext,
    });
    expect(searchWithContext).toHaveBeenCalledWith(
      "Solve Atlas",
      expect.objectContaining({
        audienceEntityId,
        audienceTerms: ["Alice", "Al", "alice"],
        entityTerms: ["Atlas", "Bob"],
        goalDescriptions: ["Ship the sprint"],
        moodState: currentMood,
        suppressionSet,
        includeOpenQuestions: false,
        traceTurnId: "turn-1",
      }),
    );

    await result.reRetrieve("verify", {
      limit: 3,
      strictTimeRange: false,
    });

    expect(search).toHaveBeenCalledWith(
      "verify",
      expect.objectContaining({
        audienceEntityId,
        limit: 3,
        strictTimeRange: false,
        traceTurnId: "turn-1",
      }),
    );
  });

  it("skips skill selection for non-problem-solving turns", async () => {
    const select = vi.fn();
    const searchWithContext = vi.fn(async () => makeRetrievedContext());
    const coordinator = new TurnRetrievalCoordinator({
      commitmentRepository: {
        getApplicable: vi.fn(() => []),
      },
      reviewQueueRepository: {
        list: vi.fn(() => []),
      },
      moodRepository: {
        current: vi.fn(() => ({
          session_id: DEFAULT_SESSION_ID,
          valence: 0,
          arousal: 0,
          updated_at: 2_000,
          half_life_hours: 24,
          recent_triggers: [],
        })),
        history: vi.fn(() => []),
      },
      retrievalPipeline: {
        searchWithContext,
        search: vi.fn(async () => []),
      },
      skillSelector: {
        select,
      },
      clock: new ManualClock(2_000),
    });

    const result = await coordinator.coordinate({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-1",
      userMessage: "Think about this",
      cognitionInput: "Think about this",
      isSelfAudience: true,
      audienceEntityId: null,
      audienceEntity: null,
      audienceProfile: null,
      perception: makePerception("reflective"),
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      selfSnapshot: makeSelfSnapshot(),
      suppressionSet: SuppressionSet.fromEntries([], 1),
      findEntityByName: () => null,
    });

    expect(result.selectedSkill).toBeNull();
    expect(result.proceduralContext).toBeNull();
    expect(select).not.toHaveBeenCalled();
    expect(searchWithContext).toHaveBeenCalledWith(
      "Think about this",
      expect.objectContaining({
        audienceTerms: [],
        includeOpenQuestions: true,
      }),
    );
  });

  it("passes selected executive goal as the primary retrieval goal without dropping other goals", async () => {
    const searchWithContext = vi.fn(async () => makeRetrievedContext());
    const coordinator = new TurnRetrievalCoordinator({
      commitmentRepository: {
        getApplicable: vi.fn(() => []),
      },
      reviewQueueRepository: {
        list: vi.fn(() => []),
      },
      moodRepository: {
        current: vi.fn(() => ({
          session_id: DEFAULT_SESSION_ID,
          valence: 0,
          arousal: 0,
          updated_at: 2_000,
          half_life_hours: 24,
          recent_triggers: [],
        })),
        history: vi.fn(() => []),
      },
      retrievalPipeline: {
        searchWithContext,
        search: vi.fn(async () => []),
      },
      skillSelector: {
        select: vi.fn(async () => null),
      },
      clock: new ManualClock(2_000),
    });
    const selfSnapshot = makeSelfSnapshot();
    const selectedGoal = {
      id: "goal_2" as GoalId,
      description: "Resolve Atlas incident",
      priority: 2,
      parent_goal_id: null,
      status: "active" as const,
      progress_notes: null,
      last_progress_ts: null,
      created_at: 200,
      target_at: null,
      provenance: {
        kind: "system" as const,
      },
    };
    const executiveFocus: ExecutiveFocus = {
      selected_goal: selectedGoal,
      selected_score: {
        goal_id: selectedGoal.id,
        goal: selectedGoal,
        score: 0.6,
        components: {
          priority: 0.8,
          deadline_pressure: 0,
          context_fit: 1,
          progress_debt: 0,
        },
        reason: "test",
      },
      candidates: [],
      threshold: 0.45,
    };

    await coordinator.coordinate({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-1",
      userMessage: "Solve Atlas",
      cognitionInput: "Solve Atlas",
      isSelfAudience: true,
      audienceEntityId: null,
      audienceEntity: null,
      audienceProfile: null,
      perception: makePerception("reflective"),
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      selfSnapshot: {
        ...selfSnapshot,
        goals: [...selfSnapshot.goals, selectedGoal],
      },
      executiveFocus,
      suppressionSet: SuppressionSet.fromEntries([], 1),
      findEntityByName: () => null,
    });

    expect(searchWithContext).toHaveBeenCalledWith(
      "Solve Atlas",
      expect.objectContaining({
        goalDescriptions: ["Resolve Atlas incident", "Ship the sprint"],
        primaryGoalDescription: "Resolve Atlas incident",
      }),
    );
  });
});
