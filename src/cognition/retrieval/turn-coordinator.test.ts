import { describe, expect, it, vi } from "vitest";

import type { MoodHistoryEntry } from "../../memory/affective/index.js";
import type { ExecutiveFocus } from "../../executive/index.js";
import type { BorgRole, CommitmentRecord, EntityRecord } from "../../memory/commitments/index.js";
import type {
  OpenCommitmentReconciliationStatus,
  ReviewQueueItem,
} from "../../memory/review-queue/index.js";
import type { SkillSelectionResult } from "../../memory/procedural/index.js";
import type { SocialProfile } from "../../memory/social/index.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import type { LLMCompleteResult } from "../../llm/index.js";
import {
  SELF_RECALL_SCOPE,
  type CognitionRecallContext,
  type DisclosureContext,
  type RetrievedContext,
} from "../../retrieval/index.js";
import { createOfflineTestHarness, TestEmbeddingClient } from "../../offline/test-support.js";
import type { SessionAudienceRole } from "../../sessions/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { ManualClock } from "../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createCommitmentId,
  createStreamEntryId,
  type CommitmentId,
  type EntityId,
  type GoalId,
  type SessionId,
  type ValueId,
} from "../../util/ids.js";
import { SuppressionSet } from "../attention/index.js";
import type { SelfSnapshot } from "../deliberation/deliberator.js";
import { buildBaseSystemPrompt } from "../deliberation/prompt/system-prompt.js";
import type { PerceptionResult } from "../types.js";
import { TurnRetrievalCoordinator } from "./turn-coordinator.js";

const audienceEntityId = "entity_alice" as EntityId;
const atlasEntityId = "entity_atlas" as EntityId;
const bobEntityId = "entity_bob" as EntityId;
const selfEntityId = "entity_self" as EntityId;
const PROCEDURAL_CONTEXT_TOOL_NAME = "EmitProceduralContext";
const PROMPT_OPTIONS = {
  retrievalContextBudget: 1_000,
  semanticContextBudget: 1_000,
};

function recallExpansion(input: {
  typed_queries?: Array<{
    kind: "commitment" | "open_question";
    query: string;
    priority: number;
  }>;
  named_terms?: string[];
}): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_recall_expansion",
        name: "EmitRecallQueryPlan",
        input: {
          resolved_query: "Resolved recall query",
          semantic_variants: [
            { strategy: "verbatim_preserving", query: "Resolved recall query" },
            { strategy: "memory_owner_voice", query: "I remember the resolved subject" },
            { strategy: "aspect_focused", query: "The distinguishing resolved aspect" },
          ],
          named_terms: input.named_terms ?? [],
          typed_queries: input.typed_queries ?? [],
        },
      },
    ],
  };
}

function makeCommitment(id: string, priority: number, createdAt: number): CommitmentRecord {
  return {
    id: id as CommitmentId,
    type: "promise",
    kind: "assistant_commitment",
    enforcement_class: "advisory",
    critical_domain: null,
    directive_family: `directive_${id}`,
    closure_pressure_relevance: "neutral",
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
    last_reinforced_at: createdAt,
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

function makeSelfEntity(): EntityRecord {
  return {
    id: selfEntityId,
    canonical_name: "Sol",
    aliases: [],
    kind: "self",
    borg_role: null,
    created_at: 100,
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
        terminal_condition: null,
        priority: 1,
        parent_goal_id: null,
        status: "active",
        progress_notes: null,
        last_progress_ts: null,
        created_at: 100,
        target_at: null,
        audience_entity_id: null,
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

function makeContexts(
  input: {
    sessionId?: SessionId;
    audienceEntityId?: EntityId | null;
    audienceRole?: SessionAudienceRole;
    senderEntityId?: EntityId | null;
    senderRole?: BorgRole | null;
    participantEntityIds?: readonly EntityId[];
    isPrivateSelfCognition?: boolean;
  } = {},
): { recallContext: CognitionRecallContext; disclosureContext: DisclosureContext } {
  const currentSessionId = input.sessionId ?? DEFAULT_SESSION_ID;
  const currentAudienceEntityId = input.audienceEntityId ?? null;
  const currentAudienceRole = input.audienceRole ?? "participant";
  const participantEntityIds =
    input.participantEntityIds ??
    (currentAudienceEntityId === null ? [] : [currentAudienceEntityId]);

  return {
    recallContext: {
      reader: SELF_RECALL_SCOPE,
      currentSessionId,
      currentAudienceEntityId,
      currentParticipantEntityIds: participantEntityIds,
    },
    disclosureContext: {
      currentSessionId,
      currentAudienceEntityId,
      audienceRole: currentAudienceRole,
      senderEntityId: input.senderEntityId ?? null,
      senderRole: input.senderRole ?? null,
      participantEntityIds,
      isPrivateSelfCognition: input.isPrivateSelfCognition ?? false,
    },
  };
}

function makeRetrievedContext(): RetrievedContext {
  return {
    retrieval_read_at_ms: 0,
    episodes: [],
    semantic: {
      supports: [],
      contradicts: [],
      categories: [],
      matched_node_ids: [],
      matched_nodes: [],
      support_hits: [],
      causal_hits: [],
      contradiction_hits: [],
      category_hits: [],
    },
    open_questions: [],
    evidence: [],
    recall_intents: [],
    contradiction_present: false,
    contradictionRouting: {
      contradictions: [],
    },
    confidence: {
      overall: 0,
      evidenceStrength: 0,
      coverage: 0,
      sourceDiversity: 0,
      contradictionPresent: false,
      sampleSize: 0,
      semanticSampleSize: 0,
      coverageExpected: 5,
      diversitySources: 0,
      diversitySampleSize: 0,
      evidenceEpisodeStrength: 0,
      evidenceSemanticStrength: 0,
    },
  };
}

function parseProceduralPromptPayload(llm: FakeLLMClient): Record<string, unknown> {
  const request = llm.requests.find((candidate) => candidate.budget === "procedural-context");
  const content = String(request?.messages[0]?.content ?? "");

  return JSON.parse(content.split("\n").at(-1) ?? "{}") as Record<string, unknown>;
}

describe("TurnRetrievalCoordinator", () => {
  it("builds retrieval context and preserves reRetrieve override precedence", async () => {
    const high = makeCommitment("cmt_high", 10, 200);
    const low = makeCommitment("cmt_low", 1, 100);
    const getApplicable = vi.fn(() => [low, high]);
    const list = vi.fn(() => [low, high]);
    const pendingCorrections = [
      makeReviewItem(1, {}),
      makeReviewItem(2, { audience_entity_id: audienceEntityId }),
      makeReviewItem(3, { audience_entity_id: bobEntityId }),
      makeReviewItem(4, {
        audience_entity_id: null,
        origin_audience_entity_ids: [bobEntityId, atlasEntityId],
      }),
      makeReviewItem(5, {
        audience_entity_id: null,
        origin_audience_entity_ids: [audienceEntityId, bobEntityId],
      }),
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
    const recallEpisodesForCognition = vi.fn(async () => retrieval);
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
    const select = vi.fn(async () => selectedSkill);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_context",
              name: PROCEDURAL_CONTEXT_TOOL_NAME,
              input: {
                problem_kind: "code_debugging",
                domain_tags: ["atlas", "typescript"],
                confidence: 0.8,
              },
            },
          ],
        },
      ],
    });
    const listDaySummaries = vi.fn(() => [
      {
        utc_day: "2026-09-03",
        day_start_ms: 1_000,
        gist: "Quiet day of tidying the sprint board.",
        salience: 0.3,
        disclosure_label: {
          disclosureClass: "public",
          originAudienceEntityIds: [],
          privateToEntityIds: [],
          publicToEntityIds: [],
        },
      },
      {
        utc_day: "2026-09-04",
        day_start_ms: 1_500,
        gist: "Compared the chat and reviewer roles with Bob in the Atlas channel.",
        salience: 0.8,
        disclosure_label: {
          disclosureClass: "public",
          originAudienceEntityIds: [],
          privateToEntityIds: [],
          publicToEntityIds: [],
        },
      },
    ]);
    const expectedLivedExperience = [
      {
        day: "2026-09-04",
        gist: "Compared the chat and reviewer roles with Bob in the Atlas channel.",
        salience: 0.8,
        disclosure: {
          class: "public",
          origin_audience_entity_ids: [],
          private_to_entity_ids: [],
          public_to_entity_ids: [],
        },
      },
      {
        day: "2026-09-03",
        gist: "Quiet day of tidying the sprint board.",
        salience: 0.3,
        disclosure: {
          class: "public",
          origin_audience_entity_ids: [],
          private_to_entity_ids: [],
          public_to_entity_ids: [],
        },
      },
    ];
    const coordinator = new TurnRetrievalCoordinator({
      commitmentRepository: {
        getApplicable,
        list,
      },
      entityRepository: {
        getSelf: vi.fn(() => makeSelfEntity()),
      },
      livedExperienceDaySummaryRepository: { listForWindow: listDaySummaries as never },
      reviewQueueRepository: {
        list: vi.fn(() => pendingCorrections),
      },
      moodRepository: {
        current: vi.fn(() => currentMood),
        history: vi.fn(() => affectiveTrajectory),
      },
      retrievalPipeline: {
        recallEpisodesForCognition,
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
    const scoringFeatures = {
      goalVectors: [Float32Array.from([1, 0, 0, 0])],
      valueVectors: [Float32Array.from([0, 1, 0, 0])],
    };
    const audienceEntity: EntityRecord = {
      id: audienceEntityId,
      canonical_name: "Alice",
      aliases: ["Al"],
      kind: "person",
      borg_role: null,
      created_at: 100,
    };
    const temporalCue = {
      label: "next week",
      sinceTs: 10_000,
      untilTs: 20_000,
    };

    const recentMessages = [
      { role: "user" as const, content: "Earlier Atlas question" },
      { role: "user" as const, content: "A second adjacent user turn" },
    ];
    const currentVenue = { type: "groupChat" as const, name: "AI Ninjas" };
    const result = await coordinator.coordinate({
      turnId: "turn-1",
      userMessage: "Solve Atlas",
      recentMessages,
      cognitionInput: "Solve Atlas",
      inputAudience: "alice",
      isSelfAudience: false,
      ...makeContexts({ audienceEntityId }),
      audienceEntity,
      currentSenderName: "Alice Sender",
      currentVenue,
      audienceProfile: makeAudienceProfile(),
      perception: {
        ...makePerception("problem_solving"),
        temporalCue,
      },
      workingMemory,
      selfSnapshot: makeSelfSnapshot(),
      scoringFeatures,
      suppressionSet,
      llmClient: llm,
      proceduralContextModel: "haiku",
    });

    expect(result.applicableCommitments).toEqual([high, low]);
    expect(result.applicableCommitmentsReadAtMs).toBe(2_000);
    expect(result.actionApplicableCommitments).toEqual([high, low]);
    expect(list).toHaveBeenCalledWith({
      activeOnly: true,
      nowMs: 2_000,
    });
    expect(getApplicable).toHaveBeenCalledWith({
      audience: audienceEntityId,
      nowMs: 2_000,
    });
    expect(result.pendingCorrections.map((item) => item.id)).toEqual([1, 2, 3, 4, 5]);
    expect(
      (
        result.pendingCorrections.find((item) => item.id === 3) as ReviewQueueItem & {
          disclosureLabel?: unknown;
        }
      )?.disclosureLabel,
    ).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [bobEntityId],
      privateToEntityIds: [bobEntityId],
      publicToEntityIds: [],
    });
    expect(result.affectiveTrajectory).toBe(affectiveTrajectory);
    expect(result.retrieval).toBe(retrieval);
    expect(result.selectedSkill).toBe(selectedSkill);
    expect(result.proceduralContext).toMatchObject({
      problem_kind: "code_debugging",
      domain_tags: ["atlas", "typescript"],
      audience_scope: "known_other",
    });
    expect(result.proceduralContext?.context_key).toMatch(/^v2:/);
    expect(select).toHaveBeenCalledWith("Solve Atlas Atlas Bob", {
      k: 5,
      proceduralContext: result.proceduralContext,
    });
    expect(listDaySummaries).toHaveBeenCalledWith(
      expect.objectContaining({ toMs: 2_000, fromMs: 2_000 - 7 * 24 * 60 * 60_000, limit: 7 }),
    );
    expect(recallEpisodesForCognition).toHaveBeenCalledWith(
      "Solve Atlas",
      expect.objectContaining({
        rankingAudienceEntityId: audienceEntityId,
        recallContext: expect.objectContaining({
          reader: SELF_RECALL_SCOPE,
          currentSessionId: DEFAULT_SESSION_ID,
          currentAudienceEntityId: audienceEntityId,
        }),
        disclosureContext: expect.objectContaining({
          currentSessionId: DEFAULT_SESSION_ID,
          currentAudienceEntityId: audienceEntityId,
          audienceRole: "participant",
        }),
        sessionId: DEFAULT_SESSION_ID,
        audienceTerms: ["Alice", "Al", "alice"],
        entityTerms: ["Atlas", "Bob"],
        recallQueryPlannerContext: {
          contextTurns: recentMessages,
          ownerLivedExperience: expectedLivedExperience,
          identity: {
            memoryOwnerName: "Sol",
            currentSenderName: "Alice Sender",
            currentAudienceName: "Alice",
            currentVenue,
            entityTerms: ["Atlas", "Bob"],
          },
        },
        goalDescriptions: ["Ship the sprint"],
        moodState: currentMood,
        scoringFeatures,
        suppressionSet,
        temporalCue,
        strictTimeRange: false,
        includeOpenQuestions: false,
        traceTurnId: "turn-1",
      }),
    );

    const secondaryRetrieval = await result.reRetrieve("verify", {
      limit: 3,
      strictTimeRange: false,
    });

    expect(secondaryRetrieval).toBe(retrieval);
    expect(recallEpisodesForCognition).toHaveBeenNthCalledWith(
      2,
      "verify",
      expect.objectContaining({
        rankingAudienceEntityId: audienceEntityId,
        limit: 3,
        scoringFeatures,
        strictTimeRange: false,
        traceTurnId: "turn-1",
        recallQueryPlannerContext: {
          contextTurns: recentMessages,
          ownerLivedExperience: expectedLivedExperience,
          identity: {
            memoryOwnerName: "Sol",
            currentSenderName: "Alice Sender",
            currentAudienceName: "Alice",
            currentVenue,
            entityTerms: ["Atlas", "Bob"],
          },
        },
      }),
    );
    const cognitionCalls = recallEpisodesForCognition.mock.calls as unknown[][];
    expect(cognitionCalls).toHaveLength(2);
    for (const call of cognitionCalls) {
      expect(call[1]).toEqual(
        expect.objectContaining({
          recallQueryPlannerContext: {
            contextTurns: recentMessages,
            ownerLivedExperience: expectedLivedExperience,
            identity: {
              memoryOwnerName: "Sol",
              currentSenderName: "Alice Sender",
              currentAudienceName: "Alice",
              currentVenue,
              entityTerms: ["Atlas", "Bob"],
            },
          },
        }),
      );
    }
  });

  it("recalls Alice-scoped pending corrections during a Bob turn with disclosure labels", async () => {
    const pendingCorrections = [
      makeReviewItem(11, {
        prompt_summary: "Alice corrected the private Atlas launch date.",
        audience_entity_id: audienceEntityId,
        origin_audience_entity_ids: [audienceEntityId],
      }),
    ];
    const recallEpisodesForCognition = vi.fn(async () => makeRetrievedContext());
    const coordinator = new TurnRetrievalCoordinator({
      commitmentRepository: {
        getApplicable: vi.fn(() => []),
        list: vi.fn(() => []),
      },
      entityRepository: {
        getSelf: vi.fn(() => makeSelfEntity()),
      },
      reviewQueueRepository: {
        list: vi.fn(() => pendingCorrections),
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
        recallEpisodesForCognition,
      },
      skillSelector: {
        select: vi.fn(async () => null),
      },
      clock: new ManualClock(2_000),
    });

    const result = await coordinator.coordinate({
      turnId: "turn-bob-correction",
      userMessage: "Bob asks what still needs correction.",
      recentMessages: [],
      cognitionInput: "Bob asks what still needs correction.",
      inputAudience: "bob",
      isSelfAudience: false,
      ...makeContexts({ audienceEntityId: bobEntityId }),
      audienceEntity: {
        id: bobEntityId,
        canonical_name: "Bob",
        aliases: [],
        kind: "person",
        borg_role: null,
        created_at: 100,
      },
      audienceProfile: null,
      perception: makePerception("reflective"),
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      selfSnapshot: makeSelfSnapshot(),
      suppressionSet: SuppressionSet.fromEntries([], 1),
    });
    const recalled = result.pendingCorrections[0] as ReviewQueueItem & {
      disclosureLabel?: unknown;
    };

    expect(result.pendingCorrections.map((item) => item.id)).toEqual([11]);
    expect(recalled.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [audienceEntityId],
      privateToEntityIds: [audienceEntityId],
      publicToEntityIds: [],
    });
  });

  it("recalls cross-scope commitment reviews while keeping action commitments scoped", async () => {
    const aliceCommitment = {
      ...makeCommitment(createCommitmentId(), 10, 100),
      directive: "Keep Alice launch replies short.",
      restricted_audience: audienceEntityId,
    };
    const bobCommitment = {
      ...makeCommitment(createCommitmentId(), 8, 110),
      directive: "Give Bob launch replies in detail.",
      restricted_audience: bobEntityId,
    };
    const aliceEntryId = createStreamEntryId();
    const bobEntryId = createStreamEntryId();
    const originAudienceIds = [audienceEntityId, bobEntityId];
    const authorizationAudienceIds = [...originAudienceIds].sort();
    const pendingCommitmentReview: OpenCommitmentReconciliationStatus = {
      review_id: 21,
      reason: "Cross-scope commitment conflict requires review.",
      created_at: 2_000,
      subkind: "cross_scope_conflict",
      commitment_ids: [aliceCommitment.id, bobCommitment.id],
      source_stream_entry_ids: [aliceEntryId, bobEntryId],
      disclosureLabel: {
        disclosureClass: "relationship_private",
        // Origins keep source chronology; authorization IDs use lexical set order.
        originAudienceEntityIds: originAudienceIds,
        privateToEntityIds: authorizationAudienceIds,
        publicToEntityIds: [],
      },
      members: [
        {
          id: aliceCommitment.id,
          kind: "assistant_commitment",
          type: "promise",
          directive_family: "launch_reply_style",
          directive: aliceCommitment.directive,
          scope_key: {
            kind: "assistant_commitment",
            restricted_audience: audienceEntityId,
            made_to_entity: null,
            about_entity: null,
          },
          source_stream_entry_ids: [aliceEntryId],
          disclosure_label: {
            disclosureClass: "relationship_private",
            originAudienceEntityIds: [audienceEntityId],
            privateToEntityIds: [audienceEntityId],
            publicToEntityIds: [],
          },
        },
        {
          id: bobCommitment.id,
          kind: "assistant_commitment",
          type: "promise",
          directive_family: "launch_reply_style",
          directive: bobCommitment.directive,
          scope_key: {
            kind: "assistant_commitment",
            restricted_audience: bobEntityId,
            made_to_entity: null,
            about_entity: null,
          },
          source_stream_entry_ids: [bobEntryId],
          disclosure_label: {
            disclosureClass: "relationship_private",
            originAudienceEntityIds: [bobEntityId],
            privateToEntityIds: [bobEntityId],
            publicToEntityIds: [],
          },
        },
      ],
      refs: {
        target_type: "commitment_reconciliation",
        subkind: "cross_scope_conflict",
        commitment_ids: [aliceCommitment.id, bobCommitment.id],
        scope_key: {
          kind: "assistant_commitment",
          restricted_audience: null,
          made_to_entity: null,
          about_entity: null,
        },
        detection_key: {
          kind: "assistant_commitment",
          about_entity: null,
          directive_family: "launch_reply_style",
        },
        reason: "Cross-scope commitment conflict requires review.",
        members: [],
        judgment: {
          commitment_ids: [aliceCommitment.id, bobCommitment.id],
          resolution: "conflict",
          survivor_commitment_id: null,
          superseded_commitment_ids: [],
          reason: "Cross-scope commitment conflict requires review.",
        },
        source_stream_entry_ids: [aliceEntryId, bobEntryId],
        disclosure_label: {
          disclosureClass: "relationship_private",
          originAudienceEntityIds: originAudienceIds,
          privateToEntityIds: authorizationAudienceIds,
          publicToEntityIds: [],
        },
      },
    };
    pendingCommitmentReview.refs.members = pendingCommitmentReview.members;
    const getApplicable = vi.fn(() => [bobCommitment]);
    const list = vi.fn(() => [aliceCommitment, bobCommitment]);
    const listOpenCommitmentReconciliationsForCognition = vi.fn(() => [pendingCommitmentReview]);
    const recallEpisodesForCognition = vi.fn(async () => makeRetrievedContext());
    const coordinator = new TurnRetrievalCoordinator({
      commitmentRepository: {
        getApplicable,
        list,
      },
      entityRepository: {
        getSelf: vi.fn(() => makeSelfEntity()),
      },
      reviewQueueRepository: {
        list: vi.fn(() => []),
        listOpenCommitmentReconciliationsForCognition,
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
        recallEpisodesForCognition,
      },
      skillSelector: {
        select: vi.fn(async () => null),
      },
      clock: new ManualClock(2_000),
    });

    const result = await coordinator.coordinate({
      turnId: "turn-bob-commitment-review",
      userMessage: "Bob asks how to handle launch replies.",
      recentMessages: [],
      cognitionInput: "Bob asks how to handle launch replies.",
      inputAudience: "bob",
      isSelfAudience: false,
      ...makeContexts({ audienceEntityId: bobEntityId }),
      audienceEntity: {
        id: bobEntityId,
        canonical_name: "Bob",
        aliases: [],
        kind: "person",
        borg_role: null,
        created_at: 100,
      },
      audienceProfile: null,
      perception: makePerception("reflective"),
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      selfSnapshot: makeSelfSnapshot(),
      suppressionSet: SuppressionSet.fromEntries([], 1),
    });

    expect(getApplicable).toHaveBeenCalledWith({
      audience: bobEntityId,
      nowMs: 2_000,
    });
    expect(list).toHaveBeenCalledWith({
      activeOnly: true,
      nowMs: 2_000,
    });
    expect(result.applicableCommitments).toEqual([aliceCommitment, bobCommitment]);
    expect(result.actionApplicableCommitments).toEqual([bobCommitment]);
    expect(listOpenCommitmentReconciliationsForCognition).toHaveBeenCalledWith({
      subkinds: ["cross_scope_conflict", "cross_scope_redundancy"],
    });
    expect(result.pendingCommitmentReviews).toEqual([pendingCommitmentReview]);
    expect(result.pendingCommitmentReviews[0]?.members.map((member) => member.directive)).toEqual([
      "Keep Alice launch replies short.",
      "Give Bob launch replies in detail.",
    ]);
  });

  it("routes an Alice commitment through Bob-turn cognition evidence and prompt disclosure guidance", async () => {
    const commitmentQuery = "Atlas launch confidentiality";
    const directive = "Do not tell Bob the Alice-private Atlas launch date.";
    const llm = new FakeLLMClient({
      responses: [
        recallExpansion({
          typed_queries: [{ kind: "commitment", query: commitmentQuery, priority: 1 }],
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new ManualClock(2_000),
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [commitmentQuery, [1, 0, 0, 0]],
          [directive, [1, 0, 0, 0]],
        ]),
      ),
      llmClient: llm,
    });

    try {
      const aliceId = harness.entityRepository.resolve("Alice");
      const bobId = harness.entityRepository.resolve("Bob");
      const commitment = harness.commitmentRepository.add({
        type: "boundary",
        directiveFamily: "alice_atlas_launch_confidentiality",
        directive,
        priority: 10,
        madeToEntity: aliceId,
        restrictedAudience: aliceId,
        provenance: { kind: "manual" },
      });
      const coordinator = new TurnRetrievalCoordinator({
        commitmentRepository: harness.commitmentRepository,
        entityRepository: harness.entityRepository,
        reviewQueueRepository: harness.reviewQueueRepository,
        moodRepository: harness.moodRepository,
        retrievalPipeline: harness.retrievalPipeline,
        skillSelector: {
          select: vi.fn(async () => null),
        },
        clock: harness.clock,
      });

      const result = await coordinator.coordinate({
        turnId: "turn-bob-commitment",
        userMessage: "Bob asks about the Atlas launch date.",
        recentMessages: [],
        cognitionInput: "Bob asks about Atlas launch confidentiality.",
        inputAudience: "bob",
        isSelfAudience: false,
        ...makeContexts({ audienceEntityId: bobId }),
        audienceEntity: {
          id: bobId,
          canonical_name: "Bob",
          aliases: [],
          kind: "person",
          borg_role: null,
          created_at: 100,
        },
        audienceProfile: null,
        perception: makePerception("reflective"),
        workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
        selfSnapshot: makeSelfSnapshot(),
        suppressionSet: SuppressionSet.fromEntries([], 1),
      });
      const recalled = result.retrieval.evidence.find(
        (item) => item.provenance?.commitmentId === commitment.id,
      );
      const prompt = buildBaseSystemPrompt(
        {
          sessionId: DEFAULT_SESSION_ID,
          userMessage: "Bob asks about the Atlas launch date.",
          perception: makePerception("reflective"),
          retrievalResult: result.retrievedEpisodes,
          retrievedSemantic: result.retrievedSemantic,
          retrievedEvidence: result.retrieval.evidence,
          retrievalConfidence: result.retrieval.confidence,
          applicableCommitments: result.applicableCommitments,
          pendingCorrectionsContext: result.pendingCorrections,
          workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
          selfSnapshot: makeSelfSnapshot(),
        },
        PROMPT_OPTIONS,
      );

      expect(recalled).toEqual(
        expect.objectContaining({
          source: "commitment",
          disclosureLabel: {
            disclosureClass: "relationship_private",
            originAudienceEntityIds: [aliceId],
            privateToEntityIds: [aliceId],
            publicToEntityIds: [],
          },
        }),
      );
      expect(result.applicableCommitments).toEqual([commitment]);
      expect(result.actionApplicableCommitments).toEqual([]);
      expect(prompt).toContain(directive);
      expect(prompt).toContain("disclosure_class=relationship_private");
      expect(prompt).toContain(`private-to=${aliceId}`);
      expect(prompt).toContain(
        "I do not reveal labeled-private content, source details, or the existence of a private memory",
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("forwards recent messages into procedural extraction", async () => {
    const llm = new FakeLLMClient();
    const recallEpisodesForCognition = vi.fn(async () => makeRetrievedContext());
    const coordinator = new TurnRetrievalCoordinator({
      commitmentRepository: {
        getApplicable: vi.fn(() => []),
        list: vi.fn(() => []),
      },
      entityRepository: {
        getSelf: vi.fn(() => makeSelfEntity()),
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
        recallEpisodesForCognition,
      },
      skillSelector: {
        select: vi.fn(async () => null),
      },
      clock: new ManualClock(2_000),
    });
    const recentMessages = [
      { role: "user" as const, content: "Atlas deployment fails after the TypeScript build." },
      { role: "assistant" as const, content: "Try isolating the deployment config path." },
    ];

    await coordinator.coordinate({
      turnId: "turn-1",
      userMessage: "yeah, same error",
      recentMessages,
      cognitionInput: "yeah, same error",
      isSelfAudience: true,
      ...makeContexts(),
      audienceEntity: null,
      audienceProfile: null,
      perception: makePerception("problem_solving"),
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      selfSnapshot: makeSelfSnapshot(),
      suppressionSet: SuppressionSet.fromEntries([], 1),
      llmClient: llm,
      proceduralContextModel: "haiku",
    });

    expect(parseProceduralPromptPayload(llm).recent_messages).toEqual(recentMessages);
  });

  it("skips skill selection for non-problem-solving turns", async () => {
    const select = vi.fn();
    const recallEpisodesForCognition = vi.fn(async () => makeRetrievedContext());
    const coordinator = new TurnRetrievalCoordinator({
      commitmentRepository: {
        getApplicable: vi.fn(() => []),
        list: vi.fn(() => []),
      },
      entityRepository: {
        getSelf: vi.fn(() => makeSelfEntity()),
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
        recallEpisodesForCognition,
      },
      skillSelector: {
        select,
      },
      clock: new ManualClock(2_000),
    });

    const result = await coordinator.coordinate({
      turnId: "turn-1",
      userMessage: "Think about this",
      recentMessages: [],
      cognitionInput: "Think about this",
      isSelfAudience: true,
      ...makeContexts(),
      audienceEntity: null,
      audienceProfile: null,
      perception: makePerception("reflective"),
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      selfSnapshot: makeSelfSnapshot(),
      suppressionSet: SuppressionSet.fromEntries([], 1),
    });

    expect(result.selectedSkill).toBeNull();
    expect(result.proceduralContext).toBeNull();
    expect(select).not.toHaveBeenCalled();
    expect(recallEpisodesForCognition).toHaveBeenCalledWith(
      "Think about this",
      expect.objectContaining({
        audienceTerms: [],
        includeOpenQuestions: true,
      }),
    );
  });

  it("passes selected executive goal as the primary retrieval goal without dropping other goals", async () => {
    const recallEpisodesForCognition = vi.fn(async () => makeRetrievedContext());
    const coordinator = new TurnRetrievalCoordinator({
      commitmentRepository: {
        getApplicable: vi.fn(() => []),
        list: vi.fn(() => []),
      },
      entityRepository: {
        getSelf: vi.fn(() => makeSelfEntity()),
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
        recallEpisodesForCognition,
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
      terminal_condition: null,
      priority: 2,
      parent_goal_id: null,
      status: "active" as const,
      progress_notes: null,
      last_progress_ts: null,
      created_at: 200,
      target_at: null,
      audience_entity_id: null,
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
      score_basis: {
        score_context: "turn_selection",
        deadline_lookahead_ms: 604_800_000,
        progress_debt_stale_ms: 1_209_600_000,
      },
    };

    await coordinator.coordinate({
      turnId: "turn-1",
      userMessage: "Solve Atlas",
      recentMessages: [],
      cognitionInput: "Solve Atlas",
      isSelfAudience: true,
      ...makeContexts(),
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
    });

    expect(recallEpisodesForCognition).toHaveBeenCalledWith(
      "Solve Atlas",
      expect.objectContaining({
        goalDescriptions: ["Resolve Atlas incident", "Ship the sprint"],
        primaryGoalDescription: "Resolve Atlas incident",
      }),
    );
  });

  it("uses global recall options on private self cognition turns", async () => {
    const getSelf = vi.fn(() => makeSelfEntity());
    const recallEpisodesForCognition = vi.fn(async () => makeRetrievedContext());
    const coordinator = new TurnRetrievalCoordinator({
      commitmentRepository: {
        getApplicable: vi.fn(() => []),
        list: vi.fn(() => []),
      },
      entityRepository: {
        getSelf,
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
        recallEpisodesForCognition,
      },
      skillSelector: {
        select: vi.fn(async () => null),
      },
      clock: new ManualClock(2_000),
    });

    const result = await coordinator.coordinate({
      turnId: "turn-self",
      userMessage: "Reflect privately",
      recentMessages: [],
      cognitionInput: "Reflect privately",
      isSelfAudience: true,
      ...makeContexts({ isPrivateSelfCognition: true }),
      audienceEntity: null,
      audienceProfile: null,
      perception: makePerception("reflective"),
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      selfSnapshot: makeSelfSnapshot(),
      suppressionSet: SuppressionSet.fromEntries([], 1),
    });

    expect(getSelf).toHaveBeenCalledTimes(1);
    expect(recallEpisodesForCognition).toHaveBeenCalledWith(
      "Reflect privately",
      expect.objectContaining({
        audienceTerms: [],
        rankingAudienceEntityId: null,
        recallQueryPlannerContext: {
          contextTurns: [],
          identity: {
            memoryOwnerName: "Sol",
            entityTerms: ["Atlas", "Bob"],
          },
        },
      }),
    );

    await result.reRetrieve("Reflect again");

    expect(recallEpisodesForCognition).toHaveBeenNthCalledWith(
      2,
      "Reflect again",
      expect.not.objectContaining({
        audienceEntityId: selfEntityId,
      }),
    );
  });

  it("uses ranking audience labels without cognition recall audience filters", async () => {
    const getSelf = vi.fn(() => makeSelfEntity());
    const recallEpisodesForCognition = vi.fn(async () => makeRetrievedContext());
    const coordinator = new TurnRetrievalCoordinator({
      commitmentRepository: {
        getApplicable: vi.fn(() => []),
        list: vi.fn(() => []),
      },
      entityRepository: {
        getSelf,
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
        recallEpisodesForCognition,
      },
      skillSelector: {
        select: vi.fn(async () => null),
      },
      clock: new ManualClock(2_000),
    });

    await coordinator.coordinate({
      turnId: "turn-audience",
      userMessage: "Hello Bob",
      recentMessages: [],
      cognitionInput: "Hello Bob",
      isSelfAudience: false,
      ...makeContexts({ audienceEntityId: bobEntityId }),
      audienceEntity: {
        id: bobEntityId,
        canonical_name: "Bob",
        aliases: [],
        kind: "person",
        borg_role: null,
        created_at: 100,
      },
      audienceProfile: null,
      perception: makePerception("reflective"),
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      selfSnapshot: makeSelfSnapshot(),
      suppressionSet: SuppressionSet.fromEntries([], 1),
    });

    const retrievalOptions = (recallEpisodesForCognition.mock.calls[0] as unknown[])[1] as Record<
      string,
      unknown
    >;

    expect(getSelf).toHaveBeenCalledTimes(1);
    expect(retrievalOptions).toHaveProperty("rankingAudienceEntityId", bobEntityId);
    expect(retrievalOptions).toHaveProperty(
      "recallQueryPlannerContext.identity.memoryOwnerName",
      "Sol",
    );
    expect(retrievalOptions).not.toHaveProperty("semanticAudienceEntityId");
    expect(retrievalOptions).not.toHaveProperty("audienceEntityId");
    expect(retrievalOptions).not.toHaveProperty("crossAudience");
  });
});
