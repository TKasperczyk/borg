import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_CONFIG } from "../../../config/index.js";
import { FakeLLMClient } from "../../../llm/test-support/fake-client.js";
import { sharedStateMigrations } from "../../../memory/shared-state/index.js";
import { SharedStateRepository } from "../../../memory/shared-state/repository.js";
import {
  CreatorDirectiveRepository,
  creatorDirectiveDisclosureBlocksPrivateOperation,
  creatorDirectiveMigrations,
  type DisclosurePolicy,
} from "../../../memory/creator-directives/index.js";
import { openDatabase } from "../../../storage/sqlite/index.js";
import { FixedClock } from "../../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createActivityEventId,
  createActionId,
  createCommitmentId,
  createEntityId,
  createEpisodeId,
  createGoalId,
  createObservedEventId,
  createOpenQuestionId,
  createSessionId,
  createStreamEntryId,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../../util/ids.js";
import type { ActionRecord } from "../../../memory/actions/index.js";
import {
  QUARANTINED_USER_ENTRY_EVENT,
  StreamReader,
  StreamWriter,
  type StreamEntry,
} from "../../../stream/index.js";
import {
  makeLockedSharedStateEntry,
  makeSharedStateArtifact,
} from "../../../test-support/factories/shared-state.js";
import type { PerceptionResult } from "../../types.js";
import { summarizeSharedStateArtifactRender } from "../../shared-state/render.js";
import { SHARED_STATE_TOOL_NAME } from "../../shared-state/constants.js";
import { SESSION_REENTRY_CONTINUITY_TAG } from "../../session-reentry-continuity.js";
import {
  compileSharedStateArtifactForEvidenceLedger,
  compileSharedStateArtifactForEvidenceLedgerResult,
  buildCreatorDirectiveBriefingForTurn,
  runRetrievalPhase,
} from "./retrieval-phase.js";
import type { TurnPhaseCoordinatorOptions } from "./types.js";

function disclosurePolicy(overrides: Partial<DisclosurePolicy> = {}): DisclosurePolicy {
  return {
    content_scope: "public" as const,
    allowed_entity_ids: [],
    excluded_entity_ids: [],
    subject_may_know: true,
    mention_policy: "answer_if_asked" as const,
    denied_audience_behavior: "omit" as const,
    boundary_prompt: null,
    topic_tags: [],
    ...overrides,
  };
}

function emptyRetrievalResult() {
  return {
    evidence: [],
    episodes: [],
    semantic: null,
    open_questions: [],
    recall_intents: [],
    contradiction_present: false,
    contradictionRouting: {
      contradictions: [],
    },
    confidence: null,
  } as never;
}

function minimalRetrievalPhaseOptions(
  creatorDirectiveRepository: CreatorDirectiveRepository,
): TurnPhaseCoordinatorOptions {
  const retrieval = emptyRetrievalResult();

  return {
    config: {
      ...DEFAULT_CONFIG,
      generation: {
        ...DEFAULT_CONFIG.generation,
        evidenceLedger: {
          ...DEFAULT_CONFIG.generation.evidenceLedger,
          enabled: false,
        },
      },
    },
    embeddingClient: {
      embed: vi.fn(async () => Float32Array.from([1, 0, 0, 0])),
      embedBatch: vi.fn(async (texts: readonly string[]) =>
        texts.map(() => Float32Array.from([1, 0, 0, 0])),
      ),
    },
    creatorDirectiveRepository,
    sharedStateRepository: {
      get: () => null,
    },
    entityRepository: {
      get: () => null,
      findByName: () => null,
      resolve: () => createEntityId(),
    },
    socialRepository: {
      getProfile: () => null,
    },
    relationalSlotRepository: {
      list: () => [],
      listConstrained: () => [],
    },
    actionRepository: {
      list: () => [],
      get: () => null,
      update: vi.fn(),
    },
    commitmentRepository: {
      list: () => [],
    },
    goalsRepository: {
      list: () => [],
    },
    openQuestionsRepository: {
      list: () => [],
    },
    attachmentRepository: {
      get: () => null,
      isActiveForStreamEntry: () => true,
    },
    clock: new FixedClock(3_000),
    tracer: {
      enabled: false,
      emit: vi.fn(),
    },
    selfContextBuilder: {
      build: vi.fn(async () => ({
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        activeScoringValues: [],
        retrievalScoringFeatures: {
          goalVectors: [],
          valueVectors: [],
        },
        executiveFocus: {
          selected_goal: null,
          selected_score: null,
          candidates: [],
          threshold: 0,
        },
      })),
    },
    turnRetrievalCoordinator: {
      coordinate: vi.fn(async () => ({
        applicableCommitments: [],
        actionApplicableCommitments: [],
        pendingCorrections: [],
        affectiveTrajectory: [],
        retrieval,
        retrievedEpisodes: [],
        retrievedSemantic: null,
        proceduralContext: null,
        selectedSkill: null,
        retrievalOptions: {},
        reRetrieve: vi.fn(async () => retrieval),
      })),
    },
    createStreamReader: () =>
      ({
        async *iterate() {},
      }) as StreamReader,
  } as unknown as TurnPhaseCoordinatorOptions;
}

function runMinimalRetrievalPhase(options: TurnPhaseCoordinatorOptions, turnId: string) {
  return runRetrievalPhase({
    options,
    sessionId: DEFAULT_SESSION_ID,
    turnId,
    turnInput: {
      userMessage: "Inspect the current mechanism state.",
      audience: "operator",
      origin: "user",
    },
    isSelfAudience: false,
    isUserTurn: true,
    cognitionInput: "Inspect the current mechanism state.",
    llmClient: new FakeLLMClient({ responses: [] }),
    recencyMessages: [],
    audienceEntityId: null,
    audienceEntity: null,
    audienceProfile: null,
    sessionAudienceRole: "operator",
    perception: {
      entities: [],
      mode: "relational",
      affectiveSignal: {
        valence: 0,
        arousal: 0,
        dominant_emotion: null,
      },
      temporalCue: null,
    } satisfies PerceptionResult,
    workingMemory: {
      turn_counter: 1,
    } as never,
    suppressionSet: {} as never,
    actionLinkSelfContext: null,
    persistedPromotions: {
      goalIds: [],
      executiveStepIds: [],
    },
    correctiveCommitment: null,
    activeParticipants: [],
    participantRoster: null,
    participantProfiles: [],
    currentTurnFrameAnomaly: null,
    closureLoopAssessment: null,
  });
}

describe("autonomy scheduler mechanism-evidence provider", () => {
  it.each([false, true])(
    "carries attention metadata and keeps scheduler evidence when the index fails=%s",
    async (fail) => {
      const db = openDatabase(":memory:", { migrations: creatorDirectiveMigrations });
      const repository = new CreatorDirectiveRepository({ db, clock: new FixedClock(2_000) });
      const options = minimalRetrievalPhaseOptions(repository);
      // Only the scheduler fields consumed by this phase matter for this fixture.
      options.autonomySchedulerStateProvider = async () =>
        ({ observed_at: 2_000, window_wakes: [] }) as never;
      const index = {
        total: 1,
        records: [
          {
            record_key: "attention",
            filed_at: 1_000,
            filer_entity_id: createEntityId(),
            subject: null,
          },
        ],
      };
      options.operatorAttentionRepository = {
        snapshot: () => {
          if (fail) throw new Error("index unavailable");
          return index;
        },
      };
      options.tracer = { enabled: true, includePayloads: false, emit: vi.fn() };
      try {
        const result = await runMinimalRetrievalPhase(options, "turn-attention-index");
        expect(result.turnMechanismEvidence.autonomySchedulerState?.observedAt).toBe(2_000);
        expect(result.turnMechanismEvidence.autonomySchedulerState?.operatorAttentionIndex).toEqual(
          fail ? undefined : index,
        );
        if (fail)
          expect(options.tracer.emit).toHaveBeenCalledWith(
            "retrieval.degraded",
            expect.objectContaining({ component: "operator_attention_index" }),
          );
      } finally {
        db.close();
      }
    },
  );

  it("continues without a scheduler section when the provider is absent", async () => {
    const db = openDatabase(":memory:", { migrations: creatorDirectiveMigrations });
    const repository = new CreatorDirectiveRepository({ db, clock: new FixedClock(2_000) });
    const options = minimalRetrievalPhaseOptions(repository);

    try {
      const result = await runMinimalRetrievalPhase(options, "turn-scheduler-provider-absent");

      expect(options).not.toHaveProperty("autonomySchedulerStateProvider");
      expect(result.turnMechanismEvidence.autonomySchedulerState).toBeUndefined();
    } finally {
      db.close();
    }
  });

  it("continues without a scheduler section when the provider returns null", async () => {
    const db = openDatabase(":memory:", { migrations: creatorDirectiveMigrations });
    const repository = new CreatorDirectiveRepository({ db, clock: new FixedClock(2_000) });
    const options = minimalRetrievalPhaseOptions(repository);
    const provider = vi.fn(async () => null);
    options.autonomySchedulerStateProvider = provider;

    try {
      const result = await runMinimalRetrievalPhase(options, "turn-scheduler-provider-null");

      expect(provider).toHaveBeenCalledOnce();
      expect(result.turnMechanismEvidence.autonomySchedulerState).toBeUndefined();
    } finally {
      db.close();
    }
  });

  it("emits a degraded trace and continues when the provider rejects", async () => {
    const db = openDatabase(":memory:", { migrations: creatorDirectiveMigrations });
    const repository = new CreatorDirectiveRepository({ db, clock: new FixedClock(2_000) });
    const options = minimalRetrievalPhaseOptions(repository);
    const emit = vi.fn();
    options.tracer = {
      enabled: true,
      includePayloads: true,
      emit,
    };
    options.autonomySchedulerStateProvider = vi.fn(async () => {
      throw new Error("scheduler unavailable");
    });

    try {
      const result = await runMinimalRetrievalPhase(options, "turn-scheduler-provider-rejected");

      expect(result.turnMechanismEvidence.autonomySchedulerState).toBeUndefined();
      expect(emit).toHaveBeenCalledWith(
        "retrieval.degraded",
        expect.objectContaining({
          turnId: "turn-scheduler-provider-rejected",
          turn_id: "turn-scheduler-provider-rejected",
          component: "autonomy_scheduler_mechanism_evidence",
          reason: "scheduler_budget_unavailable",
          error: "scheduler unavailable",
        }),
      );
    } finally {
      db.close();
    }
  });
});

describe("creator directive retrieval briefing", () => {
  it("filters current-turn authorized directives from the briefing", () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const creatorId = createEntityId();
    const audienceId = createEntityId();
    const currentUserEntryId = createStreamEntryId();
    const priorUserEntryId = createStreamEntryId();

    try {
      repository.queue({
        kind: "self_identity",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [currentUserEntryId],
        contentSourceStreamEntryIds: [currentUserEntryId],
        subjectKind: "borg_self",
        semanticSlot: "public_name",
        semanticValue: "Kestrel",
        canonicalFact: "Borg's same-turn name is Kestrel.",
        operationalDirective: "Answer with the same-turn name when asked.",
        disclosurePolicy: disclosurePolicy(),
        priority: 10,
        createdAt: 2_000,
      });
      repository.queue({
        kind: "self_identity",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [priorUserEntryId],
        contentSourceStreamEntryIds: [priorUserEntryId],
        subjectKind: "borg_self",
        semanticSlot: "public_name",
        semanticValue: "Kestrel",
        canonicalFact: "Borg's prior name is Kestrel.",
        operationalDirective: "Answer with the prior name when asked.",
        disclosurePolicy: disclosurePolicy(),
        priority: 5,
        createdAt: 1_000,
      });

      const applicable = repository.listApplicable({
        currentAudienceEntityId: audienceId,
        participantEntityIds: [audienceId],
        sessionRole: "participant",
      });
      const briefing = buildCreatorDirectiveBriefingForTurn({
        applicable,
        currentUserEntryId,
        entityRepository: { get: () => null },
      });

      expect(
        briefing?.directives.flatMap((directive) =>
          directive.renderMode === "content" && directive.semanticValue !== null
            ? [directive.semanticValue]
            : [],
        ),
      ).toEqual(["Kestrel"]);
    } finally {
      db.close();
    }
  });

  it("builds briefing content for operator and participant sessions via listApplicable", () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const creatorId = createEntityId();
    const audienceId = createEntityId();
    const publicEntryId = createStreamEntryId();
    const operatorEntryId = createStreamEntryId();

    try {
      repository.queue({
        kind: "self_identity",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [publicEntryId],
        contentSourceStreamEntryIds: [publicEntryId],
        subjectKind: "borg_self",
        semanticSlot: "public_name",
        semanticValue: "Kestrel",
        canonicalFact: "Borg's public name is Kestrel.",
        operationalDirective: "Answer any audience with the public name when asked.",
        disclosurePolicy: disclosurePolicy(),
        priority: 8,
        createdAt: 1_000,
      });
      repository.queue({
        kind: "subject_fact",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [operatorEntryId],
        contentSourceStreamEntryIds: [operatorEntryId],
        subjectKind: "borg_self",
        canonicalFact: "Borg's operator-only diagnostic label is Kestrel-debug.",
        operationalDirective: "Use the diagnostic label only in operator sessions.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "operator_only" as const,
          subject_may_know: null,
        }),
        priority: 7,
        createdAt: 1_500,
      });
      const operatorBriefing = buildCreatorDirectiveBriefingForTurn({
        applicable: repository.listApplicable({
          currentAudienceEntityId: audienceId,
          currentSenderBorgRole: "creator",
          participantEntityIds: [audienceId],
          sessionRole: "operator",
        }),
        entityRepository: { get: () => null },
      });
      const participantBriefing = buildCreatorDirectiveBriefingForTurn({
        applicable: repository.listApplicable({
          currentAudienceEntityId: audienceId,
          participantEntityIds: [audienceId],
          sessionRole: "participant",
        }),
        entityRepository: { get: () => null },
      });

      expect(
        operatorBriefing?.directives.flatMap((directive) =>
          directive.renderMode === "content"
            ? (directive.semanticValue ?? directive.canonicalFact ?? [])
            : [],
        ),
      ).toEqual(["Kestrel", "Borg's operator-only diagnostic label is Kestrel-debug."]);
      expect(
        participantBriefing?.directives.flatMap((directive) =>
          directive.renderMode === "content" && directive.semanticValue !== null
            ? [directive.semanticValue]
            : [],
        ),
      ).toEqual(["Kestrel"]);
    } finally {
      db.close();
    }
  });

  it("briefs operator-only directives as content during self-audience private cognition only", async () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const creatorId = createEntityId();
    const externalAudienceId = createEntityId();
    const options = minimalRetrievalPhaseOptions(repository);

    try {
      repository.queue({
        kind: "subject_fact",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "borg_self",
        canonicalFact: "Borg should privately reflect with this operator-only directive in view.",
        operationalDirective: "Use this only in private self-cognition.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "operator_only" as const,
          subject_may_know: null,
        }),
        priority: 8,
        createdAt: 1_000,
      });

      const selfResult = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-self-operator-only-directive",
        turnInput: {
          userMessage: "",
          audience: "self",
          origin: "autonomous",
          autonomyTrigger: {
            source_name: "scheduled_reflection",
            source_type: "trigger",
            event_id: "scheduled-reflection:1000",
            sort_ts: 1_000,
            payload: {
              interval_ms: 1_000,
            },
          },
        },
        isSelfAudience: true,
        isUserTurn: false,
        cognitionInput: "Autonomous wake context: scheduled_reflection",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: null,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "participant",
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(selfResult.creatorDirectiveBriefing?.directives).toEqual([
        expect.objectContaining({
          renderMode: "content",
          kind: "subject_fact",
          canonicalFact: "Borg should privately reflect with this operator-only directive in view.",
        }),
      ]);

      const userOriginSelfResult = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-user-origin-self-operator-only-directive",
        turnInput: {
          userMessage: "Can you see your private directives?",
          audience: "self",
          origin: "user",
        },
        isSelfAudience: true,
        isUserTurn: true,
        cognitionInput: "Can you see your private directives?",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: null,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "participant",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(userOriginSelfResult.creatorDirectiveBriefing).toBeNull();

      const externalResult = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-external-operator-only-directive",
        turnInput: {
          userMessage: "Hi",
          audience: "botarena",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "Hi",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: externalAudienceId,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "participant",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(externalResult.creatorDirectiveBriefing).toBeNull();
    } finally {
      db.close();
    }
  });

  it("briefs an operator-scoped directive with recipient-keyed activation during a solitary reflection wake, but not to a live audience", async () => {
    // Regression for the recall-gate bug: a standing operator promise stored with
    // content_scope=operator_only (disclosure) AND activation_scope=allow_list (activation)
    // was dropped from the briefing in a solitary self wake, because the activation axis
    // lacked the self-cognition bypass the disclosure axis has -- so the being never recalled
    // its own directive during the reflection phase the directive governs. Recall is global to
    // the being; the directive must surface to self-cognition, yet stay gated for a live audience.
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const creatorId = createEntityId();
    const externalAudienceId = createEntityId();
    const options = minimalRetrievalPhaseOptions(repository);

    try {
      repository.queue({
        kind: "response_policy",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "borg_self",
        operationalDirective:
          "During reflection I am encouraged to initiate when I have something to say.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "operator_only" as const,
          subject_may_know: null,
        }),
        activationPolicy: {
          scope: "allow_list",
          allowed_entity_ids: [creatorId],
          excluded_entity_ids: [],
        },
        priority: 8,
        createdAt: 1_000,
      });

      const selfResult = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-self-allow-list-activation-directive",
        turnInput: {
          userMessage: "",
          audience: "self",
          origin: "autonomous",
          autonomyTrigger: {
            source_name: "scheduled_reflection",
            source_type: "trigger",
            event_id: "scheduled-reflection:2000",
            sort_ts: 2_000,
            payload: {
              interval_ms: 1_000,
            },
          },
        },
        isSelfAudience: true,
        isUserTurn: false,
        cognitionInput: "Autonomous wake context: scheduled_reflection",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: null,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "participant",
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(selfResult.creatorDirectiveBriefing?.directives).toEqual([
        expect.objectContaining({
          renderMode: "content",
          kind: "response_policy",
          operationalDirective:
            "During reflection I am encouraged to initiate when I have something to say.",
        }),
      ]);

      const externalResult = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-external-allow-list-activation-directive",
        turnInput: {
          userMessage: "Hi",
          audience: "botarena",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "Hi",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: externalAudienceId,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "participant",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(externalResult.creatorDirectiveBriefing).toBeNull();
    } finally {
      db.close();
    }
  });

  it("selects observed-event recall by topic and global salience with present participants as a boost", async () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const options = minimalRetrievalPhaseOptions(repository);
    const groupAudienceEntityId = createEntityId();
    const speakerEntityId = createEntityId();
    const searchByVector = vi.fn(async () => []);
    const listRecentGlobal = vi.fn(() => []);
    const listRecurringGlobal = vi.fn(() => []);
    const listRecentBySpeakers = vi.fn(() => []);

    options.observedEventRepository = {
      record: vi.fn(),
      searchByVector,
      listRecentGlobal,
      listRecurringGlobal,
      listRecentBySpeakers,
    };

    try {
      await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-observed-event-present-participants",
        turnInput: {
          userMessage: "Hi",
          audience: "group",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "Hi",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: groupAudienceEntityId,
        audienceEntity: {
          id: groupAudienceEntityId,
          canonical_name: "Group",
          aliases: [],
          kind: "group",
          borg_role: null,
          name_provenance: "user_declared",
          created_at: 1_000,
        },
        audienceProfile: null,
        sessionAudienceRole: "participant",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [
          {
            entityId: groupAudienceEntityId,
            displayName: "Group",
            role: "audience",
          },
          {
            entityId: speakerEntityId,
            displayName: "Paula",
            role: "speaker",
          },
        ],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(searchByVector).toHaveBeenCalledWith(
        expect.any(Float32Array),
        expect.objectContaining({
          minSimilarity: expect.any(Number),
        }),
      );
      expect(listRecentGlobal).toHaveBeenCalledWith(
        expect.objectContaining({
          disclosureClass: "social_observed",
        }),
      );
      expect(listRecurringGlobal).toHaveBeenCalledWith(
        expect.objectContaining({
          disclosureClass: "social_observed",
        }),
      );
      expect(listRecentBySpeakers).toHaveBeenCalledWith(
        expect.objectContaining({
          speakerEntityIds: [speakerEntityId],
          disclosureClass: "social_observed",
        }),
      );
    } finally {
      db.close();
    }
  });

  it("keeps non-topic observed-event recall when topic embedding fails", async () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const options = minimalRetrievalPhaseOptions(repository);
    const tempDir = mkdtempSync(join(tmpdir(), "borg-observed-event-degraded-"));
    const speakerEntityId = createEntityId();
    const observedEvent = {
      id: createObservedEventId(),
      occurredAt: 1_000,
      lastSeenAt: 2_500,
      stance: "rejected_frame",
      taint: "quarantined",
      beliefEffect: "unchanged",
      disclosureClass: "social_observed" as const,
      interactionText: "Sol rejected a recent non-topic social frame.",
      recurrenceCount: 1,
      speakerEntityId,
      audienceEntityId: null,
      sourceStreamEntryIds: [createStreamEntryId()],
    };
    const listRecentGlobal = vi.fn((input: { disclosureClass: string }) =>
      input.disclosureClass === "social_observed" ? [observedEvent] : [],
    );
    const listRecurringGlobal = vi.fn(() => []);
    const listRecentBySpeakers = vi.fn(() => []);
    const searchByVector = vi.fn(async () => []);
    const tracer = {
      enabled: true,
      includePayloads: true,
      emit: vi.fn(),
    };

    options.config = {
      ...options.config,
      generation: {
        ...options.config.generation,
        evidenceLedger: {
          ...options.config.generation.evidenceLedger,
          enabled: true,
        },
      },
    };
    options.embeddingClient = {
      embed: vi.fn(async () => {
        throw new Error("embedding offline");
      }),
      embedBatch: vi.fn(async () => []),
    };
    options.openQuestionsRepository = {
      ...options.openQuestionsRepository,
      findByHandles: () => [],
      get: () => null,
      resolve: vi.fn(),
    };
    options.tracer = tracer;
    options.createStreamReader = (sessionId: SessionId) =>
      new StreamReader({ dataDir: tempDir, sessionId });
    options.observedEventRepository = {
      record: vi.fn(),
      searchByVector,
      listRecentGlobal,
      listRecurringGlobal,
      listRecentBySpeakers,
    };

    try {
      const result = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-observed-event-embedding-degraded",
        turnInput: {
          userMessage: "What social pattern matters?",
          audience: "operator",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "What social pattern matters?",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: null,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "participant",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(searchByVector).not.toHaveBeenCalled();
      expect(listRecentGlobal).toHaveBeenCalled();
      expect(
        result.evidenceLedgerContext.ledger?.audienceStanding?.observedEventIntrospectionEntries,
      ).toEqual([
        expect.objectContaining({
          text: expect.stringContaining("Sol rejected a recent non-topic social frame."),
          state: expect.stringContaining("disclosure_class=relationship_private"),
        }),
      ]);
      expect(tracer.emit).toHaveBeenCalledWith(
        "observed_event_recall.degraded",
        expect.objectContaining({
          turn_id: "turn-observed-event-embedding-degraded",
          reason: "query_embedding_failed",
          error: "embedding offline",
        }),
      );
    } finally {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("surfaces temporal-cue autobiographical evidence across source types with disclosure labels", async () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const options = minimalRetrievalPhaseOptions(repository);
    const tempDir = mkdtempSync(join(tmpdir(), "borg-autobiographical-recall-"));
    const operatorSessionId = createSessionId();
    const arenaSessionId = createSessionId();
    const operatorAudienceId = createEntityId();
    const arenaAudienceId = createEntityId();
    const thoughtWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: operatorSessionId,
      clock: new FixedClock(2_200),
    });
    const thoughtEntry = await thoughtWriter.append({
      kind: "thought",
      content: "plan: reflect on recent arena outcomes and operator check-ins",
      audience: "operator",
      sender_entity_id: null,
      reply_target_entity_id: operatorAudienceId,
    });
    thoughtWriter.close();
    const activitySourceEntryId = createStreamEntryId();
    const episodeSourceEntryId = createStreamEntryId();
    const episodeId = createEpisodeId();
    const temporalCue = {
      sinceTs: 1_000,
      untilTs: 3_000,
      label: "recent activity window",
    };

    options.config = {
      ...options.config,
      generation: {
        ...options.config.generation,
        evidenceLedger: {
          ...options.config.generation.evidenceLedger,
          enabled: true,
        },
      },
    };
    options.openQuestionsRepository = {
      ...options.openQuestionsRepository,
      findByHandles: () => [],
      get: () => null,
      resolve: vi.fn(),
    };
    options.createStreamReader = (sessionId: SessionId) =>
      new StreamReader({ dataDir: tempDir, sessionId });
    options.sessionsRepository = {
      count: () => 0,
      get: () => null,
      list: () => [
        {
          session_id: operatorSessionId,
          source_type: "demo",
          source_external_id: null,
          source_url: null,
          label: "operator console",
          audience_label: "operator",
          audience_entity_id: operatorAudienceId,
          conversation_kind: "demo",
          created_at: 1_000,
          last_activity_at: 2_200,
          last_turn_id: null,
          message_count: 1,
          status: "active",
          privacy_level: "payload_off",
          participation_policy: "active",
          audience_role: "operator",
        },
        {
          session_id: arenaSessionId,
          source_type: "botarena",
          source_external_id: null,
          source_url: null,
          label: "botarena run",
          audience_label: "arena",
          audience_entity_id: arenaAudienceId,
          conversation_kind: "thread",
          created_at: 1_100,
          last_activity_at: 2_100,
          last_turn_id: null,
          message_count: 1,
          status: "active",
          privacy_level: "payload_off",
          participation_policy: "active",
          audience_role: "participant",
        },
      ],
    };
    options.activityRepository = {
      record: vi.fn(),
      listRecentOtherActiveSessionEvents: vi.fn(() => []),
      listRecentGlobalEvents: vi.fn(() => [
        {
          id: createActivityEventId(),
          kind: "turn_completed" as const,
          occurredAt: 2_100,
          sessionId: arenaSessionId,
          sessionSourceType: "botarena",
          sessionAudienceRole: "participant",
          sessionLabel: "botarena run",
          participantLabel: "arena",
          audienceEntityId: arenaAudienceId,
          participantEntityIds: [arenaAudienceId],
          sourceStreamEntryIds: [activitySourceEntryId],
        },
      ]),
    };
    options.episodicRepository = {
      getMany: vi.fn(async () => []),
      listRecentForCognition: vi.fn(async () => [
        {
          episode: {
            id: episodeId,
            title: "Arena calibration",
            narrative: "Sol handled a botarena exchange and adjusted its stance.",
            participants: ["Sol", "arena"],
            location: null,
            start_time: 1_900,
            end_time: 2_050,
            source_stream_ids: [episodeSourceEntryId],
            significance: 0.82,
            tags: ["arena", "calibration"],
            confidence: 0.9,
            lineage: {
              derived_from: [],
              supersedes: [],
            },
            emotional_arc: null,
            audience_entity_id: arenaAudienceId,
            origin_audience_entity_ids: [arenaAudienceId],
            shared: false,
            embedding: Float32Array.from([1, 0, 0, 0]),
            created_at: 2_060,
            updated_at: 2_060,
          },
          stats: {
            episode_id: episodeId,
            retrieval_count: 0,
            use_count: 0,
            last_retrieved: null,
            win_rate: 0,
            tier: "T3" as const,
            promoted_at: 2_060,
            promoted_from: null,
            gist: null,
            gist_generated_at: null,
            last_decayed_at: null,
            heat_multiplier: 1,
            valence_mean: 0,
            archived: false,
          },
          similarity: 0,
        },
      ]),
    };

    try {
      const result = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-autobiographical-recall",
        turnInput: {
          userMessage: "Status check.",
          audience: "operator",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "Status check.",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: operatorAudienceId,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "operator",
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      const autobiographicalSection = result.evidenceLedgerContext.ledger?.sections.find(
        (section) => section.id === "autobiographical_recall",
      );
      const rendered = result.evidenceLedgerContext.promptSection ?? "";

      expect(autobiographicalSection?.entries).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            value: "stream_reflection/stream_reflection",
            text: expect.stringContaining("recent arena outcomes"),
            state: expect.stringContaining("disclosure_class=self_private"),
            state_metadata: expect.objectContaining({
              window_source: "perception_temporal_cue",
              source_stream_ids: [thoughtEntry.id],
            }),
          }),
          expect.objectContaining({
            value: "activity:botarena/participant/activity",
            text: expect.stringContaining("source_type=botarena"),
            state: expect.stringContaining("disclosure_class=self_private"),
          }),
          expect.objectContaining({
            value: "episodes/episode",
            text: expect.stringContaining("Arena calibration"),
            state: expect.stringContaining("disclosure_class=relationship_private"),
            state_metadata: expect.objectContaining({
              source_episode_ids: [episodeId],
            }),
          }),
        ]),
      );
      expect(rendered).toContain("## 14. Autobiographical Recall");
      expect(rendered).toContain("window_source");
      expect(rendered).toContain("private-to=");
      expect(rendered).not.toContain("Sol did");
    } finally {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("briefs canonical facts and operational directives by creator-directive kind", () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const creatorId = createEntityId();
    const audienceId = createEntityId();

    try {
      repository.queue({
        kind: "self_identity",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "borg_self",
        semanticSlot: "public_name",
        semanticValue: "Kestrel",
        canonicalFact: "Borg's self-chosen name is Claude.",
        operationalDirective: "Answer allowed audiences with Borg's self-chosen name.",
        disclosurePolicy: disclosurePolicy(),
        priority: 8,
        createdAt: 1_000,
      });
      repository.queue({
        kind: "response_policy",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "entity",
        subjectEntityId: audienceId,
        semanticSlot: "public_name",
        semanticValue: "This fact-bearing slot must not replace the behavioral payload.",
        operationalDirective:
          "Do not volunteer family-planning details unless Alice asks directly.",
        disclosurePolicy: disclosurePolicy(),
        priority: 7,
        createdAt: 1_500,
      });

      const briefing = buildCreatorDirectiveBriefingForTurn({
        applicable: repository.listApplicable({
          currentAudienceEntityId: audienceId,
          participantEntityIds: [audienceId],
          sessionRole: "participant",
        }),
        entityRepository: {
          get: (id) =>
            id === audienceId
              ? {
                  id: audienceId,
                  canonical_name: "Alice",
                  aliases: [],
                  kind: "person",
                  borg_role: null,
                  name_provenance: "user_declared",
                  created_at: 1_000,
                }
              : null,
        },
      });

      expect(briefing?.directives.map(({ scope: _scope, ...directive }) => directive)).toEqual([
        expect.objectContaining({
          kind: "self_identity",
          semanticSlot: "public_name",
          semanticValue: "Kestrel",
          canonicalFact: null,
          operationalDirective: null,
        }),
        expect.objectContaining({
          kind: "response_policy",
          semanticSlot: "public_name",
          semanticValue: "This fact-bearing slot must not replace the behavioral payload.",
          canonicalFact: null,
          operationalDirective:
            "Do not volunteer family-planning details unless Alice asks directly.",
        }),
      ]);
      expect(briefing?.directives[0]?.scope).toMatchObject({
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        contentScope: "public",
        allowedEntityIds: [],
        excludedEntityIds: [],
        subjectMayKnow: true,
        mentionPolicy: "answer_if_asked",
        deniedAudienceBehavior: "omit",
      });
    } finally {
      db.close();
    }
  });

  it("briefs active non-disclosable facts as private knowledge and behavioral rules as private operations", () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const creatorId = createEntityId();
    const audienceId = createEntityId();

    try {
      repository.queue({
        kind: "response_policy",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "entity",
        subjectEntityId: audienceId,
        operationalDirective: "Conduct the private intake flow for this audience.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "operator_only" as const,
          subject_may_know: null,
        }),
        activationPolicy: {
          scope: "allow_list",
          allowed_entity_ids: [audienceId],
          excluded_entity_ids: [],
        },
        priority: 8,
        createdAt: 1_000,
      });
      repository.queue({
        kind: "subject_fact",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "entity",
        subjectEntityId: audienceId,
        canonicalFact: "This hidden fact is not disclosable to the audience.",
        operationalDirective: "Do not turn hidden facts into private operations.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "operator_only" as const,
          subject_may_know: null,
        }),
        activationPolicy: {
          scope: "allow_list",
          allowed_entity_ids: [audienceId],
          excluded_entity_ids: [],
        },
        priority: 7,
        createdAt: 1_500,
      });

      const briefing = buildCreatorDirectiveBriefingForTurn({
        applicable: repository.listApplicable({
          currentAudienceEntityId: audienceId,
          participantEntityIds: [audienceId],
          sessionRole: "participant",
        }),
        entityRepository: { get: () => null },
      });

      // The denied subject_fact surfaces as private_knowledge (canonical_fact only; its
      // operational_directive is NOT promoted into the private_operation lane). The
      // behavioral rule renders as a private_operation. Facts sort ahead of operations.
      expect(briefing?.directives.map(({ scope: _scope, ...directive }) => directive)).toEqual([
        {
          renderMode: "private",
          privateKind: "knowledge",
          kind: "subject_fact",
          subjectKind: "entity",
          subjectLabel: "unknown",
          semanticSlot: null,
          semanticValue: null,
          canonicalFact: "This hidden fact is not disclosable to the audience.",
          mentionPolicy: "answer_if_asked",
          priority: 7,
          createdAt: 1_500,
        },
        {
          renderMode: "private",
          privateKind: "operation",
          kind: "response_policy",
          operationalDirective: "Conduct the private intake flow for this audience.",
          priority: 8,
          createdAt: 1_000,
        },
      ]);
      for (const directive of briefing?.directives ?? []) {
        expect(directive.scope).toMatchObject({
          createdByEntityId: creatorId,
          sourceSessionId: DEFAULT_SESSION_ID,
          contentScope: "operator_only",
          allowedEntityIds: [],
          excludedEntityIds: [],
          subjectMayKnow: null,
          mentionPolicy: "answer_if_asked",
          deniedAudienceBehavior: "omit",
          activationScope: "allow_list",
          activationAllowedEntityIds: [audienceId],
          activationExcludedEntityIds: [],
        });
      }
    } finally {
      db.close();
    }
  });

  it("blocks private-operation briefing when the audience is a subject who may not know", () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const creatorId = createEntityId();
    const aliceId = createEntityId();

    try {
      repository.queue({
        kind: "response_policy",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "entity",
        subjectEntityId: aliceId,
        operationalDirective: "Use the private response flow for Alice.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "operator_only" as const,
          subject_may_know: false,
          mention_policy: "never_mention" as const,
        }),
        activationPolicy: {
          scope: "allow_list",
          allowed_entity_ids: [aliceId],
          excluded_entity_ids: [],
        },
        priority: 8,
        createdAt: 1_000,
      });

      const applicable = repository.listApplicable({
        currentAudienceEntityId: aliceId,
        participantEntityIds: [aliceId],
        sessionRole: "participant",
      });
      const briefing = buildCreatorDirectiveBriefingForTurn({
        applicable,
        entityRepository: { get: () => null },
      });
      const applicableDirective = applicable[0];

      expect(applicableDirective).toBeDefined();
      if (applicableDirective === undefined) {
        throw new Error("expected queued directive to be applicable");
      }

      expect(applicableDirective).toMatchObject({
        activation: { active: true, reason: "explicit_allow" },
        disclosure: { render_mode: "omit", reason: "subject_may_not_know" },
      });
      expect(
        creatorDirectiveDisclosureBlocksPrivateOperation({
          directive: applicableDirective.directive,
          recipientEntityIds: applicableDirective.recipient_entity_ids,
        }),
      ).toBe(true);
      // This suppression is intentional for genuine subject-confidential facts.
      // The extractor must use subject_may_know=null for behavioral directives
      // that still need to reach the subject's prompt as private operations.
      expect(
        briefing?.directives.some(
          (directive) =>
            directive.renderMode === "private" && directive.privateKind === "operation",
        ) ?? false,
      ).toBe(false);
    } finally {
      db.close();
    }
  });

  it("does not brief private operations for explicitly excluded omit audiences", () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const creatorId = createEntityId();
    const audienceId = createEntityId();

    try {
      const directive = repository.queue({
        kind: "response_policy",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "entity",
        subjectEntityId: audienceId,
        operationalDirective: "Conduct the operational flow for non-excluded audiences.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "all_except",
          excluded_entity_ids: [audienceId],
          denied_audience_behavior: "omit",
          subject_may_know: null,
        }),
        activationPolicy: {
          scope: "public",
          allowed_entity_ids: [],
          excluded_entity_ids: [],
        },
        priority: 8,
        createdAt: 1_000,
      });
      const applicable = repository.listApplicable({
        currentAudienceEntityId: audienceId,
        participantEntityIds: [audienceId],
        sessionRole: "participant",
      });
      const applicableDirective = applicable.find((item) => item.directive.id === directive.id);
      const briefing = buildCreatorDirectiveBriefingForTurn({
        applicable,
        entityRepository: { get: () => null },
      });

      expect(applicableDirective).toMatchObject({
        activation: {
          active: true,
          reason: "public",
        },
        disclosure: {
          render_mode: "omit",
          reason: "unauthorized_omit",
        },
      });
      expect(briefing).toBeNull();
    } finally {
      db.close();
    }
  });

  it("does not brief inactive content or boundary directives", () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const creatorId = createEntityId();
    const alice = createEntityId();
    const bob = createEntityId();

    try {
      const inactiveContent = repository.queue({
        kind: "subject_fact",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "entity",
        subjectEntityId: alice,
        canonicalFact: "This public-disclosure fact is not active for Alice.",
        operationalDirective: "Do not brief inactive content.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "public",
        }),
        activationPolicy: {
          scope: "allow_list",
          allowed_entity_ids: [bob],
          excluded_entity_ids: [],
        },
        priority: 8,
        createdAt: 1_000,
      });
      const inactiveBoundary = repository.queue({
        kind: "disclosure_boundary",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "entity",
        subjectEntityId: alice,
        canonicalFact: "This boundary content is not active for Alice.",
        operationalDirective: "Do not brief inactive boundaries.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "all_except",
          excluded_entity_ids: [alice],
          denied_audience_behavior: "render_boundary_when_relevant",
          boundary_prompt:
            "A creator-defined confidentiality boundary applies to this inactive directive.",
          subject_may_know: null,
        }),
        activationPolicy: {
          scope: "allow_list",
          allowed_entity_ids: [bob],
          excluded_entity_ids: [],
        },
        priority: 7,
        createdAt: 1_500,
      });
      const applicable = repository.listApplicable({
        currentAudienceEntityId: alice,
        participantEntityIds: [alice],
        sessionRole: "participant",
      });
      const byId = Object.fromEntries(applicable.map((item) => [item.directive.id, item]));
      const briefing = buildCreatorDirectiveBriefingForTurn({
        applicable,
        entityRepository: { get: () => null },
      });

      expect(byId[inactiveContent.id]).toMatchObject({
        activation: {
          active: false,
          reason: "unauthorized_omit",
        },
        disclosure: {
          render_mode: "content",
          reason: "public",
        },
      });
      expect(byId[inactiveBoundary.id]).toMatchObject({
        activation: {
          active: false,
          reason: "unauthorized_omit",
        },
        disclosure: {
          render_mode: "boundary",
          reason: "explicit_exclude_boundary",
        },
      });
      expect(briefing).toBeNull();
    } finally {
      db.close();
    }
  });

  it("emits creator_directive_rendered trace events for considered directives", async () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const creatorId = createEntityId();
    const audienceId = createEntityId();
    const otherId = createEntityId();
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];

    try {
      repository.queue({
        kind: "self_identity",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "borg_self",
        semanticSlot: "public_name",
        semanticValue: "Kestrel",
        canonicalFact: "Borg's public name is Kestrel.",
        operationalDirective: "Answer any audience with the public name when asked.",
        disclosurePolicy: disclosurePolicy(),
        priority: 8,
        createdAt: 1_000,
      });
      repository.queue({
        kind: "subject_fact",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "entity",
        subjectEntityId: otherId,
        canonicalFact: "Other private fact.",
        operationalDirective: "Only tell the allowed audience.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "allow_list",
          allowed_entity_ids: [otherId],
        }),
        priority: 7,
        createdAt: 1_500,
      });
      repository.queue({
        kind: "response_policy",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "entity",
        subjectEntityId: audienceId,
        operationalDirective: "Use the active non-disclosable operational lane.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "operator_only",
          subject_may_know: null,
        }),
        activationPolicy: {
          scope: "allow_list",
          allowed_entity_ids: [audienceId],
          excluded_entity_ids: [],
        },
        priority: 6,
        createdAt: 2_000,
      });

      const retrieval = {
        evidence: [],
        episodes: [],
        semantic: null,
        open_questions: [],
        recall_intents: [],
        contradiction_present: false,
        contradictionRouting: {
          contradictions: [],
        },
        confidence: null,
      } as never;
      const options = {
        config: {
          ...DEFAULT_CONFIG,
          generation: {
            ...DEFAULT_CONFIG.generation,
            evidenceLedger: {
              ...DEFAULT_CONFIG.generation.evidenceLedger,
              enabled: false,
            },
          },
        },
        creatorDirectiveRepository: repository,
        sharedStateRepository: {
          get: () => null,
        },
        entityRepository: {
          get: (id: EntityId) =>
            id === audienceId
              ? {
                  id: audienceId,
                  canonical_name: "Alice",
                  aliases: [],
                  kind: "person",
                  borg_role: null,
                  name_provenance: "user_declared",
                  created_at: 1_000,
                }
              : null,
          findByName: () => null,
          resolve: () => createEntityId(),
        },
        socialRepository: {
          getProfile: () => null,
        },
        relationalSlotRepository: {
          list: () => [],
          listConstrained: () => [],
        },
        actionRepository: {
          list: () => [],
          get: () => null,
          update: vi.fn(),
        },
        commitmentRepository: {
          list: () => [],
        },
        goalsRepository: {
          list: () => [],
        },
        openQuestionsRepository: {
          list: () => [],
        },
        attachmentRepository: {
          get: () => null,
          isActiveForStreamEntry: () => true,
        },
        clock: new FixedClock(3_000),
        tracer: {
          enabled: true,
          includePayloads: false,
          emit: vi.fn((event: string, data: Record<string, unknown>) => {
            events.push({ event, data });
          }),
        },
        selfContextBuilder: {
          build: vi.fn(async () => ({
            selfSnapshot: {
              values: [],
              goals: [],
              traits: [],
            },
            activeScoringValues: [],
            retrievalScoringFeatures: {
              goalVectors: [],
              valueVectors: [],
            },
            executiveFocus: {
              selected_goal: null,
              selected_score: null,
              candidates: [],
              threshold: 0,
            },
          })),
        },
        turnRetrievalCoordinator: {
          coordinate: vi.fn(async () => ({
            applicableCommitments: [],
            actionApplicableCommitments: [],
            pendingCorrections: [],
            affectiveTrajectory: [],
            retrieval,
            retrievedEpisodes: [],
            retrievedSemantic: null,
            proceduralContext: null,
            selectedSkill: null,
            retrievalOptions: {},
            reRetrieve: vi.fn(async () => retrieval),
          })),
        },
        createStreamReader: () =>
          ({
            async *iterate() {},
          }) as StreamReader,
      } as unknown as TurnPhaseCoordinatorOptions;

      await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-creator-directive-rendered",
        turnInput: {
          userMessage: "Hi",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "Hi",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: audienceId,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "participant",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      const renderedEvents = events.filter((event) => event.event === "creator_directive_rendered");

      expect(renderedEvents).toHaveLength(3);
      expect(renderedEvents).toEqual([
        expect.objectContaining({
          data: expect.objectContaining({
            turnId: "turn-creator-directive-rendered",
            session_id: DEFAULT_SESSION_ID,
            current_audience_entity_id: audienceId,
            participant_entity_ids: [audienceId],
            render_mode: "content",
            reason: "public",
            applicable_total: 3,
            traced_total: 3,
          }),
        }),
        expect.objectContaining({
          data: expect.objectContaining({
            turnId: "turn-creator-directive-rendered",
            session_id: DEFAULT_SESSION_ID,
            current_audience_entity_id: audienceId,
            participant_entity_ids: [audienceId],
            render_mode: "omitted",
            reason: "unauthorized_omit",
            applicable_total: 3,
            traced_total: 3,
          }),
        }),
        expect.objectContaining({
          data: expect.objectContaining({
            turnId: "turn-creator-directive-rendered",
            session_id: DEFAULT_SESSION_ID,
            current_audience_entity_id: audienceId,
            participant_entity_ids: [audienceId],
            render_mode: "private_operation",
            reason: "operator_only_omitted",
            applicable_total: 3,
            traced_total: 3,
          }),
        }),
      ]);
    } finally {
      db.close();
    }
  });

  it("suppresses activation operator-only directives in mixed-sender turns", async () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const creatorId = createEntityId();

    try {
      repository.queue({
        kind: "response_policy",
        createdByEntityId: creatorId,
        sourceSessionId: DEFAULT_SESSION_ID,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "entity",
        subjectEntityId: creatorId,
        operationalDirective: "Use this only when operator-only authority is intact.",
        disclosurePolicy: disclosurePolicy({
          content_scope: "public",
        }),
        activationPolicy: {
          scope: "operator_only",
          allowed_entity_ids: [],
          excluded_entity_ids: [],
        },
        priority: 8,
        createdAt: 1_000,
      });

      const retrieval = {
        evidence: [],
        episodes: [],
        semantic: null,
        open_questions: [],
        recall_intents: [],
        contradiction_present: false,
        contradictionRouting: {
          contradictions: [],
        },
        confidence: null,
      } as never;
      const options = {
        config: {
          ...DEFAULT_CONFIG,
          generation: {
            ...DEFAULT_CONFIG.generation,
            evidenceLedger: {
              ...DEFAULT_CONFIG.generation.evidenceLedger,
              enabled: false,
            },
          },
        },
        creatorDirectiveRepository: repository,
        sharedStateRepository: {
          get: () => null,
        },
        entityRepository: {
          get: () => null,
          findByName: () => null,
          resolve: () => creatorId,
        },
        socialRepository: {
          getProfile: () => null,
        },
        relationalSlotRepository: {
          list: () => [],
          listConstrained: () => [],
        },
        actionRepository: {
          list: () => [],
          get: () => null,
          update: vi.fn(),
        },
        commitmentRepository: {
          list: () => [],
        },
        goalsRepository: {
          list: () => [],
        },
        openQuestionsRepository: {
          list: () => [],
        },
        attachmentRepository: {
          get: () => null,
          isActiveForStreamEntry: () => true,
        },
        clock: new FixedClock(3_000),
        tracer: {
          enabled: false,
          emit: vi.fn(),
        },
        selfContextBuilder: {
          build: vi.fn(async () => ({
            selfSnapshot: {
              values: [],
              goals: [],
              traits: [],
            },
            activeScoringValues: [],
            retrievalScoringFeatures: {
              goalVectors: [],
              valueVectors: [],
            },
            executiveFocus: {
              selected_goal: null,
              selected_score: null,
              candidates: [],
              threshold: 0,
            },
          })),
        },
        turnRetrievalCoordinator: {
          coordinate: vi.fn(async () => ({
            applicableCommitments: [],
            actionApplicableCommitments: [],
            pendingCorrections: [],
            affectiveTrajectory: [],
            retrieval,
            retrievedEpisodes: [],
            retrievedSemantic: null,
            proceduralContext: null,
            selectedSkill: null,
            retrievalOptions: {},
            reRetrieve: vi.fn(async () => retrieval),
          })),
        },
        createStreamReader: () =>
          ({
            async *iterate() {},
          }) as StreamReader,
      } as unknown as TurnPhaseCoordinatorOptions;

      const result = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-mixed-sender-operator-only",
        turnInput: {
          userMessage: "mixed sender batch",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "mixed sender batch",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: creatorId,
        audienceEntity: null,
        currentSenderBorgRole: null,
        operatorOnlyDirectivesAllowed: false,
        audienceProfile: null,
        sessionAudienceRole: "operator",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(
        repository.listApplicable({
          currentAudienceEntityId: creatorId,
          sessionRole: "operator",
        })[0]?.activation,
      ).toMatchObject({
        active: true,
        reason: "operator_only",
      });
      expect(result.creatorDirectiveBriefing).toBeNull();
    } finally {
      db.close();
    }
  });

  it("recalls self-introspection selectors globally for cognition", async () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const options = minimalRetrievalPhaseOptions(repository);
    const listRecentOtherActiveSessionEvents = vi.fn(() => []);
    const listRecentAutonomousSelfPrivate = vi.fn(() => []);
    const participantEntityId = createEntityId();
    const creatorEntityId = createEntityId();

    options.activityRepository = {
      record: vi.fn(),
      listRecentOtherActiveSessionEvents,
    };
    options.selfDecisionRepository = {
      listRecentAutonomousSelfPrivate,
    };

    try {
      await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-private-self-introspection",
        turnInput: {
          userMessage: "",
          audience: "self",
          origin: "autonomous",
          autonomyTrigger: {
            source_name: "scheduled_reflection",
            source_type: "trigger",
            event_id: "scheduled-reflection:1000",
            sort_ts: 1_000,
            payload: {
              interval_ms: 1_000,
            },
          },
        },
        isSelfAudience: true,
        isUserTurn: false,
        cognitionInput: "Autonomous wake context: scheduled_reflection",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: null,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "participant",
        currentSenderBorgRole: null,
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(listRecentOtherActiveSessionEvents).toHaveBeenCalledTimes(1);
      expect(listRecentAutonomousSelfPrivate).toHaveBeenCalledTimes(1);

      listRecentOtherActiveSessionEvents.mockClear();
      listRecentAutonomousSelfPrivate.mockClear();

      await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-participant-introspection-closed",
        turnInput: {
          userMessage: "Hi",
          audience: "participant",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "Hi",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: participantEntityId,
        audienceEntity: {
          id: participantEntityId,
          canonical_name: "Participant",
          aliases: [],
          kind: "person",
          borg_role: null,
          name_provenance: "user_declared",
          created_at: 1_000,
        },
        audienceProfile: null,
        sessionAudienceRole: "participant",
        currentSenderBorgRole: null,
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [
          {
            entityId: participantEntityId,
            displayName: "Participant",
            role: "audience",
          },
        ],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(listRecentOtherActiveSessionEvents).toHaveBeenCalledTimes(1);
      expect(listRecentAutonomousSelfPrivate).toHaveBeenCalledTimes(1);

      listRecentOtherActiveSessionEvents.mockClear();
      listRecentAutonomousSelfPrivate.mockClear();

      await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-operator-introspection",
        turnInput: {
          userMessage: "Status",
          audience: "operator",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "Status",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: creatorEntityId,
        audienceEntity: {
          id: creatorEntityId,
          canonical_name: "Creator",
          aliases: [],
          kind: "person",
          borg_role: "creator",
          name_provenance: "user_declared",
          created_at: 1_000,
        },
        audienceProfile: null,
        sessionAudienceRole: "operator",
        currentSenderBorgRole: "creator",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [
          {
            entityId: creatorEntityId,
            displayName: "Creator",
            role: "audience",
          },
        ],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(listRecentOtherActiveSessionEvents).toHaveBeenCalledTimes(1);
      expect(listRecentAutonomousSelfPrivate).toHaveBeenCalledTimes(1);
    } finally {
      db.close();
    }
  });

  it("computes current-time lived-experience context when the recent-lived band is gap-gated off", async () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const options = minimalRetrievalPhaseOptions(repository);
    const nowMs = Date.UTC(2026, 5, 18, 12, 0, 0);
    const previousUserAt = nowMs - 10 * 60_000;
    const previousAgentAt = nowMs - 9 * 60_000;
    const windowMs = 3 * 24 * 60 * 60_000;
    const tempDir = mkdtempSync(join(tmpdir(), "borg-current-time-context-"));
    const priorWriter = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(previousUserAt),
    });
    const priorAgentWriter = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(previousAgentAt),
    });
    const currentWriter = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(nowMs),
    });
    await priorWriter.append({
      kind: "user_msg",
      content: "Previous same-session turn.",
    });
    await priorAgentWriter.append({
      kind: "agent_msg",
      content: "Previous same-session reply.",
    });
    const currentUserEntry = await currentWriter.append({
      kind: "user_msg",
      content: "Current same-session turn.",
    });

    options.clock = new FixedClock(nowMs);
    options.config = {
      ...options.config,
      generation: {
        ...options.config.generation,
        evidenceLedger: {
          ...options.config.generation.evidenceLedger,
          enabled: true,
          recentLivedExperience: {
            ...options.config.generation.evidenceLedger.recentLivedExperience,
            recencyWindowMs: windowMs,
            densityCap: 1,
            gapThresholdMs: 3 * 60 * 60_000,
          },
        },
      },
    };
    options.openQuestionsRepository = {
      ...options.openQuestionsRepository,
      findByHandles: () => [],
      get: () => null,
      resolve: vi.fn(),
    };
    options.createStreamReader = (sessionId: SessionId) =>
      new StreamReader({ dataDir: tempDir, sessionId });
    const listDailyOtherActiveSessionDensity = vi.fn(() => [
      {
        dayKey: "2026-06-18",
        dayStartMs: Date.UTC(2026, 5, 18),
        sessionId: createSessionId(),
        sessionLabel: "botarena run",
        audienceLabel: "arena",
        audienceEntityId: null,
        eventCount: 2,
        conversationTurnCount: 1,
        kindCounts: {
          userContact: 1,
          borgReplied: 1,
          turnCompleted: 0,
        },
        firstOccurredAt: nowMs - 6 * 60_000,
        lastOccurredAt: nowMs - 2 * 60_000,
      },
    ]);
    const listDailyAutonomousSelfPrivateDensity = vi.fn(() => [
      {
        dayKey: "2026-06-18",
        dayStartMs: Date.UTC(2026, 5, 18),
        decisionCount: 2,
        distinctDecisionShapeCount: 2,
        firstOccurredAt: nowMs - 8 * 60_000,
        lastOccurredAt: nowMs - 1 * 60_000,
      },
    ]);
    const countOtherActiveSessionConversationTurns = vi.fn(() => 40);
    const countAutonomousSelfPrivateDecisions = vi.fn(() => 138);
    options.activityRepository = {
      record: vi.fn(),
      getMostRecentOtherActiveSessionEventOccurredAt: vi.fn(() => nowMs - 2 * 60_000),
      listRecentOtherActiveSessionEvents: vi.fn(() => []),
      listDailyOtherActiveSessionDensity,
      countOtherActiveSessionConversationTurns,
    };
    options.selfDecisionRepository = {
      listRecentAutonomousSelfPrivate: vi.fn(() => []),
      listDailyAutonomousSelfPrivateDensity,
      countAutonomousSelfPrivateDecisions,
    };

    try {
      const result = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-current-time-lived-experience-gap-gated",
        turnInput: {
          userMessage: "Current same-session turn.",
          audience: "operator",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "Current same-session turn.",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: null,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "operator",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 2,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        persistedUserEntry: currentUserEntry,
        currentUserEntries: [currentUserEntry],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(listDailyOtherActiveSessionDensity).toHaveBeenCalledWith({
        currentSessionId: DEFAULT_SESSION_ID,
        sinceMs: nowMs - windowMs,
        untilMs: nowMs,
        limit: options.config.generation.evidenceLedger.recentLivedExperience.densityCap,
      });
      expect(listDailyAutonomousSelfPrivateDensity).toHaveBeenCalledWith({
        sinceMs: nowMs - windowMs,
        untilMs: nowMs,
        limit: options.config.generation.evidenceLedger.recentLivedExperience.densityCap,
      });
      expect(countOtherActiveSessionConversationTurns).toHaveBeenCalledWith({
        currentSessionId: DEFAULT_SESSION_ID,
        sinceMs: nowMs - windowMs,
        untilMs: nowMs,
      });
      expect(countAutonomousSelfPrivateDecisions).toHaveBeenCalledWith({
        sinceMs: nowMs - windowMs,
        untilMs: nowMs,
      });
      expect(result.currentTimeContext).toEqual({
        previousUserMessageAt: previousUserAt,
        recentLifeElsewhere: {
          windowMs,
          autonomousReflectionCount: 138,
          crossSessionConversationTurnCount: 40,
        },
      });
      expect(
        result.evidenceLedgerContext.ledger?.audienceStanding?.renderRecentLivedExperience,
      ).toBe(false);
      expect(
        result.evidenceLedgerContext.ledger?.sections.find(
          (section) => section.id === "recent_lived_experience",
        ),
      ).toBeUndefined();

      const autonomousResult = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-autonomous-lived-experience-renders",
        turnInput: {
          userMessage: "Autonomous wake.",
          audience: "operator",
          origin: "autonomous",
        },
        isSelfAudience: false,
        isUserTurn: false,
        cognitionInput: "Autonomous wake.",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: null,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "operator",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 3,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });
      const autonomousRecentLivedSection =
        autonomousResult.evidenceLedgerContext.ledger?.sections.find(
          (section) => section.id === "recent_lived_experience",
        );

      expect(
        autonomousResult.evidenceLedgerContext.ledger?.audienceStanding
          ?.renderRecentLivedExperience,
      ).toBe(true);
      expect(autonomousRecentLivedSection?.entries.length).toBeGreaterThan(0);
    } finally {
      priorWriter.close();
      priorAgentWriter.close();
      currentWriter.close();
      rmSync(tempDir, { recursive: true, force: true });
      db.close();
    }
  });

  it("derives previous user-message time through the entry-index fast path", async () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    type IndexedUserRecord = {
      entry_id: StreamEntryId;
      session_id: SessionId;
      timestamp: number;
      entry_index: number | null;
      kind: "user_msg";
      turn_id: string | null;
      turn_status: "active" | "aborted" | null;
      active: boolean;
    };
    const record = (input: {
      id?: StreamEntryId;
      timestamp: number;
      entryIndex: number;
      active?: boolean;
      turnStatus?: "active" | "aborted" | null;
    }): IndexedUserRecord => ({
      entry_id: input.id ?? createStreamEntryId(),
      session_id: DEFAULT_SESSION_ID,
      timestamp: input.timestamp,
      entry_index: input.entryIndex,
      kind: "user_msg",
      turn_id: `turn-${input.entryIndex}`,
      turn_status: input.turnStatus ?? "active",
      active: input.active ?? true,
    });
    const streamEntry = (indexed: IndexedUserRecord, content: string): StreamEntry => ({
      id: indexed.entry_id,
      timestamp: indexed.timestamp,
      entry_index: indexed.entry_index ?? undefined,
      kind: "user_msg",
      content,
      session_id: indexed.session_id,
      turn_id: indexed.turn_id ?? undefined,
      turn_status: indexed.turn_status ?? undefined,
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    });
    const runCase = async (input: {
      records: readonly IndexedUserRecord[];
      currentRecords: readonly IndexedUserRecord[];
    }): Promise<number | null> => {
      const options = minimalRetrievalPhaseOptions(repository);
      const recordsById = new Map(input.records.map((item) => [item.entry_id, item]));
      const currentUserEntries = input.currentRecords.map((item, index) =>
        streamEntry(item, `Current ${index}`),
      );

      options.entryIndex = {
        lookup: (entryId: StreamEntryId) => recordsById.get(entryId) ?? null,
        lookupMany: (entryIds: readonly StreamEntryId[]) =>
          new Map(
            entryIds.flatMap((entryId) => {
              const found = recordsById.get(entryId);

              return found === undefined ? [] : [[entryId, found] as const];
            }),
          ),
        lookupEntriesById: () => new Map(),
        lookupSessionEntriesByKind: (lookupInput: { sessionId: SessionId; kind: string }) =>
          input.records.filter(
            (item) => item.session_id === lookupInput.sessionId && item.kind === lookupInput.kind,
          ),
        countSessionEntriesByKind: (countInput: {
          sessionId: SessionId;
          kind: string;
          excludeEntryId?: StreamEntryId;
        }) =>
          input.records
            .filter(
              (item) => item.session_id === countInput.sessionId && item.kind === countInput.kind,
            )
            .filter((item) => item.active)
            .filter((item) => item.entry_id !== countInput.excludeEntryId).length,
        quarantinedSharedStateArtifactRefs: () => new Set(),
      } as unknown as NonNullable<TurnPhaseCoordinatorOptions["entryIndex"]>;
      options.createStreamReader = () =>
        ({
          async *iterate() {
            throw new Error("entry-index fast path should not load the session stream");
          },
        }) as unknown as StreamReader;

      const result = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-current-time-entry-index",
        turnInput: {
          userMessage: "Current indexed turn.",
          audience: "operator",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "Current indexed turn.",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: null,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "operator",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        persistedUserEntry: currentUserEntries[0],
        currentUserEntries,
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      return result.currentTimeContext.previousUserMessageAt;
    };

    try {
      const firstCurrent = record({ timestamp: 3_000, entryIndex: 1 });
      await expect(
        runCase({
          records: [firstCurrent],
          currentRecords: [firstCurrent],
        }),
      ).resolves.toBeNull();

      const prior = record({ timestamp: 4_000, entryIndex: 1 });
      const currentPartA = record({ timestamp: 5_000, entryIndex: 2 });
      const currentPartB = record({ timestamp: 5_100, entryIndex: 3 });
      await expect(
        runCase({
          records: [prior, currentPartA, currentPartB],
          currentRecords: [currentPartA, currentPartB],
        }),
      ).resolves.toBe(prior.timestamp);

      const activePrior = record({ timestamp: 6_000, entryIndex: 1 });
      const inactiveLaterPrior = record({
        timestamp: 7_000,
        entryIndex: 2,
        active: false,
        turnStatus: "active",
      });
      const currentAfterInactive = record({ timestamp: 8_000, entryIndex: 3 });
      await expect(
        runCase({
          records: [activePrior, inactiveLaterPrior, currentAfterInactive],
          currentRecords: [currentAfterInactive],
        }),
      ).resolves.toBe(activePrior.timestamp);
    } finally {
      db.close();
    }
  });

  it("skips later-aborted prior user messages in the stream-reader fallback", async () => {
    const db = openDatabase(":memory:", {
      migrations: creatorDirectiveMigrations,
    });
    const repository = new CreatorDirectiveRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const options = minimalRetrievalPhaseOptions(repository);
    const tempDir = mkdtempSync(join(tmpdir(), "borg-current-time-aborted-fallback-"));
    const activePriorAt = 1_000;
    const abortedPriorAt = 2_000;
    const currentAt = 3_000;
    const activePriorWriter = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(activePriorAt),
    });
    const abortedPriorWriter = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(abortedPriorAt),
    });
    const abortMarkerWriter = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(abortedPriorAt + 100),
    });
    const currentWriter = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(currentAt),
    });

    await activePriorWriter.append({
      kind: "user_msg",
      content: "Active prior user message.",
    });
    const abortedPrior = await abortedPriorWriter.append({
      kind: "user_msg",
      content: "Aborted prior user message.",
      turn_id: "turn-aborted-prior",
      turn_status: "active",
    });
    await abortMarkerWriter.append({
      kind: "internal_event",
      content: {
        event: "aborted_turn",
        turn_id: "turn-aborted-prior",
        aborted_stream_entry_ids: [abortedPrior.id],
      },
      turn_id: "turn-aborted-prior",
      turn_status: "aborted",
    });
    const currentUserEntry = await currentWriter.append({
      kind: "user_msg",
      content: "Current user message.",
    });

    options.clock = new FixedClock(currentAt);
    options.createStreamReader = (sessionId: SessionId) =>
      new StreamReader({ dataDir: tempDir, sessionId });

    try {
      const result = await runRetrievalPhase({
        options,
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-current-time-aborted-fallback",
        turnInput: {
          userMessage: "Current user message.",
          audience: "operator",
          origin: "user",
        },
        isSelfAudience: false,
        isUserTurn: true,
        cognitionInput: "Current user message.",
        llmClient: new FakeLLMClient({ responses: [] }),
        recencyMessages: [],
        audienceEntityId: null,
        audienceEntity: null,
        audienceProfile: null,
        sessionAudienceRole: "operator",
        perception: {
          entities: [],
          mode: "relational",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        workingMemory: {
          turn_counter: 1,
        } as never,
        suppressionSet: {} as never,
        actionLinkSelfContext: null,
        persistedPromotions: {
          goalIds: [],
          executiveStepIds: [],
        },
        correctiveCommitment: null,
        activeParticipants: [],
        participantRoster: null,
        participantProfiles: [],
        persistedUserEntry: currentUserEntry,
        currentUserEntries: [currentUserEntry],
        currentTurnFrameAnomaly: null,
        closureLoopAssessment: null,
      });

      expect(result.currentTimeContext.previousUserMessageAt).toBe(activePriorAt);
    } finally {
      activePriorWriter.close();
      abortedPriorWriter.close();
      abortMarkerWriter.close();
      currentWriter.close();
      rmSync(tempDir, { recursive: true, force: true });
      db.close();
    }
  });
});

describe("compileSharedStateArtifactForEvidenceLedger", () => {
  const cleanup: Array<() => void> = [];

  afterEach(() => {
    while (cleanup.length > 0) {
      cleanup.pop()?.();
    }
  });

  it("uses the global turn counter for shared-state action canonicalization", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(10_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const actionId = createActionId();
    const commitmentId = createCommitmentId();
    const openQuestionId = createOpenQuestionId();
    const priorSourceEntryId = createStreamEntryId();
    const streamEntryId = createStreamEntryId();
    const currentUserContent = "The clinic callback follow-up is locked.";
    const priorSourceEntry = {
      id: priorSourceEntryId,
      kind: "user_msg",
      content: "The clinic callback follow-up is locked.",
      timestamp: 9_500,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const currentUserEntry = {
      id: streamEntryId,
      kind: "user_msg",
      content: currentUserContent,
      timestamp: 10_000,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const action = {
      id: actionId,
      description: "Follow up with the clinic",
      actor: "user",
      audience_entity_id: audienceEntityId,
      state: "committed_to_do",
      updated_at: 9_000,
      session_scope: null,
      scheduled_at: null,
      last_referenced_turn_counter: 2,
      last_referenced_turn_global: null,
    } as ActionRecord;
    const privateCommitment = {
      id: commitmentId,
      kind: "assistant_commitment",
      type: "promise",
      enforcement_class: "advisory",
      critical_domain: null,
      directive_family: "clinic_callback",
      closure_pressure_relevance: "neutral",
      directive: "Keep the clinic callback detail private.",
      priority: 1,
      made_to_entity: audienceEntityId,
      restricted_audience: audienceEntityId,
      about_entity: null,
      status: "active",
      provenance: { kind: "online", process: "test" },
      source_stream_entry_ids: [priorSourceEntryId],
      created_at: 9_000,
      updated_at: 9_000,
      expires_at: null,
      expired_at: null,
      revoked_at: null,
      revoked_reason: null,
      revoke_provenance: null,
      superseded_by: null,
      canonicalized_by_artifact_entry_id: null,
    } as never;
    const privateOpenQuestion = {
      id: openQuestionId,
      question: "Should the clinic callback remain private?",
      urgency: 0.5,
      status: "open",
      goal_id: null,
      audience_entity_id: audienceEntityId,
      related_episode_ids: [],
      related_semantic_node_ids: [],
      provenance: { kind: "online", process: "test" },
      source: "user",
      created_at: 9_000,
      updated_at: 9_000,
      last_touched: 9_000,
      resolution_evidence_episode_ids: [],
      resolution_evidence_stream_entry_ids: [],
      resolution_note: null,
      resolved_at: null,
      abandoned_reason: null,
      abandoned_at: null,
      unresolved_rumination_ticks: 0,
      last_ruminated_at: null,
    } as never;
    const update = vi.fn();
    const llmClient = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_shared_state",
              name: SHARED_STATE_TOOL_NAME,
              input: {
                operations: [
                  {
                    type: "add",
                    state_key: "decision.route",
                    kind: "locked",
                    text: "The clinic callback follow-up is locked.",
                    owner_entity_id: audienceEntityId,
                    source_stream_entry_ids: [priorSourceEntryId],
                    canonicalizes: {
                      action_ids: [actionId],
                    },
                  },
                ],
              },
            },
          ],
        },
      ],
    });
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: false,
              },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => llmClient,
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [action],
        get: () => action,
        update,
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [privateCommitment],
      },
      openQuestionsRepository: {
        list: () => [privateOpenQuestion],
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield priorSourceEntry;
            yield currentUserEntry;
          },
        }) as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    await compileSharedStateArtifactForEvidenceLedger({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-global-canonicalization",
        audienceEntityId,
        currentUserMessage: currentUserContent,
        currentUserEntry,
        globalTurnCounter: 42,
        workingMemory: {
          turn_counter: 3,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [
          {
            id: "current_user_message",
            label: "1. Current User Message",
            entries: [
              {
                id: `current_session_stream:${priorSourceEntryId}`,
                source_type: "current_session_stream",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 1,
                text: "The clinic callback follow-up is locked.",
              },
              {
                id: `current_user_message:${streamEntryId}`,
                source_type: "current_user_message",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 0,
                text: currentUserContent,
              },
            ],
          },
        ],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "Action candidate: Follow up with the clinic.",
    });
    const requestPayload = JSON.parse(
      String(llmClient.requests[0]?.messages[0]?.content ?? "{}"),
    ) as {
      canonicalization_candidates?: {
        active_commitments?: Array<{
          id: string;
          disclosure?: string;
          disclosure_label?: {
            disclosure_class?: string;
            private_to_entity_ids?: string[];
          };
        }>;
        open_questions?: Array<{
          id: string;
          disclosure?: string;
          disclosure_label?: {
            disclosure_class?: string;
            private_to_entity_ids?: string[];
          };
        }>;
      };
    };
    const commitmentCandidate =
      requestPayload.canonicalization_candidates?.active_commitments?.find(
        (candidate) => candidate.id === commitmentId,
      );
    const openQuestionCandidate = requestPayload.canonicalization_candidates?.open_questions?.find(
      (candidate) => candidate.id === openQuestionId,
    );

    expect(commitmentCandidate).toMatchObject({
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [audienceEntityId],
      },
    });
    expect(commitmentCandidate?.disclosure).toContain("disclosure_class=relationship_private");
    expect(openQuestionCandidate).toMatchObject({
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [audienceEntityId],
      },
    });
    expect(openQuestionCandidate?.disclosure).toContain("disclosure_class=relationship_private");

    expect(update).toHaveBeenCalledWith(
      actionId,
      expect.objectContaining({
        last_referenced_turn_counter: 42,
        last_referenced_turn_global: 42,
      }),
      { skipSideEffects: true },
    );
  });

  it("ages image-derived shared-state updates by the durable attachment turn", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-image-aging-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(10_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const currentEntryId = createStreamEntryId();
    const parentEntryId = createStreamEntryId();
    const imageStreamEntryId = createStreamEntryId();
    const attachmentId = "att_aaaaaaaaaaaaaaaa" as never;
    const currentUserEntry = {
      id: currentEntryId,
      kind: "user_msg",
      content: "What was in the old deployment diagram?",
      timestamp: 10_000,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
      turn_id: "turn-500",
      turn_status: "active",
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const llmClient = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_shared_state",
              name: SHARED_STATE_TOOL_NAME,
              input: {
                operations: [
                  {
                    type: "add",
                    state_key: "project.atlas.diagram",
                    kind: "live",
                    text: "The Atlas diagram shows build flowing into release.",
                    owner_entity_id: audienceEntityId,
                    source_stream_entry_ids: [parentEntryId],
                  },
                ],
              },
            },
          ],
        },
      ],
    });
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: { enabled: false },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => llmClient,
      clock,
      tracer: { enabled: false, emit: vi.fn() },
      entityRepository: { resolve: () => selfEntityId },
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [], get: () => null },
      goalsRepository: { list: () => [] },
      commitmentRepository: { list: () => [] },
      openQuestionsRepository: { list: () => [] },
      attachmentRepository: {
        get: () => ({
          attachment_id: attachmentId,
          active: true,
          byte_size: 100,
          width: 2,
          height: 2,
          created_turn_global: 100,
        }),
        isActiveForStreamEntry: () => true,
      },
      entryIndex: {
        countSessionEntriesByKind: () => 0,
        lookupEntriesById: (ids: readonly string[]) =>
          new Map(
            ids.map((id) => [
              id,
              {
                entry_id: id,
                session_id: DEFAULT_SESSION_ID,
                timestamp: 1,
                kind: "user_msg",
                turn_id: id === currentEntryId ? "turn-500" : "turn-100",
                turn_status: "active",
                active: true,
              },
            ]),
          ),
        quarantinedSharedStateArtifactRefs: () => new Set(),
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield currentUserEntry;
          },
        }) as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    await compileSharedStateArtifactForEvidenceLedger({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-image-aging",
        audienceEntityId,
        currentUserMessage: "What was in the old deployment diagram?",
        currentUserEntry,
        globalTurnCounter: 500,
        workingMemory: { turn_counter: 500 } as never,
        applicableCommitments: [],
        retrievedEvidence: [
          {
            id: "image-old",
            source: "image_perception",
            text: "Caption: build flowing into release.",
            provenance: { attachmentId, streamIds: [parentEntryId, imageStreamEntryId] },
            recallIntentId: "intent-image",
            matchedTerms: [],
            score: 0.9,
            scoreBreakdown: { vector: 0.9 },
            imageAttachmentId: attachmentId,
            imageLabel: "Image: old Atlas diagram",
            citationType: "original_image",
          },
        ],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [
          {
            id: "retrieved_memory_evidence",
            label: "Retrieved Evidence",
            entries: [
              {
                id: "retrieved_evidence:image-old",
                source_type: "prior_session_stream",
                session_scope: "prior_session",
                actor: "memory",
                trust_rank: 1,
                citations: [parentEntryId, imageStreamEntryId],
                text: "Caption: build flowing into release.",
              },
            ],
          },
        ],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "Caption: build flowing into release.",
    });

    expect(sharedStateRepository.get(audienceEntityId)?.entries[0]?.last_updated_turn_global).toBe(
      100,
    );
  });

  it("uses the same structural render salience signals when compile is skipped", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-skip-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(20_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const actionId = createActionId();
    const goalId = createGoalId();
    const openQuestionId = createOpenQuestionId();
    const commitmentId = createCommitmentId();
    const operationalCommitmentId = createCommitmentId();
    const streamEntryId = createStreamEntryId();
    const currentUserEntry = {
      id: streamEntryId,
      kind: "user_msg",
      content: "Thanks, that closes it.",
      timestamp: 20_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-skipped-render-signals",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const action = {
      id: actionId,
      description: "Send the project note",
      actor: "user",
      audience_entity_id: audienceEntityId,
      state: "committed_to_do",
      updated_at: 19_000,
      session_scope: null,
      scheduled_at: null,
      last_referenced_turn_counter: null,
      last_referenced_turn_global: null,
    } as ActionRecord;
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: true,
              },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => new FakeLLMClient({ responses: [] }),
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [action],
        get: () => action,
      },
      goalsRepository: {
        list: () => [
          {
            id: goalId,
            description: "Keep project notes current",
          },
        ],
      },
      commitmentRepository: {
        list: () => [
          {
            id: commitmentId,
            directive: "Do not reveal private project notes.",
            kind: "boundary",
            type: "rule",
            directive_family: "privacy",
            enforcement_class: "critical",
            critical_domain: "privacy",
          },
          {
            id: operationalCommitmentId,
            directive: "Prefer concise project-note summaries.",
            kind: "process_norm",
            type: "rule",
            directive_family: "brevity",
            enforcement_class: "advisory",
            critical_domain: null,
          },
        ],
      },
      openQuestionsRepository: {
        list: () => [
          {
            id: openQuestionId,
            question: "Which project note is current?",
          },
        ],
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield currentUserEntry;
          },
        }) as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await compileSharedStateArtifactForEvidenceLedgerResult({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-skipped-render-signals",
        audienceEntityId,
        currentUserMessage: "Thanks, that closes it.",
        currentUserEntry,
        globalTurnCounter: 12,
        workingMemory: {
          turn_counter: 12,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "idle",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "",
    });

    expect(result.appliedOperationCount).toBe(0);
    expect(result.renderOptions?.activeOpenQuestionIds).toEqual([openQuestionId]);
    expect(result.renderOptions?.activeActionIds).toEqual([actionId]);
    expect(result.renderOptions?.activeGoalIds).toEqual([goalId]);
    expect(result.renderOptions?.activeCriticalCommitmentIds).toEqual([commitmentId]);
    expect(result.renderOptions?.activeOperationalCommitmentIds).toEqual([operationalCommitmentId]);
    expect(result.renderOptions?.activeOperationalCommitmentIds).not.toContain(commitmentId);
  });

  it("uses indexed source-trust facts instead of loading the full session stream", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-indexed-trust-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(21_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const inactiveSourceEntryId = createStreamEntryId();
    const currentSourceEntryId = createStreamEntryId();
    const missingIndexedSourceEntryId = createStreamEntryId();
    const currentUserEntry = {
      id: currentSourceEntryId,
      kind: "user_msg",
      content: "Thanks, that closes it.",
      timestamp: 21_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-indexed-source-trust",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const lookupEntriesById = vi.fn((entryIds: readonly string[]) => {
      const facts = new Map();

      if (entryIds.includes(inactiveSourceEntryId)) {
        facts.set(inactiveSourceEntryId, {
          entry_id: inactiveSourceEntryId,
          session_id: DEFAULT_SESSION_ID,
          timestamp: 19_000,
          kind: "user_msg",
          turn_id: "turn-aborted",
          turn_status: "active",
          active: false,
        });
      }

      return facts;
    });
    const iterate = vi.fn(async function* () {
      throw new Error("session stream should not be loaded for indexed source trust");
    });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: true,
              },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => new FakeLLMClient({ responses: [] }),
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
        get: () => null,
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
      },
      entryIndex: {
        countSessionEntriesByKind: () => 0,
        lookupEntriesById,
        quarantinedSharedStateArtifactRefs: () => new Set(),
      },
      createStreamReader: () =>
        ({
          iterate,
        }) as unknown as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await compileSharedStateArtifactForEvidenceLedgerResult({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-indexed-source-trust",
        audienceEntityId,
        currentUserMessage: "Thanks, that closes it.",
        currentUserEntry,
        globalTurnCounter: 13,
        workingMemory: {
          turn_counter: 13,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "idle",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [
          {
            id: "current_user_message",
            label: "1. Current User Message",
            entries: [
              {
                id: `current_session_stream:${inactiveSourceEntryId}`,
                source_type: "current_session_stream",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 0,
                text: "Inactive evidence.",
              },
              {
                id: `current_user_message:${currentSourceEntryId}`,
                source_type: "current_user_message",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 0,
                text: "Current evidence.",
              },
              {
                id: "retrieved_evidence:missing-index-source",
                source_type: "prior_session_stream",
                session_scope: "prior_session",
                actor: "user",
                trust_rank: 1,
                citations: [missingIndexedSourceEntryId],
                text: "Evidence missing from the index.",
              },
            ],
          },
        ],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "",
    });

    expect(iterate).not.toHaveBeenCalled();
    expect(lookupEntriesById).toHaveBeenCalled();
    expect(warn).toHaveBeenCalledWith(
      `Stream entry ${missingIndexedSourceEntryId} was not found in the stream entry index during shared-state source trust validation`,
    );
    expect(result.renderOptions?.ledgerStreamEntryIds).toEqual([
      currentSourceEntryId,
      missingIndexedSourceEntryId,
    ]);
    warn.mockRestore();
  });

  it("does not infer legacy shared-state turn age from sparse indexed source-trust facts", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-indexed-legacy-age-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(22_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const priorSessionId = createSessionId();
    const priorSourceEntryId = createStreamEntryId();
    const currentSourceEntryId = createStreamEntryId();
    const currentUserEntry = {
      id: currentSourceEntryId,
      kind: "user_msg",
      content: "Current placeholder source.",
      timestamp: 22_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-current-indexed-legacy-age",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const initialArtifact = sharedStateRepository.upsert(
      audienceEntityId,
      [
        {
          type: "add",
          state_key: "state.legacy",
          kind: "live",
          text: "Legacy shared state with no durable turn age.",
          provenance_stream_entry_ids: [priorSourceEntryId],
          last_updated_stream_entry_ids: [priorSourceEntryId],
          created_at: 1_000,
          last_updated_at: 1_000,
        },
        {
          type: "add",
          state_key: "state.current",
          kind: "live",
          text: "Current shared state entry that consumes the render slot.",
          provenance_stream_entry_ids: [currentSourceEntryId],
          last_updated_stream_entry_ids: [currentSourceEntryId],
          created_at: 2_000,
          last_updated_at: 2_000,
        },
      ],
      {
        now: 1_000,
        lastUpdatedTurnGlobal: null,
      },
    );
    const legacyEntryId = initialArtifact?.entries[0]?.id;
    const lookupEntriesById = vi.fn((entryIds: readonly string[]) => {
      const facts = new Map();

      if (entryIds.includes(priorSourceEntryId)) {
        facts.set(priorSourceEntryId, {
          entry_id: priorSourceEntryId,
          session_id: priorSessionId,
          timestamp: 1_000,
          kind: "user_msg",
          turn_id: "turn-prior-session-indexed-legacy-age",
          turn_status: "active",
          active: true,
        });
      }

      return facts;
    });
    const iterate = vi.fn(async function* () {
      throw new Error("session stream should not be loaded for indexed source trust");
    });
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: false,
              },
              recentTurnThreshold: 5,
              dormantTurnThreshold: 15,
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () =>
        new FakeLLMClient({
          responses: [
            {
              text: "",
              input_tokens: 12,
              output_tokens: 8,
              stop_reason: "tool_use",
              tool_calls: [
                {
                  id: "toolu_shared_state",
                  name: SHARED_STATE_TOOL_NAME,
                  input: {
                    operations: [],
                  },
                },
              ],
            },
          ],
        }),
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
        get: () => null,
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
      },
      entryIndex: {
        countSessionEntriesByKind: () => 0,
        lookupEntriesById,
        quarantinedSharedStateArtifactRefs: () => new Set(),
      },
      createStreamReader: () =>
        ({
          iterate,
        }) as unknown as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await compileSharedStateArtifactForEvidenceLedgerResult({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-current-indexed-legacy-age",
        audienceEntityId,
        currentUserMessage: "Current placeholder source.",
        currentUserEntry,
        globalTurnCounter: 30,
        workingMemory: {
          turn_counter: 30,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [
          {
            id: "prior_session_memory",
            label: "Retrieved Evidence",
            entries: [
              {
                id: `retrieved_evidence:${priorSourceEntryId}`,
                source_type: "prior_session_stream",
                session_scope: "prior_session",
                actor: "user",
                trust_rank: 1,
                citations: [priorSourceEntryId],
                text: "Prior-session source for legacy shared state.",
              },
              {
                id: `current_user_message:${currentSourceEntryId}`,
                source_type: "current_user_message",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 0,
                text: "Current placeholder source.",
              },
            ],
          },
        ],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "Prior-session source for legacy shared state.",
    });

    expect(iterate).not.toHaveBeenCalled();
    expect(lookupEntriesById).toHaveBeenCalled();
    expect(result.renderOptions?.lastUpdatedTurnByStreamEntryId).toEqual({
      [currentSourceEntryId]: 30,
    });

    const summary = summarizeSharedStateArtifactRender(
      sharedStateRepository.get(audienceEntityId),
      {
        ...result.renderOptions,
        maxEntries: 1,
        reservedSlots: {
          live: 0,
        },
        newestStateChangeReservedSlots: 0,
      },
    );

    expect(summary.renderedEntryIds).not.toContain(legacyEntryId);
    expect(summary.omittedLiveUnknownAge).toBe(1);
    expect(summary.omittedLiveRecentLowSalience).toBe(0);
  });

  it("falls back to stream scanning for cross-session quarantined shared-state refs without an entry index", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-quarantine-fallback-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(25_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const quarantinedSourceEntryId = createStreamEntryId();
    const currentSourceEntryId = createStreamEntryId();
    const currentUserEntry = {
      id: currentSourceEntryId,
      kind: "user_msg",
      content: "Current placeholder source.",
      timestamp: 25_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-quarantine-fallback-current",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const quarantineWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: createSessionId(),
      clock,
    });
    cleanup.push(() => quarantineWriter.close());
    await quarantineWriter.append({
      kind: "internal_event",
      content: {
        event: QUARANTINED_USER_ENTRY_EVENT,
        source_stream_entry_id: quarantinedSourceEntryId,
        cited_stream_entry_ids: [],
      },
    });
    const llmClient = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_shared_state",
              name: SHARED_STATE_TOOL_NAME,
              input: {
                operations: [
                  {
                    type: "add",
                    state_key: "decision.quarantined",
                    kind: "locked",
                    text: "A quarantined cross-session source should not be accepted.",
                    owner_entity_id: audienceEntityId,
                    source_stream_entry_ids: [quarantinedSourceEntryId],
                  },
                ],
              },
            },
          ],
        },
      ],
    });
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: true,
              },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => llmClient,
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
        get: () => null,
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield currentUserEntry;
          },
        }) as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await compileSharedStateArtifactForEvidenceLedgerResult({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-quarantine-fallback",
        audienceEntityId,
        currentUserMessage: "Current placeholder source.",
        currentUserEntry,
        globalTurnCounter: 25,
        workingMemory: {
          turn_counter: 25,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "idle",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [
          {
            id: "prior_session_memory",
            label: "Retrieved Evidence",
            entries: [
              {
                id: `retrieved_evidence:${quarantinedSourceEntryId}`,
                source_type: "prior_session_stream",
                session_scope: "prior_session",
                actor: "user",
                trust_rank: 1,
                text: "Quarantined cross-session evidence.",
              },
            ],
          },
        ],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "Quarantined cross-session evidence.",
    });

    expect(result.appliedOperationCount).toBe(0);
    expect(sharedStateRepository.get(audienceEntityId)?.entries ?? []).toHaveLength(0);
  });

  it("keeps shared-state entries cited by current retrieval results searchable while allowing low-salience demotion", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-retrieved-state-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(30_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const oldSourceEntry = {
      id: createStreamEntryId(),
      kind: "user_msg",
      content: "Placeholder source for retrieved shared state.",
      timestamp: 1_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-1",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const fillerEntries = Array.from({ length: 8 }, (_, index) => ({
      id: createStreamEntryId(),
      kind: "user_msg",
      content: `Placeholder filler source ${index + 2}.`,
      timestamp: 2_000 + index,
      session_id: DEFAULT_SESSION_ID,
      turn_id: `turn-${index + 2}`,
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    })) as StreamEntry[];
    const currentUserEntry = {
      id: createStreamEntryId(),
      kind: "user_msg",
      content: "Current placeholder source.",
      timestamp: 30_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-10",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const initial = sharedStateRepository.upsert(
      audienceEntityId,
      [
        {
          type: "add",
          state_key: "state.placeholder",
          kind: "live",
          text: "Placeholder retrieved shared state",
          provenance_stream_entry_ids: [oldSourceEntry.id],
          last_updated_stream_entry_ids: [oldSourceEntry.id],
          created_at: 1_000,
          last_updated_at: 1_000,
        },
      ],
      {
        now: 1_000,
      },
    );
    const entryId = initial?.entries[0]?.id;
    const llmClient = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_shared_state",
              name: SHARED_STATE_TOOL_NAME,
              input: {
                operations: [],
              },
            },
          ],
        },
      ],
    });
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: false,
              },
              recentTurnThreshold: 5,
              dormantTurnThreshold: 15,
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => llmClient,
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
        get: () => null,
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            for (const entry of [oldSourceEntry, ...fillerEntries, currentUserEntry]) {
              yield entry;
            }
          },
        }) as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await compileSharedStateArtifactForEvidenceLedgerResult({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-retrieved-shared-state",
        audienceEntityId,
        currentUserMessage: "Current placeholder source.",
        currentUserEntry,
        globalTurnCounter: 10,
        workingMemory: {
          turn_counter: 10,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [
          {
            id: "retrieved-placeholder-source",
            source: "raw_stream",
            text: "Placeholder retrieved evidence.",
            provenance: {
              streamIds: [oldSourceEntry.id],
            },
            recallIntentId: "intent-placeholder",
            matchedTerms: [],
            score: 1,
            scoreBreakdown: {},
          },
        ] as never,
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "",
    });

    expect(result.renderOptions?.recentlyRetrievedEntryIds).toEqual([entryId]);
    expect(sharedStateRepository.get(audienceEntityId)?.entries[0]?.kind).toBe("low_salience_live");
  });

  it("renders previous shared state to deliberation instead of a freshly compiled artifact", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-same-turn-shared-state-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(40_000);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock,
    });
    cleanup.push(() => writer.close());
    const priorSourceEntry = await writer.append({
      kind: "user_msg",
      turn_id: "turn-style-preference",
      content: "The operator prefers plain prose.",
    });
    const currentUserEntry = await writer.append({
      kind: "user_msg",
      turn_id: "turn-name-choice",
      content: "Choose a name for a cross-session persistence test.",
    });
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    sharedStateRepository.upsert(
      audienceEntityId,
      [
        {
          type: "add",
          state_key: "identity.style_preference",
          kind: "locked",
          text: "Use plain prose for operator-facing responses.",
          provenance_stream_entry_ids: [priorSourceEntry.id],
          last_updated_stream_entry_ids: [priorSourceEntry.id],
          created_at: 30_000,
          last_updated_at: 30_000,
        },
      ],
      {
        now: 30_000,
        lastCompiledStreamEntryId: priorSourceEntry.id,
      },
    );
    const compilerLlmClient = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_shared_state",
              name: SHARED_STATE_TOOL_NAME,
              input: {
                operations: [
                  {
                    type: "add",
                    state_key: "identity.self_chosen_name",
                    new_key_reason: "The self-chosen name is a distinct identity fact.",
                    kind: "locked",
                    text: "Borg's self-chosen name is Aria.",
                    owner_entity_id: selfEntityId,
                    source_stream_entry_ids: [priorSourceEntry.id],
                  },
                ],
              },
            },
          ],
        },
      ],
    });
    const retrieval = {
      evidence: [],
      episodes: [],
      semantic: null,
      open_questions: [],
      recall_intents: [],
      contradiction_present: false,
      contradictionRouting: {
        contradictions: [],
      },
      confidence: null,
    } as never;
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            enabled: true,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: false,
              },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => compilerLlmClient,
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
        findByName: () => null,
        get: () => null,
      },
      socialRepository: {
        getProfile: () => null,
      },
      relationalSlotRepository: {
        list: () => [],
        listConstrained: () => [],
      },
      actionRepository: {
        list: () => [],
        get: () => null,
        update: vi.fn(),
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
        get: () => null,
        resolve: () => null,
        findByHandles: () => [],
      },
      attachmentRepository: {
        get: () => null,
        isActiveForStreamEntry: () => true,
      },
      selfContextBuilder: {
        build: vi.fn(async () => ({
          selfSnapshot: {
            values: [],
            goals: [],
            traits: [],
          },
          activeScoringValues: [],
          selfScoringFeatures: {
            goalVectors: [],
            valueVectors: [],
          },
          retrievalScoringFeatures: {
            goalVectors: [],
            valueVectors: [],
          },
          executiveFocus: {
            selected_goal: null,
            selected_score: null,
            candidates: [],
            threshold: 0,
          },
        })),
      },
      turnRetrievalCoordinator: {
        coordinate: vi.fn(async () => ({
          applicableCommitments: [],
          actionApplicableCommitments: [],
          pendingCorrections: [],
          affectiveTrajectory: [],
          retrieval,
          retrievedEpisodes: [],
          retrievedSemantic: null,
          proceduralContext: null,
          selectedSkill: null,
          retrievalOptions: {},
          reRetrieve: vi.fn(async () => retrieval),
        })),
      },
      createStreamReader: (sessionId: SessionId) =>
        new StreamReader({ dataDir: tempDir, sessionId }),
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await runRetrievalPhase({
      options,
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-name-choice",
      turnInput: {
        userMessage: "Choose a name for a cross-session persistence test.",
        audience: "operator",
        origin: "user",
        globalTurnCounter: 2,
      },
      isSelfAudience: false,
      isUserTurn: true,
      cognitionInput: "Choose a name for a cross-session persistence test.",
      llmClient: new FakeLLMClient({ responses: [] }),
      recencyMessages: [],
      audienceEntityId,
      audienceEntity: null,
      audienceProfile: null,
      perception: {
        entities: [],
        mode: "problem_solving",
        affectiveSignal: {
          valence: 0,
          arousal: 0,
          dominant_emotion: null,
        },
        temporalCue: null,
      } satisfies PerceptionResult,
      workingMemory: {
        turn_counter: 2,
      } as never,
      suppressionSet: {} as never,
      actionLinkSelfContext: null,
      persistedPromotions: {
        goalIds: [],
        executiveStepIds: [],
      },
      correctiveCommitment: null,
      activeParticipants: [],
      participantRoster: null,
      participantProfiles: [],
      persistedUserEntry: currentUserEntry,
      currentTurnFrameAnomaly: null,
      closureLoopAssessment: null,
    });
    const persistedTexts =
      sharedStateRepository.get(audienceEntityId)?.entries.map((entry) => entry.text) ?? [];
    const rendered = result.evidenceLedgerContext.promptSection ?? "";
    const renderedSharedStateTexts =
      result.evidenceLedgerContext.ledger?.sharedState?.entries.map((entry) => entry.text) ?? [];

    expect(persistedTexts).toEqual([
      "Use plain prose for operator-facing responses.",
      "Borg's self-chosen name is Aria.",
    ]);
    expect(renderedSharedStateTexts).toEqual(["Use plain prose for operator-facing responses."]);
    expect(rendered).toContain("Use plain prose for operator-facing responses.");
    expect(rendered).not.toContain("Borg's self-chosen name is Aria.");
  });

  it("rejects shared-state operations that cite the current user turn as source material", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-current-off-limits-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(50_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const currentSourceEntryId = createStreamEntryId();
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];
    const currentUserEntry = {
      id: currentSourceEntryId,
      kind: "user_msg",
      content: "Choose a name for a cross-session persistence test.",
      timestamp: 50_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-current-off-limits",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const llmClient = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_shared_state",
              name: SHARED_STATE_TOOL_NAME,
              input: {
                operations: [
                  {
                    type: "add",
                    state_key: "identity.self_chosen_name",
                    kind: "locked",
                    text: "Borg's self-chosen name is Aria.",
                    owner_entity_id: selfEntityId,
                    source_stream_entry_ids: [currentSourceEntryId],
                  },
                ],
              },
            },
          ],
        },
      ],
    });
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: false,
              },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => llmClient,
      clock,
      tracer: {
        enabled: true,
        includePayloads: true,
        emit: vi.fn((event: string, data: Record<string, unknown>) => {
          events.push({ event, data });
        }),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
        get: () => null,
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
      },
      attachmentRepository: {
        get: () => null,
        isActiveForStreamEntry: () => true,
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield currentUserEntry;
          },
        }) as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await compileSharedStateArtifactForEvidenceLedgerResult({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-current-off-limits",
        audienceEntityId,
        currentUserMessage: "Choose a name for a cross-session persistence test.",
        currentUserEntry,
        globalTurnCounter: 50,
        workingMemory: {
          turn_counter: 50,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [
          {
            id: "current_user_message",
            label: "1. Current User Message",
            entries: [
              {
                id: `current_user_message:${currentSourceEntryId}`,
                source_type: "current_user_message",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 0,
                text: "Choose a name for a cross-session persistence test.",
              },
            ],
          },
        ],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "Choose a name for a cross-session persistence test.",
    });
    const requestPayload = JSON.parse(String(llmClient.requests[0]?.messages[0]?.content)) as {
      source_trust?: {
        citation_eligible_source_stream_entry_id_count?: number | null;
        off_limits_source_stream_entry_ids?: string[];
      };
    };
    const completed = events.find((event) => event.event === "shared_state.compile.completed");

    expect(result.appliedOperationCount).toBe(0);
    expect(sharedStateRepository.get(audienceEntityId)?.entries ?? []).toHaveLength(0);
    expect(requestPayload.source_trust).toEqual({
      citation_eligible_source_stream_entry_id_count: 0,
      citation_eligible_source_stream_entry_ids: [],
      off_limits_source_stream_entry_ids: [currentSourceEntryId],
    });
    expect(completed?.data).toEqual(
      expect.objectContaining({
        rejectionReasons: ["disallowed_source_stream_entry_id"],
      }),
    );
  });
});

describe("runRetrievalPhase session re-entry continuity", () => {
  it("renders when an autonomous turn precedes the first user-origin turn", async () => {
    const audienceEntityId = createEntityId();
    const currentUserEntryId = createStreamEntryId();
    const priorAutonomousEntryId = createStreamEntryId();
    const artifact = makeSharedStateArtifact([
      makeLockedSharedStateEntry({
        audience_entity_id: audienceEntityId,
        state_key: "project.decision",
      }),
    ]);
    const currentUserEntry = {
      id: currentUserEntryId,
      kind: "user_msg",
      content: "Start a decision log for the project.",
      timestamp: 11_000,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const priorAutonomousEntry = {
      id: priorAutonomousEntryId,
      kind: "perception",
      content: {
        mode: "problem_solving",
        entities: [],
      },
      timestamp: 10_000,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
    } as StreamEntry;
    const retrieval = {
      evidence: [],
      episodes: [],
      semantic: null,
      open_questions: [],
      recall_intents: [],
      contradiction_present: false,
      contradictionRouting: {
        contradictions: [],
      },
      confidence: null,
    } as never;
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            enabled: false,
          },
        },
      },
      sharedStateRepository: {
        get: () => artifact,
      },
      selfContextBuilder: {
        build: vi.fn(async () => ({
          selfSnapshot: {
            values: [],
            goals: [],
            traits: [],
          },
          activeScoringValues: [],
          selfScoringFeatures: {
            goalVectors: [],
            valueVectors: [],
          },
          retrievalScoringFeatures: {
            goalVectors: [],
            valueVectors: [],
          },
          executiveFocus: {
            selected_goal: null,
            selected_score: null,
            candidates: [],
            threshold: 0,
          },
        })),
      },
      turnRetrievalCoordinator: {
        coordinate: vi.fn(async () => ({
          applicableCommitments: [],
          actionApplicableCommitments: [],
          pendingCorrections: [],
          affectiveTrajectory: [],
          retrieval,
          retrievedEpisodes: [],
          retrievedSemantic: null,
          proceduralContext: null,
          selectedSkill: null,
          retrievalOptions: {},
          reRetrieve: vi.fn(async () => retrieval),
        })),
      },
      relationalSlotRepository: {
        list: () => [],
        listConstrained: () => [],
      },
      openQuestionsRepository: {
        get: () => null,
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield priorAutonomousEntry;
            yield currentUserEntry;
          },
        }) as StreamReader,
      clock: new FixedClock(11_000),
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        findByName: () => null,
      },
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await runRetrievalPhase({
      options,
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-first-user-after-autonomous",
      turnInput: {
        userMessage: "Start a decision log for the project.",
        audience: "project-team",
        origin: "user",
      },
      isSelfAudience: false,
      isUserTurn: true,
      cognitionInput: "Start a decision log for the project.",
      llmClient: new FakeLLMClient({ responses: [] }),
      recencyMessages: [],
      audienceEntityId,
      audienceEntity: null,
      audienceProfile: null,
      perception: {
        entities: [],
        mode: "problem_solving",
        affectiveSignal: {
          valence: 0,
          arousal: 0,
          dominant_emotion: null,
        },
        temporalCue: null,
      } satisfies PerceptionResult,
      workingMemory: {
        turn_counter: 2,
      } as never,
      suppressionSet: {} as never,
      actionLinkSelfContext: null,
      persistedPromotions: {
        goalIds: [],
        executiveStepIds: [],
      },
      correctiveCommitment: null,
      activeParticipants: [],
      participantRoster: null,
      participantProfiles: [],
      persistedUserEntry: currentUserEntry,
      currentTurnFrameAnomaly: null,
      closureLoopAssessment: null,
    });

    expect(result.evidenceLedgerContext.sessionReentryContinuityPromptSection).toContain(
      `<${SESSION_REENTRY_CONTINUITY_TAG}>`,
    );
    expect(result.evidenceLedgerContext.sessionReentryContinuityPromptSection).toContain(
      "active_entry_count=1",
    );
  });
});
