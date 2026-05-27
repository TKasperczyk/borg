import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_CONFIG } from "../../../config/index.js";
import { FakeLLMClient } from "../../../llm/test-support/fake-client.js";
import { sharedStateMigrations } from "../../../memory/decision-artifacts/index.js";
import { SharedStateRepository } from "../../../memory/decision-artifacts/repository.js";
import {
  CreatorDirectiveRepository,
  creatorDirectiveMigrations,
  type DisclosurePolicy,
} from "../../../memory/creator-directives/index.js";
import { openDatabase } from "../../../storage/sqlite/index.js";
import { FixedClock } from "../../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createActionId,
  createCommitmentId,
  createEntityId,
  createGoalId,
  createOpenQuestionId,
  createSessionId,
  createStreamEntryId,
  type EntityId,
  type SessionId,
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
import { SHARED_STATE_TOOL_NAME } from "../../shared-state/schema.js";
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
        canonicalFact: "Borg's prior name is Kestrel.",
        operationalDirective: "Answer with the prior name when asked.",
        disclosurePolicy: disclosurePolicy(),
        priority: 5,
        createdAt: 1_000,
      });

      const applicable = repository.listApplicable({
        currentAudienceEntityId: audienceId,
        participantEntityIds: [audienceId],
        topicTags: [],
        sessionRole: "participant",
      });
      const briefing = buildCreatorDirectiveBriefingForTurn({
        applicable,
        currentUserEntryId,
        entityRepository: { get: () => null },
      });

      expect(
        briefing?.directives.flatMap((directive) =>
          directive.renderMode === "content" && directive.canonicalFact !== null
            ? [directive.canonicalFact]
            : [],
        ),
      ).toEqual(["Borg's prior name is Kestrel."]);
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
          topicTags: [],
          sessionRole: "operator",
        }),
        entityRepository: { get: () => null },
      });
      const participantBriefing = buildCreatorDirectiveBriefingForTurn({
        applicable: repository.listApplicable({
          currentAudienceEntityId: audienceId,
          participantEntityIds: [audienceId],
          topicTags: [],
          sessionRole: "participant",
        }),
        entityRepository: { get: () => null },
      });

      expect(
        operatorBriefing?.directives.flatMap((directive) =>
          directive.renderMode === "content" && directive.canonicalFact !== null
            ? [directive.canonicalFact]
            : [],
        ),
      ).toEqual([
        "Borg's public name is Kestrel.",
        "Borg's operator-only diagnostic label is Kestrel-debug.",
      ]);
      expect(
        participantBriefing?.directives.flatMap((directive) =>
          directive.renderMode === "content" && directive.canonicalFact !== null
            ? [directive.canonicalFact]
            : [],
        ),
      ).toEqual(["Borg's public name is Kestrel."]);
    } finally {
      db.close();
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
        canonicalFact: "Borg's self-chosen name is Kestrel.",
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
        canonicalFact: null,
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
          topicTags: [],
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

      expect(briefing?.directives).toEqual([
        expect.objectContaining({
          kind: "self_identity",
          canonicalFact: "Borg's self-chosen name is Kestrel.",
          operationalDirective: null,
        }),
        expect.objectContaining({
          kind: "response_policy",
          canonicalFact: null,
          operationalDirective:
            "Do not volunteer family-planning details unless Alice asks directly.",
        }),
      ]);
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

      expect(renderedEvents).toHaveLength(2);
      expect(renderedEvents).toEqual([
        expect.objectContaining({
          data: expect.objectContaining({
            turnId: "turn-creator-directive-rendered",
            session_id: DEFAULT_SESSION_ID,
            current_audience_entity_id: audienceId,
            participant_entity_ids: [audienceId],
            render_mode: "content",
            reason: "public",
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
          }),
        }),
      ]);
    } finally {
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
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
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
