import { describe, expect, it } from "vitest";

import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../../cognition/index.js";
import { computeWeights } from "../../cognition/attention/index.js";
import { FakeLLMClient } from "../../llm/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import {
  createActionId,
  createEpisodeId,
  createSemanticNodeId,
  createStreamEntryId,
} from "../../util/ids.js";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  TestEmbeddingClient,
} from "../test-support.js";
import { RuminatorProcess } from "./index.js";

const RUMINATOR_TOOL_NAME = "EmitRuminatorDecisions";
const DAY_MS = 24 * 60 * 60 * 1_000;

class CaptureTracer implements TurnTracer {
  readonly enabled = true;
  readonly includePayloads = false;
  readonly events: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    this.events.push({ event, data });
  }
}

function createRuminatorResponse(input: {
  resolution_note: string;
  growth_marker: null | {
    category: string;
    what_changed: string;
    before_description: string | null;
    after_description: string | null;
    confidence: number;
  };
}) {
  return {
    text: "",
    input_tokens: 50,
    output_tokens: 40,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: RUMINATOR_TOOL_NAME,
        input,
      },
    ],
  };
}

describe("RuminatorProcess", () => {
  it("plans and applies a resolution with capped growth confidence, and apply is idempotent", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createRuminatorResponse({
          resolution_note: "Atlas now succeeds after the rollback rehearsal.",
          growth_marker: {
            category: "understanding",
            what_changed: "I understand Atlas rollback sequencing better.",
            before_description: "The deployment order was unclear.",
            after_description: "The rollback rehearsal clarified the order.",
            confidence: 0.95,
          },
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const episode = createEpisodeFixture(
        {
          title: "Atlas rollback rehearsal",
          narrative: "Atlas stabilized after a rollback rehearsal.",
          tags: ["atlas", "deploy"],
          created_at: 2_000_000,
          updated_at: 2_000_000,
        },
        [1, 0, 0, 0],
      );
      await harness.episodicRepository.insert(episode);
      const question = harness.openQuestionsRepository.add({
        question: "Why does Atlas deploy fail?",
        urgency: 0.7,
        related_episode_ids: [createEpisodeId()],
        related_semantic_node_ids: [createSemanticNodeId()],
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
      });

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items).toHaveLength(1);
      expect(llm.requests[0]?.tool_choice).toEqual({
        type: "tool",
        name: RUMINATOR_TOOL_NAME,
      });
      expect(plan.items[0]).toMatchObject({
        action: "resolve",
        question_id: question.id,
        resolution_evidence_episode_ids: [episode.id],
        resolution_evidence_stream_entry_ids: [],
      });
      expect(
        plan.items[0]?.action === "resolve" ? plan.items[0].growth_marker?.confidence : 0,
      ).toBe(0.6);

      await process.apply(harness.createContext(), plan);
      await process.apply(harness.createContext(), plan);

      expect(harness.openQuestionsRepository.get(question.id)?.status).toBe("resolved");
      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        resolution_evidence_episode_ids: [episode.id],
        resolution_evidence_stream_entry_ids: [],
      });
      expect(harness.growthMarkersRepository.list()).toHaveLength(1);
      expect(harness.identityEventRepository.list({ recordType: "open_question" })).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "resolve",
            record_id: question.id,
            provenance: {
              kind: "offline",
              process: "ruminator",
            },
          }),
        ]),
      );
      expect(harness.identityEventRepository.list({ recordType: "growth_marker" })).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "create",
            record_type: "growth_marker",
            provenance: {
              kind: "offline",
              process: "ruminator",
            },
          }),
        ]),
      );

      const audits = harness.auditLog.list({ process: "ruminator" });
      const growthAudit = audits.find((item) => item.action === "add_growth_marker");
      const resolveAudit = audits.find((item) => item.action === "resolve");

      expect(growthAudit).toBeDefined();
      expect(resolveAudit).toBeDefined();

      if (growthAudit !== undefined) {
        await harness.auditLog.revert(growthAudit.id, "test");
      }

      if (resolveAudit !== undefined) {
        await harness.auditLog.revert(resolveAudit.id, "test");
      }

      expect(harness.growthMarkersRepository.list()).toHaveLength(0);
      expect(harness.openQuestionsRepository.get(question.id)?.status).toBe("open");
    } finally {
      await harness.cleanup();
    }
  });

  it("does not resolve when high relevance score has low retrieval confidence", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map([["Why does Atlas deploy fail?", [1, 0, 0, 0]]]),
      ),
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const episode = createEpisodeFixture(
        {
          title: "Atlas deploy keyword match",
          narrative: "Atlas deploy fix is mentioned without settled evidence.",
          tags: ["atlas", "deploy"],
          significance: 0.05,
          created_at: 2_000_000,
          updated_at: 2_000_000,
        },
        [1, 0, 0, 0],
      );
      await harness.episodicRepository.insert(episode);
      const question = harness.openQuestionsRepository.add({
        question: "Why does Atlas deploy fail?",
        urgency: 0.7,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });
      const retrieval = await harness.retrievalPipeline.searchWithContext(question.question, {
        limit: 3,
        globalIdentitySelfAudienceEntityId: harness.entityRepository.findByName("self"),
        attentionWeights: computeWeights("reflective", {
          currentGoals: [],
          hasActiveValues: false,
          hasTemporalCue: false,
        }),
        goalDescriptions: [],
        includeOpenQuestions: false,
      });

      expect(retrieval.episodes[0]?.score).toBeGreaterThan(
        harness.config.offline.ruminator.resolveConfidenceThreshold,
      );
      expect(retrieval.confidence.overall).toBeLessThan(
        harness.config.offline.ruminator.resolveConfidenceThreshold,
      );

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items).toEqual([
        expect.objectContaining({
          action: "mark_unresolved",
          question_id: question.id,
          next_unresolved_rumination_ticks: 1,
        }),
      ]);
      expect(llm.requests).toHaveLength(0);
    } finally {
      await harness.cleanup();
    }
  });

  it("gates audience-scoped resolution on merged fresh evidence confidence", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createRuminatorResponse({
          resolution_note: "Sam-scoped evidence resolved the planning question.",
          growth_marker: null,
        }),
      ],
    });
    const questionText = "What resolved Sam's planning uncertainty?";
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(new Map([[questionText, [0, 1, 0, 0]]])),
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const sam = harness.entityRepository.resolve("Sam");
      const oldGlobalEpisode = createEpisodeFixture(
        {
          title: "Public planning evidence",
          narrative: "Public planning evidence is strong but predates the open question.",
          tags: ["planning"],
          significance: 0.95,
          participants: ["public-planning-group"],
          created_at: 900_000,
          updated_at: 900_000,
        },
        [0, 1, 0, 0],
      );
      const weakAudienceEpisode = createEpisodeFixture(
        {
          title: "Sam weak private planning update",
          narrative: "Sam mentioned a possible planning resolution without settled evidence.",
          tags: ["planning"],
          audience_entity_id: sam,
          shared: false,
          significance: 0.05,
          participants: ["sam"],
          created_at: 2_000_000,
          updated_at: 2_000_000,
        },
        [0, 1, 0, 0],
      );
      await harness.episodicRepository.insert(oldGlobalEpisode);
      await harness.episodicRepository.insert(weakAudienceEpisode);
      const question = harness.openQuestionsRepository.add({
        question: questionText,
        urgency: 0.7,
        source: "reflection",
        audience_entity_id: sam,
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });

      const globalRetrieval = await harness.retrievalPipeline.searchWithContext(questionText, {
        limit: 3,
        globalIdentitySelfAudienceEntityId: harness.entityRepository.findByName("self"),
        attentionWeights: computeWeights("reflective", {
          currentGoals: [],
          hasActiveValues: false,
          hasTemporalCue: false,
        }),
        goalDescriptions: [],
        includeOpenQuestions: false,
      });

      expect(globalRetrieval.confidence.overall).toBeGreaterThan(
        harness.config.offline.ruminator.resolveConfidenceThreshold,
      );

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items).toEqual([
        expect.objectContaining({
          action: "mark_unresolved",
          question_id: question.id,
          next_unresolved_rumination_ticks: 1,
        }),
      ]);
      expect(llm.requests).toHaveLength(0);
    } finally {
      await harness.cleanup();
    }
  });

  it("plans urgency bumps and abandonments without LLM calls when evidence is weak", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(40 * 24 * 60 * 60 * 1_000),
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const staleQuestion = harness.openQuestionsRepository.add({
        question: "What was the exact rollback order?",
        urgency: 0.1,
        source: "user",
        created_at: 0,
        last_touched: 0,
        provenance: { kind: "manual" },
      });
      const agingQuestion = harness.openQuestionsRepository.add({
        question: "Should I revisit Atlas logging?",
        urgency: 0.4,
        source: "reflection",
        created_at: 0,
        provenance: { kind: "manual" },
        last_touched: 0,
      });

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "abandon",
            question_id: staleQuestion.id,
          }),
          expect.objectContaining({
            action: "bump_urgency",
            question_id: agingQuestion.id,
          }),
        ]),
      );

      await process.apply(harness.createContext(), plan);

      expect(harness.openQuestionsRepository.get(staleQuestion.id)?.status).toBe("abandoned");
      expect(harness.openQuestionsRepository.get(agingQuestion.id)?.urgency).toBe(0.45);
      expect(harness.openQuestionsRepository.get(agingQuestion.id)).toMatchObject({
        unresolved_rumination_ticks: 1,
        last_ruminated_at: harness.clock.now(),
      });
      expect(harness.identityEventRepository.list({ recordType: "open_question" })).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "abandon",
            record_id: staleQuestion.id,
            provenance: {
              kind: "offline",
              process: "ruminator",
            },
          }),
          expect.objectContaining({
            action: "bump_urgency",
            record_id: agingQuestion.id,
            provenance: {
              kind: "offline",
              process: "ruminator",
            },
          }),
        ]),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("abandons by unresolved rumination ticks when the day clock has not advanced", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(1_000_000),
      configOverrides: {
        offline: {
          ruminator: {
            stalenessTicks: 2,
          },
        },
      },
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const question = harness.openQuestionsRepository.add({
        question: "What should happen with the unresolved Sunday ritual question?",
        urgency: 0.1,
        source: "reflection",
        created_at: harness.clock.now(),
        last_touched: harness.clock.now(),
        provenance: { kind: "manual" },
      });

      const firstPlan = await process.plan(harness.createContext(), {});

      expect(firstPlan.items).toEqual([
        expect.objectContaining({
          action: "mark_unresolved",
          question_id: question.id,
          next_unresolved_rumination_ticks: 1,
        }),
      ]);
      expect(process.preview(firstPlan).changes).toEqual([]);

      await process.apply(harness.createContext(), firstPlan);

      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        status: "open",
        unresolved_rumination_ticks: 1,
      });

      const secondPlan = await process.plan(harness.createContext(), {});

      expect(harness.clock.now() - question.last_touched).toBe(0);
      expect(secondPlan.items).toEqual([
        expect.objectContaining({
          action: "mark_unresolved",
          question_id: question.id,
          next_unresolved_rumination_ticks: 2,
        }),
      ]);

      await process.apply(harness.createContext(), secondPlan);

      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        status: "open",
        unresolved_rumination_ticks: 2,
      });

      const thirdPlan = await process.plan(harness.createContext(), {});

      expect(thirdPlan.items).toEqual([
        expect.objectContaining({
          action: "abandon",
          question_id: question.id,
        }),
      ]);

      await process.apply(harness.createContext(), thirdPlan);

      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        status: "abandoned",
        unresolved_rumination_ticks: 0,
        last_ruminated_at: null,
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("dismisses stale no-traction questions but keeps ones with active actions", async () => {
    const tracer = new CaptureTracer();
    const harness = await createOfflineTestHarness({
      tracer,
      configOverrides: {
        offline: {
          ruminator: {
            staleNoTractionTicks: 2,
          },
        },
      },
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const stale = harness.openQuestionsRepository.add({
        question: "Should the stale Atlas question stay open?",
        urgency: 0.2,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      const active = harness.openQuestionsRepository.add({
        question: "Should the active Atlas action stay open?",
        urgency: 0.2,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      harness.openQuestionsRepository.markRuminated(stale.id, 2);
      harness.openQuestionsRepository.markRuminated(active.id, 2);
      harness.actionRepository.add({
        id: createActionId(),
        description: "Follow up on the active Atlas question",
        actor: "borg",
        audience_entity_id: null,
        goal_id: null,
        open_question_id: active.id,
        state: "committed_to_do",
        confidence: 0.8,
        provenance_episode_ids: [],
        provenance_stream_entry_ids: [createStreamEntryId()],
        created_at: harness.clock.now(),
        updated_at: harness.clock.now(),
        considering_at: null,
        committed_at: harness.clock.now(),
        scheduled_at: null,
        completed_at: null,
        not_done_at: null,
        unknown_at: null,
      });

      const ctx = harness.createContext();
      const plan = await process.plan(ctx, {});

      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "abandon",
            question_id: stale.id,
            reason: "stale_no_traction",
          }),
          expect.objectContaining({
            action: "mark_unresolved",
            question_id: active.id,
            next_unresolved_rumination_ticks: 3,
          }),
        ]),
      );

      await process.apply(ctx, plan);

      expect(harness.openQuestionsRepository.get(stale.id)).toMatchObject({
        status: "abandoned",
        abandoned_reason: "stale_no_traction",
      });
      expect(harness.openQuestionsRepository.get(active.id)?.status).toBe("open");
      expect(tracer.events).toContainEqual({
        event: "open_question_stale_dismissed",
        data: expect.objectContaining({
          turnId: ctx.runId,
          question_id: stale.id,
          reason: "stale_no_traction",
        }),
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("increments unresolved ticks on urgency bumps and resets them on resolution", async () => {
    const clock = new ManualClock(8 * DAY_MS);
    const questionText = "Why does Atlas deploy fail?";
    const llm = new FakeLLMClient({
      responses: [
        createRuminatorResponse({
          resolution_note: "Atlas now succeeds after the rollback rehearsal.",
          growth_marker: null,
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock,
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(new Map([[questionText, [1, 0, 0, 0]]])),
      configOverrides: {
        offline: {
          ruminator: {
            stalenessTicks: 10,
          },
        },
      },
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const question = harness.openQuestionsRepository.add({
        question: questionText,
        urgency: 0.2,
        source: "reflection",
        created_at: 0,
        last_touched: 0,
        provenance: { kind: "manual" },
      });

      for (let tick = 1; tick <= 4; tick += 1) {
        const plan = await process.plan(harness.createContext(), {});

        expect(plan.items).toEqual([
          expect.objectContaining({
            action: "bump_urgency",
            question_id: question.id,
            next_unresolved_rumination_ticks: tick,
          }),
        ]);

        await process.apply(harness.createContext(), plan);

        expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
          status: "open",
          unresolved_rumination_ticks: tick,
          last_ruminated_at: clock.now(),
        });

        clock.advance(8 * DAY_MS);
      }

      expect(llm.requests).toHaveLength(0);

      const resolutionEpisode = createEpisodeFixture(
        {
          title: "Atlas rollback rehearsal",
          narrative: "Atlas stabilized after a rollback rehearsal.",
          tags: ["atlas", "deploy"],
          significance: 0.95,
          created_at: clock.now(),
          updated_at: clock.now(),
        },
        [1, 0, 0, 0],
      );
      await harness.episodicRepository.insert(resolutionEpisode);

      const resolutionPlan = await process.plan(harness.createContext(), {});

      expect(resolutionPlan.items).toEqual([
        expect.objectContaining({
          action: "resolve",
          question_id: question.id,
          resolution_evidence_episode_ids: [resolutionEpisode.id],
        }),
      ]);

      await process.apply(harness.createContext(), resolutionPlan);

      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        status: "resolved",
        unresolved_rumination_ticks: 0,
        last_ruminated_at: null,
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("abandons by tick staleness after repeated urgency bumps", async () => {
    const clock = new ManualClock(8 * DAY_MS);
    const harness = await createOfflineTestHarness({
      clock,
      configOverrides: {
        offline: {
          ruminator: {
            stalenessTicks: 4,
          },
        },
      },
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const question = harness.openQuestionsRepository.add({
        question: "Should the Sunday ritual question stay open?",
        urgency: 0,
        source: "reflection",
        created_at: 0,
        last_touched: 0,
        provenance: { kind: "manual" },
      });

      for (let tick = 1; tick <= 4; tick += 1) {
        const plan = await process.plan(harness.createContext(), {});

        expect(plan.items).toEqual([
          expect.objectContaining({
            action: "bump_urgency",
            question_id: question.id,
            next_unresolved_rumination_ticks: tick,
          }),
        ]);

        await process.apply(harness.createContext(), plan);

        expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
          status: "open",
          unresolved_rumination_ticks: tick,
        });

        clock.advance(8 * DAY_MS);
      }

      const stalePlan = await process.plan(harness.createContext(), {});

      expect(stalePlan.items).toEqual([
        expect.objectContaining({
          action: "abandon",
          question_id: question.id,
        }),
      ]);

      await process.apply(harness.createContext(), stalePlan);

      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        status: "abandoned",
        unresolved_rumination_ticks: 0,
        last_ruminated_at: null,
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("merges near-duplicate open questions that share entity scope", async () => {
    const sharedNodeId = createSemanticNodeId();
    const firstEpisodeId = createEpisodeId();
    const secondEpisodeId = createEpisodeId();
    const firstQuestion = "Should Madrid practice stay attached to the trip prep goal?";
    const secondQuestion = "Does the Madrid prep question still belong with trip practice?";
    const harness = await createOfflineTestHarness({
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [firstQuestion, [1, 0, 0, 0]],
          [secondQuestion, [1, 0, 0, 0]],
        ]),
      ),
      configOverrides: {
        offline: {
          ruminator: {
            duplicateSimilarityThreshold: 0.9,
          },
        },
      },
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const older = harness.openQuestionsRepository.add({
        question: firstQuestion,
        urgency: 0.4,
        related_episode_ids: [firstEpisodeId],
        related_semantic_node_ids: [sharedNodeId],
        source: "reflection",
        created_at: 1_000,
        last_touched: 1_000,
      });
      const newer = harness.openQuestionsRepository.add({
        question: secondQuestion,
        urgency: 0.8,
        related_episode_ids: [secondEpisodeId],
        related_semantic_node_ids: [sharedNodeId],
        source: "reflection",
        created_at: 2_000,
        last_touched: 2_000,
      });
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "merge_duplicate",
            primary_question_id: older.id,
            duplicate_question_id: newer.id,
          }),
        ]),
      );

      await process.apply(harness.createContext(), plan);

      expect(harness.openQuestionsRepository.get(newer.id)).toBeNull();
      expect(harness.openQuestionsRepository.get(older.id)).toMatchObject({
        status: "open",
        urgency: 0.8,
        related_episode_ids: [firstEpisodeId, secondEpisodeId],
        related_semantic_node_ids: [sharedNodeId],
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("resolves global open questions only from public or self evidence", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createRuminatorResponse({
          resolution_note: "Public evidence resolved the planning question.",
          growth_marker: null,
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const sam = harness.entityRepository.resolve("Sam");
      const alex = harness.entityRepository.resolve("Alex");
      const publicEpisode = createEpisodeFixture(
        {
          title: "Public planning resolution",
          narrative: "Public planning evidence resolved the open question.",
          tags: ["planning"],
          created_at: 2_000_000,
          updated_at: 2_000_000,
        },
        [0, 1, 0, 0],
      );
      const samEpisode = createEpisodeFixture(
        {
          title: "Sam private planning resolution",
          narrative: "Sam shared a private planning resolution.",
          tags: ["planning"],
          audience_entity_id: sam,
          shared: false,
          created_at: 3_000_000,
          updated_at: 3_000_000,
        },
        [0, 1, 0, 0],
      );
      const alexEpisode = createEpisodeFixture(
        {
          title: "Alex private planning resolution",
          narrative: "Alex shared a private planning resolution.",
          tags: ["planning"],
          audience_entity_id: alex,
          shared: false,
          created_at: 4_000_000,
          updated_at: 4_000_000,
        },
        [0, 1, 0, 0],
      );
      await harness.episodicRepository.insert(publicEpisode);
      await harness.episodicRepository.insert(samEpisode);
      await harness.episodicRepository.insert(alexEpisode);
      const question = harness.openQuestionsRepository.add({
        question: "What resolved the planning uncertainty?",
        urgency: 0.7,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items[0]).toMatchObject({
        action: "resolve",
        question_id: question.id,
        resolution_evidence_episode_ids: [publicEpisode.id],
      });
      expect(llm.requests[0]?.messages[0]?.content).toContain("Public planning resolution");
      expect(llm.requests[0]?.messages[0]?.content).not.toContain("Sam private planning");
      expect(llm.requests[0]?.messages[0]?.content).not.toContain("Alex private planning");
    } finally {
      await harness.cleanup();
    }
  });

  it("resolves audience-tagged open questions from audience-visible evidence", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createRuminatorResponse({
          resolution_note: "Sam-scoped evidence resolved the planning question.",
          growth_marker: null,
        }),
      ],
    });
    const questionText = "What resolved Sam's planning uncertainty?";
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(new Map([[questionText, [0, 1, 0, 0]]])),
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const sam = harness.entityRepository.resolve("Sam");
      const alex = harness.entityRepository.resolve("Alex");
      const samEpisode = createEpisodeFixture(
        {
          title: "Sam private planning resolution",
          narrative: "Sam shared the private planning resolution.",
          tags: ["planning"],
          audience_entity_id: sam,
          shared: false,
          created_at: 2_000_000,
          updated_at: 2_000_000,
        },
        [0, 1, 0, 0],
      );
      const alexEpisode = createEpisodeFixture(
        {
          title: "Alex private planning resolution",
          narrative: "Alex shared a different private planning resolution.",
          tags: ["planning"],
          audience_entity_id: alex,
          shared: false,
          created_at: 3_000_000,
          updated_at: 3_000_000,
        },
        [0, 1, 0, 0],
      );
      await harness.episodicRepository.insert(samEpisode);
      await harness.episodicRepository.insert(alexEpisode);
      const question = harness.openQuestionsRepository.add({
        question: questionText,
        urgency: 0.7,
        source: "reflection",
        audience_entity_id: sam,
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items[0]).toMatchObject({
        action: "resolve",
        question_id: question.id,
        resolution_evidence_episode_ids: [samEpisode.id],
      });
      expect(llm.requests[0]?.messages[0]?.content).toContain("Sam private planning resolution");
      expect(llm.requests[0]?.messages[0]?.content).not.toContain("Alex private planning");
    } finally {
      await harness.cleanup();
    }
  });

  it("halts on budget exhaustion without making further LLM calls", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          ...createRuminatorResponse({
            resolution_note: "First answer",
            growth_marker: null,
          }),
          input_tokens: 20,
          output_tokens: 20,
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock: new FixedClock(40 * 24 * 60 * 60 * 1_000),
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const firstEpisode = createEpisodeFixture(
        {
          title: "Atlas deploy fix",
          narrative: "Atlas deploy fix landed.",
          tags: ["atlas", "deploy"],
          created_at: 2_000_000,
          updated_at: 2_000_000,
        },
        [1, 0, 0, 0],
      );
      const secondEpisode = createEpisodeFixture(
        {
          title: "Atlas retry plan",
          narrative: "Atlas retry plan landed.",
          tags: ["atlas", "deploy"],
          created_at: 2_100_000,
          updated_at: 2_100_000,
        },
        [1, 0, 0, 0],
      );
      await harness.episodicRepository.insert(firstEpisode);
      await harness.episodicRepository.insert(secondEpisode);
      harness.openQuestionsRepository.add({
        question: "Why does Atlas deploy fail?",
        urgency: 0.7,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });
      harness.openQuestionsRepository.add({
        question: "Why does Atlas deploy fail again?",
        urgency: 0.65,
        source: "reflection",
        created_at: 1_000_000,
        provenance: { kind: "manual" },
        last_touched: 1_000_000,
      });

      const plan = await process.plan(harness.createContext(), {
        budget: 10,
      });

      expect(plan.budget_exhausted).toBe(true);
      expect(llm.requests).toHaveLength(1);
    } finally {
      await harness.cleanup();
    }
  });
});
