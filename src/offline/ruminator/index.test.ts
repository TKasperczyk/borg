import { describe, expect, it } from "vitest";

import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../../tracing/tracer.js";
import { computeWeights } from "../../cognition/attention/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { expectedRecordVersion } from "../../memory/common/cas.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";
import {
  createActionId,
  DEFAULT_SESSION_ID,
  createEpisodeId,
  createOpenQuestionId,
  createSemanticNodeId,
  createStreamEntryId,
} from "../../util/ids.js";
import { SELF_RECALL_SCOPE, type RetrievedEpisode } from "../../retrieval/index.js";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  TestEmbeddingClient,
} from "../test-support.js";
import { RUMINATOR_SYSTEM_PROMPT, RuminatorProcess } from "./index.js";

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
        input: {
          outcome: "resolved",
          ...input,
        },
      },
    ],
  };
}

function createStillOpenRuminatorResponse(input: {
  reasoning: string;
  tensions: string[];
  connected_open_question_ids?: string[];
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
        input: {
          outcome: "still_open",
          reasoning: input.reasoning,
          tensions: input.tensions,
          connected_open_question_ids: input.connected_open_question_ids ?? [],
        },
      },
    ],
  };
}

function retrievedEpisode(
  episode: ReturnType<typeof createEpisodeFixture>,
  score: number,
): RetrievedEpisode {
  return {
    episode,
    score,
    rawScore: score,
    scoreBreakdown: {
      similarity: score,
      decayedSalience: score,
      heat: 0,
      goalRelevance: 0,
      valueAlignment: 0,
      timeRelevance: 0,
      moodBoost: 0,
      socialRelevance: 0,
      entityRelevance: 0,
      suppressionPenalty: 0,
    },
    citationChain: [],
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
      await harness.episodicRepository.createEpisode(episode);
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

  it("leaves sqlite retrieval state and Lance episode state unchanged during dry-run planning", async () => {
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
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(new Map([[questionText, [1, 0, 0, 0]]])),
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
      await harness.episodicRepository.createEpisode(episode);
      harness.openQuestionsRepository.add({
        question: questionText,
        urgency: 0.7,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });
      const readSqliteRetrievalState = () => ({
        stats: harness.db
          .prepare("SELECT * FROM episode_stats WHERE episode_id = ?")
          .all(episode.id),
        index: harness.db
          .prepare("SELECT * FROM episode_index WHERE episode_id = ?")
          .all(episode.id),
        logs: harness.db
          .prepare("SELECT * FROM retrieval_log WHERE episode_id = ? ORDER BY timestamp, score")
          .all(episode.id),
        recall: harness.db.prepare("SELECT * FROM recall_state ORDER BY scope_key").all(),
      });
      const sqliteBefore = readSqliteRetrievalState();
      const lanceBefore = await harness.episodicRepository.get(episode.id, {
        includeArchived: true,
      });

      const plan = await process.plan(harness.createContext(), { dryRun: true });

      expect(plan.items).toHaveLength(1);
      expect(readSqliteRetrievalState()).toEqual(sqliteBefore);
      expect(await harness.episodicRepository.get(episode.id, { includeArchived: true })).toEqual(
        lanceBefore,
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("persists still-open deliberations and renders recent rumination history", async () => {
    const clock = new FixedClock(3_000_000);
    const questionText = "What still explains the Atlas rollout tension?";
    const tracer = new CaptureTracer();
    const invalidConnectedOpenQuestionId = createOpenQuestionId();
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock,
      tracer,
      embeddingClient: new TestEmbeddingClient(new Map([[questionText, [1, 0, 0, 0]]])),
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const episode = createEpisodeFixture(
        {
          title: "Atlas rollout tension",
          narrative: "Atlas rollout evidence clarified timing but did not settle ownership.",
          tags: ["atlas", "rollout"],
          significance: 0.95,
          created_at: 2_000_000,
          updated_at: 2_000_000,
        },
        [1, 0, 0, 0],
      );
      await harness.episodicRepository.createEpisode(episode);
      const question = harness.openQuestionsRepository.add({
        question: questionText,
        urgency: 0.7,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });
      const connected = harness.openQuestionsRepository.add({
        question: "Is rollout ownership still unresolved?",
        urgency: 0.5,
        source: "reflection",
        created_at: 1_100_000,
        last_touched: 2_500_000,
        provenance: { kind: "manual" },
      });
      llm.pushResponse(
        createStillOpenRuminatorResponse({
          reasoning:
            "The fresh evidence narrows the rollout tension, but it does not yet settle whether scheduling or ownership is the main cause.",
          tensions: [
            "Scheduling evidence points one way while ownership evidence remains unresolved.",
          ],
          connected_open_question_ids: [connected.id, invalidConnectedOpenQuestionId],
        }),
      );
      harness.openQuestionsRepository.recordRumination({
        open_question_id: question.id,
        note: "Earlier I saw timing as the live tension.",
        tensions: ["Timing was visible, ownership was not."],
        evidence_episode_ids: [episode.id],
        source_process: "test",
        provenance: { kind: "manual" },
        created_at: 1_500_000,
      });

      const plan = await process.plan(harness.createContext(), {});
      const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

      expect(llm.requests[0]?.system).toBe(RUMINATOR_SYSTEM_PROMPT);
      expect(prompt).toContain("Connected open-question candidates:");
      expect(prompt).toContain(connected.id);
      expect(prompt).toContain("Is rollout ownership still unresolved?");
      expect(prompt).not.toContain(invalidConnectedOpenQuestionId);
      expect(prompt).toContain("Earlier I saw timing as the live tension.");
      expect(prompt).toContain("disclosure_class=self_private");
      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "mark_unresolved",
            question_id: question.id,
            next_unresolved_rumination_ticks: 1,
            rumination_note:
              "The fresh evidence narrows the rollout tension, but it does not yet settle whether scheduling or ownership is the main cause.",
            tensions: [
              "Scheduling evidence points one way while ownership evidence remains unresolved.",
            ],
            evidence_episode_ids: [episode.id],
            connected_open_question_ids: [connected.id, invalidConnectedOpenQuestionId],
          }),
        ]),
      );

      const applyContext = harness.createContext();
      await process.apply(applyContext, plan);

      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        status: "open",
        unresolved_rumination_ticks: 1,
        last_ruminated_at: clock.now(),
      });
      const ruminations = harness.openQuestionsRepository.listRecentRuminations(question.id, {
        limit: 5,
      });
      expect(ruminations.map((rumination) => rumination.note)).toEqual([
        "The fresh evidence narrows the rollout tension, but it does not yet settle whether scheduling or ownership is the main cause.",
        "Earlier I saw timing as the live tension.",
      ]);
      expect(ruminations[0]).toMatchObject({
        connected_open_question_ids: [connected.id],
        source_process: "ruminator",
        source_run_id: applyContext.runId,
        source_turn_id: null,
        provenance: { kind: "offline", process: "ruminator" },
      });
      expect(tracer.events).toContainEqual({
        event: "open_question_rumination.connected_ids_dropped",
        data: {
          turnId: applyContext.runId,
          oq_id: question.id,
          dropped_connected_open_question_ids: [invalidConnectedOpenQuestionId],
          reason: "missing_or_not_open",
        },
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("coerces a string tensions value into an array instead of rejecting the rumination", async () => {
    const clock = new FixedClock(3_000_000);
    const questionText = "What still explains the Atlas rollout tension?";
    const tensionText = "Timing is visible but ownership of the rollout is not.";
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock,
      embeddingClient: new TestEmbeddingClient(new Map([[questionText, [1, 0, 0, 0]]])),
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const episode = createEpisodeFixture(
        {
          title: "Atlas rollout tension",
          narrative: "Atlas rollout evidence clarified timing but did not settle ownership.",
          tags: ["atlas", "rollout"],
          significance: 0.95,
          created_at: 2_000_000,
          updated_at: 2_000_000,
        },
        [1, 0, 0, 0],
      );
      await harness.episodicRepository.createEpisode(episode);
      const question = harness.openQuestionsRepository.add({
        question: questionText,
        urgency: 0.5,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });
      // The model returns `tensions` as a bare string instead of an array.
      llm.pushResponse({
        text: "",
        input_tokens: 50,
        output_tokens: 40,
        stop_reason: "tool_use" as const,
        tool_calls: [
          {
            id: "toolu_1",
            name: RUMINATOR_TOOL_NAME,
            input: {
              outcome: "still_open",
              reasoning: "The evidence narrows the tension but does not settle the question.",
              tensions: tensionText,
              connected_open_question_ids: [],
            },
          },
        ],
      });

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.errors).toEqual([]);
      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "mark_unresolved",
            question_id: question.id,
            tensions: [tensionText],
          }),
        ]),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("persists disclosure from every rendered row when private strong evidence is outside the top hits", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createRuminatorResponse({
          resolution_note: "Alex-private evidence resolved the dormant planning question.",
          growth_marker: {
            category: "understanding",
            what_changed: "I understand the private planning resolution.",
            before_description: "The planning resolution was unclear.",
            after_description: "Alex-private evidence supplied the answer.",
            confidence: 0.8,
          },
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      configOverrides: {
        offline: {
          ruminator: {
            resolveConfidenceThreshold: 0.01,
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
      const alex = harness.entityRepository.resolve("Alex");
      const stalePublicEpisodes = [0, 1, 2].map((index) =>
        createEpisodeFixture({
          title: `Public stale planning note ${index}`,
          narrative: `Public stale planning note ${index}.`,
          tags: ["planning"],
          created_at: 1_100_000 + index,
          updated_at: 1_100_000 + index,
        }),
      );
      const privateStrongEpisode = createEpisodeFixture({
        title: "Alex private resolution evidence",
        narrative: "Alex-private evidence gives the actual resolution.",
        tags: ["planning"],
        audience_entity_id: alex,
        shared: false,
        created_at: 2_500_000,
        updated_at: 2_500_000,
      });
      const question = harness.openQuestionsRepository.add({
        question: "What resolved the planning uncertainty?",
        urgency: 0.8,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 2_000_000,
        provenance: { kind: "manual" },
      });
      const ctx = harness.createContext();
      const plan = await process.plan(
        {
          ...ctx,
          retrievalPipeline: {
            ...ctx.retrievalPipeline,
            recallEpisodesForCognition: async () => ({
              episodes: [
                ...stalePublicEpisodes.map((episode) => retrievedEpisode(episode, 0.95)),
                retrievedEpisode(privateStrongEpisode, 0.9),
              ],
            }),
          } as unknown as typeof ctx.retrievalPipeline,
        },
        {},
      );
      const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

      expect(prompt).toContain("Alex private resolution evidence");
      expect(prompt).toContain("Public stale planning note 0");
      expect(prompt).toContain("Public stale planning note 1");
      expect(prompt).toContain(`"id":"${question.id}"`);
      expect(prompt).toContain("disclosure_class=self_private");
      expect(prompt).toContain(SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE);
      expect(plan.items[0]).toMatchObject({
        action: "resolve",
        question_id: question.id,
        resolution_evidence_episode_ids: expect.arrayContaining([
          privateStrongEpisode.id,
          stalePublicEpisodes[0]!.id,
          stalePublicEpisodes[1]!.id,
        ]),
        resolution_disclosure_label: {
          disclosureClass: "relationship_private",
          originAudienceEntityIds: [alex],
          privateToEntityIds: [alex],
          publicToEntityIds: [],
        },
        growth_marker: expect.objectContaining({
          evidence_episode_ids: expect.arrayContaining([
            privateStrongEpisode.id,
            stalePublicEpisodes[0]!.id,
            stalePublicEpisodes[1]!.id,
          ]),
          disclosure_label: {
            disclosureClass: "relationship_private",
            originAudienceEntityIds: [alex],
            privateToEntityIds: [alex],
            publicToEntityIds: [],
          },
        }),
      });

      await process.apply(harness.createContext(), plan);

      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        resolution_disclosure_label: {
          disclosureClass: "relationship_private",
          originAudienceEntityIds: [alex],
          privateToEntityIds: [alex],
          publicToEntityIds: [],
        },
      });
      expect(harness.growthMarkersRepository.list()[0]).toMatchObject({
        disclosure_label: {
          disclosureClass: "relationship_private",
          originAudienceEntityIds: [alex],
          privateToEntityIds: [alex],
          publicToEntityIds: [],
        },
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("persists private disclosure when strong public evidence is rendered with private context", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createRuminatorResponse({
          resolution_note: "Public and Alex-private evidence resolved the planning question.",
          growth_marker: {
            category: "understanding",
            what_changed: "I understand the mixed-source planning resolution.",
            before_description: "The planning answer was incomplete.",
            after_description: "The rendered evidence included the missing private context.",
            confidence: 0.8,
          },
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      configOverrides: {
        offline: {
          ruminator: {
            resolveConfidenceThreshold: 0.01,
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
      const alex = harness.entityRepository.resolve("Alex");
      const publicStrongEpisode = createEpisodeFixture({
        title: "Public strongest planning evidence",
        narrative: "Public evidence is the highest-scoring fresh answer.",
        tags: ["planning"],
        created_at: 2_500_000,
        updated_at: 2_500_000,
      });
      const privateRenderedEpisode = createEpisodeFixture({
        title: "Alex private rendered planning context",
        narrative: "Alex-private context is also rendered to the ruminator.",
        tags: ["planning"],
        audience_entity_id: alex,
        shared: false,
        created_at: 2_400_000,
        updated_at: 2_400_000,
      });
      const publicRenderedEpisode = createEpisodeFixture({
        title: "Public secondary planning evidence",
        narrative: "Another public row is rendered as context.",
        tags: ["planning"],
        created_at: 2_300_000,
        updated_at: 2_300_000,
      });
      const question = harness.openQuestionsRepository.add({
        question: "What resolved the mixed planning uncertainty?",
        urgency: 0.8,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 2_000_000,
        provenance: { kind: "manual" },
      });
      const ctx = harness.createContext();
      const plan = await process.plan(
        {
          ...ctx,
          retrievalPipeline: {
            ...ctx.retrievalPipeline,
            recallEpisodesForCognition: async () => ({
              episodes: [
                retrievedEpisode(publicStrongEpisode, 0.95),
                retrievedEpisode(privateRenderedEpisode, 0.8),
                retrievedEpisode(publicRenderedEpisode, 0.7),
              ],
            }),
          } as unknown as typeof ctx.retrievalPipeline,
        },
        {},
      );
      const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

      expect(prompt).toContain("Public strongest planning evidence");
      expect(prompt).toContain("Alex private rendered planning context");
      expect(plan.items[0]).toMatchObject({
        action: "resolve",
        question_id: question.id,
        resolution_evidence_episode_ids: expect.arrayContaining([
          publicStrongEpisode.id,
          privateRenderedEpisode.id,
          publicRenderedEpisode.id,
        ]),
        resolution_disclosure_label: {
          disclosureClass: "relationship_private",
          originAudienceEntityIds: [alex],
          privateToEntityIds: [alex],
          publicToEntityIds: [],
        },
        growth_marker: expect.objectContaining({
          evidence_episode_ids: expect.arrayContaining([
            publicStrongEpisode.id,
            privateRenderedEpisode.id,
            publicRenderedEpisode.id,
          ]),
          disclosure_label: {
            disclosureClass: "relationship_private",
            originAudienceEntityIds: [alex],
            privateToEntityIds: [alex],
            publicToEntityIds: [],
          },
        }),
      });

      await process.apply(harness.createContext(), plan);

      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        resolution_disclosure_label: {
          disclosureClass: "relationship_private",
          originAudienceEntityIds: [alex],
          privateToEntityIds: [alex],
          publicToEntityIds: [],
        },
      });
      expect(harness.growthMarkersRepository.list()[0]).toMatchObject({
        disclosure_label: {
          disclosureClass: "relationship_private",
          originAudienceEntityIds: [alex],
          privateToEntityIds: [alex],
          publicToEntityIds: [],
        },
      });
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
      await harness.episodicRepository.createEpisode(episode);
      const question = harness.openQuestionsRepository.add({
        question: "Why does Atlas deploy fail?",
        urgency: 0.7,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });
      const retrieval = await harness.retrievalPipeline.recallEpisodesForCognition(
        question.question,
        {
          limit: 3,
          recallContext: {
            reader: SELF_RECALL_SCOPE,
            currentSessionId: DEFAULT_SESSION_ID,
            currentAudienceEntityId: question.audience_entity_id,
            currentParticipantEntityIds:
              question.audience_entity_id === null ? [] : [question.audience_entity_id],
          },
          attentionWeights: computeWeights("reflective", {
            currentGoals: [],
            hasActiveValues: false,
            hasTemporalCue: false,
          }),
          goalDescriptions: [],
          includeOpenQuestions: false,
        },
      );

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

  it("requires merged eligible evidence confidence for audience-scoped resolution", async () => {
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
      await harness.episodicRepository.createEpisode(oldGlobalEpisode);
      await harness.episodicRepository.createEpisode(weakAudienceEpisode);
      const question = harness.openQuestionsRepository.add({
        question: questionText,
        urgency: 0.7,
        source: "reflection",
        audience_entity_id: sam,
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });

      const globalRetrieval = await harness.retrievalPipeline.recallEpisodesForCognition(
        questionText,
        {
          limit: 3,
          recallContext: {
            reader: SELF_RECALL_SCOPE,
            currentSessionId: DEFAULT_SESSION_ID,
            currentAudienceEntityId: question.audience_entity_id,
            currentParticipantEntityIds:
              question.audience_entity_id === null ? [] : [question.audience_entity_id],
          },
          attentionWeights: computeWeights("reflective", {
            currentGoals: [],
            hasActiveValues: false,
            hasTemporalCue: false,
          }),
          goalDescriptions: [],
          includeOpenQuestions: false,
        },
      );

      expect(globalRetrieval.episodes.map((item) => item.episode.id)).toEqual(
        expect.arrayContaining([oldGlobalEpisode.id, weakAudienceEpisode.id]),
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

  it("rejects stale apply plans when the saved open question snapshot lacks a version", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(40 * DAY_MS),
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const question = harness.openQuestionsRepository.add({
        question: "Which unversioned rumination plan is stale?",
        urgency: 0.4,
        source: "reflection",
        created_at: 0,
        last_touched: 0,
        provenance: { kind: "manual" },
      });
      const { record_version: _recordVersion, ...previousWithoutVersion } = question;

      harness.openQuestionsRepository.bumpUrgency(question.id, 0.1, {
        expectedVersion: expectedRecordVersion(question),
      });

      await expect(
        process.apply(harness.createContext(), {
          process: "ruminator",
          items: [
            {
              action: "mark_unresolved",
              question_id: question.id,
              previous: previousWithoutVersion,
              next_unresolved_rumination_ticks: 1,
              rumination_note: null,
              tensions: [],
              connected_open_question_ids: [],
              evidence_episode_ids: [],
              evidence_stream_entry_ids: [],
            },
          ],
          errors: [],
          tokens_used: 0,
          budget_exhausted: false,
        }),
      ).rejects.toThrow(IdentityCasMismatchError);
      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        unresolved_rumination_ticks: 0,
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
        expired_at: null,
        archived_at: null,
        unknown_at: null,
        canonicalized_by_artifact_entry_id: null,
        session_scope: null,
        session_anchor_id: null,
        last_referenced_at_ms: harness.clock.now(),
        last_referenced_turn_counter: null,
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
        event: "open_question_resolution.rejected",
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
            staleNoTractionTicks: 10,
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
      await harness.episodicRepository.createEpisode(resolutionEpisode);

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

  it("merges near-duplicate open questions that share entity scope", async () => {
    const tracer = new CaptureTracer();
    const sharedNodeId = createSemanticNodeId();
    const firstEpisodeId = createEpisodeId();
    const secondEpisodeId = createEpisodeId();
    const duplicateStreamId = createStreamEntryId();
    const firstQuestion = "Should Madrid practice stay attached to the trip prep goal?";
    const secondQuestion = "Does the Madrid prep question still belong with trip practice?";
    const harness = await createOfflineTestHarness({
      tracer,
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
        provenance: { kind: "episodes", episode_ids: [firstEpisodeId] },
        source: "reflection",
        created_at: 1_000,
        last_touched: 1_000,
      });
      const newer = harness.openQuestionsRepository.add({
        question: secondQuestion,
        urgency: 0.8,
        related_semantic_node_ids: [sharedNodeId],
        provenance: { kind: "episodes", episode_ids: [secondEpisodeId] },
        source: "reflection",
        created_at: 2_000,
        last_touched: 2_000,
      });
      harness.openQuestionsRepository.update(newer.id, {
        resolution_evidence_stream_entry_ids: [duplicateStreamId],
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
        provenance: { kind: "episodes", episode_ids: [firstEpisodeId] },
        resolution_evidence_stream_entry_ids: [duplicateStreamId],
      });
      expect(tracer.events).toContainEqual({
        event: "open_question_resolution.transitioned",
        data: expect.objectContaining({
          kept_oq_id: older.id,
          deleted_oq_id: newer.id,
          similarity_score: 1,
          evidence_folded_count: 3,
        }),
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("folds a duplicate whose merged dedupe_key collides with the primary's", async () => {
    // Regression: merge_duplicate recomputes the primary's dedupe_key from the
    // folded id-set. When that recomputed key matched the duplicate's own key,
    // updating the primary while the duplicate still existed raised a UNIQUE
    // violation on open_questions.dedupe_key and aborted the whole ruminator run.
    // The duplicate must be removed first so the fold can claim the key.
    const harness = await createOfflineTestHarness({});
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const sharedQuestion = "Does the connector still need a session id?";
      const epA = createEpisodeId();
      const epB = createEpisodeId();

      // Same question text, different id-sets -> distinct dedupe_keys at insert.
      const primary = harness.openQuestionsRepository.add({
        question: sharedQuestion,
        urgency: 0.4,
        related_episode_ids: [epA],
        provenance: { kind: "manual" },
        source: "reflection",
        created_at: 1_000,
        last_touched: 1_000,
      });
      const duplicate = harness.openQuestionsRepository.add({
        question: sharedQuestion,
        urgency: 0.8,
        related_episode_ids: [epA, epB],
        provenance: { kind: "manual" },
        source: "reflection",
        created_at: 2_000,
        last_touched: 2_000,
      });
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      // Folding the duplicate's ids into the primary yields related_episode_ids
      // [epA, epB] with identical text -> the exact dedupe_key the duplicate holds.
      const plan = {
        process: "ruminator" as const,
        items: [
          {
            action: "merge_duplicate" as const,
            primary_question_id: primary.id,
            duplicate_question_id: duplicate.id,
            previous_primary: primary,
            previous_duplicate: duplicate,
            similarity: 1,
          },
        ],
        errors: [],
        tokens_used: 0,
        budget_exhausted: false,
      };

      await expect(process.apply(harness.createContext(), plan)).resolves.toMatchObject({
        process: "ruminator",
        errors: [],
      });

      expect(harness.openQuestionsRepository.get(duplicate.id)).toBeNull();
      expect(harness.openQuestionsRepository.get(primary.id)).toMatchObject({
        status: "open",
        urgency: 0.8,
        related_episode_ids: [epA, epB],
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("merges near-duplicate open questions when both lack semantic-node handles", async () => {
    const tracer = new CaptureTracer();
    const firstQuestion = "Should the trip prep checklist stay open?";
    const secondQuestion = "Is the trip prep checklist still relevant?";
    const harness = await createOfflineTestHarness({
      tracer,
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
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 1_000,
        last_touched: 1_000,
      });
      const newer = harness.openQuestionsRepository.add({
        question: secondQuestion,
        urgency: 0.8,
        source: "reflection",
        provenance: { kind: "manual" },
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
    } finally {
      await harness.cleanup();
    }
  });

  it("does not merge a scoped OQ with an unscoped OQ", async () => {
    const tracer = new CaptureTracer();
    const sharedNodeId = createSemanticNodeId();
    const scopedQuestion = "Should the scoped Madrid prep question stay open?";
    const unscopedQuestion = "Is the unscoped Madrid prep checklist still relevant?";
    const harness = await createOfflineTestHarness({
      tracer,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [scopedQuestion, [1, 0, 0, 0]],
          [unscopedQuestion, [1, 0, 0, 0]],
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
      const scoped = harness.openQuestionsRepository.add({
        question: scopedQuestion,
        urgency: 0.4,
        related_semantic_node_ids: [sharedNodeId],
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 1_000,
        last_touched: 1_000,
      });
      const unscoped = harness.openQuestionsRepository.add({
        question: unscopedQuestion,
        urgency: 0.8,
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 2_000,
        last_touched: 2_000,
      });
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      const plan = await process.plan(harness.createContext(), {});

      const mergeItems = plan.items.filter((item) => item.action === "merge_duplicate");
      expect(mergeItems).toHaveLength(0);
      expect(harness.openQuestionsRepository.get(scoped.id)?.status).toBe("open");
      expect(harness.openQuestionsRepository.get(unscoped.id)?.status).toBe("open");
    } finally {
      await harness.cleanup();
    }
  });

  it("does not stale-dismiss a merge primary in the same plan", async () => {
    const tracer = new CaptureTracer();
    const sharedNodeId = createSemanticNodeId();
    const primaryQuestion = "Should the long-running Madrid practice question stay open?";
    const duplicateQuestion = "Is the Madrid practice question still on the docket?";
    const harness = await createOfflineTestHarness({
      tracer,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [primaryQuestion, [1, 0, 0, 0]],
          [duplicateQuestion, [1, 0, 0, 0]],
        ]),
      ),
      configOverrides: {
        offline: {
          ruminator: {
            duplicateSimilarityThreshold: 0.9,
            staleNoTractionTicks: 2,
            maxQuestionsPerRun: 1,
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
      const primary = harness.openQuestionsRepository.add({
        question: primaryQuestion,
        urgency: 0.4,
        related_semantic_node_ids: [sharedNodeId],
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 1_000,
        last_touched: 1_000,
      });
      const duplicate = harness.openQuestionsRepository.add({
        question: duplicateQuestion,
        urgency: 0.8,
        related_semantic_node_ids: [sharedNodeId],
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 2_000,
        last_touched: 2_000,
      });
      harness.openQuestionsRepository.markRuminated(primary.id, 2);
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "merge_duplicate",
            primary_question_id: primary.id,
            duplicate_question_id: duplicate.id,
          }),
        ]),
      );
      expect(
        plan.items.some((item) => item.action === "abandon" && item.question_id === primary.id),
      ).toBe(false);
    } finally {
      await harness.cleanup();
    }
  });

  it("dismisses stale OQs outside the urgency-bounded LLM window", async () => {
    const tracer = new CaptureTracer();
    const harness = await createOfflineTestHarness({
      tracer,
      configOverrides: {
        offline: {
          ruminator: {
            staleNoTractionTicks: 2,
            maxQuestionsPerRun: 1,
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
      const highUrgency = harness.openQuestionsRepository.add({
        question: "High-urgency recent OQ that should not be dismissed?",
        urgency: 0.95,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      const lowUrgencyStale = harness.openQuestionsRepository.add({
        question: "Low-urgency stale OQ outside the LLM window?",
        urgency: 0.1,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      harness.openQuestionsRepository.markRuminated(lowUrgencyStale.id, 2);

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "abandon",
            question_id: lowUrgencyStale.id,
            reason: "stale_no_traction",
          }),
        ]),
      );
      expect(
        plan.items.some((item) => item.action === "abandon" && item.question_id === highUrgency.id),
      ).toBe(false);
    } finally {
      await harness.cleanup();
    }
  });

  it("resolves global open questions from global labeled evidence", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createRuminatorResponse({
          resolution_note: "Labeled evidence resolved the planning question.",
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
      await harness.episodicRepository.createEpisode(publicEpisode);
      await harness.episodicRepository.createEpisode(samEpisode);
      await harness.episodicRepository.createEpisode(alexEpisode);
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
        resolution_evidence_episode_ids: expect.arrayContaining([
          alexEpisode.id,
          samEpisode.id,
          publicEpisode.id,
        ]),
        resolution_disclosure_label: expect.objectContaining({
          disclosureClass: "relationship_private",
          originAudienceEntityIds: expect.arrayContaining([sam, alex]),
          privateToEntityIds: expect.arrayContaining([sam, alex]),
        }),
      });
      const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

      expect(prompt).toContain("Public planning resolution");
      expect(prompt).toContain("Sam private planning resolution");
      expect(prompt).toContain("Alex private planning resolution");
      expect(prompt).toContain("relationship_private");
      expect(prompt).toContain(sam);
      expect(prompt).toContain(alex);
    } finally {
      await harness.cleanup();
    }
  });

  it("uses an audience tag as a ranking hint without gating global evidence", async () => {
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
      await harness.episodicRepository.createEpisode(samEpisode);
      await harness.episodicRepository.createEpisode(alexEpisode);
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
        resolution_disclosure_label: expect.objectContaining({
          disclosureClass: "relationship_private",
        }),
      });
      const resolvedItem = plan.items[0];
      const resolvedEpisodeId =
        resolvedItem?.action === "resolve" ? resolvedItem.resolution_evidence_episode_ids[0] : null;
      const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

      expect([samEpisode.id, alexEpisode.id]).toContain(resolvedEpisodeId);
      expect(prompt).toContain("Sam private planning resolution");
      expect(prompt).toContain("Alex private planning");
      expect(prompt).toContain("relationship_private");
      expect(prompt).toContain(sam);
      expect(prompt).toContain(alex);
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
      await harness.episodicRepository.createEpisode(firstEpisode);
      await harness.episodicRepository.createEpisode(secondEpisode);
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
