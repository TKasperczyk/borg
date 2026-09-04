import { describe, expect, it, vi } from "vitest";

import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../../tracing/tracer.js";
import { computeWeights } from "../../cognition/attention/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { expectedRecordVersion } from "../../memory/common/cas.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";
import {
  createActionId,
  createEntityId,
  createMaintenanceRunId,
  DEFAULT_SESSION_ID,
  createEpisodeId,
  createOpenQuestionId,
  createSemanticNodeId,
  createSharedStateEntryId,
  createStreamEntryId,
} from "../../util/ids.js";
import { SELF_RECALL_SCOPE, type RetrievedEpisode } from "../../retrieval/index.js";
import { makeActionRecord } from "../../test-support/factories/memory.js";

import {
  createEpisodeFixture,
  createOfflineTestHarness as createOfflineTestHarnessBase,
  TestEmbeddingClient,
} from "../test-support.js";
import {
  RUMINATOR_DUPLICATE_TOOL,
  RUMINATOR_SYSTEM_PROMPT,
  RuminatorProcess,
  ruminatorPlanSchema,
  unwrapTensionParameterScaffolding,
  unwrapTensionParameterScaffoldingForParse,
} from "./index.js";

const RUMINATOR_TOOL_NAME = "EmitRuminatorDecisions";
const DAY_MS = 24 * 60 * 60 * 1_000;

type OfflineTestHarnessOptions = NonNullable<Parameters<typeof createOfflineTestHarnessBase>[0]>;

async function createOfflineTestHarness(options: OfflineTestHarnessOptions = {}) {
  return createOfflineTestHarnessBase({
    ...options,
    clock: options.clock ?? new FixedClock(3_000_000),
    configOverrides: {
      ...options.configOverrides,
      offline: {
        ...options.configOverrides?.offline,
        ruminator: {
          revisitPeriodMinDays: 0.001,
          revisitPeriodMaxDays: 0.001,
          ...options.configOverrides?.offline?.ruminator,
        },
      },
    },
  });
}

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

function createDuplicateJudgmentResponse(
  judgments: Array<{
    question_id: string;
    duplicate_of_open_question_id: string;
  }>,
  usage: { input_tokens: number; output_tokens: number } = {
    input_tokens: 50,
    output_tokens: 40,
  },
) {
  return {
    text: "",
    ...usage,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_duplicates",
        name: RUMINATOR_DUPLICATE_TOOL.name,
        input: { duplicate_judgments: judgments },
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

describe("unwrapTensionParameterScaffolding", () => {
  it.each([
    {
      label: "the tensions field with a closed JSON-array payload",
      input:
        '<parameter name="tensions">["First synthetic tension","Second synthetic tension"]</parameter>',
      expected: ["First synthetic tension", "Second synthetic tension"],
    },
    {
      label: "an integer-index field without a closing tag",
      input: '<parameter name="0">Indexed synthetic tension',
      expected: ["Indexed synthetic tension"],
    },
    {
      label: "an empty-name field with a closing tag",
      input: '<parameter name="">Empty-name synthetic tension</parameter>',
      expected: ["Empty-name synthetic tension"],
    },
    {
      label: "an unnamed field without a closing tag",
      input: "<parameter>Unnamed synthetic tension",
      expected: ["Unnamed synthetic tension"],
    },
    {
      label: "an item field with a closing tag",
      input: '<parameter name="item">Item synthetic tension</parameter>',
      expected: ["Item synthetic tension"],
    },
  ])("unwraps $label", ({ input, expected }) => {
    expect(unwrapTensionParameterScaffolding(input)).toEqual(expected);
  });

  it("collects multiple parameter payloads with mixed closing-tag presence", () => {
    expect(
      unwrapTensionParameterScaffolding(
        '<parameter name="0">First synthetic tension</parameter>\n' +
          '<parameter name="1">Second synthetic tension',
      ),
    ).toEqual(["First synthetic tension", "Second synthetic tension"]);
  });

  it("preserves the existing bare-string and JSON-encoded-array tolerances", () => {
    expect(unwrapTensionParameterScaffolding("One synthetic tension")).toEqual([
      "One synthetic tension",
    ]);
    expect(
      unwrapTensionParameterScaffolding('["First synthetic tension","Second synthetic tension"]'),
    ).toEqual(["First synthetic tension", "Second synthetic tension"]);
  });

  it("rejects foreign fields and wrappers without a usable payload", () => {
    expect(() =>
      unwrapTensionParameterScaffolding(
        '<parameter name="growth_marker">synthetic marker text</parameter>',
      ),
    ).toThrow(/Unsupported tension parameter name/);
    expect(() => unwrapTensionParameterScaffolding('<parameter name="item"></parameter>')).toThrow(
      /no usable payload/,
    );
  });
});

describe("unwrapTensionParameterScaffoldingForParse", () => {
  it("keeps the tension payloads the rejecting variant keeps", () => {
    expect(
      unwrapTensionParameterScaffoldingForParse(
        '<parameter name="tensions">["First synthetic tension","Second synthetic tension"]</parameter>',
      ),
    ).toEqual(["First synthetic tension", "Second synthetic tension"]);
    expect(unwrapTensionParameterScaffoldingForParse("One synthetic tension")).toEqual([
      "One synthetic tension",
    ]);
  });

  it("drops payloads addressed to another tool parameter instead of throwing", () => {
    expect(
      unwrapTensionParameterScaffoldingForParse(
        '<parameter name="growth_marker">synthetic marker text</parameter>',
      ),
    ).toEqual([]);
    expect(
      unwrapTensionParameterScaffoldingForParse('<parameter name="growth_marker">null'),
    ).toEqual([]);
  });

  it("keeps the tension wrappers around a dropped foreign one", () => {
    expect(
      unwrapTensionParameterScaffoldingForParse(
        '<parameter name="0">First synthetic tension</parameter>\n' +
          '<parameter name="growth_marker">null</parameter>\n' +
          '<parameter name="1">Second synthetic tension',
      ),
    ).toEqual(["First synthetic tension", "Second synthetic tension"]);
  });
});

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
    const connectedQuestionText = "Is rollout ownership still unresolved?";
    const tracer = new CaptureTracer();
    const invalidConnectedOpenQuestionId = createOpenQuestionId();
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock,
      tracer,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [questionText, [1, 0, 0, 0]],
          [connectedQuestionText, [0, 1, 0, 0]],
        ]),
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
        question: connectedQuestionText,
        urgency: 0.5,
        source: "reflection",
        created_at: 1_100_000,
        last_touched: 2_500_000,
        provenance: { kind: "manual" },
      });
      await harness.openQuestionsRepository.waitForPendingEmbeddings();
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
      llm.pushResponse(createDuplicateJudgmentResponse([]));
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
        source_run_id: plan.rumination_run_id,
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

  it("unwraps parameter scaffolding from a string tensions value", async () => {
    const clock = new FixedClock(3_000_000);
    const questionText = "What still explains the Atlas rollout tension?";
    const tensionText = "Timing is visible but ownership of the rollout is not.";
    const warn = vi.fn();
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
      logger: { warn },
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
      // The model returns one array item serialized as a parameter-wrapped string.
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
              tensions: `<parameter name="item">${tensionText}`,
              connected_open_question_ids: [],
            },
          },
        ],
      });

      const context = harness.createContext();
      const plan = await process.plan(context, {});

      expect(plan.errors).toEqual([]);
      expect(plan.tension_scaffolding_drops).toEqual([]);
      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "mark_unresolved",
            question_id: question.id,
            tensions: [tensionText],
          }),
        ]),
      );
      const result = await process.apply(context, plan);
      expect(result.notes).toEqual(expect.arrayContaining(["tension scaffolding dropped: 0"]));
      expect(warn).not.toHaveBeenCalled();
      expect(
        harness.openQuestionsRepository.listRecentRuminations(question.id)[0]?.tensions,
      ).toEqual([tensionText]);
    } finally {
      await harness.cleanup();
    }
  });

  it.each([
    {
      label: "an array element",
      tensions: ['<parameter name="growth_marker">null'],
      expectedDrop: {
        kind: "foreign_parameter_payload" as const,
        parameter_name: "growth_marker",
      },
    },
    {
      label: "the whole tensions value",
      tensions: '<parameter name="growth_marker">null</parameter>',
      expectedDrop: {
        kind: "foreign_parameter_payload" as const,
        parameter_name: "growth_marker",
      },
    },
    {
      label: "a wire-delimited tension element",
      tensions: '<parameter name="item">synthetic tension <parameter residue',
      expectedDrop: {
        kind: "wire_delimited_element" as const,
      },
    },
  ])(
    "keeps the rumination when discardable parameter scaffolding arrives as $label",
    async ({ tensions, expectedDrop }) => {
      const clock = new FixedClock(3_000_000);
      const questionText = "What still explains the Atlas rollout tension?";
      const reasoningText = "The evidence narrows the tension but does not settle the question.";
      const warn = vi.fn();
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
        logger: { warn },
      });

      try {
        await harness.episodicRepository.createEpisode(
          createEpisodeFixture(
            {
              title: "Atlas rollout tension",
              narrative: "Atlas rollout evidence clarified timing but did not settle ownership.",
              tags: ["atlas", "rollout"],
              significance: 0.95,
              created_at: 2_000_000,
              updated_at: 2_000_000,
            },
            [1, 0, 0, 0],
          ),
        );
        const question = harness.openQuestionsRepository.add({
          question: questionText,
          urgency: 0.5,
          source: "reflection",
          created_at: 1_000_000,
          last_touched: 1_000_000,
          provenance: { kind: "manual" },
        });
        // Another parameter's wire fragment lands in the tensions slot: it is not
        // tension text, and it must not take the note and the tick down with it.
        llm.pushResponse({
          text: "",
          input_tokens: 50,
          output_tokens: 40,
          stop_reason: "tool_use" as const,
          tool_calls: [
            {
              id: "toolu_foreign_wire_shape",
              name: RUMINATOR_TOOL_NAME,
              input: {
                outcome: "still_open",
                reasoning: reasoningText,
                tensions,
                connected_open_question_ids: [],
              },
            },
          ],
        });

        const context = harness.createContext();
        const plan = await process.plan(context, {});

        expect(plan.errors).toEqual([]);
        expect(plan.tension_scaffolding_drops).toEqual([
          {
            open_question_id: question.id,
            ...expectedDrop,
          },
        ]);
        expect(plan.items).toEqual(
          expect.arrayContaining([
            expect.objectContaining({
              action: "mark_unresolved",
              question_id: question.id,
              rumination_note: reasoningText,
              tensions: [],
            }),
          ]),
        );
        const result = await process.apply(context, plan);
        expect(warn).toHaveBeenCalledOnce();
        expect(warn).toHaveBeenCalledWith("Ruminator dropped tension scaffolding", {
          run_id: context.runId,
          open_question_id: question.id,
          ...expectedDrop,
        });

        const stored = harness.openQuestionsRepository.listRecentRuminations(question.id)[0];
        expect(result.notes).toEqual(
          expect.arrayContaining([
            `tension scaffolding dropped: 1; rumination_ids=${stored?.id}; open_question_ids=${question.id}`,
          ]),
        );
        expect(stored?.note).toBe(reasoningText);
        expect(stored?.tensions).toEqual([]);
        expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
          status: "open",
          unresolved_rumination_ticks: 1,
        });
      } finally {
        await harness.cleanup();
      }
    },
  );

  it.each([
    {
      label: "a parameter delimiter at the start of reasoning",
      toolInput: {
        outcome: "still_open",
        reasoning: '<parameter name="reasoning">synthetic reasoning</parameter>',
        tensions: ["Synthetic tension"],
        connected_open_question_ids: [],
      },
    },
    {
      label: "a parameter delimiter at the start of a resolution note",
      toolInput: {
        outcome: "resolved",
        resolution_note: '<parameter name="resolution_note">synthetic resolution</parameter>',
        growth_marker: null,
      },
    },
  ])("fails parsing but still stamps the LLM visit for $label", async ({ toolInput }) => {
    const clock = new FixedClock(3_000_000);
    const questionText = "What remains unsettled in this synthetic rumination?";
    const response = {
      text: "",
      input_tokens: 50,
      output_tokens: 40,
      stop_reason: "tool_use" as const,
      tool_calls: [
        {
          id: "toolu_invalid_wire_shape",
          name: RUMINATOR_TOOL_NAME,
          input: toolInput,
        },
      ],
    };
    const llm = new FakeLLMClient({ responses: [response, response] });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock,
      embeddingClient: new TestEmbeddingClient(new Map([[questionText, [1, 0, 0, 0]]])),
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
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture(
          {
            title: "Synthetic rumination evidence",
            narrative: "Synthetic evidence is relevant to the test question.",
            tags: ["synthetic"],
            significance: 0.95,
            created_at: 2_000_000,
            updated_at: 2_000_000,
          },
          [1, 0, 0, 0],
        ),
      );
      const question = harness.openQuestionsRepository.add({
        question: questionText,
        urgency: 0.5,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });

      const plan = await process.plan(harness.createContext(), {});

      expect(llm.requests).toHaveLength(2);
      expect(plan.items).toEqual([
        expect.objectContaining({
          action: "mark_unresolved",
          question_id: question.id,
          rumination_note: null,
        }),
      ]);
      expect(plan.errors).toHaveLength(1);

      await process.apply(harness.createContext(), plan);

      expect(harness.openQuestionsRepository.listRecentRuminations(question.id)).toEqual([]);
      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        status: "open",
        resolution_note: null,
        unresolved_rumination_ticks: 0,
        last_ruminated_at: clock.now(),
      });
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

  it("rotates the window by overdue ratio and makes touched-since-rumination questions due", async () => {
    const clock = new ManualClock(60 * DAY_MS);
    const lowText = "Which long-idle low-urgency uncertainty remains open?";
    const highText = "Which recently-idle high-urgency uncertainty remains open?";
    const touchedText = "Which newly touched uncertainty needs another look?";
    const neverRuminatedTouchedText = "Which never-ruminated question received new evidence?";
    const futureText = "Which new uncertainty is not due yet?";
    const harness = await createOfflineTestHarnessBase({
      clock,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [lowText, [1, 0, 0, 0]],
          [highText, [0, 1, 0, 0]],
          [touchedText, [0, 0, 1, 0]],
          [neverRuminatedTouchedText, [0.5, 0.5, 0.5, 0.5]],
          [futureText, [0, 0, 0, 1]],
        ]),
      ),
      configOverrides: {
        offline: {
          ruminator: {
            maxQuestionsPerRun: 2,
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
      const low = harness.openQuestionsRepository.add({
        question: lowText,
        urgency: 0,
        source: "reflection",
        created_at: 0,
        last_touched: 0,
        provenance: { kind: "manual" },
      });
      const high = harness.openQuestionsRepository.add({
        question: highText,
        urgency: 1,
        source: "reflection",
        created_at: clock.now() - 3 * DAY_MS,
        last_touched: clock.now() - 3 * DAY_MS,
        provenance: { kind: "manual" },
      });
      const touched = harness.openQuestionsRepository.add({
        question: touchedText,
        urgency: 0.5,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      const future = harness.openQuestionsRepository.add({
        question: futureText,
        urgency: 0.5,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      harness.openQuestionsRepository.markRuminated(touched.id, 1);
      clock.advance(1);
      harness.openQuestionsRepository.touch(touched.id);
      const neverRuminatedTouched = harness.openQuestionsRepository.add({
        question: neverRuminatedTouchedText,
        urgency: 0.4,
        source: "reflection",
        created_at: clock.now() - 1,
        last_touched: clock.now(),
        provenance: { kind: "manual" },
      });
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      const firstPlan = await process.plan(harness.createContext(), {});

      expect(firstPlan.scheduling.due_question_ids).toEqual([
        low.id,
        high.id,
        touched.id,
        neverRuminatedTouched.id,
      ]);
      expect(firstPlan.scheduling.selected_question_ids).toEqual([low.id, high.id]);
      expect(firstPlan.scheduling.visited_question_ids).toEqual([low.id, high.id]);
      expect(firstPlan.scheduling.model_called_question_ids).toEqual([]);
      expect(firstPlan.scheduling.due_question_ids).not.toContain(future.id);
      await process.apply(harness.createContext(), firstPlan);

      const secondPlan = await process.plan(harness.createContext(), {});

      expect(secondPlan.scheduling.due_question_ids).toEqual([
        touched.id,
        neverRuminatedTouched.id,
      ]);
      expect(secondPlan.scheduling.selected_question_ids).toEqual([
        touched.id,
        neverRuminatedTouched.id,
      ]);
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
        unresolved_rumination_ticks: 0,
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

  it("re-reads and stamps when an old saved plan meets a CAS conflict", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(40 * DAY_MS),
    });
    const warn = vi.fn();
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
      logger: { warn },
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
      const ruminationRunId = createMaintenanceRunId();

      const savedPlan = ruminatorPlanSchema.parse({
        process: "ruminator",
        rumination_run_id: ruminationRunId,
        items: [
          {
            action: "mark_unresolved",
            question_id: question.id,
            previous: previousWithoutVersion,
            next_unresolved_rumination_ticks: 1,
            rumination_note: "The stale plan still reached this uncertainty once.",
            tensions: ["The stored version changed before apply."],
            connected_open_question_ids: [],
            evidence_episode_ids: [],
            evidence_stream_entry_ids: [],
          },
        ],
        errors: [],
        tokens_used: 0,
        budget_exhausted: false,
        tension_scaffolding_drops: [],
      });
      const result = await process.apply(harness.createContext(), savedPlan);
      await process.apply(harness.createContext(), savedPlan);

      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        unresolved_rumination_ticks: 1,
        last_ruminated_at: harness.clock.now(),
      });
      expect(harness.openQuestionsRepository.listRecentRuminations(question.id)).toEqual([
        expect.objectContaining({
          note: "The stale plan still reached this uncertainty once.",
          source_run_id: ruminationRunId,
        }),
      ]);
      expect(result.notes).toEqual(
        expect.arrayContaining([expect.stringContaining(`open_question_ids=${question.id}`)]),
      );
      expect(warn).toHaveBeenCalled();
    } finally {
      await harness.cleanup();
    }
  });

  it("shows the rumination counter and the threshold it feeds at the decision site", async () => {
    const questionText = "What still explains the Atlas dismissal threshold?";
    const llm = new FakeLLMClient();
    const clock = new ManualClock(3_000_000);
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock,
      embeddingClient: new TestEmbeddingClient(new Map([[questionText, [1, 0, 0, 0]]])),
      configOverrides: {
        offline: {
          ruminator: {
            staleNoTractionTicks: 7,
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
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture(
          {
            title: "Atlas threshold evidence",
            narrative: "Atlas evidence arrived after the question was last touched.",
            tags: ["atlas"],
            significance: 0.95,
            created_at: 2_000_000,
            updated_at: 2_000_000,
          },
          [1, 0, 0, 0],
        ),
      );
      const question = harness.openQuestionsRepository.add({
        question: questionText,
        urgency: 0.7,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });
      harness.openQuestionsRepository.markRuminated(question.id, 3);
      clock.advance(100_000);
      llm.pushResponse(
        createStillOpenRuminatorResponse({
          reasoning: "The evidence narrows the tension but does not settle it.",
          tensions: ["The threshold is close but the evidence is not in."],
        }),
      );

      await process.plan(harness.createContext(), {});
      const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

      // The count carried is the pre-note value, not the one this result would write, and the
      // threshold is interpolated from config rather than restated -- so a threshold change moves
      // the prompt with it instead of leaving a second copy of the number behind to drift.
      expect(prompt).toContain('"unresolved_rumination_ticks":3');
      expect(prompt).toContain("counts the recorded rumination notes");
      expect(prompt).not.toContain("counts the passes");
      expect(prompt).toContain("have reached 7");
      expect(prompt).not.toContain("have reached 4");
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
        ]),
      );
      expect(plan.items).not.toEqual(
        expect.arrayContaining([expect.objectContaining({ question_id: active.id })]),
      );

      await process.apply(ctx, plan);

      expect(harness.openQuestionsRepository.get(stale.id)).toMatchObject({
        status: "abandoned",
        abandoned_reason: "stale_no_traction",
      });
      expect(harness.openQuestionsRepository.get(active.id)).toMatchObject({
        status: "open",
        unresolved_rumination_ticks: 2,
      });
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

  it("dismisses after four noted stamps but not four note-less stamps", async () => {
    const noteLessText = "Which note-less visits should remain outside the dismissal count?";
    const notedText = "Which recorded ruminations reached the dismissal count?";
    const llm = new FakeLLMClient({ responses: [createDuplicateJudgmentResponse([])] });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [noteLessText, [1, 0, 0, 0]],
          [notedText, [0, 1, 0, 0]],
        ]),
      ),
      configOverrides: {
        offline: {
          ruminator: {
            staleNoTractionTicks: 4,
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
      const noteLess = harness.openQuestionsRepository.add({
        question: noteLessText,
        urgency: 0.4,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      const noted = harness.openQuestionsRepository.add({
        question: notedText,
        urgency: 0.4,
        source: "reflection",
        provenance: { kind: "manual" },
      });

      for (let tick = 1; tick <= 4; tick += 1) {
        const runId = createMaintenanceRunId();
        harness.openQuestionsRepository.stampRuminationForRun({
          open_question_id: noteLess.id,
          source_run_id: runId,
          next_unresolved_rumination_ticks: tick,
          rumination: null,
        });
        harness.openQuestionsRepository.stampRuminationForRun({
          open_question_id: noted.id,
          source_run_id: runId,
          next_unresolved_rumination_ticks: tick,
          rumination: {
            note: `Recorded rumination ${tick}.`,
            source_process: "test",
            provenance: { kind: "manual" },
          },
        });
      }
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      expect(harness.openQuestionsRepository.get(noteLess.id)).toMatchObject({
        unresolved_rumination_ticks: 0,
        last_ruminated_at: harness.clock.now(),
      });
      expect(harness.openQuestionsRepository.get(noted.id)).toMatchObject({
        unresolved_rumination_ticks: 4,
        last_ruminated_at: harness.clock.now(),
      });
      expect(harness.openQuestionsRepository.listRecentRuminations(noted.id)).toHaveLength(4);

      const ctx = harness.createContext();
      const plan = await process.plan(ctx, {});

      expect(plan.items).toEqual([
        expect.objectContaining({
          action: "abandon",
          question_id: noted.id,
          reason: "stale_no_traction",
        }),
      ]);
      expect(
        plan.items.some((item) => "question_id" in item && item.question_id === noteLess.id),
      ).toBe(false);

      await process.apply(ctx, plan);
      expect(harness.openQuestionsRepository.get(noteLess.id)?.status).toBe("open");
      expect(harness.openQuestionsRepository.get(noted.id)?.status).toBe("abandoned");
    } finally {
      await harness.cleanup();
    }
  });

  it("keeps note-less urgency bumps out of unresolved ticks and resets on resolution", async () => {
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

      for (let pass = 1; pass <= 4; pass += 1) {
        const plan = await process.plan(harness.createContext(), {});

        expect(plan.items).toEqual([
          expect.objectContaining({
            action: "bump_urgency",
            question_id: question.id,
            next_unresolved_rumination_ticks: 1,
          }),
        ]);

        await process.apply(harness.createContext(), plan);

        expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
          status: "open",
          unresolved_rumination_ticks: 0,
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
    const llm = new FakeLLMClient({ responses: [createDuplicateJudgmentResponse([])] });
    const sharedNodeId = createSemanticNodeId();
    const firstEpisodeId = createEpisodeId();
    const secondEpisodeId = createEpisodeId();
    const duplicateStreamId = createStreamEntryId();
    const firstQuestion = "Should Madrid practice stay attached to the trip prep goal?";
    const secondQuestion = "Does the Madrid prep question still belong with trip practice?";
    const harness = await createOfflineTestHarness({
      tracer,
      llmClient: llm,
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
      expect(llm.requests).toHaveLength(1);

      await process.apply(harness.createContext(), plan);

      expect(harness.openQuestionsRepository.get(newer.id)).toMatchObject({
        status: "abandoned",
        abandoned_reason: `Merged into open question ${older.id}.`,
      });
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
          abandoned_oq_id: newer.id,
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
    // The duplicate must release its active dedupe key before the fold claims it,
    // while the abandoned row remains as the merge audit trail.
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
      const plan = ruminatorPlanSchema.parse({
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
        tension_scaffolding_drops: [],
      });

      await expect(process.apply(harness.createContext(), plan)).resolves.toMatchObject({
        process: "ruminator",
        errors: [],
      });

      expect(harness.openQuestionsRepository.get(duplicate.id)).toMatchObject({
        status: "abandoned",
        abandoned_reason: `Merged into open question ${primary.id}.`,
      });
      expect(harness.openQuestionsRepository.get(primary.id)).toMatchObject({
        status: "open",
        urgency: 0.8,
        related_episode_ids: [epA, epB],
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("retains merged rows and reparents every mutable open-question reference", async () => {
    const harness = await createOfflineTestHarness({});
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const primary = harness.openQuestionsRepository.add({
        question: "Which durable merge reference should remain primary?",
        urgency: 0.4,
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 1_000,
        last_touched: 1_000,
      });
      const duplicate = harness.openQuestionsRepository.add({
        question: "Which durable merge reference is the duplicate?",
        urgency: 0.7,
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 2_000,
        last_touched: 2_000,
      });
      const rumination = harness.openQuestionsRepository.recordRumination({
        open_question_id: duplicate.id,
        note: "Durable thought from the duplicate.",
        connected_open_question_ids: [primary.id],
        source_process: "test",
        provenance: { kind: "manual" },
      });
      harness.openQuestionsRepository.recordRumination({
        open_question_id: primary.id,
        note: "Primary thought connected to the duplicate.",
        connected_open_question_ids: [duplicate.id],
        source_process: "test",
        provenance: { kind: "manual" },
      });
      const priorStampRunId = createMaintenanceRunId();
      harness.openQuestionsRepository.stampRuminationForRun({
        open_question_id: duplicate.id,
        source_run_id: priorStampRunId,
        next_unresolved_rumination_ticks: 1,
      });
      const action = makeActionRecord({ open_question_id: duplicate.id });
      harness.actionRepository.add(action);

      const audienceId = createEntityId();
      const sharedStateEntryId = createSharedStateEntryId();
      const sharedStateStreamId = createStreamEntryId();
      harness.db
        .prepare(
          `
            INSERT INTO shared_state_artifacts (
              audience_entity_id, record_version, created_at, updated_at,
              last_compiled_at, last_compiled_stream_entry_id
            ) VALUES (?, 1, 1, 1, NULL, NULL)
          `,
        )
        .run(audienceId);
      harness.db
        .prepare(
          `
            INSERT INTO shared_state_entries (
              id, audience_entity_id, state_key, kind, text, owner_entity_id,
              provenance_stream_entry_ids, last_updated_stream_entry_ids,
              created_at, last_updated_at, superseded_by_id, rank, canonicalizes,
              last_updated_turn_global
            ) VALUES (?, ?, ?, 'live', ?, NULL, ?, ?, 1, 1, NULL, 0, ?, NULL)
          `,
        )
        .run(
          sharedStateEntryId,
          audienceId,
          "merge.reference",
          "The open question is tracked here.",
          JSON.stringify([sharedStateStreamId]),
          JSON.stringify([sharedStateStreamId]),
          JSON.stringify({
            goal_ids: [],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [primary.id, duplicate.id],
          }),
        );
      const reviewResult = harness.db
        .prepare(
          `
            INSERT INTO review_queue (kind, refs, reason, created_at, resolved_at, resolution)
            VALUES ('correction', ?, 'Merge reference fixture.', 1, NULL, NULL)
          `,
        )
        .run(
          JSON.stringify({
            target_type: "open_question",
            target_id: duplicate.id,
            patch: {},
          }),
        );
      harness.recallStateRepository.save({
        scopeKey: "ruminator-merge-reference",
        activeHandles: [
          {
            handle: { source: "open_question", openQuestionId: duplicate.id },
            firstSeenTurn: 1,
            lastSeenTurn: 1,
            lastRenderedTurn: null,
            expiresAfterTurn: 3,
            reinforcementCount: 1,
          },
        ],
        suppressedHandles: { [`open_question:${duplicate.id}`]: 4 },
        lastRefreshTurn: 1,
        updatedAt: 1,
        ttlTurns: 6,
      });
      const duplicateWatermark = `autonomy:open-question-dormant:${duplicate.id}:2000`;
      harness.db
        .prepare(
          `
            INSERT INTO stream_watermarks (
              process_name, session_id, last_ts, last_entry_id, updated_at, metadata_json
            ) VALUES (?, ?, 1, ?, 1, ?)
          `,
        )
        .run(
          duplicateWatermark,
          DEFAULT_SESSION_ID,
          createStreamEntryId(),
          JSON.stringify({ open_question_id: duplicate.id }),
        );

      const plan = ruminatorPlanSchema.parse({
        process: "ruminator",
        items: [
          {
            action: "merge_duplicate",
            primary_question_id: primary.id,
            duplicate_question_id: duplicate.id,
            previous_primary: primary,
            previous_duplicate: duplicate,
            similarity: 0.8,
          },
        ],
        errors: [],
        tokens_used: 0,
      });

      await process.apply(harness.createContext(), plan);

      expect(harness.openQuestionsRepository.get(duplicate.id)).toMatchObject({
        status: "abandoned",
        abandoned_reason: `Merged into open question ${primary.id}.`,
      });
      expect(harness.openQuestionsRepository.listRecentRuminations(primary.id)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ id: rumination.id, open_question_id: primary.id }),
        ]),
      );
      expect(harness.openQuestionsRepository.listRecentRuminations(duplicate.id)).toEqual([]);
      expect(harness.actionRepository.get(action.id)?.open_question_id).toBe(primary.id);

      const sharedStateRow = harness.db
        .prepare("SELECT canonicalizes FROM shared_state_entries WHERE id = ?")
        .get(sharedStateEntryId) as { canonicalizes: string };
      expect(JSON.parse(sharedStateRow.canonicalizes)).toMatchObject({
        open_question_ids: [primary.id],
      });
      const reviewRow = harness.db
        .prepare("SELECT refs FROM review_queue WHERE id = ?")
        .get(reviewResult.lastInsertRowid) as { refs: string };
      expect(JSON.parse(reviewRow.refs)).toMatchObject({ target_id: primary.id });
      expect(harness.recallStateRepository.load("ruminator-merge-reference")).toMatchObject({
        activeHandles: [
          expect.objectContaining({
            handle: { source: "open_question", openQuestionId: primary.id },
          }),
        ],
        suppressedHandles: { [`open_question:${primary.id}`]: 4 },
      });
      const currentReferences = harness.db
        .prepare(
          `
            SELECT
              (SELECT COUNT(*) FROM open_question_ruminations
                WHERE open_question_id = ? OR instr(connected_open_question_ids, ?) > 0) AS ruminations,
              (SELECT COUNT(*) FROM open_question_rumination_stamps
                WHERE open_question_id = ?) AS stamps,
              (SELECT COUNT(*) FROM action_records WHERE open_question_id = ?) AS actions,
              (SELECT COUNT(*) FROM shared_state_entries WHERE instr(canonicalizes, ?) > 0) AS shared_state,
              (SELECT COUNT(*) FROM review_queue WHERE instr(refs, ?) > 0) AS reviews,
              (SELECT COUNT(*) FROM recall_state WHERE instr(state_json, ?) > 0) AS recall,
              (SELECT COUNT(*) FROM stream_watermarks
                WHERE instr(process_name, ?) > 0 OR instr(COALESCE(metadata_json, ''), ?) > 0) AS watermarks
          `,
        )
        .get(
          duplicate.id,
          duplicate.id,
          duplicate.id,
          duplicate.id,
          duplicate.id,
          duplicate.id,
          duplicate.id,
          duplicate.id,
          duplicate.id,
        );
      expect(currentReferences).toEqual({
        ruminations: 0,
        stamps: 0,
        actions: 0,
        shared_state: 0,
        reviews: 0,
        recall: 0,
        watermarks: 0,
      });
      expect(
        harness.db
          .prepare("SELECT process_name FROM stream_watermarks WHERE instr(process_name, ?) > 0")
          .get(primary.id),
      ).toBeDefined();
    } finally {
      await harness.cleanup();
    }
  });

  it("merges near-duplicate open questions when both lack semantic-node handles", async () => {
    const tracer = new CaptureTracer();
    const llm = new FakeLLMClient({ responses: [createDuplicateJudgmentResponse([])] });
    const firstQuestion = "Should the trip prep checklist stay open?";
    const secondQuestion = "Is the trip prep checklist still relevant?";
    const harness = await createOfflineTestHarness({
      tracer,
      llmClient: llm,
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

  it("uses one model judgment to merge across the former entity-scope boundary", async () => {
    const tracer = new CaptureTracer();
    const llm = new FakeLLMClient();
    const sharedNodeId = createSemanticNodeId();
    const scopedQuestion = "Should the scoped Madrid prep question stay open?";
    const unscopedQuestion = "Is the unscoped Madrid prep checklist still relevant?";
    const thirdQuestion = "What remains uncertain about another Madrid preparation choice?";
    const harness = await createOfflineTestHarness({
      tracer,
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [scopedQuestion, [1, 0, 0, 0]],
          [unscopedQuestion, [0.1, 0.994987, 0, 0]],
          [thirdQuestion, [-0.1, 0.994987, 0, 0]],
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
      const third = harness.openQuestionsRepository.add({
        question: thirdQuestion,
        urgency: 0.3,
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 3_000,
        last_touched: 3_000,
      });
      await harness.openQuestionsRepository.waitForPendingEmbeddings();
      llm.pushResponse(
        createDuplicateJudgmentResponse([
          {
            question_id: scoped.id,
            duplicate_of_open_question_id: unscoped.id,
          },
          {
            question_id: createOpenQuestionId(),
            duplicate_of_open_question_id: createOpenQuestionId(),
          },
        ]),
      );

      const plan = await process.plan(harness.createContext(), {});

      const mergeItems = plan.items.filter((item) => item.action === "merge_duplicate");
      expect(mergeItems).toEqual([
        expect.objectContaining({
          primary_question_id: scoped.id,
          duplicate_question_id: unscoped.id,
          similarity: expect.closeTo(0.1),
        }),
      ]);
      expect(llm.requests).toHaveLength(1);
      expect(llm.requests[0]?.tool_choice).toEqual({
        type: "tool",
        name: RUMINATOR_DUPLICATE_TOOL.name,
      });
      const duplicatePrompt = String(llm.requests[0]?.messages[0]?.content ?? "");
      expect(duplicatePrompt).toContain(scoped.id);
      expect(duplicatePrompt).toContain(unscoped.id);
      expect(duplicatePrompt).toContain(third.id);
      expect(harness.openQuestionsRepository.get(scoped.id)?.status).toBe("open");
      expect(harness.openQuestionsRepository.get(unscoped.id)?.status).toBe("open");
    } finally {
      await harness.cleanup();
    }
  });

  it("caps duplicate judgment at the highest-cosine candidate pairs", async () => {
    const questions = [
      { text: "Which route should the first plan take?", vector: [1, 0, 0, 0] },
      { text: "Which route should the second plan take?", vector: [0.8, 0.6, 0, 0] },
      { text: "Which route should the third plan take?", vector: [0, 1, 0, 0] },
      { text: "Which route should the fourth plan take?", vector: [0, 0, 1, 0] },
    ] as const;
    const llm = new FakeLLMClient({ responses: [createDuplicateJudgmentResponse([])] });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map(questions.map(({ text, vector }) => [text, vector])),
      ),
      configOverrides: {
        offline: {
          ruminator: {
            duplicateJudgmentMaxPairs: 2,
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
      for (const [index, question] of questions.entries()) {
        harness.openQuestionsRepository.add({
          question: question.text,
          urgency: 0.5,
          source: "reflection",
          provenance: { kind: "manual" },
          created_at: 1_000 + index,
          last_touched: 1_000 + index,
        });
      }
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.duplicate_judgment).toEqual({
        pairs_candidates: 6,
        pairs_presented: 2,
        pairs_skipped_budget: 4,
        judgment_ran: true,
      });
      expect(llm.requests).toHaveLength(1);
      const prompt = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "")) as {
        candidate_pairs: Array<{
          left_candidate: Record<string, unknown>;
          cosine_similarity: number;
        }>;
      };
      expect(prompt.candidate_pairs).toHaveLength(2);
      expect(prompt.candidate_pairs.map((pair) => pair.cosine_similarity)).toEqual([
        expect.closeTo(0.8),
        expect.closeTo(0.6),
      ]);
      expect(Object.keys(prompt.candidate_pairs[0]?.left_candidate ?? {})).toEqual([
        "id",
        "excerpt",
        "urgency",
        "source",
        "disclosure_label",
      ]);
      expect(process.preview(plan).notes).toContain(
        "duplicate judgment: pairs_candidates=6; pairs_presented=2; pairs_skipped_budget=4; judgment_ran=true",
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("installs no cosine backstop merge when duplicate judgment output is malformed", async () => {
    const firstText = "Which malformed duplicate verdict should fail open?";
    const secondText = "Does this malformed duplicate verdict fail open safely?";
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_malformed_duplicates",
              name: RUMINATOR_DUPLICATE_TOOL.name,
              input: { duplicate_judgments: "not-an-array" },
            },
          ],
        },
      ],
    });
    const warn = vi.fn();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [firstText, [1, 0, 0, 0]],
          [secondText, [1, 0, 0, 0]],
        ]),
      ),
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
      logger: { warn },
    });

    try {
      harness.openQuestionsRepository.add({
        question: firstText,
        urgency: 0.6,
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 1_000,
        last_touched: 1_000,
      });
      harness.openQuestionsRepository.add({
        question: secondText,
        urgency: 0.5,
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 2_000,
        last_touched: 2_000,
      });
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items.some((item) => item.action === "merge_duplicate")).toBe(false);
      expect(llm.requests).toHaveLength(1);
      expect(warn).toHaveBeenCalledWith(
        "Ruminator duplicate judgment failed open without backstop merges",
        expect.objectContaining({ error: expect.any(Error) }),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("does not stale-dismiss a merge primary in the same plan", async () => {
    const tracer = new CaptureTracer();
    const llm = new FakeLLMClient({ responses: [createDuplicateJudgmentResponse([])] });
    const sharedNodeId = createSemanticNodeId();
    const primaryQuestion = "Should the long-running Madrid practice question stay open?";
    const duplicateQuestion = "Is the Madrid practice question still on the docket?";
    const harness = await createOfflineTestHarness({
      tracer,
      llmClient: llm,
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

  it("skips duplicate judgment below its remaining-budget reserve and reports the skipped pairs", async () => {
    const firstQuestionText = "What settled the reserved-budget planning question?";
    const secondQuestionText = "What remains open in the other planning question?";
    const llm = new FakeLLMClient({
      responses: [
        {
          ...createRuminatorResponse({
            resolution_note: "The fresh evidence settled the reserved-budget question.",
            growth_marker: null,
          }),
          input_tokens: 60,
          output_tokens: 20,
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [firstQuestionText, [1, 0, 0, 0]],
          [secondQuestionText, [0, 1, 0, 0]],
        ]),
      ),
      configOverrides: {
        offline: {
          ruminator: {
            maxQuestionsPerRun: 1,
            resolveConfidenceThreshold: 0,
            duplicateJudgmentMinRemainingBudgetFraction: 0.25,
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
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture(
          {
            title: "Reserved-budget planning evidence",
            narrative: "Fresh planning evidence settled the first question.",
            tags: ["planning"],
            significance: 0.95,
            created_at: 2_000_000,
            updated_at: 2_000_000,
          },
          [1, 0, 0, 0],
        ),
      );
      const firstQuestion = harness.openQuestionsRepository.add({
        question: firstQuestionText,
        urgency: 0.9,
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 1_000_000,
        last_touched: 1_000_000,
      });
      harness.openQuestionsRepository.add({
        question: secondQuestionText,
        urgency: 0.1,
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 1_100_000,
        last_touched: 1_100_000,
      });
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      const plan = await process.plan(harness.createContext(), { budget: 100 });

      expect(plan.budget_exhausted).toBe(false);
      expect(plan.tokens_used).toBe(80);
      expect(llm.requests).toHaveLength(1);
      expect(llm.requests[0]?.tool_choice).toEqual({
        type: "tool",
        name: RUMINATOR_TOOL_NAME,
      });
      expect(plan.scheduling.visited_question_ids).toEqual([firstQuestion.id]);
      expect(plan.duplicate_judgment).toEqual({
        pairs_candidates: 1,
        pairs_presented: 0,
        pairs_skipped_budget: 1,
        judgment_ran: false,
      });
      const result = await process.apply(harness.createContext(), plan);
      expect(result.notes).toContain(
        "duplicate judgment: pairs_candidates=1; pairs_presented=0; pairs_skipped_budget=1; judgment_ran=false",
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("visits the scheduled question before an over-budget duplicate judgment", async () => {
    const firstQuestionText = "What settled the rotation-first planning question?";
    const secondQuestionText = "What remains open in the rotation companion question?";
    const warn = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        {
          ...createRuminatorResponse({
            resolution_note: "The fresh evidence settled the rotation-first question.",
            growth_marker: null,
          }),
          input_tokens: 20,
          output_tokens: 20,
        },
        createDuplicateJudgmentResponse([], { input_tokens: 70, output_tokens: 1 }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [firstQuestionText, [1, 0, 0, 0]],
          [secondQuestionText, [0, 1, 0, 0]],
        ]),
      ),
      configOverrides: {
        offline: {
          ruminator: {
            maxQuestionsPerRun: 1,
            resolveConfidenceThreshold: 0,
          },
        },
      },
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
      logger: { warn },
    });

    try {
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture(
          {
            title: "Rotation-first planning evidence",
            narrative: "Fresh planning evidence settled the first question.",
            tags: ["planning"],
            significance: 0.95,
            created_at: 2_000_000,
            updated_at: 2_000_000,
          },
          [1, 0, 0, 0],
        ),
      );
      const firstQuestion = harness.openQuestionsRepository.add({
        question: firstQuestionText,
        urgency: 0.9,
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 1_000_000,
        last_touched: 1_000_000,
      });
      harness.openQuestionsRepository.add({
        question: secondQuestionText,
        urgency: 0.1,
        source: "reflection",
        provenance: { kind: "manual" },
        created_at: 1_100_000,
        last_touched: 1_100_000,
      });
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      const plan = await process.plan(harness.createContext(), { budget: 100 });

      expect(plan.budget_exhausted).toBe(true);
      expect(plan.tokens_used).toBe(111);
      expect(llm.requests.map((request) => request.tool_choice)).toEqual([
        { type: "tool", name: RUMINATOR_TOOL_NAME },
        { type: "tool", name: RUMINATOR_DUPLICATE_TOOL.name },
      ]);
      expect(plan.scheduling).toMatchObject({
        selected_question_ids: [firstQuestion.id],
        visited_question_ids: [firstQuestion.id],
        model_called_question_ids: [firstQuestion.id],
        budget_cut_question_ids: [],
      });
      expect(plan.duplicate_judgment).toEqual({
        pairs_candidates: 1,
        pairs_presented: 1,
        pairs_skipped_budget: 0,
        judgment_ran: true,
      });
      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ action: "resolve", question_id: firstQuestion.id }),
        ]),
      );
      expect(warn).toHaveBeenCalledWith(
        "Ruminator duplicate judgment failed open without backstop merges",
        expect.objectContaining({ error: expect.any(Error) }),
      );
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
        createDuplicateJudgmentResponse([], { input_tokens: 1, output_tokens: 1 }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock: new FixedClock(40 * 24 * 60 * 60 * 1_000),
      embeddingClient: new TestEmbeddingClient(
        new Map([
          ["Why does Atlas deploy fail?", [1, 0, 0, 0]],
          ["Why does Atlas deploy fail again?", [0, 1, 0, 0]],
        ]),
      ),
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
          created_at: harness.clock.now() - 1_000,
          updated_at: harness.clock.now() - 1_000,
        },
        [1, 0, 0, 0],
      );
      const secondEpisode = createEpisodeFixture(
        {
          title: "Atlas retry plan",
          narrative: "Atlas retry plan landed.",
          tags: ["atlas", "deploy"],
          created_at: harness.clock.now() - 500,
          updated_at: harness.clock.now() - 500,
        },
        [1, 0, 0, 0],
      );
      await harness.episodicRepository.createEpisode(firstEpisode);
      await harness.episodicRepository.createEpisode(secondEpisode);
      const firstQuestion = harness.openQuestionsRepository.add({
        question: "Why does Atlas deploy fail?",
        urgency: 0.7,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });
      const secondQuestion = harness.openQuestionsRepository.add({
        question: "Why does Atlas deploy fail again?",
        urgency: 0.65,
        source: "reflection",
        created_at: 1_000_000,
        provenance: { kind: "manual" },
        last_touched: 1_000_000,
      });
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      const plan = await process.plan(harness.createContext(), {
        budget: 10,
      });

      expect(plan.budget_exhausted).toBe(true);
      expect(llm.requests).toHaveLength(1);
      expect(plan.scheduling).toMatchObject({
        due_question_ids: [firstQuestion.id, secondQuestion.id],
        selected_question_ids: [firstQuestion.id, secondQuestion.id],
        visited_question_ids: [firstQuestion.id],
        model_called_question_ids: [firstQuestion.id],
        budget_cut_question_ids: [secondQuestion.id],
      });
      const result = await process.apply(harness.createContext(), plan);

      expect(harness.openQuestionsRepository.get(firstQuestion.id)).toMatchObject({
        unresolved_rumination_ticks: 0,
        last_ruminated_at: harness.clock.now(),
      });
      expect(harness.openQuestionsRepository.get(secondQuestion.id)).toMatchObject({
        unresolved_rumination_ticks: 0,
        last_ruminated_at: null,
      });
      expect(result.notes).toEqual(
        expect.arrayContaining([
          expect.stringContaining("due=2; selected=2; visited=1; model_calls=1; budget_cut=1"),
          expect.stringContaining(secondQuestion.id),
        ]),
      );
    } finally {
      await harness.cleanup();
    }
  });
});
