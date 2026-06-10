import { afterEach, describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import {
  memoryDisclosurePayloadFields,
  openQuestionMemoryDisclosureLabel,
} from "../../memory/common/disclosure-serializers.js";
import { StreamWriter } from "../../stream/index.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../../tracing/tracer.js";
import { createEntityId, DEFAULT_SESSION_ID } from "../../util/ids.js";

import { MaintenanceOrchestrator, type OfflineProcess, type OfflineProcessName } from "../index.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../test-support.js";
import { ASSOCIATOR_PROMPT, ASSOCIATOR_TOOL, AssociatorProcess } from "./index.js";

type OfflineHarness = Awaited<ReturnType<typeof createOfflineTestHarness>>;

function createAssociatorResponse(
  findings: Array<Record<string, unknown>>,
  usage: { inputTokens?: number; outputTokens?: number } = {},
) {
  return {
    text: "",
    input_tokens: usage.inputTokens ?? 18,
    output_tokens: usage.outputTokens ?? 12,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_association",
        name: ASSOCIATOR_TOOL.name,
        input: {
          findings,
        },
      },
    ],
  };
}

function createProcess(harness: OfflineHarness) {
  return new AssociatorProcess({
    semanticNodeRepository: harness.semanticNodeRepository,
    semanticEdgeRepository: harness.semanticEdgeRepository,
    reviewQueueRepository: harness.reviewQueueRepository,
    openQuestionsRepository: harness.openQuestionsRepository,
    registry: harness.registry,
    clock: harness.clock,
  });
}

function createAssociatorOrchestrator(harness: OfflineHarness, process: AssociatorProcess) {
  const context = harness.createContext();
  const {
    runId: _runId,
    auditLog: _auditLog,
    streamWriter: _streamWriter,
    ...baseContext
  } = context;

  return new MaintenanceOrchestrator({
    baseContext,
    auditLog: harness.auditLog,
    createStreamWriter: () =>
      new StreamWriter({
        dataDir: harness.tempDir,
        sessionId: DEFAULT_SESSION_ID,
        clock: harness.clock,
        entryIndex: context.entryIndex,
      }),
    processRegistry: {
      associator: process,
    } as unknown as Record<OfflineProcessName, OfflineProcess>,
  });
}

async function createEpisodes(harness: OfflineHarness, count = 3) {
  const episodes = Array.from({ length: count }, (_, index) =>
    createEpisodeFixture(
      {
        title: `Association episode ${index}`,
        narrative: `A distant life episode number ${index}.`,
        tags: [`tag-${index}`],
        created_at: 10_000 + index * 10_000,
        updated_at: 10_000 + index * 10_000,
        start_time: 10_000 + index * 10_000,
        end_time: 11_000 + index * 10_000,
        significance: index === 0 ? 0.95 : 0.25,
      },
      [index + 1, 0, 0, 0],
    ),
  );

  for (const episode of episodes) {
    await harness.episodicRepository.createEpisode(episode);
  }

  return episodes;
}

describe("associator process", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("routes strong findings through new_insight review before semantic materialization", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({ llmClient: llm });
    cleanup.push(() => harness.cleanup());
    const episodes = await createEpisodes(harness, 3);
    const evidenceAnchor = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          kind: "entity",
          label: "Early preparation",
          description: "An anchor from the first cited episode.",
          source_episode_ids: [episodes[0]!.id],
          confidence: 0.8,
        },
        [1, 0, 0, 0],
      ),
    );
    llm.pushResponse(
      createAssociatorResponse([
        {
          kind: "new_insight",
          label: "Preparation turns later ambiguity into recoverable work",
          description:
            "The cited distant episodes both show prior preparation making later uncertainty easier to recover from.",
          confidence: 0.9,
          source_episode_ids: [episodes[0]!.id, episodes[2]!.id],
        },
      ]),
    );

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), { dryRun: false });

    expect(result.errors).toEqual([]);
    expect(result.changes).toHaveLength(1);
    expect(llm.requests[0]?.budget).toBe("offline-associator");
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: ASSOCIATOR_TOOL.name,
    });
    expect(String(llm.requests[0]?.messages[0]?.content)).toContain(ASSOCIATOR_PROMPT);

    const nodesBeforeReview = await harness.semanticNodeRepository.list({
      includeArchived: true,
      limit: 10,
    });
    expect(
      nodesBeforeReview.some(
        (node) => node.label === "Preparation turns later ambiguity into recoverable work",
      ),
    ).toBe(false);

    const openReview = harness.reviewQueueRepository.getOpen()[0];
    expect(openReview).toEqual(
      expect.objectContaining({
        kind: "new_insight",
        refs: expect.objectContaining({
          episode_ids: [episodes[0]!.id, episodes[2]!.id],
          evidence_cluster_size: 2,
          reflector_pending_insight: expect.objectContaining({
            candidate_support_edges: [
              expect.objectContaining({
                target_node_id: evidenceAnchor.id,
                source_episode_ids: [episodes[0]!.id],
              }),
            ],
            evidence_cluster: expect.objectContaining({
              episode_ids: [episodes[0]!.id, episodes[2]!.id],
              size: 2,
            }),
          }),
        }),
      }),
    );

    await harness.reviewQueueRepository.resolve(openReview!.id, "accept");

    const nodes = await harness.semanticNodeRepository.list({
      includeArchived: true,
      limit: 10,
    });
    const insightNode = nodes.find(
      (node) => node.label === "Preparation turns later ambiguity into recoverable work",
    );

    expect(insightNode).toMatchObject({
      confidence: 0.5,
      source_episode_ids: [episodes[0]!.id, episodes[2]!.id],
    });
    expect(harness.semanticEdgeRepository.listEdges({ relation: "supports" })).toEqual([
      expect.objectContaining({
        from_node_id: evidenceAnchor.id,
        to_node_id: insightNode?.id,
        evidence_episode_ids: [episodes[0]!.id],
      }),
    ]);
  });

  it("always enqueues strong findings as insert-mode pending insights", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({ llmClient: llm });
    cleanup.push(() => harness.cleanup());
    const episodes = await createEpisodes(harness, 2);
    const label = "Repeated structure should still be reviewed downstream";
    const existing = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          kind: "proposition",
          label,
          description: "Existing semantic memory with the same label.",
          source_episode_ids: [episodes[0]!.id],
          confidence: 0.8,
        },
        [1, 0, 0, 0],
      ),
    );
    llm.pushResponse(
      createAssociatorResponse([
        {
          kind: "new_insight",
          label,
          description: "A new associative candidate with a matching label.",
          confidence: 0.4,
          source_episode_ids: [episodes[1]!.id],
        },
      ]),
    );

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());
    const finding = plan.samples[0]?.findings[0];

    expect(finding).toMatchObject({
      kind: "new_insight",
      target: {
        mode: "insert",
        node: expect.objectContaining({
          label,
          source_episode_ids: [episodes[1]!.id],
        }),
      },
    });

    await process.apply(harness.createContext(), plan);
    const openReview = harness.reviewQueueRepository.getOpen()[0];

    expect(openReview?.refs.reflector_pending_insight).toMatchObject({
      target: {
        mode: "insert",
        node: expect.objectContaining({
          label,
        }),
      },
    });
    expect(await harness.semanticNodeRepository.get(existing.id)).toMatchObject({
      id: existing.id,
      source_episode_ids: [episodes[0]!.id],
    });
  });

  it("creates open questions and reinforces exact normalized duplicates", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({ llmClient: llm });
    cleanup.push(() => harness.cleanup());
    const episodes = await createEpisodes(harness, 2);
    const process = createProcess(harness);

    llm.pushResponse(
      createAssociatorResponse([
        {
          kind: "open_question",
          question: "Is there a real connection between rehearsal and recovery?",
          urgency: 0.42,
          source_episode_ids: [episodes[0]!.id, episodes[1]!.id],
        },
      ]),
    );
    await process.run(harness.createContext(), { dryRun: false });

    const createdQuestions = harness.openQuestionsRepository.list({
      source: "associator",
      status: "open",
    });
    expect(createdQuestions).toHaveLength(1);
    expect(createdQuestions[0]).toMatchObject({
      question: "Is there a real connection between rehearsal and recovery?",
      urgency: 0.42,
      related_episode_ids: [episodes[0]!.id, episodes[1]!.id],
      provenance: {
        kind: "offline",
        process: "associator",
      },
    });

    llm.pushResponse(
      createAssociatorResponse([
        {
          kind: "open_question",
          question: "  is there a real connection between rehearsal and recovery?  ",
          urgency: 0.8,
          source_episode_ids: [episodes[0]!.id],
        },
      ]),
    );
    await process.run(harness.createContext(), { dryRun: false });

    const reinforcedQuestions = harness.openQuestionsRepository.list({
      source: "associator",
      status: "open",
    });
    expect(reinforcedQuestions).toHaveLength(1);
    expect(reinforcedQuestions[0]?.urgency).toBeCloseTo(0.44);
  });

  it("rejects out-of-sample episode ids and surfaces them through generic offline traces", async () => {
    const traceEvents: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: true,
      emit: (event, data) => {
        traceEvents.push({ event, data });
      },
    };
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({ llmClient: llm, tracer });
    cleanup.push(() => harness.cleanup());
    const episodes = await createEpisodes(harness, 2);
    llm.pushResponse(
      createAssociatorResponse([
        {
          kind: "open_question",
          question: "Does this invalid citation get rejected?",
          urgency: 0.2,
          source_episode_ids: [episodes[0]!.id, "ep_outsideoutside"],
        },
      ]),
    );
    const process = createProcess(harness);
    const orchestrator = createAssociatorOrchestrator(harness, process);

    const plan = await orchestrator.plan({ processes: [process] });
    const result = await orchestrator.apply(plan);

    expect(result.errors).toEqual([
      expect.objectContaining({
        process: "associator",
        code: "ASSOCIATOR_INVALID_REF",
      }),
    ]);
    expect(result.changes).toEqual([]);
    expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([]);
    expect(traceEvents).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          event: "offline_process.completed",
          data: expect.objectContaining({
            process_name: "associator",
            phase: "apply",
            errors: 1,
            error_details: [
              expect.objectContaining({
                code: "ASSOCIATOR_INVALID_REF",
              }),
            ],
          }),
        }),
      ]),
    );
  });

  it("reports finding-cap truncation in candidate stats and generic traces", async () => {
    const traceEvents: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: true,
      emit: (event, data) => {
        traceEvents.push({ event, data });
      },
    };
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      configOverrides: {
        offline: {
          associator: {
            maxFindingsPerRun: 1,
          },
        },
      },
    });
    cleanup.push(() => harness.cleanup());
    const episodes = await createEpisodes(harness, 2);
    llm.pushResponse(
      createAssociatorResponse([
        {
          kind: "open_question",
          question: "Is the first associative question retained?",
          urgency: 0.2,
          source_episode_ids: [episodes[0]!.id],
        },
        {
          kind: "open_question",
          question: "Is the second associative question truncated?",
          urgency: 0.2,
          source_episode_ids: [episodes[1]!.id],
        },
        {
          kind: "open_question",
          question: "Is the third associative question truncated?",
          urgency: 0.2,
          source_episode_ids: [episodes[0]!.id, episodes[1]!.id],
        },
      ]),
    );
    const process = createProcess(harness);
    const orchestrator = createAssociatorOrchestrator(harness, process);
    const plan = await orchestrator.plan({ processes: [process] });
    const result = await orchestrator.apply(plan);
    const associatorResult = result.results[0];

    expect(associatorResult?.candidate_stats).toEqual({
      proposed: 3,
      accepted: 1,
      rejected: 2,
      truncated: 2,
    });
    expect(associatorResult?.run_capped).toBe(true);
    expect(harness.openQuestionsRepository.list({ source: "associator" })).toHaveLength(1);
    expect(traceEvents).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          event: "offline_process.completed",
          data: expect.objectContaining({
            process_name: "associator",
            phase: "apply",
            candidates_proposed: 3,
            candidates_accepted: 1,
            candidates_rejected: 2,
            candidates_truncated: 2,
            notes: ["candidate_cap_truncated:2"],
          }),
        }),
      ]),
    );
  });

  it("skips findings whose planned episode refs are archived before apply", async () => {
    const traceEvents: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: true,
      emit: (event, data) => {
        traceEvents.push({ event, data });
      },
    };
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({ llmClient: llm, tracer });
    cleanup.push(() => harness.cleanup());
    const episodes = await createEpisodes(harness, 2);
    llm.pushResponse(
      createAssociatorResponse([
        {
          kind: "open_question",
          question: "Does an archived planned episode block this finding?",
          urgency: 0.3,
          source_episode_ids: [episodes[0]!.id],
        },
      ]),
    );
    const process = createProcess(harness);
    const ctx = harness.createContext();
    const plan = await process.plan(ctx);

    harness.episodicRepository.archiveEpisode(episodes[0]!.id, {
      caller: "associator.test",
      reason: "archive planned episode between plan and apply",
      process: "consolidator",
    });

    const result = await process.apply(ctx, plan);

    expect(result.changes).toEqual([]);
    expect(result.candidate_stats).toEqual({
      proposed: 1,
      accepted: 0,
      rejected: 1,
    });
    expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([]);
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
    expect(traceEvents).toContainEqual({
      event: "semantic_insert.skipped",
      data: expect.objectContaining({
        kind: "episode",
        reason: "episode_archived_post_plan",
      }),
    });
  });

  it("combines disclosure labels from cited mixed-label episodes", async () => {
    const sam = createEntityId();
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({ llmClient: llm });
    cleanup.push(() => harness.cleanup());
    const publicEpisode = createEpisodeFixture({
      title: "Public association source",
      narrative: "Public evidence.",
      shared: true,
      origin_audience_entity_ids: [],
      created_at: 10_000,
      updated_at: 10_000,
      start_time: 10_000,
      end_time: 11_000,
    });
    const scopedEpisode = createEpisodeFixture({
      title: "Scoped association source",
      narrative: "Relationship-private evidence.",
      audience_entity_id: sam,
      shared: false,
      created_at: 20_000,
      updated_at: 20_000,
      start_time: 20_000,
      end_time: 21_000,
    });

    await harness.episodicRepository.createEpisode(publicEpisode);
    await harness.episodicRepository.createEpisode(scopedEpisode);
    llm.pushResponse(
      createAssociatorResponse([
        {
          kind: "open_question",
          question: "Is public evidence connected with scoped evidence?",
          urgency: 0.3,
          source_episode_ids: [publicEpisode.id, scopedEpisode.id],
        },
      ]),
    );

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());
    const finding = plan.samples[0]?.findings[0];

    expect(finding?.source_disclosure_label).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [sam],
      privateToEntityIds: [sam],
      publicToEntityIds: [],
    });

    await process.apply(harness.createContext(), plan);
    const [question] = harness.openQuestionsRepository.list({
      source: "associator",
      status: "open",
    });

    expect(question?.disclosure_label).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [sam],
      privateToEntityIds: [sam],
      publicToEntityIds: [],
    });
    expect(
      memoryDisclosurePayloadFields(openQuestionMemoryDisclosureLabel(question!)).disclosure_label,
    ).toEqual({
      disclosure_class: "relationship_private",
      origin_audience_entity_ids: [sam],
      private_to_entity_ids: [sam],
      public_to_entity_ids: [],
    });
  });

  it("reports budget exhaustion as a process result", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createAssociatorResponse(
          [
            {
              kind: "open_question",
              question: "Will this fit the budget?",
              urgency: 0.2,
              source_episode_ids: [],
            },
          ],
          {
            inputTokens: 100,
            outputTokens: 100,
          },
        ),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      configOverrides: {
        offline: {
          associator: {
            budget: 10,
          },
        },
      },
    });
    cleanup.push(() => harness.cleanup());
    await createEpisodes(harness, 2);

    const result = await createProcess(harness).run(harness.createContext(), { dryRun: false });

    expect(result.budget_exhausted).toBe(true);
    expect(result.tokens_used).toBe(200);
    expect(result.errors).toEqual([
      expect.objectContaining({
        code: "OFFLINE_BUDGET_EXCEEDED",
      }),
    ]);
    expect(result.changes).toEqual([]);
  });

  it("persists sampled episode ids in the plan and apply never resamples", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({ llmClient: llm });
    cleanup.push(() => harness.cleanup());
    const episodes = await createEpisodes(harness, 3);
    llm.pushResponse(
      createAssociatorResponse([
        {
          kind: "new_insight",
          label: "Stable sampling preserves cited evidence",
          description: "The finding should keep using the planned cited episodes.",
          confidence: 0.4,
          source_episode_ids: [episodes[0]!.id, episodes[1]!.id],
        },
      ]),
    );
    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());
    const plannedSampleIds = plan.samples[0]?.episode_ids;

    const lateEpisode = await harness.episodicRepository.createEpisode(
      createEpisodeFixture({
        title: "Late extra episode",
        narrative: "This episode arrived after planning.",
        created_at: 99_000,
        updated_at: 99_000,
        start_time: 99_000,
        end_time: 100_000,
      }),
    );

    await process.apply(harness.createContext(), plan);
    const openReview = harness.reviewQueueRepository.getOpen()[0];

    expect(llm.requests).toHaveLength(1);
    expect(plan.samples[0]?.seed).toEqual(expect.any(String));
    expect(plannedSampleIds).toEqual(expect.arrayContaining([episodes[0]!.id, episodes[1]!.id]));
    expect(openReview?.refs.episode_ids).toEqual([episodes[0]!.id, episodes[1]!.id]);
    expect(openReview?.refs.episode_ids).not.toContain(lateEpisode.id);
  });

  it("treats empty findings as a clean no-op result", async () => {
    const llm = new FakeLLMClient({
      responses: [createAssociatorResponse([])],
    });
    const harness = await createOfflineTestHarness({ llmClient: llm });
    cleanup.push(() => harness.cleanup());
    await createEpisodes(harness, 2);

    const result = await createProcess(harness).run(harness.createContext(), { dryRun: false });

    expect(result).toMatchObject({
      process: "associator",
      dryRun: false,
      changes: [],
      errors: [],
      budget_exhausted: false,
      candidate_stats: {
        proposed: 0,
        accepted: 0,
        rejected: 0,
      },
    });
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
    expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([]);
  });
});
