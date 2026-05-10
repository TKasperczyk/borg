import { afterEach, describe, expect, it, vi } from "vitest";

import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../../cognition/index.js";
import { FakeLLMClient } from "../../llm/index.js";
import { StreamWriter } from "../../stream/index.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticEdgeFixture,
  createSemanticNodeFixture,
} from "../test-support.js";
import { MaintenanceOrchestrator } from "../orchestrator.js";
import { MaintenanceScheduler } from "../scheduler.js";
import type {
  OfflineContext,
  OfflineProcess,
  OfflineProcessName,
  OfflineProcessPlan,
  OfflineResult,
} from "../types.js";
import { SemanticExtractorProcess } from "./index.js";

const SEMANTIC_TOOL_NAME = "EmitSemanticCandidates";

function createSemanticToolResponse(input: { nodes: unknown[]; edges: unknown[] }) {
  return {
    text: "",
    input_tokens: 10,
    output_tokens: 5,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: SEMANTIC_TOOL_NAME,
        input,
      },
    ],
  };
}

class CaptureTracer implements TurnTracer {
  readonly enabled = true;
  readonly includePayloads = false;
  readonly events: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    this.events.push({ event, data });
  }
}

function createProcess(harness: Awaited<ReturnType<typeof createOfflineTestHarness>>) {
  return new SemanticExtractorProcess({
    semanticNodeRepository: harness.semanticNodeRepository,
    semanticEdgeRepository: harness.semanticEdgeRepository,
    registry: harness.registry,
    clock: harness.clock,
  });
}

function emptyResult(name: OfflineProcessName): OfflineResult {
  return {
    process: name,
    dryRun: false,
    changes: [],
    tokens_used: 0,
    errors: [],
    budget_exhausted: false,
  };
}

function emptyPlan(name: OfflineProcessName): OfflineProcessPlan {
  return {
    process: name,
    items: [],
    errors: [],
    tokens_used: 0,
    budget_exhausted: false,
  } as unknown as OfflineProcessPlan;
}

function fakeProcess(
  name: OfflineProcessName,
  onApply: (ctx: OfflineContext) => Promise<void> | void = () => undefined,
): OfflineProcess {
  return {
    name,
    plan: async () => emptyPlan(name),
    preview: () => emptyResult(name),
    apply: async (ctx) => {
      await onApply(ctx);
      return emptyResult(name);
    },
    run: async (ctx) => {
      await onApply(ctx);
      return emptyResult(name);
    },
  };
}

function createProcessRegistry(
  overrides: Partial<Record<OfflineProcessName, OfflineProcess>>,
): Record<OfflineProcessName, OfflineProcess> {
  const names: OfflineProcessName[] = [
    "consolidator",
    "reflector",
    "semantic-extractor",
    "curator",
    "overseer",
    "ruminator",
    "self-narrator",
    "procedural-synthesizer",
    "belief-reviser",
  ];

  return Object.fromEntries(
    names.map((name) => [name, overrides[name] ?? fakeProcess(name)]),
  ) as Record<OfflineProcessName, OfflineProcess>;
}

function baseContextFrom(ctx: OfflineContext) {
  const { runId: _runId, auditLog: _auditLog, streamWriter: _streamWriter, ...baseContext } = ctx;

  return baseContext;
}

describe("semantic extractor process", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("extracts unrepresented episodes through maintenance apply and records trace/audit", async () => {
    const tracer = new CaptureTracer();
    const episode = createEpisodeFixture({
      title: "Anarres and Urras comparison",
      narrative: "The conversation compared Anarres and Urras as paired social concepts.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "concept",
              label: "Anarres-Urras contrast",
              description: "A conceptual contrast between two social orders.",
              domain: "literature",
              aliases: [],
              confidence: 0.64,
              source_episode_ids: [episode.id],
            },
          ],
          edges: [],
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
    });
    cleanup.push(harness.cleanup);
    await harness.episodicRepository.insert(episode);
    const process = createProcess(harness);
    const queueDuplicateReview = vi.fn();
    const ctx = {
      ...harness.createContext(),
      semanticReviewService: {
        queueDuplicateReview,
        reviewDuplicateCandidate: vi.fn(),
      } as unknown as OfflineContext["semanticReviewService"],
    };

    const plan = await process.plan(ctx);
    const result = await process.apply(ctx, plan);

    expect(plan.episode_ids).toEqual([episode.id]);
    expect(result.candidate_stats).toEqual({
      proposed: 1,
      accepted: 1,
      rejected: 0,
    });
    await expect(harness.semanticNodeRepository.list()).resolves.toEqual([
      expect.objectContaining({
        label: "Anarres-Urras contrast",
        source_episode_ids: [episode.id],
      }),
    ]);
    expect(harness.auditLog.list({ process: "semantic-extractor" })).toHaveLength(1);
    expect(queueDuplicateReview).toHaveBeenCalledWith(
      expect.objectContaining({
        label: "Anarres-Urras contrast",
      }),
      {
        sourceProcess: "semantic-extractor",
        traceTurnId: ctx.runId,
      },
    );
    expect(tracer.events).toContainEqual({
      event: "semantic_extractor_invoked",
      data: expect.objectContaining({
        turnId: ctx.runId,
        input_episode_count: 1,
        parsed_node_count: 1,
        accepted_node_count: 1,
      }),
    });
  });

  it("does not replan episodes after a zero-candidate extraction audit", async () => {
    const episode = createEpisodeFixture({
      title: "Sparse semantic content",
      narrative: "A short exchange that produces no durable semantic candidates.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [],
          edges: [],
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    await harness.episodicRepository.insert(episode);
    const process = createProcess(harness);
    const ctx = harness.createContext();

    const firstPlan = await process.plan(ctx);
    const result = await process.apply(ctx, firstPlan);
    const secondPlan = await process.plan(ctx);

    expect(firstPlan.episode_ids).toEqual([episode.id]);
    expect(result.candidate_stats).toEqual({
      proposed: 0,
      accepted: 0,
      rejected: 0,
    });
    expect(secondPlan.episode_ids).toEqual([]);
  });

  it("skips invalid edges while keeping valid batch candidates", async () => {
    const tracer = new CaptureTracer();
    const episode = createEpisodeFixture({
      title: "Partial extraction failure",
      narrative: "The episode mentions a support relation before an invalid edge candidate.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "concept",
              label: "First concept",
              description: "The first extracted concept.",
              domain: null,
              aliases: [],
              confidence: 0.62,
              source_episode_ids: [episode.id],
            },
            {
              kind: "concept",
              label: "Second concept",
              description: "The second extracted concept.",
              domain: null,
              aliases: [],
              confidence: 0.61,
              source_episode_ids: [episode.id],
            },
          ],
          edges: [
            {
              from_label: "First concept",
              to_label: "Second concept",
              relation: "supports",
              confidence: 0.6,
              evidence_episode_ids: [episode.id],
              valid_from_ts: null,
              valid_to_ts: null,
            },
            {
              from_label: "First concept",
              to_label: "Missing concept",
              relation: "supports",
              confidence: 0.6,
              evidence_episode_ids: [episode.id],
              valid_from_ts: null,
              valid_to_ts: null,
            },
          ],
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
    });
    cleanup.push(harness.cleanup);
    await harness.episodicRepository.insert(episode);
    const process = createProcess(harness);
    const queueDuplicateReview = vi.fn();
    const ctx = {
      ...harness.createContext(),
      semanticReviewService: {
        queueDuplicateReview,
        reviewDuplicateCandidate: vi.fn(),
      } as unknown as OfflineContext["semanticReviewService"],
    };

    const plan = await process.plan(ctx);
    const result = await process.apply(ctx, plan);
    const retryPlan = await process.plan(harness.createContext());

    expect(result.errors).toEqual([]);
    expect(result.candidate_stats).toEqual({
      proposed: 4,
      accepted: 3,
      rejected: 1,
    });
    await expect(harness.semanticNodeRepository.list({ includeArchived: true })).resolves.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "First concept" }),
        expect.objectContaining({ label: "Second concept" }),
      ]),
    );
    expect(harness.semanticEdgeRepository.listEdges({ includeInvalid: true })).toHaveLength(1);
    expect(harness.auditLog.list({ process: "semantic-extractor" })).toHaveLength(1);
    expect(queueDuplicateReview).toHaveBeenCalledTimes(2);
    expect(retryPlan.episode_ids).toEqual([]);
    expect(tracer.events).toContainEqual({
      event: "semantic_extractor_partial_failure",
      data: expect.objectContaining({
        turnId: ctx.runId,
        skipped_edge_count: 1,
        skip_reasons: expect.arrayContaining(["invalid_endpoint"]),
        skipped_edge_details: [
          expect.objectContaining({
            candidate_index: 1,
            from_label: "First concept",
            to_label: "Missing concept",
            relation: "supports",
            evidence_ids: [episode.id],
            reason: "invalid_endpoint",
          }),
        ],
      }),
    });
  });

  it("restores merged edge evidence when a later batch step fails", async () => {
    const priorEpisode = createEpisodeFixture({
      title: "Prior edge evidence",
      narrative: "Earlier evidence supported the existing semantic edge.",
    });
    const episode = createEpisodeFixture({
      title: "Fresh edge evidence",
      narrative: "Fresh evidence reasserts the same semantic edge before review enqueue fails.",
    });
    const fromNode = createSemanticNodeFixture({
      label: "Atlas rollout",
      description: "The Atlas rollout is tracked as a release concern.",
      source_episode_ids: [priorEpisode.id],
    });
    const toNode = createSemanticNodeFixture({
      label: "Rollback planning",
      description: "Rollback planning is tracked as a release practice.",
      source_episode_ids: [priorEpisode.id],
    });
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "concept",
              label: "Review queue trigger",
              description: "A new concept that triggers duplicate review after extraction.",
              domain: null,
              aliases: [],
              confidence: 0.61,
              source_episode_ids: [episode.id],
            },
          ],
          edges: [
            {
              from_label: fromNode.label,
              to_label: toNode.label,
              relation: "supports",
              confidence: 0.89,
              evidence_episode_ids: [episode.id],
              valid_from_ts: null,
              valid_to_ts: null,
            },
          ],
        }),
      ],
    });
    const harness = await createOfflineTestHarness({ llmClient: llm });
    cleanup.push(harness.cleanup);
    await harness.episodicRepository.insert(priorEpisode);
    await harness.episodicRepository.insert(episode);
    await harness.semanticNodeRepository.insert(fromNode);
    await harness.semanticNodeRepository.insert(toNode);
    const originalEdge = harness.semanticEdgeRepository.addEdge(
      createSemanticEdgeFixture({
        from_node_id: fromNode.id,
        to_node_id: toNode.id,
        relation: "supports",
        confidence: 0.42,
        evidence_episode_ids: [priorEpisode.id],
        created_at: 900_000,
        last_verified_at: 900_000,
        valid_from: 900_000,
      }),
    );
    const process = createProcess(harness);
    const ctx = {
      ...harness.createContext(),
      semanticReviewService: {
        queueDuplicateReview: vi.fn(() => {
          throw new Error("review enqueue failed");
        }),
        reviewDuplicateCandidate: vi.fn(),
      } as unknown as OfflineContext["semanticReviewService"],
    };

    const plan = await process.plan(ctx);
    const result = await process.apply(ctx, plan);

    expect(result.changes).toEqual([]);
    expect(result.errors).toHaveLength(1);
    expect(harness.semanticEdgeRepository.getEdge(originalEdge.id)).toEqual(originalEdge);
    await expect(
      harness.semanticNodeRepository.list({ includeArchived: true }),
    ).resolves.not.toEqual(
      expect.arrayContaining([expect.objectContaining({ label: "Review queue trigger" })]),
    );
  });

  it("traces node dedupe when extraction updates an existing compatible node", async () => {
    const tracer = new CaptureTracer();
    const priorEpisode = createEpisodeFixture({
      title: "Prior Minds note",
      narrative: "A prior public episode established the Minds-as-kind concept.",
    });
    const episode = createEpisodeFixture({
      title: "Minds follow-up",
      narrative: "The conversation returns to Minds-as-kind with fresh wording.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "concept",
              label: "Minds-as-kind",
              description: "Minds are discussed as a kind rather than a single agent.",
              domain: "philosophy",
              aliases: [],
              confidence: 0.66,
              source_episode_ids: [episode.id],
            },
          ],
          edges: [],
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
    });
    cleanup.push(harness.cleanup);
    await harness.episodicRepository.insert(priorEpisode);
    await harness.episodicRepository.insert(episode);
    await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        kind: "concept",
        label: "Minds-as-kind",
        description: "Minds are a durable conceptual kind.",
        source_episode_ids: [priorEpisode.id],
      }),
    );
    const process = createProcess(harness);
    const ctx = harness.createContext();

    const plan = await process.plan(ctx);
    const result = await process.apply(ctx, plan);

    expect(plan.episode_ids).toEqual([episode.id]);
    expect(result.candidate_stats).toEqual({
      proposed: 1,
      accepted: 1,
      rejected: 0,
    });
    expect(tracer.events).toContainEqual({
      event: "semantic_insert_skipped",
      data: expect.objectContaining({
        turnId: ctx.runId,
        kind: "node",
        reason: "dedupe_match",
      }),
    });
  });

  it("light maintenance skips episodes archived by consolidator after planning", async () => {
    const tracer = new CaptureTracer();
    const episode = createEpisodeFixture({
      title: "Consolidated before extraction",
      narrative: "This episode is visible at planning time and archived before semantic apply.",
    });
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
    });
    cleanup.push(harness.cleanup);
    await harness.episodicRepository.insert(episode);
    const semanticProcess = createProcess(harness);
    const processRegistry = createProcessRegistry({
      consolidator: fakeProcess("consolidator", (ctx) => {
        ctx.episodicRepository.updateStats(episode.id, {
          archived: true,
        });
      }),
      "semantic-extractor": semanticProcess,
      curator: fakeProcess("curator"),
    });
    const orchestratorContext = harness.createContext();
    const orchestrator = new MaintenanceOrchestrator({
      baseContext: baseContextFrom(orchestratorContext),
      auditLog: harness.auditLog,
      createStreamWriter: () =>
        new StreamWriter({
          dataDir: harness.tempDir,
          clock: harness.clock,
        }),
      processRegistry,
    });
    const scheduler = new MaintenanceScheduler({
      enabled: true,
      lightIntervalMs: 1,
      heavyIntervalMs: 1,
      lightProcesses: ["consolidator", "semantic-extractor", "curator"],
      heavyProcesses: [],
      orchestrator,
      processRegistry,
      clock: harness.clock,
    });

    const tick = await scheduler.tick("light");

    expect(tick.status).toBe("ok");
    expect(tick.processes).toEqual(["consolidator", "semantic-extractor", "curator"]);
    expect(tick.result?.results.map((result) => result.process)).toEqual([
      "consolidator",
      "semantic-extractor",
      "curator",
    ]);
    expect(tick.result?.results[1]?.candidate_stats).toEqual({
      proposed: 1,
      accepted: 0,
      rejected: 1,
    });
    expect(llm.requests).toEqual([]);
    expect(tracer.events).toContainEqual({
      event: "semantic_insert_skipped",
      data: expect.objectContaining({
        kind: "episode",
        reason: "episode_archived_post_plan",
      }),
    });
  });
});
