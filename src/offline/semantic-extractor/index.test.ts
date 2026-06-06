import { afterEach, describe, expect, it, vi } from "vitest";

import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../../cognition/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { StreamWriter } from "../../stream/index.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticEdgeFixture,
  createSemanticNodeFixture,
  TestEmbeddingClient,
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

function semanticNodeEmbeddingText(input: { label: string; description: string }): string {
  return [input.label, input.description, ""].join("\n");
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
    "review-resolver",
    "ruminator",
    "self-narrator",
    "procedural-synthesizer",
    "belief-reviser",
    "creator-directive-reconciler",
    "commitment-reconciler",
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
    const selfEntityId = harness.entityRepository.resolve("self", {
      kind: "self",
      provenance: "assistant_seeded",
    });
    await harness.episodicRepository.createEpisode(episode);
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
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    expect(prompt).toContain(`- self (id: ${selfEntityId}`);
    expect(prompt).toContain(
      `Entity ${selfEntityId} is yourself; refer to all entities by name, including yourself.`,
    );
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
      event: "semantic_extractor.started",
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
    await harness.episodicRepository.createEpisode(episode);
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

  it("reports capped semantic extraction backlog after selected episodes", async () => {
    const processedEpisode = createEpisodeFixture({ title: "Already represented" });
    const archivedEpisode = createEpisodeFixture({ title: "Archived episode" });
    const candidateEpisodes = [
      createEpisodeFixture({
        title: "Candidate oldest",
        created_at: 1_000,
        updated_at: 1_000,
      }),
      createEpisodeFixture({
        title: "Candidate middle",
        created_at: 2_000,
        updated_at: 2_000,
      }),
      createEpisodeFixture({
        title: "Candidate newest",
        created_at: 3_000,
        updated_at: 3_000,
      }),
    ];
    const harness = await createOfflineTestHarness({
      configOverrides: {
        offline: {
          semanticExtractor: {
            maxEpisodesPerRun: 2,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    for (const episode of [processedEpisode, archivedEpisode, ...candidateEpisodes]) {
      await harness.episodicRepository.createEpisode(episode);
    }

    harness.episodicRepository.updateStats(archivedEpisode.id, { archived: true });
    await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        source_episode_ids: [processedEpisode.id],
      }),
    );

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());
    const result = process.preview(plan);

    expect(plan.episode_ids).toEqual(candidateEpisodes.slice(0, 2).map((episode) => episode.id));
    expect(plan.episode_ids).not.toContain(processedEpisode.id);
    expect(plan.episode_ids).not.toContain(archivedEpisode.id);
    expect(plan.pending_episode_count).toBe(1);
    expect(plan.run_capped).toBe(true);
    expect(result.pending_episode_count).toBe(1);
    expect(result.run_capped).toBe(true);
  });

  it("drains unprocessed episodes oldest-first across repeated extraction runs", async () => {
    const episodes = Array.from({ length: 5 }, (_, index) =>
      createEpisodeFixture({
        title: `Backlog candidate ${index + 1}`,
        narrative: `Backlog episode ${index + 1} has sparse durable content.`,
        created_at: (index + 1) * 1_000,
        updated_at: (index + 1) * 1_000,
      }),
    );
    const llm = new FakeLLMClient({
      responses: Array.from({ length: 3 }, () =>
        createSemanticToolResponse({
          nodes: [],
          edges: [],
        }),
      ),
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      configOverrides: {
        offline: {
          semanticExtractor: {
            maxEpisodesPerRun: 2,
            maxInputTokensPerRun: 10_000,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    for (const episode of episodes) {
      await harness.episodicRepository.createEpisode(episode);
    }

    const process = createProcess(harness);
    const ctx = harness.createContext();
    const firstPlan = await process.plan(ctx);
    await process.apply(ctx, firstPlan);
    const secondPlan = await process.plan(ctx);
    await process.apply(ctx, secondPlan);
    const thirdPlan = await process.plan(ctx);
    await process.apply(ctx, thirdPlan);
    const fourthPlan = await process.plan(ctx);

    expect([...firstPlan.episode_ids, ...secondPlan.episode_ids, ...thirdPlan.episode_ids]).toEqual(
      episodes.map((episode) => episode.id),
    );
    expect(firstPlan.episode_ids).toEqual(episodes.slice(0, 2).map((episode) => episode.id));
    expect(secondPlan.episode_ids).toEqual(episodes.slice(2, 4).map((episode) => episode.id));
    expect(thirdPlan.episode_ids).toEqual(episodes.slice(4).map((episode) => episode.id));
    expect(fourthPlan.episode_ids).toEqual([]);
  });

  it("caps extraction selection by estimated input tokens and still selects one oversized episode", async () => {
    const tokenBoundedHarness = await createOfflineTestHarness({
      configOverrides: {
        offline: {
          semanticExtractor: {
            maxEpisodesPerRun: 8,
            maxInputTokensPerRun: 540,
          },
        },
      },
    });
    cleanup.push(tokenBoundedHarness.cleanup);
    const tokenBoundedEpisodes = Array.from({ length: 4 }, (_, index) =>
      createEpisodeFixture({
        title: `Token bounded ${index + 1}`,
        narrative: "x".repeat(40),
        created_at: (index + 1) * 1_000,
        updated_at: (index + 1) * 1_000,
      }),
    );

    for (const episode of tokenBoundedEpisodes) {
      await tokenBoundedHarness.episodicRepository.createEpisode(episode);
    }

    const tokenBoundedProcess = createProcess(tokenBoundedHarness);
    const tokenBoundedPlan = await tokenBoundedProcess.plan(tokenBoundedHarness.createContext());

    expect(tokenBoundedPlan.episode_ids).toEqual(
      tokenBoundedEpisodes.slice(0, 2).map((episode) => episode.id),
    );
    expect(tokenBoundedPlan.pending_episode_count).toBe(2);
    expect(tokenBoundedPlan.run_capped).toBe(true);

    const oversizedHarness = await createOfflineTestHarness({
      configOverrides: {
        offline: {
          semanticExtractor: {
            maxEpisodesPerRun: 8,
            maxInputTokensPerRun: 100,
          },
        },
      },
    });
    cleanup.push(oversizedHarness.cleanup);
    const oversizedEpisode = createEpisodeFixture({
      title: "Oversized episode",
      narrative: "x".repeat(2_000),
      created_at: 1_000,
      updated_at: 1_000,
    });
    const followingEpisode = createEpisodeFixture({
      title: "Following episode",
      narrative: "x".repeat(40),
      created_at: 2_000,
      updated_at: 2_000,
    });

    await oversizedHarness.episodicRepository.createEpisode(oversizedEpisode);
    await oversizedHarness.episodicRepository.createEpisode(followingEpisode);

    const oversizedProcess = createProcess(oversizedHarness);
    const oversizedPlan = await oversizedProcess.plan(oversizedHarness.createContext());

    expect(oversizedPlan.episode_ids).toEqual([oversizedEpisode.id]);
    expect(oversizedPlan.pending_episode_count).toBe(1);
    expect(oversizedPlan.run_capped).toBe(true);
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
    await harness.episodicRepository.createEpisode(episode);
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
      event: "semantic_extractor.degraded",
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
    await harness.episodicRepository.createEpisode(priorEpisode);
    await harness.episodicRepository.createEpisode(episode);
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
    const label = "Minds-as-kind";
    const candidateDescription = "Minds are discussed as a kind rather than a single agent.";
    const compatibleVector = [0, 0, 1, 0];
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
              label,
              description: candidateDescription,
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
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [
            semanticNodeEmbeddingText({
              label,
              description: candidateDescription,
            }),
            compatibleVector,
          ],
        ]),
      ),
      tracer,
    });
    cleanup.push(harness.cleanup);
    await harness.episodicRepository.createEpisode(priorEpisode);
    await harness.episodicRepository.createEpisode(episode);
    await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          kind: "concept",
          label,
          description: "Minds are a durable conceptual kind.",
          source_episode_ids: [priorEpisode.id],
        },
        compatibleVector,
      ),
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
      event: "semantic_insert.skipped",
      data: expect.objectContaining({
        turnId: ctx.runId,
        kind: "node",
        reason: "dedupe_match",
      }),
    });
  });

  it("keeps a same-label concept separate when embedding compatibility is low", async () => {
    const tracer = new CaptureTracer();
    const label = "Minds-as-kind";
    const candidateDescription = "Minds-as-kind names a separate taxonomy label in this context.";
    const priorEpisode = createEpisodeFixture({
      title: "Prior Minds note",
      narrative: "A prior public episode established one Minds-as-kind concept.",
    });
    const episode = createEpisodeFixture({
      title: "Distinct Minds follow-up",
      narrative: "The conversation uses the same label for a distinct conceptual taxonomy.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "concept",
              label,
              description: candidateDescription,
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
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [
            semanticNodeEmbeddingText({
              label,
              description: candidateDescription,
            }),
            [1, 0, 0, 0],
          ],
        ]),
      ),
      tracer,
    });
    cleanup.push(harness.cleanup);
    await harness.episodicRepository.createEpisode(priorEpisode);
    await harness.episodicRepository.createEpisode(episode);
    const existing = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          kind: "concept",
          label,
          description: "Minds are a durable conceptual kind.",
          source_episode_ids: [priorEpisode.id],
        },
        [0, 0, 1, 0],
      ),
    );
    const process = createProcess(harness);
    const ctx = harness.createContext();

    const plan = await process.plan(ctx);
    const result = await process.apply(ctx, plan);
    const nodes = await harness.semanticNodeRepository.list({
      includeArchived: true,
      limit: 10,
    });
    const sameLabelNodes = nodes.filter((node) => node.label === label);

    expect(result.candidate_stats).toEqual({
      proposed: 1,
      accepted: 1,
      rejected: 0,
    });
    expect(sameLabelNodes).toHaveLength(2);
    expect(sameLabelNodes.find((node) => node.id === existing.id)?.source_episode_ids).toEqual([
      priorEpisode.id,
    ]);
    expect(sameLabelNodes.find((node) => node.id !== existing.id)?.source_episode_ids).toEqual([
      episode.id,
    ]);
    expect(tracer.events).not.toContainEqual({
      event: "semantic_insert.skipped",
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
    await harness.episodicRepository.createEpisode(episode);
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
      event: "semantic_insert.skipped",
      data: expect.objectContaining({
        kind: "episode",
        reason: "episode_archived_post_plan",
      }),
    });
  });
});
