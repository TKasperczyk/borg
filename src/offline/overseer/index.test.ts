import { afterEach, describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "../../config/index.js";
import { FakeLLMClient } from "../../llm/index.js";
import type { ReviewOpenQuestionExtractorLike } from "../../memory/self/index.js";
import { StreamReader } from "../../stream/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { createEntityId, createMaintenanceRunId, createStreamEntryId } from "../../util/ids.js";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../test-support.js";
import { OverseerProcess } from "./index.js";

const OVERSEER_TOOL_NAME = "EmitOverseerFlags";
type OfflineHarness = Awaited<ReturnType<typeof createOfflineTestHarness>>;

function createOverseerResponse(flags: unknown[], inputTokens = 12, outputTokens = 8) {
  return {
    text: "",
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: OVERSEER_TOOL_NAME,
        input: { flags },
      },
    ],
  };
}

function supportedMisattributionFlag(
  citedStreamIds: readonly string[],
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    kind: "misattribution",
    reason: "The target memory attribution is unsupported by its cited source.",
    confidence: 0.8,
    patch: {
      participants: ["team", "Maya"],
    },
    source_assessment: "supports_flag",
    cited_stream_ids: [...citedStreamIds],
    ...overrides,
  };
}

async function appendSourceEntry(
  harness: OfflineHarness,
  content: string,
  kind: "user_msg" | "agent_msg" = "user_msg",
) {
  return harness.streamWriter.append({
    kind,
    content,
  });
}

function requestPrompt(llm: FakeLLMClient, index = 0): string {
  const content = llm.requests[index]?.messages[0]?.content;
  return typeof content === "string" ? content : JSON.stringify(content);
}

function maxChecksConfig(maxChecksPerRun = 1) {
  return {
    offline: {
      ...DEFAULT_CONFIG.offline,
      overseer: {
        ...DEFAULT_CONFIG.offline.overseer,
        maxChecksPerRun,
      },
    },
  };
}

describe("overseer process", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("flags misattribution-like issues and can revert the audit item", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient();
    const reviewOpenQuestionExtractor: ReviewOpenQuestionExtractorLike = {
      async extract(_item, context) {
        return {
          question: "¿Qué atribución debería corregirse?",
          urgency: 0.61,
          related_episode_ids: [...context.allowed_episode_ids],
          related_semantic_node_ids: [],
        };
      },
    };
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      reviewOpenQuestionExtractor,
      configOverrides: {
        anthropic: {
          models: {
            cognition: "cog-model",
            background: "bg-model",
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    const sourceEntry = await appendSourceEntry(harness, "Alex led the meeting.");
    llm.pushResponse(
      createOverseerResponse([
        supportedMisattributionFlag([sourceEntry.id], {
          reason: "The narrative mentions Alex, but Alex is missing from participants.",
          patch: {
            participants: ["team", "Alex"],
          },
        }),
      ]),
    );

    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Misattributed meeting",
          narrative: "Alex led the meeting, but the participants only mention the team.",
          participants: ["team"],
          source_stream_ids: [sourceEntry.id],
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
        },
        [0, 1, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });
    await harness.flushHookLogs();

    expect(result.errors).toEqual([]);
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: OVERSEER_TOOL_NAME,
    });
    expect(llm.requests[0]?.model).toBe("bg-model");
    expect(result.changes[0]).toMatchObject({
      action: "flag",
      targets: {
        kind: "misattribution",
      },
    });
    expect(harness.reviewQueueRepository.getOpen()[0]).toMatchObject({
      kind: "misattribution",
    });
    expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([
      expect.objectContaining({
        source: "overseer",
        question: "¿Qué atribución debería corregirse?",
        urgency: 0.61,
      }),
    ]);

    const auditRow = harness.auditLog.list({ process: "overseer" })[0];
    await harness.auditLog.revert(auditRow!.id, "test");
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
  });

  it("stays quiet on clean fixtures", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient({
      responses: [createOverseerResponse([], 8, 4)],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Clean semantic memory",
          description: "This proposition is aligned with the supporting evidence.",
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
          last_verified_at: nowMs - 1_000,
          source_episode_ids: [createEpisodeFixture().id],
        },
        [0, 0, 1, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });

    expect(result.changes).toEqual([]);
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
  });

  it("includes raw source entries for episode targets", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient({
      responses: [createOverseerResponse([])],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      configOverrides: maxChecksConfig(),
    });
    cleanup.push(harness.cleanup);

    const userSource = await appendSourceEntry(harness, "Maya is my partner.");
    const agentSource = await appendSourceEntry(
      harness,
      "I will remember Maya as your partner.",
      "agent_msg",
    );
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Maya partner source",
          narrative: "The user said Maya is their partner.",
          participants: ["user", "Maya"],
          source_stream_ids: [userSource.id, agentSource.id],
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
        },
        [1, 0, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    await process.run(harness.createContext(), {
      dryRun: true,
    });

    const prompt = requestPrompt(llm);
    expect(prompt).toContain(`stream_id=${userSource.id}`);
    expect(prompt).toContain(`stream_id=${agentSource.id}`);
    expect(prompt).toContain("kind=user_msg");
    expect(prompt).toContain("kind=agent_msg");
    expect(prompt).toContain("Maya is my partner.");
  });

  it("includes raw source entries for semantic node targets", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient({
      responses: [createOverseerResponse([])],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      configOverrides: maxChecksConfig(),
    });
    cleanup.push(harness.cleanup);

    const source = await appendSourceEntry(harness, "The user said Maya is my partner.");
    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Maya source episode",
          narrative: "Maya was identified by the user.",
          source_stream_ids: [source.id],
          created_at: nowMs - 3_000,
          updated_at: nowMs - 3_000,
        },
        [1, 0, 0, 0],
      ),
    );
    await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Maya is the user's partner",
          description: "Maya is the user's partner.",
          source_episode_ids: [episode.id],
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
          last_verified_at: nowMs - 1_000,
        },
        [0, 1, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    await process.run(harness.createContext(), {
      dryRun: true,
    });

    const prompt = requestPrompt(llm);
    expect(prompt).toContain(`source_episode_ids: ${episode.id}`);
    expect(prompt).toContain(`stream_id=${source.id}`);
    expect(prompt).toContain("The user said Maya is my partner.");
  });

  it("includes raw source entries for semantic edge targets", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient({
      responses: [createOverseerResponse([])],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      configOverrides: maxChecksConfig(),
    });
    cleanup.push(harness.cleanup);

    const source = await appendSourceEntry(harness, "Maya supports the user's household planning.");
    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Maya edge evidence",
          narrative: "Maya supports household planning.",
          source_stream_ids: [source.id],
          created_at: nowMs - 4_000,
          updated_at: nowMs - 4_000,
        },
        [1, 0, 0, 0],
      ),
    );
    const first = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Maya",
          description: "Maya.",
          source_episode_ids: [episode.id],
          created_at: nowMs - 3_000,
          updated_at: nowMs - 3_000,
        },
        [1, 0, 0, 0],
      ),
    );
    const second = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Household planning",
          description: "Household planning.",
          source_episode_ids: [episode.id],
          created_at: nowMs - 2_000,
          updated_at: nowMs - 2_000,
        },
        [0, 1, 0, 0],
      ),
    );
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "supports",
      confidence: 0.8,
      evidence_episode_ids: [episode.id],
      created_at: nowMs - 1_000,
      last_verified_at: nowMs - 1_000,
      valid_from: nowMs - 1_000,
    });

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    await process.run(harness.createContext(), {
      dryRun: true,
    });

    const prompt = requestPrompt(llm);
    expect(prompt).toContain(`target_id: ${edge.id}`);
    expect(prompt).toContain(`stream_id=${source.id}`);
    expect(prompt).toContain("Maya supports the user's household planning.");
  });

  it("marks missing stream source IDs as provenance-insufficient in the prompt", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient({
      responses: [createOverseerResponse([])],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      configOverrides: maxChecksConfig(),
    });
    cleanup.push(harness.cleanup);

    const missingStreamId = createStreamEntryId();
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Missing source episode",
          source_stream_ids: [missingStreamId],
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
        },
        [1, 0, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    await process.run(harness.createContext(), {
      dryRun: true,
    });

    const prompt = requestPrompt(llm);
    expect(prompt).toContain("PROVENANCE-INSUFFICIENT missing source_stream_ids");
    expect(prompt).toContain(missingStreamId);
  });

  it("resolves audience-private source episodes for audit grounding", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient({
      responses: [createOverseerResponse([])],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      configOverrides: maxChecksConfig(),
    });
    cleanup.push(harness.cleanup);

    const source = await appendSourceEntry(harness, "Private audience source names Maya.");
    const privateEpisode = await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Private source episode",
          narrative: "Private source names Maya.",
          source_stream_ids: [source.id],
          audience_entity_id: createEntityId(),
          shared: false,
          created_at: nowMs - 3_000,
          updated_at: nowMs - 3_000,
        },
        [1, 0, 0, 0],
      ),
    );
    await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Maya private source",
          description: "Maya is grounded by private audience evidence.",
          source_episode_ids: [privateEpisode.id],
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
          last_verified_at: nowMs - 1_000,
        },
        [0, 1, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    await process.run(harness.createContext(), {
      dryRun: true,
    });

    const prompt = requestPrompt(llm);
    expect(prompt).toContain(`source_episode_ids: ${privateEpisode.id}`);
    expect(prompt).toContain(`stream_id=${source.id}`);
    expect(prompt).toContain("Private audience source names Maya.");
  });

  it("suppresses Maya misattribution flags when source entries contradict the flag", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      configOverrides: maxChecksConfig(),
    });
    cleanup.push(harness.cleanup);

    const source = await appendSourceEntry(harness, "Maya is my partner.");
    llm.pushResponse(
      createOverseerResponse([
        supportedMisattributionFlag([source.id], {
          reason: "Borg fabricated Maya.",
          source_assessment: "contradicts_flag",
        }),
      ]),
    );
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Maya source",
          narrative: "The user said Maya is their partner.",
          source_stream_ids: [source.id],
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
        },
        [1, 0, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    const ctx = harness.createContext();
    const plan = await process.plan(ctx, {});
    await process.apply(ctx, plan);

    expect(plan.items).toEqual([]);
    expect(plan.suppressed_flags).toEqual([
      expect.objectContaining({
        reason: "SOURCE-CONTRADICTS",
        cited_ids: [source.id],
      }),
    ]);
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
  });

  it("suppresses misattribution flags without cited stream IDs", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      configOverrides: maxChecksConfig(),
    });
    cleanup.push(harness.cleanup);

    const source = await appendSourceEntry(harness, "Maya is my partner.");
    llm.pushResponse(
      createOverseerResponse([
        supportedMisattributionFlag([], {
          reason: "Misattribution without citations.",
        }),
      ]),
    );
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Maya source",
          source_stream_ids: [source.id],
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
        },
        [1, 0, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    const ctx = harness.createContext();
    const plan = await process.plan(ctx, {});
    await process.apply(ctx, plan);

    expect(plan.items).toEqual([]);
    expect(plan.suppressed_flags).toEqual([
      expect.objectContaining({
        reason: "PROVENANCE-INSUFFICIENT",
        cited_ids: [],
      }),
    ]);
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
  });

  it("suppresses misattribution flags with citations outside the target source bundle", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      configOverrides: maxChecksConfig(),
    });
    cleanup.push(harness.cleanup);

    const source = await appendSourceEntry(harness, "Maya is my partner.");
    const invalidCitation = createStreamEntryId();
    llm.pushResponse(
      createOverseerResponse([
        supportedMisattributionFlag([invalidCitation], {
          reason: "Misattribution with unrelated citation.",
        }),
      ]),
    );
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Maya source",
          source_stream_ids: [source.id],
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
        },
        [1, 0, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    const ctx = harness.createContext();
    const plan = await process.plan(ctx, {});
    await process.apply(ctx, plan);

    expect(plan.items).toEqual([]);
    expect(plan.suppressed_flags).toEqual([
      expect.objectContaining({
        reason: "INVALID-CITATION",
        cited_ids: [invalidCitation],
      }),
    ]);
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
  });

  it("creates misattribution reviews only when cited source entries support the flag", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      configOverrides: maxChecksConfig(),
    });
    cleanup.push(harness.cleanup);

    const source = await appendSourceEntry(
      harness,
      "The user said Riley, not Maya, is my partner.",
    );
    llm.pushResponse(
      createOverseerResponse([
        supportedMisattributionFlag([source.id], {
          reason: "The target names Maya, but the source supports Riley.",
          patch: {
            participants: ["user", "Riley"],
            narrative: "The user said Riley is their partner.",
          },
        }),
      ]),
    );
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Partner source",
          narrative: "The target incorrectly names Maya.",
          participants: ["user", "Maya"],
          source_stream_ids: [source.id],
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
        },
        [1, 0, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    const ctx = harness.createContext();
    const plan = await process.plan(ctx, {});
    await process.apply(ctx, plan);

    expect(plan.suppressed_flags).toEqual([]);
    expect(plan.items).toHaveLength(1);
    expect(harness.reviewQueueRepository.getOpen()[0]).toMatchObject({
      kind: "misattribution",
      refs: {
        evidence_stream_ids: [source.id],
      },
    });
  });

  it("queues temporal drift reviews for semantic edges without mutating them", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const suggestedValidTo = nowMs - 500;
    const llm = new FakeLLMClient({
      responses: [
        createOverseerResponse([
          {
            kind: "temporal_drift",
            reason: "The edge only held before the later rollback evidence.",
            confidence: 0.82,
            suggested_valid_to: suggestedValidTo,
          },
        ]),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
      configOverrides: {
        offline: {
          ...DEFAULT_CONFIG.offline,
          overseer: {
            ...DEFAULT_CONFIG.offline.overseer,
            maxChecksPerRun: 1,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    const episodeId = createEpisodeFixture().id;
    const first = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Atlas edge source",
          description: "Atlas was stable.",
          source_episode_ids: [episodeId],
          created_at: nowMs - 2_000,
          updated_at: nowMs - 2_000,
        },
        [1, 0, 0, 0],
      ),
    );
    const second = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Atlas edge target",
          description: "Rollback had completed.",
          source_episode_ids: [episodeId],
          created_at: nowMs - 1_900,
          updated_at: nowMs - 1_900,
        },
        [0, 1, 0, 0],
      ),
    );
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "supports",
      confidence: 0.8,
      evidence_episode_ids: [episodeId],
      created_at: nowMs - 100,
      last_verified_at: nowMs - 100,
      valid_from: nowMs - 100,
    });

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });

    expect(result.errors).toEqual([]);
    expect(result.changes[0]).toMatchObject({
      targets: {
        kind: "temporal_drift",
        target_type: "semantic_edge",
        target_id: edge.id,
      },
      preview: {
        suggested_valid_to: suggestedValidTo,
      },
    });
    expect(harness.reviewQueueRepository.getOpen()[0]).toMatchObject({
      kind: "temporal_drift",
      refs: {
        target_type: "semantic_edge",
        target_kind: "semantic_edge",
        target_id: edge.id,
        suggested_valid_to: suggestedValidTo,
      },
    });
    expect(harness.semanticEdgeRepository.getEdge(edge.id)?.valid_to).toBeNull();
  });

  it("halts further llm work after budget exhaustion", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createOverseerResponse(
          [
            {
              kind: "temporal_drift",
              reason: "First target issue.",
              confidence: 0.8,
            },
          ],
          35,
          25,
        ),
        createOverseerResponse(
          [
            {
              kind: "temporal_drift",
              reason: "Second target issue.",
              confidence: 0.8,
            },
          ],
          35,
          25,
        ),
        createOverseerResponse(
          [
            {
              kind: "identity_inconsistency",
              reason: "Third target issue.",
              confidence: 0.8,
            },
          ],
          35,
          25,
        ),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      configOverrides: {
        offline: {
          ...DEFAULT_CONFIG.offline,
          overseer: {
            ...DEFAULT_CONFIG.offline.overseer,
            maxChecksPerRun: 3,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Target one",
          created_at: 1_000,
          updated_at: 1_000,
        },
        [1, 0, 0, 0],
      ),
    );
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Target two",
          created_at: 2_000,
          updated_at: 2_000,
        },
        [1, 0, 0, 0],
      ),
    );
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Target three",
          created_at: 3_000,
          updated_at: 3_000,
        },
        [1, 0, 0, 0],
      ),
    );

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    const result = await process.run(harness.createContext(), {
      dryRun: false,
      budget: 100,
    });

    expect(result.budget_exhausted).toBe(true);
    expect(llm.requests).toHaveLength(2);
    expect(result.changes).toHaveLength(1);
  });

  it("respects the lookback window when prior audit history is stale", async () => {
    const nowMs = 20 * 24 * 60 * 60 * 1_000;
    const clock = new ManualClock(nowMs - 10 * 24 * 60 * 60 * 1_000);
    const recentEpisode = createEpisodeFixture(
      {
        title: "Recent target",
        created_at: nowMs - 60 * 60 * 1_000,
        updated_at: nowMs - 60 * 60 * 1_000,
      },
      [1, 0, 0, 0],
    );
    const oldEpisode = createEpisodeFixture(
      {
        title: "Old target",
        created_at: nowMs - 5 * 24 * 60 * 60 * 1_000,
        updated_at: nowMs - 5 * 24 * 60 * 60 * 1_000,
      },
      [1, 0, 0, 0],
    );
    const llm = new FakeLLMClient({
      responses: [
        createOverseerResponse([
          {
            kind: "temporal_drift",
            reason: "Recent target issue.",
            confidence: 0.8,
          },
        ]),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock,
      llmClient: llm,
      configOverrides: {
        offline: {
          ...DEFAULT_CONFIG.offline,
          overseer: {
            ...DEFAULT_CONFIG.offline.overseer,
            lookbackHours: 24,
            maxChecksPerRun: 8,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    harness.auditLog.record({
      run_id: createMaintenanceRunId(),
      process: "overseer",
      action: "flag",
      targets: {
        seed: true,
      },
      reversal: {},
    });
    clock.set(nowMs);

    await harness.episodicRepository.insert(oldEpisode);
    await harness.episodicRepository.insert(recentEpisode);

    const process = new OverseerProcess({
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
    });
    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });

    expect(result.errors).toEqual([]);
    expect(llm.requests).toHaveLength(1);
    expect(harness.reviewQueueRepository.getOpen()[0]).toMatchObject({
      refs: {
        target_id: recentEpisode.id,
      },
    });
  });

  it("continues and logs when the review-to-open-question hook fails", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    const reviewQueueRepository = harness.reviewQueueRepository as unknown as {
      options: {
        onEnqueue?: (item: unknown, input: unknown) => void;
      };
    };
    const originalHook = reviewQueueRepository.options.onEnqueue;
    reviewQueueRepository.options.onEnqueue = () => {
      throw new Error("hook exploded");
    };

    try {
      const sourceEntry = await appendSourceEntry(harness, "Alex led the meeting.");
      llm.pushResponse(
        createOverseerResponse([
          supportedMisattributionFlag([sourceEntry.id], {
            reason: "The narrative mentions Alex, but Alex is missing from participants.",
            patch: {
              participants: ["team", "Alex"],
            },
          }),
        ]),
      );

      await harness.episodicRepository.insert(
        createEpisodeFixture(
          {
            title: "Misattributed meeting",
            narrative: "Alex led the meeting, but the participants only mention the team.",
            participants: ["team"],
            source_stream_ids: [sourceEntry.id],
            created_at: nowMs - 1_000,
            updated_at: nowMs - 1_000,
          },
          [0, 1, 0, 0],
        ),
      );

      const process = new OverseerProcess({
        reviewQueueRepository: harness.reviewQueueRepository,
        registry: harness.registry,
      });

      const result = await process.run(harness.createContext(), {
        dryRun: false,
      });

      await harness.flushHookLogs();

      const entries = new StreamReader({
        dataDir: harness.tempDir,
      }).tail(1);

      expect(result.errors).toEqual([]);
      expect(harness.reviewQueueRepository.getOpen()).toHaveLength(1);
      expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([]);
      expect(entries[0]).toMatchObject({
        kind: "internal_event",
        content: {
          hook: "review_queue_open_question",
        },
      });
    } finally {
      reviewQueueRepository.options.onEnqueue = originalHook;
    }
  });
});
