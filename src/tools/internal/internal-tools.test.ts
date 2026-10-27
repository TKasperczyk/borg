import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { Borg, DEFAULT_SESSION_ID } from "../../index.js";
import { ScheduledWakesRepository } from "../../autonomy/index.js";
import { DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET } from "../../cognition/deliberation/constants.js";
import { PromptSurfaceHistoryRepository } from "../../cognition/prompts/prompt-surface-history.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { buildToolDispatcher } from "../../borg/tools-setup.js";
import { deriveProceduralContextKey } from "../../memory/procedural/index.js";
import { SemanticGraph } from "../../memory/semantic/index.js";
import { TrainOfThoughtRepository } from "../../memory/train-of-thought/index.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticEdgeFixture,
  createSemanticNodeFixture,
  createTestConfig,
  TestEmbeddingClient,
} from "../../offline/test-support.js";
import { StreamEntryIndexRepository, StreamWriter } from "../../stream/index.js";
import { SourceStreamAudienceDisclosureResolver } from "../../memory/common/index.js";
import {
  ToolDispatcher,
  createCommitmentsListTool,
  createEpisodicSearchTool,
  createIdentityEventsListForCognitionTool,
  createOpenQuestionsCreateTool,
  createSemanticWalkTool,
  createSkillsListTool,
} from "../../tools/index.js";
import { ManualClock } from "../../util/clock.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import {
  createAttachmentId,
  createConsolidationFamilyId,
  createEntityId,
  createEpisodeId,
  createScheduledWakeId,
  createSemanticNodeId,
  createSessionId,
  createStreamEntryId,
} from "../../util/ids.js";

const TYPESCRIPT_DEBUG_CONTEXT_KEY = deriveProceduralContextKey({
  problem_kind: "code_debugging",
  domain_tags: ["typescript"],
  audience_scope: "self",
});

async function openTestBorg(tempDir: string, llm = new FakeLLMClient()) {
  return Borg.open({
    config: createTestConfig({
      dataDir: tempDir,
      perception: {
        llmEnabled: false,
      },
      embedding: {
        baseUrl: "http://localhost:1234/v1",
        apiKey: "test",
        model: "test-embed",
        dims: 4,
      },
      anthropic: {
        auth: "api-key",
        apiKey: "test",
        models: {
          cognition: "test-cognition",
          background: "test-background",
          extraction: "test-extraction",
        },
      },
    }),
    clock: new ManualClock(1_000_000),
    embeddingDimensions: 4,
    embeddingClient: new TestEmbeddingClient(),
    llmClient: llm,
    liveExtraction: false,
  });
}

function createHarnessToolDispatcher(
  harness: Awaited<ReturnType<typeof createOfflineTestHarness>>,
) {
  const clock = new ManualClock(1_000_100);
  const semanticGraph = new SemanticGraph({
    nodeRepository: harness.semanticNodeRepository,
    edgeRepository: harness.semanticEdgeRepository,
  });
  const promptSurfaceHistoryRepository = new PromptSurfaceHistoryRepository({
    db: harness.db,
    clock,
  });
  promptSurfaceHistoryRepository.observeCurrent();
  const entryIndex = new StreamEntryIndexRepository({
    db: harness.db,
    dataDir: harness.tempDir,
  });

  return buildToolDispatcher({
    dataDir: harness.tempDir,
    entryIndex,
    sourceStreamAudienceDisclosureResolver: new SourceStreamAudienceDisclosureResolver({
      dataDir: harness.tempDir,
      entryIndex,
      sessionsRepository: { getMany: () => [] },
      entityRepository: harness.entityRepository,
    }),
    retrievalPipeline: harness.retrievalPipeline,
    episodicRepository: harness.episodicRepository,
    semanticNodeRepository: harness.semanticNodeRepository,
    semanticGraph,
    commitmentRepository: harness.commitmentRepository,
    entityRepository: harness.entityRepository,
    goalsRepository: harness.goalsRepository,
    openQuestionsRepository: harness.openQuestionsRepository,
    identityService: harness.identityService,
    skillRepository: harness.skillRepository,
    trainOfThoughtRepository: new TrainOfThoughtRepository({ db: harness.db, clock }),
    scheduledWakesRepository: new ScheduledWakesRepository({ db: harness.db, clock }),
    promptSurfaceHistoryRepository,
    createStreamWriter: (sessionId) =>
      new StreamWriter({
        dataDir: harness.tempDir,
        sessionId,
        clock,
        entryIndex,
      }),
    clock,
  });
}

describe("internal tools", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("searches episodic memory", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm);

    try {
      const entry = await borg.stream.append({
        kind: "user_msg",
        content: "planning sync for sprint follow-up",
      });
      llm.pushResponse({
        text: "",
        input_tokens: 10,
        output_tokens: 5,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "toolu_extract_2",
            name: "EmitEpisodeCandidates",
            input: {
              episodes: [
                {
                  title: "Planning sync",
                  narrative: "The team aligned on the sprint plan and next steps.",
                  source_stream_ids: [entry.id],
                  participants: ["team"],
                  location: null,
                  tags: ["planning"],
                  confidence: 0.8,
                  significance: 0.8,
                },
              ],
            },
          },
        ],
      });
      await borg.episodic.extract({
        session: DEFAULT_SESSION_ID,
      });

      const tool = createEpisodicSearchTool({
        searchEpisodes: (query, limit, context) =>
          borg.episodic.search(query, {
            limit,
            audienceEntityId: context.audienceEntityId,
          }),
      });
      const result = await tool.invoke(
        {
          query: "planning",
        },
        {
          sessionId: DEFAULT_SESSION_ID,
          origin: "autonomous",
        },
      );

      expect(result.episodes[0]?.title).toBe("Planning sync");
    } finally {
      await borg.close();
    }
  });

  it("returns usable episodic evidence from search results", async () => {
    const longNarrative = `${"The team traced the retrieval path and compared the tool payload to the prompt evidence. ".repeat(
      8,
    )}The final sentence should be truncated.`;
    const citationAudience = createEntityId();
    const episode = createEpisodeFixture({
      title: "Retrieval evidence review",
      narrative: longNarrative,
      participants: ["Ari", "Sam"],
      tags: ["retrieval", "tools"],
      start_time: 1_700_000,
      end_time: 1_701_000,
      audience_entity_id: citationAudience,
      shared: false,
    });
    const tool = createEpisodicSearchTool({
      searchEpisodes: async (_query, limit, _context) => {
        expect(limit).toBe(5);

        return [
          {
            episode,
            score: 0.82,
            rawScore: 0.82,
            scoreBreakdown: {
              similarity: 0.91,
              decayedSalience: 0.73,
              heat: 2,
              goalRelevance: 0,
              valueAlignment: 0,
              timeRelevance: 0.44,
              moodBoost: 0,
              socialRelevance: 0,
              entityRelevance: 0,
              suppressionPenalty: 0,
            },
            citationChain: [
              {
                id: episode.source_stream_ids[0]!,
                timestamp: 1_699_990,
                kind: "user_msg",
                content: "We need the search tool to return evidence, not only ids.",
                session_id: DEFAULT_SESSION_ID,
                compressed: false,
                sender_entity_id: null,
                reply_target_entity_id: null,
              },
            ],
          },
        ];
      },
    });

    const result = await tool.invoke(
      {
        query: "retrieval evidence",
      },
      {
        sessionId: DEFAULT_SESSION_ID,
        origin: "deliberator",
      },
    );

    expect(result.episodes).toHaveLength(1);
    expect(result.episodes[0]).toMatchObject({
      id: episode.id,
      title: "Retrieval evidence review",
      participants: ["Ari", "Sam"],
      tags: ["retrieval", "tools"],
      start_time: 1_700_000,
      end_time: 1_701_000,
      source_stream_ids: episode.source_stream_ids,
      score: 0.82,
      score_breakdown: {
        similarity: 0.91,
        decayed_salience: 0.73,
        time_relevance: 0.44,
      },
      citation_chain: [
        expect.objectContaining({
          id: episode.source_stream_ids[0],
          kind: "user_msg",
          content: "We need the search tool to return evidence, not only ids.",
          disclosure: expect.stringContaining("disclosure_class=relationship_private"),
          disclosure_label: expect.objectContaining({
            disclosure_class: "relationship_private",
            private_to_entity_ids: [citationAudience],
          }),
        }),
      ],
    });
    expect(result.episodes[0]?.narrative.length).toBeLessThanOrEqual(400);
    expect(result.episodes[0]?.narrative).toContain("The team traced the retrieval path");
  });

  it("reads one full episode or indexed stream record across sessions with disclosure labels", async () => {
    const harness = await createOfflineTestHarness();
    try {
      const alice = harness.entityRepository.resolve("Alice");
      const bob = harness.entityRepository.resolve("Bob");
      const sessionId = createSessionId();
      const fullText = "条件と具体例。".repeat(150) + "最後の例外も保持する。";
      const entryIndex = new StreamEntryIndexRepository({
        db: harness.db,
        dataDir: harness.tempDir,
      });
      const writer = new StreamWriter({
        dataDir: harness.tempDir,
        sessionId,
        entryIndex,
        clock: harness.clock,
      });
      const entry = await writer.append({ kind: "user_msg", content: fullText, audience: "Bob" });
      const episode = await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          narrative: fullText,
          audience_entity_id: bob,
          shared: false,
          source_stream_ids: [entry.id],
        }),
      );
      const dispatcher = createHarnessToolDispatcher(harness);
      const readEpisode = await dispatcher.dispatch({
        toolName: "tool.episodic.get",
        input: { episode_id: episode.id },
        origin: "autonomous",
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: alice,
      });
      expect(readEpisode).toMatchObject({
        ok: true,
        output: {
          episode: {
            id: episode.id,
            narrative: fullText,
            source_stream_ids: [entry.id],
            lineage: episode.lineage,
            archived: false,
            lifecycle_status: "active",
            disclosure: expect.stringContaining("relationship_private"),
            disclosure_label: { private_to_entity_ids: [bob] },
          },
        },
      });
      const readStream = await dispatcher.dispatch({
        toolName: "tool.stream.get",
        input: { stream_entry_id: entry.id },
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: alice,
      });
      expect(readStream).toMatchObject({
        ok: true,
        output: {
          entry: {
            id: entry.id,
            session_id: sessionId,
            text: fullText,
            active: true,
            disclosure: expect.stringContaining("relationship_private"),
            disclosure_label: { private_to_entity_ids: [bob] },
          },
        },
      });
      const familyId = createConsolidationFamilyId();
      const summary = await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          episode_kind: "consolidation_version",
          consolidation_family_id: familyId,
          consolidation_coverage_hash: "fixture-coverage",
          audience_entity_id: bob,
          shared: false,
        }),
      );
      harness.db
        .prepare(
          `INSERT INTO consolidation_families
        (family_id, current_version_episode_id, coverage_hash, policy_version, created_at, updated_at)
        VALUES (?, ?, ?, 1, ?, ?)`,
        )
        .run(familyId, summary.id, "fixture-coverage", harness.clock.now(), harness.clock.now());
      harness.db
        .prepare(
          `INSERT INTO consolidation_members
        (family_id, raw_episode_id, source_stream_ids_json, added_by_version_episode_id)
        VALUES (?, ?, ?, ?)`,
        )
        .run(familyId, episode.id, JSON.stringify(episode.source_stream_ids), summary.id);
      expect(await harness.episodicRepository.get(episode.id)).toBeNull();
      expect(
        await dispatcher.dispatch({
          toolName: "tool.episodic.get",
          input: { episode_id: episode.id },
          origin: "autonomous",
          sessionId: DEFAULT_SESSION_ID,
          audienceEntityId: alice,
        }),
      ).toMatchObject({
        ok: true,
        output: {
          episode: {
            narrative: fullText,
            archived: false,
            lifecycle_status: "inactive",
            disclosure_label: { private_to_entity_ids: [bob] },
          },
        },
      });
      harness.episodicRepository.archiveEpisode(episode.id, {
        caller: "internal-tools.test",
        reason: "lifecycle fixture",
        process: "curator",
      });
      expect(
        await dispatcher.dispatch({
          toolName: "tool.episodic.get",
          input: { episode_id: episode.id },
          origin: "autonomous",
          sessionId: DEFAULT_SESSION_ID,
        }),
      ).toMatchObject({
        ok: true,
        output: {
          episode: {
            narrative: fullText,
            archived: true,
            lifecycle_status: "archived",
          },
        },
      });
      for (const [toolName, input, field] of [
        ["tool.episodic.get", { episode_id: createEpisodeId() }, "episode"],
        ["tool.stream.get", { stream_entry_id: createStreamEntryId() }, "entry"],
      ] as const) {
        expect(
          await dispatcher.dispatch({
            toolName,
            input,
            origin: "autonomous",
            sessionId: DEFAULT_SESSION_ID,
          }),
        ).toMatchObject({ ok: true, output: { [field]: null } });
      }
    } finally {
      await harness.cleanup();
    }
  });

  it("recalls episodic search globally with disclosure labels", async () => {
    const harness = await createOfflineTestHarness({
      embeddingClient: new TestEmbeddingClient(new Map([["planning roadmap", [0, 1, 0, 0]]])),
    });

    try {
      const alice = harness.entityRepository.resolve("Alice");
      const bob = harness.entityRepository.resolve("Bob");
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          title: "Alice private planning",
          narrative: "Alice discussed a private roadmap planning note.",
          participants: ["Alice"],
          tags: ["planning", "roadmap"],
          audience_entity_id: alice,
          shared: false,
        }),
      );
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          title: "Bob private planning",
          narrative: "Bob discussed a private roadmap planning note.",
          participants: ["Bob"],
          tags: ["planning", "roadmap"],
          audience_entity_id: bob,
          shared: false,
        }),
      );

      const dispatcher = createHarnessToolDispatcher(harness);
      const result = await dispatcher.dispatch({
        toolName: "tool.episodic.search",
        input: {
          query: "planning roadmap",
          limit: 5,
        },
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: alice,
      });

      expect(result.ok).toBe(true);
      if (!result.ok) {
        throw new Error(result.error);
      }
      const output = result.output as {
        episodes: Array<{
          title: string;
          disclosure: string;
          disclosure_label: {
            disclosure_class: string;
            private_to_entity_ids: string[];
          };
        }>;
      };
      const byTitle = new Map(output.episodes.map((episode) => [episode.title, episode]));
      expect(byTitle.get("Alice private planning")?.disclosure_label).toMatchObject({
        disclosure_class: "relationship_private",
        private_to_entity_ids: [alice],
      });
      expect(byTitle.get("Alice private planning")?.disclosure).toContain(
        "disclosure_class=relationship_private",
      );
      expect(byTitle.get("Bob private planning")?.disclosure_label).toMatchObject({
        disclosure_class: "relationship_private",
        private_to_entity_ids: [bob],
      });
      expect(byTitle.get("Bob private planning")?.disclosure).toContain(
        "disclosure_class=relationship_private",
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("walks the semantic graph", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openTestBorg(tempDir);

    try {
      const root = await borg.semantic.nodes.add({
        kind: "concept",
        label: "Planning",
        description: "Planning work",
        domain: "process",
        sourceEpisodeIds: [createEpisodeId()],
      });
      const child = await borg.semantic.nodes.add({
        kind: "concept",
        label: "Sprint 10",
        description: "Autonomy work",
        sourceEpisodeIds: [createEpisodeId()],
      });
      expect(root.domain).toBe("process");
      borg.semantic.edges.add({
        from_node_id: root.id,
        to_node_id: child.id,
        relation: "supports",
        confidence: 0.8,
        evidence_episode_ids: [createEpisodeId()],
        created_at: 1_000_000,
        last_verified_at: 1_000_000,
      });

      const tool = createSemanticWalkTool({
        walkGraph: (fromId, options) => borg.semantic.walk(fromId, options),
      });
      const result = await tool.invoke(
        {
          node_id: root.id,
          relation: "supports",
        },
        {
          sessionId: DEFAULT_SESSION_ID,
          origin: "autonomous",
        },
      );

      expect(result.steps[0]?.node.id).toBe(child.id);
    } finally {
      await borg.close();
    }
  });

  it("carries corrected_by on semantic walk node output", async () => {
    const rootId = createSemanticNodeId();
    const correctedBy = createSemanticNodeId();
    const node = createSemanticNodeFixture({
      status: "superseded",
      corrected_by: correctedBy,
      superseded_at: 1_250,
    });
    const edge = createSemanticEdgeFixture({
      from_node_id: rootId,
      to_node_id: node.id,
    });
    const tool = createSemanticWalkTool({
      walkGraph: async () => [
        {
          node,
          edgePath: [edge],
        },
      ],
    });

    const result = await tool.invoke(
      {
        node_id: rootId,
        relation: "supports",
      },
      {
        sessionId: DEFAULT_SESSION_ID,
        origin: "deliberator",
      },
    );

    expect(result.steps[0]?.node.status).toBe("superseded");
    expect(result.steps[0]?.node.superseded_at).toBe(1_250);
    expect(result.steps[0]?.node.corrected_by).toBe(correctedBy);
    expect(result.steps[0]?.node.disclosure_label).toEqual({
      disclosure_class: "unknown",
      origin_audience_entity_ids: [],
      private_to_entity_ids: [],
      public_to_entity_ids: [],
    });
    expect(result.steps[0]?.edgePath[0]?.disclosure_label).toEqual({
      disclosure_class: "unknown",
      origin_audience_entity_ids: [],
      private_to_entity_ids: [],
      public_to_entity_ids: [],
    });
  });

  it("forwards semantic walk as-of to the graph", async () => {
    let receivedOptions: Parameters<Parameters<typeof createSemanticWalkTool>[0]["walkGraph"]>[1];
    const nodeId = createSemanticNodeId();
    const tool = createSemanticWalkTool({
      walkGraph: async (_fromId, options) => {
        receivedOptions = options;
        return [];
      },
    });

    const result = await tool.invoke(
      {
        node_id: nodeId,
        relation: "supports",
        asOf: 1_250,
      },
      {
        sessionId: DEFAULT_SESSION_ID,
        origin: "autonomous",
      },
    );

    expect(result.steps).toEqual([]);
    expect(receivedOptions).toMatchObject({
      asOf: 1_250,
      depth: 2,
      maxNodes: 16,
      relations: ["supports"],
    });
  });

  it("walks semantic graph globally with disclosure labels", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const alice = harness.entityRepository.resolve("Alice");
      const bob = harness.entityRepository.resolve("Bob");
      const publicEpisode = await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Public semantic root",
          narrative: "Public evidence anchors the root node.",
        }),
      );
      const aliceEpisode = await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          title: "Alice semantic support",
          narrative: "Alice-only evidence supports the root node.",
          audience_entity_id: alice,
          shared: false,
        }),
      );
      const bobEpisode = await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_cccccccccccccccc" as never,
          title: "Bob semantic support",
          narrative: "Bob-only evidence supports the root node.",
          audience_entity_id: bob,
          shared: false,
        }),
      );
      const root = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture({
          label: "Planning root",
          source_episode_ids: [publicEpisode.id],
        }),
      );
      const aliceNode = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture({
          label: "Alice support",
          source_episode_ids: [aliceEpisode.id],
        }),
      );
      const bobNode = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture({
          label: "Bob support",
          source_episode_ids: [bobEpisode.id],
        }),
      );
      harness.semanticEdgeRepository.addEdge({
        from_node_id: root.id,
        to_node_id: aliceNode.id,
        relation: "supports",
        confidence: 0.8,
        evidence_episode_ids: [aliceEpisode.id],
        created_at: 1_000_000,
        last_verified_at: 1_000_000,
      });
      harness.semanticEdgeRepository.addEdge({
        from_node_id: root.id,
        to_node_id: bobNode.id,
        relation: "supports",
        confidence: 0.8,
        evidence_episode_ids: [bobEpisode.id],
        created_at: 1_000_000,
        last_verified_at: 1_000_000,
      });

      const dispatcher = createHarnessToolDispatcher(harness);
      const result = await dispatcher.dispatch({
        toolName: "tool.semantic.walk",
        input: {
          node_id: root.id,
          relation: "supports",
        },
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: alice,
      });

      expect(result.ok).toBe(true);
      if (!result.ok) {
        throw new Error(result.error);
      }

      const output = result.output as {
        steps: Array<{
          node: {
            id: string;
            label: string;
            disclosure_label?: unknown;
          };
          edgePath: Array<{ disclosure_label?: unknown }>;
        }>;
      };
      const stepsByNodeId = new Map(output.steps.map((step) => [step.node.id, step]));
      expect(stepsByNodeId.has(aliceNode.id)).toBe(true);
      expect(stepsByNodeId.has(bobNode.id)).toBe(true);
      expect(stepsByNodeId.get(aliceNode.id)?.node.disclosure_label).toEqual({
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [alice],
        private_to_entity_ids: [alice],
        public_to_entity_ids: [],
      });
      expect(stepsByNodeId.get(aliceNode.id)?.edgePath.at(-1)?.disclosure_label).toEqual({
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [alice],
        private_to_entity_ids: [alice],
        public_to_entity_ids: [],
      });
      expect(stepsByNodeId.get(bobNode.id)?.node.disclosure_label).toEqual({
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [bob],
        private_to_entity_ids: [bob],
        public_to_entity_ids: [],
      });
      expect(stepsByNodeId.get(bobNode.id)?.edgePath.at(-1)?.disclosure_label).toEqual({
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [bob],
        private_to_entity_ids: [bob],
        public_to_entity_ids: [],
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("returns structural prompt-surface changes for autonomous reflection", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const dispatcher = createHarnessToolDispatcher(harness);
      const result = await dispatcher.dispatch({
        toolName: "tool.promptSurface.changes",
        input: {},
        origin: "autonomous",
        sessionId: DEFAULT_SESSION_ID,
      });

      expect(result.ok).toBe(true);
      if (!result.ok) {
        throw new Error(result.error);
      }

      const output = result.output as {
        current: Record<string, unknown>;
        changes: Array<Record<string, unknown>>;
      };
      expect(Object.keys(output).sort()).toEqual(["changes", "current"]);
      expect(Object.keys(output.current).sort()).toEqual([
        "block_ids",
        "hash",
        "observed_at",
        "surface_placements",
      ]);
      expect(output.current.block_ids).toContain("borg_autonomous_reflection");
      expect(output.current.surface_placements).toContainEqual({
        block_id: "borg_autonomous_reflection",
        surface: "base_trusted_guidance_sections",
        order: 50,
      });
      expect(output.changes).toHaveLength(1);
      expect(Object.keys(output.changes[0] ?? {}).sort()).toEqual([
        "added_block_ids",
        "added_surface_placements",
        "from_hash",
        "observed_at",
        "removed_block_ids",
        "removed_surface_placements",
        "to_hash",
      ]);
      expect(output.changes[0]).toMatchObject({
        from_hash: null,
        to_hash: output.current.hash,
        added_block_ids: [],
        removed_block_ids: [],
        added_surface_placements: [],
        removed_surface_placements: [],
      });
      expect(JSON.stringify(output)).not.toContain("renderCondition");
      expect(JSON.stringify(output)).not.toContain("purpose");
    } finally {
      await harness.cleanup();
    }
  });

  // Pins the claim tool.journal.append's description makes about the registry,
  // so adding a journal mutator later fails here instead of leaving the surface
  // asserting something the tool list no longer supports.
  it("composes exactly one journal tool, the append that calls itself immutable", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const journalTools = createHarnessToolDispatcher(harness)
        .listTools("autonomous")
        .filter((tool) => tool.name.startsWith("tool.journal."));

      expect(journalTools.map((tool) => tool.name)).toEqual(["tool.journal.append"]);
      expect(journalTools[0]?.description).toContain("no tool amends or deletes one");
    } finally {
      await harness.cleanup();
    }
  });

  it("lists exact thought and journal records globally through the composed dispatcher", async () => {
    const harness = await createOfflineTestHarness();
    const otherSessionId = createSessionId();
    const sharedEntryIndex = new StreamEntryIndexRepository({
      db: harness.db,
      dataDir: harness.tempDir,
    });
    const otherWriter = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: otherSessionId,
      clock: new ManualClock(1_000_050),
      entryIndex: sharedEntryIndex,
    });

    try {
      const localThought = await harness.streamWriter.append({
        kind: "thought",
        content: "本地 thought — dokładnie",
        turn_id: "turn-local-thought",
      });
      const crossSessionThought = await otherWriter.append({
        kind: "thought",
        content: "عبر جلسة أخرى",
        turn_id: "turn-cross-session-thought",
      });
      const dispatcher = createHarnessToolDispatcher(harness);
      const journalResult = await dispatcher.dispatch({
        toolName: "tool.journal.append",
        input: { text: "Private journal exact payload" },
        origin: "autonomous",
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-journal-anchor",
      });

      expect(journalResult.ok).toBe(true);

      const globalResult = await dispatcher.dispatch({
        toolName: "tool.ownRecords.list",
        input: {
          since: new Date(999_000).toISOString(),
          until: new Date(1_001_000).toISOString(),
        },
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
      });

      expect(globalResult.ok).toBe(true);
      if (!globalResult.ok) {
        throw new Error(globalResult.error);
      }

      const globalRecords = (
        globalResult.output as {
          records: Array<{
            kind: string;
            content: string | null;
            session_id: string | null;
            stream_entry_id: string | null;
            disclosure_label: { disclosure_class: string };
          }>;
        }
      ).records;

      expect(globalRecords.map((record) => record.kind)).toEqual(["journal", "thought", "thought"]);
      expect(globalRecords.map((record) => record.content)).toEqual([
        "Private journal exact payload",
        "عبر جلسة أخرى",
        "本地 thought — dokładnie",
      ]);
      expect(globalRecords.map((record) => record.disclosure_label.disclosure_class)).toEqual([
        "self_private",
        "self_private",
        "self_private",
      ]);
      expect(globalRecords.map((record) => record.stream_entry_id)).toEqual([
        null,
        crossSessionThought.id,
        localThought.id,
      ]);

      const filteredResult = await dispatcher.dispatch({
        toolName: "tool.ownRecords.list",
        input: {
          since: new Date(999_000).toISOString(),
          until: new Date(1_001_000).toISOString(),
          session_id: otherSessionId,
        },
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
      });

      expect(filteredResult.ok).toBe(true);
      if (!filteredResult.ok) {
        throw new Error(filteredResult.error);
      }
      expect(
        (
          filteredResult.output as {
            records: Array<{ stream_entry_id: string | null; session_id: string | null }>;
          }
        ).records,
      ).toEqual([
        expect.objectContaining({
          stream_entry_id: crossSessionThought.id,
          session_id: otherSessionId,
        }),
      ]);
    } finally {
      otherWriter.close();
      await harness.cleanup();
    }
  });

  // Replaces the old repository regression that accepted dangling citations.
  // The composed tool owns reference validation before identity governance writes.
  it.each(["episode", "stream"] as const)(
    "returns an unresolvable %s citation as a structural tool error",
    async (kind) => {
      const harness = await createOfflineTestHarness();
      try {
        const question = harness.openQuestionsRepository.add({
          question: "What does the record establish?",
          urgency: 0.4,
          source: "reflection",
          provenance: { kind: "manual" },
        });
        const missingId = kind === "episode" ? createEpisodeId() : createStreamEntryId();
        const evidence =
          kind === "episode"
            ? { resolution_evidence_episode_ids: [missingId] }
            : { resolution_evidence_stream_entry_ids: [missingId] };
        const dispatcher = createHarnessToolDispatcher(harness);
        const result = await dispatcher.dispatch({
          toolName: "tool.openQuestions.resolve",
          input: { open_question_id: question.id, resolution_note: "My conclusion.", ...evidence },
          origin: "autonomous",
        });

        expect(result).toMatchObject({ ok: false, error: expect.stringContaining(missingId) });
        expect(result).toMatchObject({
          error: expect.stringContaining("do not name stored records"),
        });
        expect(harness.openQuestionsRepository.get(question.id)).toEqual(question);
      } finally {
        await harness.cleanup();
      }
    },
  );

  it("accepts archived episodes and inactive stream references without treating existence as disclosure permission", async () => {
    const harness = await createOfflineTestHarness();
    try {
      const audience = createEntityId();
      const episode = createEpisodeFixture({
        audience_entity_id: audience,
        origin_audience_entity_ids: [audience],
        shared: false,
      });
      await harness.episodicRepository.createEpisode(episode);
      harness.episodicRepository.archiveEpisode(episode.id, {
        caller: "internal-tools.test",
        reason: "Archived references still exist.",
        process: "curator",
      });
      expect(await harness.episodicRepository.get(episode.id)).toBeNull();
      const entry = await harness.streamWriter.append({
        kind: "thought",
        content: "An earlier thought.",
      });
      harness.db
        .prepare("UPDATE stream_entry_index SET active = 0 WHERE entry_id = ?")
        .run(entry.id);
      const dispatcher = createHarnessToolDispatcher(harness);

      for (const [evidence, disclosureClass] of [
        [{ resolution_evidence_episode_ids: [episode.id] }, "relationship_private"],
        [{ resolution_evidence_stream_entry_ids: [entry.id] }, "unknown"],
      ] as const) {
        const question = harness.openQuestionsRepository.add({
          question: `Can I cite the ${disclosureClass} record?`,
          urgency: 0.4,
          source: "reflection",
          provenance: { kind: "manual" },
        });
        const missingId = createEpisodeId();
        const rejected = await dispatcher.dispatch({
          toolName: "tool.openQuestions.resolve",
          input: {
            open_question_id: question.id,
            resolution_note: "A mixed set of references.",
            ...evidence,
            resolution_evidence_episode_ids: [episode.id, missingId],
          },
          origin: "autonomous",
        });
        expect(rejected).toMatchObject({ ok: false, error: expect.stringContaining(missingId) });
        expect(harness.openQuestionsRepository.get(question.id)?.status).toBe("open");

        const result = await dispatcher.dispatch({
          toolName: "tool.openQuestions.resolve",
          input: {
            open_question_id: question.id,
            resolution_note: "Re-cited from the stored record.",
            ...evidence,
          },
          origin: "autonomous",
          audienceEntityId: createEntityId(),
        });
        expect(result).toMatchObject({
          ok: true,
          output: {
            status: "applied",
            openQuestion: {
              status: "resolved",
              resolution_disclosure_label: { disclosureClass },
            },
          },
        });
      }
    } finally {
      await harness.cleanup();
    }
  });

  it("rereads goal dispositions in prose alongside thoughts and journal with stable tied-time pagination", async () => {
    const clock = new ManualClock(1_000_100);
    const harness = await createOfflineTestHarness({ clock });
    try {
      const dispatcher = createHarnessToolDispatcher(harness);
      const goal = harness.goalsRepository.add({
        description: "检查计划 — sprawdzić plan",
        priority: 5,
        audienceEntityId: createEntityId(),
        provenance: { kind: "manual" },
      });
      await harness.streamWriter.append({ kind: "thought", content: "Earlier thought." });
      expect(
        await dispatcher.dispatch({
          toolName: "tool.journal.append",
          input: { text: "A separate journal entry." },
          origin: "autonomous",
        }),
      ).toMatchObject({ ok: true });

      const reasons = ["试过了，还不可用。", "Dostęp przywrócony.", `Zakończone: ${goal.id}`];
      for (const [toolName, input] of [
        [
          "tool.goals.block",
          {
            goal_id: goal.id,
            blocker: { kind: "until", until: clock.now() + 50_000 },
            attempt_status: "attempted_unavailable",
            reason: reasons[0],
          },
        ],
        ["tool.goals.unblock", { goal_id: goal.id, reason: reasons[1] }],
        ["tool.goals.retire", { goal_id: goal.id, reason: reasons[2] }],
      ] as const) {
        expect(await dispatcher.dispatch({ toolName, input, origin: "autonomous" })).toMatchObject({
          ok: true,
        });
      }
      // A no-op and unrelated same-status update must not masquerade as dispositions.
      await dispatcher.dispatch({
        toolName: "tool.goals.retire",
        input: { goal_id: goal.id, reason: "No-op." },
        origin: "autonomous",
      });
      harness.goalsRepository.update(
        goal.id,
        { description: "Later description" },
        { kind: "manual" },
      );
      const range = {
        since: new Date(clock.now()).toISOString(),
        until: new Date(clock.now()).toISOString(),
      };
      const records: Array<{
        kind: string;
        content: string;
        occurred_at: number;
        origin_time_basis: string;
        disclosure_label: { disclosure_class: string };
      }> = [];
      let cursor: string | undefined;
      for (let pageNumber = 0; pageNumber < 6; pageNumber += 1) {
        const page = await dispatcher.dispatch({
          toolName: "tool.ownRecords.list",
          input: { ...range, limit: 1, ...(cursor === undefined ? {} : { cursor }) },
          origin: "deliberator",
          audienceEntityId: createEntityId(),
        });
        if (!page.ok) throw new Error(page.error);
        const output = page.output as {
          records: typeof records;
          has_more: boolean;
          next_cursor: string | null;
        };
        records.push(...output.records);
        cursor = output.next_cursor ?? undefined;
        if (!output.has_more) break;
      }
      expect(cursor).toBeUndefined();
      expect(records.map((record) => record.kind)).toEqual([
        "thought",
        "journal",
        "goal_disposition",
        "goal_disposition",
        "goal_disposition",
      ]);
      const dispositions = records.filter((record) => record.kind === "goal_disposition");
      expect(dispositions.map((record) => record.content)).toEqual([
        `At ${range.since}, the goal “${goal.description}” was retired. Reason: Zakończone: [internal_id]`,
        `At ${range.since}, the goal “${goal.description}” was unblocked. Reason: ${reasons[1]}`,
        `At ${range.since}, the goal “${goal.description}” was blocked. Reason: ${reasons[0]}`,
      ]);
      expect(JSON.stringify(dispositions)).not.toContain(goal.id);
      expect(
        dispositions.every(
          (record) =>
            record.occurred_at === clock.now() &&
            record.origin_time_basis === "identity_event_timestamp",
        ),
      ).toBe(true);
      expect(
        dispositions.every((record) => record.disclosure_label.disclosure_class !== "public"),
      ).toBe(true);
      expect(
        harness.db.prepare("SELECT COUNT(*) AS count FROM train_of_thought_journal_entries").get(),
      ).toMatchObject({ count: 1 });

      const onlyDispositions = await dispatcher.dispatch({
        toolName: "tool.ownRecords.list",
        input: { ...range, kinds: ["goal_disposition"], session_id: createSessionId() },
        origin: "deliberator",
      });
      expect(onlyDispositions).toMatchObject({
        ok: true,
        output: { records: dispositions, has_more: false },
      });
      const outsideRange = await dispatcher.dispatch({
        toolName: "tool.ownRecords.list",
        input: {
          since: new Date(clock.now() + 1).toISOString(),
          until: new Date(clock.now() + 10).toISOString(),
          kinds: ["goal_disposition"],
        },
        origin: "deliberator",
      });
      expect(outsideRange).toMatchObject({ ok: true, output: { records: [], has_more: false } });
    } finally {
      await harness.cleanup();
    }
  });

  it("rereads an automatic unblock with both its effective and recorded time", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    try {
      const goal = harness.goalsRepository.add({
        description: "Wait for the scheduled opening",
        priority: 5,
        provenance: { kind: "manual" },
      });
      const dispatcher = createHarnessToolDispatcher(harness);
      const effectiveAt = clock.now() + 50;
      expect(
        await dispatcher.dispatch({
          toolName: "tool.goals.block",
          input: {
            goal_id: goal.id,
            blocker: { kind: "until", until: effectiveAt },
            attempt_status: "attempted_unavailable",
            reason: "Not available yet.",
          },
          origin: "autonomous",
        }),
      ).toMatchObject({ ok: true });
      clock.advance(100);
      harness.goalsRepository.reconcileBlocks();

      const result = await dispatcher.dispatch({
        toolName: "tool.ownRecords.list",
        input: {
          since: new Date(clock.now()).toISOString(),
          until: new Date(clock.now()).toISOString(),
          kinds: ["goal_disposition"],
        },
        origin: "deliberator",
      });
      expect(result).toMatchObject({
        ok: true,
        output: {
          has_more: false,
          records: [
            {
              kind: "goal_disposition",
              occurred_at: clock.now(),
              content: `At ${new Date(clock.now()).toISOString()}, the goal “${goal.description}” was unblocked. Effective at ${new Date(effectiveAt).toISOString()}. Reason: until timestamp ${effectiveAt} passed; observed at ${clock.now()}`,
              disclosure_label: { disclosure_class: "unknown" },
            },
          ],
        },
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("bounds oversized disposition disclosure labels and still advances the cursor", async () => {
    const clock = new ManualClock(1_000_100);
    const harness = await createOfflineTestHarness({ clock });
    try {
      const dispatcher = createHarnessToolDispatcher(harness);
      const earlier = harness.goalsRepository.add({
        description: "Earlier goal",
        priority: 5,
        provenance: { kind: "manual" },
      });
      harness.goalsRepository.retire(earlier.id, "Finished earlier.", { kind: "manual" });
      const audience = Array.from({ length: 2_000 }, () => createEntityId());
      const episode = createEpisodeFixture({
        audience_entity_id: audience[0]!,
        origin_audience_entity_ids: audience,
        shared: false,
      });
      await harness.episodicRepository.createEpisode(episode);
      const goal = harness.goalsRepository.add({
        description: "Goal with a large source audience",
        priority: 5,
        provenance: { kind: "episodes", episode_ids: [episode.id] },
      });
      expect(
        await dispatcher.dispatch({
          toolName: "tool.goals.retire",
          input: { goal_id: goal.id, reason: "Finished." },
          origin: "autonomous",
        }),
      ).toMatchObject({ ok: true });
      const range = {
        since: new Date(clock.now()).toISOString(),
        until: new Date(clock.now()).toISOString(),
        kinds: ["goal_disposition"],
        limit: 1,
      };
      const first = await dispatcher.dispatch({
        toolName: "tool.ownRecords.list",
        input: range,
        origin: "deliberator",
      });
      if (!first.ok) throw new Error(first.error);
      expect(estimatePromptTokens(JSON.stringify(first.output))).toBeLessThanOrEqual(
        DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET,
      );
      expect(first.output).toMatchObject({
        records: [
          {
            kind: "goal_disposition",
            content: null,
            payload_status: "check_not_completed_budget",
            oversized_anchors_omitted: true,
            disclosure_label: {
              disclosure_class: "unknown",
              origin_audience_entity_ids: [],
              private_to_entity_ids: [],
            },
          },
        ],
        has_more: true,
        page_end_reason: "context_budget",
        next_cursor: expect.any(String),
      });
      expect(JSON.stringify(first.output)).not.toContain(audience[0]);
      const second = await dispatcher.dispatch({
        toolName: "tool.ownRecords.list",
        input: { ...range, cursor: (first.output as { next_cursor: string }).next_cursor },
        origin: "deliberator",
      });
      expect(second).toMatchObject({
        ok: true,
        output: {
          records: [
            { content: expect.stringContaining(earlier.description), payload_status: "exact" },
          ],
          has_more: false,
          next_cursor: null,
        },
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("scrubs scheduled-wake, attachment, and Unicode-adjacent goal ids from disposition prose", async () => {
    const clock = new ManualClock(1_000_100);
    const harness = await createOfflineTestHarness({ clock });
    try {
      const dispatcher = createHarnessToolDispatcher(harness);
      const attachmentId = createAttachmentId();
      const wakeId = createScheduledWakeId();
      const goal = harness.goalsRepository.add({
        description: `检查${attachmentId}文档`,
        priority: 5,
        provenance: { kind: "manual" },
      });
      expect(
        await dispatcher.dispatch({
          toolName: "tool.goals.retire",
          input: {
            goal_id: goal.id,
            reason: `已完成${goal.id}，无需继续；${wakeId}已处理。保留8和1000100。`,
          },
          origin: "autonomous",
        }),
      ).toMatchObject({ ok: true });
      const result = await dispatcher.dispatch({
        toolName: "tool.ownRecords.list",
        input: {
          since: new Date(clock.now()).toISOString(),
          until: new Date(clock.now()).toISOString(),
          kinds: ["goal_disposition"],
        },
        origin: "deliberator",
      });
      expect(result).toMatchObject({
        ok: true,
        output: {
          records: [
            {
              content: `At ${new Date(clock.now()).toISOString()}, the goal “检查[internal_id]文档” was retired. Reason: 已完成[internal_id]，无需继续；[internal_id]已处理。保留8和1000100。`,
            },
          ],
        },
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("renders automatic dependency audit references without hiding ordinary numbers or times", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    try {
      const dispatcher = createHarnessToolDispatcher(harness);
      const dependency = harness.goalsRepository.add({
        description: "Prepare 8 checks",
        priority: 5,
        provenance: { kind: "manual" },
      });
      const goal = harness.goalsRepository.add({
        description: "Complete 8 checks at 1000100",
        priority: 5,
        provenance: { kind: "manual" },
      });
      harness.goalsRepository.block(
        goal.id,
        {
          blocker: { kind: "goal", goal_id: dependency.id },
          attempt_status: "attempted_unavailable",
          reason: "Tried 8 times at 1000000.",
        },
        { kind: "manual" },
      );
      clock.advance(100);
      harness.goalsRepository.retire(dependency.id, "After 8 checks at 1000100.", {
        kind: "manual",
      });
      const event = harness.identityEventRepository
        .listGoalDispositionsInRange({
          sinceMs: clock.now(),
          untilMs: clock.now(),
          limit: 10,
        })
        .find((entry) => entry.action === "unblock");
      expect(event?.reason).toMatch(/identity event \d+ at 1000100/);
      const result = await dispatcher.dispatch({
        toolName: "tool.ownRecords.list",
        input: {
          since: new Date(clock.now()).toISOString(),
          until: new Date(clock.now()).toISOString(),
          kinds: ["goal_disposition"],
        },
        origin: "deliberator",
      });
      expect(result).toMatchObject({
        ok: true,
        output: {
          records: [
            {
              content: `At ${new Date(clock.now()).toISOString()}, the goal “${goal.description}” was unblocked. Reason: blocker goal [internal_id] is abandoned; identity event [internal_id] at 1000100; observed at 1000100`,
            },
            {
              content: `At ${new Date(clock.now()).toISOString()}, the goal “${dependency.description}” was retired. Reason: After 8 checks at 1000100.`,
            },
          ],
        },
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("lists active commitments", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openTestBorg(tempDir);

    try {
      const commitment = borg.commitments.add({
        type: "promise",
        directiveFamily: "sprint10_autonomy_followup",
        directive: "Follow up on Sprint 10 autonomy work",
        priority: 8,
        provenance: { kind: "manual" },
      });

      const tool = createCommitmentsListTool({
        listCommitments: () =>
          borg.commitments.list({
            activeOnly: true,
          }),
      });
      const result = await tool.invoke(
        {},
        {
          sessionId: DEFAULT_SESSION_ID,
          origin: "autonomous",
        },
      );

      expect(result.commitments.map((item) => item.id)).toContain(commitment.id);
      expect(result.commitments[0]?.disclosure_label).toMatchObject({
        disclosure_class: "unknown",
      });
    } finally {
      await borg.close();
    }
  });

  it("lists active commitments globally for cognition with disclosure labels", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const sam = harness.entityRepository.resolve("Sam");
      const alex = harness.entityRepository.resolve("Alex");
      const publicCommitment = harness.commitmentRepository.add({
        type: "promise",
        directiveFamily: "public_planning_followup",
        directive: "Follow up on public planning",
        priority: 5,
        provenance: { kind: "manual" },
      });
      const samCommitment = harness.commitmentRepository.add({
        type: "boundary",
        directiveFamily: "sam_planning_scope",
        directive: "Keep Sam planning details scoped to Sam",
        priority: 10,
        restrictedAudience: sam,
        provenance: { kind: "manual" },
      });
      const alexCommitment = harness.commitmentRepository.add({
        type: "boundary",
        directiveFamily: "alex_planning_scope",
        directive: "Keep Alex planning details scoped to Alex",
        priority: 10,
        restrictedAudience: alex,
        provenance: { kind: "manual" },
      });
      const dispatcher = createHarnessToolDispatcher(harness);

      const defaultResult = await dispatcher.dispatch({
        toolName: "tool.commitments.list",
        input: {},
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: null,
      });

      expect(defaultResult.ok).toBe(true);
      if (!defaultResult.ok) {
        throw new Error(defaultResult.error);
      }
      expect(
        (defaultResult.output as { commitments: Array<{ id: string }> }).commitments.map(
          (item) => item.id,
        ),
      ).toEqual([samCommitment.id, alexCommitment.id, publicCommitment.id]);

      const samResult = await dispatcher.dispatch({
        toolName: "tool.commitments.list",
        input: {},
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: sam,
      });

      expect(samResult.ok).toBe(true);
      if (!samResult.ok) {
        throw new Error(samResult.error);
      }

      const samIds = (samResult.output as { commitments: Array<{ id: string }> }).commitments.map(
        (item) => item.id,
      );
      expect(samIds).toContain(publicCommitment.id);
      expect(samIds).toContain(samCommitment.id);
      expect(samIds).toContain(alexCommitment.id);
      expect(
        (
          samResult.output as {
            commitments: Array<{
              id: string;
              disclosure_label: {
                disclosure_class: string;
                private_to_entity_ids: string[];
              };
            }>;
          }
        ).commitments.find((item) => item.id === alexCommitment.id)?.disclosure_label,
      ).toMatchObject({
        disclosure_class: "relationship_private",
        private_to_entity_ids: [alex],
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("creates open questions with autonomy provenance", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openTestBorg(tempDir);

    try {
      const tool = createOpenQuestionsCreateTool({
        createOpenQuestion: (input) => borg.self.openQuestions.add(input),
      });
      const result = await tool.invoke(
        {
          question: "Should I revisit the autonomy scheduler cadence?",
        },
        {
          sessionId: DEFAULT_SESSION_ID,
          origin: "autonomous",
        },
      );

      expect(result.openQuestion.source).toBe("autonomy");
      expect(result.openQuestion.question).toContain("scheduler cadence");
      expect(result.openQuestion).toMatchObject({
        disclosure: expect.stringContaining("disclosure_class=self_private"),
        disclosure_label: {
          disclosure_class: "self_private",
        },
      });
    } finally {
      await borg.close();
    }
  });

  it("creates open questions with deliberator provenance", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openTestBorg(tempDir);

    try {
      const tool = createOpenQuestionsCreateTool({
        createOpenQuestion: (input) => borg.self.openQuestions.add(input),
      });
      const result = await tool.invoke(
        {
          question: "What should I clarify before answering the user?",
        },
        {
          sessionId: DEFAULT_SESSION_ID,
          origin: "deliberator",
        },
      );

      expect(result.openQuestion.source).toBe("deliberator");
      expect(result.openQuestion.question).toContain("clarify before answering");
    } finally {
      await borg.close();
    }
  });

  it("lists identity events for cognition with disclosure labels", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openTestBorg(tempDir);

    try {
      borg.self.values.add({
        label: "clarity",
        description: "Prefer explicit state.",
        priority: 5,
        provenance: {
          kind: "manual",
        },
      });

      const tool = createIdentityEventsListForCognitionTool({
        listEvents: (options) => borg.identity.listEvents(options),
      });
      const result = await tool.invoke(
        {
          limit: 5,
        },
        {
          sessionId: DEFAULT_SESSION_ID,
          origin: "autonomous",
        },
      );

      const valueEvent = result.events.find((event) => event.record_type === "value");

      expect(valueEvent).toBeDefined();
      expect(valueEvent?.disclosure_label).toMatchObject({
        disclosure_class: "unknown",
      });
    } finally {
      await borg.close();
    }
  });

  it("lists private commitment identity events for cognition with disclosure labels", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const sam = harness.entityRepository.resolve("Sam");
      const alex = harness.entityRepository.resolve("Alex");
      const samCommitment = harness.commitmentRepository.add({
        type: "boundary",
        directiveFamily: "sam_identity_event",
        directive: "Sam identity event",
        priority: 8,
        restrictedAudience: sam,
        provenance: { kind: "manual" },
      });
      const dispatcher = createHarnessToolDispatcher(harness);

      const result = await dispatcher.dispatch({
        toolName: "tool.identityEvents.listForCognition",
        input: {
          recordType: "commitment",
          limit: 10,
        },
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: alex,
      });

      expect(result.ok).toBe(true);
      if (!result.ok) {
        throw new Error(result.error);
      }

      const returnedEvent = (
        result.output as {
          events: Array<{
            record_id: string;
            disclosure_label?: {
              disclosure_class?: string;
              origin_audience_entity_ids?: string[];
              private_to_entity_ids?: string[];
            };
          }>;
        }
      ).events.find((event) => event.record_id === samCommitment.id);

      expect(returnedEvent).toBeDefined();
      expect(returnedEvent?.disclosure_label).toMatchObject({
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [sam],
        private_to_entity_ids: [sam],
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("labels source-backed value and trait identity events for cognition without failing open", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const sam = harness.entityRepository.resolve("Sam");
      const alex = harness.entityRepository.resolve("Alex");
      const privateEpisode = createEpisodeFixture({
        title: "Sam private identity event evidence",
        audience_entity_id: sam,
        origin_audience_entity_ids: [sam],
        shared: false,
      });
      await harness.episodicRepository.createEpisode(privateEpisode);
      const missingEpisodeId = createEpisodeId();

      harness.identityEventRepository.record({
        record_type: "value",
        record_id: "val_private_evidence",
        action: "create",
        old_value: null,
        new_value: {
          id: "val_private_evidence",
          label: "protected value",
          evidence_episode_ids: [privateEpisode.id],
        },
        reason: "private source label fixture",
        provenance: { kind: "manual" },
      });
      harness.identityEventRepository.record({
        record_type: "trait",
        record_id: "trait_unresolved_evidence",
        action: "create",
        old_value: null,
        new_value: {
          id: "trait_unresolved_evidence",
          label: "unresolved source trait",
          evidence_episode_ids: [missingEpisodeId],
        },
        reason: "unresolved source label fixture",
        provenance: { kind: "manual" },
      });
      harness.identityEventRepository.record({
        record_type: "value",
        record_id: "val_shared_evidence_ids",
        action: "create",
        old_value: null,
        new_value: {
          id: "val_shared_evidence_ids",
          label: "shared evidence id fixture",
          evidence_episode_ids: [privateEpisode.id, missingEpisodeId],
        },
        reason: "deduplicated source label fixture",
        provenance: { kind: "manual" },
      });
      const getMany = vi.spyOn(harness.episodicRepository, "getMany");
      const dispatcher = createHarnessToolDispatcher(harness);

      const result = await dispatcher.dispatch({
        toolName: "tool.identityEvents.listForCognition",
        input: {
          limit: 10,
        },
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: alex,
      });

      expect(result.ok).toBe(true);
      if (!result.ok) {
        throw new Error(result.error);
      }

      const eventsByRecordId = new Map(
        (
          result.output as {
            events: Array<{
              record_id: string;
              disclosure_label?: {
                disclosure_class?: string;
                origin_audience_entity_ids?: string[];
                private_to_entity_ids?: string[];
              };
            }>;
          }
        ).events.map((event) => [event.record_id, event]),
      );
      const valueEvent = eventsByRecordId.get("val_private_evidence");
      const traitEvent = eventsByRecordId.get("trait_unresolved_evidence");

      expect(valueEvent?.disclosure_label).toMatchObject({
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [sam],
        private_to_entity_ids: [sam],
      });
      expect(valueEvent?.disclosure_label?.disclosure_class).not.toBe("public");
      expect(traitEvent?.disclosure_label).toMatchObject({
        disclosure_class: "unknown",
      });
      expect(traitEvent?.disclosure_label?.disclosure_class).not.toBe("public");
      expect(getMany).toHaveBeenCalledTimes(1);
      const lookedUpEpisodeIds = getMany.mock.calls[0]?.[0] ?? [];
      expect(lookedUpEpisodeIds).toHaveLength(2);
      expect(new Set(lookedUpEpisodeIds)).toEqual(new Set([privateEpisode.id, missingEpisodeId]));
    } finally {
      await harness.cleanup();
    }
  });

  it("scopes commitment identity events on the disclosure path", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const sam = harness.entityRepository.resolve("Sam");
      const alex = harness.entityRepository.resolve("Alex");
      const publicCommitment = harness.commitmentRepository.add({
        type: "promise",
        directiveFamily: "public_identity_event",
        directive: "Public identity event",
        priority: 5,
        provenance: { kind: "manual" },
      });
      const samCommitment = harness.commitmentRepository.add({
        type: "boundary",
        directiveFamily: "sam_identity_event",
        directive: "Sam identity event",
        priority: 8,
        restrictedAudience: sam,
        provenance: { kind: "manual" },
      });
      const alexCommitment = harness.commitmentRepository.add({
        type: "boundary",
        directiveFamily: "alex_identity_event",
        directive: "Alex identity event",
        priority: 8,
        restrictedAudience: alex,
        provenance: { kind: "manual" },
      });
      const events = harness.identityService.listEventsForDisclosure(
        {
          recordType: "commitment",
          limit: 10,
        },
        sam,
      );

      const recordIds = events.map((event) => event.record_id);
      expect(recordIds).toContain(publicCommitment.id);
      expect(recordIds).toContain(samCommitment.id);
      expect(recordIds).not.toContain(alexCommitment.id);
    } finally {
      await harness.cleanup();
    }
  });

  it("does not expose multi-origin private episode identity-event values on the disclosure path", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const alice = createEntityId();
      const bob = createEntityId();
      const carol = createEntityId();
      const episode = createEpisodeFixture({
        title: "Alice Bob private correction target",
        audience_entity_id: null,
        origin_audience_entity_ids: [alice, bob],
        shared: false,
      });
      const nestedEpisode = createEpisodeFixture({
        title: "Nested Alice Bob private correction target",
        audience_entity_id: null,
        origin_audience_entity_ids: [alice, bob],
        shared: false,
      });

      harness.identityEventRepository.record({
        record_type: "episode",
        record_id: episode.id,
        action: "correction_apply",
        old_value: null,
        new_value: {
          id: episode.id,
          title: episode.title,
          audience_entity_id: episode.audience_entity_id ?? null,
          origin_audience_entity_ids: episode.origin_audience_entity_ids ?? [],
          shared: episode.shared ?? false,
        },
        reason: "test correction",
        provenance: { kind: "manual" },
      });
      harness.identityEventRepository.record({
        record_type: "episode",
        record_id: nestedEpisode.id,
        action: "forget",
        old_value: {
          episode: {
            id: nestedEpisode.id,
            title: nestedEpisode.title,
            audience_entity_id: nestedEpisode.audience_entity_id ?? null,
            origin_audience_entity_ids: nestedEpisode.origin_audience_entity_ids ?? [],
            shared: nestedEpisode.shared ?? false,
          },
          stats: {
            archived: false,
          },
        },
        new_value: {
          episode: {
            id: nestedEpisode.id,
            title: nestedEpisode.title,
            audience_entity_id: nestedEpisode.audience_entity_id ?? null,
            origin_audience_entity_ids: nestedEpisode.origin_audience_entity_ids ?? [],
            shared: nestedEpisode.shared ?? false,
          },
          stats: {
            archived: true,
          },
        },
        reason: "test nested correction",
        provenance: { kind: "manual" },
      });

      const carolEvents = harness.identityService.listEventsForDisclosure(
        {
          recordType: "episode",
          limit: 10,
        },
        carol,
      );
      const aliceEvents = harness.identityService.listEventsForDisclosure(
        {
          recordType: "episode",
          limit: 10,
        },
        alice,
      );

      const carolRecordIds = carolEvents.map((event) => event.record_id);
      const aliceRecordIds = aliceEvents.map((event) => event.record_id);

      expect(carolRecordIds).not.toContain(episode.id);
      expect(carolRecordIds).not.toContain(nestedEpisode.id);
      expect(aliceRecordIds).toContain(episode.id);
      expect(aliceRecordIds).toContain(nestedEpisode.id);
    } finally {
      await harness.cleanup();
    }
  });

  it("lists procedural skills", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openTestBorg(tempDir);

    try {
      const sourceEpisode = createEpisodeId();
      const skill = await borg.skills.add({
        applies_when: "debugging pgvector similarity drift after rollback",
        approach: "Verify dimensions, compare operator class, then rebuild the index safely.",
        sourceEpisodes: [sourceEpisode],
      });

      const tool = createSkillsListTool({
        listSkills: (limit) => borg.skills.list(limit),
        listContextStatsForSkill: (skillId) =>
          skillId === skill.id
            ? [
                {
                  skill_id: skill.id,
                  context_key: TYPESCRIPT_DEBUG_CONTEXT_KEY,
                  alpha: 3,
                  beta: 1,
                  attempts: 2,
                  successes: 2,
                  failures: 0,
                  last_used: 1_000,
                  last_successful: 1_000,
                  updated_at: 1_000,
                },
              ]
            : [],
      });
      const result = await tool.invoke(
        {
          limit: 5,
        },
        {
          sessionId: DEFAULT_SESSION_ID,
          origin: "deliberator",
        },
      );

      expect(result.skills.map((item) => item.id)).toContain(skill.id);
      expect(result.skills.find((item) => item.id === skill.id)).not.toHaveProperty(
        "source_episode_ids",
      );
      expect(result.skills.find((item) => item.id === skill.id)).toMatchObject({
        disclosure: expect.stringContaining("disclosure_class=unknown"),
        disclosure_label: {
          disclosure_class: "unknown",
          origin_audience_entity_ids: [],
          private_to_entity_ids: [],
          public_to_entity_ids: [],
        },
      });
      expect(result.context_stats_by_skill_id?.[skill.id]).toEqual([
        expect.objectContaining({
          context_key: TYPESCRIPT_DEBUG_CONTEXT_KEY,
          attempts: 2,
        }),
      ]);
    } finally {
      await borg.close();
    }
  });

  it("returns an empty skills list when the registry is empty", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openTestBorg(tempDir);

    try {
      const tool = createSkillsListTool({
        listSkills: (limit) => borg.skills.list(limit),
      });
      const result = await tool.invoke(
        {},
        {
          sessionId: DEFAULT_SESSION_ID,
          origin: "deliberator",
        },
      );

      expect(result.skills).toEqual([]);
    } finally {
      await borg.close();
    }
  });
});
