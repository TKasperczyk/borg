import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { Borg, DEFAULT_SESSION_ID, FakeLLMClient } from "../../index.js";
import { buildToolDispatcher } from "../../borg/tools-setup.js";
import { SemanticGraph } from "../../memory/semantic/index.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
  createTestConfig,
  TestEmbeddingClient,
} from "../../offline/test-support.js";
import { StreamWriter } from "../../stream/index.js";
import {
  ToolDispatcher,
  createCommitmentsListTool,
  createEpisodicSearchTool,
  createIdentityEventsListTool,
  createOpenQuestionsCreateTool,
  createSemanticWalkTool,
  createSkillsListTool,
} from "../../tools/index.js";
import { ManualClock } from "../../util/clock.js";
import { createEpisodeId, createSemanticNodeId } from "../../util/ids.js";

async function openTestBorg(tempDir: string, llm = new FakeLLMClient()) {
  return Borg.open({
    config: createTestConfig({
      dataDir: tempDir,
      perception: {
        useLlmFallback: false,
        modeWhenLlmAbsent: "problem_solving",
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

  return buildToolDispatcher({
    retrievalPipeline: harness.retrievalPipeline,
    episodicRepository: harness.episodicRepository,
    semanticNodeRepository: harness.semanticNodeRepository,
    semanticGraph,
    commitmentRepository: harness.commitmentRepository,
    identityService: harness.identityService,
    skillRepository: harness.skillRepository,
    createStreamWriter: (sessionId) =>
      new StreamWriter({
        dataDir: harness.tempDir,
        sessionId,
        clock,
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
    const episode = createEpisodeFixture({
      title: "Retrieval evidence review",
      narrative: longNarrative,
      participants: ["Ari", "Sam"],
      tags: ["retrieval", "tools"],
      start_time: 1_700_000,
      end_time: 1_701_000,
    });
    const tool = createEpisodicSearchTool({
      searchEpisodes: async (_query, limit, _context) => {
        expect(limit).toBe(5);

        return [
          {
            episode,
            score: 0.82,
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
        }),
      ],
    });
    expect(result.episodes[0]?.narrative.length).toBeLessThanOrEqual(400);
    expect(result.episodes[0]?.narrative).toContain("The team traced the retrieval path");
  });

  it("scopes episodic search to the invocation audience", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm);

    try {
      const aliceEntry = await borg.stream.append({
        kind: "user_msg",
        content: "Alice private planning note about the roadmap.",
        audience: "Alice",
      });
      const bobEntry = await borg.stream.append({
        kind: "user_msg",
        content: "Bob private planning note about the roadmap.",
        audience: "Bob",
      });
      llm.pushResponse({
        text: "",
        input_tokens: 10,
        output_tokens: 5,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "toolu_extract_3",
            name: "EmitEpisodeCandidates",
            input: {
              episodes: [
                {
                  title: "Alice private planning",
                  narrative: "Alice discussed a private roadmap planning note.",
                  source_stream_ids: [aliceEntry.id],
                  participants: ["Alice"],
                  location: null,
                  tags: ["planning", "roadmap"],
                  confidence: 0.8,
                  significance: 0.8,
                },
                {
                  title: "Bob private planning",
                  narrative: "Bob discussed a private roadmap planning note.",
                  source_stream_ids: [bobEntry.id],
                  participants: ["Bob"],
                  location: null,
                  tags: ["planning", "roadmap"],
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

      const episodes = await borg.episodic.list({ limit: 10 });
      const aliceEpisode = episodes.items.find(
        (episode) => episode.title === "Alice private planning",
      );
      if (aliceEpisode === undefined || aliceEpisode.audience_entity_id === null) {
        throw new Error("Expected an Alice-scoped episode");
      }

      const clock = new ManualClock(1_000_100);
      const dispatcher = new ToolDispatcher({
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: tempDir,
            sessionId,
            clock,
          }),
      });
      dispatcher.register(
        createEpisodicSearchTool({
          searchEpisodes: (query, limit, context) =>
            borg.episodic.search(query, {
              limit,
              audienceEntityId: context.audienceEntityId,
            }),
        }),
      );

      const result = await dispatcher.dispatch({
        toolName: "tool.episodic.search",
        input: {
          query: "planning roadmap",
          limit: 5,
        },
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: aliceEpisode.audience_entity_id,
      });

      expect(result.ok).toBe(true);
      if (!result.ok) {
        throw new Error(result.error);
      }
      const output = result.output as {
        episodes: Array<{ title: string }>;
      };
      const titles = output.episodes.map((episode) => episode.title);
      expect(titles).toContain("Alice private planning");
      expect(titles).not.toContain("Bob private planning");
    } finally {
      await borg.close();
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

  it("scopes semantic walks to the invocation audience", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const alice = harness.entityRepository.resolve("Alice");
      const bob = harness.entityRepository.resolve("Bob");
      const publicEpisode = await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Public semantic root",
          narrative: "Public evidence anchors the root node.",
        }),
      );
      const aliceEpisode = await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          title: "Alice semantic support",
          narrative: "Alice-only evidence supports the root node.",
          audience_entity_id: alice,
          shared: false,
        }),
      );
      const bobEpisode = await harness.episodicRepository.insert(
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
        steps: Array<{ node: { id: string; label: string } }>;
      };
      expect(output.steps.map((step) => step.node.id)).toContain(aliceNode.id);
      expect(output.steps.map((step) => step.node.id)).not.toContain(bobNode.id);
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
    } finally {
      await borg.close();
    }
  });

  it("scopes commitment lists to the invocation audience", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const sam = harness.entityRepository.resolve("Sam");
      const alex = harness.entityRepository.resolve("Alex");
      const publicCommitment = harness.commitmentRepository.add({
        type: "promise",
        directive: "Follow up on public planning",
        priority: 5,
        provenance: { kind: "manual" },
      });
      const samCommitment = harness.commitmentRepository.add({
        type: "boundary",
        directive: "Keep Sam planning details scoped to Sam",
        priority: 10,
        restrictedAudience: sam,
        provenance: { kind: "manual" },
      });
      const alexCommitment = harness.commitmentRepository.add({
        type: "boundary",
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
      ).toEqual([publicCommitment.id]);

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
      expect(samIds).not.toContain(alexCommitment.id);
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

  it("lists identity events", async () => {
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

      const tool = createIdentityEventsListTool({
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

      expect(result.events.some((event) => event.record_type === "value")).toBe(true);
    } finally {
      await borg.close();
    }
  });

  it("scopes commitment identity events to the invocation audience", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const sam = harness.entityRepository.resolve("Sam");
      const alex = harness.entityRepository.resolve("Alex");
      const publicCommitment = harness.commitmentRepository.add({
        type: "promise",
        directive: "Public identity event",
        priority: 5,
        provenance: { kind: "manual" },
      });
      const samCommitment = harness.commitmentRepository.add({
        type: "boundary",
        directive: "Sam identity event",
        priority: 8,
        restrictedAudience: sam,
        provenance: { kind: "manual" },
      });
      const alexCommitment = harness.commitmentRepository.add({
        type: "boundary",
        directive: "Alex identity event",
        priority: 8,
        restrictedAudience: alex,
        provenance: { kind: "manual" },
      });
      const dispatcher = createHarnessToolDispatcher(harness);

      const result = await dispatcher.dispatch({
        toolName: "tool.identityEvents.list",
        input: {
          recordType: "commitment",
          limit: 10,
        },
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: sam,
      });

      expect(result.ok).toBe(true);
      if (!result.ok) {
        throw new Error(result.error);
      }

      const recordIds = (result.output as { events: Array<{ record_id: string }> }).events.map(
        (event) => event.record_id,
      );
      expect(recordIds).toContain(publicCommitment.id);
      expect(recordIds).toContain(samCommitment.id);
      expect(recordIds).not.toContain(alexCommitment.id);
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
                  context_key: "code_debugging:typescript:self",
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
      expect(result.context_stats_by_skill_id?.[skill.id]).toEqual([
        expect.objectContaining({
          context_key: "code_debugging:typescript:self",
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
