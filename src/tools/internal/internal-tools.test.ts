import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  Borg,
  DEFAULT_SESSION_ID,
  FakeLLMClient,
  ManualClock,
  StreamWriter,
  ToolDispatcher,
  createEpisodeId,
  createCommitmentsListTool,
  createEpisodicSearchTool,
  createIdentityEventsListTool,
  createOpenQuestionsCreateTool,
  createSemanticWalkTool,
  createSkillsListTool,
} from "../../index.js";
import { TestEmbeddingClient } from "../../offline/test-support.js";

async function openTestBorg(tempDir: string, llm = new FakeLLMClient()) {
  return Borg.open({
    config: {
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
    },
    clock: new ManualClock(1_000_000),
    embeddingDimensions: 4,
    embeddingClient: new TestEmbeddingClient(),
    llmClient: llm,
    liveExtraction: false,
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
          limit: 10,
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
