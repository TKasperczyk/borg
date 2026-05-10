import { afterEach, describe, expect, it, vi } from "vitest";

import {
  Reflector,
  ReflectorOptions,
  FakeLLMClient,
  LLMClient,
  commitmentMigrations,
  episodicMigrations,
  EpisodicRepository,
  createEpisodesTableSchema,
  selfMigrations,
  retrievalMigrations,
  LanceDbStore,
  composeMigrations,
  openDatabase,
  ManualClock,
  createTestConfig,
  Borg,
  EPISODE_TOOL_NAME,
  ScriptedEmbeddingClient,
  borgInternals,
  createEmptyReflectionResponse,
  createGenerationGateResponse,
  createTurnPlanResponse,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

describe("Borg", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("runs the full cognitive turn loop", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(episodicMigrations, selfMigrations, retrievalMigrations),
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock,
    });

    await repo.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Atlas release incident",
      narrative: "Atlas release hit a pnpm failure during deploy.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.8,
      tags: ["atlas", "release"],
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    db.close();
    await store.close();

    const expectedIntent = {
      description: "Follow up on the Atlas deployment after rerunning pnpm install",
      next_action: "rerun the deploy",
    };
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan_1",
              name: "EmitTurnPlan",
              input: {
                uncertainty: "the best rerun order",
                verification_steps: ["check pnpm lockfile"],
                tensions: [],
                voice_note: "",
                referenced_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
                intents: [expectedIntent],
              },
            },
          ],
        },
        {
          text: "To stabilize the Atlas release, rerun pnpm install. Next step: rerun the deploy.",
          input_tokens: 20,
          output_tokens: 10,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_reflection",
              name: "EmitTurnReflection",
              input: {
                advanced_goals: [
                  {
                    goal_id: "goal_aaaaaaaaaaaaaaaa",
                    evidence: "Reran the Atlas release stabilization plan.",
                  },
                ],
                trait_demonstrations: [
                  {
                    trait_label: "engaged",
                    evidence:
                      "The response gave a concrete next action grounded in the Atlas episode.",
                    strength_delta: 0.05,
                  },
                ],
              },
            },
          ],
        },
      ],
    });
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      const goal = borg.self.goals.add({
        id: "goal_aaaaaaaaaaaaaaaa" as never,
        description: "stabilize atlas release",
        priority: 5,
        provenance: { kind: "manual" },
      });
      const result = await borg.turn({
        userMessage: "Project Atlas has a pnpm error and this is high stakes.",
        stakes: "high",
      });

      expect(result.mode).toBe("problem_solving");
      expect(result.path).toBe("system_2");
      expect(result.response).toContain("rerun pnpm install");
      expect(result.retrievedEpisodeIds).toEqual(["ep_aaaaaaaaaaaaaaaa"]);
      expect(result.intents).toEqual([expectedIntent]);
      expect(borg.workmem.load().turn_counter).toBe(1);
      expect(borg.workmem.load().pending_actions).toEqual([
        {
          ...expectedIntent,
          created_at: 1_000,
        },
      ]);
      expect(borg.self.goals.list({ status: "active" })[0]?.id).toBe(goal.id);
      expect(borg.self.goals.list({ status: "active" })[0]?.progress_notes).toContain(
        "Reran the Atlas release stabilization plan.",
      );
      expect(borg.self.goals.list({ status: "active" })[0]?.provenance).toEqual({
        kind: "episodes",
        episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      });
      expect(borg.self.traits.list()).toEqual([]);
      // Sprint 56: trait demonstration is now anchored to the
      // demonstrating turn's stream entries, not arbitrary planner-
      // referenced episodes. The actual stream entry ids are auto-
      // generated; assert their shape and length rather than literal ids.
      const pendingTrait = borg.workmem.load().pending_trait_attribution;
      expect(pendingTrait).toMatchObject({
        trait_label: "engaged",
        audience_entity_id: null,
      });
      expect(pendingTrait?.source_stream_entry_ids).toHaveLength(2);
      // Phase D: the planner's EmitTurnPlan tool-call shows up as a
      // compact "plan: ..." thought entry persisted before the agent_msg.
      expect(borg.stream.tail(4).map((entry) => entry.kind)).toEqual([
        "user_msg",
        "perception",
        "thought",
        "agent_msg",
      ]);
    } finally {
      await borg.close();
    }
  });

  it("does not reinforce a trait when no episodes are retrieved", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan_1",
              name: "EmitTurnPlan",
              input: {
                uncertainty: "the best rerun order",
                verification_steps: ["check pnpm lockfile"],
                tensions: [],
                voice_note: "",
                referenced_episode_ids: [],
                intents: [],
              },
            },
          ],
        },
        {
          text: "Try the deployment again after checking the lockfile.",
          input_tokens: 20,
          output_tokens: 10,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      const result = await borg.turn({
        userMessage: "The deployment is flaky again.",
        stakes: "high",
      });

      expect(result.retrievedEpisodeIds).toEqual([]);
      expect(borg.self.traits.list()).toEqual([]);
      expect(borg.workmem.load().pending_trait_attribution).toBeNull();
    } finally {
      await borg.close();
    }
  });

  it("logs deliberator tool calls between the user and agent messages on a normal turn", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const llm = new FakeLLMClient();
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
      liveExtraction: false,
    });

    try {
      const seedEntry = await borg.stream.append({
        kind: "user_msg",
        content: "planning sync notes",
      });

      llm.pushResponse({
        text: "",
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "toolu_extract_1",
            name: EPISODE_TOOL_NAME,
            input: {
              episodes: [
                {
                  title: "Planning sync",
                  narrative: "The team aligned on the sprint plan and follow-up work.",
                  source_stream_ids: [seedEntry.id],
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
        sinceTs: seedEntry.timestamp,
      });

      llm.pushResponse([
        {
          type: "tool_use",
          id: "toolu_1",
          name: "tool.episodic.search",
          input: {
            query: "planning sync",
          },
        },
      ]);
      llm.pushResponse("I found the planning sync in memory.");
      llm.pushResponse(createEmptyReflectionResponse());

      const result = await borg.turn({
        userMessage: "What do you remember about the planning sync?",
      });

      expect(result.response).toBe("I found the planning sync in memory.");
      expect(result.toolCalls).toMatchObject([
        {
          callId: "toolu_1",
          name: "tool.episodic.search",
          input: {
            query: "planning sync",
          },
          ok: true,
        },
      ]);
      const entries = borg.stream.tail(5);
      expect(entries.map((entry) => entry.kind)).toEqual([
        "user_msg",
        "perception",
        "tool_call",
        "tool_result",
        "agent_msg",
      ]);
      expect(entries[2]?.content).toMatchObject({
        tool_name: "tool.episodic.search",
        origin: "deliberator",
      });
      expect(entries[3]?.content).toMatchObject({
        ok: true,
      });
      expect(entries[4]?.tool_calls).toMatchObject([
        {
          callId: "toolu_1",
          name: "tool.episodic.search",
          input: {
            query: "planning sync",
          },
          ok: true,
        },
      ]);
    } finally {
      await borg.close();
    }
  });

  it("pulls commitments for all perceived entities in a turn", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(episodicMigrations, selfMigrations, retrievalMigrations),
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock,
    });

    await repo.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Atlas and Borealis status",
      narrative: "Atlas and Borealis updates were discussed together.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.9,
      tags: ["atlas", "status"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    db.close();
    await store.close();

    const llm = new FakeLLMClient({
      responses: [
        // S2 planning (Haiku)
        createTurnPlanResponse(["ep_aaaaaaaaaaaaaaaa"]),
        // S2 final (Sonnet) -- refusal-only, judge will find no violations
        {
          text: "I can't discuss Atlas or Borealis with Sam.",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        // Commitment judge: no violations on the refusal-only response
        {
          text: "",
          input_tokens: 8,
          output_tokens: 2,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_judge",
              name: "EmitCommitmentViolations",
              input: { violations: [] },
            },
          ],
        },
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "reflective",
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      borg.commitments.add({
        type: "boundary",
        directiveFamily: "atlas_sam_boundary",
        directive: "Do not discuss Atlas with Sam",
        priority: 10,
        audience: "Sam",
        about: "Atlas",
        provenance: { kind: "manual" },
      });
      borg.commitments.add({
        type: "boundary",
        directiveFamily: "borealis_sam_boundary",
        directive: "Do not discuss Borealis with Sam",
        priority: 9,
        audience: "Sam",
        about: "Borealis",
        provenance: { kind: "manual" },
      });

      const result = await borg.turn({
        userMessage: "Can you update Sam on Atlas and Borealis?",
        audience: "Sam",
      });
      // The commitment judge now uses the background model, so the sonnet
      // request with commitments-awareness is the deliberation response.
      const sonnetRequest = llm.requests.find(
        (request) =>
          request.model === "sonnet" &&
          typeof request.system === "string" &&
          request.system.includes("Commitments you made to this person"),
      );

      expect(sonnetRequest?.system).toContain("Do not discuss Atlas with Sam");
      expect(sonnetRequest?.system).toContain("Do not discuss Borealis with Sam");
      expect(result.response).toContain("can't discuss Atlas or Borealis");
    } finally {
      await borg.close();
    }
  });

  it("uses background for commitment detection and cognition for rewrite through the turn orchestrator", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(episodicMigrations, commitmentMigrations, selfMigrations),
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock,
    });

    await repo.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Atlas status",
      narrative: "Atlas status was discussed.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.9,
      tags: ["atlas", "status"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    db.close();
    await store.close();

    const llm = new FakeLLMClient();
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      const commitment = borg.commitments.add({
        type: "boundary",
        directiveFamily: "atlas_sam_boundary",
        directive: "Do not discuss Atlas with Sam",
        priority: 10,
        audience: "Sam",
        about: "Atlas",
        provenance: { kind: "manual" },
      });
      llm.pushResponse({
        text: "Atlas is down right now.",
        input_tokens: 10,
        output_tokens: 5,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse({
        text: "",
        input_tokens: 8,
        output_tokens: 2,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "toolu_judge_1",
            name: "EmitCommitmentViolations",
            input: {
              violations: [
                {
                  commitment_id: commitment.id,
                  reason: "Discloses Atlas status to Sam",
                  confidence: 0.9,
                },
              ],
            },
          },
        ],
      });
      llm.pushResponse({
        text: "I can't share Atlas details with Sam.",
        input_tokens: 10,
        output_tokens: 5,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse({
        text: "",
        input_tokens: 8,
        output_tokens: 2,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "toolu_judge_2",
            name: "EmitCommitmentViolations",
            input: { violations: [] },
          },
        ],
      });
      llm.pushResponse(createEmptyReflectionResponse());
      const result = await borg.turn({
        userMessage: "Update Sam on Atlas.",
        audience: "Sam",
      });

      expect(result.response).toBe("I can't share Atlas details with Sam.");
      const nonCorrectiveRequests = llm.requests.filter(
        (request) =>
          request.budget !== "corrective-preference-extractor" &&
          request.budget !== "action-state-extractor" &&
          request.budget !== "goal-promotion-extractor" &&
          request.budget !== "frame-anomaly-classifier",
      );
      expect(
        llm.requests.some((request) => request.budget === "corrective-preference-extractor"),
      ).toBe(true);
      expect(llm.requests.some((request) => request.budget === "action-state-extractor")).toBe(
        true,
      );
      expect(llm.requests.some((request) => request.budget === "goal-promotion-extractor")).toBe(
        true,
      );
      expect(llm.requests.some((request) => request.budget === "frame-anomaly-classifier")).toBe(
        true,
      );
      expect(nonCorrectiveRequests.map((request) => request.model)).toEqual([
        "haiku",
        "sonnet",
        "haiku",
        "sonnet",
        "haiku",
        "haiku",
        "haiku",
        "haiku",
        "haiku",
      ]);
      expect(nonCorrectiveRequests[0]?.budget).toBe("procedural-context");
      expect(nonCorrectiveRequests[2]?.budget).toBe("commitment-judge");
      expect(nonCorrectiveRequests[3]?.budget).toBe("commitment-revision");
      expect(nonCorrectiveRequests[4]?.budget).toBe("commitment-judge");
      expect(nonCorrectiveRequests[5]?.budget).toBe("relational-claim-auditor");
      expect(nonCorrectiveRequests[6]?.budget).toBe("closure-response-auditor");
      expect(nonCorrectiveRequests[7]?.budget).toBe("generation-stop-commitment");
      expect(nonCorrectiveRequests[8]?.budget).toBe("reflection");
    } finally {
      await borg.close();
    }
  });

  it("persists suppression across turns and Borg reopen", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(episodicMigrations, selfMigrations, retrievalMigrations),
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock,
    });

    await repo.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Atlas deploy fix",
      narrative: "Rerun pnpm install to recover the Atlas deploy.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.9,
      tags: ["atlas", "deploy"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    await repo.insert({
      id: "ep_bbbbbbbbbbbbbbbb" as never,
      title: "Fallback checklist",
      narrative: "Use the backup recovery checklist if the first fix fails.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_bbbbbbbbbbbbbbbb" as never],
      significance: 0.85,
      tags: ["fallback"],
      confidence: 0.85,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    db.close();
    await store.close();

    const firstBorg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createTurnPlanResponse(["ep_aaaaaaaaaaaaaaaa"]),
          {
            text: "Rerun pnpm install for the Atlas deploy.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createEmptyReflectionResponse(),
        ],
      }),
      liveExtraction: false,
    });

    try {
      const firstResult = await firstBorg.turn({
        userMessage: "Atlas deploy failed with pnpm",
        stakes: "high",
      });

      expect(firstResult.retrievedEpisodeIds[0]).toBe("ep_aaaaaaaaaaaaaaaa");
      expect(firstBorg.workmem.load().suppressed).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "ep_aaaaaaaaaaaaaaaa",
            reason: "already surfaced",
          }),
        ]),
      );
    } finally {
      await firstBorg.close();
    }

    const reopenedBorg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createGenerationGateResponse({
            decision: "proceed",
            substantive: true,
            reason: "The repeated short deploy message is a real request.",
          }),
          {
            text: "Use the rollback fallback.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      liveExtraction: false,
    });

    try {
      const secondResult = await reopenedBorg.turn({
        userMessage: "Atlas deploy failed with pnpm",
      });

      expect(reopenedBorg.workmem.load().suppressed).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "ep_aaaaaaaaaaaaaaaa",
          }),
        ]),
      );
      expect(secondResult.retrievedEpisodeIds).toContain("ep_aaaaaaaaaaaaaaaa");
    } finally {
      await reopenedBorg.close();
    }
  });

  it("rolls back working memory and logs an aborted marker when a turn fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "Check the deploy state before answering.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
    });

    try {
      await expect(
        borg.turn({
          userMessage: "Atlas deploy failed with pnpm and this is high stakes.",
          stakes: "high",
        }),
      ).rejects.toThrow("FakeLLMClient has no scripted response available");

      expect(borg.workmem.load()).toMatchObject({
        turn_counter: 0,
        mode: null,
      });
      const entries = borg.stream.tail(3);

      expect(entries.map((entry) => entry.kind)).toEqual([
        "user_msg",
        "perception",
        "internal_event",
      ]);
      expect(entries[2]).toMatchObject({
        turn_status: "aborted",
        content: expect.objectContaining({
          event: "aborted_turn",
        }),
      });
    } finally {
      await borg.close();
    }
  });

  it("keeps a turn running when the reflection open-question hook fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "reflective",
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "toolu_plan_open_q",
                name: "EmitTurnPlan",
                input: {
                  uncertainty: "why the open-question hook would fire",
                  verification_steps: ["compare Atlas evidence"],
                  tensions: [],
                  voice_note: "",
                  referenced_episode_ids: [],
                  intents: [],
                },
              },
            ],
          },
          {
            text: "I need to compare more evidence before answering.",
            input_tokens: 12,
            output_tokens: 6,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createEmptyReflectionResponse([
            {
              question: "What uncertainty remains about Atlas?",
              urgency: 0.6,
              related_episode_ids: [],
            },
          ]),
        ],
      }),
    });

    try {
      const internal = borgInternals<{
        deps: Pick<
          ReflectorOptions,
          | "episodicRepository"
          | "goalsRepository"
          | "traitsRepository"
          | "reviewQueueRepository"
          | "skillRepository"
          | "proceduralEvidenceRepository"
        > & {
          turnOrchestrator: {
            options: {
              createReflector: (llmClient: LLMClient) => Reflector;
            };
          };
        };
      }>(borg);
      const brokenIdentityService = {
        addOpenQuestion() {
          throw new Error("hook exploded");
        },
        updateGoal() {
          throw new Error("unexpected goal update");
        },
        updateGoalProgressFromReflection() {
          throw new Error("unexpected goal progress update");
        },
        resolveOpenQuestion() {
          throw new Error("unexpected open question resolution");
        },
      };
      internal.deps.turnOrchestrator.options.createReflector = (llmClient) =>
        new Reflector({
          clock,
          llmClient,
          model: "haiku",
          episodicRepository: internal.deps.episodicRepository,
          goalsRepository: internal.deps.goalsRepository,
          traitsRepository: internal.deps.traitsRepository,
          identityService: brokenIdentityService,
          reviewQueueRepository: internal.deps.reviewQueueRepository,
          skillRepository: internal.deps.skillRepository,
          proceduralEvidenceRepository: internal.deps.proceduralEvidenceRepository,
        });

      const result = await borg.turn({
        userMessage: "Why is Atlas still failing?",
        stakes: "high",
      });

      expect(result.path).toBe("system_2");
      expect(result.response).toContain("compare more evidence");
      expect(borg.self.openQuestions.list({ status: "open" })).toEqual([]);
      expect(borg.stream.tail(5).map((entry) => entry.kind)).toEqual([
        "user_msg",
        "perception",
        "thought",
        "agent_msg",
        "internal_event",
      ]);
    } finally {
      await borg.close();
    }
  });
});
