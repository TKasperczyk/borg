import { readFileSync } from "node:fs";

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
  createStreamEntryId,
  Borg,
  EPISODE_TOOL_NAME,
  ScriptedEmbeddingClient,
  borgInternals,
  createEmptyReflectionResponse,
  createEmitAnswerResponse,
  createGenerationGateResponse,
  createTurnPlanResponse,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";
import type { BorgDependencies } from "../types.js";

function createEntityDetectionResponse(entities: string[] = []) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_entity",
        name: "EmitEntityExtraction",
        input: { entities },
      },
    ],
  };
}

function createModeDetectionResponse(
  mode: "problem_solving" | "relational" | "reflective" | "idle",
  isOperational = false,
) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_mode",
        name: "EmitModeDetection",
        input: { mode, is_operational: isOperational },
      },
    ],
  };
}

function createNoTemporalCueResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_temporal",
        name: "EmitTemporalCue",
        input: { has_cue: false },
      },
    ],
  };
}

function requestSystemText(request: { system?: unknown } | undefined): string {
  const system = request?.system;

  if (typeof system === "string") {
    return system;
  }

  if (Array.isArray(system)) {
    return system
      .map((block) =>
        block !== null &&
        typeof block === "object" &&
        "text" in block &&
        typeof block.text === "string"
          ? block.text
          : "",
      )
      .join("\n");
  }

  return "";
}

function extractTaggedPromptBlock(prompt: string, tag: string): string {
  const openTag = `<${tag}>`;
  const closeTag = `</${tag}>`;
  const start = prompt.indexOf(openTag);
  const end = prompt.indexOf(closeTag, start + openTag.length);

  expect(start).toBeGreaterThanOrEqual(0);
  expect(end).toBeGreaterThan(start);

  return prompt.slice(start, end + closeTag.length);
}

function createNoCorrectivePreferenceResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_corrective",
        name: "EmitCorrectivePreference",
        input: {
          classification: "none",
          type: null,
          directive: null,
          directive_family: null,
          closure_pressure_relevance: null,
          priority: null,
          reason: "No durable correction detected.",
          confidence: 0,
          supersedes_commitment_id: null,
          slot_negations: [],
        },
      },
    ],
  };
}

function createNoActionStatesResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_actions",
        name: "EmitActionStates",
        input: { action_states: [] },
      },
    ],
  };
}

function createNoGoalPromotionResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_goals",
        name: "EmitGoalPromotion",
        input: { promotions: [] },
      },
    ],
  };
}

function createSharedStateArtifactPatchResponse(input: { operations: unknown[] }) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_decision_artifact",
        name: "EmitDecisionArtifactPatch",
        input,
      },
    ],
  };
}

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
        createEntityDetectionResponse(["Project Atlas"]),
        createModeDetectionResponse("problem_solving"),
        createNoTemporalCueResponse(),
        createNoCorrectivePreferenceResponse(),
        createNoActionStatesResponse(),
        createNoGoalPromotionResponse(),
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
                intents: [expectedIntent],
              },
            },
          ],
        },
        createEmitAnswerResponse(
          "To stabilize the Atlas release, rerun pnpm install. Next step: rerun the deploy.",
          { inputTokens: 20, outputTokens: 10 },
        ),
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
          llmEnabled: true,
        },
        affective: {
          llmEnabled: false,
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
                intents: [],
              },
            },
          ],
        },
        createEmitAnswerResponse("Try the deployment again after checking the lockfile.", {
          inputTokens: 20,
          outputTokens: 10,
        }),
      ],
    });
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
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
          llmEnabled: false,
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

      llm.pushResponse(createEmitAnswerResponse("I found the planning sync in memory."));
      llm.pushResponse(createEmptyReflectionResponse());

      const result = await borg.turn({
        userMessage: "What do you remember about the planning sync?",
      });

      expect(result.response).toBe("I found the planning sync in memory.");
      expect(result.toolCalls).toEqual([]);
      const entries = borg.stream.tail(3);
      expect(entries.map((entry) => entry.kind)).toEqual(["user_msg", "perception", "agent_msg"]);
      expect(
        entries.some((entry) => entry.kind === "tool_call" || entry.kind === "tool_result"),
      ).toBe(false);
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
        createTurnPlanResponse(),
        // S2 final (Sonnet) -- refusal-only, judge will find no violations
        createEmitAnswerResponse("I can't discuss Atlas or Borealis with Sam.", {
          inputTokens: 10,
          outputTokens: 5,
        }),
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
          llmEnabled: false,
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
          requestSystemText(request).includes(
            "Active commitment / rule / preference / boundary records",
          ),
      );
      const sonnetSystem = requestSystemText(sonnetRequest);

      expect(sonnetSystem).toContain("Do not discuss Atlas with Sam");
      expect(sonnetSystem).toContain("Do not discuss Borealis with Sam");
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
          llmEnabled: true,
        },
        affective: {
          llmEnabled: false,
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
        commitments: {
          enforce: {
            regenerateBeforeSuppress: false,
            rewriteOnViolation: true,
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
        kind: "boundary",
        directiveFamily: "atlas_sam_boundary",
        directive: "Do not discuss Atlas with Sam",
        priority: 10,
        audience: "Sam",
        about: "Atlas",
        provenance: { kind: "manual" },
      });
      llm.pushResponse(createEntityDetectionResponse(["Sam", "Atlas"]));
      llm.pushResponse(createModeDetectionResponse("problem_solving"));
      llm.pushResponse(createNoTemporalCueResponse());
      llm.pushResponse(createNoCorrectivePreferenceResponse());
      llm.pushResponse(createNoActionStatesResponse());
      llm.pushResponse(createNoGoalPromotionResponse());
      llm.pushResponse(
        createEmitAnswerResponse("Atlas is down right now.", {
          inputTokens: 10,
          outputTokens: 5,
        }),
      );
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
          request.budget !== "frame-anomaly-classifier" &&
          request.budget !== "perception-entity-fallback" &&
          request.budget !== "perception-mode-fallback" &&
          request.budget !== "perception-temporal-cue",
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
      ]);
      expect(nonCorrectiveRequests[0]?.budget).toBe("procedural-context");
      expect(nonCorrectiveRequests[2]?.budget).toBe("commitment-judge");
      expect(nonCorrectiveRequests[3]?.budget).toBe("commitment-revision");
      expect(nonCorrectiveRequests[4]?.budget).toBe("commitment-judge");
      expect(nonCorrectiveRequests[5]?.budget).toBe("closure-response-auditor");
      expect(nonCorrectiveRequests[6]?.budget).toBe("generation-stop-commitment");
      expect(nonCorrectiveRequests[7]?.budget).toBe("reflection");
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
          llmEnabled: true,
        },
        affective: {
          llmEnabled: false,
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
          createEntityDetectionResponse(["Atlas"]),
          createModeDetectionResponse("problem_solving"),
          createNoTemporalCueResponse(),
          createTurnPlanResponse(),
          createEmitAnswerResponse("Rerun pnpm install for the Atlas deploy.", {
            inputTokens: 10,
            outputTokens: 5,
          }),
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
          llmEnabled: true,
        },
        affective: {
          llmEnabled: false,
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
          createEntityDetectionResponse(["Atlas"]),
          createModeDetectionResponse("problem_solving"),
          createNoTemporalCueResponse(),
          createGenerationGateResponse({
            decision: "proceed",
            substantive: true,
            reason: "The repeated short deploy message is a real request.",
          }),
          createEmitAnswerResponse("Use the rollback fallback.", {
            inputTokens: 10,
            outputTokens: 5,
          }),
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
          llmEnabled: false,
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
          createEmitAnswerResponse("Check the deploy state before answering.", {
            inputTokens: 10,
            outputTokens: 5,
          }),
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
          llmEnabled: false,
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
                  intents: [],
                },
              },
            ],
          },
          createEmitAnswerResponse("I need to compare more evidence before answering.", {
            inputTokens: 12,
            outputTokens: 6,
          }),
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

  it("renders the shared audience state above the compact planner ledger", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000);
    const llm = new FakeLLMClient({
      responses: [
        createEntityDetectionResponse(["Spain planning", "Ben"]),
        createModeDetectionResponse("problem_solving"),
        createNoTemporalCueResponse(),
        createGenerationGateResponse({
          decision: "proceed",
          substantive: true,
          reason: "planning turn",
        }),
        createTurnPlanResponse(),
        createEmitAnswerResponse("I will keep the locked route order visible.", {
          inputTokens: 10,
          outputTokens: 5,
        }),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: true,
        },
        generation: {
          evidenceLedger: {
            enabled: true,
          },
        },
      }),
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
      clock,
      liveExtraction: false,
    });

    try {
      const internal = borgInternals<{
        deps: Pick<BorgDependencies, "sharedStateRepository" | "entityRepository">;
      }>(borg);
      const audience = internal.deps.entityRepository.resolve("Spain planning", {
        kind: "group",
        provenance: "transport_audience_label",
      });
      const ben = internal.deps.entityRepository.resolve("Ben", {
        kind: "person",
        provenance: "user_declared",
      });
      const artifactSource = createStreamEntryId();

      internal.deps.sharedStateRepository.upsert(audience, [
        {
          type: "add",
          kind: "locked",
          text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
          owner_entity_id: audience,
          provenance_stream_entry_ids: [artifactSource],
        },
      ]);

      const result = await borg.turn({
        userMessage:
          "Ben here. Could we add a Granada to SS leg after Seville, or does that break the route?",
        audience: "Spain planning",
        senderEntityId: ben,
        stakes: "high",
      });
      const plannerRequest = llm.requests.find((request) => request.budget === "cognition-plan");
      const finalizerRequest = llm.requests.find(
        (request) => request.budget === "cognition-system-2",
      );
      const plannerSystem = requestSystemText(plannerRequest);
      const finalizerSystem = requestSystemText(finalizerRequest);
      const artifactIndex = plannerSystem.indexOf("## 0. Shared Audience State");
      const compactLedgerIndex = plannerSystem.indexOf("CompactPlannerLedger");
      const finalizerArtifactIndex = finalizerSystem.indexOf("## 0. Shared Audience State");
      const currentUserSectionIndex = finalizerSystem.indexOf("## 1. Current User Message");

      expect(result.path).toBe("system_2");
      expect(plannerSystem).toContain(
        "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
      );
      expect(finalizerSystem).toContain(
        "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
      );
      expect(artifactIndex).toBeGreaterThanOrEqual(0);
      expect(compactLedgerIndex).toBeGreaterThan(artifactIndex);
      expect(finalizerArtifactIndex).toBeGreaterThanOrEqual(0);
      expect(currentUserSectionIndex).toBeGreaterThan(finalizerArtifactIndex);
    } finally {
      await borg.close();
    }
  });

  it("surfaces locked route order across artifact, ledger, and prompts for phantom-route turns", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000);
    const lockedOrder = "Itinerary order: Madrid -> San Sebastian -> Seville -> Granada";
    const lockedFinalBase = "Granada is the final base before fly-home.";
    const llm = new FakeLLMClient({
      responses: [
        createEntityDetectionResponse(["Spain planning", "Ben"]),
        createModeDetectionResponse("problem_solving"),
        createNoTemporalCueResponse(),
        createGenerationGateResponse({
          decision: "proceed",
          substantive: true,
          reason: "planning turn",
        }),
        createTurnPlanResponse(),
        createEmitAnswerResponse("Context surfaced.", {
          inputTokens: 10,
          outputTokens: 5,
        }),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: true,
        },
        generation: {
          evidenceLedger: {
            enabled: true,
          },
        },
      }),
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
      clock,
      liveExtraction: false,
    });

    try {
      const internal = borgInternals<{
        deps: Pick<BorgDependencies, "sharedStateRepository" | "entityRepository">;
      }>(borg);
      const audience = internal.deps.entityRepository.resolve("Spain planning", {
        kind: "group",
        provenance: "transport_audience_label",
      });
      const ben = internal.deps.entityRepository.resolve("Ben", {
        kind: "person",
        provenance: "user_declared",
      });
      const orderEntry = await borg.stream.append({
        kind: "user_msg",
        audience: "Spain planning",
        sender_entity_id: ben,
        content: `We locked this route: ${lockedOrder}.`,
      });
      const finalBaseEntry = await borg.stream.append({
        kind: "agent_msg",
        audience: "Spain planning",
        content: `Locked note: ${lockedFinalBase}`,
      });

      internal.deps.sharedStateRepository.upsert(audience, [
        {
          type: "add",
          kind: "locked",
          text: lockedOrder,
          owner_entity_id: audience,
          provenance_stream_entry_ids: [orderEntry.id],
        },
        {
          type: "add",
          kind: "locked",
          text: lockedFinalBase,
          owner_entity_id: audience,
          provenance_stream_entry_ids: [finalBaseEntry.id],
        },
      ]);

      await borg.turn({
        userMessage: "Ben here. What if we flew from Granada to San Sebastian directly?",
        audience: "Spain planning",
        senderEntityId: ben,
        stakes: "high",
      });

      const plannerRequest = llm.requests.find((request) => request.budget === "cognition-plan");
      const finalizerRequest = llm.requests.find(
        (request) => request.budget === "cognition-system-2",
      );
      const plannerSystem = requestSystemText(plannerRequest);
      const finalizerSystem = requestSystemText(finalizerRequest);
      const compactPlannerLedgerBlock = extractTaggedPromptBlock(
        plannerSystem,
        "borg_compact_planner_ledger",
      );
      const finalizerArtifactStart = finalizerSystem.indexOf("## 0. Shared Audience State");
      const finalizerCurrentUserStart = finalizerSystem.indexOf("## 1. Current User Message");
      const finalizerArtifactSection = finalizerSystem.slice(
        finalizerArtifactStart,
        finalizerCurrentUserStart,
      );
      const persistedLockedTexts =
        internal.deps.sharedStateRepository
          .get(audience)
          ?.entries.filter((entry) => entry.kind === "locked" && entry.superseded_by_id === null)
          .map((entry) => entry.text) ?? [];

      expect(compactPlannerLedgerBlock).toContain(lockedOrder);
      expect(compactPlannerLedgerBlock).toContain(lockedFinalBase);
      expect(finalizerArtifactStart).toBeGreaterThanOrEqual(0);
      expect(finalizerCurrentUserStart).toBeGreaterThan(finalizerArtifactStart);
      expect(finalizerArtifactSection).toContain(lockedOrder);
      expect(finalizerArtifactSection).toContain(lockedFinalBase);
      expect(persistedLockedTexts).toEqual(expect.arrayContaining([lockedOrder, lockedFinalBase]));
    } finally {
      await borg.close();
    }
  });

  it("rejects shared state citations outside the prompt-visible ledger", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "trace.jsonl");
    const clock = new ManualClock(1_000);
    const inventedStreamId = "strm_zzzzzzzzzzzzzzzz";
    const llm = new FakeLLMClient({
      responses: [
        createEntityDetectionResponse(["Spain planning", "Ben"]),
        createModeDetectionResponse("problem_solving"),
        createNoTemporalCueResponse(),
        createGenerationGateResponse({
          decision: "proceed",
          substantive: true,
          reason: "planning turn",
        }),
        createSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Invented route fact",
              owner_entity_id: null,
              source_stream_entry_ids: [inventedStreamId],
            },
          ],
        }),
        createTurnPlanResponse(),
        createEmitAnswerResponse("I will not persist unsupported provenance.", {
          inputTokens: 10,
          outputTokens: 5,
        }),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: true,
        },
        generation: {
          evidenceLedger: {
            enabled: true,
          },
        },
      }),
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
      clock,
      tracerPath: tracePath,
      liveExtraction: false,
    });

    try {
      const internal = borgInternals<{
        deps: Pick<BorgDependencies, "sharedStateRepository" | "entityRepository">;
      }>(borg);
      const audience = internal.deps.entityRepository.resolve("Spain planning", {
        kind: "group",
        provenance: "transport_audience_label",
      });
      const ben = internal.deps.entityRepository.resolve("Ben", {
        kind: "person",
        provenance: "user_declared",
      });

      await borg.turn({
        userMessage: "Ben here. Lock the invented route fact.",
        audience: "Spain planning",
        senderEntityId: ben,
        stakes: "high",
      });

      const traceEvents = readFileSync(tracePath, "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line) as Record<string, unknown>);

      expect(internal.deps.sharedStateRepository.get(audience)).toBeNull();
      expect(traceEvents).toContainEqual(
        expect.objectContaining({
          event: "shared_state.compile.completed",
          rejectedCount: 1,
          rejectionReasons: ["disallowed_source_stream_entry_id"],
          applied: false,
        }),
      );
    } finally {
      await borg.close();
    }
  });
});
