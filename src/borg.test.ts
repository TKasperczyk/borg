import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_CONFIG } from "./config/index.js";
import type { EmbeddingClient } from "./embeddings/index.js";
import { FakeLLMClient } from "./llm/index.js";
import { EntityRepository, commitmentMigrations } from "./memory/commitments/index.js";
import { episodicMigrations } from "./memory/episodic/index.js";
import { EpisodicRepository, createEpisodesTableSchema } from "./memory/episodic/repository.js";
import { selfMigrations, type OpenQuestionsRepository } from "./memory/self/index.js";
import { retrievalMigrations } from "./retrieval/index.js";
import { LanceDbStore } from "./storage/lancedb/index.js";
import { openDatabase, SqliteDatabase } from "./storage/sqlite/index.js";
import { ManualClock } from "./util/clock.js";
import { createEpisodeId, createStreamEntryId } from "./util/ids.js";
import { Borg } from "./borg.js";

const EPISODE_TOOL_NAME = "EmitEpisodeCandidates";

class ScriptedEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    if (/Planning sync|planning|Atlas|atlas|pnpm|deploy|rollback/.test(text)) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    return Float32Array.from([0, 1, 0, 0]);
  }
}

describe("Borg", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("opens the sprint 2 facade and reuses injected clients", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const llm = new FakeLLMClient();
    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      const entry = await borg.stream.append({
        kind: "user_msg",
        content: "planning kickoff",
      });
      llm.pushResponse({
        text: "",
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "toolu_1",
            name: EPISODE_TOOL_NAME,
            input: {
              episodes: [
                {
                  title: "Planning sync",
                  narrative:
                    "The team aligned on the sprint plan. They captured the first follow-up actions.",
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

      const extracted = await borg.episodic.extract({
        sinceTs: entry.timestamp,
      });
      const results = await borg.episodic.search("planning", {
        limit: 1,
      });
      const value = borg.self.values.add({
        label: "clarity",
        description: "Prefer explicit, auditable state.",
        priority: 5,
        provenance: { kind: "manual" },
      });

      expect(extracted.inserted).toBe(1);
      expect(results[0]?.citationChain[0]?.id).toBe(entry.id);
      expect(borg.stream.tail(1)).toHaveLength(1);
      expect(borg.self.values.list()).toEqual([
        expect.objectContaining({
          id: value.id,
        }),
      ]);
    } finally {
      await borg.close();
    }
  });

  it("defaults episodic public APIs to public-only visibility unless audience access is explicit", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [
        ...episodicMigrations,
        ...selfMigrations,
        ...retrievalMigrations,
        ...commitmentMigrations,
      ],
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
    const entities = new EntityRepository({
      db,
      clock,
    });
    const alice = entities.resolve("Alice");

    await repo.insert({
      id: "ep_publicpublicpub1" as never,
      title: "Public planning note",
      narrative: "A public planning note.",
      participants: ["team"],
      location: null,
      start_time: 1,
      end_time: 2,
      source_stream_ids: ["strm_publicpublic0001" as never],
      significance: 0.8,
      tags: ["planning"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      audience_entity_id: null,
      shared: true,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 1,
      updated_at: 1,
    });
    await repo.insert({
      id: "ep_privateprivate01" as never,
      title: "Alice planning note",
      narrative: "A planning note only for Alice.",
      participants: ["Alice"],
      location: null,
      start_time: 3,
      end_time: 4,
      source_stream_ids: ["strm_privateprivate01" as never],
      significance: 0.8,
      tags: ["planning"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      audience_entity_id: alice,
      shared: false,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 3,
      updated_at: 3,
    });
    db.close();
    await store.close();

    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      expect((await borg.episodic.search("planning", { limit: 5 })).map((item) => item.episode.id)).toEqual([
        "ep_publicpublicpub1",
      ]);
      expect((await borg.episodic.get("ep_privateprivate01" as never))?.episode.id).toBeUndefined();
      expect(
        (await borg.episodic.search("planning", { limit: 5, audience: "Alice" })).map(
          (item) => item.episode.id,
        ),
      ).toContain("ep_privateprivate01");
      expect(
        (await borg.episodic.get("ep_privateprivate01" as never, { audience: "Alice" }))?.episode
          .id,
      ).toBe("ep_privateprivate01");
      expect(
        (await borg.episodic.get("ep_privateprivate01" as never, { crossAudience: true }))?.episode
          .id,
      ).toBe("ep_privateprivate01");
    } finally {
      await borg.close();
    }
  });

  it("lets the public episodic search API rescue explicit entity matches", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [
        ...episodicMigrations,
        ...selfMigrations,
        ...retrievalMigrations,
        ...commitmentMigrations,
      ],
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

    for (let index = 0; index < 12; index += 1) {
      await repo.insert({
        id: `ep_publicentity${String(index).padStart(4, "0")}` as never,
        title: `Decoy ${index}`,
        narrative: "A strong semantic match that lacks the entity cue.",
        participants: ["team"],
        location: null,
        start_time: 1 + index,
        end_time: 2 + index,
        source_stream_ids: [`strm_publicentity${String(index).padStart(4, "0")}` as never],
        significance: 0.2,
        tags: ["decoy"],
        confidence: 0.9,
        lineage: {
          derived_from: [],
          supersedes: [],
        },
        audience_entity_id: null,
        shared: true,
        embedding: Float32Array.from([1, 0, 0, 0]),
        created_at: 1 + index,
        updated_at: 1 + index,
      });
    }

    const rescuedId = "ep_entityrescue0001" as never;
    await repo.insert({
      id: rescuedId,
      title: "Atlas entity rescue",
      narrative: "A hot Atlas note that should be rescued by explicit entity terms.",
      participants: ["team"],
      location: null,
      start_time: 500,
      end_time: 600,
      source_stream_ids: ["strm_entityrescue0001" as never],
      significance: 1,
      tags: ["Atlas"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      audience_entity_id: null,
      shared: true,
      embedding: Float32Array.from([0, 1, 0, 0]),
      created_at: 500,
      updated_at: clock.now(),
    });
    repo.updateStats(rescuedId, {
      retrieval_count: 12,
      win_rate: 0.9,
      last_retrieved: clock.now() - 1_000,
    });
    db.close();
    await store.close();

    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      expect(
        (await borg.episodic.search("deploy", { limit: 3, entityTerms: ["atlas"] })).map(
          (item) => item.episode.id,
        ),
      ).toContain(rescuedId);
    } finally {
      await borg.close();
    }
  });

  it("treats explicit public API timeRange as a strict filter", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(10_000_000_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [
        ...episodicMigrations,
        ...selfMigrations,
        ...retrievalMigrations,
        ...commitmentMigrations,
      ],
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

    const hotOutOfRangeId = createEpisodeId();
    await repo.insert({
      id: hotOutOfRangeId,
      title: "Hot out-of-range deploy note",
      narrative: "A recent hot semantic match outside the requested time window.",
      participants: ["team"],
      location: null,
      start_time: 900_000,
      end_time: 901_000,
      source_stream_ids: [createStreamEntryId()],
      significance: 1,
      tags: ["deploy"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      audience_entity_id: null,
      shared: true,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: clock.now(),
      updated_at: clock.now(),
    });
    repo.updateStats(hotOutOfRangeId, {
      retrieval_count: 12,
      win_rate: 0.9,
      last_retrieved: clock.now() - 1_000,
    });

    const inRangeId = createEpisodeId();
    await repo.insert({
      id: inRangeId,
      title: "In-range deploy incident",
      narrative: "An older in-range note that should survive the strict time filter.",
      participants: ["team"],
      location: null,
      start_time: 150_000,
      end_time: 160_000,
      source_stream_ids: [createStreamEntryId()],
      significance: 1,
      tags: ["incident"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      audience_entity_id: null,
      shared: true,
      embedding: Float32Array.from([0, 1, 0, 0]),
      created_at: 10,
      updated_at: 10,
    });
    db.close();
    await store.close();

    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const results = await borg.episodic.search("deploy", {
        limit: 3,
        timeRange: {
          start: 140_000,
          end: 170_000,
        },
      });

      expect(results.map((item) => item.episode.id)).toEqual([inRangeId]);
    } finally {
      await borg.close();
    }
  });

  it("bootstraps a current autobiographical period once by default", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(Date.UTC(2026, 3, 22));

    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const period = borg.self.autobiographical.currentPeriod();

      expect(period).not.toBeNull();
      expect(period?.label).toBe("2026-Q2");
      expect(borg.self.autobiographical.listPeriods({ limit: 10 })).toHaveLength(1);
    } finally {
      await borg.close();
    }

    const reopened = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      expect(reopened.self.autobiographical.listPeriods({ limit: 10 })).toHaveLength(1);
    } finally {
      await reopened.close();
    }
  });

  it("can disable autobiographical bootstrap", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        self: {
          autoBootstrapPeriod: false,
        },
        embedding: {
          ...DEFAULT_CONFIG.embedding,
          dims: 4,
        },
      },
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      expect(borg.self.autobiographical.currentPeriod()).toBeNull();
    } finally {
      await borg.close();
    }
  });

  it("closes opened resources if a later Borg.open step fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const sqliteCloseSpy = vi.spyOn(SqliteDatabase.prototype, "close");
    const lanceCloseSpy = vi.spyOn(LanceDbStore.prototype, "close");
    const failure = new Error("embedding init failed");
    const openOptions = {
      dataDir: tempDir,
    } as {
      dataDir: string;
      embeddingClient?: ScriptedEmbeddingClient;
    };

    Object.defineProperty(openOptions, "embeddingClient", {
      get() {
        throw failure;
      },
    });

    await expect(Borg.open(openOptions)).rejects.toThrow(failure);
    expect(sqliteCloseSpy).toHaveBeenCalledTimes(1);
    expect(lanceCloseSpy).toHaveBeenCalledTimes(1);
  });

  it("runs the full cognitive turn loop", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
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
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    db.close();
    await store.close();

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
      ],
    });
    const borg = await Borg.open({
      config: {
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
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      const goal = borg.self.goals.add({
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
      expect(result.intents[0]?.next_action).toContain("rerun the deploy");
      expect(borg.workmem.load().turn_counter).toBe(1);
      expect(borg.self.goals.list({ status: "active" })[0]?.id).toBe(goal.id);
      expect(borg.self.goals.list({ status: "active" })[0]?.progress_notes).toContain(
        "Heuristic turn progress",
      );
      expect(borg.self.goals.list({ status: "active" })[0]?.provenance).toEqual({
        kind: "episodes",
        episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      });
      expect(borg.self.traits.list()[0]).toMatchObject({
        label: "engaged",
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
        },
      });
      // Phase D: the planner's EmitTurnPlan tool-call shows up as a
      // compact "plan: ..." thought entry persisted before the agent_msg.
      expect(borg.stream.tail(3).map((entry) => entry.kind)).toEqual([
        "user_msg",
        "thought",
        "agent_msg",
      ]);
    } finally {
      await borg.close();
    }
  });

  it("uses offline reflector provenance for durable reflection updates when no episodes are retrieved", async () => {
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
      config: {
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
      },
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
      expect(borg.self.traits.list()[0]).toMatchObject({
        label: "engaged",
        provenance: {
          kind: "offline",
          process: "reflector",
        },
      });
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
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
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
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    db.close();
    await store.close();

    const llm = new FakeLLMClient({
      responses: [
        // S2 planning (Haiku)
        {
          text: "List the commitments that apply before answering.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
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
      ],
    });
    const borg = await Borg.open({
      config: {
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
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      borg.commitments.add({
        type: "boundary",
        directive: "Do not discuss Atlas with Sam",
        priority: 10,
        audience: "Sam",
        about: "Atlas",
        provenance: { kind: "manual" },
      });
      borg.commitments.add({
        type: "boundary",
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
      migrations: [...episodicMigrations, ...commitmentMigrations, ...selfMigrations],
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
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    db.close();
    await store.close();

    const llm = new FakeLLMClient();
    const borg = await Borg.open({
      config: {
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
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      const commitment = borg.commitments.add({
        type: "boundary",
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
      const result = await borg.turn({
        userMessage: "Update Sam on Atlas.",
        audience: "Sam",
      });

      expect(result.response).toBe("I can't share Atlas details with Sam.");
      expect(llm.requests.map((request) => request.model)).toEqual([
        "sonnet",
        "haiku",
        "sonnet",
        "haiku",
      ]);
      expect(llm.requests[1]?.budget).toBe("commitment-judge");
      expect(llm.requests[2]?.budget).toBe("commitment-revision");
      expect(llm.requests[3]?.budget).toBe("commitment-judge");
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
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
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
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    db.close();
    await store.close();

    const firstBorg = await Borg.open({
      config: {
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
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "Rerun pnpm install for the Atlas deploy.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
    });

    try {
      const firstResult = await firstBorg.turn({
        userMessage: "Atlas deploy failed with pnpm",
      });

      expect(firstResult.retrievedEpisodeIds[0]).toBe("ep_aaaaaaaaaaaaaaaa");
      expect(firstBorg.workmem.load().suppressed).toEqual([
        expect.objectContaining({
          id: "ep_aaaaaaaaaaaaaaaa",
          reason: "already surfaced",
        }),
      ]);
    } finally {
      await firstBorg.close();
    }

    const reopenedBorg = await Borg.open({
      config: {
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
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "Use the rollback fallback.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
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
      expect(secondResult.retrievedEpisodeIds[0]).toBe("ep_bbbbbbbbbbbbbbbb");
    } finally {
      await reopenedBorg.close();
    }
  });

  it("saves working memory early and logs an internal event when a turn fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: {
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
      },
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
        turn_counter: 1,
        mode: "problem_solving",
      });
      expect(borg.stream.tail(2).map((entry) => entry.kind)).toEqual([
        "user_msg",
        "internal_event",
      ]);
    } finally {
      await borg.close();
    }
  });

  it("keeps a turn running when the reflection open-question hook fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: {
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
      },
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
        ],
      }),
    });

    try {
      const internal = borg as unknown as {
        deps: {
          turnOrchestrator: {
            options: {
              openQuestionsRepository: OpenQuestionsRepository;
            };
          };
        };
      };
      internal.deps.turnOrchestrator.options.openQuestionsRepository = {
        add() {
          throw new Error("hook exploded");
        },
      } as unknown as OpenQuestionsRepository;

      const result = await borg.turn({
        userMessage: "Why is Atlas still failing?",
        stakes: "high",
      });

      expect(result.path).toBe("system_2");
      expect(result.response).toContain("compare more evidence");
      expect(borg.self.openQuestions.list({ status: "open" })).toEqual([]);
      expect(borg.stream.tail(4).map((entry) => entry.kind)).toEqual([
        "user_msg",
        "thought",
        "agent_msg",
        "internal_event",
      ]);
    } finally {
      await borg.close();
    }
  });

  it("keeps a turn running when mood update fails and logs an internal event", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: {
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
        },
        affective: {
          useLlmFallback: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
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
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "Try the rollback plan.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
    });

    try {
      const internal = borg as unknown as {
        deps: {
          turnOrchestrator: {
            options: {
              moodRepository: {
                update: (sessionId: string, update: unknown) => unknown;
              };
            };
          };
        };
      };
      vi.spyOn(internal.deps.turnOrchestrator.options.moodRepository, "update").mockImplementation(
        () => {
          throw new Error("mood exploded");
        },
      );

      const result = await borg.turn({
        userMessage: "Atlas deploy failed again.",
      });

      expect(result.response).toContain("rollback plan");
      expect(borg.stream.tail(4)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              hook: "mood_update",
            }),
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });

  it("keeps a turn running when social update fails and logs an internal event", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: {
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "relational",
        },
        affective: {
          useLlmFallback: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
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
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "Focus on the audience and clarify the tone first.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "I'll keep this short for Sam.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
    });

    try {
      const internal = borg as unknown as {
        deps: {
          turnOrchestrator: {
            options: {
              socialRepository: {
                recordInteraction: (entityId: string, interaction: unknown) => unknown;
              };
            };
          };
        };
      };
      vi.spyOn(
        internal.deps.turnOrchestrator.options.socialRepository,
        "recordInteraction",
      ).mockImplementation(() => {
        throw new Error("social exploded");
      });

      const result = await borg.turn({
        userMessage: "Can you phrase this carefully for Sam?",
        audience: "Sam",
      });

      expect(result.response).toContain("short for Sam");
      expect(borg.stream.tail(4)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              hook: "social_update",
            }),
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });

  it("falls back to neutral affect when affective extraction fails and logs an internal event", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: {
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
        },
        affective: {
          useLlmFallback: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
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
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "Let's inspect the deploy state first.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
    });

    try {
      const internal = borg as unknown as {
        deps: {
          turnOrchestrator: {
            options: {
              affectiveSignalDetector?: (
                text: string,
                recentHistory?: readonly string[],
                options?: unknown,
              ) => Promise<unknown>;
            };
          };
        };
      };
      internal.deps.turnOrchestrator.options.affectiveSignalDetector = async () => {
        throw new Error("affect exploded");
      };

      const result = await borg.turn({
        userMessage: "Atlas deploy failed and I'm upset.",
      });

      expect(result.response).toContain("inspect the deploy state");
      expect(borg.workmem.load().mood).toEqual({
        valence: 0,
        arousal: 0,
        dominant_emotion: null,
      });
      expect(borg.stream.tail(4)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              hook: "affective_extraction",
            }),
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });

  it("runs offline maintenance through the Borg facade and exposes audit reversal", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const nowMs = 100 * 24 * 60 * 60 * 1_000;
    const clock = new ManualClock(nowMs);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
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
      id: "ep_cccccccccccccccc" as never,
      title: "Old quiet note",
      narrative: "A stale note that should be archived by the curator.",
      participants: ["team"],
      location: null,
      start_time: nowMs - 50 * 24 * 60 * 60 * 1_000,
      end_time: nowMs - 50 * 24 * 60 * 60 * 1_000 + 1,
      source_stream_ids: ["strm_cccccccccccccccc" as never],
      significance: 0.2,
      tags: ["quiet"],
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([0, 1, 0, 0]),
      created_at: nowMs - 50 * 24 * 60 * 60 * 1_000,
      updated_at: nowMs - 50 * 24 * 60 * 60 * 1_000,
    });
    db.close();
    await store.close();

    const borg = await Borg.open({
      config: {
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
        offline: {
          consolidator: {
            enabled: true,
            similarityThreshold: 0.82,
            minClusterSize: 2,
            maxClustersPerRun: 2,
            budget: 15_000,
          },
          reflector: {
            enabled: true,
            minSupport: 3,
            ceilingConfidence: 0.5,
            maxInsightsPerRun: 2,
            budget: 30_000,
          },
          curator: {
            enabled: true,
            t1Heat: 5,
            t2Heat: 15,
            t3DemoteHeat: 3,
            archiveAgeDays: 45,
            archiveMinHeat: 1,
          },
          overseer: {
            enabled: true,
            lookbackHours: 24,
            maxChecksPerRun: 8,
            budget: 20_000,
          },
        },
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const result = await borg.dream.curate();
      expect(result.results[0]?.process).toBe("curator");

      const audits = borg.audit.list({
        process: "curator",
      });
      expect(audits.length).toBeGreaterThan(0);
      expect((await borg.episodic.get("ep_cccccccccccccccc" as never))?.episode.id).toBe(
        "ep_cccccccccccccccc",
      );

      const reverted = await borg.audit.revert(audits[0]!.id);
      expect(reverted?.reverted_at).not.toBeNull();
    } finally {
      await borg.close();
    }
  });
});
