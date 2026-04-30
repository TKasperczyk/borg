import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_CONFIG } from "./config/index.js";
import type { EmbeddingClient } from "./embeddings/index.js";
import { Reflector, type ReflectorOptions } from "./cognition/index.js";
import { FakeLLMClient, type LLMClient } from "./llm/index.js";
import { EntityRepository, commitmentMigrations } from "./memory/commitments/index.js";
import { episodicMigrations } from "./memory/episodic/index.js";
import { EpisodicRepository, createEpisodesTableSchema } from "./memory/episodic/repository.js";
import { selfMigrations } from "./memory/self/index.js";
import { retrievalMigrations } from "./retrieval/index.js";
import { LanceDbStore } from "./storage/lancedb/index.js";
import { openDatabase, SqliteDatabase } from "./storage/sqlite/index.js";
import { ManualClock } from "./util/clock.js";
import { createEpisodeId, createSessionId, createStreamEntryId } from "./util/ids.js";
import { createTestConfig } from "./offline/test-support.js";
import { resolveBorgConfig } from "./borg/storage-setup.js";
import { Borg } from "./borg.js";

const EPISODE_TOOL_NAME = "EmitEpisodeCandidates";
const ENTITY_TOOL_NAME = "EmitEntityExtraction";
const MODE_TOOL_NAME = "EmitModeDetection";
const TEMPORAL_TOOL_NAME = "EmitTemporalCue";

function createTurnPlanResponse(referencedEpisodeIds: string[] = []) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_plan",
        name: "EmitTurnPlan",
        input: {
          uncertainty: "",
          verification_steps: [],
          tensions: [],
          voice_note: "",
          referenced_episode_ids: referencedEpisodeIds,
          intents: [],
        },
      },
    ],
  };
}

function createGenerationGateResponse(input: {
  decision: "proceed" | "suppress";
  substantive: boolean;
  reason?: string;
}) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_generation_gate",
        name: "EmitGenerationGateDecision",
        input: {
          decision: input.decision,
          substantive: input.substantive,
          reason: input.reason ?? "classified by generation gate",
          confidence: 0.9,
        },
      },
    ],
  };
}

function createTraitReflectionResponse(input: {
  traitLabel: string;
  evidence: string;
  strengthDelta?: number;
  advancedGoals?: Array<{ goal_id: string; evidence: string }>;
}) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: input.advancedGoals ?? [],
          procedural_outcomes: [],
          trait_demonstrations: [
            {
              trait_label: input.traitLabel,
              evidence: input.evidence,
              strength_delta: input.strengthDelta ?? 0.05,
            },
          ],
          intent_updates: [],
        },
      },
    ],
  };
}

function createEmptyReflectionResponse(
  openQuestions: Array<{
    question: string;
    urgency: number;
    related_episode_ids: string[];
  }> = [],
) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: [],
          procedural_outcomes: [],
          trait_demonstrations: [],
          intent_updates: [],
          open_questions: openQuestions,
        },
      },
    ],
  };
}

function createInvalidEntityClassifierResponse() {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_entity",
        name: ENTITY_TOOL_NAME,
        input: { entities: [1] },
      },
    ],
  };
}

function createInvalidModeClassifierResponse() {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_mode",
        name: MODE_TOOL_NAME,
        input: { mode: "unknown" },
      },
    ],
  };
}

function createNoTemporalCueResponse() {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_temporal",
        name: TEMPORAL_TOOL_NAME,
        input: { has_cue: false },
      },
    ],
  };
}

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

  it("merges sparse Borg.open config with required defaults", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = resolveBorgConfig({
      config: {
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
            cognition: "test-cognition",
          },
        },
      } as never,
    });

    expect(config.dataDir).toBe(tempDir);
    expect(config.perception.useLlmFallback).toBe(false);
    expect(config.perception.modeWhenLlmAbsent).toBe("idle");
    expect(config.embedding.dims).toBe(4);
    expect(config.anthropic.auth).toBe("api-key");
    expect(config.anthropic.models).toEqual({
      ...DEFAULT_CONFIG.anthropic.models,
      cognition: "test-cognition",
    });
    expect(config.affective).toEqual(DEFAULT_CONFIG.affective);
    expect(config.procedural).toEqual(DEFAULT_CONFIG.procedural);
    expect(config.retrieval).toEqual(DEFAULT_CONFIG.retrieval);
    expect(config.executive).toEqual(DEFAULT_CONFIG.executive);
    expect(config.offline.beliefReviser).toEqual(DEFAULT_CONFIG.offline.beliefReviser);
    expect(config.maintenance).toEqual(DEFAULT_CONFIG.maintenance);
    expect(config.autonomy.executiveFocus).toEqual(DEFAULT_CONFIG.autonomy.executiveFocus);
  });

  it("omits a disabled belief reviser from default dream plans", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        offline: {
          consolidator: { enabled: false },
          reflector: { enabled: false },
          curator: { enabled: false },
          overseer: { enabled: false },
          ruminator: { enabled: false },
          selfNarrator: { enabled: false },
          proceduralSynthesizer: { enabled: false },
          beliefReviser: { enabled: false },
        },
      }),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const plan = await borg.dream.plan();

      expect(plan.processes).toEqual([]);
    } finally {
      await borg.close();
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

  it("exposes self writes through the identity guard instead of raw repositories", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      dataDir: tempDir,
      clock: new ManualClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const value = borg.self.values.add({
        label: "evidence-backed clarity",
        description: "Prefer evidence-backed changes.",
        priority: 5,
        provenance: {
          kind: "episodes",
          episode_ids: [createEpisodeId(), createEpisodeId(), createEpisodeId()],
        },
      });

      expect(value.state).toBe("established");
      expect("remove" in (borg.self.values as Record<string, unknown>)).toBe(false);
      expect("recordContradiction" in (borg.self.values as Record<string, unknown>)).toBe(false);

      const result = borg.self.values.update(
        value.id,
        {
          description: "Manual overwrite should not bypass review.",
        },
        {
          kind: "manual",
        },
      );

      expect(result).toEqual({
        status: "requires_review",
        current: value,
      });
      expect(borg.self.values.get(value.id)?.description).toBe("Prefer evidence-backed changes.");

      const periodEpisodeId = createEpisodeId();
      const period = borg.self.autobiographical.upsertPeriod({
        label: "2026-Q2",
        start_ts: 1_100,
        narrative: "Episode-backed period.",
        key_episode_ids: [periodEpisodeId],
        themes: ["guard"],
        provenance: {
          kind: "episodes",
          episode_ids: [periodEpisodeId],
        },
      });
      const closeResult = borg.self.autobiographical.closePeriod(period.id, 1_200, {
        kind: "manual",
      });

      expect(closeResult).toEqual({
        status: "requires_review",
        current: period,
      });
      expect(borg.self.autobiographical.getPeriod(period.id)?.end_ts).toBeNull();

      const markerEpisodeId = createEpisodeId();
      const marker = borg.self.growthMarkers.add({
        ts: 1_150,
        category: "understanding",
        what_changed: "Facade growth marker writes are audited.",
        evidence_episode_ids: [markerEpisodeId],
        confidence: 0.7,
        source_process: "manual",
        provenance: {
          kind: "episodes",
          episode_ids: [markerEpisodeId],
        },
      });

      expect(
        borg.identity.listEvents({
          recordType: "growth_marker",
          recordId: marker.id,
        })[0]?.action,
      ).toBe("create");

      const questionEpisodeId = createEpisodeId();
      const question = borg.self.openQuestions.add({
        question: "Does the facade guard open question state changes?",
        urgency: 0.5,
        related_episode_ids: [questionEpisodeId],
        provenance: {
          kind: "episodes",
          episode_ids: [questionEpisodeId],
        },
        source: "reflection",
      });
      const bumpResult = borg.self.openQuestions.bumpUrgency(question.id, 0.2, {
        kind: "manual",
      });

      expect(bumpResult).toEqual({
        status: "requires_review",
        current: question,
      });
      expect(borg.self.openQuestions.list({ status: "open" })[0]?.urgency).toBe(0.5);
    } finally {
      await borg.close();
    }
  });

  it("lets facade upsertPeriod update an existing autobiographical period", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      dataDir: tempDir,
      clock: new ManualClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const episodeId = createEpisodeId();
      const period = borg.self.autobiographical.upsertPeriod({
        label: "2026-Q2",
        start_ts: 1_100,
        narrative: "Initial period narrative.",
        key_episode_ids: [episodeId],
        themes: ["identity"],
        provenance: {
          kind: "episodes",
          episode_ids: [episodeId],
        },
      });
      const result = borg.self.autobiographical.upsertPeriod({
        id: period.id,
        label: "2026-Q2 revised",
        start_ts: 1_100,
        end_ts: 1_900,
        narrative: "Updated period narrative.",
        key_episode_ids: [episodeId],
        themes: ["identity", "revision"],
        provenance: {
          kind: "episodes",
          episode_ids: [episodeId],
        },
      });

      expect(result).toEqual({
        status: "applied",
        record: expect.objectContaining({
          id: period.id,
          label: "2026-Q2 revised",
          end_ts: 1_900,
          narrative: "Updated period narrative.",
          themes: ["identity", "revision"],
        }),
      });
      expect(borg.self.autobiographical.getPeriod(period.id)).toMatchObject({
        label: "2026-Q2 revised",
        end_ts: 1_900,
      });
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
      emotional_arc: null,
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
      emotional_arc: null,
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
      expect(
        (await borg.episodic.search("planning", { limit: 5 })).map((item) => item.episode.id),
      ).toEqual(["ep_publicpublicpub1"]);
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
        emotional_arc: null,
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
      emotional_arc: null,
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
      emotional_arc: null,
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
      emotional_arc: null,
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

  it("does not bootstrap an autobiographical period before evidence", async () => {
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
      expect(borg.self.autobiographical.currentPeriod()).toBeNull();
      expect(borg.self.autobiographical.listPeriods({ limit: 10 })).toHaveLength(0);
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
      expect(reopened.self.autobiographical.listPeriods({ limit: 10 })).toHaveLength(0);
    } finally {
      await reopened.close();
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

  it("waits for live ingestion to flush before closing", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
            text: "Try the rollback plan.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createEmptyReflectionResponse(),
        ],
      }),
      liveExtraction: true,
    });

    let closePromise: Promise<void> | undefined;
    let closed = false;
    let resolveExtraction:
      | ((value: { inserted: number; updated: number; skipped: number }) => void)
      | undefined;

    try {
      const internal = borg as unknown as {
        deps: {
          streamIngestionCoordinator?: {
            options: {
              extractor: {
                extractFromStream(): Promise<{
                  inserted: number;
                  updated: number;
                  skipped: number;
                }>;
              };
            };
          };
        };
      };
      const coordinator = internal.deps.streamIngestionCoordinator;
      expect(coordinator).toBeDefined();

      let notifyExtractionStarted: (() => void) | undefined;
      const extractionStarted = new Promise<void>((resolve) => {
        notifyExtractionStarted = resolve;
      });
      let extractionCalls = 0;
      coordinator!.options.extractor = {
        async extractFromStream(): Promise<{ inserted: number; updated: number; skipped: number }> {
          extractionCalls += 1;
          notifyExtractionStarted?.();

          return await new Promise((resolve) => {
            resolveExtraction = resolve;
          });
        },
      };

      await borg.turn({
        userMessage: "Atlas deploy failed again.",
      });
      await extractionStarted;

      closePromise = borg.close();
      void closePromise.then(() => {
        closed = true;
      });

      await Promise.resolve();
      expect(extractionCalls).toBe(1);
      expect(closed).toBe(false);

      resolveExtraction?.({
        inserted: 1,
        updated: 0,
        skipped: 0,
      });
      await closePromise;
      expect(closed).toBe(true);
    } finally {
      if (!closed) {
        resolveExtraction?.({
          inserted: 1,
          updated: 0,
          skipped: 0,
        });
        await closePromise?.catch(() => undefined);

        if (closePromise === undefined) {
          await borg.close().catch(() => undefined);
        }
      }
    }
  });

  it("enables live extraction by default", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
            text: "Try the rollback plan.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createEmptyReflectionResponse(),
        ],
      }),
    });

    try {
      const internal = borg as unknown as {
        deps: {
          streamIngestionCoordinator?: {
            options: {
              extractor: {
                extractFromStream(): Promise<{
                  inserted: number;
                  updated: number;
                  skipped: number;
                }>;
              };
            };
          };
        };
      };
      const coordinator = internal.deps.streamIngestionCoordinator;
      expect(coordinator).toBeDefined();
      let extractionCalls = 0;
      coordinator!.options.extractor = {
        async extractFromStream(): Promise<{ inserted: number; updated: number; skipped: number }> {
          extractionCalls += 1;
          return {
            inserted: 0,
            updated: 0,
            skipped: 0,
          };
        },
      };

      await borg.turn({
        userMessage: "Atlas deploy failed again.",
      });
      await borg.close();

      expect(extractionCalls).toBe(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("logs perception entries after the user message", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
      clock: new ManualClock(1_000),
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
          createEmptyReflectionResponse(),
        ],
      }),
      liveExtraction: false,
    });

    try {
      await borg.turn({
        userMessage: "Atlas deploy failed again.",
        stakes: "low",
      });

      expect(borg.stream.tail(3).map((entry) => entry.kind)).toEqual([
        "user_msg",
        "perception",
        "agent_msg",
      ]);
    } finally {
      await borg.close();
    }
  });

  it("logs live extraction failures to the triggering session stream", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const sessionId = createSessionId();
    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
            text: "Try the rollback plan.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      liveExtraction: true,
    });

    try {
      const internal = borg as unknown as {
        deps: {
          streamIngestionCoordinator?: {
            options: {
              extractor: {
                extractFromStream(): Promise<{
                  inserted: number;
                  updated: number;
                  skipped: number;
                }>;
              };
            };
          };
        };
      };
      const coordinator = internal.deps.streamIngestionCoordinator;
      expect(coordinator).toBeDefined();
      coordinator!.options.extractor = {
        async extractFromStream(): Promise<never> {
          throw new Error("boom");
        },
      };

      await borg.turn({
        userMessage: "Atlas deploy failed again.",
        sessionId,
      });
      await borg.close();

      const failedSessionEntries = borg.stream.tail(10, {
        session: sessionId,
      });
      const defaultSessionEntries = borg.stream.tail(10);

      expect(
        failedSessionEntries.some(
          (entry) =>
            entry.kind === "internal_event" &&
            String(entry.content).includes("Live episodic extraction failed: boom"),
        ),
      ).toBe(true);
      expect(
        defaultSessionEntries.some(
          (entry) =>
            entry.kind === "internal_event" &&
            String(entry.content).includes("Live episodic extraction failed: boom"),
        ),
      ).toBe(false);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("continues the turn and logs an internal event when pre-turn catch-up throws", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
      clock: new ManualClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "The turn still completes.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createEmptyReflectionResponse(),
        ],
      }),
      liveExtraction: true,
    });

    try {
      const internal = borg as unknown as {
        deps: {
          streamIngestionCoordinator?: {
            catchUp(): Promise<never>;
            ingest(): Promise<{
              ran: boolean;
              processedEntries: number;
            }>;
          };
        };
      };
      const coordinator = internal.deps.streamIngestionCoordinator;
      expect(coordinator).toBeDefined();
      coordinator!.catchUp = async (): Promise<never> => {
        throw new Error("catch-up exploded");
      };
      coordinator!.ingest = async () => ({
        ran: false,
        processedEntries: 0,
      });

      const result = await borg.turn({
        userMessage: "Please continue despite ingestion trouble.",
      });

      expect(result.response).toBe("The turn still completes.");
      expect(
        borg.stream.tail(10).some((entry) => {
          if (entry.kind !== "internal_event" || typeof entry.content !== "object") {
            return false;
          }

          return (
            entry.content !== null &&
            "hook" in entry.content &&
            entry.content.hook === "stream_ingestion_pre_turn_catchup" &&
            "error" in entry.content &&
            String(entry.content.error).includes("catch-up exploded")
          );
        }),
      ).toBe(true);
    } finally {
      await borg.close().catch(() => undefined);
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
      expect(borg.workmem.load().pending_intents).toEqual([expectedIntent]);
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
        source_episode_ids: [],
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
      expect(llm.requests.map((request) => request.model)).toEqual([
        "haiku",
        "sonnet",
        "haiku",
        "sonnet",
        "haiku",
        "haiku",
        "haiku",
      ]);
      expect(llm.requests[0]?.budget).toBe("procedural-context");
      expect(llm.requests[2]?.budget).toBe("commitment-judge");
      expect(llm.requests[3]?.budget).toBe("commitment-revision");
      expect(llm.requests[4]?.budget).toBe("commitment-judge");
      expect(llm.requests[5]?.budget).toBe("generation-stop-commitment");
      expect(llm.requests[6]?.budget).toBe("reflection");
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

  it("saves working memory early and logs an internal event when a turn fails", async () => {
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
        turn_counter: 1,
        mode: "problem_solving",
      });
      expect(borg.stream.tail(3).map((entry) => entry.kind)).toEqual([
        "user_msg",
        "perception",
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
      const internal = borg as unknown as {
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
      };
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

  it("keeps a turn running when mood update fails and logs an internal event", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
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
      }),
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
      liveExtraction: false,
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
      (
        internal.deps.turnOrchestrator.options as {
          affectiveSignalDetector?: () => Promise<unknown>;
        }
      ).affectiveSignalDetector = async () => ({
        valence: -0.7,
        arousal: 0.4,
        dominant_emotion: "fear",
      });

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

  it("feeds current-turn perceived mood into retrieval before mood persistence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      dataDir: tempDir,
      config: createTestConfig({
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
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "We can slow down and inspect the failure.",
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
      const internal = borg as unknown as {
        deps: {
          turnOrchestrator: {
            options: {
              affectiveSignalDetector?: () => Promise<unknown>;
              retrievalPipeline: {
                searchWithContext: (
                  query: string,
                  options?: Record<string, unknown>,
                ) => Promise<unknown>;
              };
            };
          };
        };
      };
      internal.deps.turnOrchestrator.options.affectiveSignalDetector = async () => ({
        valence: -0.9,
        arousal: 0.85,
        dominant_emotion: "fear",
      });
      const searchSpy = vi.spyOn(
        internal.deps.turnOrchestrator.options.retrievalPipeline,
        "searchWithContext",
      );

      await borg.turn({
        userMessage: "Atlas deploy failed and I am panicking.",
      });

      expect(searchSpy.mock.calls[0]?.[1]).toMatchObject({
        moodState: {
          valence: -0.9,
          arousal: 0.85,
          dominant_emotion: "fear",
        },
      });
    } finally {
      await borg.close();
    }
  });

  it("keeps a turn running when social update fails and logs an internal event", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
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
      }),
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
                recordInteractionWithId: (entityId: string, interaction: unknown) => unknown;
              };
            };
          };
        };
      };
      vi.spyOn(
        internal.deps.turnOrchestrator.options.socialRepository,
        "recordInteractionWithId",
      ).mockImplementation(() => {
        throw new Error("social exploded");
      });

      const result = await borg.turn({
        userMessage: "Can you phrase this carefully for Sam?",
        audience: "Sam",
      });

      expect(result.response).toContain("clarify the tone first");
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

  it("attributes social sentiment from the next user turn instead of the agent response", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
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
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createTurnPlanResponse([]),
          {
            text: "Warm, supportive reply for Sam.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createTurnPlanResponse([]),
          {
            text: "I hear that landed badly.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "I hear that landed badly.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "I hear that landed badly.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      liveExtraction: false,
    });

    try {
      const internal = borg as unknown as {
        deps: {
          turnOrchestrator: {
            options: {
              affectiveSignalDetector?: (text: string) => Promise<unknown>;
            };
          };
        };
      };
      internal.deps.turnOrchestrator.options.affectiveSignalDetector = async (text) =>
        text.includes("frustrated")
          ? {
              valence: -1,
              arousal: 0.6,
              dominant_emotion: "anger",
            }
          : {
              valence: 0,
              arousal: 0,
              dominant_emotion: null,
            };

      await borg.turn({
        userMessage: "Can you phrase this carefully for Sam?",
        audience: "Sam",
      });

      const profileAfterFirst = borg.social.getProfile("Sam");
      const pendingAfterFirst = borg.workmem.load().pending_social_attribution;

      expect(profileAfterFirst?.interaction_count).toBe(1);
      expect(profileAfterFirst?.sentiment_history).toEqual([]);
      expect(pendingAfterFirst).not.toBeNull();
      expect(pendingAfterFirst?.interaction_id).toBeGreaterThan(0);

      clock.advance(1_000);
      await borg.turn({
        userMessage: "I'm frustrated and upset with how that landed.",
        audience: "Sam",
      });

      const profileAfterSecond = borg.social.getProfile("Sam");
      const pendingAfterSecond = borg.workmem.load().pending_social_attribution;

      expect(profileAfterSecond?.interaction_count).toBe(2);
      expect(profileAfterSecond?.sentiment_history).toEqual([
        {
          ts: pendingAfterFirst?.turn_completed_ts ?? 0,
          valence: -1,
        },
      ]);
      expect(profileAfterSecond?.last_interaction_at).toBe(2_000);
      expect(pendingAfterSecond?.turn_completed_ts).toBe(2_000);
    } finally {
      await borg.close();
    }
  });

  it("keeps pending social attribution across an autonomous wake until the next user reply", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
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
        autonomy: {
          enabled: true,
          intervalMs: 60_000,
          maxWakesPerWindow: 6,
          budgetWindowMs: 86_400_000,
          executiveFocus: {
            enabled: false,
            stalenessSec: 86_400,
            dueLeadSec: 0,
          },
          triggers: {
            commitmentExpiring: {
              enabled: false,
              lookaheadMs: 86_400_000,
            },
            openQuestionDormant: {
              enabled: false,
              dormantMs: 604_800_000,
            },
            scheduledReflection: {
              enabled: true,
              intervalMs: 60_000,
            },
            goalFollowupDue: {
              enabled: false,
              lookaheadMs: 604_800_000,
              staleMs: 1_209_600_000,
            },
          },
          conditions: {
            commitmentRevoked: {
              enabled: false,
            },
            moodValenceDrop: {
              enabled: false,
              threshold: -0.5,
              windowN: 5,
              activationPeriodMs: 86_400_000,
            },
            openQuestionUrgencyBump: {
              enabled: false,
              threshold: 0.9,
            },
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "First reply for Sam.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Autonomous reflection.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Follow-up reply for Sam.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
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
              affectiveSignalDetector?: (text: string) => Promise<unknown>;
            };
          };
        };
      };
      internal.deps.turnOrchestrator.options.affectiveSignalDetector = async (text) =>
        text.includes("frustrated")
          ? {
              valence: -1,
              arousal: 0.6,
              dominant_emotion: "anger",
            }
          : {
              valence: 0,
              arousal: 0,
              dominant_emotion: null,
            };

      await borg.turn({
        userMessage: "Can you phrase this carefully for Sam?",
        audience: "Sam",
      });

      const pendingAfterFirst = borg.workmem.load().pending_social_attribution;
      expect(pendingAfterFirst).not.toBeNull();

      clock.advance(1_000);
      const wakeResult = await borg.autonomy.scheduler.tick();
      expect(wakeResult.firedEvents).toBe(1);
      expect(borg.workmem.load().pending_social_attribution).toEqual(pendingAfterFirst);
      expect(borg.social.getProfile("Sam")?.interaction_count).toBe(1);

      clock.advance(1_000);
      await borg.turn({
        userMessage: "I'm frustrated and upset with how that landed.",
        audience: "Sam",
      });

      const profileAfterSecond = borg.social.getProfile("Sam");
      expect(profileAfterSecond?.interaction_count).toBe(2);
      expect(profileAfterSecond?.sentiment_history).toEqual([
        {
          ts: pendingAfterFirst?.turn_completed_ts ?? 0,
          valence: -1,
        },
      ]);
    } finally {
      await borg.close();
    }
  });

  it("reinforces a pending trait from the next positive user turn with episode-backed provenance", async () => {
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
      title: "Atlas status update",
      narrative: "Atlas needed a warmer explanation.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.9,
      tags: ["atlas", "tone"],
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

    const borg = await Borg.open({
      config: createTestConfig({
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
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createTurnPlanResponse(["ep_aaaaaaaaaaaaaaaa"]),
          {
            text: "Here is a warmer Atlas update.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update.",
          }),
          {
            text: "Glad that helped.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      liveExtraction: false,
    });

    try {
      await borg.turn({
        userMessage: "Can you make the Atlas update sound warmer?",
        stakes: "high",
      });

      expect(borg.self.traits.list()).toEqual([]);
      // Sprint 56: trait evidence is the demonstrating turn's stream
      // entries; the legacy episode_ids field is empty until the offline
      // extraction completes (disabled here via liveExtraction:false).
      const pendingTraitFirst = borg.workmem.load().pending_trait_attribution;
      expect(pendingTraitFirst).toMatchObject({
        trait_label: "warm",
        source_episode_ids: [],
        turn_completed_ts: 1_000,
        audience_entity_id: null,
      });
      expect(pendingTraitFirst?.source_stream_entry_ids).toHaveLength(2);

      clock.advance(1_000);
      await borg.turn({
        userMessage: "Thanks!",
      });

      // Sprint 56: with liveExtraction off, the demonstrating turn's
      // stream entries do not resolve to an episode, so reinforcement
      // stays pending until extraction completes or TTL expires. Confirm
      // the attribution survived the second turn instead of getting
      // credited to unrelated retrieved memories.
      expect(borg.self.traits.list()).toEqual([]);
      expect(borg.workmem.load().pending_trait_attribution).toMatchObject({
        trait_label: "warm",
        source_episode_ids: [],
      });
    } finally {
      await borg.close();
    }
  });

  it("clears pending trait attribution without reinforcement on a non-positive follow-up", async () => {
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
      title: "Atlas status update",
      narrative: "Atlas needed a warmer explanation.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.9,
      tags: ["atlas", "tone"],
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

    const borg = await Borg.open({
      config: createTestConfig({
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
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createTurnPlanResponse(["ep_aaaaaaaaaaaaaaaa"]),
          {
            text: "Here is a warmer Atlas update.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update.",
          }),
          {
            text: "Understood.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      liveExtraction: false,
    });

    try {
      await borg.turn({
        userMessage: "Can you make the Atlas update sound warmer?",
        stakes: "high",
      });

      clock.advance(1_000);
      await borg.turn({
        userMessage: "Okay.",
      });

      expect(borg.self.traits.list()).toEqual([]);
    } finally {
      await borg.close();
    }
  });

  it("keeps pending trait attribution alive when no demonstrating-turn episode has been extracted yet", async () => {
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
      title: "Atlas status update",
      narrative: "Atlas needed a warmer explanation.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.9,
      tags: ["atlas", "tone"],
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

    const borg = await Borg.open({
      config: createTestConfig({
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
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createTurnPlanResponse(["ep_aaaaaaaaaaaaaaaa"]),
          {
            text: "Here is a warmer Atlas update.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update.",
          }),
          {
            text: "Still here.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      liveExtraction: false,
    });

    try {
      await borg.turn({
        userMessage: "Can you make the Atlas update sound warmer?",
        stakes: "high",
      });

      const pendingAfterFirst = borg.workmem.load().pending_trait_attribution;
      expect(pendingAfterFirst).not.toBeNull();

      // Sprint 56: with liveExtraction off, the demonstrating turn never
      // gets an episode; the next user turn cannot resolve evidence so
      // the attribution stays pending instead of crediting some unrelated
      // memory the planner happened to reference. TTL eventually expires
      // it (covered by a separate test).
      clock.advance(1_000);
      await borg.turn({
        userMessage: "Thanks!",
      });

      expect(borg.self.traits.list()).toEqual([]);
      expect(borg.workmem.load().pending_trait_attribution).toEqual(pendingAfterFirst);
    } finally {
      await borg.close();
    }
  });

  it("keeps pending trait attribution across an autonomous wake until the next user reply", async () => {
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
      title: "Atlas status update",
      narrative: "Atlas needed a warmer explanation.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.9,
      tags: ["atlas", "tone"],
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

    const borg = await Borg.open({
      config: createTestConfig({
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
        autonomy: {
          enabled: true,
          intervalMs: 60_000,
          maxWakesPerWindow: 6,
          budgetWindowMs: 86_400_000,
          executiveFocus: {
            enabled: false,
            stalenessSec: 86_400,
            dueLeadSec: 0,
          },
          triggers: {
            commitmentExpiring: {
              enabled: false,
              lookaheadMs: 86_400_000,
            },
            openQuestionDormant: {
              enabled: false,
              dormantMs: 604_800_000,
            },
            scheduledReflection: {
              enabled: true,
              intervalMs: 60_000,
            },
            goalFollowupDue: {
              enabled: false,
              lookaheadMs: 604_800_000,
              staleMs: 1_209_600_000,
            },
          },
          conditions: {
            commitmentRevoked: {
              enabled: false,
            },
            moodValenceDrop: {
              enabled: false,
              threshold: -0.5,
              windowN: 5,
              activationPeriodMs: 86_400_000,
            },
            openQuestionUrgencyBump: {
              enabled: false,
              threshold: 0.9,
            },
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
            text: "Here is a warmer Atlas update.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update.",
          }),
          {
            text: "Autonomous reflection.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Glad that helped.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      liveExtraction: false,
    });

    try {
      await borg.turn({
        userMessage: "Can you make the Atlas update sound warmer?",
        stakes: "high",
      });

      const pendingAfterFirst = borg.workmem.load().pending_trait_attribution;
      expect(pendingAfterFirst).not.toBeNull();

      clock.advance(1_000);
      const wakeResult = await borg.autonomy.scheduler.tick();
      expect(wakeResult.firedEvents).toBe(1);
      expect(borg.workmem.load().pending_trait_attribution).toEqual(pendingAfterFirst);
      expect(borg.self.traits.list()).toEqual([]);

      clock.advance(1_000);
      await borg.turn({
        userMessage: "I appreciate that, it was helpful.",
      });

      // Sprint 56: with liveExtraction off, the demonstrating turn never
      // gets an episode, so the next user reply cannot resolve evidence
      // and the trait stays pending instead of being credited to an
      // unrelated retrieved memory. The test still confirms the autonomous
      // wake didn't consume the attribution.
      expect(borg.self.traits.list()).toEqual([]);
      expect(borg.workmem.load().pending_trait_attribution).not.toBeNull();
    } finally {
      await borg.close();
    }
  });

  it("drops expired pending trait attribution and logs an internal event", async () => {
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
      title: "Atlas status update",
      narrative: "Atlas needed a warmer explanation.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.9,
      tags: ["atlas", "tone"],
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

    const borg = await Borg.open({
      config: createTestConfig({
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
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createTurnPlanResponse(["ep_aaaaaaaaaaaaaaaa"]),
          {
            text: "Here is a warmer Atlas update.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update.",
          }),
          {
            text: "Glad that helped.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      liveExtraction: false,
    });

    try {
      await borg.turn({
        userMessage: "Can you make the Atlas update sound warmer?",
        stakes: "high",
      });

      clock.advance(60 * 60 * 1_000 + 1);
      await borg.turn({
        userMessage: "I appreciate that, it was helpful.",
      });

      expect(borg.self.traits.list()).toEqual([]);
      expect(borg.stream.tail(8)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              kind: "trait_attribution_drop",
              reason: "expired",
            }),
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });

  it("drops pending trait attribution on audience mismatch and logs an internal event", async () => {
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
      title: "Atlas status update",
      narrative: "Atlas needed a warmer explanation.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.9,
      tags: ["atlas", "tone"],
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

    const borg = await Borg.open({
      config: createTestConfig({
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
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createTurnPlanResponse(["ep_aaaaaaaaaaaaaaaa"]),
          {
            text: "Here is a warmer Atlas update for Sam.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update for Sam.",
          }),
          {
            text: "Glad that helped.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Extra fallback.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      liveExtraction: false,
    });

    try {
      await borg.turn({
        userMessage: "Can you make the Atlas update sound warmer?",
        stakes: "high",
        audience: "Sam",
      });

      clock.advance(1_000);
      await borg.turn({
        userMessage: "I appreciate that, it was helpful.",
        audience: "Alex",
      });

      expect(borg.self.traits.list()).toEqual([]);
      expect(borg.stream.tail(10)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              kind: "trait_attribution_drop",
              reason: "audience_mismatch",
            }),
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });

  it("keeps mood neutral when only the agent response is enthusiastic", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
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
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "Amazing, great, wonderful progress! I'm thrilled this is working!",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
    });

    try {
      await borg.turn({
        userMessage: "Status update on Atlas build.",
      });

      expect(borg.mood.current("default" as never).valence).toBe(0);
      expect(borg.mood.history("default" as never)).toEqual([]);
    } finally {
      await borg.close();
    }
  });

  it("falls back to neutral affect when affective extraction fails and logs an internal event", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
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
      }),
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
          createEmptyReflectionResponse(),
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

  it("logs internal events when perception classifiers degrade", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: true,
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
      }),
      clock: new ManualClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createInvalidEntityClassifierResponse(),
          createInvalidModeClassifierResponse(),
          createNoTemporalCueResponse(),
          {
            text: "The turn still completed.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createEmptyReflectionResponse(),
        ],
      }),
    });

    try {
      const result = await borg.turn({
        userMessage: 'Talk to @alice about "Project Atlas".',
      });

      expect(result.response).toContain("still completed");
      // hot_entities is empty when entity-extractor classifier degrades:
      // the regex-heuristic fallback was removed in favor of LLM-only
      // extraction, so a failed LLM call yields empty entities rather
      // than false positives.
      expect(borg.workmem.load()).toMatchObject({
        mode: "idle",
        hot_entities: [],
      });
      expect(borg.stream.tail(10)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              hook: "perception_classifier",
              classifier: "entity_extractor",
              error: expect.stringContaining("Entity fallback returned invalid payload"),
            }),
          }),
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              hook: "perception_classifier",
              classifier: "mode_detector",
              error: expect.stringContaining("Mode fallback returned invalid payload"),
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
      emotional_arc: null,
      embedding: Float32Array.from([0, 1, 0, 0]),
      created_at: nowMs - 50 * 24 * 60 * 60 * 1_000,
      updated_at: nowMs - 50 * 24 * 60 * 60 * 1_000,
    });
    db.close();
    await store.close();

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
      }),
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
      expect((await borg.episodic.get("ep_cccccccccccccccc" as never))?.episode.id).toBeUndefined();

      const archiveAudit = audits.find((audit) => audit.action === "archive");
      expect(archiveAudit).toBeDefined();

      const reverted = await borg.audit.revert(archiveAudit!.id);
      expect(reverted?.reverted_at).not.toBeNull();
      expect((await borg.episodic.get("ep_cccccccccccccccc" as never))?.episode.id).toBe(
        "ep_cccccccccccccccc",
      );
    } finally {
      await borg.close();
    }
  });
});
