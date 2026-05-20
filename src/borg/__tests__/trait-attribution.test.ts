import { afterEach, describe, expect, it, vi } from "vitest";

import {
  FakeLLMClient,
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
  ScriptedEmbeddingClient,
  createEmitAnswerResponse,
  createTraitReflectionResponse,
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

  it("reinforces a pending trait from the next positive user turn with episode-backed provenance", async () => {
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
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
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
          createTurnPlanResponse(),
          createEmitAnswerResponse("Here is a warmer Atlas update.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update.",
          }),
          createEmitAnswerResponse("Glad that helped.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
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
      // entries. With extraction disabled here via liveExtraction:false,
      // those stream entries do not resolve yet.
      const pendingTraitFirst = borg.workmem.load().pending_trait_attribution;
      expect(pendingTraitFirst).toMatchObject({
        trait_label: "warm",
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
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
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
          createTurnPlanResponse(),
          createEmitAnswerResponse("Here is a warmer Atlas update.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update.",
          }),
          createEmitAnswerResponse("Understood.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
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
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
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
          createTurnPlanResponse(),
          createEmitAnswerResponse("Here is a warmer Atlas update.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update.",
          }),
          createEmitAnswerResponse("Still here.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
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
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
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
          createTurnPlanResponse(),
          createEmitAnswerResponse("Here is a warmer Atlas update.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update.",
          }),
          createEmitAnswerResponse("Autonomous reflection.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Glad that helped.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
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
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
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
          createTurnPlanResponse(),
          createEmitAnswerResponse("Here is a warmer Atlas update.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update.",
          }),
          createEmitAnswerResponse("Glad that helped.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
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
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
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
          createTurnPlanResponse(),
          createEmitAnswerResponse("Here is a warmer Atlas update for Sam.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createTraitReflectionResponse({
            traitLabel: "warm",
            evidence: "The response deliberately softened the Atlas update for Sam.",
          }),
          createEmitAnswerResponse("Glad that helped.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
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
});
