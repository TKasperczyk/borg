import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { SuppressionSet } from "../attention/index.js";
import { Reflector } from "./reflector.js";
import { FakeLLMClient } from "../../llm/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { selfMigrations } from "../../memory/self/migrations.js";
import { episodicMigrations } from "../../memory/episodic/migrations.js";
import { EpisodicRepository, createEpisodesTableSchema } from "../../memory/episodic/repository.js";
import {
  GoalsRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
} from "../../memory/self/index.js";
import { retrievalMigrations } from "../../retrieval/migrations.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import type { RetrievalConfidence, RetrievedEpisode } from "../../retrieval/index.js";
import { createEpisodeFixture, createOfflineTestHarness } from "../../offline/test-support.js";

function createReflectionResponse(
  advancedGoals: Array<{ goal_id: string; evidence: string }> = [],
) {
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
          advanced_goals: advancedGoals,
        },
      },
    ],
  };
}

function createRetrievedEpisode(
  episode: RetrievedEpisode["episode"],
  score = 0.8,
): RetrievedEpisode {
  return {
    episode,
    score,
    scoreBreakdown: {
      similarity: score,
      decayedSalience: 0.4,
      heat: 0.3,
      goalRelevance: 0,
      valueAlignment: 0,
      timeRelevance: 0,
      moodBoost: 0,
      socialRelevance: 0,
      entityRelevance: 0,
      suppressionPenalty: 0,
    },
    citationChain: [],
    semantic_context: {
      supports: [],
      contradicts: [],
      categories: [],
    },
  };
}

function createRetrievalConfidence(
  overrides: Partial<RetrievalConfidence> = {},
): RetrievalConfidence {
  return {
    overall: overrides.overall ?? 0.8,
    evidenceStrength: overrides.evidenceStrength ?? 0.8,
    coverage: overrides.coverage ?? 1,
    sourceDiversity: overrides.sourceDiversity ?? 1,
    contradictionPresent: overrides.contradictionPresent ?? false,
    sampleSize: overrides.sampleSize ?? 3,
  };
}

describe("reflector", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("bumps LLM-marked goal progress, marks episode use, skips S1 trait evidence, and ticks suppression", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new FixedClock(1_000);
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
    const episodicRepository = new EpisodicRepository({
      table,
      db,
      clock,
    });
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const traitsRepository = new TraitsRepository({
      db,
      clock,
    });
    const valuesRepository = new ValuesRepository({
      db,
      clock,
    });
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const goal = goalsRepository.add({
      description: "stabilize atlas release",
      priority: 5,
      provenance: { kind: "manual" },
    });
    const episode = await episodicRepository.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Atlas incident",
      narrative: "Atlas deployment failed.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.8,
      tags: ["atlas"],
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    const suppressionSet = SuppressionSet.fromEntries(
      [
        {
          id: "ep_stale",
          reason: "temporary",
          until_turn: 1,
        },
      ],
      1,
    );

    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse([
          {
            goal_id: goal.id,
            evidence: "Updated the Atlas release stabilization plan.",
          },
        ]),
      ],
    });
    const reflector = new Reflector({
      clock,
      llmClient: llm,
      model: "haiku",
    });
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.9,
      scoreBreakdown: {
        similarity: 0.9,
        decayedSalience: 0.3,
        heat: 1,
        goalRelevance: 0.2,
        timeRelevance: 0,
        suppressionPenalty: 0,
      },
      citationChain: [],
      semantic_context: {
        supports: [],
        contradicts: [],
        categories: [],
      },
    };
    const reflected = await reflector.reflect(
      {
        userMessage: "We need to stabilize the Atlas release",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: "Atlas",
          hot_entities: ["Atlas"],
          pending_intents: [],
          pending_trait_attribution: null,
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: valuesRepository.list(),
          goals: [goal],
          traits: traitsRepository.list(),
        },
        deliberationResult: {
          path: "system_1",
          response: "To stabilize the Atlas release, check the atlas deployment.",
          thoughts: ["brief thought"],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: null,
          thoughtsPersisted: true,
        },
        actionResult: {
          response: "To stabilize the Atlas release, check the atlas deployment.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: "Atlas",
            hot_entities: ["Atlas"],
            pending_intents: [],
            pending_trait_attribution: null,
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository,
        goalsRepository,
        traitsRepository,
        openQuestionsRepository,
        suppressionSet,
      },
      writer,
    );

    expect(goalsRepository.list({ status: "active" })[0]?.progress_notes).toContain(
      "Updated the Atlas release stabilization plan.",
    );
    expect(episodicRepository.getStats(episode.id)?.use_count).toBe(1);
    expect(traitsRepository.list()).toEqual([]);
    expect(reflected.pending_trait_attribution).toBeNull();
    expect(suppressionSet.isSuppressed(episode.id)).toBe(true);
    expect(suppressionSet.isSuppressed("ep_stale")).toBe(false);
    // Phase E removed scratchpad/recent_thoughts from working memory. The
    // reflector no longer clears scratchpad or pushes thoughts into the
    // cache; thoughts live in the stream (persisted by the deliberator),
    // and working memory holds derived live-turn state only.
    expect(reflected.turn_counter).toBe(1);
    expect(reflected.current_focus).toBe("Atlas");
  });

  it("queues review instead of silently overwriting episode-backed active goals from offline reflection", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(4_000),
    });
    cleanup.push(harness.cleanup);

    const goal = harness.goalsRepository.add({
      description: "stabilize atlas release",
      priority: 5,
      provenance: {
        kind: "episodes",
        episode_ids: ["ep_aaaaaaaaaaaaaaaa" as const],
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse([
          {
            goal_id: goal.id,
            evidence: "Updated the deployment checklist.",
          },
        ]),
      ],
    });
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
    });

    await reflector.reflect(
      {
        userMessage: "We need to stabilize the Atlas release",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: "Atlas",
          hot_entities: ["Atlas"],
          pending_intents: [],
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: harness.valuesRepository.list(),
          goals: [goal],
          traits: harness.traitsRepository.list(),
        },
        deliberationResult: {
          path: "system_1",
          response: "To stabilize the Atlas release, update the deployment checklist.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          thoughtsPersisted: true,
        },
        actionResult: {
          response: "To stabilize the Atlas release, update the deployment checklist.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: "Atlas",
            hot_entities: ["Atlas"],
            pending_intents: [],
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        identityService: harness.identityService,
        reviewQueueRepository: harness.reviewQueueRepository,
        suppressionSet: new SuppressionSet(),
      },
      harness.streamWriter,
    );

    expect(harness.goalsRepository.get(goal.id)?.progress_notes).toBeNull();
    expect(harness.reviewQueueRepository.getOpen()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "identity_inconsistency",
          refs: expect.objectContaining({
            target_type: "goal",
            target_id: goal.id,
            repair_op: "patch",
            proposed_provenance: {
              kind: "online",
              process: "reflector",
            },
          }),
        }),
      ]),
    );
  });

  it("does not update goal progress when reflection output is empty even if text overlaps", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(5_000),
    });
    cleanup.push(harness.cleanup);

    const goal = harness.goalsRepository.add({
      description: "stabilize atlas release",
      priority: 5,
      provenance: { kind: "manual" },
    });
    const llm = new FakeLLMClient({
      responses: [createReflectionResponse()],
    });
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
    });

    await reflector.reflect(
      {
        userMessage: "We need to stabilize the Atlas release.",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: "Atlas",
          hot_entities: ["Atlas"],
          pending_intents: [],
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: harness.valuesRepository.list(),
          goals: [goal],
          traits: harness.traitsRepository.list(),
        },
        deliberationResult: {
          path: "system_1",
          response: "To stabilize the Atlas release, we should discuss the risk.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          thoughtsPersisted: true,
        },
        actionResult: {
          response: "To stabilize the Atlas release, we should discuss the risk.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: "Atlas",
            hot_entities: ["Atlas"],
            pending_intents: [],
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        suppressionSet: new SuppressionSet(),
      },
      harness.streamWriter,
    );

    expect(harness.goalsRepository.get(goal.id)?.progress_notes).toBeNull();
  });

  it("counts an episode as used when the response echoes title or narrative tokens", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new FixedClock(1_000);
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
    const episodicRepository = new EpisodicRepository({
      table,
      db,
      clock,
    });
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const traitsRepository = new TraitsRepository({
      db,
      clock,
    });
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episode = await episodicRepository.insert({
      id: "ep_bbbbbbbbbbbbbbbb" as never,
      title: "Unexpected rollback plan",
      narrative: "Database migration blocked the release and required a rollback.",
      participants: ["ops-team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_bbbbbbbbbbbbbbbb" as never],
      significance: 0.8,
      tags: ["ops"],
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    const suppressionSet = new SuppressionSet(1);
    const reflector = new Reflector({
      clock,
    });
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.9,
      scoreBreakdown: {
        similarity: 0.9,
        decayedSalience: 0.3,
        heat: 1,
        goalRelevance: 0.2,
        timeRelevance: 0,
        suppressionPenalty: 0,
      },
      citationChain: [],
      semantic_context: {
        supports: [],
        contradicts: [],
        categories: [],
      },
    };

    await reflector.reflect(
      {
        userMessage: "How should we recover the release?",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: null,
          hot_entities: [],
          pending_intents: [],
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "The database migration blocked the release, so prepare a rollback plan.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "The database migration blocked the release, so prepare a rollback plan.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: null,
            hot_entities: [],
            pending_intents: [],
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository,
        goalsRepository,
        traitsRepository,
        openQuestionsRepository,
        suppressionSet,
      },
      writer,
    );

    expect(episodicRepository.getStats(episode.id)?.use_count).toBe(1);
  });

  it("adds an open question when high relevance score has low retrieval confidence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new FixedClock(1_000);
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
    const episodicRepository = new EpisodicRepository({
      table,
      db,
      clock,
    });
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const traitsRepository = new TraitsRepository({
      db,
      clock,
    });
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episode = await episodicRepository.insert({
      id: "ep_cccccccccccccccc" as never,
      title: "Atlas uncertainty",
      narrative: "The logs were incomplete and the root cause stayed unclear.",
      participants: ["Atlas"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_cccccccccccccccc" as never],
      significance: 0.5,
      tags: ["atlas"],
      confidence: 0.4,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });

    const reflector = new Reflector({
      clock,
    });
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.92,
      scoreBreakdown: {
        similarity: 0.9,
        decayedSalience: 0.1,
        heat: 1,
        goalRelevance: 0,
        timeRelevance: 0,
        suppressionPenalty: 0,
      },
      citationChain: [],
      semantic_context: {
        supports: [],
        contradicts: [],
        categories: [],
      },
    };

    await reflector.reflect(
      {
        userMessage: "Why is Atlas still failing?",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: "Atlas",
          hot_entities: ["Atlas"],
          pending_intents: [],
          suppressed: [],
          mode: "reflective",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "I still need to compare more evidence.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "low confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: null,
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "I still need to compare more evidence.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: "Atlas",
            hot_entities: ["Atlas"],
            pending_intents: [],
            suppressed: [],
            mode: "reflective",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence({
          overall: 0.2,
          evidenceStrength: 0.1,
          coverage: 0.2,
          sampleSize: 1,
        }),
        episodicRepository,
        goalsRepository,
        traitsRepository,
        openQuestionsRepository,
        suppressionSet: new SuppressionSet(1),
      },
      writer,
    );

    expect(openQuestionsRepository.list({ status: "open" })).toEqual([
      expect.objectContaining({
        source: "reflection",
        related_episode_ids: [episode.id],
        provenance: {
          kind: "episodes",
          episode_ids: [episode.id],
        },
      }),
    ]);
  });

  it("does not add an open question when low relevance score has high retrieval confidence", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture({
        title: "Atlas settled cause",
        narrative: "Atlas failures were traced to a known rollback gap.",
        tags: ["atlas"],
      }),
    );
    const reflector = new Reflector({
      clock: harness.clock,
    });
    const retrieved = createRetrievedEpisode(episode, 0.1);

    await reflector.reflect(
      {
        userMessage: "Why is Atlas still failing?",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: "Atlas",
          hot_entities: ["Atlas"],
          pending_intents: [],
          suppressed: [],
          mode: "reflective",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "The rollback gap explains the failure.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "low score",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: null,
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "The rollback gap explains the failure.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: "Atlas",
            hot_entities: ["Atlas"],
            pending_intents: [],
            suppressed: [],
            mode: "reflective",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence({
          overall: 0.82,
          evidenceStrength: 0.82,
        }),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([]);
  });

  it("logs and continues when the reflection open-question hook fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new FixedClock(1_000);
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
    const episodicRepository = new EpisodicRepository({
      table,
      db,
      clock,
    });
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const traitsRepository = new TraitsRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episode = await episodicRepository.insert({
      id: "ep_dddddddddddddddd" as never,
      title: "Atlas unknowns",
      narrative: "Atlas remains uncertain.",
      participants: ["Atlas"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_dddddddddddddddd" as never],
      significance: 0.4,
      tags: ["atlas"],
      confidence: 0.4,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    const reflector = new Reflector({
      clock,
    });
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.2,
      scoreBreakdown: {
        similarity: 0.2,
        decayedSalience: 0.2,
        heat: 0.1,
        goalRelevance: 0,
        timeRelevance: 0,
        suppressionPenalty: 0,
      },
      citationChain: [],
      semantic_context: {
        supports: [],
        contradicts: [],
        categories: [],
      },
    };
    const brokenOpenQuestionsRepository = {
      add() {
        throw new Error("hook exploded");
      },
    } as unknown as OpenQuestionsRepository;

    const reflected = await reflector.reflect(
      {
        userMessage: "Why is Atlas still failing?",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: "Atlas",
          hot_entities: ["Atlas"],
          pending_intents: [],
          suppressed: [],
          mode: "reflective",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "I still need to compare more evidence.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "low confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: null,
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "I still need to compare more evidence.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: "Atlas",
            hot_entities: ["Atlas"],
            pending_intents: [],
            suppressed: [],
            mode: "reflective",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence({
          overall: 0.2,
          evidenceStrength: 0.2,
          coverage: 0.2,
          sampleSize: 1,
        }),
        episodicRepository,
        goalsRepository,
        traitsRepository,
        openQuestionsRepository: brokenOpenQuestionsRepository,
        suppressionSet: new SuppressionSet(1),
      },
      writer,
    );

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(1);

    expect(reflected.turn_counter).toBe(1);
    expect(entries[0]).toMatchObject({
      kind: "internal_event",
      content: {
        hook: "reflection_open_question",
      },
    });
  });

  it("keeps a newly surfaced skill pending and records explicit follow-up failure once", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = createEpisodeFixture({
      title: "Rust lifetime frustration",
      narrative: "Rust lifetime errors kept blocking progress.",
      tags: ["rust", "lifetimes"],
    });
    await harness.episodicRepository.insert(episode);

    const skill = await harness.skillRepository.add({
      applies_when: "Rust lifetime debugging",
      approach: "Shrink borrow scopes and use intermediate bindings.",
      sourceEpisodes: [episode.id],
    });
    const reflector = new Reflector({
      clock: harness.clock,
    });
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.8,
      scoreBreakdown: {
        similarity: 0.8,
        decayedSalience: 0.4,
        heat: 0.3,
        goalRelevance: 0,
        timeRelevance: 0,
        moodBoost: 0,
        socialRelevance: 0,
        suppressionPenalty: 0,
      },
      citationChain: [],
      semantic_context: {
        supports: [],
        contradicts: [],
        categories: [],
      },
    };

    const firstTurnMemory = await reflector.reflect(
      {
        userMessage: "I hit a Rust lifetime error again.",
        perception: {
          entities: ["Rust"],
          mode: "problem_solving",
          affectiveSignal: {
            valence: -0.7,
            arousal: 0.5,
            dominant_emotion: "anger",
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: "Rust",
          hot_entities: ["Rust"],
          pending_intents: [],
          suppressed: [],
          mood: null,
          last_selected_skill_id: null,
          last_selected_skill_turn: null,
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "Try shrinking the borrow scope.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Try shrinking the borrow scope.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: "Rust",
            hot_entities: ["Rust"],
            pending_intents: [],
            suppressed: [],
            mood: null,
            last_selected_skill_id: null,
            last_selected_skill_turn: null,
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        skillRepository: harness.skillRepository,
        selectedSkillId: skill.id,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(harness.skillRepository.get(skill.id)?.attempts).toBe(0);
    expect(firstTurnMemory.last_selected_skill_id).toBe(skill.id);

    const secondTurnMemory = await reflector.reflect(
      {
        userMessage: "That didn't work; I'm still failing with the Rust lifetime error.",
        perception: {
          entities: ["Rust"],
          mode: "problem_solving",
          affectiveSignal: {
            valence: -0.6,
            arousal: 0.5,
            dominant_emotion: "anger",
          },
          temporalCue: null,
        },
        workingMemory: {
          ...firstTurnMemory,
          turn_counter: 2,
          updated_at: 1,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "Let's inspect the borrow spans next.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Let's inspect the borrow spans next.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            ...firstTurnMemory,
            turn_counter: 2,
            updated_at: 1,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        skillRepository: harness.skillRepository,
        selectedSkillId: null,
        suppressionSet: new SuppressionSet(2),
      },
      harness.streamWriter,
    );

    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      attempts: 1,
      failures: 1,
      beta: 2,
    });
    expect(secondTurnMemory.last_selected_skill_id).toBeNull();
  });

  it("does not attribute outcomes from assistant wording or generic negative affect", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = createEpisodeFixture({
      title: "Rust lifetime frustration",
      narrative: "Rust lifetime errors kept blocking progress.",
      tags: ["rust", "lifetimes"],
    });
    await harness.episodicRepository.insert(episode);

    const skill = await harness.skillRepository.add({
      applies_when: "Rust lifetime debugging",
      approach: "Shrink borrow scopes and use intermediate bindings.",
      sourceEpisodes: [episode.id],
    });
    const reflector = new Reflector({
      clock: harness.clock,
    });
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.8,
      scoreBreakdown: {
        similarity: 0.8,
        decayedSalience: 0.4,
        heat: 0.3,
        goalRelevance: 0,
        timeRelevance: 0,
        moodBoost: 0,
        socialRelevance: 0,
        suppressionPenalty: 0,
      },
      citationChain: [],
      semantic_context: {
        supports: [],
        contradicts: [],
        categories: [],
      },
    };

    const firstTurnMemory = await reflector.reflect(
      {
        userMessage: "I hit a Rust lifetime issue again.",
        perception: {
          entities: ["Rust"],
          mode: "problem_solving",
          affectiveSignal: {
            valence: -0.5,
            arousal: 0.5,
            dominant_emotion: "anger",
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: "Rust",
          hot_entities: ["Rust"],
          pending_intents: [],
          suppressed: [],
          mood: null,
          last_selected_skill_id: null,
          last_selected_skill_turn: null,
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "This works when you shrink the borrow scope.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "This works when you shrink the borrow scope.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: "Rust",
            hot_entities: ["Rust"],
            pending_intents: [],
            suppressed: [],
            mood: null,
            last_selected_skill_id: null,
            last_selected_skill_turn: null,
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        skillRepository: harness.skillRepository,
        selectedSkillId: skill.id,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(harness.skillRepository.get(skill.id)?.attempts).toBe(0);
    expect(firstTurnMemory.last_selected_skill_id).toBe(skill.id);

    const secondTurnMemory = await reflector.reflect(
      {
        userMessage: "I'm frustrated and tired of this.",
        perception: {
          entities: ["Rust"],
          mode: "problem_solving",
          affectiveSignal: {
            valence: -0.8,
            arousal: 0.3,
            dominant_emotion: "sadness",
          },
          temporalCue: null,
        },
        workingMemory: {
          ...firstTurnMemory,
          turn_counter: 2,
          updated_at: 1,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "Let's narrow the borrow lifetime a bit more.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: null,
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Let's narrow the borrow lifetime a bit more.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            ...firstTurnMemory,
            turn_counter: 2,
            updated_at: 1,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        skillRepository: harness.skillRepository,
        selectedSkillId: null,
        suppressionSet: new SuppressionSet(2),
      },
      harness.streamWriter,
    );

    expect(harness.skillRepository.get(skill.id)?.attempts).toBe(0);
    expect(secondTurnMemory.last_selected_skill_id).toBe(skill.id);
  });

  it("uses planner-referenced retrieved episodes for S2 trait evidence", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episodeA = await harness.episodicRepository.insert(
      createEpisodeFixture({
        id: "ep_aaaaaaaaaaaaaaaa" as never,
        title: "Planning sync A",
      }),
    );
    const episodeB = await harness.episodicRepository.insert(
      createEpisodeFixture({
        id: "ep_bbbbbbbbbbbbbbbb" as never,
        title: "Planning sync B",
      }),
    );
    const retrievedA = createRetrievedEpisode(episodeA);
    const retrievedB = createRetrievedEpisode(episodeB);
    const reflector = new Reflector({
      clock: harness.clock,
    });

    const reflected = await reflector.reflect(
      {
        userMessage: "Let's work through the plan.",
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: null,
          hot_entities: [],
          pending_intents: [],
          pending_trait_attribution: null,
          suppressed: [],
          mood: null,
          last_selected_skill_id: null,
          last_selected_skill_turn: null,
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "Use both planning syncs as evidence.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "reflective",
          retrievedEpisodes: [retrievedA, retrievedB],
          referencedEpisodeIds: [episodeA.id, episodeB.id],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Use both planning syncs as evidence.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: null,
            hot_entities: [],
            pending_intents: [],
            pending_trait_attribution: null,
            suppressed: [],
            mood: null,
            last_selected_skill_id: null,
            last_selected_skill_turn: null,
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrievedA, retrievedB],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(reflected.pending_trait_attribution).toMatchObject({
      trait_label: "engaged",
      source_episode_ids: [episodeA.id, episodeB.id],
    });
  });

  it("does not attach S2 trait evidence when the planner referenced no episodes", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.insert(createEpisodeFixture());
    const retrieved = createRetrievedEpisode(episode);
    const reflector = new Reflector({
      clock: harness.clock,
    });

    const reflected = await reflector.reflect(
      {
        userMessage: "Let's work through the plan.",
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: null,
          hot_entities: [],
          pending_intents: [],
          pending_trait_attribution: null,
          suppressed: [],
          mood: null,
          last_selected_skill_id: null,
          last_selected_skill_turn: null,
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "No episode evidence was needed.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "reflective",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "No episode evidence was needed.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: null,
            hot_entities: [],
            pending_intents: [],
            pending_trait_attribution: null,
            suppressed: [],
            mood: null,
            last_selected_skill_id: null,
            last_selected_skill_turn: null,
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(reflected.pending_trait_attribution).toBeNull();
  });

  it("filters S2 planner-referenced trait evidence to episodes retrieved this turn", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture({
        id: "ep_aaaaaaaaaaaaaaaa" as never,
      }),
    );
    const retrieved = createRetrievedEpisode(episode);
    const reflector = new Reflector({
      clock: harness.clock,
    });

    const reflected = await reflector.reflect(
      {
        userMessage: "Let's work through the plan.",
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: null,
          hot_entities: [],
          pending_intents: [],
          pending_trait_attribution: null,
          suppressed: [],
          mood: null,
          last_selected_skill_id: null,
          last_selected_skill_turn: null,
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "Only the retrieved episode should count.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "reflective",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id, "ep_bbbbbbbbbbbbbbbb"],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Only the retrieved episode should count.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: null,
            hot_entities: [],
            pending_intents: [],
            pending_trait_attribution: null,
            suppressed: [],
            mood: null,
            last_selected_skill_id: null,
            last_selected_skill_turn: null,
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(reflected.pending_trait_attribution).toMatchObject({
      trait_label: "engaged",
      source_episode_ids: [episode.id],
    });
  });

  it("does not reinforce traits when no episodes were retrieved", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const reflector = new Reflector({
      clock: harness.clock,
    });

    await reflector.reflect(
      {
        userMessage: "Thinking out loud.",
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: null,
          hot_entities: [],
          pending_intents: [],
          pending_trait_attribution: null,
          suppressed: [],
          mood: null,
          last_selected_skill_id: null,
          last_selected_skill_turn: null,
          mode: "reflective",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "Staying with it.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Staying with it.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: null,
            hot_entities: [],
            pending_intents: [],
            pending_trait_attribution: null,
            suppressed: [],
            mood: null,
            last_selected_skill_id: null,
            last_selected_skill_turn: null,
            mode: "reflective",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(harness.traitsRepository.list()).toEqual([]);
  });

  it("maps reflective mode reinforcement onto introspective pending attribution", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = createEpisodeFixture({
      title: "Reflective walk",
      narrative: "A slow reflective walk helped untangle a hard feeling.",
      tags: ["reflection"],
    });
    await harness.episodicRepository.insert(episode);

    const reflector = new Reflector({
      clock: harness.clock,
    });
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.7,
      scoreBreakdown: {
        similarity: 0.7,
        decayedSalience: 0.4,
        heat: 0.2,
        goalRelevance: 0,
        valueAlignment: 0,
        timeRelevance: 0,
        moodBoost: 0,
        socialRelevance: 0,
        suppressionPenalty: 0,
      },
      citationChain: [],
      semantic_context: {
        supports: [],
        contradicts: [],
        categories: [],
      },
    };

    const reflected = await reflector.reflect(
      {
        userMessage: "I want to sit with this feeling for a minute.",
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: -0.2,
            arousal: 0.1,
            dominant_emotion: "sadness",
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: null,
          hot_entities: [],
          pending_intents: [],
          pending_trait_attribution: null,
          suppressed: [],
          mood: null,
          last_selected_skill_id: null,
          last_selected_skill_turn: null,
          mode: "reflective",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "Let me stay with it and trace what keeps resurfacing.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Let me stay with it and trace what keeps resurfacing.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: null,
            hot_entities: [],
            pending_intents: [],
            pending_trait_attribution: null,
            suppressed: [],
            mood: null,
            last_selected_skill_id: null,
            last_selected_skill_turn: null,
            mode: "reflective",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(harness.traitsRepository.list()).toEqual([]);
    expect(reflected.pending_trait_attribution).toMatchObject({
      trait_label: "introspective",
      source_episode_ids: [episode.id],
      audience_entity_id: null,
    });
  });

  it("does not queue pending trait attribution for autonomous turns", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const reflector = new Reflector({
      clock: harness.clock,
    });
    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture({
        title: "Internal reflective note",
        narrative: "A private autonomous reflection about an earlier feeling.",
      }),
    );
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.7,
      scoreBreakdown: {
        similarity: 0.7,
        decayedSalience: 0.4,
        heat: 0.2,
        goalRelevance: 0,
        valueAlignment: 0,
        timeRelevance: 0,
        moodBoost: 0,
        socialRelevance: 0,
        entityRelevance: 0,
        suppressionPenalty: 0,
      },
      citationChain: [],
      semantic_context: {
        supports: [],
        contradicts: [],
        categories: [],
      },
    };

    const reflected = await reflector.reflect(
      {
        origin: "autonomous",
        userMessage: "Let me sit with this for a moment.",
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: 0,
            arousal: 0.1,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          current_focus: null,
          hot_entities: [],
          pending_intents: [],
          pending_trait_attribution: null,
          suppressed: [],
          mood: null,
          last_selected_skill_id: null,
          last_selected_skill_turn: null,
          mode: "reflective",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "I'll keep tracing this privately.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: null,
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "I'll keep tracing this privately.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            current_focus: null,
            hot_entities: [],
            pending_intents: [],
            pending_trait_attribution: null,
            suppressed: [],
            mood: null,
            last_selected_skill_id: null,
            last_selected_skill_turn: null,
            mode: "reflective",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        episodicRepository: harness.episodicRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(harness.traitsRepository.list()).toEqual([]);
    expect(reflected.pending_trait_attribution).toBeNull();
  });
});
