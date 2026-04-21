import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { SuppressionSet } from "../attention/index.js";
import { Reflector } from "./reflector.js";
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
import type { RetrievedEpisode } from "../../retrieval/index.js";
import { createEpisodeFixture, createOfflineTestHarness } from "../../offline/test-support.js";

describe("reflector", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("bumps goal progress, marks episode use, and ticks suppression", async () => {
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
    const reflected = await reflector.reflect(
      {
        userMessage: "We need to stabilize the Atlas release",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          scratchpad: "thinking",
          current_focus: "Atlas",
          recent_thoughts: [],
          hot_entities: ["Atlas"],
          pending_intents: [],
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
          thoughtsPersisted: true,
        },
        actionResult: {
          response: "To stabilize the Atlas release, check the atlas deployment.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            scratchpad: "brief thought",
            current_focus: "Atlas",
            recent_thoughts: [],
            hot_entities: ["Atlas"],
            pending_intents: [],
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        episodicRepository,
        goalsRepository,
        traitsRepository,
        openQuestionsRepository,
        suppressionSet,
      },
      writer,
    );

    expect(goalsRepository.list({ status: "active" })[0]?.progress_notes).toContain(
      "Heuristic turn progress",
    );
    expect(episodicRepository.getStats(episode.id)?.use_count).toBe(1);
    expect(traitsRepository.list()[0]?.label).toBe("engaged");
    expect(suppressionSet.isSuppressed(episode.id)).toBe(true);
    expect(suppressionSet.isSuppressed("ep_stale")).toBe(false);
    expect(reflected.scratchpad).toBe("");
    expect(reflected.recent_thoughts).toContain("brief thought");
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
          scratchpad: "",
          current_focus: null,
          recent_thoughts: [],
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
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "The database migration blocked the release, so prepare a rollback plan.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            scratchpad: "",
            current_focus: null,
            recent_thoughts: [],
            hot_entities: [],
            pending_intents: [],
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
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

  it("adds an open question when system 2 finishes with low retrieval confidence", async () => {
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

    await reflector.reflect(
      {
        userMessage: "Why is Atlas still failing?",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          scratchpad: "",
          current_focus: "Atlas",
          recent_thoughts: [],
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
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "I still need to compare more evidence.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            scratchpad: "",
            current_focus: "Atlas",
            recent_thoughts: [],
            hot_entities: ["Atlas"],
            pending_intents: [],
            suppressed: [],
            mode: "reflective",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
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
      }),
    ]);
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
          scratchpad: "",
          current_focus: "Atlas",
          recent_thoughts: [],
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
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "I still need to compare more evidence.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            scratchpad: "",
            current_focus: "Atlas",
            recent_thoughts: [],
            hot_entities: ["Atlas"],
            pending_intents: [],
            suppressed: [],
            mode: "reflective",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
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

    expect(reflected.scratchpad).toBe("");
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
          scratchpad: "",
          current_focus: "Rust",
          recent_thoughts: [],
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
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Try shrinking the borrow scope.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            scratchpad: "",
            current_focus: "Rust",
            recent_thoughts: [],
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
          scratchpad: "",
          current_focus: "Rust",
          recent_thoughts: [],
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
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "This works when you shrink the borrow scope.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            scratchpad: "",
            current_focus: "Rust",
            recent_thoughts: [],
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
});
