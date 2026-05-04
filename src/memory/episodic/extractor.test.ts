import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { FakeLLMClient, type LLMCompleteOptions } from "../../llm/index.js";
import { createOfflineTestHarness } from "../../offline/test-support.js";
import { StreamWriter } from "../../stream/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { EmbeddingError, LLMError } from "../../util/errors.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { retrievalMigrations } from "../../retrieval/migrations.js";
import { commitmentMigrations, EntityRepository } from "../commitments/index.js";
import { RelationalSlotRepository, relationalSlotMigrations } from "../relational-slots/index.js";
import { selfMigrations } from "../self/migrations.js";
import { createWorkingMemory, WorkingMemoryStore } from "../working/index.js";
import { episodicMigrations } from "./migrations.js";
import { EpisodicExtractor } from "./extractor.js";
import { EpisodicRepository, createEpisodesTableSchema } from "./repository.js";

const EPISODE_TOOL_NAME = "EmitEpisodeCandidates";

function createEpisodeToolResponse(episodes: unknown[], relationalSlotUpdates: unknown[] = []) {
  return {
    text: "",
    input_tokens: 10,
    output_tokens: 20,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: EPISODE_TOOL_NAME,
        input: {
          episodes: episodes.map((episode) =>
            typeof episode === "object" && episode !== null && !("location" in episode)
              ? { ...episode, location: null }
              : episode,
          ),
          relational_slot_updates: relationalSlotUpdates,
        },
      },
    ],
  };
}

class TitleEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    if (text.includes("Planning sync")) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    return Float32Array.from([0, 1, 0, 0]);
  }
}

class FailingOnceEmbeddingClient implements EmbeddingClient {
  private failed = false;

  async embed(text: string): Promise<Float32Array> {
    if (!this.failed && text.includes("Skip me")) {
      this.failed = true;
      throw new EmbeddingError("embedding failed");
    }

    return Float32Array.from([1, 0, 0, 0]);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map((text) => this.embed(text)));
  }
}

describe("episodic extractor", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  async function createRelationalExtractorHarness(clock = new ManualClock(1_000)) {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        commitmentMigrations,
        relationalSlotMigrations,
      ),
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
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const relationalSlotRepository = new RelationalSlotRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    return {
      tempDir,
      clock,
      repo,
      entityRepository,
      relationalSlotRepository,
      writer,
    };
  }

  it("keeps repeated similar episodes on different days as distinct episodes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const first = await writer.append({
      kind: "user_msg",
      content: "We reviewed the borg architecture and memory bands together.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Reviewed borg architecture",
            narrative:
              "We reviewed the borg architecture and discussed the memory bands. The conversation focused on how the pieces fit together.",
            source_stream_ids: [first.id],
            participants: ["team"],
            tags: ["architecture", "borg"],
            confidence: 0.8,
            significance: 0.7,
          },
        ]),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      entityRepository,
      clock,
    });

    const firstRun = await extractor.extractFromStream();
    clock.advance(4 * 24 * 60 * 60 * 1_000);
    const second = await writer.append({
      kind: "agent_msg",
      content:
        "We reviewed the borg architecture again and compared the retrieval pipeline changes.",
    });
    llm.pushResponse(
      createEpisodeToolResponse([
        {
          title: "Reviewed borg architecture",
          narrative:
            "We revisited the borg architecture and compared the retrieval pipeline changes. This was a later review, not the original conversation.",
          source_stream_ids: [second.id],
          participants: ["team", "pm"],
          tags: ["architecture", "retrieval"],
          confidence: 0.9,
          significance: 0.9,
        },
      ]),
    );
    const secondRun = await extractor.extractFromStream({
      sinceTs: second.timestamp,
    });
    const listed = await repo.listAll();

    expect(firstRun).toEqual({
      inserted: 1,
      updated: 0,
      skipped: 0,
    });
    expect(secondRun).toEqual({
      inserted: 1,
      updated: 0,
      skipped: 0,
    });
    expect(listed).toHaveLength(2);
    expect(listed.map((episode) => episode.source_stream_ids)).toEqual(
      expect.arrayContaining([[first.id], [second.id]]),
    );
    expect(listed.map((episode) => episode.start_time)).toEqual(
      expect.arrayContaining([first.timestamp, second.timestamp]),
    );
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: EPISODE_TOOL_NAME,
    });
  });

  it("filters internal scaffolding out of episodic extraction chunks", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const user = await writer.append({
      kind: "user_msg",
      content: "Sam asked about the Atlas deployment issue.",
    });
    clock.advance(10);
    await writer.append({
      kind: "thought",
      content: "internal plan: inspect pnpm lockfile before answering",
    });
    clock.advance(10);
    await writer.append({
      kind: "internal_event",
      content: {
        kind: "debug_hook",
        detail: "non-conversational scaffolding",
      },
    });
    clock.advance(10);
    const perception = await writer.append({
      kind: "perception",
      content: {
        mode: "problem_solving",
        entities: ["Atlas"],
        temporalCue: null,
        affectiveSignal: {
          valence: -0.3,
          arousal: 0.4,
          dominant_emotion: "anger",
        },
      },
    });
    clock.advance(10);
    await writer.append({
      kind: "tool_call",
      content: {
        call_id: "call_1",
        tool_name: "tool.test.echo",
        input: {
          value: "Atlas",
        },
        origin: "deliberator",
      },
    });
    clock.advance(10);
    const agent = await writer.append({
      kind: "agent_msg",
      content: "I suggested rerunning pnpm install before the next deploy.",
    });

    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Atlas deploy debugging",
            narrative:
              "Sam asked about the Atlas deployment issue. I suggested rerunning pnpm install before the next deploy.",
            source_stream_ids: [user.id, agent.id],
            participants: ["Sam"],
            tags: ["atlas", "deploy"],
            confidence: 0.8,
            significance: 0.7,
          },
        ]),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      entityRepository,
      clock,
    });

    const result = await extractor.extractFromStream();
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    const listed = await repo.listAll();

    expect(result).toEqual({
      inserted: 1,
      updated: 0,
      skipped: 0,
    });
    expect(prompt).not.toContain("internal plan");
    expect(prompt).toContain("<perception_context>");
    expect(prompt).toContain('"mode":"problem_solving"');
    expect(prompt).toContain('"entities":["Atlas"]');
    expect(prompt).toContain('"affectiveSignal"');
    expect(prompt).not.toContain('"tool_call"');
    expect(prompt).not.toContain("non-conversational scaffolding");
    expect(listed[0]?.source_stream_ids).toEqual([user.id, agent.id]);
    expect(listed[0]?.source_stream_ids).not.toContain(perception.id);
  });

  it("applies relational slot updates emitted with episodic extraction", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new ManualClock(1_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        commitmentMigrations,
        relationalSlotMigrations,
      ),
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
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const relationalSlotRepository = new RelationalSlotRepository({
      db,
      clock,
    });
    const tom = entityRepository.resolve("Tom");
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const user = await writer.append({
      kind: "user_msg",
      content: "My partner's name is Sarah.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [
            {
              title: "Tom named his partner",
              narrative: "Tom said his partner's name is Sarah.",
              source_stream_ids: [user.id],
              participants: ["Tom"],
              tags: ["relationship"],
              confidence: 0.9,
              significance: 0.7,
            },
          ],
          [
            {
              subject_entity_id: tom,
              slot_key: "partner.name",
              asserted_value: "Sarah",
              source_stream_entry_ids: [user.id],
            },
          ],
        ),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      entityRepository,
      relationalSlotRepository,
      defaultUser: "Tom",
      clock,
    });

    const result = await extractor.extractFromStream();
    const slot = relationalSlotRepository.findBySubjectAndKey(tom, "partner.name");
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(result.inserted).toBe(1);
    expect(prompt).toContain("<relational_slot_subjects>");
    expect(prompt).toContain(tom);
    expect(slot).toMatchObject({
      subject_entity_id: tom,
      slot_key: "partner.name",
      value: "Sarah",
      state: "established",
      evidence_stream_entry_ids: [user.id],
    });
  });

  it("resolves bare user relational slot subjects to the human audience entity", async () => {
    const harness = await createRelationalExtractorHarness();
    const user = await harness.writer.append({
      kind: "user_msg",
      content: "My dog's name is Otto.",
      audience: "Tom",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [
            {
              title: "Tom named his dog",
              narrative: "Tom said his dog's name is Otto.",
              source_stream_ids: [user.id],
              participants: ["Tom"],
              tags: ["dog"],
              confidence: 0.9,
              significance: 0.7,
            },
          ],
          [
            {
              subject_entity_id: "user",
              slot_key: "dog.name",
              asserted_value: "Otto",
              source_stream_entry_ids: [user.id],
            },
          ],
        ),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      entityRepository: harness.entityRepository,
      relationalSlotRepository: harness.relationalSlotRepository,
      clock: harness.clock,
    });

    await extractor.extractFromStream();

    const tom = harness.entityRepository.resolve("Tom");
    const fallbackUser = harness.entityRepository.resolve("user");

    expect(harness.relationalSlotRepository.findBySubjectAndKey(tom, "dog.name")).toMatchObject({
      subject_entity_id: tom,
      slot_key: "dog.name",
      value: "Otto",
      evidence_stream_entry_ids: [user.id],
    });
    expect(
      harness.relationalSlotRepository.findBySubjectAndKey(fallbackUser, "dog.name"),
    ).toBeNull();
  });

  it("keeps bare user relational slot subjects on the default user for self audience", async () => {
    const harness = await createRelationalExtractorHarness();
    const user = await harness.writer.append({
      kind: "user_msg",
      content: "My dog's name is Otto.",
      audience: "self",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [],
          [
            {
              subject_entity_id: "user",
              slot_key: "dog.name",
              asserted_value: "Otto",
              source_stream_entry_ids: [user.id],
            },
          ],
        ),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      entityRepository: harness.entityRepository,
      relationalSlotRepository: harness.relationalSlotRepository,
      clock: harness.clock,
    });

    await extractor.extractFromStream();

    const defaultUser = harness.entityRepository.resolve("user");
    const selfAudience = harness.entityRepository.resolve("self");

    expect(
      harness.relationalSlotRepository.findBySubjectAndKey(defaultUser, "dog.name"),
    ).toMatchObject({
      subject_entity_id: defaultUser,
      slot_key: "dog.name",
      value: "Otto",
      evidence_stream_entry_ids: [user.id],
    });
    expect(
      harness.relationalSlotRepository.findBySubjectAndKey(selfAudience, "dog.name"),
    ).toBeNull();
  });

  it("converges relational slots for default user and bare user subjects under one audience", async () => {
    const harness = await createRelationalExtractorHarness();
    const tom = harness.entityRepository.resolve("Tom");
    const dog = await harness.writer.append({
      kind: "user_msg",
      content: "My dog's name is Otto.",
      audience: "Tom",
    });
    harness.clock.advance(10);
    const partner = await harness.writer.append({
      kind: "user_msg",
      content: "My partner's name is Elena.",
      audience: "Tom",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [
            {
              title: "Tom named his dog",
              narrative: "Tom said his dog's name is Otto.",
              source_stream_ids: [dog.id],
              participants: ["Tom"],
              tags: ["dog"],
              confidence: 0.9,
              significance: 0.7,
            },
            {
              title: "Tom named his partner",
              narrative: "Tom said his partner's name is Elena.",
              source_stream_ids: [partner.id],
              participants: ["Tom"],
              tags: ["relationship"],
              confidence: 0.9,
              significance: 0.7,
            },
          ],
          [
            {
              subject_entity_id: "user",
              slot_key: "dog.name",
              asserted_value: "Otto",
              source_stream_entry_ids: [dog.id],
            },
            {
              subject_entity_id: tom,
              slot_key: "partner.name",
              asserted_value: "Elena",
              source_stream_entry_ids: [partner.id],
            },
          ],
        ),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      entityRepository: harness.entityRepository,
      relationalSlotRepository: harness.relationalSlotRepository,
      defaultUser: "Tom",
      clock: harness.clock,
    });

    await extractor.extractFromStream();

    expect(harness.relationalSlotRepository.list().map((slot) => slot.subject_entity_id)).toEqual([
      tom,
      tom,
    ]);
    expect(harness.entityRepository.findByName("user")).toBeNull();
  });

  it("sanitizes pending actions when relational slot extraction quarantines a value", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new ManualClock(1_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        commitmentMigrations,
        relationalSlotMigrations,
      ),
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
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const relationalSlotRepository = new RelationalSlotRepository({
      db,
      clock,
    });
    const workingMemoryStore = new WorkingMemoryStore({
      dataDir: tempDir,
      clock,
    });
    const tom = entityRepository.resolve("Tom");
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    workingMemoryStore.save({
      ...createWorkingMemory(DEFAULT_SESSION_ID, clock.now()),
      pending_actions: [
        {
          description: "Track whether Tom raises the planning comment with Sarah directly",
          next_action: "Ask Sarah if Tom brings up the planning comment",
        },
      ],
      updated_at: clock.now(),
    });

    const sarah = await writer.append({
      kind: "user_msg",
      content: "My partner's name is Sarah.",
    });
    clock.advance(10);
    const maya = await writer.append({
      kind: "user_msg",
      content: "Actually, my partner's name is Maya.",
    });
    clock.advance(10);
    const clara = await writer.append({
      kind: "user_msg",
      content: "No, my partner's name is Clara.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [],
          [
            {
              subject_entity_id: tom,
              slot_key: "partner.name",
              asserted_value: "Sarah",
              source_stream_entry_ids: [sarah.id],
            },
            {
              subject_entity_id: tom,
              slot_key: "partner.name",
              asserted_value: "Maya",
              source_stream_entry_ids: [maya.id],
            },
            {
              subject_entity_id: tom,
              slot_key: "partner.name",
              asserted_value: "Clara",
              source_stream_entry_ids: [clara.id],
            },
          ],
        ),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      entityRepository,
      relationalSlotRepository,
      workingMemoryStore,
      defaultUser: "Tom",
      clock,
    });

    await extractor.extractFromStream();

    const slot = relationalSlotRepository.findBySubjectAndKey(tom, "partner.name");
    const workingMemory = workingMemoryStore.load(DEFAULT_SESSION_ID);

    expect(slot?.state).toBe("quarantined");
    expect(workingMemory.pending_actions).toEqual([
      {
        description: "Track whether Tom raises the planning comment with your partner directly",
        next_action: "Ask your partner if Tom brings up the planning comment",
      },
    ]);
  });

  it("passes perception-only entities and mode through LLM-emitted episode fields", async () => {
    const llm = new FakeLLMClient();
    const clock = new ManualClock(1_000);
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: "default" as never,
      clock,
    });

    cleanup.push(harness.cleanup);
    cleanup.push(async () => {
      writer.close();
    });

    const user = await writer.append({
      kind: "user_msg",
      content: "Can we make sense of the recent planning thread?",
    });
    clock.advance(10);
    await writer.append({
      kind: "perception",
      content: {
        mode: "reflective",
        entities: ["LatentRook"],
        temporalCue: {
          label: "recent planning thread",
        },
        affectiveSignal: {
          valence: 0.15,
          arousal: 0.3,
          dominant_emotion: "curiosity",
        },
      },
    });
    clock.advance(10);
    const agent = await writer.append({
      kind: "agent_msg",
      content: "I mapped the thread into the current decision points.",
    });

    llm.pushResponse((options: LLMCompleteOptions) => {
      const prompt = String(options.messages[0]?.content ?? "");

      expect(prompt).toContain("<perception_context>");
      expect(prompt).toContain("LatentRook");
      expect(prompt).toContain('"mode":"reflective"');

      return createEpisodeToolResponse([
        {
          title: "Planning thread reflection",
          narrative: "The turn was framed as a reflective planning-thread review.",
          source_stream_ids: [user.id, agent.id],
          participants: ["LatentRook"],
          tags: ["LatentRook", "reflective"],
          confidence: 0.8,
          significance: 0.7,
        },
      ]);
    });

    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.episodicRepository,
      embeddingClient: harness.embeddingClient,
      llmClient: llm,
      model: "claude-haiku",
      entityRepository: harness.entityRepository,
      clock,
    });

    await extractor.extractFromStream();
    const [episode] = await harness.episodicRepository.listAll();

    expect(episode?.source_stream_ids).toEqual([user.id, agent.id]);
    expect(episode?.participants).toContain("LatentRook");
    expect(episode?.tags).toEqual(expect.arrayContaining(["LatentRook", "reflective"]));
  });

  it("omits perception context when the chunk has no perception entries", async () => {
    const llm = new FakeLLMClient();
    const clock = new ManualClock(1_000);
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: "default" as never,
      clock,
    });

    cleanup.push(harness.cleanup);
    cleanup.push(async () => {
      writer.close();
    });

    const user = await writer.append({
      kind: "user_msg",
      content: "Let's summarize the release checklist.",
    });
    clock.advance(10);
    const agent = await writer.append({
      kind: "agent_msg",
      content: "I grouped the checklist by risk and owner.",
    });

    llm.pushResponse(
      createEpisodeToolResponse([
        {
          title: "Release checklist summary",
          narrative: "The release checklist was summarized by risk and owner.",
          source_stream_ids: [user.id, agent.id],
          participants: ["team"],
          tags: ["release"],
          confidence: 0.8,
          significance: 0.7,
        },
      ]),
    );

    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.episodicRepository,
      embeddingClient: harness.embeddingClient,
      llmClient: llm,
      model: "claude-haiku",
      entityRepository: harness.entityRepository,
      clock,
    });

    await extractor.extractFromStream();
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(prompt).not.toContain("<perception_context>");
    expect(prompt).not.toContain("</perception_context>");
  });

  it("omits perception context when no perception entries match the chunk audience", async () => {
    const llm = new FakeLLMClient();
    const clock = new ManualClock(1_000);
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: "default" as never,
      clock,
    });

    cleanup.push(harness.cleanup);
    cleanup.push(async () => {
      writer.close();
    });

    const user = await writer.append({
      kind: "user_msg",
      content: "Can you help Sam with the private deployment note?",
      audience: "Sam",
    });
    clock.advance(10);
    await writer.append({
      kind: "perception",
      content: {
        mode: "relational",
        entities: ["AlexOnlySignal"],
        temporalCue: null,
        affectiveSignal: {
          valence: -0.2,
          arousal: 0.35,
          dominant_emotion: "fear",
        },
      },
      audience: "Alex",
    });
    clock.advance(10);
    const agent = await writer.append({
      kind: "agent_msg",
      content: "I drafted a scoped response for Sam.",
      audience: "Sam",
    });

    llm.pushResponse(
      createEpisodeToolResponse([
        {
          title: "Scoped deployment note",
          narrative: "The private deployment note for Sam was handled in a scoped turn.",
          source_stream_ids: [user.id, agent.id],
          participants: ["Sam"],
          tags: ["deployment"],
          confidence: 0.8,
          significance: 0.7,
        },
      ]),
    );

    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.episodicRepository,
      embeddingClient: harness.embeddingClient,
      llmClient: llm,
      model: "claude-haiku",
      entityRepository: harness.entityRepository,
      clock,
    });

    await extractor.extractFromStream();
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(prompt).not.toContain("<perception_context>");
    expect(prompt).not.toContain("AlexOnlySignal");
  });

  it("stores LLM-emitted emotional arcs without agent affect contamination", async () => {
    const llm = new FakeLLMClient();
    const clock = new ManualClock(1_000);
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: "default" as never,
      clock,
    });

    cleanup.push(harness.cleanup);
    cleanup.push(async () => {
      writer.close();
    });

    const user = await writer.append({
      kind: "user_msg",
      content: "I'm frustrated with this.",
    });
    clock.advance(10);
    const agent = await writer.append({
      kind: "agent_msg",
      content: "Great, happy, helpful, supportive, kind, and glad to help.",
    });

    llm.pushResponse(
      createEpisodeToolResponse([
        {
          title: "Frustrated implementation turn",
          narrative: "The user was frustrated and the agent offered help.",
          source_stream_ids: [user.id, agent.id],
          participants: ["user"],
          tags: ["implementation"],
          emotional_arc: {
            start: {
              valence: -0.7,
              arousal: 0.45,
            },
            peak: {
              valence: -0.7,
              arousal: 0.45,
            },
            end: {
              valence: -0.55,
              arousal: 0.35,
            },
            dominant_emotion: "anger",
          },
          confidence: 0.8,
          significance: 0.7,
        },
      ]),
    );

    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.episodicRepository,
      embeddingClient: harness.embeddingClient,
      llmClient: llm,
      model: "claude-haiku",
      entityRepository: harness.entityRepository,
      clock,
    });

    await extractor.extractFromStream();
    const [episode] = await harness.episodicRepository.listAll();

    expect(episode?.emotional_arc).not.toBeNull();
    expect(episode?.emotional_arc).toMatchObject({
      start: {
        valence: -0.7,
        arousal: 0.45,
      },
      peak: {
        valence: -0.7,
        arousal: 0.45,
      },
      end: {
        valence: -0.55,
        arousal: 0.35,
      },
      dominant_emotion: "anger",
    });
  });

  it("falls back to perception affective signals when LLM omits emotional arc", async () => {
    const llm = new FakeLLMClient();
    const clock = new ManualClock(1_000);
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: "default" as never,
      clock,
    });

    cleanup.push(harness.cleanup);
    cleanup.push(async () => {
      writer.close();
    });

    const user = await writer.append({
      kind: "user_msg",
      content: "Honestly, I am fine.",
    });
    clock.advance(10);
    await writer.append({
      kind: "perception",
      content: {
        mode: "relational",
        entities: [],
        temporalCue: null,
        affectiveSignal: {
          valence: -0.65,
          arousal: 0.55,
          dominant_emotion: "anger",
        },
      },
    });
    clock.advance(10);
    const agent = await writer.append({
      kind: "agent_msg",
      content: "Wonderful, happy, helpful, supportive, kind, and glad to help.",
    });

    llm.pushResponse(
      createEpisodeToolResponse([
        {
          title: "Guarded implementation turn",
          narrative: "The user signaled guarded frustration while the agent responded warmly.",
          source_stream_ids: [user.id, agent.id],
          participants: ["user"],
          tags: ["implementation"],
          confidence: 0.8,
          significance: 0.7,
        },
      ]),
    );

    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.episodicRepository,
      embeddingClient: harness.embeddingClient,
      llmClient: llm,
      model: "claude-haiku",
      entityRepository: harness.entityRepository,
      clock,
    });

    await extractor.extractFromStream();
    const [episode] = await harness.episodicRepository.listAll();

    expect(episode?.source_stream_ids).toEqual([user.id, agent.id]);
    expect(episode?.emotional_arc).toEqual({
      start: {
        valence: -0.65,
        arousal: 0.55,
      },
      peak: {
        valence: -0.65,
        arousal: 0.55,
      },
      end: {
        valence: -0.65,
        arousal: 0.55,
      },
      dominant_emotion: "anger",
    });
  });

  it("treats replayed chunks as idempotent no-ops keyed by source stream ids", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const first = await writer.append({
      kind: "user_msg",
      content: "We reviewed the retrieval boundary.",
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEpisodeToolResponse([
            {
              title: "Retrieval boundary review",
              narrative: "We reviewed the retrieval boundary and hard audience scoping.",
              source_stream_ids: [first.id],
              participants: ["team"],
              tags: ["retrieval"],
              confidence: 0.8,
              significance: 0.7,
            },
          ]),
          createEpisodeToolResponse([
            {
              title: "Retrieval boundary review",
              narrative: "We reviewed the retrieval boundary and hard audience scoping.",
              source_stream_ids: [first.id],
              participants: ["team"],
              tags: ["retrieval"],
              confidence: 0.8,
              significance: 0.7,
            },
          ]),
        ],
      }),
      model: "claude-haiku",
      entityRepository,
      clock,
    });

    const firstRun = await extractor.extractFromStream();
    const secondRun = await extractor.extractFromStream();

    expect(firstRun).toEqual({
      inserted: 1,
      updated: 0,
      skipped: 0,
    });
    expect(secondRun).toEqual({
      inserted: 0,
      updated: 0,
      skipped: 1,
    });
    expect((await repo.listAll()).map((episode) => episode.source_stream_ids)).toEqual([
      [first.id],
    ]);
  });

  it("does not extract a user-only turn marked as agent_suppressed", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const user = await writer.append({
      kind: "user_msg",
      content: "No.",
    });
    clock.advance(1);
    await writer.append({
      kind: "agent_suppressed",
      content: {
        reason: "generation_gate",
        user_entry_id: user.id,
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Should not be read",
            narrative: "This response should not be consumed.",
            source_stream_ids: [user.id],
            participants: ["Borg"],
            tags: ["suppression"],
            confidence: 0.8,
            significance: 0.7,
          },
        ]),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      entityRepository,
      clock,
    });

    await expect(extractor.extractFromStream()).resolves.toEqual({
      inserted: 0,
      updated: 0,
      skipped: 0,
    });
    expect(llm.requests).toHaveLength(0);
  });

  it("rejects hallucinated source stream ids", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
    });
    const entityRepository = new EntityRepository({
      db,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
    });
    const entry = await writer.append({
      kind: "user_msg",
      content: "hello",
    });
    void entry;

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEpisodeToolResponse([
            {
              title: "Planning sync",
              narrative: "A grounded narrative.",
              source_stream_ids: ["strm_missingmissing"],
              participants: [],
              tags: [],
              confidence: 0.8,
              significance: 0.8,
            },
          ]),
        ],
      }),
      model: "claude-haiku",
      entityRepository,
    });

    await expect(extractor.extractFromStream()).rejects.toBeInstanceOf(LLMError);
  });

  it("raises a typed error naming the tool when the llm returns bare text", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
    });
    const entityRepository = new EntityRepository({
      db,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
    });

    await writer.append({
      kind: "user_msg",
      content: "hello",
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: '{"episodes":[]}',
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      model: "claude-haiku",
      entityRepository,
    });

    await expect(extractor.extractFromStream()).rejects.toMatchObject({
      code: "EXTRACTOR_OUTPUT_INVALID",
      message: expect.stringContaining(EPISODE_TOOL_NAME),
    });
  });

  it("propagates embedding failures so ingestion can retry the candidate", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const first = await writer.append({
      kind: "user_msg",
      content: "candidate one",
    });
    const second = await writer.append({
      kind: "agent_msg",
      content: "candidate two",
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new FailingOnceEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEpisodeToolResponse([
            {
              title: "Skip me",
              narrative: "This candidate will fail embedding.",
              source_stream_ids: [first.id],
              participants: [],
              tags: [],
              confidence: 0.5,
              significance: 0.5,
            },
            {
              title: "Keep me",
              narrative: "This candidate should still be inserted.",
              source_stream_ids: [second.id],
              participants: [],
              tags: ["kept"],
              confidence: 0.9,
              significance: 0.9,
            },
          ]),
        ],
      }),
      model: "claude-haiku",
      entityRepository,
      clock,
    });

    await expect(extractor.extractFromStream()).rejects.toBeInstanceOf(EmbeddingError);
    const listed = await repo.list();

    expect(listed.items).toHaveLength(0);
  });
});
