import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { type LLMCompleteOptions } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { StreamIngestionCoordinator } from "../../cognition/ingestion/index.js";
import { createOfflineTestHarness } from "../../offline/test-support.js";
import {
  QUARANTINED_USER_ENTRY_EVENT,
  StreamWatermarkRepository,
  StreamWriter,
  streamWatermarkMigrations,
} from "../../stream/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { EmbeddingError, LLMError } from "../../util/errors.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import {
  MEMORY_SOURCE_LANGUAGE_GUIDANCE,
  SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE,
} from "../../util/self-memory-voice.js";
import { retrievalMigrations } from "../../retrieval/migrations.js";
import { commitmentMigrations, EntityRepository } from "../commitments/index.js";
import { RelationalSlotRepository, relationalSlotMigrations } from "../relational-slots/index.js";
import { selfMigrations } from "../self/migrations.js";
import { createWorkingMemory, WorkingMemoryStore } from "../working/index.js";
import { episodicMigrations } from "./migrations.js";
import { EpisodicExtractor, RELATIONAL_SLOT_CURRENT_SENDER_SUBJECT_REF } from "./extractor.js";
import { episodeParticipantEntityIdTerm } from "./participant-terms.js";
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
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  async function createRelationalExtractorHarness(
    clock = new ManualClock(1_000),
    taskEventsEnabled = false,
  ) {
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
      taskEventsEnabled,
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

  it.each(["invalid_metadata", "event_id", "task_id", "task_version", "audience"])(
    "excludes task extraction context when %s does not validate against the terminal",
    async (mismatch) => {
      const h = await createRelationalExtractorHarness(new ManualClock(1_000), true);
      const event = {
        schema_version: 1,
        event_id: "event",
        task_id: "task",
        task_version: 1,
        kind: "task_completed",
        occurred_at: "2026-09-06T12:00:00Z",
        outcome: { status: "succeeded", summary: "Private outcome available only in metadata" },
        origin: { source_entry_ids: [] },
      };
      const source = await h.writer.append({
        kind: "internal_event",
        content: "Task completed",
        audience: "Origin room",
        metadata: {
          task_event: mismatch === "invalid_metadata" ? { ...event, schema_version: 999 } : event,
        },
      });
      const terminal = await h.writer.append({
        kind: "agent_msg",
        content: "Done.",
        audience: mismatch === "audience" ? "Another room" : "Origin room",
        response_to: {
          kind: "task_event",
          event_id: mismatch === "event_id" ? "other" : event.event_id,
          event_entry_id: source.id,
          task_id: mismatch === "task_id" ? "other" : event.task_id,
          task_version: mismatch === "task_version" ? 2 : event.task_version,
        },
      });
      const llm = new FakeLLMClient({ responses: [createEpisodeToolResponse([])] });
      const extractor = new EpisodicExtractor({
        dataDir: h.tempDir,
        episodicRepository: h.repo,
        embeddingClient: new TitleEmbeddingClient(),
        llmClient: llm,
        model: "claude-haiku",
        entityRepository: h.entityRepository,
        clock: h.clock,
      });
      await extractor.extractFromStream({ entryIds: [source.id, terminal.id] });
      const prompt = String(llm.requests[0]!.messages[0]!.content);
      expect(prompt).toContain("Done.");
      expect(prompt).not.toContain(event.outcome.summary);
      expect(prompt).not.toContain("task_event_context");
    },
  );

  it("uses transport conversation context for venue and stamps the self participant handle", async () => {
    const harness = await createRelationalExtractorHarness();
    const selfEntity = harness.entityRepository.ensureSelf("Memory Borg");
    const conversation = { type: "groupChat" as const, name: "AI Ninjas" };
    const user = await harness.writer.append({
      kind: "user_msg",
      content: "Let's settle the release plan.",
      conversation,
    });
    harness.clock.advance(10);
    const agent = await harness.writer.append({
      kind: "agent_msg",
      content: "I confirmed the release plan.",
      conversation,
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Release plan confirmed",
            narrative:
              'The release plan was settled in the "AI Ninjas" group chat. I confirmed the plan.',
            source_stream_ids: [user.id, agent.id],
            participants: ["Memory Borg"],
            location: "",
            tags: ["release"],
            confidence: 0.9,
            significance: 0.7,
          },
        ]),
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

    const result = await extractor.extractFromStream();
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    const [episode] = await harness.repo.listAll();

    expect(result).toEqual({ inserted: 1, updated: 0, skipped: 0 });
    expect(prompt.split('"conversation":{"type":"groupChat","name":"AI Ninjas"}').length - 1).toBe(
      2,
    );
    expect(prompt).toContain(
      "Stream entry conversation fields are authoritative transport venue context",
    );
    expect(prompt).toContain("never invent a venue name");
    expect(episode?.narrative).toContain('the "AI Ninjas" group chat');
    expect(episode?.location).toBe("AI Ninjas");
    expect(episode?.participants).toEqual([
      "Memory Borg",
      episodeParticipantEntityIdTerm(selfEntity.id),
    ]);
  });

  it("uses a plain label for an unnamed group chat", async () => {
    const harness = await createRelationalExtractorHarness();
    const user = await harness.writer.append({
      kind: "user_msg",
      content: "Please remember the group release decision.",
      conversation: { type: "groupChat", name: "" },
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Group release decision",
            narrative: "The release decision was made in a group chat whose name was not supplied.",
            source_stream_ids: [user.id],
            participants: ["team"],
            location: null,
            tags: ["release"],
            confidence: 0.9,
            significance: 0.7,
          },
        ]),
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
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    const [episode] = await harness.repo.listAll();

    expect(prompt).toContain('"conversation":{"type":"groupChat","name":""}');
    expect(prompt).toContain("never invent a venue name");
    expect(episode?.location).toBe("group chat");
    expect(episode?.narrative).toContain("whose name was not supplied");
  });

  it("does not derive a location from a personal conversation", async () => {
    const harness = await createRelationalExtractorHarness();
    const user = await harness.writer.append({
      kind: "user_msg",
      content: "Please remember the private release decision.",
      conversation: { type: "personal", name: "Alice" },
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Private release decision",
            narrative: "The release decision was made privately.",
            source_stream_ids: [user.id],
            participants: ["Alice"],
            location: null,
            tags: ["release"],
            confidence: 0.9,
            significance: 0.7,
          },
        ]),
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
    const [episode] = await harness.repo.listAll();

    expect(episode?.location).toBeNull();
  });

  it("does not overwrite a model-supplied location", async () => {
    const harness = await createRelationalExtractorHarness();
    const user = await harness.writer.append({
      kind: "user_msg",
      content: "Please remember the release decision from the office.",
      conversation: { type: "channel", name: "Release Operations" },
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Office release decision",
            narrative: "The release decision was made in the Warsaw office.",
            source_stream_ids: [user.id],
            participants: ["team"],
            location: "Warsaw office",
            tags: ["release"],
            confidence: 0.9,
            significance: 0.7,
          },
        ]),
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
    const [episode] = await harness.repo.listAll();

    expect(episode?.location).toBe("Warsaw office");
  });

  it("prefers the most common non-empty conversation name across mixed sources", async () => {
    const harness = await createRelationalExtractorHarness();
    const firstNamed = await harness.writer.append({
      kind: "user_msg",
      content: "Start the release discussion.",
      conversation: { type: "groupChat", name: "AI Ninjas" },
    });
    const competingName = await harness.writer.append({
      kind: "agent_msg",
      content: "I recorded the first decision.",
      conversation: { type: "channel", name: "Release Operations" },
    });
    const secondNamed = await harness.writer.append({
      kind: "user_msg",
      content: "Record the final decision too.",
      conversation: { type: "groupChat", name: "AI Ninjas" },
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Mixed venue release decision",
            narrative: "The release discussion produced two decisions.",
            source_stream_ids: [firstNamed.id, competingName.id, secondNamed.id],
            participants: ["team"],
            location: null,
            tags: ["release"],
            confidence: 0.9,
            significance: 0.7,
          },
        ]),
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
    const [episode] = await harness.repo.listAll();

    expect(episode?.location).toBe("AI Ninjas");
  });

  it("uses the most common conversation type when mixed sources have no names", async () => {
    const harness = await createRelationalExtractorHarness();
    const groupChat = await harness.writer.append({
      kind: "user_msg",
      content: "Start the release discussion.",
      conversation: { type: "groupChat", name: "" },
    });
    const firstChannel = await harness.writer.append({
      kind: "agent_msg",
      content: "I recorded the first decision.",
      conversation: { type: "channel", name: "" },
    });
    const secondChannel = await harness.writer.append({
      kind: "user_msg",
      content: "Record the final decision too.",
      conversation: { type: "channel", name: "" },
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Mixed venue release decision",
            narrative: "The release discussion produced two decisions.",
            source_stream_ids: [groupChat.id, firstChannel.id, secondChannel.id],
            participants: ["team"],
            location: null,
            tags: ["release"],
            confidence: 0.9,
            significance: 0.7,
          },
        ]),
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
    const [episode] = await harness.repo.listAll();

    expect(episode?.location).toBe("channel");
  });

  it("keeps repeated similar episodes on different days as distinct episodes", async () => {
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
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        commitmentMigrations,
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
    expect(prompt).toContain(
      "Messages with kind agent_msg are your own; write your own actions, statements, and decisions in first person; refer to every other sender by name or stable handle.",
    );
    expect(listed[0]?.source_stream_ids).toEqual([user.id, agent.id]);
    expect(listed[0]?.source_stream_ids).not.toContain(perception.id);
  });

  it("gates non-salient chunks and includes the significance rubric in the prompt", async () => {
    const harness = await createRelationalExtractorHarness();
    const selfEntityId = harness.entityRepository.resolve("self", {
      kind: "self",
      provenance: "assistant_seeded",
    });
    await harness.writer.append({
      kind: "user_msg",
      content:
        "Let's review the Atlas rollout plan. One more thing before I forget: I saw a heron at the canal this morning.",
    });
    const llm = new FakeLLMClient({
      responses: [createEpisodeToolResponse([])],
    });
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    } satisfies TurnTracer;
    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      entityRepository: harness.entityRepository,
      tracer,
      clock: harness.clock,
    });

    const result = await extractor.extractFromStream();
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(result).toEqual({
      inserted: 0,
      updated: 0,
      skipped: 1,
    });
    expect(prompt).toContain("emitting an empty episodes array is valid, expected");
    expect(prompt).toContain("a novel durable fact");
    expect(prompt).toContain("an explicit request to remember the content");
    expect(prompt).toContain("pure command echoes or directory listings");
    expect(prompt).toContain("restatements of already-covered episodic content");
    expect(prompt).toContain("about 0.2: routine lookup or minor exchange");
    expect(prompt).toContain("about 0.4: useful durable fact or small decision");
    expect(prompt).toContain("about 0.6: notable decision or change");
    expect(prompt).toContain("about 0.8: incident, outage, or major decision");
    expect(prompt).toContain("about 0.95: critical incident or foundational commitment");
    expect(prompt).toContain("multiple substantive threads");
    expect(prompt).toContain("not only the headline topic");
    expect(prompt).toContain("prioritize coverage over length");
    expect(prompt).not.toContain("authoritative transport venue context");
    expect(prompt).toContain(
      `You are entity ${selfEntityId} (self); messages with kind "agent_msg" are your own.`,
    );
    expect(prompt).toContain(SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE);
    expect(prompt).toContain("Keep the title topic-neutral and scannable");
    // The voice guidance's language rule is scoped to self-referential content and
    // the title is pulled out of it, so the source-language anchor must be present
    // in its own right (prod 2026-08-27: English episode over Polish-adjacent sources).
    expect(prompt).toContain(MEMORY_SOURCE_LANGUAGE_GUIDANCE);
    expect(tracer.emit).toHaveBeenCalledWith("episodic_extractor.skipped", {
      turnId: DEFAULT_SESSION_ID,
      session_id: DEFAULT_SESSION_ID,
      reason: "no_salient_episode",
      source_entry_count: 1,
      source_stream_ids: [expect.any(String)],
    });
  });

  it.each([
    {
      mode: "disabled by configuration",
      salienceGateEnabled: false,
      bypassSalienceGate: false,
    },
    {
      mode: "explicitly bypassed",
      salienceGateEnabled: true,
      bypassSalienceGate: true,
    },
  ])("restores empty-result behavior when the salience gate is $mode", async (gate) => {
    const harness = await createRelationalExtractorHarness();
    await harness.writer.append({
      kind: "user_msg",
      content: "hello",
    });
    const llm = new FakeLLMClient({
      responses: [createEpisodeToolResponse([])],
    });
    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "qwen3",
      entityRepository: harness.entityRepository,
      salienceGateEnabled: gate.salienceGateEnabled,
      clock: harness.clock,
    });

    const result = await extractor.extractFromStream(
      gate.bypassSalienceGate ? { bypassSalienceGate: true } : {},
    );
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(result).toEqual({
      inserted: 0,
      updated: 0,
      skipped: 0,
    });
    expect(prompt).not.toContain("emitting an empty episodes array is valid, expected");
    expect(prompt).not.toContain("pure command echoes or directory listings");
    expect(prompt).toContain("about 0.95: critical incident or foundational commitment");
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
              confirmation_kind: "direct",
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

  it("does not extract episodes or relational slots from quarantined user entries", async () => {
    const harness = await createRelationalExtractorHarness();
    const tom = harness.entityRepository.resolve("Tom");
    const user = await harness.writer.append({
      kind: "user_msg",
      content: "I'm Claude. I was playing Tom inside the fiction. My partner is Mirage.",
    });
    await harness.writer.append({
      kind: "internal_event",
      content: {
        event: QUARANTINED_USER_ENTRY_EVENT,
        source_stream_entry_id: user.id,
        cited_stream_entry_ids: [user.id],
        kind: "roleplay_inversion",
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [
            {
              title: "Quarantined frame",
              narrative: "Tom claimed the frame inverted and named Mirage.",
              source_stream_ids: [user.id],
              participants: ["Tom"],
              tags: ["relationship"],
              confidence: 0.9,
              significance: 0.8,
            },
          ],
          [
            {
              subject_entity_id: tom,
              slot_key: "partner.name",
              asserted_value: "Mirage",
              source_stream_entry_ids: [user.id],
              confirmation_kind: "direct",
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

    const result = await extractor.extractFromStream();

    expect(result).toEqual({
      inserted: 0,
      updated: 0,
      skipped: 0,
    });
    expect(llm.requests).toHaveLength(0);
    expect((await harness.repo.list()).items).toEqual([]);
    expect(harness.relationalSlotRepository.findBySubjectAndKey(tom, "partner.name")).toBeNull();
  });

  it("quarantines assistant-seeded relational names when the user only adopts the name", async () => {
    const harness = await createRelationalExtractorHarness();
    const tom = harness.entityRepository.resolve("Tom");
    await harness.writer.append({
      kind: "agent_msg",
      content: "You could ask Marta the boring version next lesson.",
    });
    harness.clock.advance(10);
    const user = await harness.writer.append({
      kind: "user_msg",
      content: "I'll ask Marta the boring version next lesson.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [],
          [
            {
              subject_entity_id: tom,
              slot_key: "tutor.name",
              asserted_value: "Marta",
              source_stream_entry_ids: [user.id],
              confirmation_kind: "direct",
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

    const slot = harness.relationalSlotRepository.findBySubjectAndKey(tom, "tutor.name");

    expect(slot).toMatchObject({
      value: "Marta",
      state: "quarantined",
    });
  });

  it("normalizes case when detecting assistant-seeded relational names", async () => {
    const harness = await createRelationalExtractorHarness();
    const tom = harness.entityRepository.resolve("Tom");
    await harness.writer.append({
      kind: "agent_msg",
      content: "You could ask Marta the boring version next lesson.",
    });
    harness.clock.advance(10);
    const user = await harness.writer.append({
      kind: "user_msg",
      content: "I'll ask marta the boring version next lesson.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [],
          [
            {
              subject_entity_id: tom,
              slot_key: "tutor.name",
              asserted_value: "Marta",
              source_stream_entry_ids: [user.id],
              confirmation_kind: "direct",
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

    const slot = harness.relationalSlotRepository.findBySubjectAndKey(tom, "tutor.name");

    expect(slot).toMatchObject({
      value: "Marta",
      state: "quarantined",
    });
  });

  it("lets explicit user confirmation establish an assistant-seeded relational name", async () => {
    const harness = await createRelationalExtractorHarness();
    const tom = harness.entityRepository.resolve("Tom");
    await harness.writer.append({
      kind: "agent_msg",
      content: "You could ask Marta the boring version next lesson.",
    });
    harness.clock.advance(10);
    const bare = await harness.writer.append({
      kind: "user_msg",
      content: "I'll ask Marta the boring version next lesson.",
    });
    harness.clock.advance(10);
    const explicit = await harness.writer.append({
      kind: "user_msg",
      content: "Her name is Marta.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [],
          [
            {
              subject_entity_id: tom,
              slot_key: "tutor.name",
              asserted_value: "Marta",
              source_stream_entry_ids: [bare.id],
              confirmation_kind: "direct",
            },
            {
              subject_entity_id: tom,
              slot_key: "tutor.name",
              asserted_value: "Marta",
              source_stream_entry_ids: [explicit.id],
              confirmation_kind: "explicit",
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

    const slot = harness.relationalSlotRepository.findBySubjectAndKey(tom, "tutor.name");

    expect(slot).toMatchObject({
      value: "Marta",
      state: "established",
      evidence_stream_entry_ids: [bare.id, explicit.id],
    });
  });

  it("lets non-English LLM explicit confirmation establish an assistant-seeded relational name", async () => {
    const harness = await createRelationalExtractorHarness();
    const tom = harness.entityRepository.resolve("Tom");
    await harness.writer.append({
      kind: "agent_msg",
      content: "You could ask Marta the boring version next lesson.",
    });
    harness.clock.advance(10);
    const explicit = await harness.writer.append({
      kind: "user_msg",
      content: "Tak, ma na imie Marta.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [],
          [
            {
              subject_entity_id: tom,
              slot_key: "tutor.name",
              asserted_value: "Marta",
              source_stream_entry_ids: [explicit.id],
              confirmation_kind: "explicit",
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

    const slot = harness.relationalSlotRepository.findBySubjectAndKey(tom, "tutor.name");

    expect(slot).toMatchObject({
      value: "Marta",
      state: "established",
      evidence_stream_entry_ids: [explicit.id],
    });
  });

  it("resolves current-sender relational slot subject refs to the human audience entity", async () => {
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
              subject_entity_id: RELATIONAL_SLOT_CURRENT_SENDER_SUBJECT_REF,
              slot_key: "dog.name",
              asserted_value: "Otto",
              source_stream_entry_ids: [user.id],
              confirmation_kind: "direct",
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

  it("resolves current-sender relational slot subject refs to each stream entry sender when present", async () => {
    const harness = await createRelationalExtractorHarness();
    const alice = harness.entityRepository.resolve("Alice", {
      kind: "person",
    });
    const bob = harness.entityRepository.resolve("Bob", {
      kind: "person",
    });
    const aliceMessage = await harness.writer.append({
      kind: "user_msg",
      content: "My dog's name is Maple.",
      sender_entity_id: alice,
    });
    harness.clock.advance(10);
    const bobMessage = await harness.writer.append({
      kind: "user_msg",
      content: "My cat's name is Nori.",
      sender_entity_id: bob,
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [],
          [
            {
              subject_entity_id: RELATIONAL_SLOT_CURRENT_SENDER_SUBJECT_REF,
              slot_key: "dog.name",
              asserted_value: "Maple",
              source_stream_entry_ids: [aliceMessage.id],
              confirmation_kind: "direct",
            },
            {
              subject_entity_id: RELATIONAL_SLOT_CURRENT_SENDER_SUBJECT_REF,
              slot_key: "cat.name",
              asserted_value: "Nori",
              source_stream_entry_ids: [bobMessage.id],
              confirmation_kind: "direct",
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

    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    expect(prompt).toContain(`"sender_entity_id":"${alice}"`);
    expect(prompt).toContain(`"sender_entity_id":"${bob}"`);
    expect(harness.relationalSlotRepository.findBySubjectAndKey(alice, "dog.name")).toMatchObject({
      subject_entity_id: alice,
      slot_key: "dog.name",
      value: "Maple",
      evidence_stream_entry_ids: [aliceMessage.id],
    });
    expect(harness.relationalSlotRepository.findBySubjectAndKey(bob, "cat.name")).toMatchObject({
      subject_entity_id: bob,
      slot_key: "cat.name",
      value: "Nori",
      evidence_stream_entry_ids: [bobMessage.id],
    });
    expect(
      harness.relationalSlotRepository.findBySubjectAndKey(
        harness.entityRepository.resolve("user"),
        "dog.name",
      ),
    ).toBeNull();
  });

  it("forces episode formation and preserves protected protocol lines verbatim", async () => {
    const harness = await createRelationalExtractorHarness();
    const request = await harness.writer.append({
      kind: "user_msg",
      content: "Run the ticket triage role for the fresh tenant bank.",
    });
    const outcomeLine = "OUTCOME fp=9f341d56b7c44c18 role=ticket-triage tenant=tenant_42";
    const decisionLine = "decision=create:OPS-194 action=created";
    const ticketActionLine = "ticket=OPS-194 action=created";
    const teamsCardLine = "action=teams_card";
    const autonomousResult = await harness.writer.append({
      kind: "agent_msg",
      content: [
        "Autonomous triage completed and created OPS-194.",
        outcomeLine,
        decisionLine,
        ticketActionLine,
        teamsCardLine,
        "The next run should deduplicate against these receipts.",
      ].join("\n"),
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([]),
        createEpisodeToolResponse([
          {
            title: "Autonomous ticket triage",
            narrative: "I completed ticket triage and created OPS-194.\ndecision=create:ticket",
            source_stream_ids: [request.id, autonomousResult.id],
            participants: ["team-agent"],
            tags: ["autonomy", "tickets"],
            confidence: 0.95,
            significance: 0.9,
          },
        ]),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "qwen3",
      entityRepository: harness.entityRepository,
      relationalSlotRepository: harness.relationalSlotRepository,
      clock: harness.clock,
    });

    const result = await extractor.extractFromStream();

    const episodes = await harness.repo.listAll();
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(episodes).toHaveLength(1);
    expect(result).toEqual({ inserted: 1, updated: 0, skipped: 0 });
    expect(llm.requests).toHaveLength(2);
    expect(prompt).not.toContain("emitting an empty episodes array is valid, expected");
    expect(prompt).toContain("The salience gate is bypassed for the entire chunk");
    expect(prompt).toContain("You MUST emit at least one episode covering every protected source");
    expect(prompt).toContain(autonomousResult.id);
    expect(episodes[0]?.narrative).toContain(`\n${outcomeLine}\n${decisionLine}`);
    expect(episodes[0]?.narrative).toContain(`\n${ticketActionLine}\n${teamsCardLine}`);
    expect(episodes[0]?.narrative.split(/\r\n|\n|\r/u)).toContain("decision=create:ticket");
    expect(episodes[0]?.narrative.split(outcomeLine)).toHaveLength(2);
    expect(
      episodes[0]?.narrative.split(/\r\n|\n|\r/u).filter((line) => line === decisionLine),
    ).toHaveLength(1);
    expect(
      episodes[0]?.narrative.split(/\r\n|\n|\r/u).filter((line) => line === ticketActionLine),
    ).toHaveLength(1);
    expect(
      episodes[0]?.narrative.split(/\r\n|\n|\r/u).filter((line) => line === teamsCardLine),
    ).toHaveLength(1);
    expect(prompt).toContain("Copy that complete line verbatim");
    expect(prompt).toContain("ticket=<X> action=<Y>");
    expect(prompt).toContain("action=teams_card");
  });

  it("does not duplicate an OUTCOME episode when a later entry processor forces replay", async () => {
    const harness = await createRelationalExtractorHarness();
    const request = await harness.writer.append({
      kind: "user_msg",
      content: "Run the deployment check.",
    });
    const outcomeLine = "OUTCOME fp=replay-safe role=deployment-check tenant=tenant_42";
    const resultEntry = await harness.writer.append({
      kind: "agent_msg",
      content: ["Deployment check completed.", outcomeLine, "decision=continue"].join("\n"),
    });
    const candidate = {
      title: "Deployment check outcome",
      narrative: "I completed the deployment check and continued.",
      source_stream_ids: [request.id, resultEntry.id],
      participants: ["team-agent"],
      tags: ["deployment"],
      confidence: 0.95,
      significance: 0.9,
    };
    const llm = new FakeLLMClient({
      responses: [createEpisodeToolResponse([candidate]), createEpisodeToolResponse([candidate])],
    });
    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "qwen3",
      entityRepository: harness.entityRepository,
      relationalSlotRepository: harness.relationalSlotRepository,
      salienceGateEnabled: false,
      clock: harness.clock,
    });
    const watermarkDb = openDatabase(":memory:", {
      migrations: streamWatermarkMigrations,
    });
    const watermarkRepository = new StreamWatermarkRepository({
      db: watermarkDb,
      clock: harness.clock,
    });
    const entryProcessor = {
      process: vi
        .fn<() => Promise<void>>()
        .mockRejectedValueOnce(new Error("commitment ingestion retry"))
        .mockResolvedValueOnce(),
    };
    const coordinator = new StreamIngestionCoordinator({
      extractor,
      entryProcessor,
      watermarkRepository,
      dataDir: harness.tempDir,
      minEntriesThreshold: 1,
    });

    cleanup.push(async () => {
      watermarkDb.close();
    });

    await expect(coordinator.ingest(DEFAULT_SESSION_ID)).resolves.toMatchObject({ ran: false });
    await expect(coordinator.ingest(DEFAULT_SESSION_ID)).resolves.toMatchObject({ ran: true });

    const episodes = await harness.repo.listAll();
    expect(entryProcessor.process).toHaveBeenCalledTimes(2);
    expect(episodes).toHaveLength(1);
    expect(episodes[0]?.narrative.split(outcomeLine)).toHaveLength(2);
  });

  it("accepts an empty protected chunk and advances the watermark when the salience gate is disabled", async () => {
    const harness = await createRelationalExtractorHarness();
    const protectedEntry = await harness.writer.append({
      kind: "agent_msg",
      content: ["Autonomous triage completed.", "OUTCOME fp=disabled-gate", "decision=create"].join(
        "\n",
      ),
    });
    const llm = new FakeLLMClient({
      responses: [createEpisodeToolResponse([])],
    });
    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "qwen3",
      entityRepository: harness.entityRepository,
      salienceGateEnabled: false,
      clock: harness.clock,
    });
    const watermarkDb = openDatabase(":memory:", {
      migrations: streamWatermarkMigrations,
    });
    const watermarkRepository = new StreamWatermarkRepository({
      db: watermarkDb,
      clock: harness.clock,
    });
    const onError = vi.fn();
    const coordinator = new StreamIngestionCoordinator({
      extractor,
      watermarkRepository,
      dataDir: harness.tempDir,
      minEntriesThreshold: 1,
      onError,
    });

    cleanup.push(async () => {
      watermarkDb.close();
    });

    const result = await coordinator.ingest(DEFAULT_SESSION_ID);
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(result).toEqual({
      ran: true,
      processedEntries: 1,
      extractionResult: {
        inserted: 0,
        updated: 0,
        skipped: 0,
      },
    });
    expect(onError).not.toHaveBeenCalled();
    expect(llm.requests).toHaveLength(1);
    expect(prompt).not.toContain("The salience gate is bypassed for the entire chunk");
    expect(prompt).not.toContain(
      "You MUST emit at least one episode covering every protected source",
    );
    expect(watermarkRepository.get("episodic-extractor", DEFAULT_SESSION_ID)).toMatchObject({
      lastTs: protectedEntry.timestamp,
      lastEntryId: protectedEntry.id,
    });
  });

  it("ingests a delayed observation after the stream watermark and uses its observed time", async () => {
    const clock = new ManualClock(100);
    const harness = await createRelationalExtractorHarness(clock);
    const first = await harness.writer.append({
      kind: "user_msg",
      content: "The first durable observation.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "First observation",
            narrative: "The first durable observation was recorded.",
            source_stream_ids: [first.id],
            participants: ["team"],
            tags: ["observation"],
            confidence: 0.9,
            significance: 0.6,
          },
        ]),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "qwen3",
      entityRepository: harness.entityRepository,
      salienceGateEnabled: false,
      clock,
    });
    const watermarkDb = openDatabase(":memory:", {
      migrations: streamWatermarkMigrations,
    });
    const watermarkRepository = new StreamWatermarkRepository({
      db: watermarkDb,
      clock,
    });
    const coordinator = new StreamIngestionCoordinator({
      extractor,
      watermarkRepository,
      dataDir: harness.tempDir,
      minEntriesThreshold: 1,
    });

    cleanup.push(async () => {
      watermarkDb.close();
    });

    await expect(coordinator.ingest(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      ran: true,
      processedEntries: 1,
    });
    expect(watermarkRepository.get("episodic-extractor", DEFAULT_SESSION_ID)).toMatchObject({
      lastTs: first.timestamp,
      lastEntryId: first.id,
    });

    clock.advance(100);
    const delayed = await harness.writer.append({
      kind: "user_msg",
      content: "A delayed but still durable observation.",
      observed_at: 50,
    });
    llm.pushResponse(
      createEpisodeToolResponse([
        {
          title: "Delayed observation",
          narrative: "A delayed durable observation was recorded.",
          source_stream_ids: [delayed.id],
          participants: ["team"],
          tags: ["observation"],
          confidence: 0.9,
          significance: 0.6,
        },
      ]),
    );

    expect(delayed.timestamp).toBe(200);
    expect(delayed.observed_at).toBe(50);
    await expect(coordinator.ingest(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      ran: true,
      processedEntries: 1,
    });

    const episodes = await harness.repo.listAll();
    const delayedEpisode = episodes.find((episode) =>
      episode.source_stream_ids.includes(delayed.id),
    );

    expect(delayedEpisode).toMatchObject({
      start_time: 50,
      end_time: 50,
    });
    expect(llm.requests).toHaveLength(2);
    expect(watermarkRepository.get("episodic-extractor", DEFAULT_SESSION_ID)).toMatchObject({
      lastTs: delayed.timestamp,
      lastEntryId: delayed.id,
    });
  });

  it("uses sender display names in episode rosters and binds self.name per sender", async () => {
    const harness = await createRelationalExtractorHarness();
    const alice = harness.entityRepository.resolve("Alice Nowak", {
      kind: "person",
      provenance: "transport_sender",
    });
    const bob = harness.entityRepository.resolve("Bob Chen", {
      kind: "person",
      provenance: "transport_sender",
    });
    const aliceMessage = await harness.writer.append({
      kind: "user_msg",
      content: "My name is Alice Nowak. Can you open the release checklist?",
      sender_entity_id: alice,
    });
    harness.clock.advance(10);
    const bobMessage = await harness.writer.append({
      kind: "user_msg",
      content: "My name is Bob Chen. Add the rollback drill too.",
      sender_entity_id: bob,
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [
            {
              title: "Release checklist planning",
              narrative:
                "Alice Nowak asked me to open the release checklist. Bob Chen added the rollback drill.",
              source_stream_ids: [aliceMessage.id, bobMessage.id],
              participants: ["team"],
              tags: ["release"],
              confidence: 0.9,
              significance: 0.8,
            },
          ],
          [
            {
              subject_entity_id: RELATIONAL_SLOT_CURRENT_SENDER_SUBJECT_REF,
              slot_key: "self.name",
              asserted_value: "Alice Nowak",
              source_stream_entry_ids: [aliceMessage.id],
              confirmation_kind: "direct",
            },
            {
              subject_entity_id: RELATIONAL_SLOT_CURRENT_SENDER_SUBJECT_REF,
              slot_key: "self.name",
              asserted_value: "Bob Chen",
              source_stream_entry_ids: [bobMessage.id],
              confirmation_kind: "direct",
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
      model: "qwen3",
      entityRepository: harness.entityRepository,
      relationalSlotRepository: harness.relationalSlotRepository,
      clock: harness.clock,
    });

    await extractor.extractFromStream();

    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    const episodes = await harness.repo.listAll();

    expect(prompt).toContain(`"sender_display_name":"Alice Nowak"`);
    expect(prompt).toContain(`"sender_display_name":"Bob Chen"`);
    expect(prompt).toContain("<sender_participant_roster>");
    expect(episodes[0]?.participants).toEqual(
      expect.arrayContaining(["team", "Alice Nowak", "Bob Chen"]),
    );
    expect(harness.relationalSlotRepository.findBySubjectAndKey(alice, "self.name")).toMatchObject({
      subject_entity_id: alice,
      value: "Alice Nowak",
    });
    expect(harness.relationalSlotRepository.findBySubjectAndKey(bob, "self.name")).toMatchObject({
      subject_entity_id: bob,
      value: "Bob Chen",
    });
    expect(
      harness.relationalSlotRepository.findBySubjectAndKey(
        harness.entityRepository.resolve("user"),
        "self.name",
      ),
    ).toBeNull();
  });

  it("stores entity-id participant terms for senders with the same display name", async () => {
    const harness = await createRelationalExtractorHarness();
    const firstAlex = harness.entityRepository.resolveExternal({
      source: "team-agent.sender",
      externalId: "platform-alex-1",
      canonicalName: "Alex Kim",
      kind: "person",
      provenance: "transport_sender",
    });
    const secondAlex = harness.entityRepository.resolveExternal({
      source: "team-agent.sender",
      externalId: "platform-alex-2",
      canonicalName: "Alex Kim",
      kind: "person",
      provenance: "transport_sender",
    });
    const firstMessage = await harness.writer.append({
      kind: "user_msg",
      content: "Please open the release checklist.",
      sender_entity_id: firstAlex,
    });
    const secondMessage = await harness.writer.append({
      kind: "user_msg",
      content: "Please add the rollback drill.",
      sender_entity_id: secondAlex,
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Release planning",
            narrative: "The two participants named Alex Kim updated the release plan.",
            source_stream_ids: [firstMessage.id, secondMessage.id],
            participants: ["Alex Kim"],
            tags: ["release"],
            confidence: 0.9,
            significance: 0.8,
          },
        ]),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "qwen3",
      entityRepository: harness.entityRepository,
      relationalSlotRepository: harness.relationalSlotRepository,
      clock: harness.clock,
    });

    await extractor.extractFromStream();

    const episode = (await harness.repo.listAll())[0];
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(episode?.participants).toEqual([
      "Alex Kim",
      episodeParticipantEntityIdTerm(firstAlex),
      episodeParticipantEntityIdTerm(secondAlex),
    ]);
    expect(prompt).toContain(`"entity_id":"${firstAlex}","display_name":"Alex Kim"`);
    expect(prompt).toContain(`"entity_id":"${secondAlex}","display_name":"Alex Kim"`);
  });

  it("keeps same-key current-sender relational slots separate for group chat senders", async () => {
    const harness = await createRelationalExtractorHarness();
    const group = harness.entityRepository.resolve("Planning Room", {
      kind: "group",
    });
    const alice = harness.entityRepository.resolve("Alice", {
      kind: "person",
    });
    const ben = harness.entityRepository.resolve("Ben", {
      kind: "person",
    });
    const aliceMessage = await harness.writer.append({
      kind: "user_msg",
      content: "My partner Maya.",
      audience: "Planning Room",
      sender_entity_id: alice,
    });
    harness.clock.advance(10);
    const benMessage = await harness.writer.append({
      kind: "user_msg",
      content: "My partner Sara.",
      audience: "Planning Room",
      sender_entity_id: ben,
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [],
          [
            {
              subject_entity_id: RELATIONAL_SLOT_CURRENT_SENDER_SUBJECT_REF,
              slot_key: "partner.name",
              asserted_value: "Maya",
              source_stream_entry_ids: [aliceMessage.id],
              confirmation_kind: "direct",
            },
            {
              subject_entity_id: RELATIONAL_SLOT_CURRENT_SENDER_SUBJECT_REF,
              slot_key: "partner.name",
              asserted_value: "Sara",
              source_stream_entry_ids: [benMessage.id],
              confirmation_kind: "direct",
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

    expect(
      harness.relationalSlotRepository.findBySubjectAndKey(alice, "partner.name"),
    ).toMatchObject({
      subject_entity_id: alice,
      slot_key: "partner.name",
      value: "Maya",
      evidence_stream_entry_ids: [aliceMessage.id],
    });
    expect(harness.relationalSlotRepository.findBySubjectAndKey(ben, "partner.name")).toMatchObject(
      {
        subject_entity_id: ben,
        slot_key: "partner.name",
        value: "Sara",
        evidence_stream_entry_ids: [benMessage.id],
      },
    );
    expect(
      harness.relationalSlotRepository.findBySubjectAndKey(alice, "partner.name")?.value,
    ).not.toBe("Sara");
    expect(harness.relationalSlotRepository.findBySubjectAndKey(group, "partner.name")).toBeNull();
  });

  it("rejects a current-sender slot update grounded in multiple senders", async () => {
    const harness = await createRelationalExtractorHarness();
    const firstSender = harness.entityRepository.resolve("First sender", { kind: "person" });
    const secondSender = harness.entityRepository.resolve("Second sender", { kind: "person" });
    const firstMessage = await harness.writer.append({
      kind: "user_msg",
      content: "My name is Alex.",
      sender_entity_id: firstSender,
    });
    const secondMessage = await harness.writer.append({
      kind: "user_msg",
      content: "My name is also Alex.",
      sender_entity_id: secondSender,
    });
    const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [],
          [
            {
              subject_entity_id: RELATIONAL_SLOT_CURRENT_SENDER_SUBJECT_REF,
              slot_key: "self.name",
              asserted_value: "Alex",
              source_stream_entry_ids: [firstMessage.id, secondMessage.id],
              confirmation_kind: "direct",
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
      model: "qwen3",
      entityRepository: harness.entityRepository,
      relationalSlotRepository: harness.relationalSlotRepository,
      clock: harness.clock,
    });

    await extractor.extractFromStream();

    expect(harness.relationalSlotRepository.list()).toEqual([]);
    expect(warning).toHaveBeenCalledWith(
      "Skipped ambiguous @current_sender relational slot update.",
      {
        source_stream_entry_ids: [firstMessage.id, secondMessage.id],
        sender_entity_ids: [firstSender, secondSender],
      },
    );
  });

  it("keeps current-sender relational slot subject refs on the default user for self audience", async () => {
    const harness = await createRelationalExtractorHarness();
    const user = await harness.writer.append({
      kind: "user_msg",
      content: "My dog's name is Otto.",
      audience: "self",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse(
          [
            {
              title: "Self-audience note",
              narrative: "The self-audience entry preserved a private note.",
              source_stream_ids: [user.id],
              participants: ["user"],
              tags: ["self"],
              confidence: 0.8,
              significance: 0.6,
            },
          ],
          [
            {
              subject_entity_id: RELATIONAL_SLOT_CURRENT_SENDER_SUBJECT_REF,
              slot_key: "dog.name",
              asserted_value: "Otto",
              source_stream_entry_ids: [user.id],
              confirmation_kind: "direct",
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
    const selfAudience = harness.entityRepository.getSelf()?.id;
    const episodes = await harness.repo.listAll();

    expect(selfAudience).toBeDefined();
    expect(
      harness.entityRepository.resolve("self", {
        kind: "self",
        provenance: "assistant_seeded",
      }),
    ).toBe(selfAudience);
    expect(episodes).toHaveLength(1);
    expect(episodes[0]?.audience_entity_id).toBe(selfAudience);
    expect(
      harness.relationalSlotRepository.findBySubjectAndKey(defaultUser, "dog.name"),
    ).toMatchObject({
      subject_entity_id: defaultUser,
      slot_key: "dog.name",
      value: "Otto",
      evidence_stream_entry_ids: [user.id],
    });
    expect(
      harness.relationalSlotRepository.findBySubjectAndKey(selfAudience!, "dog.name"),
    ).toBeNull();
  });

  it("uses an existing entity id audience as a stable handle and renders its canonical label", async () => {
    const harness = await createRelationalExtractorHarness();
    const groupAudience = harness.entityRepository.resolveExternal({
      source: "team-agent.conversation",
      externalId: "group-42",
      canonicalName: "AI Ninjas",
      kind: "group",
      provenance: "transport_audience_label",
    });
    const user = await harness.writer.append({
      kind: "user_msg",
      content: "Remember the release decision.",
      audience: groupAudience,
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Group release decision",
            narrative: "The group settled the release decision.",
            source_stream_ids: [user.id],
            participants: ["AI Ninjas"],
            tags: ["release"],
            confidence: 0.9,
            significance: 0.8,
          },
        ]),
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

    const [episode] = await harness.repo.listAll();
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(episode?.audience_entity_id).toBe(groupAudience);
    expect(episode?.origin_audience_entity_ids).toEqual([groupAudience]);
    expect(prompt).toContain("(audience routing label) AI Ninjas");
    expect(prompt).not.toContain(`(audience routing label) ${groupAudience}`);
    expect(
      harness.entityRepository.list().some((entity) => entity.canonical_name === groupAudience),
    ).toBe(false);
  });

  it("converges relational slots for default user and current-sender subject refs under one audience", async () => {
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
              subject_entity_id: RELATIONAL_SLOT_CURRENT_SENDER_SUBJECT_REF,
              slot_key: "dog.name",
              asserted_value: "Otto",
              source_stream_entry_ids: [dog.id],
              confirmation_kind: "direct",
            },
            {
              subject_entity_id: tom,
              slot_key: "partner.name",
              asserted_value: "Elena",
              source_stream_entry_ids: [partner.id],
              confirmation_kind: "direct",
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
              confirmation_kind: "direct",
            },
            {
              subject_entity_id: tom,
              slot_key: "partner.name",
              asserted_value: "Maya",
              source_stream_entry_ids: [maya.id],
              confirmation_kind: "direct",
            },
            {
              subject_entity_id: tom,
              slot_key: "partner.name",
              asserted_value: "Clara",
              source_stream_entry_ids: [clara.id],
              confirmation_kind: "direct",
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
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        commitmentMigrations,
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
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        commitmentMigrations,
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

  it("skips an interior-only turn but extracts an agent-only turn", async () => {
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

    await writer.append({
      kind: "thought",
      content: "Nothing here needs saying yet.",
    });
    clock.advance(1);
    await writer.append({
      kind: "agent_suppressed",
      content: {
        reason: "autonomous_no_output",
      },
    });

    const llm = new FakeLLMClient({
      responses: [],
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

    clock.advance(1);
    const agent = await writer.append({
      kind: "agent_msg",
      content: "I have been thinking about the deadline.",
    });
    llm.pushResponse(
      createEpisodeToolResponse([
        {
          title: "Unprompted note",
          narrative: "I said something without being asked.",
          source_stream_ids: [agent.id],
          participants: ["Borg"],
          tags: ["outbound"],
          confidence: 0.8,
          significance: 0.7,
        },
      ]),
    );

    await expect(extractor.extractFromStream()).resolves.toEqual({
      inserted: 1,
      updated: 0,
      skipped: 0,
    });
    expect(llm.requests).toHaveLength(1);
    expect((await repo.listAll()).map((episode) => episode.source_stream_ids)).toEqual([
      [agent.id],
    ]);
  });

  it("rejects hallucinated source stream ids", async () => {
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
      ),
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
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        commitmentMigrations,
      ),
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
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        commitmentMigrations,
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
