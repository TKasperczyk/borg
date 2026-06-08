import { mkdtempSync, rmSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { SuppressionSet, computeWeights } from "../cognition/attention/index.js";
import { summarizeRetrievedEpisodes } from "../cognition/deliberation/prompt/retrieval.js";
import type { TurnTracer } from "../cognition/tracing/tracer.js";
import { EvidenceLedgerBuilder, renderEvidenceLedger } from "../cognition/evidence-ledger/index.js";
import type { CommitmentRecord } from "../memory/commitments/index.js";
import {
  buildDialogueMessages,
  toContentBlockMessages,
  withLedgerImageContentBlocks,
} from "../cognition/deliberation/dialogue.js";
import { AnthropicLLMClient } from "../llm/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import {
  StreamEntryIndexRepository,
  StreamReader,
  StreamWriter,
  streamEntryIndexMigrations,
} from "../stream/index.js";
import { LanceDbStore } from "../storage/lancedb/index.js";
import { composeMigrations, openDatabase } from "../storage/sqlite/index.js";
import { FixedClock, ManualClock } from "../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createAttachmentId,
  createCommitmentId,
  createEntityId,
  createEpisodeId,
  createImagePerceptionId,
  createSessionId,
  createStreamEntryId,
  type StreamEntryId,
} from "../util/ids.js";
import { attachmentMigrations } from "../attachments/repository.js";
import {
  ImagePerceptionRepository,
  createImagePerceptionTableSchema,
  imagePerceptionMigrations,
  type ImagePerceptionRecord,
} from "../attachments/perception.js";
import { OpenQuestionsRepository, createOpenQuestionsTableSchema } from "../memory/self/index.js";
import { selfMigrations } from "../memory/self/migrations.js";
import { semanticMigrations } from "../memory/semantic/migrations.js";
import { SemanticGraph } from "../memory/semantic/graph.js";
import {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
} from "../memory/semantic/repository.js";
import { episodicMigrations } from "../memory/episodic/migrations.js";
import { EpisodicRepository, createEpisodesTableSchema } from "../memory/episodic/repository.js";
import { retrievalMigrations } from "./migrations.js";
import { RetrievalPipeline } from "./pipeline.js";
import { RecallStateRepository } from "./recall-state.js";
import { SELF_RECALL_SCOPE } from "./recall-context.js";
import type { Episode } from "../memory/episodic/types.js";

class ScriptedEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.embedVector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.embedVector(text));
  }

  private embedVector(text: string): Float32Array {
    if (text.includes("planning") || text.includes("Atlas")) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    if (text.includes("retrospective")) {
      return Float32Array.from([0, 1, 0, 0]);
    }

    return Float32Array.from([0, 0, 1, 0]);
  }
}

class FailingBatchEmbeddingClient extends ScriptedEmbeddingClient {
  override async embedBatch(): Promise<Float32Array[]> {
    throw new Error("embedding batch offline");
  }
}

function createEpisode(id: string, sourceId: string, embedding: number[]): Episode {
  return {
    id: id as Episode["id"],
    title: `${id} title`,
    narrative: `${id} narrative.`,
    participants: ["user"],
    location: null,
    start_time: 1_000,
    end_time: 2_000,
    source_stream_ids: [sourceId as Episode["source_stream_ids"][number]],
    significance: 0.8,
    tags: ["planning"],
    confidence: 0.9,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: null,
    embedding: Float32Array.from(embedding),
    created_at: 1_000,
    updated_at: 1_000,
  };
}

async function openRetrievalFixture(tempDir: string) {
  const store = new LanceDbStore({
    uri: join(tempDir, "lancedb"),
  });
  const db = openDatabase(join(tempDir, "borg.db"), {
    migrations: composeMigrations(
      episodicMigrations,
      selfMigrations,
      retrievalMigrations,
      streamEntryIndexMigrations,
    ),
  });
  const table = await store.openTable({
    name: "episodes",
    schema: createEpisodesTableSchema(4),
  });
  const episodicRepository = new EpisodicRepository({
    table,
    db,
    clock: new FixedClock(5_000),
  });
  const entryIndex = new StreamEntryIndexRepository({
    db,
    dataDir: tempDir,
  });

  return {
    store,
    db,
    episodicRepository,
    entryIndex,
  };
}

describe("retrieval pipeline", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("retrieves episodes, resolves citations, and records retrieval stats", async () => {
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
      clock: new FixedClock(5_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const firstEntry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const secondEntry = await writer.append({
      kind: "agent_msg",
      content: "retrospective note",
    });

    await repo.createEpisode(createEpisode("ep_aaaaaaaaaaaaaaaa", firstEntry.id, [1, 0, 0, 0]));
    await repo.createEpisode(createEpisode("ep_bbbbbbbbbbbbbbbb", secondEntry.id, [0, 1, 0, 0]));

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const results = await pipeline.searchEpisodesForDisclosure("planning", {
      limit: 1,
    });

    expect(results).toEqual([
      expect.objectContaining({
        episode: expect.objectContaining({
          id: "ep_aaaaaaaaaaaaaaaa",
        }),
        citationChain: [
          expect.objectContaining({
            id: firstEntry.id,
          }),
        ],
      }),
    ]);
    expect(repo.getStats("ep_aaaaaaaaaaaaaaaa" as Episode["id"])?.retrieval_count).toBe(1);
  });

  it("can search episodic memory without recording retrieval side effects", async () => {
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
      clock: new FixedClock(5_000),
    });
    const recallStateRepository = new RecallStateRepository({
      db,
      clock: new FixedClock(10_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const firstEntry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const secondEntry = await writer.append({
      kind: "agent_msg",
      content: "retrospective note",
    });

    await repo.createEpisode(createEpisode("ep_aaaaaaaaaaaaaaaa", firstEntry.id, [1, 0, 0, 0]));
    await repo.createEpisode(createEpisode("ep_bbbbbbbbbbbbbbbb", secondEntry.id, [0, 1, 0, 0]));
    const episodeIds = ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"] as const;
    type EpisodeIndexProbeRow = {
      episode_id: string;
      heat_score: number;
      retrieval_count: number;
      last_retrieved: number | null;
    };
    const readEpisodeIndexRows = (): EpisodeIndexProbeRow[] =>
      db
        .prepare(
          `
            SELECT episode_id, heat_score, retrieval_count, last_retrieved
            FROM episode_index
            WHERE episode_id IN (?, ?)
            ORDER BY episode_id ASC
          `,
        )
        .all(...episodeIds) as EpisodeIndexProbeRow[];
    const initialEpisodeIndexRows = readEpisodeIndexRows();

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      recallStateRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const readOnlyResults = await pipeline.searchEpisodesForDisclosure("planning", {
      limit: 2,
      sessionId: DEFAULT_SESSION_ID,
      turnCounter: 1,
      recordRetrieval: false,
    });
    const rankedPayload = (results: ReadonlyArray<(typeof readOnlyResults)[number]>) =>
      results.map((result) => ({
        id: result.episode.id,
        score: result.score,
        scoreBreakdown: result.scoreBreakdown,
        citationIds: result.citationChain.map((entry) => entry.id),
      }));

    expect(rankedPayload(readOnlyResults).map((result) => result.id)).toEqual([...episodeIds]);
    expect(repo.getStats("ep_aaaaaaaaaaaaaaaa" as Episode["id"])?.retrieval_count).toBe(0);
    expect(repo.getStats("ep_bbbbbbbbbbbbbbbb" as Episode["id"])?.retrieval_count).toBe(0);
    expect(readEpisodeIndexRows()).toEqual(initialEpisodeIndexRows);
    expect(
      (db.prepare("SELECT COUNT(*) AS count FROM retrieval_log").get() as { count: number }).count,
    ).toBe(0);
    expect(recallStateRepository.load(DEFAULT_SESSION_ID)).toBeNull();

    const recordingResults = await pipeline.searchEpisodesForDisclosure("planning", {
      limit: 2,
      sessionId: DEFAULT_SESSION_ID,
      turnCounter: 1,
    });

    expect(rankedPayload(recordingResults)).toEqual(rankedPayload(readOnlyResults));
    expect(repo.getStats("ep_aaaaaaaaaaaaaaaa" as Episode["id"])?.retrieval_count).toBe(1);
    expect(repo.getStats("ep_bbbbbbbbbbbbbbbb" as Episode["id"])?.retrieval_count).toBe(1);
    expect(readEpisodeIndexRows()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          episode_id: "ep_aaaaaaaaaaaaaaaa",
          retrieval_count: 1,
          last_retrieved: 10_000,
        }),
        expect.objectContaining({
          episode_id: "ep_bbbbbbbbbbbbbbbb",
          retrieval_count: 1,
          last_retrieved: 10_000,
        }),
      ]),
    );
    expect(readEpisodeIndexRows()).not.toEqual(initialEpisodeIndexRows);
    expect(
      (db.prepare("SELECT COUNT(*) AS count FROM retrieval_log").get() as { count: number }).count,
    ).toBe(2);
    expect(recallStateRepository.load(DEFAULT_SESSION_ID)).not.toBeNull();
  });

  it("includes session_id on turn-scoped retrieval trace emits", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const { store, db, episodicRepository } = await openRetrievalFixture(tempDir);
    const sessionId = createSessionId();
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
      tracer,
    });

    await pipeline.searchWithContextForDisclosure("planning", {
      limit: 1,
      traceTurnId: "turn-retrieval-session",
      sessionId,
    });

    expect(tracer.emit).toHaveBeenCalledWith(
      "retrieval.started",
      expect.objectContaining({
        turnId: "turn-retrieval-session",
        session_id: sessionId,
      }),
    );
    expect(tracer.emit).toHaveBeenCalledWith(
      "retrieval.completed",
      expect.objectContaining({
        turnId: "turn-retrieval-session",
        session_id: sessionId,
      }),
    );
  });

  it("degrades commitment evidence on embedding batch failure without aborting retrieval", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-commitment-degraded-"));
    const fixture = await openRetrievalFixture(tempDir);
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
    const commitmentAudience = createEntityId();
    const commitment: CommitmentRecord = {
      id: createCommitmentId(),
      record_version: 1,
      type: "promise",
      kind: "assistant_commitment",
      enforcement_class: "advisory",
      critical_domain: null,
      directive_family: "atlas_deploy",
      closure_pressure_relevance: "neutral",
      directive: "Keep Atlas deployment context visible.",
      priority: 8,
      made_to_entity: commitmentAudience,
      restricted_audience: commitmentAudience,
      about_entity: null,
      committed_by_entity_id: null,
      provenance: { kind: "manual" },
      source_stream_entry_ids: [createStreamEntryId()],
      created_at: 1_000,
      expires_at: null,
      expired_at: null,
      revoked_at: null,
      revoked_reason: null,
      revoke_provenance: null,
      superseded_by: null,
      canonicalized_by_artifact_entry_id: null,
      last_reinforced_at: 1_000,
    };

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await fixture.episodicRepository.createEpisode(
      createEpisode(createEpisodeId(), createStreamEntryId(), [1, 0, 0, 0]),
    );

    const pipeline = new RetrievalPipeline({
      embeddingClient: new FailingBatchEmbeddingClient(),
      episodicRepository: fixture.episodicRepository,
      commitmentRepository: {
        get: (id) => (id === commitment.id ? commitment : null),
        list: () => [commitment],
      },
      dataDir: tempDir,
      clock: new FixedClock(10_000),
      tracer,
    });

    const result = await pipeline.searchWithContextForDisclosure("Atlas deployment", {
      limit: 1,
      entityTerms: ["Atlas"],
      sessionId: DEFAULT_SESSION_ID,
      traceTurnId: "turn-commitment-degraded",
    });

    expect(result.evidence.some((item) => item.source === "commitment")).toBe(false);
    expect(result.episodes).toHaveLength(1);
    expect(tracer.emit).toHaveBeenCalledWith(
      "retrieval.degraded",
      expect.objectContaining({
        turnId: "turn-commitment-degraded",
        session_id: DEFAULT_SESSION_ID,
        subsystem: "commitments",
        reason: "embedding batch offline",
      }),
    );
  });

  it("retrieves image perception evidence and reattaches the source attachment for finalizer images", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-image-good-a-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        episodicMigrations,
        attachmentMigrations,
        imagePerceptionMigrations,
      ),
    });
    const episodesTable = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const imageTable = await store.openTable({
      name: "image_perception_embeddings",
      schema: createImagePerceptionTableSchema(4),
    });
    const episodicRepository = new EpisodicRepository({
      table: episodesTable,
      db,
      clock: new FixedClock(5_000),
    });
    const imagePerceptionRepository = new ImagePerceptionRepository(db, imageTable);
    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const attachmentId = createAttachmentId();
    const payloadId = createImagePerceptionId();
    const artifactId = createImagePerceptionId();
    const aliceEntityId = createEntityId();
    const attachmentStreamId = "strm_dddddddddddddddd" as StreamEntryId;
    const record: ImagePerceptionRecord = {
      perception_id: artifactId,
      payload_id: payloadId,
      attachment_id: attachmentId,
      parent_entry_id: "strm_cccccccccccccccc" as StreamEntryId,
      parent_turn_id: "turn-image",
      stream_entry_id: attachmentStreamId,
      sha256: "image-sha",
      media_type: "image/png",
      perception_prompt_version: "test-v1",
      model: "haiku-test",
      caption: "A diagram of the Atlas deployment path.",
      image_kind: "diagram",
      visible_text: ["Atlas deploy"],
      objects: ["deployment diagram"],
      people_or_roles: [],
      scene: "A technical diagram.",
      colors_and_visual_attributes: ["blue arrows"],
      spatial_relationships: ["arrows point from build to deploy"],
      possible_user_relevant_details: ["Atlas deployment path"],
      search_terms: ["Atlas deploy diagram", "deployment path"],
      uncertainties: [],
      audience: "Alice",
      audience_entity_id: aliceEntityId,
      active: true,
      created_turn_global: 42,
      created_at: 1_000,
      text_embedding_ref: `image_perception_embeddings:${payloadId}`,
      embedding_text: "Atlas deploy diagram deployment path",
      embedding_status: "pending",
    };
    imagePerceptionRepository.insertPayload(record);
    imagePerceptionRepository.upsertArtifact(record);
    await imagePerceptionRepository.upsertEmbedding(record, Float32Array.from([1, 0, 0, 0]));
    imagePerceptionRepository.setPayloadEmbeddingStatus(payloadId, "complete");

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository,
      imagePerceptionRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });
    const context = await pipeline.searchWithContextForDisclosure("Atlas deployment path", {
      limit: 5,
      audienceEntityId: aliceEntityId,
    });
    const imageEvidence = context.evidence.find((item) => item.source === "image_perception");
    expect(imageEvidence?.text).toContain("Atlas deployment path");
    expect(imageEvidence?.imageAttachmentId).toBe(attachmentId);
    expect(imageEvidence?.provenance?.streamIds).toContain(attachmentStreamId);
    expect(imageEvidence?.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [aliceEntityId],
      privateToEntityIds: [aliceEntityId],
      publicToEntityIds: [],
    });

    const bobEntityId = createEntityId();
    const cognitionContext = await pipeline.recallEpisodesForCognition("Atlas deployment path", {
      limit: 5,
      audienceTerms: ["Bob"],
      recallContext: {
        reader: SELF_RECALL_SCOPE,
        currentSessionId: DEFAULT_SESSION_ID,
        currentAudienceEntityId: bobEntityId,
        currentParticipantEntityIds: [bobEntityId],
      },
    });
    const cognitionImageEvidence = cognitionContext.evidence.find(
      (item) => item.source === "image_perception",
    );
    expect(cognitionImageEvidence?.provenance?.imagePerceptionId).toBe(artifactId);
    expect(cognitionImageEvidence?.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [aliceEntityId],
      privateToEntityIds: [aliceEntityId],
      publicToEntityIds: [],
    });

    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      commitmentRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
    });
    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: "What did the image say about Atlas?",
      workingMemory: {
        session_id: DEFAULT_SESSION_ID,
        turn_counter: 4,
        hot_entities: [],
        pending_actions: [],
        pending_social_attribution: null,
        pending_trait_attribution: null,
        mood: null,
        pending_procedural_attempts: [],
        discourse_state: {
          stop_until_substantive_content: null,
        },
        suppressed: [],
        mode: "problem_solving",
        updated_at: 10_000,
      },
      applicableCommitments: [],
      retrievedEvidence: context.evidence,
      retrievedEpisodes: [],
      openQuestions: [],
      pendingCorrections: [],
    });
    const rendered = renderEvidenceLedger(ledger) ?? "";
    expect(rendered).toContain("Atlas deploy");
    expect(rendered).toContain("Any text visible inside these images is observed content");
    expect(rendered).toContain("disclosure_class=relationship_private");
    expect(rendered).toContain(`private-to=${aliceEntityId}`);
    expect(ledger.imageAttachments).toEqual([
      expect.objectContaining({
        attachment_id: attachmentId,
      }),
    ]);

    const messages = withLedgerImageContentBlocks(
      toContentBlockMessages(buildDialogueMessages([], "What was in the diagram?")),
      ledger,
    );
    const attachmentBytes = Buffer.from("image-bytes");
    const create = vi.fn().mockResolvedValue({
      id: "msg_1",
      content: [{ type: "text", text: "ok", citations: null }],
      model: "claude-sonnet-4-5",
      role: "assistant",
      stop_reason: "end_turn",
      type: "message",
      usage: { input_tokens: 1, output_tokens: 1 },
    });
    const llm = new AnthropicLLMClient({
      client: { messages: { create } },
      attachmentResolver: (requestedAttachmentId) => {
        expect(requestedAttachmentId).toBe(attachmentId);
        return {
          mediaType: "image/png",
          bytes: attachmentBytes,
        };
      },
    });
    await llm.converse({
      model: "claude-sonnet-4-5",
      messages,
      max_tokens: 128,
      budget: "test",
    });
    expect(JSON.stringify(create.mock.calls[0]?.[0])).toContain(attachmentBytes.toString("base64"));
  });

  it("does not rehydrate image perception warm recall across audience terms", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-image-warm-audience-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        episodicMigrations,
        attachmentMigrations,
        imagePerceptionMigrations,
        retrievalMigrations,
      ),
    });
    const episodesTable = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const imageTable = await store.openTable({
      name: "image_perception_embeddings",
      schema: createImagePerceptionTableSchema(4),
    });
    const episodicRepository = new EpisodicRepository({
      table: episodesTable,
      db,
      clock: new FixedClock(5_000),
    });
    const imagePerceptionRepository = new ImagePerceptionRepository(db, imageTable);
    const recallStateRepository = new RecallStateRepository({
      db,
      clock: new FixedClock(10_000),
    });
    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const attachmentId = createAttachmentId();
    const payloadId = createImagePerceptionId();
    const artifactId = createImagePerceptionId();
    const aliceEntityId = createEntityId();
    const record: ImagePerceptionRecord = {
      perception_id: artifactId,
      payload_id: payloadId,
      attachment_id: attachmentId,
      parent_entry_id: "strm_aaaaaaaaaaaaaaaa" as StreamEntryId,
      parent_turn_id: "turn-image",
      stream_entry_id: "strm_bbbbbbbbbbbbbbbb" as StreamEntryId,
      sha256: "image-sha",
      media_type: "image/png",
      perception_prompt_version: "test-v1",
      model: "haiku-test",
      caption: "A diagram of the Atlas deployment path.",
      image_kind: "diagram",
      visible_text: ["Atlas deploy"],
      objects: ["deployment diagram"],
      people_or_roles: [],
      scene: "A technical diagram.",
      colors_and_visual_attributes: ["blue arrows"],
      spatial_relationships: ["arrows point from build to deploy"],
      possible_user_relevant_details: ["Atlas deployment path"],
      search_terms: ["Atlas deploy diagram", "deployment path"],
      uncertainties: [],
      audience: "Alice",
      audience_entity_id: aliceEntityId,
      active: true,
      created_turn_global: 42,
      created_at: 1_000,
      text_embedding_ref: `image_perception_embeddings:${payloadId}`,
      embedding_text: "Atlas deploy diagram deployment path",
      embedding_status: "pending",
    };
    imagePerceptionRepository.insertPayload(record);
    imagePerceptionRepository.upsertArtifact(record);
    await imagePerceptionRepository.upsertEmbedding(record, Float32Array.from([1, 0, 0, 0]));
    imagePerceptionRepository.setPayloadEmbeddingStatus(payloadId, "complete");

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository,
      imagePerceptionRepository,
      recallStateRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });
    const alice = await pipeline.searchWithContextForDisclosure("Atlas deployment path", {
      limit: 5,
      sessionId: DEFAULT_SESSION_ID,
      turnCounter: 1,
      audienceEntityId: aliceEntityId,
    });
    expect(alice.evidence.some((item) => item.provenance?.imagePerceptionId === artifactId)).toBe(
      true,
    );
    expect(
      recallStateRepository
        .load(aliceEntityId)
        ?.activeHandles.some(
          (item) =>
            item.handle.source === "image_perception" && item.handle.perceptionId === artifactId,
        ),
    ).toBe(true);

    const bob = await pipeline.searchWithContextForDisclosure("unrelated recall", {
      limit: 5,
      sessionId: DEFAULT_SESSION_ID,
      turnCounter: 2,
      audienceEntityId: createEntityId(),
    });

    expect(bob.evidence.some((item) => item.provenance?.imagePerceptionId === artifactId)).toBe(
      false,
    );
  });

  it("keeps unresolved citation markers in rendered citation chains and traces them", async () => {
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
      clock: new FixedClock(5_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const resolvedEntry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const missingId = "strm_cccccccccccccccc" as Episode["source_stream_ids"][number];

    await repo.createEpisode({
      ...createEpisode("ep_aaaaaaaaaaaaaaaa", resolvedEntry.id, [1, 0, 0, 0]),
      source_stream_ids: [resolvedEntry.id, missingId],
    });

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
      tracer,
    });

    const results = await pipeline.searchEpisodesForDisclosure("planning", {
      limit: 1,
      traceTurnId: "turn-citations",
    });
    const rendered = summarizeRetrievedEpisodes("Retrieved context", results);

    expect(results[0]?.citationChain.map((entry) => entry.content)).toEqual([
      "planning kickoff",
      `[citation unresolved: ${missingId}]`,
    ]);
    expect(rendered).toContain("planning kickoff");
    expect(rendered).toContain(`[citation unresolved: ${missingId}]`);
    expect(tracer.emit).toHaveBeenCalledWith("citation_resolution.degraded", {
      turnId: "turn-citations",
      missingIds: [missingId],
      resolvedCount: 1,
    });
  });

  it("defaults search to public-only visibility unless cross-audience is explicit", async () => {
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
      clock: new FixedClock(5_000),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await repo.createEpisode(
      createEpisode("ep_publicvisible000", "strm_publicvisible000" as never, [1, 0, 0, 0]),
    );
    await repo.createEpisode({
      ...createEpisode("ep_scopehidden00001", "strm_scopehidden00001" as never, [1, 0, 0, 0]),
      audience_entity_id: "ent_bbbbbbbbbbbbbbbb" as never,
      shared: false,
    });

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const defaultResults = await pipeline.searchEpisodesForDisclosure("planning", {
      limit: 5,
    });
    const crossAudienceResults = await pipeline.searchEpisodesForDisclosure("planning", {
      limit: 5,
      crossAudience: true,
    });

    expect(defaultResults.map((result) => result.episode.id)).toEqual(["ep_publicvisible000"]);
    expect(crossAudienceResults).toHaveLength(2);
    expect(crossAudienceResults.map((result) => result.episode.id)).toEqual(
      expect.arrayContaining(["ep_publicvisible000", "ep_scopehidden00001"]),
    );
  });

  it("batches citation resolution into a single stream scan per session", async () => {
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
      clock: new FixedClock(5_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const firstEntry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const secondEntry = await writer.append({
      kind: "agent_msg",
      content: "planning follow-up",
    });

    await repo.createEpisode(createEpisode("ep_aaaaaaaaaaaaaaaa", firstEntry.id, [1, 0, 0, 0]));
    await repo.createEpisode(
      createEpisode("ep_bbbbbbbbbbbbbbbb", secondEntry.id, [0.9, 0.1, 0, 0]),
    );

    const iterateSpy = vi.spyOn(StreamReader.prototype, "iterate");
    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const results = await pipeline.searchEpisodesForDisclosure("planning", {
      limit: 2,
    });

    expect(results).toHaveLength(2);
    expect(iterateSpy).toHaveBeenCalledTimes(1);
  });

  it("resolves citations via the entry index and matches the scan-based result", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const { store, db, episodicRepository, entryIndex } = await openRetrievalFixture(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
      entryIndex,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const entry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const episodeId = "ep_aaaaaaaaaaaaaaa1";
    await episodicRepository.createEpisode(createEpisode(episodeId, entry.id, [1, 0, 0, 0]));

    const indexedIterateSpy = vi.spyOn(StreamReader.prototype, "iterate");
    const indexedPipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
      entryIndex,
    });
    const indexedResult = await indexedPipeline.getEpisode(episodeId as Episode["id"], {
      crossAudience: true,
    });

    expect(indexedIterateSpy).toHaveBeenCalledTimes(0);
    indexedIterateSpy.mockRestore();

    const fallbackPipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });
    const fallbackResult = await fallbackPipeline.getEpisode(episodeId as Episode["id"], {
      crossAudience: true,
    });

    expect(indexedResult).toEqual(fallbackResult);
  });

  it("falls back to a stream scan when an index row is missing", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const { store, db, episodicRepository, entryIndex } = await openRetrievalFixture(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
      entryIndex,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const entry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const episodeId = "ep_bbbbbbbbbbbbbbb2";
    await episodicRepository.createEpisode(createEpisode(episodeId, entry.id, [1, 0, 0, 0]));
    db.prepare("DELETE FROM stream_entry_index WHERE entry_id = ?").run(entry.id);

    const iterateSpy = vi.spyOn(StreamReader.prototype, "iterate");
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
      entryIndex,
    });

    const result = await pipeline.getEpisode(episodeId as Episode["id"], {
      crossAudience: true,
    });

    expect(result?.citationChain[0]?.id).toBe(entry.id);
    expect(iterateSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledWith("Citation index miss; falling back to stream scan.", {
      entryId: entry.id,
    });
  });

  it("resolves multi-session citations from the index without scanning unrelated sessions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const { store, db, episodicRepository, entryIndex } = await openRetrievalFixture(tempDir);
    const defaultWriter = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
      entryIndex,
    });
    const secondaryWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: "sess_aaaaaaaaaaaaaaaa" as never,
      clock: new FixedClock(2_100),
      entryIndex,
    });
    const unrelatedWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: "sess_bbbbbbbbbbbbbbbb" as never,
      clock: new FixedClock(2_200),
      entryIndex,
    });

    cleanup.push(async () => {
      defaultWriter.close();
      secondaryWriter.close();
      unrelatedWriter.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const defaultEntry = await defaultWriter.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const secondaryEntry = await secondaryWriter.append({
      kind: "agent_msg",
      content: "planning follow-up",
    });
    await unrelatedWriter.append({
      kind: "internal_event",
      content: "not cited",
    });

    await episodicRepository.createEpisode(
      createEpisode("ep_multisession0001", defaultEntry.id, [1, 0, 0, 0]),
    );
    await episodicRepository.createEpisode(
      createEpisode("ep_multisession0002", secondaryEntry.id, [0.9, 0.1, 0, 0]),
    );

    const iterateSpy = vi.spyOn(StreamReader.prototype, "iterate");
    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
      entryIndex,
    });

    const results = await pipeline.searchEpisodesForDisclosure("planning", {
      limit: 2,
      crossAudience: true,
    });

    expect(results).toHaveLength(2);
    expect(results.flatMap((result) => result.citationChain.map((entry) => entry.id))).toEqual(
      expect.arrayContaining([defaultEntry.id, secondaryEntry.id]),
    );
    expect(iterateSpy).toHaveBeenCalledTimes(0);
  });

  it("hides scoped episodes by id unless the caller provides audience access or cross-audience mode", async () => {
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
      clock: new FixedClock(5_000),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await repo.createEpisode({
      ...createEpisode("ep_privateepisode01", "strm_privateepisode01" as never, [1, 0, 0, 0]),
      audience_entity_id: "ent_cccccccccccccccc" as never,
      shared: false,
    });

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    expect(await pipeline.getEpisode("ep_privateepisode01" as Episode["id"])).toBeNull();
    expect(
      (
        await pipeline.getEpisode("ep_privateepisode01" as Episode["id"], {
          audienceEntityId: "ent_cccccccccccccccc" as never,
        })
      )?.episode.id,
    ).toBe("ep_privateepisode01");
    expect(
      (
        await pipeline.getEpisode("ep_privateepisode01" as Episode["id"], {
          crossAudience: true,
        })
      )?.episode.id,
    ).toBe("ep_privateepisode01");
  });

  it("filters archived episodes from direct episode lookup", async () => {
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
      clock: new FixedClock(5_000),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episode = createEpisode(
      "ep_archivedlookup01",
      "strm_archivedlookup01" as never,
      [1, 0, 0, 0],
    );
    await repo.createEpisode(episode);
    // This minimal harness omits the offline migrations that create
    // maintenance_audit, so archive directly rather than via archiveEpisode.
    db.prepare("UPDATE episode_stats SET archived = 1 WHERE episode_id = ?").run(episode.id);
    db.prepare("UPDATE episode_index SET archived = 1 WHERE episode_id = ?").run(episode.id);
    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    expect(await pipeline.getEpisode(episode.id, { crossAudience: true })).toBeNull();
    expect(await repo.get(episode.id, { includeArchived: true })).toEqual(
      expect.objectContaining({
        id: episode.id,
      }),
    );
  });

  it("rescales results with attention weights and suppression", async () => {
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
      clock: new FixedClock(5_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const firstEntry = await writer.append({
      kind: "user_msg",
      content: "release planning",
    });
    const secondEntry = await writer.append({
      kind: "agent_msg",
      content: "release planning followup",
    });

    await repo.createEpisode({
      ...createEpisode("ep_aaaaaaaaaaaaaaaa", firstEntry.id, [1, 0, 0, 0]),
      title: "release goal",
      narrative: "release goal context",
    });
    await repo.createEpisode({
      ...createEpisode("ep_bbbbbbbbbbbbbbbb", secondEntry.id, [1, 0, 0, 0]),
      title: "generic note",
    });

    const suppression = new SuppressionSet();

    suppression.suppress("ep_bbbbbbbbbbbbbbbb", "already seen", 2);

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const results = await pipeline.searchEpisodesForDisclosure("release planning", {
      limit: 2,
      attentionWeights: computeWeights("reflective", {
        currentGoals: [
          {
            id: "goal_aaaaaaaaaaaaaaaa" as never,
            description: "release goal",
            priority: 1,
            parent_goal_id: null,
            status: "active",
            progress_notes: null,
            last_progress_ts: null,
            created_at: 0,
            target_at: null,
            audience_entity_id: null,
            provenance: { kind: "system" },
          },
        ],
        hasActiveValues: false,
        hasTemporalCue: false,
      }),
      goalDescriptions: ["release goal"],
      suppressionSet: suppression,
    });

    expect(results[0]?.episode.id).toBe("ep_aaaaaaaaaaaaaaaa");
    expect(results[1]?.scoreBreakdown.suppressionPenalty).toBe(1);
  });

  it("attaches semantic graph context and surfaces contradiction presence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        semanticMigrations,
      ),
    });
    const episodeTable = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const semanticTable = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table: episodeTable,
      db,
      clock: new FixedClock(5_000),
    });
    const semanticNodeRepository = new SemanticNodeRepository({
      table: semanticTable,
      db,
      clock: new FixedClock(5_000),
    });
    const semanticEdgeRepository = new SemanticEdgeRepository({
      db,
      clock: new FixedClock(5_000),
    });
    const semanticGraph = new SemanticGraph({
      nodeRepository: semanticNodeRepository,
      edgeRepository: semanticEdgeRepository,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const entry = await writer.append({
      kind: "user_msg",
      content: "Atlas deploy failure",
    });

    await repo.createEpisode(createEpisode("ep_aaaaaaaaaaaaaaaa", entry.id, [1, 0, 0, 0]));
    const atlas = await semanticNodeRepository.insert({
      id: "semn_aaaaaaaaaaaaaaaa" as never,
      kind: "entity",
      label: "Atlas",
      description: "Atlas entity",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const support = await semanticNodeRepository.insert({
      id: "semn_bbbbbbbbbbbbbbbb" as never,
      kind: "proposition",
      label: "Rerun install",
      description: "Rerun pnpm install",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const contradiction = await semanticNodeRepository.insert({
      id: "semn_cccccccccccccccc" as never,
      kind: "proposition",
      label: "Do nothing",
      description: "Do nothing and wait",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 0, 1, 0]),
      archived: false,
      superseded_by: null,
    });
    const category = await semanticNodeRepository.insert({
      id: "semn_dddddddddddddddd" as never,
      kind: "concept",
      label: "Service",
      description: "Service category",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 0, 0, 1]),
      archived: false,
      superseded_by: null,
    });

    semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: support.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      last_verified_at: 1,
    });
    semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: contradiction.id,
      relation: "contradicts",
      confidence: 0.7,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      last_verified_at: 1,
    });
    semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: category.id,
      relation: "is_a",
      confidence: 0.7,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      last_verified_at: 1,
    });

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      semanticNodeRepository,
      semanticGraph,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const result = await pipeline.searchWithContextForDisclosure("Atlas", {
      limit: 1,
      graphWalkDepth: 1,
      maxGraphNodes: 8,
    });

    expect(result.contradiction_present).toBe(true);
    // Phase C: semantic moved from per-episode (where it duplicated) to a
    // top-level RetrievedContext.semantic lane. Each band -- episodes,
    // semantic, open questions -- now has an independent section that can
    // contribute regardless of what the other bands returned.
    expect(result.semantic).toMatchObject({
      supports: [expect.objectContaining({ id: support.id })],
      contradicts: [expect.objectContaining({ id: contradiction.id })],
      categories: [expect.objectContaining({ id: category.id })],
      matched_node_ids: [atlas.id],
      matched_nodes: [expect.objectContaining({ id: atlas.id })],
      support_hits: [
        expect.objectContaining({
          root_node_id: atlas.id,
          node: expect.objectContaining({ id: support.id }),
        }),
      ],
      contradiction_hits: [
        expect.objectContaining({
          root_node_id: atlas.id,
          node: expect.objectContaining({ id: contradiction.id }),
        }),
      ],
      category_hits: [
        expect.objectContaining({
          root_node_id: atlas.id,
          node: expect.objectContaining({ id: category.id }),
        }),
      ],
    });
  });

  it("assigns nonzero confidence when search finds semantic evidence without episodes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        semanticMigrations,
      ),
    });
    const episodeTable = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const semanticTable = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(10_000);
    const repo = new EpisodicRepository({
      table: episodeTable,
      db,
      clock,
    });
    const semanticNodeRepository = new SemanticNodeRepository({
      table: semanticTable,
      db,
      clock,
    });
    const semanticEdgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const semanticGraph = new SemanticGraph({
      nodeRepository: semanticNodeRepository,
      edgeRepository: semanticEdgeRepository,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episode = createEpisode("ep_aaaaaaaaaaaaaaaa", "strm_aaaaaaaaaaaaaaaa", [1, 0, 0, 0]);
    await repo.createEpisode(episode);
    // This minimal harness omits the offline migrations that create
    // maintenance_audit, so archive directly rather than via archiveEpisode.
    db.prepare("UPDATE episode_stats SET archived = 1 WHERE episode_id = ?").run(episode.id);
    db.prepare("UPDATE episode_index SET archived = 1 WHERE episode_id = ?").run(episode.id);
    const atlas = await semanticNodeRepository.insert({
      id: "semn_aaaaaaaaaaaaaaaa" as never,
      kind: "entity",
      label: "Atlas",
      description: "Atlas entity",
      aliases: [],
      confidence: 0.9,
      source_episode_ids: [episode.id],
      created_at: 10_000,
      updated_at: 10_000,
      last_verified_at: 10_000,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const support = await semanticNodeRepository.insert({
      id: "semn_bbbbbbbbbbbbbbbb" as never,
      kind: "proposition",
      label: "Atlas deploys are steadier with rollback plans",
      description: "Rollback plans support steadier Atlas deploys.",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 10_000,
      updated_at: 10_000,
      last_verified_at: 10_000,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: support.id,
      relation: "supports",
      confidence: 0.8,
      evidence_episode_ids: [episode.id],
      created_at: 10_000,
      last_verified_at: 10_000,
    });
    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      semanticNodeRepository,
      semanticGraph,
      dataDir: tempDir,
      clock,
    });

    const result = await pipeline.searchWithContextForDisclosure("Atlas", {
      crossAudience: true,
      graphWalkDepth: 1,
      limit: 1,
    });

    expect(result.episodes).toEqual([]);
    expect(result.semantic.matched_nodes).toEqual(
      expect.arrayContaining([expect.objectContaining({ id: atlas.id })]),
    );
    expect(result.semantic.support_hits).toHaveLength(1);
    expect(result.confidence.overall).toBeGreaterThan(0);
  });

  it("threads semantic as-of through graph retrieval and confidence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        semanticMigrations,
      ),
    });
    const episodeTable = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const semanticTable = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new ManualClock(1_000_000);
    const repo = new EpisodicRepository({
      table: episodeTable,
      db,
      clock,
    });
    const semanticNodeRepository = new SemanticNodeRepository({
      table: semanticTable,
      db,
      clock,
    });
    const semanticEdgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const semanticGraph = new SemanticGraph({
      nodeRepository: semanticNodeRepository,
      edgeRepository: semanticEdgeRepository,
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

    const entry = await writer.append({
      kind: "user_msg",
      content: "Atlas deploy note",
    });

    await repo.createEpisode(createEpisode("ep_aaaaaaaaaaaaaaaa", entry.id, [1, 0, 0, 0]));
    const atlas = await semanticNodeRepository.insert({
      id: "semn_aaaaaaaaaaaaaaaa" as never,
      kind: "entity",
      label: "Atlas",
      description: "Atlas entity",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1_000_000,
      updated_at: 1_000_000,
      last_verified_at: 1_000_000,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const contradiction = await semanticNodeRepository.insert({
      id: "semn_bbbbbbbbbbbbbbbb" as never,
      kind: "proposition",
      label: "Atlas needs no deployment work",
      description: "A stale claim that Atlas deployment needs no action.",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1_000_000,
      updated_at: 1_000_000,
      last_verified_at: 1_000_000,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const edge = semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: contradiction.id,
      relation: "contradicts",
      confidence: 0.7,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1_000_000,
      last_verified_at: 1_000_000,
    });
    semanticEdgeRepository.invalidateEdge(edge.id, {
      at: 1_000_500,
      by_process: "manual",
    });
    clock.set(1_001_000);

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      semanticNodeRepository,
      semanticGraph,
      dataDir: tempDir,
      clock,
    });

    const current = await pipeline.searchWithContextForDisclosure("Atlas", {
      limit: 1,
      graphWalkDepth: 1,
      maxGraphNodes: 4,
    });
    const historical = await pipeline.searchWithContextForDisclosure("Atlas", {
      limit: 1,
      graphWalkDepth: 1,
      maxGraphNodes: 4,
      asOf: 1_000_250,
    });

    expect(current.contradiction_present).toBe(false);
    expect(current.semantic.contradiction_hits).toEqual([]);
    expect(current.confidence.contradictionPresent).toBe(false);
    expect(historical.semantic.as_of).toBe(1_000_250);
    expect(historical.contradiction_present).toBe(true);
    expect(historical.semantic.contradiction_hits[0]?.edgePath[0]?.id).toBe(edge.id);
    expect(historical.confidence.contradictionPresent).toBe(true);
    expect(historical.contradictionRouting.contradictions).toEqual([
      expect.objectContaining({
        edgeId: edge.id,
        nodeIds: [atlas.id, contradiction.id].sort(),
        sourceEpisodeIds: ["ep_aaaaaaaaaaaaaaaa"],
        validUntil: 1_000_500,
        sessionScope: "unknown",
        linkedOpenQuestionIds: [],
        fingerprint: createHash("sha1")
          .update(
            [`edge:${edge.id}`, `node:${atlas.id}`, `node:${contradiction.id}`].sort().join("|"),
          )
          .digest("hex"),
      }),
    ]);
  });

  it("attaches relevant open questions when requested", async () => {
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
    const openQuestionsTable = await store.openTable({
      name: "open_questions",
      schema: createOpenQuestionsTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock: new FixedClock(5_000),
    });
    const embeddingClient = new ScriptedEmbeddingClient();
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      table: openQuestionsTable,
      embeddingClient,
      clock: new FixedClock(5_000),
    });
    const alice = createEntityId();
    const bob = createEntityId();

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await repo.createEpisode({
      ...createEpisode("ep_aaaaaaaaaaaaaaaa", "strm_aaaaaaaaaaaaaaaa", [1, 0, 0, 0]),
      title: "Atlas deployment note",
    });
    openQuestionsRepository.add({
      question: "Why does Atlas deployment keep failing?",
      urgency: 0.8,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    openQuestionsRepository.add({
      question: "What snacks should we order?",
      urgency: 0.9,
      source: "user",
      provenance: { kind: "manual" },
    });
    openQuestionsRepository.add({
      question: "Why does Atlas deployment keep failing for Alice?",
      urgency: 1,
      audience_entity_id: alice,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    await openQuestionsRepository.waitForPendingEmbeddings();

    const pipeline = new RetrievalPipeline({
      embeddingClient,
      episodicRepository: repo,
      openQuestionsRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const reflective = await pipeline.searchWithContextForDisclosure("Atlas deployment", {
      limit: 1,
      includeOpenQuestions: true,
      audienceEntityId: bob,
    });
    const defaultResult = await pipeline.searchWithContextForDisclosure("Atlas deployment", {
      limit: 1,
    });

    expect(reflective.open_questions.map((question) => question.question)).toEqual(
      expect.arrayContaining([
        "Why does Atlas deployment keep failing?",
        "Why does Atlas deployment keep failing for Alice?",
      ]),
    );
    expect(reflective.open_questions.map((question) => question.question)).not.toContain(
      "What snacks should we order?",
    );
    expect(defaultResult.open_questions).toEqual([]);
  });

  it("recalls audience-scoped open questions globally for cognition", async () => {
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
    const openQuestionsTable = await store.openTable({
      name: "open_questions",
      schema: createOpenQuestionsTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock: new FixedClock(5_000),
    });
    const embeddingClient = new ScriptedEmbeddingClient();
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      table: openQuestionsTable,
      embeddingClient,
      clock: new FixedClock(5_000),
    });
    const group = createEntityId();
    const alice = createEntityId();

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await repo.createEpisode({
      ...createEpisode("ep_groupvisibility0", "strm_groupvisible0000" as never, [1, 0, 0, 0]),
      title: "Atlas deployment note",
    });
    openQuestionsRepository.add({
      question: "Why does Atlas deployment keep failing for everyone?",
      urgency: 0.8,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    openQuestionsRepository.add({
      question: "Why does Atlas deployment keep failing in the group?",
      urgency: 0.9,
      audience_entity_id: group,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    openQuestionsRepository.add({
      question: "Por que falla Atlas deployment para Alicia en privado?",
      urgency: 1,
      audience_entity_id: alice,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    await openQuestionsRepository.waitForPendingEmbeddings();

    const pipeline = new RetrievalPipeline({
      embeddingClient,
      episodicRepository: repo,
      openQuestionsRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const result = await pipeline.searchWithContextForDisclosure("Atlas deployment", {
      limit: 1,
      includeOpenQuestions: true,
      audienceEntityId: group,
      openQuestionsLimit: 10,
    });
    const questions = result.open_questions.map((question) => question.question);
    const aliceEvidence = result.evidence.find(
      (item) => item.source === "open_question" && item.text.includes("Alicia en privado"),
    );

    expect(questions).toEqual(
      expect.arrayContaining([
        "Why does Atlas deployment keep failing for everyone?",
        "Why does Atlas deployment keep failing in the group?",
        "Por que falla Atlas deployment para Alicia en privado?",
      ]),
    );
    expect(aliceEvidence?.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [alice],
      privateToEntityIds: [alice],
      publicToEntityIds: [],
    });
  });
});
