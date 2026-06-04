import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { LLMError } from "../../util/errors.js";
import {
  createEntityId,
  createRelationalSlotId,
  createSemanticNodeId,
  createStreamEntryId,
  type EpisodeId,
} from "../../util/ids.js";
import type { ParticipantRoster } from "../../cognition/perception/index.js";
import type { RelationshipClaim } from "../../cognition/relationship-claims.js";
import type { TurnTracer } from "../../cognition/tracing/tracer.js";
import type { Episode } from "../episodic/types.js";
import { SemanticExtractor } from "./extractor.js";
import { semanticMigrations } from "./migrations.js";
import {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
} from "./repository.js";

const SEMANTIC_TOOL_NAME = "EmitSemanticCandidates";

function createSemanticToolResponse(input: { nodes: unknown[]; edges: unknown[] }) {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: SEMANTIC_TOOL_NAME,
        input,
      },
    ],
  };
}

function relationshipClaim(overrides: Partial<RelationshipClaim> = {}): RelationshipClaim {
  return {
    label_family: "kinship",
    subject_entity_id: null,
    object_entity_id: null,
    object_text: "relationship participant",
    requires_grounding: true,
    evidence_relational_slot_ids: [],
    evidence_stream_entry_ids: [],
    ...overrides,
  };
}

class SemanticEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    if (/atlas/i.test(text)) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    return Float32Array.from([0, 1, 0, 0]);
  }
}

function buildEpisode(id: Episode["id"], title: string, overrides: Partial<Episode> = {}): Episode {
  return {
    id,
    title,
    narrative: overrides.narrative ?? `${title} narrative.`,
    participants: overrides.participants ?? ["team"],
    location: overrides.location ?? null,
    start_time: overrides.start_time ?? 1,
    end_time: overrides.end_time ?? 2,
    source_stream_ids: overrides.source_stream_ids ?? [
      "strm_aaaaaaaaaaaaaaaa" as Episode["source_stream_ids"][number],
    ],
    significance: overrides.significance ?? 0.8,
    tags: overrides.tags ?? ["atlas"],
    confidence: overrides.confidence ?? 0.8,
    lineage: overrides.lineage ?? {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: overrides.emotional_arc ?? null,
    audience_entity_id: overrides.audience_entity_id,
    shared: overrides.shared,
    embedding: overrides.embedding ?? Float32Array.from([1, 0, 0, 0]),
    created_at: overrides.created_at ?? 1,
    updated_at: overrides.updated_at ?? 1,
  };
}

function createEpisodeLookup(episodes: readonly Episode[]) {
  const episodeById = new Map(episodes.map((episode) => [episode.id, episode]));

  return {
    getMany: async (ids: readonly Episode["id"][]) =>
      ids
        .map((id) => episodeById.get(id))
        .filter((episode): episode is Episode => episode !== undefined),
  };
}

async function createSemanticRepositories(cleanup: Array<() => Promise<void>>) {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
  const store = new LanceDbStore({
    uri: join(tempDir, "lancedb"),
  });
  const db = openDatabase(join(tempDir, "borg.db"), {
    migrations: semanticMigrations,
  });
  const table = await store.openTable({
    name: "semantic_nodes",
    schema: createSemanticNodesTableSchema(4),
  });
  const clock = new FixedClock(1_000);
  const nodeRepository = new SemanticNodeRepository({
    table,
    db,
    clock,
  });
  const edgeRepository = new SemanticEdgeRepository({
    db,
    clock,
  });

  cleanup.push(async () => {
    db.close();
    await store.close();
    rmSync(tempDir, { recursive: true, force: true });
  });

  return {
    nodeRepository,
    edgeRepository,
    clock,
  };
}

describe("semantic extractor", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("routes relationship-claim grounding through the promoted shared gate", () => {
    const semanticSource = readFileSync(new URL("./extractor.ts", import.meta.url), "utf8");
    const sharedStateSource = readFileSync(
      new URL("../../cognition/shared-state/patch-validation.ts", import.meta.url),
      "utf8",
    );

    expect(semanticSource).toContain("checkRelationshipClaimGroundingAsync");
    expect(sharedStateSource).toContain("checkRelationshipClaimGrounding");
    expect(semanticSource).toContain("relationship_claims");
  });

  it("includes event-vs-state guidance in the extraction prompt", async () => {
    const privateAudience = createEntityId();
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Madrid travel note", {
      narrative: "The user described arrival and return dates for a Madrid visit.",
      audience_entity_id: privateAudience,
      shared: false,
    });
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [],
          edges: [],
        }),
      ],
    });
    const extractor = new SemanticExtractor({
      nodeRepository: {
        listDistinctKinds: () => ["process_explanation"],
      } as unknown as SemanticNodeRepository,
      edgeRepository: {} as SemanticEdgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: llm,
      model: "haiku",
      participantRoster: {
        participants: [
          {
            entity_id:
              "ent_noraaaaaaaaaaaa" as ParticipantRoster["participants"][number]["entity_id"],
            display_name: "Nora",
            known_relationships: ["spouse:Priya"],
            audience_role: "speaker",
            relationship_source: "relational_slot:rslot_grounded",
          },
        ],
        non_chat_subjects: [],
        unknown_or_uncertain: [],
      },
    });

    await extractor.extractFromEpisodes([episode]);

    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(prompt).toContain("Distinguish temporally bounded events");
    expect(prompt).toContain("do not collapse event-scoped language");
    expect(prompt).toContain("prefer the narrower event-scoped interpretation");
    expect(prompt).toContain("observation_metadata");
    expect(prompt).toContain("Distinct witness, timeframe/date, count_or_intensity");
    expect(prompt).toContain("Memory-write relationship claim grounding");
    expect(prompt).toContain("Headcount and set grounding");
    expect(prompt).toContain("relationship_claims");
    expect(prompt).toContain("evidence_relational_slot_ids");
    expect(prompt).toContain("evidence_stream_entry_ids");
    expect(prompt).toContain("Known semantic node kinds already in graph: process_explanation.");
    expect(prompt).toContain(
      "coin a new lowercase_slug kind only for a genuinely new information shape",
    );
    expect(prompt).toContain("Thread roster:");
    expect(prompt).toContain("relational_slot:rslot_grounded");
    expect(prompt).toContain("disclosure_class=relationship_private");
    expect(prompt).toContain("usable internally");
    expect(prompt).toContain(privateAudience);
  });

  it("traces and skips semantic nodes with ungrounded relationship claims", async () => {
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Birthday lunch", {
      narrative: "The user discussed birthday lunch attendance.",
    });
    const reviewEnqueue = vi.fn();
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
    const extractor = new SemanticExtractor({
      nodeRepository: {
        listDistinctKinds: () => [],
        findByExactLabelOrAlias: async () => [],
      } as unknown as SemanticNodeRepository,
      edgeRepository: {} as SemanticEdgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "proposition",
                label: "Birthday lunch siblings",
                description: "The four siblings plus Mom and Dad are attending lunch.",
                aliases: [],
                confidence: 0.7,
                relationship_claims: [
                  relationshipClaim({ object_text: "les membres de la famille" }),
                ],
                source_episode_ids: [episode.id],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      reviewEnqueue,
      tracer,
      participantRoster: {
        participants: [],
        non_chat_subjects: [],
        unknown_or_uncertain: [],
      },
      traceTurnId: "turn_ungrounded_relationship",
    });

    const result = await extractor.extractFromEpisodes([episode]);

    expect(result).toMatchObject({
      insertedNodes: 0,
      updatedNodes: 0,
      skippedNodes: 1,
      insertedEdges: 0,
      skippedEdges: 0,
    });
    expect(reviewEnqueue).not.toHaveBeenCalled();
    expect(tracer.emit).toHaveBeenCalledWith("semantic_insert.skipped", {
      turnId: "turn_ungrounded_relationship",
      kind: "node",
      reason: "relationship_claim_ungrounded",
      relationship_claim_label_families: ["kinship"],
      relationship_claims: [expect.objectContaining({ object_text: "les membres de la famille" })],
      ungrounded_relationship_claims: [
        expect.objectContaining({ object_text: "les membres de la famille" }),
      ],
    });
  });

  it("rejects semantic relationship claims grounded only by uncertain roster slots", async () => {
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Birthday lunch", {
      narrative: "The user discussed birthday lunch attendance.",
    });
    const contestedSlotId = createRelationalSlotId();
    const reviewEnqueue = vi.fn();
    const extractor = new SemanticExtractor({
      nodeRepository: {
        listDistinctKinds: () => [],
        findByExactLabelOrAlias: async () => [],
      } as unknown as SemanticNodeRepository,
      edgeRepository: {} as SemanticEdgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "proposition",
                label: "Birthday lunch siblings",
                description: "The siblings are attending lunch.",
                aliases: [],
                confidence: 0.7,
                relationship_claims: [
                  relationshipClaim({
                    evidence_relational_slot_ids: [contestedSlotId],
                  }),
                ],
                source_episode_ids: [episode.id],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      reviewEnqueue,
      participantRoster: {
        participants: [],
        non_chat_subjects: [],
        unknown_or_uncertain: [
          {
            entity_id: null,
            display_name: "uncertain family group",
            known_relationships: ["sibling:uncertain"],
            reason: "relational_slot_state:contested",
            relationship_source: `relational_slot:${contestedSlotId}`,
            relationship_sources: [`relational_slot:${contestedSlotId}`],
          },
        ],
      },
      traceTurnId: "turn_uncertain_relationship",
    });

    const result = await extractor.extractFromEpisodes([episode]);

    expect(result).toMatchObject({
      insertedNodes: 0,
      skippedNodes: 1,
    });
    expect(reviewEnqueue).not.toHaveBeenCalled();
  });

  it("inserts semantic relationship claims with grounded participant relational slot evidence", async () => {
    const { nodeRepository, edgeRepository, clock } = await createSemanticRepositories(cleanup);
    const slotId = createRelationalSlotId();
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Sibling planning", {
      narrative: "The user stated the sibling relationship was established in the roster.",
    });
    const reviewEnqueue = vi.fn();
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "proposition",
                label: "Sibling planning",
                description: "Nora's sibling relationship is relevant to the planning thread.",
                aliases: [],
                confidence: 0.7,
                relationship_claims: [
                  relationshipClaim({
                    evidence_relational_slot_ids: [slotId],
                  }),
                ],
                source_episode_ids: [episode.id],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      reviewEnqueue,
      participantRoster: {
        participants: [
          {
            entity_id:
              "ent_noraaaaaaaaaaaa" as ParticipantRoster["participants"][number]["entity_id"],
            display_name: "Nora",
            known_relationships: ["sibling:Julian"],
            audience_role: "speaker",
            relationship_source: `relational_slot:${slotId}`,
            relationship_sources: [`relational_slot:${slotId}`],
          },
        ],
        non_chat_subjects: [],
        unknown_or_uncertain: [],
      },
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);
    const nodes = await nodeRepository.list();

    expect(result).toMatchObject({
      insertedNodes: 1,
      skippedNodes: 0,
    });
    expect(nodes.map((node) => node.label)).toContain("Sibling planning");
    expect(reviewEnqueue).not.toHaveBeenCalled();
  });

  it("inserts semantic relationship claims with direct trusted user stream evidence", async () => {
    const { nodeRepository, edgeRepository, clock } = await createSemanticRepositories(cleanup);
    const userStreamId = "strm_userdirect000001" as Episode["source_stream_ids"][number];
    const episode = buildEpisode("ep_userdirect000001" as Episode["id"], "Direct sibling note", {
      narrative: "The user directly stated Nora and Julian are siblings.",
      source_stream_ids: [userStreamId],
    });
    const reviewEnqueue = vi.fn();
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "proposition",
                label: "Nora and Julian siblings",
                description: "Nora and Julian are siblings.",
                aliases: [],
                confidence: 0.7,
                relationship_claims: [
                  relationshipClaim({
                    evidence_stream_entry_ids: [userStreamId],
                  }),
                ],
                source_episode_ids: [episode.id],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      reviewEnqueue,
      relationshipEvidenceStreamEntryTrust: async (streamEntryId) => ({
        allowed: streamEntryId === userStreamId,
      }),
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);
    const nodes = await nodeRepository.list();

    expect(result).toMatchObject({
      insertedNodes: 1,
      skippedNodes: 0,
    });
    expect(nodes.map((node) => node.label)).toContain("Nora and Julian siblings");
    expect(reviewEnqueue).not.toHaveBeenCalled();
  });

  it("rejects semantic relationship claims grounded only by assistant output under review", async () => {
    const { nodeRepository, edgeRepository, clock } = await createSemanticRepositories(cleanup);
    const assistantStreamId = "strm_assistant0000001" as Episode["source_stream_ids"][number];
    const episode = buildEpisode("ep_assistant0000001" as Episode["id"], "Assistant sibling note", {
      narrative: "The assistant output under review stated Nora and Julian are siblings.",
      source_stream_ids: [assistantStreamId],
    });
    const reviewEnqueue = vi.fn();
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "proposition",
                label: "Assistant sibling claim",
                description: "Nora and Julian are siblings.",
                aliases: [],
                confidence: 0.7,
                relationship_claims: [
                  relationshipClaim({
                    evidence_stream_entry_ids: [assistantStreamId],
                  }),
                ],
                source_episode_ids: [episode.id],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      reviewEnqueue,
      relationshipEvidenceStreamEntryTrust: async () => ({
        allowed: false,
        reason: "not_user_msg",
      }),
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);

    expect(result).toMatchObject({
      insertedNodes: 0,
      skippedNodes: 1,
    });
    expect(reviewEnqueue).not.toHaveBeenCalled();
  });

  it("rejects semantic relationship claims with stream evidence outside the source bundle", async () => {
    const { nodeRepository, edgeRepository, clock } = await createSemanticRepositories(cleanup);
    const sourceStreamId = "strm_srcbundle0000001" as Episode["source_stream_ids"][number];
    const citedOutsideBundleId = "strm_notbundle0000001" as Episode["source_stream_ids"][number];
    const episode = buildEpisode("ep_srcbundle0000001" as Episode["id"], "Source-bundled note", {
      narrative: "The user directly stated Nora and Julian are siblings.",
      source_stream_ids: [sourceStreamId],
    });
    const reviewEnqueue = vi.fn();
    const trust = vi.fn(async () => ({
      allowed: true,
    }));
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "proposition",
                label: "Outside-bundle sibling claim",
                description: "Nora and Julian are siblings.",
                aliases: [],
                confidence: 0.7,
                relationship_claims: [
                  relationshipClaim({
                    evidence_stream_entry_ids: [citedOutsideBundleId],
                  }),
                ],
                source_episode_ids: [episode.id],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      reviewEnqueue,
      relationshipEvidenceStreamEntryTrust: trust,
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);

    expect(result).toMatchObject({
      insertedNodes: 0,
      skippedNodes: 1,
    });
    expect(trust).not.toHaveBeenCalled();
    expect(reviewEnqueue).not.toHaveBeenCalled();
  });

  it("inserts semantic nodes without relationship claims, including service context nouns", async () => {
    const { nodeRepository, edgeRepository, clock } = await createSemanticRepositories(cleanup);
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Appointment logistics");
    const reviewEnqueue = vi.fn();
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "concept",
                label: "Doctor appointment",
                description: "Patient portal paperwork is pending.",
                aliases: [],
                confidence: 0.6,
                source_episode_ids: [episode.id],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      reviewEnqueue,
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);
    const nodes = await nodeRepository.list();

    expect(result).toMatchObject({
      insertedNodes: 1,
      skippedNodes: 0,
    });
    expect(nodes.map((node) => node.label)).toContain("Doctor appointment");
    expect(reviewEnqueue).not.toHaveBeenCalled();
  });

  it("extracts nodes and edges, rejects hallucinated refs, and merges duplicates", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Atlas",
      description: "Atlas existing node",
      domain: "tech",
      aliases: ["Project Atlas"],
      confidence: 0.6,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });

    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "entity",
              label: "Atlas",
              description: "Atlas updated node",
              domain: " Technology ",
              aliases: ["Atlas service"],
              confidence: 0.7,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
            {
              kind: "concept",
              label: "Rollback",
              description: "Rollback plan",
              aliases: [],
              confidence: 0.6,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
          ],
          edges: [
            {
              from_label: "Atlas",
              to_label: "Rollback",
              relation: "supports",
              confidence: 0.6,
              evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
          ],
        }),
      ],
    });
    const semanticReviewService = {
      queueDuplicateReview: vi.fn(),
    };
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([
        buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
      ]),
      llmClient: llm,
      model: "haiku",
      semanticReviewService,
      clock,
    });
    const nodeInsertSpy = vi.spyOn(nodeRepository, "insert");
    const edgeAddSpy = vi.spyOn(edgeRepository, "addEdge");

    const result = await extractor.extractFromEpisodes([
      buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
    ]);

    expect(result).toEqual({
      insertedNodes: 1,
      updatedNodes: 1,
      skippedNodes: 0,
      insertedEdges: 1,
      updatedEdges: 0,
      skippedEdges: 0,
    });
    const nodesAfterMerge = await nodeRepository.list();

    expect(nodesAfterMerge.map((node) => node.label)).toEqual(
      expect.arrayContaining(["Atlas", "Rollback"]),
    );
    expect(nodesAfterMerge.find((node) => node.label === "Atlas")?.domain).toBe("tech");
    expect(edgeRepository.listEdges()).toHaveLength(1);
    expect(semanticReviewService.queueDuplicateReview).toHaveBeenCalledWith(
      expect.objectContaining({ label: "Rollback" }),
    );
    expect(nodeInsertSpy).toHaveBeenCalled();
    expect(edgeAddSpy).toHaveBeenCalled();
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: SEMANTIC_TOOL_NAME,
    });
    expect(Math.max(...nodeInsertSpy.mock.invocationCallOrder)).toBeLessThan(
      edgeAddSpy.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER,
    );

    const hallucinatingExtractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([
        buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
      ]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "concept",
                label: "Bad node",
                description: "Bad node",
                aliases: [],
                confidence: 0.6,
                source_episode_ids: ["ep_missing"],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    await expect(
      hallucinatingExtractor.extractFromEpisodes([
        buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
      ]),
    ).rejects.toThrow("unknown source_episode_ids");
  });

  it("creates edges between existing nodes even when the batch omits node candidates", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Alice and Bob");

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const alice = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Alice",
      description: "Alice existing node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const bob = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Bob",
      description: "Bob existing node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [],
            edges: [
              {
                from_label: "Alice",
                to_label: "Bob",
                relation: "related_to",
                confidence: 0.8,
                evidence_episode_ids: [episode.id],
              },
            ],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);
    const [edge] = edgeRepository.listEdges();

    expect(result.insertedEdges).toBe(1);
    expect(edge).toMatchObject({
      from_node_id: alice.id,
      to_node_id: bob.id,
      relation: "related_to",
    });
  });

  it("inserts vector-only semantic matches separately and queues duplicate review metadata", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas vector note");

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const existing = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Atlas platform",
      description: "Atlas existing node",
      domain: "tech",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const reviewEnqueue = vi.fn();
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "entity",
                label: "Deployment platform",
                description: "Atlas service used for deployments.",
                domain: "tech",
                aliases: [],
                confidence: 0.65,
                source_episode_ids: [episode.id],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      reviewEnqueue,
      clock,
      traceTurnId: "turn_vector_only",
    });

    const result = await extractor.extractFromEpisodes([episode]);
    const nodes = await nodeRepository.list();
    const inserted = nodes.find((node) => node.label === "Deployment platform");

    expect(result).toMatchObject({
      insertedNodes: 1,
      updatedNodes: 0,
    });
    expect(nodes).toHaveLength(2);
    expect(inserted).toBeDefined();
    expect(await nodeRepository.get(existing.id)).toMatchObject({
      label: "Atlas platform",
      source_episode_ids: [episode.id],
    });
    expect(reviewEnqueue).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "duplicate",
        sourceProcess: "semantic-extractor",
        traceTurnId: "turn_vector_only",
        refs: expect.objectContaining({
          node_ids: [inserted?.id, existing.id],
          node_labels: ["Deployment platform", "Atlas platform"],
          duplicate_subtype: "vector_only_merge_candidate",
          vector_similarity: expect.closeTo(1, 5),
          source_overlap: {
            candidate_source_episode_ids: [episode.id],
            matched_source_episode_ids: [episode.id],
            overlapping_source_episode_ids: [episode.id],
            overlap_count: 1,
          },
        }),
      }),
    );
  });

  it("skips malformed and dangling edges while preserving valid candidates", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas partial edges");

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "entity",
                label: "Atlas",
                description: "Atlas service.",
                domain: "tech",
                aliases: [],
                confidence: 0.7,
                source_episode_ids: [episode.id],
              },
              {
                kind: "concept",
                label: "Rollback",
                description: "Rollback plan.",
                domain: "process",
                aliases: [],
                confidence: 0.6,
                source_episode_ids: [episode.id],
              },
            ],
            edges: [
              {
                from_label: "Atlas",
                to_label: "Rollback",
                relation: "supports",
                confidence: 0.6,
                evidence_episode_ids: [episode.id],
              },
              {
                from_label: "Atlas",
                to_label: "Rollback",
                confidence: 0.6,
                evidence_episode_ids: [episode.id],
              },
              {
                from_label: "Atlas",
                to_label: "Missing concept",
                relation: "supports",
                confidence: 0.6,
                evidence_episode_ids: [episode.id],
              },
            ],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);

    expect(result).toMatchObject({
      insertedNodes: 2,
      insertedEdges: 1,
      skippedEdges: 2,
    });
    expect(edgeRepository.listEdges()).toHaveLength(1);
  });

  it("merges duplicate node and edge evidence across repeated cross-scope extraction", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const privateAudience = createEntityId();
    const firstEpisode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas public note", {
      audience_entity_id: null,
      shared: true,
    });
    const secondEpisode = buildEpisode(
      "ep_bbbbbbbbbbbbbbbb" as Episode["id"],
      "Atlas private note",
      {
        audience_entity_id: privateAudience,
        shared: false,
      },
    );
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "entity",
              label: "Atlas",
              description: "Atlas service.",
              domain: "tech",
              aliases: [],
              confidence: 0.7,
              source_episode_ids: [firstEpisode.id],
            },
            {
              kind: "concept",
              label: "Rollback",
              description: "Rollback plan.",
              domain: "process",
              aliases: [],
              confidence: 0.6,
              source_episode_ids: [firstEpisode.id],
            },
          ],
          edges: [
            {
              from_label: "Atlas",
              to_label: "Rollback",
              relation: "supports",
              confidence: 0.6,
              evidence_episode_ids: [firstEpisode.id],
            },
          ],
        }),
        createSemanticToolResponse({
          nodes: [
            {
              kind: "entity",
              label: "Atlas",
              description: "Atlas service with new evidence.",
              domain: "tech",
              aliases: [],
              confidence: 0.7,
              source_episode_ids: [secondEpisode.id],
            },
            {
              kind: "concept",
              label: "Rollback",
              description: "Rollback plan with new evidence.",
              domain: "process",
              aliases: [],
              confidence: 0.6,
              source_episode_ids: [secondEpisode.id],
            },
          ],
          edges: [
            {
              from_label: "Atlas",
              to_label: "Rollback",
              relation: "supports",
              confidence: 0.65,
              evidence_episode_ids: [secondEpisode.id],
            },
          ],
        }),
      ],
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([firstEpisode, secondEpisode]),
      llmClient: llm,
      model: "haiku",
      clock,
    });

    await extractor.extractFromEpisodes([firstEpisode]);
    const result = await extractor.extractFromEpisodes([secondEpisode]);
    const [edge] = edgeRepository.listEdges({ includeInvalid: true });
    const atlas = (
      await nodeRepository.findByExactLabelOrAlias("Atlas", 1, {
        includeArchived: true,
      })
    )[0];

    expect(result).toMatchObject({
      insertedEdges: 0,
      updatedEdges: 1,
      skippedEdges: 0,
    });
    expect(edge?.evidence_episode_ids).toEqual([firstEpisode.id, secondEpisode.id]);
    expect(atlas?.source_episode_ids).toEqual([firstEpisode.id, secondEpisode.id]);
  });

  it("sets edge valid_from from an explicit temporal relation hint", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(Date.UTC(2026, 4, 1));
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas temporal note", {
      narrative: "Atlas has depended on rollback drills since 2026-03-01.",
      start_time: Date.UTC(2026, 2, 10),
      end_time: Date.UTC(2026, 2, 10, 1),
      created_at: Date.UTC(2026, 2, 10),
      updated_at: Date.UTC(2026, 2, 10),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "entity",
                label: "Atlas",
                description: "Atlas service.",
                domain: "tech",
                aliases: [],
                confidence: 0.7,
                source_episode_ids: [episode.id],
              },
              {
                kind: "concept",
                label: "Rollback drills",
                description: "Rollback drill practice.",
                domain: "process",
                aliases: [],
                confidence: 0.7,
                source_episode_ids: [episode.id],
              },
            ],
            edges: [
              {
                from_label: "Atlas",
                to_label: "Rollback drills",
                relation: "related_to",
                confidence: 0.7,
                evidence_episode_ids: [episode.id],
                valid_from_ts: Date.UTC(2026, 2, 1),
              },
            ],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);
    const [edge] = edgeRepository.listEdges({
      includeInvalid: true,
    });

    expect(result.insertedEdges).toBe(1);
    expect(edge).toMatchObject({
      valid_from: Date.UTC(2026, 2, 1),
      valid_to: null,
    });
  });

  it("keeps default edge validity when no temporal relation hint is present", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas plain note", {
      narrative: "Atlas depends on rollback drills.",
      start_time: 500,
      end_time: 600,
      created_at: 500,
      updated_at: 600,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "entity",
                label: "Atlas",
                description: "Atlas service.",
                domain: "tech",
                aliases: [],
                confidence: 0.7,
                source_episode_ids: [episode.id],
              },
              {
                kind: "concept",
                label: "Rollback drills",
                description: "Rollback drill practice.",
                domain: "process",
                aliases: [],
                confidence: 0.7,
                source_episode_ids: [episode.id],
              },
            ],
            edges: [
              {
                from_label: "Atlas",
                to_label: "Rollback drills",
                relation: "related_to",
                confidence: 0.7,
                evidence_episode_ids: [episode.id],
              },
            ],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);
    const [edge] = edgeRepository.listEdges({
      includeInvalid: true,
    });

    expect(result.insertedEdges).toBe(1);
    expect(edge?.valid_from).toBe(1_000);
    expect(edge?.valid_to).toBeNull();
  });

  it("does not resolve label-only edges to archived nodes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Archived Alice");

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Alice",
      description: "Archived Alice node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: true,
      superseded_by: null,
    });
    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Bob",
      description: "Bob existing node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [],
            edges: [
              {
                from_label: "Alice",
                to_label: "Bob",
                relation: "related_to",
                confidence: 0.8,
                evidence_episode_ids: [episode.id],
              },
            ],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);

    expect(result).toMatchObject({
      insertedEdges: 0,
      skippedEdges: 1,
    });
    expect(edgeRepository.listEdges()).toHaveLength(0);
  });

  it("resolves label-only edges across audience scopes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const bobAudience = createEntityId();
    const publicEpisode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Public Alice");
    const privateEpisode = buildEpisode("ep_bbbbbbbbbbbbbbbb" as Episode["id"], "Private Alice", {
      audience_entity_id: bobAudience,
      shared: false,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Alice",
      description: "Private Alice node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [privateEpisode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Bob",
      description: "Public Bob node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [publicEpisode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([publicEpisode, privateEpisode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [],
            edges: [
              {
                from_label: "Alice",
                to_label: "Bob",
                relation: "related_to",
                confidence: 0.8,
                evidence_episode_ids: [publicEpisode.id],
              },
            ],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    const result = await extractor.extractFromEpisodes([publicEpisode]);

    expect(result).toMatchObject({
      insertedEdges: 1,
      skippedEdges: 0,
    });
    expect(edgeRepository.listEdges()).toHaveLength(1);
  });

  it("skips label-only edges when an existing endpoint label is ambiguous", async () => {
    const { nodeRepository, edgeRepository, clock } = await createSemanticRepositories(cleanup);
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Ambiguous Alice");
    const firstAliceId = createSemanticNodeId();
    const secondAliceId = createSemanticNodeId();

    await nodeRepository.insert({
      id: firstAliceId,
      kind: "entity",
      label: "Alice",
      description: "Alice from the public project context.",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    await nodeRepository.insert({
      id: secondAliceId,
      kind: "entity",
      label: "Alice",
      description: "Alice from a separate private context.",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 2,
      updated_at: 2,
      last_verified_at: 2,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Bob",
      description: "Bob existing node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 0, 1, 0]),
      archived: false,
      superseded_by: null,
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [],
            edges: [
              {
                from_label: "Alice",
                to_label: "Bob",
                relation: "related_to",
                confidence: 0.8,
                evidence_episode_ids: [episode.id],
              },
            ],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);

    expect(result).toMatchObject({
      insertedEdges: 0,
      skippedEdges: 1,
    });
    expect(edgeRepository.listEdges()).toHaveLength(0);
  });

  it("re-embeds updated nodes from the final stored text", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const existing = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Atlas",
      description: "Atlas existing node",
      domain: "tech",
      aliases: ["Project Atlas"],
      confidence: 0.8,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const embed = vi.fn(async (_text: string) => Float32Array.from([1, 0, 0, 0]));
    const embeddingClient: EmbeddingClient = {
      embed,
      embedBatch: async (texts) => Promise.all(texts.map((text) => embed(text))),
    };
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient,
      episodicRepository: createEpisodeLookup([
        buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
      ]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "entity",
                label: "Atlas",
                description: "Atlas newer description that should not win",
                domain: "tech",
                aliases: ["Atlas service"],
                confidence: 0.4,
                source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    await extractor.extractFromEpisodes([
      buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
    ]);
    const updated = await nodeRepository.get(existing.id);

    expect(updated?.description).toBe("Atlas existing node");
    expect(updated?.aliases).toEqual(["Project Atlas", "Atlas", "Atlas service"]);
    expect(embed.mock.calls.at(-1)?.[0]).toBe(
      "Atlas\nAtlas existing node\nProject Atlas Atlas Atlas service",
    );
  });

  it("rejects malformed string-wrapped tool payloads", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident");
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: SEMANTIC_TOOL_NAME,
              input: {
                nodes:
                  '[{"kind":"entity","label":"Atlas","description":"Atlas node","aliases":[],"confidence":0.7,"source_episode_ids":["ep_aaaaaaaaaaaaaaaa"]},{"kind":"concept","label":"Rollback","description":"Rollback concept","aliases":[],"confidence":0.6,"source_episode_ids":["ep_aaaaaaaaaaaaaaaa"]}]<parameter name="edges">[{"from_label":"Atlas","to_label":"Rollback","relation":"related_to","confidence":0.5,"evidence_episode_ids":["ep_aaaaaaaaaaaaaaaa"]}]',
              },
            },
          ],
        },
      ],
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: llm,
      model: "haiku",
      clock,
    });

    let error: unknown;

    try {
      await extractor.extractFromEpisodes([episode]);
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(LLMError);
    expect(error).toMatchObject({
      code: "SEMANTIC_EXTRACTOR_INVALID",
    });
    expect(edgeRepository.listEdges()).toHaveLength(0);
  });

  it("keeps same-label distinct concepts separate by kind, not access scope", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const episodeA = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Tomasz conversation", {
      narrative: "Tomasz joined the call and asked about plans.",
      participants: ["Alice", "Tomasz"],
      audience_entity_id: "ent_aaaaaaaaaaaaaaaa" as Episode["audience_entity_id"],
      shared: false,
      tags: ["people"],
    });
    const episodeB = buildEpisode("ep_bbbbbbbbbbbbbbbb" as Episode["id"], "Tomasz travel note", {
      narrative: "We discussed the city of Tomasz and nearby routes.",
      participants: ["team"],
      tags: ["travel", "places"],
      location: "Poland",
    });
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "entity",
              label: "Tomasz",
              description: "A person participating in the conversation.",
              domain: "people",
              aliases: [],
              confidence: 0.7,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
          ],
          edges: [],
        }),
        createSemanticToolResponse({
          nodes: [
            {
              kind: "place",
              label: "Tomasz",
              description: "A city mentioned in the travel discussion.",
              domain: "places",
              aliases: [],
              confidence: 0.7,
              source_episode_ids: ["ep_bbbbbbbbbbbbbbbb"],
            },
          ],
          edges: [],
        }),
      ],
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episodeA, episodeB]),
      llmClient: llm,
      model: "haiku",
      clock,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await extractor.extractFromEpisodes([episodeA]);
    await extractor.extractFromEpisodes([episodeB]);

    const matches = await nodeRepository.findByExactLabelOrAlias("Tomasz", 5, {
      includeArchived: true,
    });

    expect(matches).toHaveLength(2);
    expect(matches.map((node) => node.kind).sort()).toEqual(["entity", "place"]);
    expect(matches.map((node) => node.domain).sort()).toEqual(["people", "places"]);
    expect(matches.map((node) => node.source_episode_ids[0])).toEqual(
      expect.arrayContaining([episodeA.id, episodeB.id]),
    );
  });

  it("merges compatible nodes even when one candidate has a specific domain", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Time concept note", {
      narrative: "Time came up as a broad concept.",
      participants: ["Alice"],
      audience_entity_id: "ent_aaaaaaaaaaaaaaaa" as Episode["audience_entity_id"],
      shared: false,
    });
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "concept",
              label: "Time",
              description: "A broad time concept.",
              aliases: [],
              confidence: 0.7,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
          ],
          edges: [],
        }),
        createSemanticToolResponse({
          nodes: [
            {
              kind: "concept",
              label: "Time",
              description: "A scientific time concept.",
              domain: "science",
              aliases: [],
              confidence: 0.7,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
          ],
          edges: [],
        }),
      ],
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: llm,
      model: "haiku",
      clock,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "concept",
      label: "Time",
      description: "Existing broad time concept.",
      domain: null,
      aliases: [],
      confidence: 0.6,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });

    await extractor.extractFromEpisodes([episode]);

    const afterNullCandidate = await nodeRepository.findByExactLabelOrAlias("Time", 5, {
      includeArchived: true,
    });

    expect(afterNullCandidate).toHaveLength(1);
    expect(afterNullCandidate[0]).toMatchObject({
      label: "Time",
      domain: null,
      description: "A broad time concept.",
    });

    await extractor.extractFromEpisodes([episode]);

    const afterSpecificCandidate = await nodeRepository.findByExactLabelOrAlias("Time", 5, {
      includeArchived: true,
    });

    expect(afterSpecificCandidate).toHaveLength(1);
    expect(afterSpecificCandidate[0]).toMatchObject({
      label: "Time",
      domain: "science",
      description: "A scientific time concept.",
    });
  });

  it("stores unknown domains as trimmed lowercase free-form strings", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Craft fair note", {
      narrative: "The note discussed artisanal craft details.",
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "concept",
                label: "Artisanal craft",
                description: "A handmade craft category.",
                domain: "  Artisanal-Craft  ",
                aliases: [],
                confidence: 0.7,
                source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await extractor.extractFromEpisodes([episode]);

    expect(await nodeRepository.list()).toEqual([
      expect.objectContaining({
        label: "Artisanal craft",
        domain: "artisanal-craft",
      }),
    ]);
  });

  it("preserves distinct observation nodes with different timeframe and count metadata", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const firstEpisode = buildEpisode(
      "ep_aaaaaaaaaaaaaaaa" as Episode["id"],
      "Early April video call",
      {
        narrative: "Nora observed Ruth repeat the same question three times on a video call.",
      },
    );
    const secondEpisode = buildEpisode(
      "ep_bbbbbbbbbbbbbbbb" as Episode["id"],
      "Mid April video call",
      {
        narrative: "Nora observed Ruth repeat the same question five times on a later video call.",
      },
    );
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "proposition",
              label: "Ruth repeated a question during a video call",
              description:
                "Nora observed Ruth repeat the same question three times during the early April video call.",
              domain: "family",
              aliases: [],
              observation_metadata: {
                witness: "Nora",
                timeframe: "early April video call",
                count_or_intensity: "three repetitions",
                source_kind: "direct_observation",
                confidence: 0.7,
                status: "observed",
              },
              confidence: 0.7,
              source_episode_ids: [firstEpisode.id],
            },
          ],
          edges: [],
        }),
        createSemanticToolResponse({
          nodes: [
            {
              kind: "proposition",
              label: "Ruth repeated a question during a video call",
              description:
                "Nora observed Ruth repeat the same question five times during the mid-April video call.",
              domain: "family",
              aliases: [],
              observation_metadata: {
                witness: "Nora",
                timeframe: "mid-April video call",
                count_or_intensity: "five repetitions",
                source_kind: "direct_observation",
                confidence: 0.7,
                status: "observed",
              },
              confidence: 0.7,
              source_episode_ids: [secondEpisode.id],
            },
          ],
          edges: [],
        }),
      ],
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([firstEpisode, secondEpisode]),
      llmClient: llm,
      model: "haiku",
      clock,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await extractor.extractFromEpisodes([firstEpisode]);
    const secondResult = await extractor.extractFromEpisodes([secondEpisode]);
    const nodes = await nodeRepository.list({
      limit: 10,
    });

    expect(secondResult).toMatchObject({
      insertedNodes: 1,
      updatedNodes: 0,
    });
    expect(nodes).toHaveLength(2);
    expect(nodes.map((node) => node.observation_metadata?.count_or_intensity).sort()).toEqual([
      "five repetitions",
      "three repetitions",
    ]);
  });
});
