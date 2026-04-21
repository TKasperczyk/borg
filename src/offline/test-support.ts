import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { DEFAULT_CONFIG, type Config } from "../config/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient, type LLMClient } from "../llm/index.js";
import {
  CommitmentRepository,
  EntityRepository,
  commitmentMigrations,
} from "../memory/commitments/index.js";
import {
  EpisodicRepository,
  createEpisodesTableSchema,
  episodicMigrations,
  type Episode,
} from "../memory/episodic/index.js";
import {
  GoalsRepository,
  TraitsRepository,
  ValuesRepository,
  selfMigrations,
} from "../memory/self/index.js";
import {
  ReviewQueueRepository,
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
  semanticMigrations,
  type SemanticNode,
} from "../memory/semantic/index.js";
import { retrievalMigrations } from "../retrieval/index.js";
import { StreamWriter } from "../stream/index.js";
import { LanceDbStore } from "../storage/lancedb/index.js";
import { openDatabase } from "../storage/sqlite/index.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { FixedClock, type Clock } from "../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createEpisodeId,
  createMaintenanceRunId,
  createSemanticNodeId,
  createStreamEntryId,
  type MaintenanceRunId,
} from "../util/ids.js";

import { AuditLog, ReverserRegistry, offlineMigrations, type OfflineContext } from "./index.js";

export class TestEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    if (/atlas|deploy|release|rollback|architecture/i.test(text)) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    if (/plan|planning|sprint|goal|roadmap/i.test(text)) {
      return Float32Array.from([0, 1, 0, 0]);
    }

    if (/reflect|pattern|habit|insight/i.test(text)) {
      return Float32Array.from([0, 0, 1, 0]);
    }

    return Float32Array.from([0, 0, 0, 1]);
  }
}

export type OfflineTestHarness = {
  tempDir: string;
  config: Config;
  clock: Clock;
  db: SqliteDatabase;
  embeddingClient: EmbeddingClient;
  llmClient: LLMClient;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository: SemanticNodeRepository;
  semanticEdgeRepository: SemanticEdgeRepository;
  reviewQueueRepository: ReviewQueueRepository;
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  entityRepository: EntityRepository;
  commitmentRepository: CommitmentRepository;
  registry: ReverserRegistry;
  auditLog: AuditLog;
  streamWriter: StreamWriter;
  createContext: (runId?: MaintenanceRunId) => OfflineContext;
  cleanup: () => Promise<void>;
};

export async function createOfflineTestHarness(
  options: {
    clock?: Clock;
    llmClient?: LLMClient;
    embeddingClient?: EmbeddingClient;
    configOverrides?: Partial<Config>;
  } = {},
): Promise<OfflineTestHarness> {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
  const clock = options.clock ?? new FixedClock(1_000_000);
  const embeddingClient = options.embeddingClient ?? new TestEmbeddingClient();
  const llmClient = options.llmClient ?? new FakeLLMClient();
  const config: Config = {
    ...DEFAULT_CONFIG,
    ...options.configOverrides,
    dataDir: tempDir,
    perception: {
      ...DEFAULT_CONFIG.perception,
      ...options.configOverrides?.perception,
    },
    embedding: {
      ...DEFAULT_CONFIG.embedding,
      ...options.configOverrides?.embedding,
      dims: 4,
    },
    anthropic: {
      ...DEFAULT_CONFIG.anthropic,
      ...options.configOverrides?.anthropic,
      models: {
        ...DEFAULT_CONFIG.anthropic.models,
        ...options.configOverrides?.anthropic?.models,
      },
    },
    offline: {
      ...DEFAULT_CONFIG.offline,
      ...options.configOverrides?.offline,
      consolidator: {
        ...DEFAULT_CONFIG.offline.consolidator,
        ...options.configOverrides?.offline?.consolidator,
      },
      reflector: {
        ...DEFAULT_CONFIG.offline.reflector,
        ...options.configOverrides?.offline?.reflector,
      },
      curator: {
        ...DEFAULT_CONFIG.offline.curator,
        ...options.configOverrides?.offline?.curator,
      },
      overseer: {
        ...DEFAULT_CONFIG.offline.overseer,
        ...options.configOverrides?.offline?.overseer,
      },
    },
  };
  const lance = new LanceDbStore({
    uri: join(tempDir, "lancedb"),
  });
  const db = openDatabase(join(tempDir, "borg.db"), {
    migrations: [
      ...episodicMigrations,
      ...selfMigrations,
      ...retrievalMigrations,
      ...semanticMigrations,
      ...commitmentMigrations,
      ...offlineMigrations,
    ],
  });
  const episodesTable = await lance.openTable({
    name: "episodes",
    schema: createEpisodesTableSchema(4),
  });
  const semanticNodesTable = await lance.openTable({
    name: "semantic_nodes",
    schema: createSemanticNodesTableSchema(4),
  });
  const episodicRepository = new EpisodicRepository({
    table: episodesTable,
    db,
    clock,
  });
  const registry = new ReverserRegistry();
  const semanticNodeRepository = new SemanticNodeRepository({
    table: semanticNodesTable,
    db,
    clock,
  });
  const reviewQueueRepository = new ReviewQueueRepository({
    db,
    clock,
    semanticNodeRepository,
  });
  const semanticEdgeRepository = new SemanticEdgeRepository({
    db,
    clock,
    enqueueReview: (input) => reviewQueueRepository.enqueue(input),
  });
  const valuesRepository = new ValuesRepository({
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
  const entityRepository = new EntityRepository({
    db,
    clock,
  });
  const commitmentRepository = new CommitmentRepository({
    db,
    clock,
  });
  const auditLog = new AuditLog({
    db,
    clock,
    registry,
  });
  const streamWriter = new StreamWriter({
    dataDir: tempDir,
    sessionId: DEFAULT_SESSION_ID,
    clock,
  });

  return {
    tempDir,
    config,
    clock,
    db,
    embeddingClient,
    llmClient,
    episodicRepository,
    semanticNodeRepository,
    semanticEdgeRepository,
    reviewQueueRepository,
    valuesRepository,
    goalsRepository,
    traitsRepository,
    entityRepository,
    commitmentRepository,
    registry,
    auditLog,
    streamWriter,
    createContext: (runId = createMaintenanceRunId()) => ({
      config,
      runId,
      clock,
      auditLog,
      streamWriter,
      embeddingClient,
      llm: {
        cognition: llmClient,
        background: llmClient,
        extraction: llmClient,
      },
      episodicRepository,
      semanticNodeRepository,
      semanticEdgeRepository,
      reviewQueueRepository,
      valuesRepository,
      goalsRepository,
      traitsRepository,
      entityRepository,
      commitmentRepository,
    }),
    cleanup: async () => {
      streamWriter.close();
      db.close();
      await lance.close();
      rmSync(tempDir, { recursive: true, force: true });
    },
  };
}

export function createEpisodeFixture(
  overrides: Partial<Episode> = {},
  vector = [0, 1, 0, 0],
): Episode {
  const nowMs = overrides.created_at ?? 1_000_000;

  return {
    id: overrides.id ?? createEpisodeId(),
    title: overrides.title ?? "Planning sync",
    narrative: overrides.narrative ?? "The team reviewed the plan and captured next steps.",
    participants: overrides.participants ?? ["team"],
    location: overrides.location ?? null,
    start_time: overrides.start_time ?? nowMs - 1_000,
    end_time: overrides.end_time ?? nowMs,
    source_stream_ids: overrides.source_stream_ids ?? [createStreamEntryId()],
    significance: overrides.significance ?? 0.7,
    tags: overrides.tags ?? ["planning"],
    confidence: overrides.confidence ?? 0.8,
    lineage: overrides.lineage ?? {
      derived_from: [],
      supersedes: [],
    },
    embedding: overrides.embedding ?? Float32Array.from(vector),
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
  };
}

export function createSemanticNodeFixture(
  overrides: Partial<SemanticNode> = {},
  vector = [0, 0, 1, 0],
): SemanticNode {
  const nowMs = overrides.created_at ?? 1_000_000;

  return {
    id: overrides.id ?? createSemanticNodeId(),
    kind: overrides.kind ?? "proposition",
    label: overrides.label ?? "Release stability improves after rollback planning",
    description:
      overrides.description ??
      "Rollback planning tends to reduce deployment mistakes in the release workflow.",
    aliases: overrides.aliases ?? [],
    confidence: overrides.confidence ?? 0.5,
    source_episode_ids: overrides.source_episode_ids ?? [createEpisodeId()],
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
    last_verified_at: overrides.last_verified_at ?? nowMs,
    embedding: overrides.embedding ?? Float32Array.from(vector),
    archived: overrides.archived ?? false,
    superseded_by: overrides.superseded_by ?? null,
  };
}
