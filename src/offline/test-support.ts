import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { DEFAULT_CONFIG, type Config } from "../config/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient, type LLMClient } from "../llm/index.js";
import { MoodRepository, affectiveMigrations } from "../memory/affective/index.js";
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
  SkillRepository,
  createSkillsTableSchema,
  proceduralMigrations,
} from "../memory/procedural/index.js";
import {
  IdentityEventRepository,
  IdentityService,
  identityMigrations,
} from "../memory/identity/index.js";
import {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
  selfMigrations,
} from "../memory/self/index.js";
import {
  appendOpenQuestionHookFailureEvent,
  enqueueOpenQuestionForReview,
} from "../memory/self/review-open-question-hook.js";
import {
  ReviewQueueRepository,
  SemanticGraph,
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
  semanticMigrations,
  type SemanticNode,
} from "../memory/semantic/index.js";
import { SocialRepository, socialMigrations } from "../memory/social/index.js";
import { retrievalMigrations } from "../retrieval/index.js";
import { RetrievalPipeline } from "../retrieval/index.js";
import {
  StreamEntryIndexRepository,
  StreamWriter,
  streamEntryIndexMigrations,
  streamWatermarkMigrations,
} from "../stream/index.js";
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
  identityEventRepository: IdentityEventRepository;
  identityService: IdentityService;
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  autobiographicalRepository: AutobiographicalRepository;
  growthMarkersRepository: GrowthMarkersRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  moodRepository: MoodRepository;
  socialRepository: SocialRepository;
  entityRepository: EntityRepository;
  commitmentRepository: CommitmentRepository;
  skillRepository: SkillRepository;
  retrievalPipeline: RetrievalPipeline;
  registry: ReverserRegistry;
  auditLog: AuditLog;
  streamWriter: StreamWriter;
  flushHookLogs: () => Promise<void>;
  createContext: (runId?: MaintenanceRunId) => OfflineContext;
  cleanup: () => Promise<void>;
};

export async function createOfflineTestHarness(
  options: {
    clock?: Clock;
    llmClient?: LLMClient;
    embeddingClient?: EmbeddingClient;
    embeddingDimensions?: number;
    configOverrides?: Partial<Config>;
  } = {},
): Promise<OfflineTestHarness> {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
  const clock = options.clock ?? new FixedClock(1_000_000);
  const embeddingClient = options.embeddingClient ?? new TestEmbeddingClient();
  const llmClient = options.llmClient ?? new FakeLLMClient();
  const embeddingDimensions = options.embeddingDimensions ?? 4;
  const config: Config = {
    ...DEFAULT_CONFIG,
    ...options.configOverrides,
    dataDir: tempDir,
    perception: {
      ...DEFAULT_CONFIG.perception,
      ...options.configOverrides?.perception,
    },
    affective: {
      ...DEFAULT_CONFIG.affective,
      ...options.configOverrides?.affective,
    },
    embedding: {
      ...DEFAULT_CONFIG.embedding,
      ...options.configOverrides?.embedding,
      dims: embeddingDimensions,
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
      ruminator: {
        ...DEFAULT_CONFIG.offline.ruminator,
        ...options.configOverrides?.offline?.ruminator,
      },
      selfNarrator: {
        ...DEFAULT_CONFIG.offline.selfNarrator,
        ...options.configOverrides?.offline?.selfNarrator,
      },
    },
    autonomy: {
      ...DEFAULT_CONFIG.autonomy,
      ...options.configOverrides?.autonomy,
      triggers: {
        ...DEFAULT_CONFIG.autonomy.triggers,
        ...options.configOverrides?.autonomy?.triggers,
        commitmentExpiring: {
          ...DEFAULT_CONFIG.autonomy.triggers.commitmentExpiring,
          ...options.configOverrides?.autonomy?.triggers?.commitmentExpiring,
        },
        openQuestionDormant: {
          ...DEFAULT_CONFIG.autonomy.triggers.openQuestionDormant,
          ...options.configOverrides?.autonomy?.triggers?.openQuestionDormant,
        },
        scheduledReflection: {
          ...DEFAULT_CONFIG.autonomy.triggers.scheduledReflection,
          ...options.configOverrides?.autonomy?.triggers?.scheduledReflection,
        },
        goalFollowupDue: {
          ...DEFAULT_CONFIG.autonomy.triggers.goalFollowupDue,
          ...options.configOverrides?.autonomy?.triggers?.goalFollowupDue,
        },
      },
      conditions: {
        ...DEFAULT_CONFIG.autonomy.conditions,
        ...options.configOverrides?.autonomy?.conditions,
        commitmentRevoked: {
          ...DEFAULT_CONFIG.autonomy.conditions.commitmentRevoked,
          ...options.configOverrides?.autonomy?.conditions?.commitmentRevoked,
        },
        moodValenceDrop: {
          ...DEFAULT_CONFIG.autonomy.conditions.moodValenceDrop,
          ...options.configOverrides?.autonomy?.conditions?.moodValenceDrop,
        },
        openQuestionUrgencyBump: {
          ...DEFAULT_CONFIG.autonomy.conditions.openQuestionUrgencyBump,
          ...options.configOverrides?.autonomy?.conditions?.openQuestionUrgencyBump,
        },
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
      ...affectiveMigrations,
      ...retrievalMigrations,
      ...semanticMigrations,
      ...commitmentMigrations,
      ...socialMigrations,
      ...proceduralMigrations,
      ...identityMigrations,
      ...offlineMigrations,
      ...streamWatermarkMigrations,
      ...streamEntryIndexMigrations,
    ],
  });
  const episodesTable = await lance.openTable({
    name: "episodes",
    schema: createEpisodesTableSchema(embeddingDimensions),
  });
  const semanticNodesTable = await lance.openTable({
    name: "semantic_nodes",
    schema: createSemanticNodesTableSchema(embeddingDimensions),
  });
  const skillsTable = await lance.openTable({
    name: "skills",
    schema: createSkillsTableSchema(embeddingDimensions),
  });
  const episodicRepository = new EpisodicRepository({
    table: episodesTable,
    db,
    clock,
  });
  const entryIndex = new StreamEntryIndexRepository({
    db,
    dataDir: tempDir,
  });
  const streamWriter = new StreamWriter({
    dataDir: tempDir,
    sessionId: DEFAULT_SESSION_ID,
    clock,
    entryIndex,
  });
  const pendingHookLogs = new Set<Promise<void>>();
  const registry = new ReverserRegistry();
  const semanticNodeRepository = new SemanticNodeRepository({
    table: semanticNodesTable,
    db,
    clock,
  });
  const openQuestionsRepository = new OpenQuestionsRepository({
    db,
    clock,
  });
  const moodRepository = new MoodRepository({
    db,
    clock,
    defaultHalfLifeHours: config.affective.moodHalfLifeHours,
    incomingWeight: config.affective.incomingMoodWeight,
  });
  let reviewQueueRepository: ReviewQueueRepository;
  const semanticEdgeRepository = new SemanticEdgeRepository({
    db,
    clock,
    enqueueReview: (input) => reviewQueueRepository.enqueue(input),
  });
  const semanticGraph = new SemanticGraph({
    nodeRepository: semanticNodeRepository,
    edgeRepository: semanticEdgeRepository,
  });
  const identityEventRepository = new IdentityEventRepository({
    db,
    clock,
  });
  const valuesRepository = new ValuesRepository({
    db,
    clock,
    identityEventRepository,
  });
  const goalsRepository = new GoalsRepository({
    db,
    clock,
    identityEventRepository,
  });
  const traitsRepository = new TraitsRepository({
    db,
    clock,
    identityEventRepository,
  });
  const autobiographicalRepository = new AutobiographicalRepository({
    db,
    clock,
  });
  const growthMarkersRepository = new GrowthMarkersRepository({
    db,
    clock,
  });
  const entityRepository = new EntityRepository({
    db,
    clock,
  });
  const socialRepository = new SocialRepository({
    db,
    clock,
  });
  const commitmentRepository = new CommitmentRepository({
    db,
    clock,
    identityEventRepository,
  });
  const identityService = new IdentityService({
    valuesRepository,
    goalsRepository,
    traitsRepository,
    autobiographicalRepository,
    growthMarkersRepository,
    openQuestionsRepository,
    commitmentRepository,
    identityEventRepository,
  });
  reviewQueueRepository = new ReviewQueueRepository({
    db,
    clock,
    episodicRepository,
    semanticNodeRepository,
    valuesRepository,
    goalsRepository,
    traitsRepository,
    autobiographicalRepository,
    commitmentRepository,
    identityService,
    onEnqueue: (item) => enqueueOpenQuestionForReview(openQuestionsRepository, item),
    onEnqueueError: (error) => {
      const promise = appendOpenQuestionHookFailureEvent(
        streamWriter,
        "review_queue_open_question",
        error,
      ).finally(() => {
        pendingHookLogs.delete(promise);
      });
      pendingHookLogs.add(promise);
    },
  });
  const skillRepository = new SkillRepository({
    table: skillsTable,
    db,
    embeddingClient,
    clock,
  });
  const auditLog = new AuditLog({
    db,
    clock,
    registry,
  });
  const retrievalPipeline = new RetrievalPipeline({
    embeddingClient,
    episodicRepository,
    semanticNodeRepository,
    semanticGraph,
    openQuestionsRepository,
    dataDir: tempDir,
    entryIndex,
    clock,
  });
  const flushHookLogs = async () => {
    await Promise.all([...pendingHookLogs]);
  };

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
    identityEventRepository,
    identityService,
    valuesRepository,
    goalsRepository,
    traitsRepository,
    autobiographicalRepository,
    growthMarkersRepository,
    openQuestionsRepository,
    moodRepository,
    socialRepository,
    entityRepository,
    commitmentRepository,
    skillRepository,
    retrievalPipeline,
    registry,
    auditLog,
    streamWriter,
    flushHookLogs,
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
      identityService,
      identityEventRepository,
      valuesRepository,
      goalsRepository,
      traitsRepository,
      autobiographicalRepository,
      growthMarkersRepository,
      openQuestionsRepository,
      moodRepository,
      socialRepository,
      entityRepository,
      commitmentRepository,
      skillRepository,
      retrievalPipeline,
    }),
    cleanup: async () => {
      await flushHookLogs();
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
    emotional_arc: overrides.emotional_arc ?? null,
    audience_entity_id: overrides.audience_entity_id,
    shared: overrides.shared,
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
    domain: overrides.domain ?? null,
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
