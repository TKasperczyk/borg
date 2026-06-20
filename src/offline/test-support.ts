import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { autonomyMigrations } from "../autonomy/index.js";
import { promptSurfaceHistoryMigrations } from "../cognition/prompts/prompt-surface-history-migrations.js";
import { CorrectionService } from "../correction/index.js";
import { DEFAULT_CONFIG, type Config } from "../config/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { executiveMigrations, ExecutiveStepsRepository } from "../executive/index.js";
import { type LLMClient } from "../llm/index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import { MoodRepository, affectiveMigrations } from "../memory/affective/index.js";
import {
  ActionRepository,
  actionMigrations,
  createActionRecordsTableSchema,
  resolveOpenQuestionsForCompletedAction,
} from "../memory/actions/index.js";
import {
  CommitmentRepository,
  EntityRepository,
  commitmentMigrations,
} from "../memory/commitments/index.js";
import { unknownMemoryDisclosureLabel } from "../memory/common/disclosure-label.js";
import {
  CreatorDirectiveRepository,
  creatorDirectiveMigrations,
} from "../memory/creator-directives/index.js";
import {
  RelationalSlotRepository,
  relationalSlotMigrations,
} from "../memory/relational-slots/index.js";
import { sharedStateMigrations } from "../memory/shared-state/index.js";
import {
  EpisodicRepository,
  createEpisodesTableSchema,
  episodicMigrations,
  type Episode,
} from "../memory/episodic/index.js";
import {
  ProceduralContextStatsRepository,
  ProceduralEvidenceRepository,
  SkillRepository,
  createSkillsTableSchema,
  proceduralMigrations,
  type SkillRecord,
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
  createOpenQuestionsTableSchema,
  selfMigrations,
} from "../memory/self/index.js";
import { selfDecisionMigrations } from "../memory/self-decisions/index.js";
import { trainOfThoughtMigrations } from "../memory/train-of-thought/index.js";
import {
  appendOpenQuestionHookFailureEvent,
  enqueueOpenQuestionForReview,
  type ReviewOpenQuestionExtractorLike,
} from "../memory/self/review-open-question-hook.js";
import {
  SemanticBeliefDependencyRepository,
  SemanticGraph,
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
  semanticMigrations,
  type SemanticEdge,
  type SemanticNode,
} from "../memory/semantic/index.js";
import {
  ReviewQueueRepository,
  createCorrectionReviewHandler,
  createSkillSplitReviewQueueHandler,
  registerBuiltinReviewQueueHandlers,
  type ReviewQueueItem,
} from "../memory/review-queue/index.js";
import { SocialRepository, socialMigrations } from "../memory/social/index.js";
import {
  RecallStateRepository,
  retrievalMigrations,
  type RetrievedEpisode,
} from "../retrieval/index.js";
import { RetrievalPipeline } from "../retrieval/index.js";
import {
  StreamEntryIndexRepository,
  StreamWriter,
  streamEntryIndexMigrations,
  streamWatermarkMigrations,
} from "../stream/index.js";
import { LanceDbStore } from "../storage/lancedb/index.js";
import { composeMigrations, openDatabase } from "../storage/sqlite/index.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { FixedClock, type Clock } from "../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createEpisodeId,
  createMaintenanceRunId,
  createSemanticEdgeId,
  createSemanticNodeId,
  createSessionId,
  createSkillId,
  createStreamEntryId,
  type MaintenanceRunId,
  type SessionId,
} from "../util/ids.js";
import type { WorkingMemory } from "../memory/working/index.js";

import {
  AuditLog,
  ReverserRegistry,
  createSkillSplitReviewHandler,
  offlineMigrations,
  type OfflineContext,
} from "./index.js";

export class TestEmbeddingClient implements EmbeddingClient {
  constructor(
    private readonly vectorsByText: ReadonlyMap<string, readonly number[]> = new Map(),
    private readonly dims = 4,
  ) {}

  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    const scripted = this.vectorsByText.get(text);

    if (scripted !== undefined) {
      return Float32Array.from(scripted);
    }

    const vector = new Float32Array(this.dims);
    let seed = 2_166_136_261;

    for (let index = 0; index < text.length; index += 1) {
      seed ^= text.charCodeAt(index);
      seed = Math.imul(seed, 16_777_619);
    }

    for (let index = 0; index < vector.length; index += 1) {
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      vector[index] = ((seed >>> 0) / 0xffffffff) * 2 - 1;
    }

    return vector;
  }
}

export type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends Array<infer U>
    ? Array<U>
    : T[K] extends ReadonlyArray<infer U>
      ? ReadonlyArray<U>
      : T[K] extends object
        ? DeepPartial<T[K]>
        : T[K];
};

export function testSessionId(value?: SessionId | string): SessionId {
  return value === undefined ? createSessionId() : (value as SessionId);
}

export function createTestConfig(
  overrides: DeepPartial<Config> = {},
  options: { embeddingDimensions?: number } = {},
): Config {
  const embeddingDimensions = options.embeddingDimensions ?? overrides.embedding?.dims ?? 4;

  return {
    ...DEFAULT_CONFIG,
    ...overrides,
    dataDir: overrides.dataDir ?? "/tmp/borg-test",
    perception: {
      ...DEFAULT_CONFIG.perception,
      ...overrides.perception,
    },
    affective: {
      ...DEFAULT_CONFIG.affective,
      ...overrides.affective,
    },
    embedding: {
      ...DEFAULT_CONFIG.embedding,
      ...overrides.embedding,
      dims: embeddingDimensions,
    },
    anthropic: {
      ...DEFAULT_CONFIG.anthropic,
      ...overrides.anthropic,
      models: {
        ...DEFAULT_CONFIG.anthropic.models,
        ...overrides.anthropic?.models,
      },
    },
    procedural: {
      ...DEFAULT_CONFIG.procedural,
      ...overrides.procedural,
    },
    retrieval: {
      ...DEFAULT_CONFIG.retrieval,
      ...overrides.retrieval,
      semanticOverfetchMultiplier:
        overrides.retrieval?.semanticOverfetchMultiplier ??
        DEFAULT_CONFIG.retrieval.semanticOverfetchMultiplier,
      semantic: {
        ...DEFAULT_CONFIG.retrieval.semantic,
        ...overrides.retrieval?.semantic,
        statusMultipliers: {
          ...DEFAULT_CONFIG.retrieval.semantic.statusMultipliers,
          ...overrides.retrieval?.semantic?.statusMultipliers,
        },
      },
    },
    commitments: {
      ...DEFAULT_CONFIG.commitments,
      ...overrides.commitments,
      enforce: {
        ...DEFAULT_CONFIG.commitments.enforce,
        ...overrides.commitments?.enforce,
      },
    },
    attachments: {
      ...DEFAULT_CONFIG.attachments,
      ...overrides.attachments,
    },
    deliberation: {
      ...DEFAULT_CONFIG.deliberation,
      ...overrides.deliberation,
      contradictionRouting: {
        ...DEFAULT_CONFIG.deliberation.contradictionRouting,
        ...overrides.deliberation?.contradictionRouting,
      },
    },
    cognition: {
      ...DEFAULT_CONFIG.cognition,
      ...overrides.cognition,
      actionLifecycle: {
        ...DEFAULT_CONFIG.cognition.actionLifecycle,
        ...overrides.cognition?.actionLifecycle,
      },
    },
    generation: {
      ...DEFAULT_CONFIG.generation,
      ...overrides.generation,
      cognition: {
        ...DEFAULT_CONFIG.generation.cognition,
        ...overrides.generation?.cognition,
        thinking: {
          ...DEFAULT_CONFIG.generation.cognition.thinking,
          ...overrides.generation?.cognition?.thinking,
        },
      },
      evidenceLedger: {
        ...DEFAULT_CONFIG.generation.evidenceLedger,
        // Most full-turn test fixtures use short fake LLM response queues
        // and exercise legacy prompt paths unless they explicitly opt into
        // ledger behavior.
        enabled: overrides.generation?.evidenceLedger?.enabled ?? false,
        ...overrides.generation?.evidenceLedger,
        recentLivedExperience: {
          ...DEFAULT_CONFIG.generation.evidenceLedger.recentLivedExperience,
          ...overrides.generation?.evidenceLedger?.recentLivedExperience,
        },
        decisionArtifact: {
          ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
          ...overrides.generation?.evidenceLedger?.decisionArtifact,
          kindSoftCaps: {
            ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact.kindSoftCaps,
            ...overrides.generation?.evidenceLedger?.decisionArtifact?.kindSoftCaps,
          },
          renderReservedSlots: {
            ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact.renderReservedSlots,
            ...overrides.generation?.evidenceLedger?.decisionArtifact?.renderReservedSlots,
          },
          previousArtifactSummary: {
            ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact.previousArtifactSummary,
            ...overrides.generation?.evidenceLedger?.decisionArtifact?.previousArtifactSummary,
            maxEntries: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact.previousArtifactSummary
                .maxEntries,
              ...overrides.generation?.evidenceLedger?.decisionArtifact?.previousArtifactSummary
                ?.maxEntries,
            },
          },
          compilerPrefilter: {
            ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact.compilerPrefilter,
            ...overrides.generation?.evidenceLedger?.decisionArtifact?.compilerPrefilter,
          },
          ledgerDelta: {
            ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact.ledgerDelta,
            ...overrides.generation?.evidenceLedger?.decisionArtifact?.ledgerDelta,
          },
        },
      },
      postGenerationGuards: {
        ...DEFAULT_CONFIG.generation.postGenerationGuards,
        ...overrides.generation?.postGenerationGuards,
        commitment: {
          ...DEFAULT_CONFIG.generation.postGenerationGuards.commitment,
          ...overrides.generation?.postGenerationGuards?.commitment,
        },
        closurePressure: {
          ...DEFAULT_CONFIG.generation.postGenerationGuards.closurePressure,
          ...overrides.generation?.postGenerationGuards?.closurePressure,
        },
      },
    },
    streamIngestion: {
      ...DEFAULT_CONFIG.streamIngestion,
      ...overrides.streamIngestion,
      settle: {
        ...DEFAULT_CONFIG.streamIngestion.settle,
        ...overrides.streamIngestion?.settle,
      },
      preTurnCatchup: {
        ...DEFAULT_CONFIG.streamIngestion.preTurnCatchup,
        ...overrides.streamIngestion?.preTurnCatchup,
      },
    },
    executive: {
      ...DEFAULT_CONFIG.executive,
      ...overrides.executive,
    },
    offline: {
      ...DEFAULT_CONFIG.offline,
      ...overrides.offline,
      consolidator: {
        ...DEFAULT_CONFIG.offline.consolidator,
        ...overrides.offline?.consolidator,
      },
      reflector: {
        ...DEFAULT_CONFIG.offline.reflector,
        ...overrides.offline?.reflector,
      },
      associator: {
        ...DEFAULT_CONFIG.offline.associator,
        ...overrides.offline?.associator,
      },
      proceduralSynthesizer: {
        ...DEFAULT_CONFIG.offline.proceduralSynthesizer,
        ...overrides.offline?.proceduralSynthesizer,
      },
      curator: {
        ...DEFAULT_CONFIG.offline.curator,
        ...overrides.offline?.curator,
      },
      overseer: {
        ...DEFAULT_CONFIG.offline.overseer,
        ...overrides.offline?.overseer,
      },
      reviewResolver: {
        ...DEFAULT_CONFIG.offline.reviewResolver,
        ...overrides.offline?.reviewResolver,
      },
      ruminator: {
        ...DEFAULT_CONFIG.offline.ruminator,
        ...overrides.offline?.ruminator,
      },
      selfNarrator: {
        ...DEFAULT_CONFIG.offline.selfNarrator,
        ...overrides.offline?.selfNarrator,
      },
      beliefReviser: {
        ...DEFAULT_CONFIG.offline.beliefReviser,
        ...overrides.offline?.beliefReviser,
      },
      creatorDirectiveReconciler: {
        ...DEFAULT_CONFIG.offline.creatorDirectiveReconciler,
        ...overrides.offline?.creatorDirectiveReconciler,
      },
      commitmentReconciler: {
        ...DEFAULT_CONFIG.offline.commitmentReconciler,
        ...overrides.offline?.commitmentReconciler,
      },
      semanticExtractor: {
        ...DEFAULT_CONFIG.offline.semanticExtractor,
        ...overrides.offline?.semanticExtractor,
      },
    },
    maintenance: {
      ...DEFAULT_CONFIG.maintenance,
      ...overrides.maintenance,
    },
    autonomy: {
      ...DEFAULT_CONFIG.autonomy,
      ...overrides.autonomy,
      proactiveOutbound: {
        ...DEFAULT_CONFIG.autonomy.proactiveOutbound,
        ...overrides.autonomy?.proactiveOutbound,
        allowByConfig: {
          ...DEFAULT_CONFIG.autonomy.proactiveOutbound.allowByConfig,
          ...overrides.autonomy?.proactiveOutbound?.allowByConfig,
          sessionIds: [
            ...(overrides.autonomy?.proactiveOutbound?.allowByConfig?.sessionIds ??
              DEFAULT_CONFIG.autonomy.proactiveOutbound.allowByConfig.sessionIds),
          ],
          sourceTypes: [
            ...(overrides.autonomy?.proactiveOutbound?.allowByConfig?.sourceTypes ??
              DEFAULT_CONFIG.autonomy.proactiveOutbound.allowByConfig.sourceTypes),
          ],
        },
      },
      executiveFocus: {
        ...DEFAULT_CONFIG.autonomy.executiveFocus,
        ...overrides.autonomy?.executiveFocus,
      },
      triggers: {
        ...DEFAULT_CONFIG.autonomy.triggers,
        ...overrides.autonomy?.triggers,
        commitmentExpiring: {
          ...DEFAULT_CONFIG.autonomy.triggers.commitmentExpiring,
          ...overrides.autonomy?.triggers?.commitmentExpiring,
        },
        openQuestionDormant: {
          ...DEFAULT_CONFIG.autonomy.triggers.openQuestionDormant,
          ...overrides.autonomy?.triggers?.openQuestionDormant,
        },
        scheduledReflection: {
          ...DEFAULT_CONFIG.autonomy.triggers.scheduledReflection,
          ...overrides.autonomy?.triggers?.scheduledReflection,
        },
        scheduledWake: {
          ...DEFAULT_CONFIG.autonomy.triggers.scheduledWake,
          ...overrides.autonomy?.triggers?.scheduledWake,
        },
        goalFollowupDue: {
          ...DEFAULT_CONFIG.autonomy.triggers.goalFollowupDue,
          ...overrides.autonomy?.triggers?.goalFollowupDue,
        },
      },
      conditions: {
        ...DEFAULT_CONFIG.autonomy.conditions,
        ...overrides.autonomy?.conditions,
        commitmentRevoked: {
          ...DEFAULT_CONFIG.autonomy.conditions.commitmentRevoked,
          ...overrides.autonomy?.conditions?.commitmentRevoked,
        },
        moodValenceDrop: {
          ...DEFAULT_CONFIG.autonomy.conditions.moodValenceDrop,
          ...overrides.autonomy?.conditions?.moodValenceDrop,
        },
        openQuestionUrgencyBump: {
          ...DEFAULT_CONFIG.autonomy.conditions.openQuestionUrgencyBump,
          ...overrides.autonomy?.conditions?.openQuestionUrgencyBump,
        },
      },
    },
  };
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
  semanticGraph: SemanticGraph;
  semanticBeliefDependencyRepository: SemanticBeliefDependencyRepository;
  recallStateRepository: RecallStateRepository;
  reviewQueueRepository: ReviewQueueRepository;
  identityEventRepository: IdentityEventRepository;
  identityService: IdentityService;
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
  executiveStepsRepository: ExecutiveStepsRepository;
  traitsRepository: TraitsRepository;
  autobiographicalRepository: AutobiographicalRepository;
  growthMarkersRepository: GrowthMarkersRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  moodRepository: MoodRepository;
  actionRepository: ActionRepository;
  socialRepository: SocialRepository;
  entityRepository: EntityRepository;
  relationalSlotRepository: RelationalSlotRepository;
  commitmentRepository: CommitmentRepository;
  creatorDirectiveRepository: CreatorDirectiveRepository;
  skillRepository: SkillRepository;
  proceduralContextStatsRepository: ProceduralContextStatsRepository;
  proceduralEvidenceRepository: ProceduralEvidenceRepository;
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
    configOverrides?: DeepPartial<Config>;
    reviewOpenQuestionExtractor?: ReviewOpenQuestionExtractorLike | null;
    tracer?: import("../tracing/tracer.js").TurnTracer;
  } = {},
): Promise<OfflineTestHarness> {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
  const clock = options.clock ?? new FixedClock(1_000_000);
  const embeddingClient = options.embeddingClient ?? new TestEmbeddingClient();
  const llmClient = options.llmClient ?? new FakeLLMClient();
  const embeddingDimensions = options.embeddingDimensions ?? 4;
  const config = createTestConfig(
    {
      ...options.configOverrides,
      dataDir: tempDir,
    },
    { embeddingDimensions },
  );
  const lance = new LanceDbStore({
    uri: join(tempDir, "lancedb"),
  });
  const db = openDatabase(join(tempDir, "borg.db"), {
    migrations: composeMigrations(
      episodicMigrations,
      selfMigrations,
      executiveMigrations,
      affectiveMigrations,
      retrievalMigrations,
      semanticMigrations,
      commitmentMigrations,
      creatorDirectiveMigrations,
      relationalSlotMigrations,
      sharedStateMigrations,
      socialMigrations,
      proceduralMigrations,
      actionMigrations,
      identityMigrations,
      offlineMigrations,
      autonomyMigrations,
      streamWatermarkMigrations,
      streamEntryIndexMigrations,
      selfDecisionMigrations,
      trainOfThoughtMigrations,
      promptSurfaceHistoryMigrations,
    ),
  });
  const episodesTable = await lance.openTable({
    name: "episodes",
    schema: createEpisodesTableSchema(embeddingDimensions),
  });
  const semanticNodesTable = await lance.openTable({
    name: "semantic_nodes",
    schema: createSemanticNodesTableSchema(embeddingDimensions),
  });
  const openQuestionsTable = await lance.openTable({
    name: "open_questions",
    schema: createOpenQuestionsTableSchema(embeddingDimensions),
  });
  const skillsTable = await lance.openTable({
    name: "skills",
    schema: createSkillsTableSchema(embeddingDimensions),
  });
  const actionRecordsTable = await lance.openTable({
    name: "action_records",
    schema: createActionRecordsTableSchema(embeddingDimensions),
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
  let applyCorrectionReview: ((item: ReviewQueueItem) => Promise<void>) | undefined;
  const semanticNodeRepository = new SemanticNodeRepository({
    table: semanticNodesTable,
    db,
    clock,
  });
  const openQuestionsRepository = new OpenQuestionsRepository({
    db,
    table: openQuestionsTable,
    embeddingClient,
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
  const semanticBeliefDependencyRepository = new SemanticBeliefDependencyRepository({
    db,
    clock,
  });
  const semanticGraph = new SemanticGraph({
    nodeRepository: semanticNodeRepository,
    edgeRepository: semanticEdgeRepository,
  });
  const recallStateRepository = new RecallStateRepository({
    db,
    clock,
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
  const executiveStepsRepository = new ExecutiveStepsRepository({
    db,
    clock,
  });
  const goalsRepository = new GoalsRepository({
    db,
    clock,
    identityEventRepository,
    executiveStepsRepository,
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
  const creatorDirectiveRepository = new CreatorDirectiveRepository({
    db,
    clock,
  });
  const relationalSlotRepository = new RelationalSlotRepository({
    db,
    clock,
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
    semanticEdgeRepository,
    valuesRepository,
    goalsRepository,
    traitsRepository,
    autobiographicalRepository,
    commitmentRepository,
    identityService,
    identityEventRepository,
    tracer: options.tracer,
    onEnqueue: (item) =>
      enqueueOpenQuestionForReview(identityService, item, {
        extractor: options.reviewOpenQuestionExtractor ?? null,
      }),
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
  registerBuiltinReviewQueueHandlers(reviewQueueRepository);
  reviewQueueRepository.registerHandler(
    createCorrectionReviewHandler({
      applyCorrection: (item) => {
        if (applyCorrectionReview === undefined) {
          throw new Error("Correction service not initialized");
        }

        return applyCorrectionReview(item);
      },
    }),
  );
  const skillRepository = new SkillRepository({
    table: skillsTable,
    db,
    embeddingClient,
    clock,
  });
  const actionRepository = new ActionRepository({
    table: actionRecordsTable,
    db,
    embeddingClient,
    clock,
    onCompleted: (record) => {
      resolveOpenQuestionsForCompletedAction({
        action: record,
        openQuestionsRepository,
        identityService,
      });
    },
  });
  const proceduralEvidenceRepository = new ProceduralEvidenceRepository({
    db,
    clock,
  });
  const proceduralContextStatsRepository = new ProceduralContextStatsRepository({
    db,
    clock,
  });
  const auditLog = new AuditLog({
    db,
    clock,
    registry,
  });
  reviewQueueRepository.registerHandler(
    createSkillSplitReviewQueueHandler(
      createSkillSplitReviewHandler({
        skillRepository,
        auditLog,
        clock,
      }),
    ),
  );
  const retrievalPipeline = new RetrievalPipeline({
    embeddingClient,
    llmClient,
    recallExpansionModel: config.anthropic.models.recallExpansion,
    episodicRepository,
    semanticNodeRepository,
    semanticEdgeRepository,
    semanticGraph,
    reviewQueueRepository,
    openQuestionsRepository,
    entityRepository,
    commitmentRepository,
    recallStateRepository,
    dataDir: tempDir,
    entryIndex,
    clock,
    semanticUnderReviewMultiplier: config.retrieval.semantic.underReviewMultiplier,
    semanticStatusMultipliers: config.retrieval.semantic.statusMultipliers,
  });
  const correctionService = new CorrectionService({
    config,
    db,
    clock,
    retrievalPipeline,
    episodicRepository,
    semanticNodeRepository,
    semanticEdgeRepository,
    semanticGraph,
    valuesRepository,
    goalsRepository,
    traitsRepository,
    openQuestionsRepository,
    socialRepository,
    entityRepository,
    commitmentRepository,
    reviewQueueRepository,
    identityService,
    identityEventRepository,
  });
  applyCorrectionReview = (item) => correctionService.applyCorrectionReview(item);
  const flushHookLogs = async () => {
    await reviewQueueRepository.flushEnqueueHooks();
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
    semanticGraph,
    semanticBeliefDependencyRepository,
    recallStateRepository,
    reviewQueueRepository,
    identityEventRepository,
    identityService,
    valuesRepository,
    goalsRepository,
    executiveStepsRepository,
    traitsRepository,
    autobiographicalRepository,
    growthMarkersRepository,
    openQuestionsRepository,
    moodRepository,
    actionRepository,
    socialRepository,
    entityRepository,
    relationalSlotRepository,
    commitmentRepository,
    creatorDirectiveRepository,
    skillRepository,
    proceduralContextStatsRepository,
    proceduralEvidenceRepository,
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
      entryIndex,
      embeddingClient,
      tracer: options.tracer,
      llm: {
        cognition: llmClient,
        background: llmClient,
        extraction: llmClient,
      },
      episodicRepository,
      semanticNodeRepository,
      semanticEdgeRepository,
      semanticBeliefDependencyRepository,
      semanticReviewService: undefined,
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
      actionRepository,
      socialRepository,
      entityRepository,
      relationalSlotRepository,
      commitmentRepository,
      creatorDirectiveRepository,
      skillRepository,
      proceduralEvidenceRepository,
      retrievalPipeline,
    }),
    cleanup: async () => {
      await flushHookLogs();
      await actionRepository.waitForPendingEmbeddings();
      await openQuestionsRepository.waitForPendingEmbeddings();
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
    origin_audience_entity_ids: overrides.origin_audience_entity_ids,
    shared: overrides.shared,
    embedding: overrides.embedding ?? Float32Array.from(vector),
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
  };
}

export function createWorkingMemoryFixture(overrides: Partial<WorkingMemory> = {}): WorkingMemory {
  return {
    session_id: overrides.session_id ?? DEFAULT_SESSION_ID,
    turn_counter: overrides.turn_counter ?? 1,
    hot_entities: overrides.hot_entities ?? [],
    pending_actions: overrides.pending_actions ?? [],
    pending_social_attribution: overrides.pending_social_attribution ?? null,
    pending_trait_attribution: overrides.pending_trait_attribution ?? null,
    suppressed: overrides.suppressed ?? [],
    mood: overrides.mood ?? null,
    pending_procedural_attempts: overrides.pending_procedural_attempts ?? [],
    discourse_state: overrides.discourse_state ?? {
      stop_until_substantive_content: null,
    },
    mode: overrides.mode ?? "problem_solving",
    updated_at: overrides.updated_at ?? 0,
  };
}

export function createRetrievalScoreFixture(
  overrides: Partial<RetrievedEpisode["scoreBreakdown"]> = {},
): RetrievedEpisode["scoreBreakdown"] {
  return {
    similarity: overrides.similarity ?? 0.8,
    decayedSalience: overrides.decayedSalience ?? 0.4,
    heat: overrides.heat ?? 0.3,
    goalRelevance: overrides.goalRelevance ?? 0,
    valueAlignment: overrides.valueAlignment ?? 0,
    timeRelevance: overrides.timeRelevance ?? 0,
    moodBoost: overrides.moodBoost ?? 0,
    socialRelevance: overrides.socialRelevance ?? 0,
    entityRelevance: overrides.entityRelevance ?? 0,
    suppressionPenalty: overrides.suppressionPenalty ?? 0,
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
    observation_metadata: overrides.observation_metadata ?? null,
    confidence: overrides.confidence ?? 0.5,
    source_episode_ids: overrides.source_episode_ids ?? [createEpisodeId()],
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
    last_verified_at: overrides.last_verified_at ?? nowMs,
    embedding: overrides.embedding ?? Float32Array.from(vector),
    archived: overrides.archived ?? false,
    superseded_by: overrides.superseded_by ?? null,
    status: overrides.status ?? "active",
    corrected_by: overrides.corrected_by ?? null,
    superseded_at: overrides.superseded_at ?? null,
  };
}

export function createSemanticEdgeFixture(overrides: Partial<SemanticEdge> = {}): SemanticEdge {
  const nowMs = overrides.created_at ?? 1_000_000;

  return {
    id: overrides.id ?? createSemanticEdgeId(),
    from_node_id: overrides.from_node_id ?? createSemanticNodeId(),
    to_node_id: overrides.to_node_id ?? createSemanticNodeId(),
    relation: overrides.relation ?? "supports",
    confidence: overrides.confidence ?? 0.7,
    evidence_episode_ids: overrides.evidence_episode_ids ?? [createEpisodeId()],
    created_at: nowMs,
    last_verified_at: overrides.last_verified_at ?? nowMs,
    valid_from: overrides.valid_from ?? nowMs,
    valid_to: overrides.valid_to ?? null,
    invalidated_at: overrides.invalidated_at ?? null,
    invalidated_by_edge_id: overrides.invalidated_by_edge_id ?? null,
    invalidated_by_review_id: overrides.invalidated_by_review_id ?? null,
    invalidated_by_process: overrides.invalidated_by_process ?? null,
    invalidated_reason: overrides.invalidated_reason ?? null,
  };
}

export function createSkillFixture(overrides: Partial<SkillRecord> = {}): SkillRecord {
  const nowMs = overrides.created_at ?? 1_000_000;

  return {
    id: overrides.id ?? createSkillId(),
    applies_when: overrides.applies_when ?? "Debugging a flaky deployment",
    approach: overrides.approach ?? "Compare the failing state with the last known-good state.",
    status: overrides.status ?? "active",
    alpha: overrides.alpha ?? 1,
    beta: overrides.beta ?? 1,
    attempts: overrides.attempts ?? 0,
    successes: overrides.successes ?? 0,
    failures: overrides.failures ?? 0,
    alternatives: overrides.alternatives ?? [],
    superseded_by: overrides.superseded_by ?? [],
    superseded_at: overrides.superseded_at ?? null,
    splitting_at: overrides.splitting_at ?? null,
    last_split_attempt_at: overrides.last_split_attempt_at ?? null,
    split_failure_count: overrides.split_failure_count ?? 0,
    last_split_error: overrides.last_split_error ?? null,
    requires_manual_review: overrides.requires_manual_review ?? false,
    source_episode_ids: overrides.source_episode_ids ?? [createEpisodeId()],
    disclosure_label: overrides.disclosure_label ?? unknownMemoryDisclosureLabel(),
    last_used: overrides.last_used ?? null,
    last_successful: overrides.last_successful ?? null,
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
  };
}
