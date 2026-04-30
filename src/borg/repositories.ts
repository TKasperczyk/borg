// Builds Borg's repository graph and the cross-repository services that sit on top of it.

import { AutonomyWakesRepository } from "../autonomy/index.js";
import type { Config } from "../config/index.js";
import { CorrectionService } from "../correction/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { ExecutiveStepsRepository } from "../executive/index.js";
import type { LLMClient } from "../llm/index.js";
import { MoodRepository } from "../memory/affective/index.js";
import { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import { EpisodicRepository } from "../memory/episodic/index.js";
import { IdentityEventRepository, IdentityService } from "../memory/identity/index.js";
import {
  ProceduralContextStatsRepository,
  ProceduralEvidenceRepository,
  SkillRepository,
  SkillSelector,
} from "../memory/procedural/index.js";
import {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
} from "../memory/self/index.js";
import {
  appendInternalFailureEvent,
  appendOpenQuestionHookFailureEvent,
  enqueueOpenQuestionForReview,
} from "../memory/self/review-open-question-hook.js";
import {
  ReviewOpenQuestionExtractor,
  type ReviewOpenQuestionExtractorDegradedEvent,
} from "../memory/self/review-open-question-extractor.js";
import {
  ReviewQueueRepository,
  ReviewQueueHandlerRegistry,
  SemanticBeliefDependencyRepository,
  SemanticEdgeRepository,
  SemanticGraph,
  SemanticNodeRepository,
  SemanticReviewService,
  createCorrectionReviewHandler,
  type ReviewQueueItem,
} from "../memory/semantic/index.js";
import { SocialRepository } from "../memory/social/index.js";
import { WorkingMemoryStore } from "../memory/working/index.js";
import { RetrievalPipeline } from "../retrieval/index.js";
import type { LanceDbTable } from "../storage/lancedb/index.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { StreamEntryIndexRepository, StreamWriter } from "../stream/index.js";
import type { Clock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import type { TurnTracer } from "../cognition/tracing/tracer.js";
import type { BorgDependencies, BorgStreamWriterFactory } from "./types.js";
import { backfillStreamEntryIndex } from "./reconciliation.js";

export type BorgRepositorySetup = Pick<
  BorgDependencies,
  | "entryIndex"
  | "episodicRepository"
  | "semanticNodeRepository"
  | "semanticEdgeRepository"
  | "semanticBeliefDependencyRepository"
  | "semanticGraph"
  | "semanticReviewService"
  | "reviewQueueRepository"
  | "identityEventRepository"
  | "identityService"
  | "valuesRepository"
  | "goalsRepository"
  | "traitsRepository"
  | "autobiographicalRepository"
  | "growthMarkersRepository"
  | "openQuestionsRepository"
  | "executiveStepsRepository"
  | "moodRepository"
  | "socialRepository"
  | "entityRepository"
  | "commitmentRepository"
  | "correctionService"
  | "skillRepository"
  | "proceduralContextStatsRepository"
  | "proceduralEvidenceRepository"
  | "skillSelector"
  | "retrievalPipeline"
  | "workingMemoryStore"
  | "autonomyWakesRepository"
> & {
  createStreamWriter: BorgStreamWriterFactory;
};

export type BuildBorgRepositoriesOptions = {
  config: Config;
  sqlite: SqliteDatabase;
  episodesTable: LanceDbTable;
  semanticNodesTable: LanceDbTable;
  openQuestionsTable: LanceDbTable;
  skillsTable: LanceDbTable;
  embeddingClient: EmbeddingClient;
  llmClient: LLMClient;
  clock: Clock;
  tracer?: TurnTracer;
};

export async function buildBorgRepositories(
  options: BuildBorgRepositoriesOptions,
): Promise<BorgRepositorySetup> {
  const { config, sqlite, clock, embeddingClient } = options;
  const autonomyWakesRepository = new AutonomyWakesRepository({
    db: sqlite,
    clock,
  });
  const episodicRepository = new EpisodicRepository({
    table: options.episodesTable,
    db: sqlite,
    clock,
  });
  await episodicRepository.reconcileCrossStoreState();

  const entryIndex = new StreamEntryIndexRepository({
    db: sqlite,
    dataDir: config.dataDir,
  });
  await backfillStreamEntryIndex({
    dataDir: config.dataDir,
    entryIndex,
  });

  const createStreamWriter = (sessionId: Parameters<BorgStreamWriterFactory>[0]) =>
    new StreamWriter({
      dataDir: config.dataDir,
      sessionId,
      clock,
      entryIndex,
    });
  const createDefaultStreamWriter = () => createStreamWriter(DEFAULT_SESSION_ID);
  let reviewQueueRepository: ReviewQueueRepository | undefined;
  let applyCorrectionReview: ((item: ReviewQueueItem) => Promise<void>) | undefined;
  const openQuestionsRepository = new OpenQuestionsRepository({
    db: sqlite,
    table: options.openQuestionsTable,
    embeddingClient,
    clock,
    onEmbeddingFailure: (error, details) => {
      const writer = createDefaultStreamWriter();
      void appendInternalFailureEvent(writer, "open_question_embedding", error, {
        operation: details.operation,
        questionId: details.questionId,
      }).finally(() => {
        writer.close();
      });
    },
  });
  void openQuestionsRepository.backfillMissingEmbeddings().catch((error) => {
    const writer = createDefaultStreamWriter();
    void appendInternalFailureEvent(writer, "open_question_embedding_backfill", error).finally(
      () => {
        writer.close();
      },
    );
  });
  const executiveStepsRepository = new ExecutiveStepsRepository({
    db: sqlite,
    clock,
  });
  const moodRepository = new MoodRepository({
    db: sqlite,
    clock,
    defaultHalfLifeHours: config.affective.moodHalfLifeHours,
    incomingWeight: config.affective.incomingMoodWeight,
  });
  const enqueueReview = (input: Parameters<ReviewQueueRepository["enqueue"]>[0]) => {
    return reviewQueueRepository?.enqueue(input);
  };
  const semanticNodeRepository = new SemanticNodeRepository({
    table: options.semanticNodesTable,
    db: sqlite,
    clock,
  });
  const semanticReviewService = new SemanticReviewService({
    nodeRepository: semanticNodeRepository,
    enqueueReview,
    llmClient: options.llmClient,
    contradictionJudgeModel: config.anthropic.models.background,
    onDuplicateReviewError: (error) => {
      const writer = createDefaultStreamWriter();
      void appendInternalFailureEvent(writer, "semantic_duplicate_review", error).finally(() => {
        writer.close();
      });
    },
  });
  const semanticEdgeRepository = new SemanticEdgeRepository({
    db: sqlite,
    clock,
    enqueueReview,
  });
  const semanticBeliefDependencyRepository = new SemanticBeliefDependencyRepository({
    db: sqlite,
    clock,
  });
  const semanticGraph = new SemanticGraph({
    nodeRepository: semanticNodeRepository,
    edgeRepository: semanticEdgeRepository,
  });
  const identityEventRepository = new IdentityEventRepository({
    db: sqlite,
    clock,
  });
  const valuesRepository = new ValuesRepository({
    db: sqlite,
    clock,
    identityEventRepository,
  });
  const goalsRepository = new GoalsRepository({
    db: sqlite,
    clock,
    identityEventRepository,
    executiveStepsRepository,
  });
  const traitsRepository = new TraitsRepository({
    db: sqlite,
    clock,
    identityEventRepository,
  });
  const autobiographicalRepository = new AutobiographicalRepository({
    db: sqlite,
    clock,
  });

  const growthMarkersRepository = new GrowthMarkersRepository({
    db: sqlite,
    clock,
  });
  const entityRepository = new EntityRepository({
    db: sqlite,
    clock,
  });
  const socialRepository = new SocialRepository({
    db: sqlite,
    clock,
  });
  const commitmentRepository = new CommitmentRepository({
    db: sqlite,
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
  const reportReviewOpenQuestionExtractorDegraded = (
    event: ReviewOpenQuestionExtractorDegradedEvent,
  ) => {
    const writer = createDefaultStreamWriter();
    const { error, ...details } = event;

    return appendInternalFailureEvent(
      writer,
      "review_open_question_extractor",
      error ?? event.reason,
      details,
    ).finally(() => {
      writer.close();
    });
  };
  const reviewOpenQuestionExtractor = new ReviewOpenQuestionExtractor({
    llmClient: options.llmClient,
    model: config.anthropic.models.background,
    onDegraded: reportReviewOpenQuestionExtractorDegraded,
  });
  const reviewHandlers = new ReviewQueueHandlerRegistry();
  reviewHandlers.register(
    createCorrectionReviewHandler({
      applyCorrection: (item) => {
        if (applyCorrectionReview === undefined) {
          throw new Error("Correction service not initialized");
        }

        return applyCorrectionReview(item);
      },
    }),
  );
  const createdReviewQueueRepository = new ReviewQueueRepository({
    db: sqlite,
    clock,
    handlers: reviewHandlers,
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
    onEnqueue: (item) =>
      enqueueOpenQuestionForReview(identityService, item, {
        extractor: reviewOpenQuestionExtractor,
      }),
    onEnqueueError: (error) => {
      const writer = createDefaultStreamWriter();
      void appendOpenQuestionHookFailureEvent(writer, "review_queue_open_question", error).finally(
        () => {
          writer.close();
        },
      );
    },
  });
  reviewQueueRepository = createdReviewQueueRepository;
  const skillRepository = new SkillRepository({
    table: options.skillsTable,
    db: sqlite,
    embeddingClient,
    clock,
  });
  const proceduralEvidenceRepository = new ProceduralEvidenceRepository({
    db: sqlite,
    clock,
  });
  const proceduralContextStatsRepository = new ProceduralContextStatsRepository({
    db: sqlite,
    clock,
  });
  const skillSelector = new SkillSelector({
    repository: skillRepository,
    contextStatsRepository: proceduralContextStatsRepository,
    minSimilarity: config.procedural.skillSelectionMinSimilarity,
  });
  const retrievalPipeline = new RetrievalPipeline({
    embeddingClient,
    episodicRepository,
    semanticNodeRepository,
    semanticGraph,
    reviewQueueRepository: createdReviewQueueRepository,
    openQuestionsRepository,
    entityRepository,
    dataDir: config.dataDir,
    entryIndex,
    clock,
    tracer: options.tracer,
    semanticUnderReviewMultiplier: config.retrieval.semantic.underReviewMultiplier,
  });
  const correctionService = new CorrectionService({
    config,
    db: sqlite,
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
    reviewQueueRepository: createdReviewQueueRepository,
    identityService,
    identityEventRepository,
  });
  applyCorrectionReview = (item) => correctionService.applyCorrectionReview(item);
  const workingMemoryStore = new WorkingMemoryStore({
    dataDir: config.dataDir,
    clock,
  });
  return {
    entryIndex,
    episodicRepository,
    semanticNodeRepository,
    semanticEdgeRepository,
    semanticBeliefDependencyRepository,
    semanticGraph,
    semanticReviewService,
    reviewQueueRepository: createdReviewQueueRepository,
    identityEventRepository,
    identityService,
    valuesRepository,
    goalsRepository,
    traitsRepository,
    autobiographicalRepository,
    growthMarkersRepository,
    openQuestionsRepository,
    executiveStepsRepository,
    moodRepository,
    socialRepository,
    entityRepository,
    commitmentRepository,
    correctionService,
    skillRepository,
    proceduralContextStatsRepository,
    proceduralEvidenceRepository,
    skillSelector,
    retrievalPipeline,
    workingMemoryStore,
    autonomyWakesRepository,
    createStreamWriter,
  };
}
