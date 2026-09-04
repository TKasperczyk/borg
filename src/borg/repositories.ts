// Builds Borg's repository graph and the cross-repository services that sit on top of it.

import { AutonomyWakesRepository, ScheduledWakesRepository } from "../autonomy/index.js";
import type { AttachmentRepository } from "../attachments/index.js";
import { ImagePerceptionRepository } from "../attachments/index.js";
import type { Config } from "../config/index.js";
import { CorrectionService } from "../correction/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { ExecutiveStepsRepository } from "../executive/index.js";
import type { LLMClient } from "../llm/index.js";
import { MoodRepository } from "../memory/affective/index.js";
import { SourceStreamAudienceDisclosureResolver } from "../memory/common/index.js";
import {
  ActivityRepository,
  LivedExperienceDaySummaryRepository,
} from "../memory/activity/index.js";
import {
  ActionRepository,
  resolveOpenQuestionsForCompletedAction,
} from "../memory/actions/index.js";
import { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import { CreatorDirectiveRepository } from "../memory/creator-directives/index.js";
import { SharedStateRepository } from "../memory/shared-state/index.js";
import { EpisodicRepository } from "../memory/episodic/index.js";
import { IdentityEventRepository, IdentityService } from "../memory/identity/index.js";
import { ObservedEventRepository } from "../memory/observed-events/index.js";
import { SelfDecisionRepository } from "../memory/self-decisions/index.js";
import { TrainOfThoughtRepository } from "../memory/train-of-thought/index.js";
import {
  ProceduralContextStatsRepository,
  ProceduralEvidenceRepository,
  SkillRepository,
  SkillSelector,
} from "../memory/procedural/index.js";
import { RelationalSlotRepository } from "../memory/relational-slots/index.js";
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
  SemanticBeliefDependencyRepository,
  SemanticEdgeRepository,
  SemanticGraph,
  SemanticNodeRepository,
  SemanticReviewService,
} from "../memory/semantic/index.js";
import {
  ReviewQueueRepository,
  ReviewQueueHandlerRegistry,
  createCorrectionReviewHandler,
  registerBuiltinReviewQueueHandlers,
  type ReviewQueueItem,
} from "../memory/review-queue/index.js";
import { SocialRepository } from "../memory/social/index.js";
import { WorkingMemoryStore } from "../memory/working/index.js";
import { RecallStateRepository, RetrievalPipeline } from "../retrieval/index.js";
import { SessionsRepository } from "../sessions/index.js";
import type { LanceDbTable } from "../storage/lancedb/index.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { StreamEntryIndexRepository, StreamWriter, type StreamEntry } from "../stream/index.js";
import type { Clock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import { PromptOverrideRepository } from "../cognition/prompts/override-repository.js";
import { PromptSurfaceHistoryRepository } from "../cognition/prompts/prompt-surface-history.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { BorgDependencies, BorgStreamWriterFactory } from "./types.js";
import {
  backfillSessionStreamEntryIndexAndAttachments,
  backfillStreamEntryIndex,
} from "./reconciliation.js";

export type BorgRepositorySetup = Pick<
  BorgDependencies,
  | "entryIndex"
  | "sourceStreamAudienceDisclosureResolver"
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
  | "actionRepository"
  | "socialRepository"
  | "entityRepository"
  | "commitmentRepository"
  | "creatorDirectiveRepository"
  | "sharedStateRepository"
  | "activityRepository"
  | "livedExperienceDaySummaryRepository"
  | "selfDecisionRepository"
  | "trainOfThoughtRepository"
  | "observedEventRepository"
  | "correctionService"
  | "skillRepository"
  | "proceduralContextStatsRepository"
  | "proceduralEvidenceRepository"
  | "relationalSlotRepository"
  | "skillSelector"
  | "retrievalPipeline"
  | "workingMemoryStore"
  | "autonomyWakesRepository"
  | "scheduledWakesRepository"
  | "sessionsRepository"
  | "attachmentRepository"
  | "imagePerceptionRepository"
  | "promptOverrideRepository"
  | "promptSurfaceHistoryRepository"
> & {
  createStreamWriter: BorgStreamWriterFactory;
  createNonNotifyingStreamWriter: BorgStreamWriterFactory;
};

export type BuildBorgRepositoriesOptions = {
  config: Config;
  sqlite: SqliteDatabase;
  episodesTable: LanceDbTable;
  semanticNodesTable: LanceDbTable;
  openQuestionsTable: LanceDbTable;
  skillsTable: LanceDbTable;
  actionRecordsTable: LanceDbTable;
  imagePerceptionsTable: LanceDbTable;
  observedEventsTable: LanceDbTable;
  embeddingClient: EmbeddingClient;
  llmClient: LLMClient;
  clock: Clock;
  tracer?: TurnTracer;
  attachmentRepository: AttachmentRepository;
  entryIndex?: StreamEntryIndexRepository;
  onStreamAppend?: (entries: readonly StreamEntry[]) => void;
};

export async function buildBorgRepositories(
  options: BuildBorgRepositoriesOptions,
): Promise<BorgRepositorySetup> {
  const { config, sqlite, clock, embeddingClient } = options;
  const autonomyWakesRepository = new AutonomyWakesRepository({
    db: sqlite,
    clock,
  });
  const scheduledWakesRepository = new ScheduledWakesRepository({
    db: sqlite,
    clock,
  });
  const sessionsRepository = new SessionsRepository({
    db: sqlite,
    clock,
  });
  const episodicRepository = new EpisodicRepository({
    table: options.episodesTable,
    db: sqlite,
    clock,
  });
  const imagePerceptionRepository = new ImagePerceptionRepository(
    sqlite,
    options.imagePerceptionsTable,
  );
  await episodicRepository.reconcileCrossStoreState();

  const entryIndex =
    options.entryIndex ??
    new StreamEntryIndexRepository({
      db: sqlite,
      dataDir: config.dataDir,
    });
  await backfillStreamEntryIndex({
    dataDir: config.dataDir,
    entryIndex,
    attachmentRepository: options.attachmentRepository,
  });
  const repairSessionStreamEntryIndex = (sessionId: Parameters<BorgStreamWriterFactory>[0]) =>
    backfillSessionStreamEntryIndexAndAttachments({
      dataDir: config.dataDir,
      sessionId,
      entryIndex,
      attachmentRepository: options.attachmentRepository,
    });

  const createStreamWriter = (sessionId: Parameters<BorgStreamWriterFactory>[0]) =>
    new StreamWriter({
      dataDir: config.dataDir,
      sessionId,
      clock,
      entryIndex,
      repairSession: repairSessionStreamEntryIndex,
      onAppend: options.onStreamAppend,
    });
  const createNonNotifyingStreamWriter = (sessionId: Parameters<BorgStreamWriterFactory>[0]) =>
    new StreamWriter({
      dataDir: config.dataDir,
      sessionId,
      clock,
      entryIndex,
      repairSession: repairSessionStreamEntryIndex,
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
  const sourceStreamAudienceDisclosureResolver = new SourceStreamAudienceDisclosureResolver({
    dataDir: config.dataDir,
    entryIndex,
    sessionsRepository,
    entityRepository,
  });
  const creatorDirectiveRepository = new CreatorDirectiveRepository({
    db: sqlite,
    clock,
  });
  const sharedStateRepository = new SharedStateRepository({
    db: sqlite,
    clock,
  });
  const activityRepository = new ActivityRepository({
    db: sqlite,
    clock,
  });
  const livedExperienceDaySummaryRepository = new LivedExperienceDaySummaryRepository({
    db: sqlite,
    clock,
  });
  const selfDecisionRepository = new SelfDecisionRepository({
    db: sqlite,
    clock,
  });
  const trainOfThoughtRepository = new TrainOfThoughtRepository({
    db: sqlite,
    clock,
  });
  const observedEventRepository = new ObservedEventRepository({
    db: sqlite,
    table: options.observedEventsTable,
    embeddingClient,
    clock,
    onEmbeddingFailure: (error, details) => {
      const writer = createDefaultStreamWriter();
      void appendInternalFailureEvent(writer, "observed_event_embedding", error, {
        operation: details.operation,
        eventId: details.eventId,
      }).finally(() => {
        writer.close();
      });
    },
  });
  if (options.tracer?.enabled === true) {
    options.tracer.emit("observed_event_embedding_backfill.started", {
      turnId: "startup",
      mode: "background",
      recall_consistency: "topic_recall_eventual_until_complete",
    });
  }
  void observedEventRepository
    .backfillMissingEmbeddings()
    .then((report) => {
      if (options.tracer?.enabled === true) {
        options.tracer.emit("observed_event_embedding_backfill.completed", {
          turnId: "startup",
          mode: "background",
          scanned: report.scanned,
          embedded: report.embedded,
          skipped: report.skipped,
          failed: report.failed,
        });
      }
    })
    .catch((error) => {
      if (options.tracer?.enabled === true) {
        options.tracer.emit("observed_event_embedding_backfill.failed", {
          turnId: "startup",
          mode: "background",
          ...(options.tracer.includePayloads
            ? { error: error instanceof Error ? error.message : String(error) }
            : {}),
        });
      }
      const writer = createDefaultStreamWriter();
      void appendInternalFailureEvent(writer, "observed_event_embedding_backfill", error).finally(
        () => {
          writer.close();
        },
      );
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
  registerBuiltinReviewQueueHandlers(reviewHandlers);
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
    tracer: options.tracer,
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
  const actionRepository = new ActionRepository({
    table: options.actionRecordsTable,
    db: sqlite,
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
    db: sqlite,
    clock,
  });
  const proceduralContextStatsRepository = new ProceduralContextStatsRepository({
    db: sqlite,
    clock,
  });
  const relationalSlotRepository = new RelationalSlotRepository({
    db: sqlite,
    clock,
  });
  const skillSelector = new SkillSelector({
    repository: skillRepository,
    contextStatsRepository: proceduralContextStatsRepository,
    minSimilarity: config.procedural.skillSelectionMinSimilarity,
  });
  const recallStateRepository = new RecallStateRepository({
    db: sqlite,
    clock,
  });
  const retrievalPipeline = new RetrievalPipeline({
    embeddingClient,
    llmClient: options.llmClient,
    recallExpansionModel: config.anthropic.models.recallExpansion,
    recallExpansionTimeoutMs: config.retrieval.recallExpansionTimeoutMs,
    episodicRepository,
    semanticNodeRepository,
    semanticEdgeRepository,
    semanticGraph,
    reviewQueueRepository: createdReviewQueueRepository,
    openQuestionsRepository,
    entityRepository,
    commitmentRepository,
    recallStateRepository,
    dataDir: config.dataDir,
    entryIndex,
    sourceStreamAudienceDisclosureResolver,
    clock,
    tracer: options.tracer,
    semanticUnderReviewMultiplier: config.retrieval.semantic.underReviewMultiplier,
    semanticStatusMultipliers: config.retrieval.semantic.statusMultipliers,
    semanticOverfetchMultiplier: config.retrieval.semanticOverfetchMultiplier,
    lexicalFusionEnabled: config.retrieval.lexicalFusion.enabled,
    imagePerceptionRepository,
    maxRetrievedImageRefs: config.attachments.maxRetrievedImageRefs,
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
    sourceStreamAudienceDisclosureResolver,
    reviewQueueRepository: createdReviewQueueRepository,
    identityService,
    identityEventRepository,
  });
  applyCorrectionReview = (item) => correctionService.applyCorrectionReview(item);
  const workingMemoryStore = new WorkingMemoryStore({
    dataDir: config.dataDir,
    clock,
  });
  const promptOverrideRepository = new PromptOverrideRepository(sqlite, clock);
  const promptSurfaceHistoryRepository = new PromptSurfaceHistoryRepository({
    db: sqlite,
    clock,
  });
  try {
    promptSurfaceHistoryRepository.observeCurrent();
  } catch (error) {
    console.error("Prompt surface history observation failed; continuing startup", {
      cause: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    });
  }
  return {
    entryIndex,
    sourceStreamAudienceDisclosureResolver,
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
    actionRepository,
    socialRepository,
    entityRepository,
    commitmentRepository,
    creatorDirectiveRepository,
    sharedStateRepository,
    activityRepository,
    livedExperienceDaySummaryRepository,
    selfDecisionRepository,
    trainOfThoughtRepository,
    observedEventRepository,
    correctionService,
    skillRepository,
    proceduralContextStatsRepository,
    proceduralEvidenceRepository,
    relationalSlotRepository,
    skillSelector,
    retrievalPipeline,
    workingMemoryStore,
    autonomyWakesRepository,
    scheduledWakesRepository,
    sessionsRepository,
    attachmentRepository: options.attachmentRepository,
    imagePerceptionRepository,
    promptOverrideRepository,
    promptSurfaceHistoryRepository,
    createStreamWriter,
    createNonNotifyingStreamWriter,
  };
}
