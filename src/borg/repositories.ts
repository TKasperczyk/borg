// Builds Borg's repository graph and the cross-repository services that sit on top of it.

import type { Config } from "../config/index.js";
import { CorrectionService } from "../correction/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { LLMClient } from "../llm/index.js";
import { MoodRepository } from "../memory/affective/index.js";
import { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import { EpisodicRepository } from "../memory/episodic/index.js";
import { IdentityEventRepository, IdentityService } from "../memory/identity/index.js";
import { SkillRepository, SkillSelector } from "../memory/procedural/index.js";
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
  ReviewQueueRepository,
  SemanticEdgeRepository,
  SemanticGraph,
  SemanticNodeRepository,
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
import type { BorgDependencies, BorgStreamWriterFactory } from "./types.js";
import { backfillStreamEntryIndex } from "./reconciliation.js";

export type BorgRepositorySetup = Pick<
  BorgDependencies,
  | "entryIndex"
  | "episodicRepository"
  | "semanticNodeRepository"
  | "semanticEdgeRepository"
  | "semanticGraph"
  | "reviewQueueRepository"
  | "identityEventRepository"
  | "identityService"
  | "valuesRepository"
  | "goalsRepository"
  | "traitsRepository"
  | "autobiographicalRepository"
  | "growthMarkersRepository"
  | "openQuestionsRepository"
  | "moodRepository"
  | "socialRepository"
  | "entityRepository"
  | "commitmentRepository"
  | "correctionService"
  | "skillRepository"
  | "skillSelector"
  | "retrievalPipeline"
  | "workingMemoryStore"
> & {
  createStreamWriter: BorgStreamWriterFactory;
};

export type BuildBorgRepositoriesOptions = {
  config: Config;
  sqlite: SqliteDatabase;
  episodesTable: LanceDbTable;
  semanticNodesTable: LanceDbTable;
  skillsTable: LanceDbTable;
  embeddingClient: EmbeddingClient;
  clock: Clock;
  getDeferredLlm: () => LLMClient | undefined;
};

function quarterLabel(timestamp: number): string {
  const date = new Date(timestamp);
  const year = date.getUTCFullYear();
  const quarter = Math.floor(date.getUTCMonth() / 3) + 1;
  return `${year}-Q${quarter}`;
}

export async function buildBorgRepositories(
  options: BuildBorgRepositoriesOptions,
): Promise<BorgRepositorySetup> {
  const { config, sqlite, clock, embeddingClient } = options;
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
  // The semantic repo only uses the LLM lazily at duplicate-review time, so
  // this getter preserves Borg.open's startup ordering while allowing the
  // factory to be resolved later in the composition flow.
  const semanticNodeRepository = new SemanticNodeRepository({
    table: options.semanticNodesTable,
    db: sqlite,
    clock,
    enqueueReview,
    get llmClient(): LLMClient | undefined {
      return options.getDeferredLlm();
    },
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

  if (config.self.autoBootstrapPeriod && autobiographicalRepository.currentPeriod() === null) {
    const nowMs = clock.now();
    autobiographicalRepository.upsertPeriod({
      label: quarterLabel(nowMs),
      start_ts: nowMs,
      end_ts: null,
      narrative: "",
      key_episode_ids: [],
      themes: [],
      provenance: {
        kind: "system",
      },
    });
  }

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
  const createdReviewQueueRepository = new ReviewQueueRepository({
    db: sqlite,
    clock,
    episodicRepository,
    semanticNodeRepository,
    valuesRepository,
    goalsRepository,
    traitsRepository,
    autobiographicalRepository,
    commitmentRepository,
    identityService,
    applyCorrection: (item) => {
      if (applyCorrectionReview === undefined) {
        throw new Error("Correction service not initialized");
      }

      return applyCorrectionReview(item);
    },
    onEnqueue: (item) => enqueueOpenQuestionForReview(openQuestionsRepository, item),
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
  const skillSelector = new SkillSelector({
    repository: skillRepository,
  });
  const retrievalPipeline = new RetrievalPipeline({
    embeddingClient,
    episodicRepository,
    semanticNodeRepository,
    semanticGraph,
    openQuestionsRepository,
    dataDir: config.dataDir,
    entryIndex,
    clock,
  });
  const correctionService = new CorrectionService({
    config,
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
    semanticGraph,
    reviewQueueRepository: createdReviewQueueRepository,
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
    correctionService,
    skillRepository,
    skillSelector,
    retrievalPipeline,
    workingMemoryStore,
    createStreamWriter,
  };
}
