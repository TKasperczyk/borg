// Wires the per-turn cognitive orchestrator and its session-scoped dependencies.

import type { StreamIngestionCoordinator } from "../cognition/ingestion/index.js";
import type { SessionLock } from "../cognition/index.js";
import { Reflector, TurnOrchestrator } from "../cognition/index.js";
import { TurnContextCompiler } from "../cognition/recency/index.js";
import type { Config } from "../config/index.js";
import type { ExecutiveStepsRepository } from "../executive/index.js";
import type { LLMClient } from "../llm/index.js";
import type { MoodRepository } from "../memory/affective/index.js";
import type { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import type { IdentityService } from "../memory/identity/index.js";
import type {
  ProceduralEvidenceRepository,
  SkillRepository,
  SkillSelector,
} from "../memory/procedural/index.js";
import type {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  TraitsRepository,
  ValuesRepository,
} from "../memory/self/index.js";
import type { ReviewQueueRepository } from "../memory/semantic/index.js";
import type { SocialRepository } from "../memory/social/index.js";
import type { WorkingMemoryStore } from "../memory/working/index.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { ToolDispatcher } from "../tools/index.js";
import type { Clock } from "../util/clock.js";
import type { TurnTracer } from "../cognition/tracing/tracer.js";
import type { BorgStreamWriterFactory } from "./types.js";

export type BuildTurnOrchestratorOptions = {
  config: Config;
  retrievalPipeline: RetrievalPipeline;
  embeddingClient: EmbeddingClient;
  episodicRepository: EpisodicRepository;
  entityRepository: EntityRepository;
  commitmentRepository: CommitmentRepository;
  reviewQueueRepository: ReviewQueueRepository;
  identityService: IdentityService;
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  autobiographicalRepository: AutobiographicalRepository;
  growthMarkersRepository: GrowthMarkersRepository;
  executiveStepsRepository: ExecutiveStepsRepository;
  moodRepository: MoodRepository;
  socialRepository: SocialRepository;
  skillRepository: SkillRepository;
  proceduralEvidenceRepository: ProceduralEvidenceRepository;
  skillSelector: SkillSelector;
  workingMemoryStore: WorkingMemoryStore;
  llmFactory: () => LLMClient;
  toolDispatcher: ToolDispatcher;
  sessionLock: SessionLock;
  streamIngestionCoordinator?: StreamIngestionCoordinator;
  createStreamWriter: BorgStreamWriterFactory;
  clock: Clock;
  tracer?: TurnTracer;
};

export function buildTurnOrchestrator(options: BuildTurnOrchestratorOptions): TurnOrchestrator {
  return new TurnOrchestrator({
    config: options.config,
    retrievalPipeline: options.retrievalPipeline,
    embeddingClient: options.embeddingClient,
    episodicRepository: options.episodicRepository,
    entityRepository: options.entityRepository,
    commitmentRepository: options.commitmentRepository,
    reviewQueueRepository: options.reviewQueueRepository,
    valuesRepository: options.valuesRepository,
    goalsRepository: options.goalsRepository,
    traitsRepository: options.traitsRepository,
    autobiographicalRepository: options.autobiographicalRepository,
    growthMarkersRepository: options.growthMarkersRepository,
    executiveStepsRepository: options.executiveStepsRepository,
    moodRepository: options.moodRepository,
    socialRepository: options.socialRepository,
    skillSelector: options.skillSelector,
    workingMemoryStore: options.workingMemoryStore,
    llmFactory: options.llmFactory,
    createReflector: (llmClient) =>
      new Reflector({
        clock: options.clock,
        llmClient,
        model: options.config.anthropic.models.background,
        episodicRepository: options.episodicRepository,
        goalsRepository: options.goalsRepository,
        traitsRepository: options.traitsRepository,
        executiveStepsRepository: options.executiveStepsRepository,
        identityService: options.identityService,
        reviewQueueRepository: options.reviewQueueRepository,
        skillRepository: options.skillRepository,
        proceduralEvidenceRepository: options.proceduralEvidenceRepository,
      }),
    toolDispatcher: options.toolDispatcher,
    sessionLock: options.sessionLock,
    clock: options.clock,
    tracer: options.tracer,
    createStreamWriter: options.createStreamWriter,
    // Explicit so borg.ts wires a single compiler instance per process;
    // turn-orchestrator.ts falls back to defaults if omitted, but doing
    // it here makes the configuration visible at the composition root.
    turnContextCompiler: new TurnContextCompiler(),
    ...(options.streamIngestionCoordinator === undefined
      ? {}
      : { streamIngestionCoordinator: options.streamIngestionCoordinator }),
  });
}
