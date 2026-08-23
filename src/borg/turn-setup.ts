// Wires the per-turn cognitive orchestrator and its session-scoped dependencies.

import type {
  ChatResponseWatermarkCoordinator,
  StreamIngestionCoordinator,
} from "../cognition/ingestion/index.js";
import type {
  AttachmentRepository,
  AttachmentService,
  ImagePerceptionRepository,
  ImagePerceptionService,
} from "../attachments/index.js";
import type { SessionLock } from "../cognition/index.js";
import { Reflector, TurnOrchestrator } from "../cognition/index.js";
import { TurnContextCompiler } from "../cognition/recency/index.js";
import type { Config } from "../config/index.js";
import type { ExecutiveStepsRepository } from "../executive/index.js";
import type { LLMClient } from "../llm/index.js";
import type { MoodRepository } from "../memory/affective/index.js";
import type {
  ActivityRepository,
  LivedExperienceDaySummaryRepository,
} from "../memory/activity/index.js";
import type { ActionRepository } from "../memory/actions/index.js";
import type { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import type { CreatorDirectiveRepository } from "../memory/creator-directives/index.js";
import type { SharedStateRepository } from "../memory/shared-state/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import type { IdentityService } from "../memory/identity/index.js";
import type { ObservedEventRepository } from "../memory/observed-events/index.js";
import type {
  ProceduralEvidenceRepository,
  SkillRepository,
  SkillSelector,
} from "../memory/procedural/index.js";
import type { RelationalSlotRepository } from "../memory/relational-slots/index.js";
import type { SelfDecisionRepository } from "../memory/self-decisions/index.js";
import type { TrainOfThoughtRepository } from "../memory/train-of-thought/index.js";
import type {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
} from "../memory/self/index.js";
import type { SemanticNodeRepository } from "../memory/semantic/index.js";
import type { ReviewQueueRepository } from "../memory/review-queue/index.js";
import type { SocialRepository } from "../memory/social/index.js";
import type { WorkingMemoryStore } from "../memory/working/index.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import type { AutonomousOutboundPolicy, OutboundDelivery } from "../outbound/index.js";
import type { SessionSourceType, SessionsRepository } from "../sessions/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { ToolDispatcher } from "../tools/index.js";
import type { Clock } from "../util/clock.js";
import type { PromptOverrideRepository } from "../cognition/prompts/override-repository.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { BorgStreamWriterFactory } from "./types.js";
import type { StreamEntryIndexRepository } from "../stream/index.js";
import type { AutonomySchedulerDescription } from "../autonomy/index.js";

export type BuildTurnOrchestratorOptions = {
  config: Config;
  retrievalPipeline: RetrievalPipeline;
  embeddingClient: EmbeddingClient;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository: SemanticNodeRepository;
  entityRepository: EntityRepository;
  commitmentRepository: CommitmentRepository;
  creatorDirectiveRepository: CreatorDirectiveRepository;
  sharedStateRepository: SharedStateRepository;
  activityRepository: ActivityRepository;
  livedExperienceDaySummaryRepository: LivedExperienceDaySummaryRepository;
  selfDecisionRepository: SelfDecisionRepository;
  trainOfThoughtRepository: TrainOfThoughtRepository;
  observedEventRepository: ObservedEventRepository;
  reviewQueueRepository: ReviewQueueRepository;
  identityService: IdentityService;
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  autobiographicalRepository: AutobiographicalRepository;
  growthMarkersRepository: GrowthMarkersRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  executiveStepsRepository: ExecutiveStepsRepository;
  moodRepository: MoodRepository;
  actionRepository: ActionRepository;
  socialRepository: SocialRepository;
  skillRepository: SkillRepository;
  proceduralEvidenceRepository: ProceduralEvidenceRepository;
  relationalSlotRepository: RelationalSlotRepository;
  skillSelector: SkillSelector;
  workingMemoryStore: WorkingMemoryStore;
  llmFactory: () => LLMClient;
  toolDispatcher: ToolDispatcher;
  sessionLock: SessionLock;
  streamIngestionCoordinator?: StreamIngestionCoordinator;
  chatResponseWatermarkCoordinator?: ChatResponseWatermarkCoordinator;
  outboundDelivery?: Pick<OutboundDelivery, "deliver">;
  autonomousOutboundPolicy?: Pick<AutonomousOutboundPolicy, "promptContext">;
  autonomySchedulerStateProvider?: () => Promise<AutonomySchedulerDescription | null>;
  outboundSourceTypes?: readonly SessionSourceType[];
  createStreamWriter: BorgStreamWriterFactory;
  entryIndex?: StreamEntryIndexRepository;
  attachmentService: AttachmentService;
  attachmentRepository: AttachmentRepository;
  imagePerceptionRepository: ImagePerceptionRepository;
  imagePerceptionService?: ImagePerceptionService;
  clock: Clock;
  tracer?: TurnTracer;
  promptOverrideRepository?: PromptOverrideRepository;
  sessionsRepository?: SessionsRepository;
};

export function buildTurnOrchestrator(options: BuildTurnOrchestratorOptions): TurnOrchestrator {
  return new TurnOrchestrator({
    config: options.config,
    retrievalPipeline: options.retrievalPipeline,
    embeddingClient: options.embeddingClient,
    episodicRepository: options.episodicRepository,
    semanticNodeRepository: options.semanticNodeRepository,
    entityRepository: options.entityRepository,
    commitmentRepository: options.commitmentRepository,
    creatorDirectiveRepository: options.creatorDirectiveRepository,
    sharedStateRepository: options.sharedStateRepository,
    activityRepository: options.activityRepository,
    livedExperienceDaySummaryRepository: options.livedExperienceDaySummaryRepository,
    selfDecisionRepository: options.selfDecisionRepository,
    trainOfThoughtRepository: options.trainOfThoughtRepository,
    observedEventRepository: options.observedEventRepository,
    identityService: options.identityService,
    reviewQueueRepository: options.reviewQueueRepository,
    valuesRepository: options.valuesRepository,
    goalsRepository: options.goalsRepository,
    traitsRepository: options.traitsRepository,
    autobiographicalRepository: options.autobiographicalRepository,
    growthMarkersRepository: options.growthMarkersRepository,
    openQuestionsRepository: options.openQuestionsRepository,
    executiveStepsRepository: options.executiveStepsRepository,
    moodRepository: options.moodRepository,
    actionRepository: options.actionRepository,
    socialRepository: options.socialRepository,
    skillSelector: options.skillSelector,
    relationalSlotRepository: options.relationalSlotRepository,
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
        actionRepository: options.actionRepository,
        identityService: options.identityService,
        openQuestionsRepository: options.openQuestionsRepository,
        reviewQueueRepository: options.reviewQueueRepository,
        skillRepository: options.skillRepository,
        proceduralEvidenceRepository: options.proceduralEvidenceRepository,
        tracer: options.tracer,
      }),
    toolDispatcher: options.toolDispatcher,
    sessionLock: options.sessionLock,
    clock: options.clock,
    tracer: options.tracer,
    createStreamWriter: options.createStreamWriter,
    entryIndex: options.entryIndex,
    attachmentService: options.attachmentService,
    attachmentRepository: options.attachmentRepository,
    imagePerceptionRepository: options.imagePerceptionRepository,
    imagePerceptionService: options.imagePerceptionService,
    // Explicit so borg.ts wires a single compiler instance per process;
    // turn-orchestrator.ts falls back to defaults if omitted, but doing
    // it here makes the configuration visible at the composition root.
    turnContextCompiler: new TurnContextCompiler(),
    ...(options.streamIngestionCoordinator === undefined
      ? {}
      : { streamIngestionCoordinator: options.streamIngestionCoordinator }),
    ...(options.chatResponseWatermarkCoordinator === undefined
      ? {}
      : { chatResponseWatermarkCoordinator: options.chatResponseWatermarkCoordinator }),
    ...(options.outboundDelivery === undefined
      ? {}
      : { outboundDelivery: options.outboundDelivery }),
    ...(options.autonomousOutboundPolicy === undefined
      ? {}
      : { autonomousOutboundPolicy: options.autonomousOutboundPolicy }),
    ...(options.autonomySchedulerStateProvider === undefined
      ? {}
      : { autonomySchedulerStateProvider: options.autonomySchedulerStateProvider }),
    ...(options.outboundSourceTypes === undefined
      ? {}
      : { outboundSourceTypes: options.outboundSourceTypes }),
    ...(options.promptOverrideRepository === undefined
      ? {}
      : { promptOverrideRepository: options.promptOverrideRepository }),
    ...(options.sessionsRepository === undefined
      ? {}
      : { sessionsRepository: options.sessionsRepository }),
  });
}
