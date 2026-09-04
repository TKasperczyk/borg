// Shared public and internal Borg composition types used by the facade and setup modules.

import type {
  AutonomyScheduler,
  AutonomyWakesRepository,
  ScheduledWakesRepository,
} from "../autonomy/index.js";
import type {
  AttachmentRepository,
  AttachmentService,
  ImageAttachmentLifecycleService,
  ImagePerceptionRepository,
} from "../attachments/index.js";
import type {
  ChatResponseCatchUpWorker,
  ChatResponseWatermarkCoordinator,
  MessageEnqueuer,
  StreamIngestionCoordinator,
} from "../cognition/ingestion/index.js";
import type { TurnOrchestrator } from "../cognition/index.js";
import type { PromptOverrideRepository } from "../cognition/prompts/override-repository.js";
import type { PromptSurfaceHistoryRepository } from "../cognition/prompts/prompt-surface-history.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { Config } from "../config/index.js";
import type { CorrectionService } from "../correction/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { ExecutiveStepsRepository } from "../executive/index.js";
import type { LLMClient } from "../llm/index.js";
import type { MoodRepository } from "../memory/affective/index.js";
import type { SourceStreamAudienceDisclosureResolver } from "../memory/common/index.js";
import type {
  ActivityRepository,
  LivedExperienceDaySummaryRepository,
} from "../memory/activity/index.js";
import type { ActionRepository } from "../memory/actions/index.js";
import type { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import type { CreatorDirectiveRepository } from "../memory/creator-directives/index.js";
import type { SharedStateRepository } from "../memory/shared-state/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import type { IdentityEventRepository, IdentityService } from "../memory/identity/index.js";
import type { ObservedEventRepository } from "../memory/observed-events/index.js";
import type {
  ProceduralContextStatsRepository,
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
import type {
  SemanticBeliefDependencyRepository,
  SemanticEdgeRepository,
  SemanticGraph,
  SemanticNodeRepository,
  SemanticReviewService,
} from "../memory/semantic/index.js";
import type { ReviewQueueRepository } from "../memory/review-queue/index.js";
import type { SocialRepository } from "../memory/social/index.js";
import type { WorkingMemoryStore } from "../memory/working/index.js";
import type {
  AuditLog,
  MaintenanceOrchestrator,
  MaintenancePlan,
  MaintenanceScheduler,
  OfflineProcess,
  OfflineProcessName,
  OrchestratorResult,
} from "../offline/index.js";
import type {
  AutonomousOutboundPolicy,
  MessageConnector,
  MessageConnectorRegistry,
  OutboundDelivery,
} from "../outbound/index.js";
import type {
  DisclosureRetrievalOptions,
  RetrievalGetEpisodeOptions,
  RetrievalPipeline,
} from "../retrieval/index.js";
import type { SessionsRepository } from "../sessions/index.js";
import type { LanceDbStore } from "../storage/lancedb/index.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import type { StreamEntry, StreamEntryIndexRepository, StreamWriter } from "../stream/index.js";
import type { Clock } from "../util/clock.js";
import type { EntityId, MaintenanceRunId, SessionId } from "../util/ids.js";

export type BorgStreamWriterFactory = (sessionId: SessionId) => StreamWriter;

export type BorgDependencies = {
  config: Config;
  sqlite: SqliteDatabase;
  lance: LanceDbStore;
  entryIndex: StreamEntryIndexRepository;
  sourceStreamAudienceDisclosureResolver: SourceStreamAudienceDisclosureResolver;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository: SemanticNodeRepository;
  semanticEdgeRepository: SemanticEdgeRepository;
  semanticBeliefDependencyRepository: SemanticBeliefDependencyRepository;
  semanticGraph: SemanticGraph;
  semanticReviewService: SemanticReviewService;
  reviewQueueRepository: ReviewQueueRepository;
  identityEventRepository: IdentityEventRepository;
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
  entityRepository: EntityRepository;
  commitmentRepository: CommitmentRepository;
  creatorDirectiveRepository: CreatorDirectiveRepository;
  sharedStateRepository: SharedStateRepository;
  activityRepository: ActivityRepository;
  livedExperienceDaySummaryRepository: LivedExperienceDaySummaryRepository;
  selfDecisionRepository: SelfDecisionRepository;
  trainOfThoughtRepository: TrainOfThoughtRepository;
  observedEventRepository: ObservedEventRepository;
  correctionService: CorrectionService;
  skillRepository: SkillRepository;
  proceduralContextStatsRepository: ProceduralContextStatsRepository;
  proceduralEvidenceRepository: ProceduralEvidenceRepository;
  relationalSlotRepository: RelationalSlotRepository;
  skillSelector: SkillSelector;
  retrievalPipeline: RetrievalPipeline;
  workingMemoryStore: WorkingMemoryStore;
  autonomyWakesRepository: AutonomyWakesRepository;
  scheduledWakesRepository: ScheduledWakesRepository;
  sessionsRepository: SessionsRepository;
  attachmentRepository: AttachmentRepository;
  imagePerceptionRepository: ImagePerceptionRepository;
  imageAttachmentLifecycleService: ImageAttachmentLifecycleService;
  attachmentService: AttachmentService;
  promptOverrideRepository: PromptOverrideRepository;
  promptSurfaceHistoryRepository: PromptSurfaceHistoryRepository;
  outboundConnectorRegistry: MessageConnectorRegistry;
  outboundDelivery: OutboundDelivery;
  autonomousOutboundPolicy: AutonomousOutboundPolicy;
  turnOrchestrator: TurnOrchestrator;
  autonomyScheduler: AutonomyScheduler;
  maintenanceScheduler: MaintenanceScheduler;
  streamIngestionCoordinator?: StreamIngestionCoordinator;
  chatResponseWatermarkCoordinator: ChatResponseWatermarkCoordinator;
  chatResponseCatchUpWorker: ChatResponseCatchUpWorker;
  messageEnqueuer: MessageEnqueuer;
  auditLog: AuditLog;
  maintenanceOrchestrator: MaintenanceOrchestrator;
  offlineProcesses: Record<OfflineProcessName, OfflineProcess>;
  createStreamWriter: BorgStreamWriterFactory;
  llmFactory: () => LLMClient;
  embeddingClient: EmbeddingClient;
  tracer: TurnTracer;
  clock: Clock;
};

export type BorgOpenOptions = {
  config?: Config;
  env?: NodeJS.ProcessEnv;
  dataDir?: string;
  embeddingDimensions?: number;
  embeddingClient?: EmbeddingClient;
  llmClient?: LLMClient;
  clock?: Clock;
  tracer?: TurnTracer;
  onStreamAppend?: (entries: readonly StreamEntry[]) => void;
  outboundConnectors?: readonly MessageConnector[];
  tracerPath?: string;
  /**
   * When true, completed turns trigger best-effort watermark-based episodic
   * extraction. The next turn retries any backlog before retrieval up to the
   * configured pre-turn catch-up bound. Tests and scripted harnesses that
   * use fake LLM response queues should opt out explicitly so extraction
   * does not consume responses out of band.
   */
  liveExtraction?: boolean;
  /**
   * When true, live stream ingestion also classifies sender-stamped user
   * entries with the corrective-preference machinery. This is disabled by
   * default because the full cognition turn loop already performs that work.
   */
  liveCommitmentExtraction?: boolean;
  /** Per-user-entry token accounting cap for live commitment extraction. */
  liveCommitmentExtractionBudget?: number | null;
};

export type BorgDreamOptions = {
  runId?: MaintenanceRunId;
  dryRun?: boolean;
  budget?: number;
  processes?: OfflineProcessName[];
  processOverrides?: Partial<
    Record<
      OfflineProcessName,
      {
        dryRun?: boolean;
        budget?: number;
        params?: Record<string, unknown>;
      }
    >
  >;
};

export type BorgDreamRunner = ((options?: BorgDreamOptions) => Promise<OrchestratorResult>) & {
  plan: (options?: Omit<BorgDreamOptions, "dryRun">) => Promise<MaintenancePlan>;
  preview: (plan: MaintenancePlan) => OrchestratorResult;
  apply: (plan: MaintenancePlan) => Promise<OrchestratorResult>;
  consolidate: (options?: { dryRun?: boolean; budget?: number }) => Promise<OrchestratorResult>;
  reflect: (options?: { dryRun?: boolean; budget?: number }) => Promise<OrchestratorResult>;
  associate: (options?: { dryRun?: boolean; budget?: number }) => Promise<OrchestratorResult>;
  extractSemantics: (options?: {
    dryRun?: boolean;
    budget?: number;
  }) => Promise<OrchestratorResult>;
  curate: (options?: { dryRun?: boolean; budget?: number }) => Promise<OrchestratorResult>;
  oversee: (options?: { dryRun?: boolean; budget?: number }) => Promise<OrchestratorResult>;
  ruminate: (options?: {
    dryRun?: boolean;
    budget?: number;
    maxQuestionsPerRun?: number;
  }) => Promise<OrchestratorResult>;
  narrate: (options?: {
    dryRun?: boolean;
    budget?: number;
    label?: string;
  }) => Promise<OrchestratorResult>;
};

export type BorgEpisodeSearchOptions = Omit<DisclosureRetrievalOptions, "audienceEntityId"> & {
  audience?: string | null;
  audienceEntityId?: EntityId | null;
};

export type BorgEpisodeGetOptions = Omit<RetrievalGetEpisodeOptions, "audienceEntityId"> & {
  audience?: string | null;
  audienceEntityId?: EntityId | null;
};
