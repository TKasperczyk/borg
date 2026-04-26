// Shared public and internal Borg composition types used by the facade and setup modules.

import type { AutonomyScheduler } from "../autonomy/index.js";
import type { StreamIngestionCoordinator } from "../cognition/ingestion/index.js";
import type { TurnOrchestrator } from "../cognition/index.js";
import type { Config } from "../config/index.js";
import type { CorrectionService } from "../correction/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { LLMClient } from "../llm/index.js";
import type { MoodRepository } from "../memory/affective/index.js";
import type { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import type { IdentityEventRepository, IdentityService } from "../memory/identity/index.js";
import type { SkillRepository, SkillSelector } from "../memory/procedural/index.js";
import type {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
} from "../memory/self/index.js";
import type {
  ReviewQueueRepository,
  SemanticEdgeRepository,
  SemanticGraph,
  SemanticNodeRepository,
} from "../memory/semantic/index.js";
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
  RetrievalGetEpisodeOptions,
  RetrievalPipeline,
  RetrievalSearchOptions,
} from "../retrieval/index.js";
import type { LanceDbStore } from "../storage/lancedb/index.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import type { StreamEntryIndexRepository, StreamWriter } from "../stream/index.js";
import type { Clock } from "../util/clock.js";
import type { EntityId, SessionId } from "../util/ids.js";

export type BorgStreamWriterFactory = (sessionId: SessionId) => StreamWriter;

export type BorgDependencies = {
  config: Config;
  sqlite: SqliteDatabase;
  lance: LanceDbStore;
  entryIndex: StreamEntryIndexRepository;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository: SemanticNodeRepository;
  semanticEdgeRepository: SemanticEdgeRepository;
  semanticGraph: SemanticGraph;
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
  correctionService: CorrectionService;
  skillRepository: SkillRepository;
  skillSelector: SkillSelector;
  retrievalPipeline: RetrievalPipeline;
  workingMemoryStore: WorkingMemoryStore;
  turnOrchestrator: TurnOrchestrator;
  autonomyScheduler: AutonomyScheduler;
  maintenanceScheduler: MaintenanceScheduler;
  streamIngestionCoordinator?: StreamIngestionCoordinator;
  auditLog: AuditLog;
  maintenanceOrchestrator: MaintenanceOrchestrator;
  offlineProcesses: Record<OfflineProcessName, OfflineProcess>;
  llmFactory: () => LLMClient;
  embeddingClient: EmbeddingClient;
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
  tracerPath?: string;
  /**
   * When true, every completed turn triggers watermark-based episodic
   * extraction so the next turn's retrieval sees material from the turn
   * that just ran. Defaults to true. Tests and scripted harnesses that use
   * fake LLM response queues should opt out explicitly so extraction does
   * not consume responses out of band.
   */
  liveExtraction?: boolean;
};

export type BorgDreamOptions = {
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

export type BorgEpisodeSearchOptions = Omit<RetrievalSearchOptions, "audienceEntityId"> & {
  audience?: string | null;
  audienceEntityId?: EntityId | null;
};

export type BorgEpisodeGetOptions = Omit<RetrievalGetEpisodeOptions, "audienceEntityId"> & {
  audience?: string | null;
  audienceEntityId?: EntityId | null;
};
