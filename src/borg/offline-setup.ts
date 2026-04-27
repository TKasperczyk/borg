// Wires Borg's offline maintenance processes into the maintenance orchestrator.

import type { Config } from "../config/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { LLMClient } from "../llm/index.js";
import type { MoodRepository } from "../memory/affective/index.js";
import type { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import type { IdentityService } from "../memory/identity/index.js";
import type { ProceduralEvidenceRepository, SkillRepository } from "../memory/procedural/index.js";
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
  SemanticBeliefDependencyRepository,
  SemanticEdgeRepository,
  SemanticNodeRepository,
} from "../memory/semantic/index.js";
import type { SocialRepository } from "../memory/social/index.js";
import {
  AuditLog,
  BeliefReviserProcess,
  ConsolidatorProcess,
  CuratorProcess,
  MaintenanceOrchestrator,
  OverseerProcess,
  ProceduralSynthesizerProcess,
  ReflectorProcess,
  ReverserRegistry,
  RuminatorProcess,
  SelfNarratorProcess,
  type OfflineProcess,
  type OfflineProcessName,
} from "../offline/index.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import type { Clock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import type { BorgStreamWriterFactory } from "./types.js";

export type BorgOfflineSetup = {
  auditLog: AuditLog;
  maintenanceOrchestrator: MaintenanceOrchestrator;
  offlineProcesses: Record<OfflineProcessName, OfflineProcess>;
};

export type BuildOfflineSetupOptions = {
  config: Config;
  sqlite: SqliteDatabase;
  clock: Clock;
  embeddingClient: EmbeddingClient;
  lazyLlmClient: LLMClient;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository: SemanticNodeRepository;
  semanticEdgeRepository: SemanticEdgeRepository;
  semanticBeliefDependencyRepository: SemanticBeliefDependencyRepository;
  reviewQueueRepository: ReviewQueueRepository;
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
  proceduralEvidenceRepository: ProceduralEvidenceRepository;
  retrievalPipeline: RetrievalPipeline;
  createStreamWriter: BorgStreamWriterFactory;
};

export function buildOfflineSetup(options: BuildOfflineSetupOptions): BorgOfflineSetup {
  const reverserRegistry = new ReverserRegistry();
  const auditLog = new AuditLog({
    db: options.sqlite,
    clock: options.clock,
    registry: reverserRegistry,
  });
  const offlineProcesses = {
    consolidator: new ConsolidatorProcess({
      episodicRepository: options.episodicRepository,
      registry: reverserRegistry,
    }),
    reflector: new ReflectorProcess({
      semanticNodeRepository: options.semanticNodeRepository,
      semanticEdgeRepository: options.semanticEdgeRepository,
      reviewQueueRepository: options.reviewQueueRepository,
      registry: reverserRegistry,
      clock: options.clock,
    }),
    curator: new CuratorProcess({
      episodicRepository: options.episodicRepository,
      traitsRepository: options.traitsRepository,
      moodRepository: options.moodRepository,
      socialRepository: options.socialRepository,
      registry: reverserRegistry,
    }),
    overseer: new OverseerProcess({
      reviewQueueRepository: options.reviewQueueRepository,
      registry: reverserRegistry,
    }),
    ruminator: new RuminatorProcess({
      openQuestionsRepository: options.openQuestionsRepository,
      growthMarkersRepository: options.growthMarkersRepository,
      registry: reverserRegistry,
    }),
    "self-narrator": new SelfNarratorProcess({
      autobiographicalRepository: options.autobiographicalRepository,
      growthMarkersRepository: options.growthMarkersRepository,
      registry: reverserRegistry,
    }),
    "procedural-synthesizer": new ProceduralSynthesizerProcess({
      skillRepository: options.skillRepository,
      proceduralEvidenceRepository: options.proceduralEvidenceRepository,
      registry: reverserRegistry,
      clock: options.clock,
    }),
    "belief-reviser": new BeliefReviserProcess({
      db: options.sqlite,
      confidenceDropMultiplier:
        options.config.offline.beliefReviser.confidenceDropMultiplier,
      confidenceFloor: options.config.offline.beliefReviser.confidenceFloor,
      regradeBatchSize: options.config.offline.beliefReviser.regradeBatchSize,
      maxEventsPerRun: options.config.offline.beliefReviser.maxEventsPerRun,
      maxReviewsPerRun: options.config.offline.beliefReviser.maxReviewsPerRun,
      claimStaleSec: options.config.offline.beliefReviser.claimStaleSec,
      maxParseFailures: options.config.offline.beliefReviser.maxParseFailures,
      budget: options.config.offline.beliefReviser.budget,
      consecutiveParseFailureLimit:
        options.config.offline.beliefReviser.consecutiveParseFailureLimit,
    }),
  } satisfies Record<OfflineProcessName, OfflineProcess>;
  const maintenanceOrchestrator = new MaintenanceOrchestrator({
    baseContext: {
      config: options.config,
      clock: options.clock,
      embeddingClient: options.embeddingClient,
      llm: {
        cognition: options.lazyLlmClient,
        background: options.lazyLlmClient,
        extraction: options.lazyLlmClient,
      },
      episodicRepository: options.episodicRepository,
      semanticNodeRepository: options.semanticNodeRepository,
      semanticEdgeRepository: options.semanticEdgeRepository,
      semanticBeliefDependencyRepository: options.semanticBeliefDependencyRepository,
      reviewQueueRepository: options.reviewQueueRepository,
      identityService: options.identityService,
      valuesRepository: options.valuesRepository,
      goalsRepository: options.goalsRepository,
      traitsRepository: options.traitsRepository,
      autobiographicalRepository: options.autobiographicalRepository,
      growthMarkersRepository: options.growthMarkersRepository,
      openQuestionsRepository: options.openQuestionsRepository,
      moodRepository: options.moodRepository,
      socialRepository: options.socialRepository,
      entityRepository: options.entityRepository,
      commitmentRepository: options.commitmentRepository,
      skillRepository: options.skillRepository,
      proceduralEvidenceRepository: options.proceduralEvidenceRepository,
      retrievalPipeline: options.retrievalPipeline,
    },
    auditLog,
    createStreamWriter: () => options.createStreamWriter(DEFAULT_SESSION_ID),
    processRegistry: offlineProcesses,
  });

  return {
    auditLog,
    maintenanceOrchestrator,
    offlineProcesses,
  };
}
