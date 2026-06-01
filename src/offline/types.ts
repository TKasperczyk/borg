import type { Config } from "../config/index.js";
import type { TurnTracer } from "../cognition/tracing/tracer.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { LLMClient } from "../llm/index.js";
import type { MoodRepository } from "../memory/affective/index.js";
import type { ActionRepository } from "../memory/actions/index.js";
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
  SemanticBeliefDependencyRepository,
  ReviewQueueRepository,
  SemanticEdgeRepository,
  SemanticNodeRepository,
  SemanticReviewService,
} from "../memory/semantic/index.js";
import type { SocialRepository } from "../memory/social/index.js";
import type { WorkingMemoryStore } from "../memory/working/index.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import type { RelationalSlotRepository } from "../memory/relational-slots/index.js";
import type { StreamEntryIndexRepository, StreamWriter } from "../stream/index.js";
import type { Clock } from "../util/clock.js";
import type { MaintenanceRunId } from "../util/ids.js";

import type { AuditLog } from "./audit-log.js";

export const OFFLINE_PROCESS_NAMES = [
  "consolidator",
  "reflector",
  "semantic-extractor",
  "curator",
  "overseer",
  "review-resolver",
  "ruminator",
  "self-narrator",
  "procedural-synthesizer",
  "belief-reviser",
] as const;

export type OfflineProcessName = (typeof OFFLINE_PROCESS_NAMES)[number];

export type OfflineChange = {
  process: OfflineProcessName;
  action: string;
  targets: Record<string, unknown>;
  preview?: Record<string, unknown>;
};

export type OfflineProcessError = {
  process: OfflineProcessName;
  message: string;
  code?: string;
  target_type?: "episode" | "semantic_node" | "semantic_edge";
  target_id?: string;
};

export type OfflineResult = {
  process: OfflineProcessName;
  dryRun: boolean;
  changes: OfflineChange[];
  tokens_used: number;
  errors: OfflineProcessError[];
  budget_exhausted: boolean;
  candidate_stats?: {
    proposed: number;
    accepted: number;
    rejected: number;
  };
  pending_episode_count?: number;
  run_capped?: boolean;
};

export type OfflineProcessPlan = {
  process: OfflineProcessName;
  tokens_used: number;
  errors: OfflineProcessError[];
  budget_exhausted: boolean;
};

export type OfflineContext = {
  config: Config;
  runId: MaintenanceRunId;
  clock: Clock;
  auditLog: AuditLog;
  streamWriter: StreamWriter;
  entryIndex: StreamEntryIndexRepository;
  embeddingClient: EmbeddingClient;
  tracer?: TurnTracer;
  llm: {
    cognition: LLMClient;
    background: LLMClient;
    extraction: LLMClient;
  };
  episodicRepository: EpisodicRepository;
  semanticNodeRepository: SemanticNodeRepository;
  semanticEdgeRepository: SemanticEdgeRepository;
  semanticBeliefDependencyRepository: SemanticBeliefDependencyRepository;
  semanticReviewService?: SemanticReviewService;
  reviewQueueRepository: ReviewQueueRepository;
  identityService: IdentityService;
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
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
  skillRepository: SkillRepository;
  proceduralEvidenceRepository: ProceduralEvidenceRepository;
  workingMemoryStore?: WorkingMemoryStore;
  retrievalPipeline: RetrievalPipeline;
};

export type OfflineProcessRunOptions = {
  dryRun?: boolean;
  budget?: number;
  params?: Record<string, unknown>;
};

export interface OfflineProcess<Plan extends OfflineProcessPlan = OfflineProcessPlan> {
  readonly name: OfflineProcessName;
  plan(ctx: OfflineContext, opts: OfflineProcessRunOptions): Promise<Plan>;
  preview(plan: Plan): OfflineResult;
  apply(ctx: OfflineContext, plan: Plan): Promise<OfflineResult>;
  run(ctx: OfflineContext, opts: OfflineProcessRunOptions): Promise<OfflineResult>;
}

export type OrchestratorResult = {
  run_id: MaintenanceRunId;
  dryRun: boolean;
  results: OfflineResult[];
  changes: OfflineChange[];
  tokens_used: number;
  errors: OfflineProcessError[];
};
