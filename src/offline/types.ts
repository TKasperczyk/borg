import type { Config } from "../config/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { LLMClient } from "../llm/index.js";
import type { MoodRepository } from "../memory/affective/index.js";
import type { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import type { SkillRepository } from "../memory/procedural/index.js";
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
  SemanticNodeRepository,
} from "../memory/semantic/index.js";
import type { SocialRepository } from "../memory/social/index.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import type { StreamWriter } from "../stream/index.js";
import type { Clock } from "../util/clock.js";
import type { MaintenanceRunId } from "../util/ids.js";

import type { AuditLog } from "./audit-log.js";

export const OFFLINE_PROCESS_NAMES = [
  "consolidator",
  "reflector",
  "curator",
  "overseer",
  "ruminator",
  "self-narrator",
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
};

export type OfflineResult = {
  process: OfflineProcessName;
  dryRun: boolean;
  changes: OfflineChange[];
  tokens_used: number;
  errors: OfflineProcessError[];
  budget_exhausted: boolean;
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
  embeddingClient: EmbeddingClient;
  llm: {
    cognition: LLMClient;
    background: LLMClient;
    extraction: LLMClient;
  };
  episodicRepository: EpisodicRepository;
  semanticNodeRepository: SemanticNodeRepository;
  semanticEdgeRepository: SemanticEdgeRepository;
  reviewQueueRepository: ReviewQueueRepository;
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
