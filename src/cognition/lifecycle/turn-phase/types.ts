import type { ToolLoopCallRecord } from "../../turn-action/index.js";
import type {
  AttachmentRepository,
  AttachmentService,
  ImagePerceptionService,
  TurnInputAttachment,
} from "../../../attachments/index.js";
import type { TurnActionCoordinator } from "../../turn-action/turn-action-coordinator.js";
import type { TurnActionStateService } from "../../actions/turn-action-state-service.js";
import type { AttributionLifecycleService } from "../../attribution/lifecycle-service.js";
import type { AutonomyTriggerContext } from "../../autonomy-trigger.js";
import type { CorrectivePreferenceTurnService } from "../../commitments/corrective-preference-service.js";
import type { TurnStakes } from "../../deliberation/deliberator.js";
import type { TurnDiscourseStateService } from "../../generation/turn-discourse-state.js";
import type { TurnEmission } from "../../generation/types.js";
import type { TurnPostGenerationGuardRunner } from "../../generation/turn-post-generation-guard.js";
import type { TurnGoalPromotionService } from "../../goals/turn-goal-promotion-service.js";
import type { StreamIngestionCoordinator } from "../../ingestion/index.js";
import type { PerceptionGateway } from "../../perception/gateway.js";
import type { TurnOpeningPersistence } from "../../persistence/turn-opening.js";
import type { TurnReflectionCoordinator } from "../../reflection/turn-reflection-coordinator.js";
import type { TurnRetrievalCoordinator } from "../../retrieval/turn-coordinator.js";
import type { TurnSelfContextBuilder } from "../../self/turn-self-context.js";
import type { TurnTerminalOutcome, TurnTracer } from "../../tracing/tracer.js";
import type { PromptOverrideRepository } from "../../prompts/override-repository.js";
import type { CognitiveMode, IntentRecord } from "../../types.js";
import type { Config } from "../../../config/index.js";
import type { EmbeddingClient } from "../../../embeddings/index.js";
import type { LLMClient } from "../../../llm/index.js";
import type { ActionRepository } from "../../../memory/actions/index.js";
import type { CommitmentRepository, EntityRepository } from "../../../memory/commitments/index.js";
import type { SharedStateRepository } from "../../../memory/decision-artifacts/index.js";
import type { EpisodicRepository } from "../../../memory/episodic/index.js";
import type { RelationalSlotRepository } from "../../../memory/relational-slots/index.js";
import type { GoalsRepository, OpenQuestionsRepository } from "../../../memory/self/index.js";
import type { SemanticNodeRepository } from "../../../memory/semantic/index.js";
import type { SocialRepository } from "../../../memory/social/index.js";
import type { WorkingMemoryStore } from "../../../memory/working/index.js";
import type { SessionsRepository } from "../../../sessions/index.js";
import type {
  StreamEntryIndexRepository,
  StreamReader,
  StreamWriter,
} from "../../../stream/index.js";
import type { ToolDispatcher } from "../../../tools/index.js";
import type { Clock } from "../../../util/clock.js";
import type { EntityId, SessionId } from "../../../util/ids.js";
import type { TurnLifecycleTracker } from "../turn-lifecycle-tracker.js";

export type TurnPhaseInput = {
  userMessage: string;
  attachments?: readonly TurnInputAttachment[];
  audience?: string;
  senderEntityId?: EntityId;
  stakes?: TurnStakes;
  sessionId?: SessionId;
  globalTurnCounter?: number;
  origin?: "user" | "autonomous";
  autonomyTrigger?: AutonomyTriggerContext | null;
};

export type TurnPhaseResult = {
  turn_id: string;
  mode: CognitiveMode;
  path: "system_1" | "system_2" | "suppressed";
  response: string;
  emitted: boolean;
  emission: TurnEmission;
  thoughts: string[];
  usage: {
    input_tokens: number;
    output_tokens: number;
    stop_reason: string | null;
  };
  retrievedEpisodeIds: string[];
  referencedEpisodeIds: string[];
  intents: IntentRecord[];
  toolCalls: ToolLoopCallRecord[];
  agentMessageId?: string;
  terminalOutcome?: TurnTerminalOutcome;
};

export type TurnPhaseCoordinatorOptions = {
  config: Config;
  embeddingClient: EmbeddingClient;
  episodicRepository?: Pick<EpisodicRepository, "getMany">;
  semanticNodeRepository?: Pick<
    SemanticNodeRepository,
    "searchByVector" | "markSuperseded" | "markContradicted"
  >;
  workingMemoryStore: WorkingMemoryStore;
  entityRepository: EntityRepository;
  socialRepository: SocialRepository;
  relationalSlotRepository: RelationalSlotRepository;
  actionRepository: Pick<ActionRepository, "get" | "list" | "update"> &
    Partial<Pick<ActionRepository, "findSimilarDescriptionPairs">>;
  commitmentRepository: CommitmentRepository;
  sharedStateRepository: Pick<SharedStateRepository, "get" | "upsert">;
  goalsRepository: GoalsRepository;
  openQuestionsRepository: Pick<
    OpenQuestionsRepository,
    "findByHandles" | "get" | "list" | "resolve"
  >;
  toolDispatcher: ToolDispatcher;
  createStreamReader: (sessionId: SessionId) => StreamReader;
  entryIndex?: Pick<
    StreamEntryIndexRepository,
    "countSessionEntriesByKind" | "lookupEntriesById" | "quarantinedSharedStateArtifactRefs"
  >;
  attachmentService: AttachmentService;
  attachmentRepository: Pick<AttachmentRepository, "get" | "isActiveForStreamEntry">;
  imagePerceptionService?: ImagePerceptionService;
  streamIngestionCoordinator?: StreamIngestionCoordinator;
  llmFactory: () => LLMClient;
  perceptionGateway: PerceptionGateway;
  turnOpeningPersistence: TurnOpeningPersistence;
  attributionLifecycleService: AttributionLifecycleService;
  correctivePreferenceTurnService: CorrectivePreferenceTurnService;
  turnActionStateService: TurnActionStateService;
  turnGoalPromotionService: TurnGoalPromotionService;
  selfContextBuilder: TurnSelfContextBuilder;
  turnRetrievalCoordinator: TurnRetrievalCoordinator;
  discourseStateService: TurnDiscourseStateService;
  postGenerationGuardRunner: Pick<TurnPostGenerationGuardRunner, "listRecentCompletedActions">;
  turnActionCoordinator: TurnActionCoordinator;
  turnReflectionCoordinator: TurnReflectionCoordinator;
  clock: Clock;
  tracer: TurnTracer;
  promptOverrideRepository?: Pick<PromptOverrideRepository, "get">;
  sessionsRepository?: Pick<SessionsRepository, "get">;
};

export type RunTurnPhasesInput = {
  input: TurnPhaseInput;
  globalTurnCounter?: number;
  sessionId: SessionId;
  turnId: string;
  streamWriter: StreamWriter;
  lifecycleTracker: TurnLifecycleTracker;
};
