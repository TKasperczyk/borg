// Shared deliberation data shapes used by the orchestrator and extracted helpers.
import type { LLMClient } from "../../llm/index.js";
import type { BorgUserContentBlock } from "../../attachments/index.js";
import type { ExecutiveFocus } from "../../executive/index.js";
import type { MoodHistoryEntry } from "../../memory/affective/index.js";
import type { ActionRecord } from "../../memory/actions/index.js";
import type {
  BorgRole,
  CommitmentRecord,
  EntityRepository,
} from "../../memory/commitments/index.js";
import type {
  CreatorDirectiveKind,
  CreatorDirectiveMentionPolicy,
  CreatorDirectiveSubjectKind,
} from "../../memory/creator-directives/index.js";
import type {
  AutobiographicalPeriod,
  GoalRecord,
  GrowthMarker,
  OpenQuestion,
  TraitRecord,
  ValueRecord,
} from "../../memory/self/index.js";
import type { SocialProfile } from "../../memory/social/index.js";
import type { SkillSelectionResult } from "../../memory/procedural/index.js";
import type {
  RelationalSlot,
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import type { ReviewQueueItem } from "../../memory/semantic/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type {
  EvidenceItem,
  RetrievedContext,
  RetrievedContradictionRouting,
  RetrievalConfidence,
  RetrievedEpisode,
  RetrievedSemantic,
  RetrievalSearchOptions,
} from "../../retrieval/index.js";
import type { ToolDispatcher } from "../../tools/index.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, SessionId, StreamEntryId } from "../../util/ids.js";
import type { ToolLoopCallRecord } from "../turn-action/index.js";
import type { AutonomyTriggerContext } from "../autonomy-trigger.js";
import type { FrameAnomalyClassification } from "../frame-anomaly/index.js";
import type { PendingTurnEmission } from "../generation/types.js";
import type { EmissionRecommendation } from "../generation/types.js";
import type { SharedStateRenderOptions, EvidenceLedger } from "../evidence-ledger/index.js";
import type { ActiveParticipant, ParticipantProfileContext } from "../participants.js";
import type { ParticipantRoster } from "../perception/index.js";
import type { RecencyMessage } from "../recency/index.js";
import type { PromptKey } from "../prompts/registry.js";
import type { SessionAudienceRole, SessionParticipationPolicy } from "../../sessions/index.js";
import type { OperatorSessionSnapshot } from "../lifecycle/turn-phase/session-snapshot.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { IntentRecord, PerceptionResult } from "../types.js";
import type { ContradictionRoutingCooldown } from "./contradiction-routing-cooldown.js";

export type TurnStakes = "low" | "medium" | "high";
export type DeliberationRoutingForcedBy = "open_question_contradiction";
export type ContradictionRoutingTier =
  | "none"
  | "annotation_only"
  | "confidence_penalty"
  | "s2_recommended"
  | "s2_forced";

export type DeliberationContradictionRoutingConfig = {
  enabled: boolean;
  cooldownTurns: number;
};

export type TrustedCreatorContext = {
  currentSenderEntityId: EntityId | null;
  currentSenderDisplayName: string | null;
  currentSenderBorgRole: BorgRole | null;
  sessionAudienceRole: SessionAudienceRole;
};

export type CreatorDirectiveBriefingContentDirective = {
  renderMode: "content";
  kind: CreatorDirectiveKind;
  subjectKind: CreatorDirectiveSubjectKind;
  subjectLabel: string;
  canonicalFact: string | null;
  operationalDirective: string | null;
  mentionPolicy: CreatorDirectiveMentionPolicy;
  priority: number;
  createdAt: number;
};

export type CreatorDirectiveBriefingBoundaryDirective = {
  renderMode: "boundary";
  priority: number;
  createdAt: number;
};

export type CreatorDirectiveBriefingDirective =
  | CreatorDirectiveBriefingContentDirective
  | CreatorDirectiveBriefingBoundaryDirective;

export type CreatorDirectiveBriefing = {
  directives: readonly CreatorDirectiveBriefingDirective[];
};

export type DeliberationRoutingOverride = {
  forceSystem2: boolean;
  reason: DeliberationRoutingForcedBy;
  forcedBy: DeliberationRoutingForcedBy;
  oqIds: readonly string[];
  openQuestions?: readonly (Pick<OpenQuestion, "id" | "question" | "source"> & {
    localHandle?: string;
  })[];
  contradictionFingerprints?: readonly string[];
  audienceEntityId?: EntityId | null;
  isOperational?: boolean;
};

export type SelfSnapshot = {
  values: ValueRecord[];
  goals: GoalRecord[];
  traits: TraitRecord[];
  /**
   * The being's current autobiographical period (label + narrative). Phase
   * F wires this into the deliberator prompt so the being has a glimpse of
   * its own arc rather than values/goals/traits alone. Null when no period
   * has been opened yet.
   */
  currentPeriod?: AutobiographicalPeriod | null;
  /**
   * Recent growth markers -- what the being has newly learned or noticed
   * about itself. Surfaced as a thin "Recent learning" section so the
   * being doesn't keep rediscovering the same ground every session.
   */
  recentGrowthMarkers?: readonly GrowthMarker[];
};

export type DeliberationContext = {
  sessionId: SessionId;
  participationPolicy?: SessionParticipationPolicy;
  creatorContext?: TrustedCreatorContext | null;
  creatorDirectiveBriefing?: CreatorDirectiveBriefing | null;
  operatorSessionSnapshot?: OperatorSessionSnapshot | null;
  turnId?: string;
  audience?: string;
  audienceEntityId?: EntityId | null;
  senderEntityId?: EntityId;
  userMessage: string;
  currentUserContent?: readonly BorgUserContentBlock[];
  userEntryId?: string;
  autonomyTrigger?: AutonomyTriggerContext | null;
  perception: PerceptionResult;
  retrievalResult: RetrievedEpisode[];
  /**
   * Semantic-band retrieval for this query: graph walks across supports,
   * causes/prevents, contradicts, and is_a relations from matched nodes. Previously
   * attached per-episode with the same value duplicated; Phase C lifted
   * it out so it can be rendered once regardless of episode count and
   * retrieved independently of episode hits.
   */
  retrievedSemantic?: RetrievedSemantic | null;
  retrievedEvidence?: readonly EvidenceItem[];
  contradictionPresent?: boolean;
  contradictionRouting?: RetrievedContradictionRouting | null;
  contradictionRoutingTier?: ContradictionRoutingTier;
  deliberationPath?: "system_1" | "system_2";
  retrievalConfidence?: RetrievalConfidence | null;
  applicableCommitments?: readonly CommitmentRecord[];
  openQuestionsContext?: readonly OpenQuestion[];
  pendingCorrectionsContext?: readonly ReviewQueueItem[];
  relationalSlots?: readonly RelationalSlot[];
  activeParticipants?: readonly ActiveParticipant[];
  participantRoster?: ParticipantRoster | null;
  participantProfiles?: readonly ParticipantProfileContext[];
  selectedSkill?: SkillSelectionResult | null;
  entityRepository?: EntityRepository;
  workingMemory: WorkingMemory;
  recentCompletedActions?: readonly ActionRecord[];
  /**
   * Recent affective history for this session, newest first. The current
   * mood snapshot remains in workingMemory; this lane shows prior turns.
   */
  affectiveTrajectory?: readonly MoodHistoryEntry[];
  selfSnapshot: SelfSnapshot;
  /**
   * Derived executive focus for this turn. It is a soft bias over active
   * goals, never a directive that overrides the current user request or
   * active commitments.
   */
  executiveFocus?: ExecutiveFocus | null;
  /**
   * Social band: the profile of the person the being is talking to, when
   * audience is known. Phase F wires a thin summary (trust, interactions,
   * last contact) into the prompt so the being has relational context
   * rather than treating every audience as a cold first contact.
   */
  audienceProfile?: SocialProfile | null;
  /**
   * Recent dialogue from this session's stream, pre-compiled as LLM-ready
   * messages. If omitted, the deliberator behaves as it did pre-Phase-A:
   * the LLM sees only the current user message. Passing a window restores
   * the being's visibility into its own just-completed turns.
   */
  recencyMessages?: readonly RecencyMessage[];
  frameAnomaly?: FrameAnomalyClassification | null;
  /**
   * Optional finalizer-only evidence ledger prompt section. This is appended
   * after the legacy base prompt and before S2 additional retrieval / plan
   * sections when enabled.
   */
  evidenceLedgerPromptSection?: string | null;
  /**
   * Trusted prompt guidance for first-turn session re-entry when durable
   * audience shared state already exists. This is substrate presentation, not
   * a post-generation output judge.
   */
  sessionReentryContinuityPromptSection?: string | null;
  /**
   * Typed ledger corresponding to evidenceLedgerPromptSection. Emission-tool
   * finalization uses this to keep prompt-visible IDs tied to evidence prose.
   */
  evidenceLedger?: EvidenceLedger | null;
  /**
   * Count of shared-state compiler operations applied for the current turn.
   * This is compiler-emitted structure, not inferred from artifact text.
   */
  sharedStateAppliedOperationCount?: number;
  /**
   * Count of open-question entries actually rendered to the finalizer prompt.
   * Available-but-omitted open questions do not count.
   */
  openQuestionsRenderedToFinalizerCount?: number;
  routingOverride?: DeliberationRoutingOverride | null;
  contradictionRoutingCooldown?: ContradictionRoutingCooldown;
  contradictionRoutingConfig?: DeliberationContradictionRoutingConfig;
  options?: {
    stakes?: TurnStakes;
    maxThinkingTokens?: number;
  };
  reRetrieve?: (query: string, options?: RetrievalSearchOptions) => Promise<RetrievedContext>;
};

export type DeliberationUsage = {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
  stop_reason: string | null;
};

export type DeliberationRegenerationInput = {
  additionalPromptSections: readonly (string | null)[];
};

export type CognitionThinkingConfig = {
  enabled: boolean;
  budget_tokens: number;
};

export type DeliberationResult = {
  path: "system_1" | "system_2";
  response: string;
  emitted?: boolean;
  emission?: PendingTurnEmission;
  emissionRecommendation?: EmissionRecommendation;
  thoughtStreamEntryIds?: readonly StreamEntryId[];
  thoughts: string[];
  tool_calls: ToolLoopCallRecord[];
  usage: DeliberationUsage;
  decision_reason: string;
  retrievedEpisodes: RetrievedEpisode[];
  referencedEpisodeIds: readonly string[] | null;
  intents: IntentRecord[];
  thoughtsPersisted: boolean;
  regenerateFinalResponse?: (input: DeliberationRegenerationInput) => Promise<DeliberationResult>;
};

export type DeliberatorOptions = {
  llmClient: LLMClient;
  toolDispatcher: ToolDispatcher;
  cognitionModel: string;
  cognitionThinking?: CognitionThinkingConfig;
  clock?: Clock;
  tracer?: TurnTracer;
  hostCapabilities?: string;
  promptBlocks?: Partial<Record<PromptKey, string>>;
  sharedStateRenderOptions?: SharedStateRenderOptions;
  maxImagesPerLlmCall?: number;
};
