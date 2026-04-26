// Borg -- public library entry point.
// See ARCHITECTURE.md for the design. This file re-exports the stable
// library surface; implementation modules stay behind internal paths.

export const VERSION = "0.1.0";

export {
  Borg,
  type BorgDreamOptions,
  type BorgDreamRunner,
  type BorgEpisodeGetOptions,
  type BorgEpisodeSearchOptions,
  type BorgOpenOptions,
} from "./borg.js";
export {
  DEFAULT_CONFIG,
  loadConfig,
  redactConfig,
  type Config,
  type LoadConfigOptions,
} from "./config/index.js";
export {
  FakeEmbeddingClient,
  OpenAICompatibleEmbeddingClient,
  type EmbeddingClient,
  type OpenAICompatibleEmbeddingClientOptions,
} from "./embeddings/index.js";
export {
  AnthropicLLMClient,
  FakeLLMClient,
  type LLMClient,
  type LLMContentBlock,
  type LLMContentBlockMessage,
  type LLMCompleteOptions,
  type LLMCompleteResult,
  type LLMConverseOptions,
  type LLMConverseResult,
  type LLMMessage,
  type LLMSystemBlock,
  type LLMTextBlock,
  type LLMToolCall,
  type LLMToolDefinition,
  type LLMToolResultBlock,
  type LLMToolUseBlock,
  type TokenUsageEvent,
  type TokenUsageSink,
} from "./llm/index.js";

export {
  COGNITIVE_MODES,
  JsonlTracer,
  NOOP_TRACER,
  NoopTracer,
  createTurnTracer,
  type AttentionWeights,
  type CognitiveMode,
  type CreateTurnTracerOptions,
  type IntentRecord,
  type JsonlTracerOptions,
  type PerceptionResult,
  type TemporalCue,
  type ToolLoopCallRecord,
  type TurnInput,
  type TurnResult,
  type TurnStakes,
  type TurnTraceData,
  type TurnTraceEventName,
  type TurnTracer,
} from "./cognition/index.js";
export {
  AUTONOMY_CONDITION_NAMES,
  AUTONOMY_TRIGGER_NAMES,
  AUTONOMY_WAKE_SOURCE_NAMES,
  type AutonomyConditionName,
  type AutonomySchedulerObserver,
  type AutonomyTickEventResult,
  type AutonomyWakeRecord,
  type AutonomyWakeRecordInput,
  type AutonomyWakeSourceName,
  type AutonomyWakeSourceType,
  type DueEvent,
  type TickResult,
} from "./autonomy/index.js";
export {
  OFFLINE_PROCESS_NAMES,
  type MaintenanceAuditRecord,
  type MaintenanceCadence,
  type MaintenancePlan,
  type MaintenanceSchedulerObserver,
  type MaintenanceSchedulerStopOptions,
  type MaintenanceTickResult,
  type OfflineChange,
  type OfflineMaintenanceProcessPlan,
  type OfflineProcessError,
  type OfflineProcessName,
  type OfflineProcessPlan,
  type OfflineResult,
  type OrchestratorResult,
} from "./offline/index.js";

export { type Provenance, type ProvenanceKind } from "./memory/common/index.js";
export {
  type AffectiveSignal as MemoryAffectiveSignal,
  type DominantEmotion,
  type EmotionalArc,
  type MoodHistoryEntry,
  type MoodState,
} from "./memory/affective/index.js";
export {
  EPISODE_TIERS,
  type Episode,
  type EpisodeListOptions,
  type EpisodeListResult,
  type EpisodePatch,
  type EpisodeSearchCandidate,
  type EpisodeSearchOptions,
  type EpisodeStats,
  type EpisodeStatsPatch,
  type EpisodeTier,
  type ExtractFromStreamResult,
} from "./memory/episodic/index.js";
export {
  IDENTITY_RECORD_TYPES,
  type IdentityEvent,
  type IdentityRecordType,
  type IdentityUpdateOptions,
  type IdentityUpdateResult,
} from "./memory/identity/index.js";
export {
  type SkillRecord,
  type SkillSearchCandidate,
  type SkillSelectionCandidate,
  type SkillSelectionResult,
  type SkillStats,
} from "./memory/procedural/index.js";
export {
  GROWTH_MARKER_CATEGORIES,
  OPEN_QUESTION_SOURCES,
  OPEN_QUESTION_STATUSES,
  type AutobiographicalPeriod,
  type GoalRecord,
  type GoalStatus,
  type GoalTreeNode,
  type GrowthMarker,
  type GrowthMarkerCategory,
  type GrowthMarkersSummary,
  type OpenQuestion,
  type OpenQuestionSource,
  type OpenQuestionStatus,
  type TraitRecord,
  type ValueRecord,
} from "./memory/self/index.js";
export {
  type SocialEvent,
  type SocialProfile,
  type SocialSentimentPoint,
} from "./memory/social/index.js";
export {
  type CommitmentRecord,
  type CommitmentType,
  type EntityRecord,
} from "./memory/commitments/index.js";
export {
  REVIEW_KINDS,
  REVIEW_RESOLUTIONS,
  type ExtractSemanticResult,
  type ReviewKind,
  type ReviewQueueItem,
  type ReviewResolution,
  type ReviewResolutionInput,
  type SemanticContext,
  type SemanticEdge,
  type SemanticNode,
  type SemanticNodeKind,
  type SemanticNodePatch,
  type SemanticNodeSearchCandidate,
  type SemanticRelation,
  type SemanticWalkOptions,
  type SemanticWalkStep,
} from "./memory/semantic/index.js";
export { type WorkingMemory } from "./memory/working/index.js";

export { type RetrievedEpisode } from "./retrieval/index.js";
export {
  STREAM_ENTRY_KINDS,
  type StreamCursor,
  type StreamEntry,
  type StreamEntryInput,
  type StreamEntryKind,
  type StreamIterateOptions,
} from "./stream/index.js";

export { FixedClock, ManualClock, SystemClock, type Clock } from "./util/clock.js";
export {
  AuthError,
  AutonomyError,
  BorgError,
  BudgetExceededError,
  CognitionError,
  CommitmentError,
  ConfigError,
  EmbeddingError,
  LLMError,
  ProvenanceError,
  SemanticError,
  SessionBusyError,
  StorageError,
  StreamError,
  ToolError,
  WorkingMemoryError,
  type BorgErrorJSON,
  type BorgErrorOptions,
} from "./util/errors.js";
export {
  DEFAULT_SESSION_ID,
  createSessionId,
  parseSessionId,
  type AuditId,
  type AutonomyWakeId,
  type BrandedId,
  type CommitmentId,
  type EntityId,
  type EpisodeId,
  type GoalId,
  type MaintenanceRunId,
  type SemanticEdgeId,
  type SemanticNodeId,
  type SessionId,
  type SkillId,
  type StreamEntryId,
  type TraitId,
  type ValueId,
} from "./util/ids.js";
