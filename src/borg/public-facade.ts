// Public Borg facade declarations. Keep these structural and free of internal
// repository/service/scheduler classes so generated declarations stay light.

import type { BorgOperatorAttentionFacade } from "../memory/operator-attention/types.js";

import type {
  AutonomyConditionName,
  AutonomySchedulerDescription,
  AutonomyWakeOutcome,
  AutonomyWakeSourceCategory,
  AutonomyWakeSourceName,
  AutonomyWakeSourceType,
  TickResult,
} from "../autonomy/types.js";
import type { MoodHistoryEntry, MoodState } from "../memory/affective/types.js";
import type {
  ActivityEvent,
  ActivityEventRecordInput,
  ActivityVisibleSessionEvent,
} from "../memory/activity/types.js";
import type {
  ActionActor,
  ActionRecord,
  ActionRecordPatch,
  ActionSessionScope,
  ActionState,
} from "../memory/actions/types.js";
import type {
  CommitmentCriticalDomain,
  CommitmentEnforcementClass,
  CommitmentKind,
  CommitmentRecord,
  CommitmentType,
  BorgRole,
  EntityKind,
  EntityRecord,
  NameProvenance,
} from "../memory/commitments/types.js";
import type {
  CreatorDirective,
  CreatorDirectiveApplicable,
  CreatorDirectiveApplicableOptions,
  CreatorDirectiveId,
  CreatorDirectiveListFilter,
  CreatorDirectiveQueueInput,
} from "../memory/creator-directives/types.js";
import type { Provenance } from "../memory/common/provenance.js";
import type { SharedStateArtifact, SharedStateEntry } from "../memory/shared-state/index.js";
import type {
  Episode,
  EpisodeListOptions,
  EpisodeListResult,
  EpisodeSearchCandidate,
  EpisodeStats,
} from "../memory/episodic/types.js";
import type { IdentityEvent, IdentityRecordType } from "../memory/identity/types.js";
import type { ProceduralContext } from "../memory/procedural/context.js";
import type {
  SkillRecord,
  SkillSearchCandidate,
  SkillSelectionResult,
} from "../memory/procedural/types.js";
import type { RelationalSlot, RelationalSlotState } from "../memory/relational-slots/types.js";
import type {
  AutobiographicalPeriod,
  GoalPatch,
  GoalRecord,
  GoalStatus,
  GoalTreeNode,
  OpenQuestion,
  OpenQuestionListOptions,
  OpenQuestionPatch,
  OpenQuestionSource,
  TraitPatch,
  TraitRecord,
  ValuePatch,
  ValueRecord,
} from "../memory/self/types.js";
import type {
  GrowthMarker,
  GrowthMarkerCategory,
  GrowthMarkersSummary,
} from "../memory/self/growth-markers.js";
import type {
  SemanticEdge,
  SemanticEdgeListOptions,
  SemanticNode,
  SemanticNodeKind,
  SemanticNodeListOptions,
  SemanticNodeListResult,
  SemanticNodeSearchCandidate,
  SemanticNodeStatus,
  SemanticRelation,
  SemanticWalkOptions,
  SemanticWalkStep,
} from "../memory/semantic/types.js";
import type { MemoryDisclosureLabel } from "../retrieval/index.js";
import type { MemoryDisclosureLabelMetadata } from "../memory/common/disclosure-label.js";
import type { SocialProfile } from "../memory/social/types.js";
import type { WorkingMemory } from "../memory/working/types.js";
import type { PromptKey } from "../cognition/prompts/registry.js";
import type {
  InboxReplyActivityReconcileInput,
  InboxReplyActivityReconcileResult,
  IngestionResult,
} from "../cognition/ingestion/index.js";
import type { MaintenancePlan } from "../offline/plan-file.js";
import type { OrchestratorResult } from "../offline/types.js";
import type {
  StreamCursor,
  StreamEntry,
  StreamEntryInput,
  StreamIterateOptions,
} from "../stream/types.js";
import type {
  SessionEnsureInput,
  SessionListOptions,
  SessionParticipationPolicy,
  SessionRecord,
  SessionTouchUpdate,
} from "../sessions/types.js";
import type { TrainOfThoughtJournalEntry } from "../memory/train-of-thought/index.js";
import type {
  ImageMediaType,
  ImagePerceptionRecord,
  StoredAttachmentRecord,
} from "../attachments/index.js";
import type {
  ActionId,
  AuditId,
  AttachmentId,
  AutobiographicalPeriodId,
  AutonomyWakeId,
  CommitmentId,
  EntityId,
  EpisodeId,
  GoalId,
  GrowthMarkerId,
  MaintenanceRunId,
  OpenQuestionId,
  SemanticEdgeId,
  SemanticNodeId,
  SessionId,
  SharedStateEntryId,
  SkillId,
  StreamEntryId,
  TraitId,
  ValueId,
} from "../util/ids.js";
import type { BorgDreamOptions, BorgEpisodeGetOptions, BorgEpisodeSearchOptions } from "./types.js";
import type {
  AutobiographicalRecallInput,
  AutobiographicalRecallResult,
  AutobiographicalRecallServiceOptions,
} from "../cognition/autobiographical-recall.js";
import type { LivedExperienceDaySummary } from "../memory/activity/lived-experience-day-summary.js";

export type BorgIdentityUpdateOptions = {
  throughReview?: boolean;
  reason?: string | null;
  reviewItemId?: number | null;
  preserveRecordProvenance?: boolean;
};

export type BorgIdentityUpdateResult<T> =
  | {
      status: "applied";
      record: T;
    }
  | {
      status: "requires_review";
      current: T;
    };

export type BorgDisclosurePayloadFields = {
  disclosure: string;
  disclosure_label: MemoryDisclosureLabelMetadata;
};

export type BorgGoalWithDisclosure = GoalRecord & BorgDisclosurePayloadFields;
export type BorgGoalTreeWithDisclosure = Omit<GoalTreeNode, "children"> &
  BorgDisclosurePayloadFields & {
    children: BorgGoalTreeWithDisclosure[];
  };
export type BorgCommitmentWithDisclosure = CommitmentRecord & BorgDisclosurePayloadFields;

export type BorgExtractFromStreamResult = {
  inserted: number;
  updated: number;
  skipped: number;
};

export type BorgRetrievedEpisode = {
  episode: Episode;
  score: number;
  // Pre-clamp fused score — the operative relevance value used for MMR
  // selection and result ordering; `score` is its clamped [0,1] form.
  rawScore: number;
  scoreBreakdown: {
    similarity: number;
    decayedSalience: number;
    heat: number;
    goalRelevance: number;
    valueAlignment: number;
    timeRelevance: number;
    moodBoost: number;
    socialRelevance: number;
    entityRelevance: number;
    suppressionPenalty: number;
  };
  citationChain: StreamEntry[];
};

export type BorgStreamReverseScanCap = "entries" | "bytes";

export type BorgStreamReverseScanOptions = {
  maxEntries?: number;
  maxBytes?: number;
  filter?: (entry: StreamEntry) => boolean;
  budgetFilter?: (entry: StreamEntry) => boolean;
  stop?: (entries: StreamEntry[]) => boolean;
};

export type BorgStreamReverseScanResult = {
  entries: StreamEntry[];
  scannedEntries: number;
  scannedBytes: number;
  capReached: BorgStreamReverseScanCap | null;
};

export type BorgStreamReader = {
  iterate(options?: StreamIterateOptions): AsyncGenerator<StreamEntry>;
  scanReverse(options?: BorgStreamReverseScanOptions): BorgStreamReverseScanResult;
  tail(n: number): StreamEntry[];
};

export type BorgStreamFacade = {
  append(input: StreamEntryInput, options?: { session?: SessionId }): Promise<StreamEntry>;
  appendMany(
    inputs: readonly StreamEntryInput[],
    options?: { session?: SessionId },
  ): Promise<StreamEntry[]>;
  tail(n: number, options?: { session?: SessionId }): StreamEntry[];
  reader(options?: { session?: SessionId }): BorgStreamReader;
  hydrateIndexed(
    streamEntryIds: readonly StreamEntryId[],
    options?: { budgetMs?: number; activeOnly?: boolean },
  ): Promise<Map<StreamEntryId, StreamEntry>>;
};

export type BorgEpisodicFacade = {
  get(id: EpisodeId, options?: BorgEpisodeGetOptions): Promise<BorgRetrievedEpisode | null>;
  inspect(id: Episode["id"]): Promise<Episode | null>;
  search(query: string, options?: BorgEpisodeSearchOptions): Promise<BorgRetrievedEpisode[]>;
  searchWithTimeRangeFallback(
    query: string,
    options: BorgEpisodeSearchOptions & { timeRange: { start: number; end: number } },
  ): Promise<{ episodes: BorgRetrievedEpisode[]; timeRangeFallback: boolean }>;
  recordRetrieval(episodeId: Episode["id"], score: number): void;
  extract(options?: {
    sinceTs?: number;
    sinceCursor?: StreamCursor;
    untilTs?: number;
    session?: SessionId;
    bypassSalienceGate?: boolean;
  }): Promise<BorgExtractFromStreamResult>;
  ingest(options?: { session?: SessionId }): Promise<IngestionResult>;
  list(options?: EpisodeListOptions): Promise<EpisodeListResult>;
  listAll(): Promise<Episode[]>;
  listRecentForSession(options: {
    sessionId: SessionId;
    sinceMs: number;
    limit?: number;
    audienceEntityId?: EntityId | null;
    visibleAudienceEntityIds?: readonly EntityId[];
    crossAudience?: boolean;
  }): Promise<EpisodeSearchCandidate[]>;
  getStats(id: Episode["id"]): EpisodeStats | null;
};

export type BorgValueAddInput = {
  id?: ValueId;
  label: string;
  description: string;
  priority: number;
  provenance: Provenance;
  createdAt?: number;
  lastAffirmed?: number | null;
};

export type BorgValueReinforcementEvent = {
  id: number;
  value_id: ValueId;
  ts: number;
  provenance: Provenance;
};

export type BorgValueContradictionEvent = {
  id: number;
  value_id: ValueId;
  ts: number;
  weight: number;
  provenance: Provenance;
};

export type BorgGoalListOptions = {
  status?: GoalStatus;
  statuses?: readonly GoalStatus[];
  visibleToAudienceEntityId?: EntityId | null;
  ownerEntityId?: EntityId | null;
};

export type BorgGoalAddInput = {
  id?: GoalId;
  description: string;
  priority: number;
  parentId?: GoalId | null;
  status?: GoalStatus;
  progressNotes?: string | null;
  provenance: Provenance;
  createdAt?: number;
  targetAt?: number | null;
  audienceEntityId?: EntityId | null;
  ownerEntityId?: EntityId | null;
  counterpartyEntityId?: EntityId | null;
  sourceStreamEntryIds?: readonly StreamEntryId[];
};

export type BorgTraitReinforceInput = {
  label: string;
  delta: number;
  provenance: Provenance;
  timestamp?: number;
  expectedVersion?: number;
};

export type BorgTraitReinforcementEvent = {
  id: number;
  trait_id: TraitId;
  delta: number;
  ts: number;
  provenance: Provenance;
};

export type BorgTraitContradictionEvent = {
  id: number;
  trait_id: TraitId;
  ts: number;
  weight: number;
  provenance: Provenance;
};

export type BorgAutobiographicalPeriodInput = {
  id?: AutobiographicalPeriodId;
  label: string;
  start_ts: number;
  end_ts?: number | null;
  narrative: string;
  key_episode_ids?: readonly EpisodeId[];
  themes?: readonly string[];
  provenance: Provenance;
  created_at?: number;
  last_updated?: number;
};

export type BorgAutobiographicalPeriodListOptions = {
  fromTs?: number;
  toTs?: number;
  limit?: number;
};

// Row budgets an interactive caller may tighten below Sol's defaults.
export type AutobiographicalRecallCaps = Pick<
  AutobiographicalRecallServiceOptions,
  "sourceCap" | "sessionCap" | "totalCap"
>;

export type BorgAutobiographicalUpsertPeriod = {
  (input: BorgAutobiographicalPeriodInput & { id?: undefined }): AutobiographicalPeriod;
  (
    input: BorgAutobiographicalPeriodInput & { id: AutobiographicalPeriodId },
  ): AutobiographicalPeriod | BorgIdentityUpdateResult<AutobiographicalPeriod>;
  (
    input: BorgAutobiographicalPeriodInput,
  ): AutobiographicalPeriod | BorgIdentityUpdateResult<AutobiographicalPeriod>;
};

export type BorgGrowthMarkerAddInput = {
  id?: GrowthMarkerId;
  ts: number;
  category: GrowthMarkerCategory;
  what_changed: string;
  before_description?: string | null;
  after_description?: string | null;
  evidence_episode_ids: readonly (EpisodeId | StreamEntryId)[];
  confidence: number;
  source_process: string;
  provenance: Provenance;
  created_at?: number;
};

export type BorgGrowthMarkerListOptions = {
  sinceTs?: number;
  untilTs?: number;
  category?: GrowthMarkerCategory;
  limit?: number;
};

export type BorgGrowthMarkerSummaryOptions = {
  periodId?: string;
  fromTs?: number;
  toTs?: number;
};

export type BorgJournalListOptions = {
  limit?: number;
};

export type BorgOpenQuestionAddInput = {
  id?: OpenQuestionId;
  question: string;
  urgency: number;
  related_episode_ids?: readonly EpisodeId[];
  related_semantic_node_ids?: readonly SemanticNodeId[];
  goal_id?: GoalId | null;
  audience_entity_id?: EntityId | null;
  provenance?: Provenance | null;
  source: OpenQuestionSource;
  created_at?: number;
  last_touched?: number;
};

export type BorgOpenQuestionResolutionInput = {
  resolution_evidence_episode_ids?: readonly EpisodeId[];
  resolution_evidence_stream_entry_ids?: readonly StreamEntryId[];
  resolution_note: string;
};

export type BorgSelfFacade = {
  // The memory owner's own closed-day summaries (offline day summarizer), newest first, limited
  // after ordering.
  livedExperience: {
    listDaySummaries(options: {
      fromMs: number;
      toMs: number;
      limit?: number;
    }): LivedExperienceDaySummary[];
  };
  values: {
    get(valueId: ValueId): ValueRecord | null;
    list(): ValueRecord[];
    add(input: BorgValueAddInput): ValueRecord;
    update(
      valueId: ValueId,
      patch: ValuePatch | unknown,
      provenance: Provenance,
      options?: BorgIdentityUpdateOptions,
    ): BorgIdentityUpdateResult<ValueRecord>;
    reinforce(
      valueId: ValueId,
      provenance: Provenance,
      timestamp?: number,
      options?: BorgIdentityUpdateOptions,
    ): BorgIdentityUpdateResult<ValueRecord>;
    listReinforcementEvents(valueId: ValueId): BorgValueReinforcementEvent[];
    listContradictionEvents(valueId: ValueId): BorgValueContradictionEvent[];
  };
  goals: {
    block(
      goalId: GoalId,
      input: import("../memory/self/goal-blocks.js").GoalBlockInput,
      provenance: Provenance,
    ): BorgGoalWithDisclosure;
    unblock(goalId: GoalId, reason: string, provenance: Provenance): BorgGoalWithDisclosure;
    get(goalId: GoalId): BorgGoalWithDisclosure | null;
    list(options?: BorgGoalListOptions): BorgGoalTreeWithDisclosure[];
    add(input: BorgGoalAddInput): BorgGoalWithDisclosure;
    update(
      goalId: GoalId,
      patch: GoalPatch | unknown,
      provenance: Provenance,
      options?: BorgIdentityUpdateOptions,
    ): BorgIdentityUpdateResult<BorgGoalWithDisclosure>;
    updateStatus(
      goalId: GoalId,
      status: GoalStatus,
      provenance: Provenance,
      options?: BorgIdentityUpdateOptions,
    ): BorgIdentityUpdateResult<BorgGoalWithDisclosure>;
    updateProgress(
      goalId: GoalId,
      progressNotes: string,
      provenance: Provenance,
      options?: BorgIdentityUpdateOptions,
    ): BorgIdentityUpdateResult<BorgGoalWithDisclosure>;
  };
  traits: {
    get(traitId: TraitId): TraitRecord | null;
    list(): TraitRecord[];
    add(input: BorgTraitReinforceInput): BorgIdentityUpdateResult<TraitRecord>;
    update(
      traitId: TraitId,
      patch: TraitPatch | unknown,
      provenance: Provenance,
      options?: BorgIdentityUpdateOptions,
    ): BorgIdentityUpdateResult<TraitRecord>;
    reinforce(
      input: BorgTraitReinforceInput,
      options?: BorgIdentityUpdateOptions,
    ): BorgIdentityUpdateResult<TraitRecord>;
    listReinforcementEvents(traitId: TraitId): BorgTraitReinforcementEvent[];
    listContradictionEvents(traitId: TraitId): BorgTraitContradictionEvent[];
  };
  autobiographical: {
    // What the memory owner did, decided, said, and observed in a period, assembled by the same
    // service Sol's evidence ledger reads. Gated like Sol: a temporal cue, a self audience, an
    // operator session, or a reflective turn opens it; otherwise null. Rows carry disclosure
    // labels and the caller filters them for its audience.
    recall(
      input: AutobiographicalRecallInput,
      options?: AutobiographicalRecallCaps,
    ): Promise<AutobiographicalRecallResult | null>;
    currentPeriod(): AutobiographicalPeriod | null;
    listPeriods(options?: BorgAutobiographicalPeriodListOptions): AutobiographicalPeriod[];
    upsertPeriod: BorgAutobiographicalUpsertPeriod;
    closePeriod(
      periodId: AutobiographicalPeriodId,
      closedAt: number,
      provenance: Provenance,
      options?: BorgIdentityUpdateOptions,
    ): BorgIdentityUpdateResult<AutobiographicalPeriod>;
    getPeriod(id: AutobiographicalPeriodId): AutobiographicalPeriod | null;
    getByLabel(label: string): AutobiographicalPeriod | null;
  };
  growthMarkers: {
    list(options?: BorgGrowthMarkerListOptions): GrowthMarker[];
    add(input: BorgGrowthMarkerAddInput): GrowthMarker;
    summarize(options?: BorgGrowthMarkerSummaryOptions): GrowthMarkersSummary;
  };
  journal: {
    latest(): TrainOfThoughtJournalEntry | null;
    list(options?: BorgJournalListOptions): TrainOfThoughtJournalEntry[];
  };
  openQuestions: {
    list(options?: OpenQuestionListOptions): OpenQuestion[];
    add(input: BorgOpenQuestionAddInput): OpenQuestion;
    resolve(
      openQuestionId: OpenQuestionId,
      resolution: BorgOpenQuestionResolutionInput,
      provenance: Provenance,
      options?: BorgIdentityUpdateOptions,
    ): BorgIdentityUpdateResult<OpenQuestion>;
    abandon(
      openQuestionId: OpenQuestionId,
      reason: string,
      provenance: Provenance,
      options?: BorgIdentityUpdateOptions,
    ): BorgIdentityUpdateResult<OpenQuestion>;
    bumpUrgency(
      openQuestionId: OpenQuestionId,
      delta: number,
      provenance: Provenance,
      options?: BorgIdentityUpdateOptions,
    ): BorgIdentityUpdateResult<OpenQuestion>;
  };
};

export type BorgSkillAddInput = {
  id?: SkillId;
  applies_when: string;
  approach: string;
  alternatives?: readonly SkillId[];
  sourceEpisodes: readonly EpisodeId[];
  priorAlpha?: number;
  priorBeta?: number;
  createdAt?: number;
};

export type BorgSkillSelectOptions = {
  k?: number;
  minSimilarity?: number;
  proceduralContext?: ProceduralContext | null;
};

export type BorgSkillsFacade = {
  list(limit?: number): SkillRecord[];
  add(input: BorgSkillAddInput): Promise<SkillRecord>;
  get(id: SkillId): SkillRecord | null;
  searchByContext(text: string, limit?: number): Promise<SkillSearchCandidate[]>;
  recordOutcome(
    skillId: SkillId,
    success: boolean,
    episodeIds?: EpisodeId | readonly EpisodeId[],
    proceduralContext?: ProceduralContext | null,
  ): SkillRecord;
  select(text: string, options?: BorgSkillSelectOptions): Promise<SkillSelectionResult | null>;
};

export type BorgMoodUpdateInput = {
  valence: number;
  arousal: number;
  reason?: string;
  provenance: Provenance;
};

export type BorgMoodHistoryOptions = {
  fromTs?: number;
  toTs?: number;
  limit?: number;
};

export type BorgMoodFacade = {
  current(sessionId: SessionId): MoodState;
  history(sessionId: SessionId, options?: BorgMoodHistoryOptions): MoodHistoryEntry[];
  update(sessionId: SessionId, input: BorgMoodUpdateInput): MoodState;
};

export type BorgActionRecordListFilter = {
  state?: ActionState;
  states?: readonly ActionState[];
  actor?: ActionActor;
  sessionScope?: ActionSessionScope | null;
  sessionAnchorId?: SessionId | null;
  audienceEntityId?: EntityId | null;
  goalId?: GoalId;
  openQuestionId?: OpenQuestionId;
  limit?: number;
};

export type BorgActionRecordCreationSource = "extractor" | "reflector" | "api" | "tool" | "unknown";
export type BorgActionCountByState = Record<ActionState, number>;
export type BorgActionCreationCountsBySource = Record<BorgActionRecordCreationSource, number>;

export type BorgActionAddOptions = {
  creationSource?: BorgActionRecordCreationSource;
};

export type BorgActionUpdateOptions = {
  skipSideEffects?: boolean;
};

export type BorgActionDescriptionSimilarityPair = {
  leftId: ActionId;
  rightId: ActionId;
  similarity: number;
};

export type BorgActionsFacade = {
  nextLifecycleTurnGlobal(): number;
  ensureLifecycleTurnGlobal(value: number): number;
  waitForPendingEmbeddings(): Promise<void>;
  add(record: ActionRecord, options?: BorgActionAddOptions): void;
  update(id: ActionId, patch: ActionRecordPatch, options?: BorgActionUpdateOptions): void;
  get(id: ActionId): ActionRecord | null;
  list(filter?: BorgActionRecordListFilter): ActionRecord[];
  count(): number;
  countByState(): BorgActionCountByState;
  countCanonicalized(): number;
  countActive(): number;
  getCreationCountsBySource(): BorgActionCreationCountsBySource;
  countCompletedSince(timestampMs: number): number;
  listCompletedIds(): ActionId[];
  latestCompletedAt(): number | null;
  findByDescription(description: string, limit: number): Promise<ActionRecord[]>;
  findSimilarDescriptionPairs(
    records: readonly ActionRecord[],
    threshold: number,
  ): Promise<BorgActionDescriptionSimilarityPair[]>;
  delete(id: ActionId): Promise<boolean>;
};

export type BorgSocialInteractionInput = {
  provenance: Provenance;
  valence?: number;
  now?: number;
};

export type BorgSocialFacade = {
  getProfile(entity: string): SocialProfile | null;
  list(limit?: number): SocialProfile[];
  upsertProfile(entity: string): SocialProfile;
  recordInteraction(entity: string, interaction: BorgSocialInteractionInput): SocialProfile;
  adjustTrust(entity: string, delta: number, provenance: Provenance): SocialProfile;
};

export type BorgEntityResolveOptions = {
  kind?: EntityKind;
  aliases?: readonly string[];
  provenance?: NameProvenance;
};

export type BorgEntityResolveExternalInput = Pick<
  BorgEntityResolveOptions,
  "kind" | "provenance"
> & {
  source: string;
  externalId: string;
  canonicalName: string;
};

export type BorgEntitiesFacade = {
  resolve(name: string, options?: BorgEntityResolveOptions): EntityId;
  resolveExternal(input: BorgEntityResolveExternalInput): EntityId;
  findByExternalId(source: string, externalId: string): EntityId | null;
  get(id: EntityId): EntityRecord | null;
  list(options?: { kind?: EntityKind }): EntityRecord[];
  getCreator(): EntityRecord | null;
  getSelf(): EntityRecord | null;
  ensureSelf(canonicalName: string, options?: { provenance?: NameProvenance }): EntityRecord;
  setBorgRole(id: EntityId, role: BorgRole | null): EntityRecord | null;
  find(name: string, options?: Pick<BorgEntityResolveOptions, "kind">): EntityRecord | null;
};

export type BorgSharedStateFacade = {
  getForAudience(audience: string): SharedStateArtifact | null;
  listEntriesForAudience(audience: string): SharedStateEntry[];
};

export type BorgAttachmentBytesResult = {
  attachment: StoredAttachmentRecord;
  mediaType: ImageMediaType;
  bytes: Uint8Array;
};

export type BorgAttachmentMetadataResult = {
  attachment: StoredAttachmentRecord;
  perception: ImagePerceptionRecord | null;
  status: {
    active: boolean;
    quarantined: boolean;
    stream_active?: boolean;
    parent_active?: boolean;
  };
};

export type BorgAttachmentsFacade = {
  get(attachmentId: AttachmentId): BorgAttachmentMetadataResult | null;
  getBytes(
    attachmentId: AttachmentId,
    options?: {
      audience?: string | null;
    },
  ): BorgAttachmentBytesResult | null;
};

export type BorgSemanticNodeAddInput = {
  kind: SemanticNodeKind;
  label: string;
  description: string;
  domain?: string | null;
  aliases?: string[];
  confidence?: number;
  sourceEpisodeIds: readonly EpisodeId[];
};

export type BorgSemanticEdgeAddInput = Omit<
  SemanticEdge,
  | "id"
  | "valid_from"
  | "valid_to"
  | "invalidated_at"
  | "invalidated_by_edge_id"
  | "invalidated_by_review_id"
  | "invalidated_by_process"
  | "invalidated_reason"
> &
  Partial<
    Pick<
      SemanticEdge,
      | "valid_from"
      | "valid_to"
      | "invalidated_at"
      | "invalidated_by_edge_id"
      | "invalidated_by_review_id"
      | "invalidated_by_process"
      | "invalidated_reason"
    >
  > & {
    id?: SemanticEdgeId;
  };

export type BorgExtractSemanticResult = {
  insertedNodes: number;
  updatedNodes: number;
  skippedNodes: number;
  insertedEdges: number;
  updatedEdges: number;
  skippedEdges: number;
};

export type BorgSemanticNodeWithDisclosure = SemanticNode & {
  disclosureLabel: MemoryDisclosureLabel;
};

export type BorgSemanticEdgeWithDisclosure = SemanticEdge & {
  disclosureLabel: MemoryDisclosureLabel;
};

export type BorgSemanticNodeSearchCandidateWithDisclosure = Omit<
  SemanticNodeSearchCandidate,
  "node"
> & {
  node: BorgSemanticNodeWithDisclosure;
};

export type BorgSemanticNodeListResultWithDisclosure = Omit<SemanticNodeListResult, "items"> & {
  items: BorgSemanticNodeWithDisclosure[];
};

export type BorgSemanticWalkStepWithDisclosure = Omit<SemanticWalkStep, "node" | "edgePath"> & {
  node: BorgSemanticNodeWithDisclosure;
  edgePath: BorgSemanticEdgeWithDisclosure[];
};

export type BorgSemanticFacade = {
  nodes: {
    add(input: BorgSemanticNodeAddInput): Promise<SemanticNode>;
    get(id: SemanticNodeId): Promise<BorgSemanticNodeWithDisclosure | null>;
    list(options?: SemanticNodeListOptions): Promise<BorgSemanticNodeWithDisclosure[]>;
    listPage(options?: SemanticNodeListOptions): Promise<BorgSemanticNodeListResultWithDisclosure>;
    countByStatus(): Record<SemanticNodeStatus, number>;
    search(
      query: string,
      options?: { limit?: number },
    ): Promise<BorgSemanticNodeSearchCandidateWithDisclosure[]>;
  };
  edges: {
    add(input: BorgSemanticEdgeAddInput): SemanticEdge;
    get(id: SemanticEdgeId): Promise<BorgSemanticEdgeWithDisclosure | null>;
    list(options?: SemanticEdgeListOptions): Promise<BorgSemanticEdgeWithDisclosure[]>;
  };
  walk(
    fromId: SemanticNodeId,
    options?: SemanticWalkOptions,
  ): Promise<BorgSemanticWalkStepWithDisclosure[]>;
  extract(episodes: readonly Episode[]): Promise<BorgExtractSemanticResult>;
};

export type BorgCommitmentAddInput = {
  type: CommitmentType;
  kind?: CommitmentKind;
  enforcementClass?: CommitmentEnforcementClass;
  criticalDomain?: CommitmentCriticalDomain | null;
  directiveFamily: string;
  directive: string;
  priority: number;
  madeTo?: string | null;
  audience?: string | null;
  about?: string | null;
  provenance: Provenance;
  expiresAt?: number | null;
};

export type BorgCommitmentListOptions = {
  activeOnly?: boolean;
  audience?: string | null;
  audienceEntityId?: EntityId | null;
  aboutEntity?: string | null;
};

export type BorgCommitmentsFacade = {
  add(input: BorgCommitmentAddInput): BorgCommitmentWithDisclosure;
  get(id: CommitmentId): BorgCommitmentWithDisclosure | null;
  revoke(
    id: CommitmentId,
    reason: string,
    provenance: Provenance,
    timestamp?: number,
    options?: {
      expectedVersion?: number;
      canonicalizedByArtifactEntryId?: SharedStateEntryId | null;
    },
  ): BorgCommitmentWithDisclosure | null;
  list(options?: BorgCommitmentListOptions): BorgCommitmentWithDisclosure[];
  countActive(): number;
  countActiveByKind(): Record<CommitmentKind, number>;
  countActiveByEnforcementClass(): Record<CommitmentEnforcementClass, number>;
  countSuperseded(): number;
  countRevoked(): number;
  countExpired(): number;
  countCanonicalized(): number;
};

export type BorgActivityFacade = {
  record(input: ActivityEventRecordInput): ActivityEvent;
  projectObservedTurn(input: BorgActivityObservedTurnProjectionInput): {
    userContact: ActivityEvent;
    session: SessionRecord;
  };
  projectRepliedTurn(input: BorgActivityRepliedTurnProjectionInput): {
    borgReplied: ActivityEvent;
    session: SessionRecord;
  };
  projectCompletedTurn(input: BorgActivityCompletedTurnProjectionInput): {
    userContact: ActivityEvent;
    borgReplied: ActivityEvent;
    session: SessionRecord;
  };
  listObservedGroupAudienceEntityIdsForSpeaker(speakerEntityId: EntityId): EntityId[];
  listRecentVisibleOtherSessionEvents(input: {
    currentSessionId: SessionId;
    audienceEntityIds: readonly EntityId[];
    sinceMs: number;
    limit: number;
    kinds?: readonly ActivityVisibleSessionEvent["kind"][];
  }): ActivityVisibleSessionEvent[];
};

export type BorgActivityObservedTurnProjectionInput = {
  session: SessionEnsureInput;
  userContact: ActivityEventRecordInput;
  touch: SessionTouchUpdate;
};

export type BorgActivityRepliedTurnProjectionInput = {
  session: SessionEnsureInput;
  borgReplied: ActivityEventRecordInput;
  touch: SessionTouchUpdate;
};

export type BorgActivityCompletedTurnProjectionInput = {
  session: SessionEnsureInput;
  userContact: ActivityEventRecordInput;
  borgReplied: ActivityEventRecordInput;
  touch: SessionTouchUpdate;
};

export type BorgCreatorDirectivesFacade = {
  queue(input: CreatorDirectiveQueueInput): CreatorDirective;
  get(id: CreatorDirectiveId): CreatorDirective | null;
  list(filter?: CreatorDirectiveListFilter): CreatorDirective[];
  listApplicable(options: CreatorDirectiveApplicableOptions): CreatorDirectiveApplicable[];
  supersede(id: CreatorDirectiveId, replacementId: CreatorDirectiveId): CreatorDirective | null;
  supersedeFamilyAtomic(input: {
    survivorId: CreatorDirectiveId;
    expectedSurvivorVersion: number;
    losers: Array<{ id: CreatorDirectiveId; expectedVersion: number }>;
  }): Array<{ id: CreatorDirectiveId; record_version: number }> | null;
  revoke(id: CreatorDirectiveId, reason: string): CreatorDirective | null;
};

export type BorgIdentityEventListOptions = {
  recordType?: IdentityRecordType;
  recordId?: string;
  limit?: number;
};

export type BorgIdentityFacade = {
  updateValue(
    valueId: ValueId,
    patch: unknown,
    provenance: Provenance,
    options?: BorgIdentityUpdateOptions,
  ): BorgIdentityUpdateResult<ValueRecord>;
  updateGoal(
    goalId: GoalId,
    patch: unknown,
    provenance: Provenance,
    options?: BorgIdentityUpdateOptions,
  ): BorgIdentityUpdateResult<BorgGoalWithDisclosure>;
  updateTrait(
    traitId: TraitId,
    patch: unknown,
    provenance: Provenance,
    options?: BorgIdentityUpdateOptions,
  ): BorgIdentityUpdateResult<TraitRecord>;
  addCommitment(input: {
    id?: CommitmentId;
    type: CommitmentType;
    kind?: CommitmentKind;
    enforcementClass?: CommitmentEnforcementClass;
    criticalDomain?: CommitmentCriticalDomain | null;
    directiveFamily: string;
    directive: string;
    priority: number;
    madeToEntity?: EntityId | null;
    restrictedAudience?: EntityId | null;
    aboutEntity?: EntityId | null;
    committedByEntityId?: EntityId | null;
    provenance: Provenance;
    sourceStreamEntryIds?: readonly StreamEntryId[];
    createdAt?: number;
    expiresAt?: number | null;
    skipDirectiveFamilyMerge?: boolean;
  }): BorgCommitmentWithDisclosure;
  updateCommitment(
    commitmentId: CommitmentId,
    patch: unknown,
    provenance: Provenance,
    options?: BorgIdentityUpdateOptions,
  ): BorgIdentityUpdateResult<BorgCommitmentWithDisclosure>;
  updatePeriod(
    periodId: AutobiographicalPeriodId,
    patch: unknown,
    provenance: Provenance,
    options?: BorgIdentityUpdateOptions,
  ): BorgIdentityUpdateResult<AutobiographicalPeriod>;
  updateGrowthMarker(
    markerId: GrowthMarkerId,
    patch: unknown,
    provenance: Provenance,
    options?: BorgIdentityUpdateOptions,
  ): BorgIdentityUpdateResult<GrowthMarker>;
  updateOpenQuestion(
    openQuestionId: OpenQuestionId,
    patch: OpenQuestionPatch | unknown,
    provenance: Provenance,
    options?: BorgIdentityUpdateOptions,
  ): BorgIdentityUpdateResult<OpenQuestion>;
  listEvents(options?: BorgIdentityEventListOptions): IdentityEvent[];
};

export type BorgCorrectionTargetType =
  | "episode"
  | "semantic_node"
  | "semantic_edge"
  | "value"
  | "goal"
  | "trait"
  | "commitment"
  | "open_question";

export type BorgForgetResult = {
  id: string;
  target_type: BorgCorrectionTargetType;
  archived: true;
  provenance: { kind: "manual" };
};

export type BorgRememberAboutMeResult = {
  entity: EntityRecord | null;
  social_profile: SocialProfile;
  active_commitments: CommitmentRecord[];
  scoped_episodes: EpisodeSearchCandidate[];
  related_episodes: EpisodeSearchCandidate[];
};

export type BorgCorrectionFacade = {
  forget(id: string): Promise<BorgForgetResult>;
  why(id: string): Promise<Record<string, unknown>>;
  invalidateSemanticEdge(id: string, options?: { at?: number; reason?: string }): SemanticEdge;
  correct(
    id: string,
    patch: Record<string, unknown>,
    provenance?: Provenance,
    options?: { reason?: string },
  ): Promise<BorgReviewQueueItem>;
  rememberAboutMe(options?: { entity?: string }): Promise<BorgRememberAboutMeResult>;
  listIdentityEvents(options?: BorgIdentityEventListOptions): IdentityEvent[];
};

export type BorgRelationalSlotsFacade = {
  list(options?: {
    subjectEntityId?: EntityId;
    states?: readonly RelationalSlotState[];
    limit?: number;
  }): RelationalSlot[];
  countByState(): Record<RelationalSlotState, number>;
};

export type BorgReviewKind =
  | "contradiction"
  | "duplicate"
  | "new_insight"
  | "misattribution"
  | "temporal_drift"
  | "identity_inconsistency"
  | "correction"
  | "belief_revision"
  | "skill_split"
  | "creator_directive_reconciliation"
  | "commitment_reconciliation";

export type BorgStoredReviewKind = BorgReviewKind | "relationship_claim_ungrounded";

export type BorgReviewResolution =
  | "keep_both"
  | "supersede"
  | "invalidate"
  | "dismiss"
  | "accept"
  | "reject"
  | "keep"
  | "weaken"
  | "archive_node"
  | "invalidate_edge";

export type BorgReviewResolutionInput =
  | BorgReviewResolution
  | {
      decision: BorgReviewResolution;
      winner_node_id?: SemanticNodeId;
      reason?: string;
    };

export type BorgReviewQueueItem = {
  id: number;
  kind: BorgStoredReviewKind;
  refs: Record<string, unknown>;
  reason: string;
  created_at: number;
  resolved_at: number | null;
  resolution: BorgReviewResolution | null;
};

export type BorgReviewResolveOptions = {
  source?: "manual" | "auto";
  sourceProcess?: string;
  traceTurnId?: string;
};

export type BorgReviewRevalidationOptions = {
  kind: "misattribution";
  maxAgeDays?: number;
};

export type BorgReviewRevalidationResult = {
  kind: "misattribution";
  revalidated: number;
  dismissed_as_suppressed: number;
  skipped_legacy: number;
  unchanged: number;
  diagnostics: Record<string, number>;
  warnings: string[];
};

export type BorgReviewFacade = {
  list(options?: { kind?: BorgReviewKind; openOnly?: boolean }): BorgReviewQueueItem[];
  resolve(
    id: number,
    decision: BorgReviewResolutionInput,
    options?: BorgReviewResolveOptions,
  ): Promise<BorgReviewQueueItem | null>;
  revalidate(options: BorgReviewRevalidationOptions): Promise<BorgReviewRevalidationResult>;
};

export type BorgOfflineChange = {
  process: string;
  action: string;
  targets: Record<string, unknown>;
  preview?: Record<string, unknown>;
};

export type BorgOfflineProcessError = {
  process: string;
  message: string;
  code?: string;
  target_type?: "episode" | "semantic_node" | "semantic_edge";
  target_id?: string;
};

export type BorgOfflineResult = {
  process: string;
  dryRun: boolean;
  changes: BorgOfflineChange[];
  tokens_used: number;
  errors: BorgOfflineProcessError[];
  budget_exhausted: boolean;
  candidate_stats?: {
    proposed: number;
    accepted: number;
    rejected: number;
  };
  pending_episode_count?: number;
  pending_family_count?: number;
  run_capped?: boolean;
};

export type BorgOrchestratorResult = {
  run_id: MaintenanceRunId;
  dryRun: boolean;
  results: BorgOfflineResult[];
  changes: BorgOfflineChange[];
  tokens_used: number;
  errors: BorgOfflineProcessError[];
};

export type BorgOfflineMaintenanceProcessPlan = {
  process: string;
  tokens_used: number;
  errors: BorgOfflineProcessError[];
  budget_exhausted: boolean;
  [key: string]: unknown;
};

export type BorgMaintenancePlan = {
  kind: "borg_maintenance_plan";
  version: 2;
  run_id: MaintenanceRunId;
  created_at: number;
  processes: BorgOfflineMaintenanceProcessPlan[];
};

export type BorgMaintenanceAuditRecord = {
  id: AuditId;
  run_id: MaintenanceRunId;
  process: string;
  action: string;
  targets: Record<string, unknown>;
  reversal: Record<string, unknown>;
  applied_at: number;
  reverted_at: number | null;
  reverted_by: string | null;
};

export type BorgAuditFacade = {
  list(options?: {
    runId?: MaintenanceRunId;
    process?: string;
    reverted?: boolean;
  }): BorgMaintenanceAuditRecord[];
  revert(id: AuditId, revertedBy?: string): Promise<BorgMaintenanceAuditRecord | null>;
};

export type BorgDreamFacade = {
  (options?: BorgDreamOptions): Promise<OrchestratorResult>;
  plan(options?: Omit<BorgDreamOptions, "dryRun">): Promise<MaintenancePlan>;
  preview(plan: MaintenancePlan): OrchestratorResult;
  apply(plan: MaintenancePlan): Promise<OrchestratorResult>;
  consolidate(options?: { dryRun?: boolean; budget?: number }): Promise<OrchestratorResult>;
  reflect(options?: { dryRun?: boolean; budget?: number }): Promise<OrchestratorResult>;
  associate(options?: { dryRun?: boolean; budget?: number }): Promise<OrchestratorResult>;
  extractSemantics(options?: { dryRun?: boolean; budget?: number }): Promise<OrchestratorResult>;
  curate(options?: { dryRun?: boolean; budget?: number }): Promise<OrchestratorResult>;
  oversee(options?: { dryRun?: boolean; budget?: number }): Promise<OrchestratorResult>;
  ruminate(options?: {
    dryRun?: boolean;
    budget?: number;
    maxQuestionsPerRun?: number;
  }): Promise<OrchestratorResult>;
  narrate(options?: {
    dryRun?: boolean;
    budget?: number;
    label?: string;
  }): Promise<OrchestratorResult>;
};

export type BorgAutonomyObserver = {
  onTick?(result: TickResult): void | Promise<void>;
  onError?(error: unknown): void | Promise<void>;
};

export type BorgAutonomyController = {
  setObserver(observer: BorgAutonomyObserver | null): void;
  isEnabled(): boolean;
  describe(): Promise<AutonomySchedulerDescription>;
  start(): void;
  stop(options?: { graceful?: boolean }): Promise<void>;
  tick(): Promise<TickResult>;
};

export type BorgAutonomyWakeRecord = {
  id: AutonomyWakeId;
  ts: number;
  trigger_name: AutonomyWakeSourceName;
  condition_name: AutonomyConditionName | null;
  session_id: SessionId | null;
  wake_source_type: AutonomyWakeSourceType;
  source_category: AutonomyWakeSourceCategory;
  selected_goal_id: GoalId | null;
  outcome: AutonomyWakeOutcome | null;
  outcome_detail: string | null;
  headway_bases: string[] | null;
  finalizer_rounds: number | null;
  stall_retries: number | null;
};

export type BorgAutonomyWakeRecordInput = {
  trigger_name: AutonomyWakeSourceName;
  condition_name?: AutonomyConditionName | null;
  session_id?: SessionId | null;
  wake_source_type: AutonomyWakeSourceType;
  source_category?: AutonomyWakeSourceCategory;
  selected_goal_id?: GoalId | null;
};

export type BorgAutonomyWakesFacade = {
  record(input: BorgAutonomyWakeRecordInput): BorgAutonomyWakeRecord;
  countSince(ts: number, options?: { sourceCategory?: AutonomyWakeSourceCategory }): number;
  listSince(ts: number, limit: number): BorgAutonomyWakeRecord[];
  prune(olderThan: number): number;
};

export type BorgAutonomyFacade = {
  scheduler: BorgAutonomyController;
  wakes: BorgAutonomyWakesFacade;
};

export type BorgMaintenanceCadence = "light" | "heavy";

export type BorgStorageOptimizationTableResult =
  | {
      table: string;
      status: "ok";
      fragmentsRemoved: number;
      fragmentsAdded: number;
      versionsPruned: number;
      bytesRemoved: number;
      durationMs: number;
    }
  | {
      table: string;
      status: "error";
      durationMs: number;
      error: {
        message: string;
        code?: string;
      };
    };

export type BorgStorageOptimizationResult = {
  cleanupOlderThan?: number;
  durationMs: number;
  tables: BorgStorageOptimizationTableResult[];
  error?: {
    message: string;
    code?: string;
  };
};

export type BorgMaintenanceTickResult = {
  status: "ok" | "skipped_busy" | "skipped_empty" | "disabled";
  cadence: BorgMaintenanceCadence;
  ts: number;
  processes: string[];
  result: BorgOrchestratorResult | null;
  storageOptimization?: BorgStorageOptimizationResult | null;
  reason?: string;
};

export type BorgMaintenanceSchedulerObserver = {
  onTick?(result: BorgMaintenanceTickResult): void | Promise<void>;
  onError?(error: unknown, cadence: BorgMaintenanceCadence): void | Promise<void>;
};

export type BorgMaintenanceScheduler = {
  setObserver(observer: BorgMaintenanceSchedulerObserver | null): void;
  isEnabled(): boolean;
  start(): void;
  stop(options?: { graceful?: boolean }): Promise<void>;
  tick(cadence: BorgMaintenanceCadence): Promise<BorgMaintenanceTickResult>;
};

export type BorgMaintenanceFacade = {
  scheduler: BorgMaintenanceScheduler;
  optimizeStorage(options?: { runId?: MaintenanceRunId }): Promise<BorgStorageOptimizationResult>;
  countPendingSemanticExtractionEpisodes(): Promise<number>;
  config(): {
    enabled: boolean;
    lightIntervalMs: number;
    heavyIntervalMs: number;
    optimizeStorage: boolean;
    lightBudget: number | null;
    heavyBudget: number | null;
    lightProcesses: readonly string[];
    heavyProcesses: readonly string[];
    processBudgets: Partial<Record<string, number | null>>;
  };
};

export type BorgInboxCatchUpDrainResult = {
  sessionId: SessionId;
  status: "empty" | "drained" | "busy" | "error";
  drained: number;
  hasMore: boolean;
  error?: string;
};

export type BorgInboxCatchUpController = {
  isEnabled(): boolean;
  start(): void;
  stop(options?: { graceful?: boolean }): Promise<void>;
  onAppend(entries: readonly StreamEntry[]): void;
  tick(sessionId: SessionId): Promise<BorgInboxCatchUpDrainResult>;
};

export type BorgAppendBacklogTerminalInput = {
  sessionId: SessionId;
  sourceEntryIds: readonly StreamEntryId[];
  terminal: { kind: "agent_msg"; content: string } | { kind: "agent_observed"; reason: string };
  audience?: string;
  turnId?: string;
};

export type BorgAppendBacklogTerminalResult = {
  terminalEntry: StreamEntry;
  responseTo: NonNullable<StreamEntry["response_to"]>;
  sourceEntries: readonly StreamEntry[];
};

export type BorgFindTerminalCoveringEntryResult =
  | { status: "unknown_entry" }
  | { status: "session_mismatch" }
  | { status: "pending" }
  | {
      status: "found";
      terminalEntry: StreamEntry;
      responseTo: NonNullable<StreamEntry["response_to"]>;
    };

export type BorgInboxFacade = {
  catchUp: BorgInboxCatchUpController;
  appendBacklogTerminal(
    input: BorgAppendBacklogTerminalInput,
  ): Promise<BorgAppendBacklogTerminalResult>;
  sealPendingBacklog(input: {
    sessionId: SessionId;
    reason?: string;
    audience?: string;
    turnId?: string;
  }): Promise<BorgAppendBacklogTerminalResult | null>;
  sealStaleBacklog(input: {
    sessionId: SessionId;
    staleBefore: number;
    reason?: string;
    audience?: string;
    turnId?: string;
  }): Promise<BorgAppendBacklogTerminalResult | null>;
  findTerminalCoveringEntry(input: {
    sessionId: SessionId;
    entryId: StreamEntryId;
  }): BorgFindTerminalCoveringEntryResult;
  reconcileReplyActivity(
    input: InboxReplyActivityReconcileInput,
  ): InboxReplyActivityReconcileResult;
};

export type BorgWorkmemFacade = {
  load(sessionId?: SessionId): WorkingMemory;
  clear(sessionId?: SessionId): void;
  getPendingActionMergeCount(): number;
};

export type BorgPromptBlockView = {
  key: PromptKey;
  label: string;
  description: string;
  default_text: string;
  current_text: string;
  current_text_kind: "static_default" | "runtime_composed" | "stored_override";
  overridden: boolean;
  updated_at: number | null;
};

export type BorgAssembledFramingPromptPreview = {
  text: string;
  sections: readonly string[];
  segments: readonly BorgAssembledFramingPromptPreviewSegment[];
};

export type BorgAssembledFramingPromptPreviewSegment = {
  id: string;
  label: string;
  editable_key: PromptKey | null;
  start: number;
  end: number;
};

export type BorgPromptsFacade = {
  list(): BorgPromptBlockView[];
  set(key: PromptKey, text: string): BorgPromptBlockView;
  clear(key: PromptKey): BorgPromptBlockView;
  previewAssembledFraming(): BorgAssembledFramingPromptPreview;
};

export type BorgSessionsFacade = {
  ensure(input: SessionEnsureInput): SessionRecord;
  touch(sessionId: SessionId, update?: SessionTouchUpdate): SessionRecord | null;
  setParticipationPolicy(
    sessionId: SessionId,
    policy: SessionParticipationPolicy,
    opts?: { reason?: string },
  ): Promise<SessionRecord>;
  get(sessionId: SessionId): SessionRecord | null;
  list(options?: SessionListOptions): SessionRecord[];
};

export type BorgFacades = {
  stream: BorgStreamFacade;
  episodic: BorgEpisodicFacade;
  self: BorgSelfFacade;
  skills: BorgSkillsFacade;
  mood: BorgMoodFacade;
  actions: BorgActionsFacade;
  social: BorgSocialFacade;
  entities: BorgEntitiesFacade;
  sharedState: BorgSharedStateFacade;
  attachments: BorgAttachmentsFacade;
  semantic: BorgSemanticFacade;
  relationalSlots: BorgRelationalSlotsFacade;
  commitments: BorgCommitmentsFacade;
  operatorAttention: BorgOperatorAttentionFacade;
  activity: BorgActivityFacade;
  creatorDirectives: BorgCreatorDirectivesFacade;
  identity: BorgIdentityFacade;
  correction: BorgCorrectionFacade;
  review: BorgReviewFacade;
  audit: BorgAuditFacade;
  dream: BorgDreamFacade;
  autonomy: BorgAutonomyFacade;
  maintenance: BorgMaintenanceFacade;
  inbox: BorgInboxFacade;
  workmem: BorgWorkmemFacade;
  prompts: BorgPromptsFacade;
  sessions: BorgSessionsFacade;
};
