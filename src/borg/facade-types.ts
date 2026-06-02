// Public facade property types exposed by the Borg class.

import type { AutonomyScheduler, AutonomyWakesRepository } from "../autonomy/index.js";
import type {
  MaintenanceScheduler,
  ReviewRevalidationOptions,
  ReviewRevalidationResult,
} from "../offline/index.js";
import type { CorrectionService } from "../correction/index.js";
import type { MoodRepository } from "../memory/affective/index.js";
import type { ActionRepository } from "../memory/actions/index.js";
import type {
  BorgRole,
  CommitmentRepository,
  EntityRepository,
} from "../memory/commitments/index.js";
import type { CreatorDirectiveRepository } from "../memory/creator-directives/index.js";
import type { Provenance } from "../memory/common/index.js";
import type { EpisodicRepository, ExtractFromStreamResult } from "../memory/episodic/index.js";
import type { IdentityService } from "../memory/identity/index.js";
import type { SkillRepository, SkillSelector } from "../memory/procedural/index.js";
import type { RelationalSlotRepository } from "../memory/relational-slots/index.js";
import type {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
} from "../memory/self/index.js";
import type {
  ReviewKind,
  ReviewQueueItem,
  ReviewResolveOptions,
  ReviewResolutionInput,
  SemanticEdgeRepository,
  SemanticGraph,
  SemanticNode,
  SemanticNodeRepository,
  SemanticNodeSearchCandidate,
} from "../memory/semantic/index.js";
import type { SemanticExtractor } from "../memory/semantic/index.js";
import type { SocialRepository } from "../memory/social/index.js";
import type { WorkingMemory, WorkingMemoryStore } from "../memory/working/index.js";
import type { ChatResponseCatchUpWorker } from "../cognition/ingestion/index.js";
import type { PromptKey } from "../cognition/prompts/registry.js";
import type { OfflineProcessName } from "../offline/index.js";
import type { RetrievedEpisode, RetrievalSearchOptions } from "../retrieval/index.js";
import type {
  SessionEnsureInput,
  SessionListOptions,
  SessionParticipationPolicy,
  SessionRecord,
  SessionTouchUpdate,
} from "../sessions/index.js";
import type { StreamCursor, StreamEntry, StreamEntryInput, StreamReader } from "../stream/index.js";
import type {
  AuditId,
  AttachmentId,
  AutobiographicalPeriodId,
  EntityId,
  EpisodeId,
  MaintenanceRunId,
  SessionId,
} from "../util/ids.js";
import type {
  BorgDependencies,
  BorgDreamRunner,
  BorgEpisodeGetOptions,
  BorgEpisodeSearchOptions,
} from "./types.js";

export type BorgStreamFacade = {
  append: (input: StreamEntryInput, options?: { session?: SessionId }) => Promise<StreamEntry>;
  tail: (n: number, options?: { session?: SessionId }) => StreamEntry[];
  reader: (options?: { session?: SessionId }) => StreamReader;
};

export type BorgEpisodicFacade = {
  get: (id: EpisodeId, options?: BorgEpisodeGetOptions) => Promise<RetrievedEpisode | null>;
  search: (query: string, options?: BorgEpisodeSearchOptions) => Promise<RetrievedEpisode[]>;
  extract: (options?: {
    sinceTs?: number;
    sinceCursor?: StreamCursor;
    untilTs?: number;
    session?: SessionId;
  }) => Promise<ExtractFromStreamResult>;
  list: (...args: Parameters<EpisodicRepository["list"]>) => ReturnType<EpisodicRepository["list"]>;
  listAll: () => ReturnType<EpisodicRepository["listAll"]>;
  getStats: (
    ...args: Parameters<EpisodicRepository["getStats"]>
  ) => ReturnType<EpisodicRepository["getStats"]>;
};

type AutobiographicalUpsertPeriodInput = Parameters<IdentityService["addPeriod"]>[0];

type BorgAutobiographicalUpsertPeriod = {
  (
    input: AutobiographicalUpsertPeriodInput & { id?: undefined },
  ): ReturnType<IdentityService["addPeriod"]>;
  (
    input: AutobiographicalUpsertPeriodInput & { id: AutobiographicalPeriodId },
  ): ReturnType<IdentityService["addPeriod"]> | ReturnType<IdentityService["updatePeriod"]>;
  (
    input: AutobiographicalUpsertPeriodInput,
  ): ReturnType<IdentityService["addPeriod"]> | ReturnType<IdentityService["updatePeriod"]>;
};

export type BorgSelfFacade = {
  values: {
    get: (...args: Parameters<ValuesRepository["get"]>) => ReturnType<ValuesRepository["get"]>;
    list: (...args: Parameters<ValuesRepository["list"]>) => ReturnType<ValuesRepository["list"]>;
    add: (
      ...args: Parameters<IdentityService["addValue"]>
    ) => ReturnType<IdentityService["addValue"]>;
    update: (
      ...args: Parameters<IdentityService["updateValue"]>
    ) => ReturnType<IdentityService["updateValue"]>;
    reinforce: (
      ...args: Parameters<IdentityService["reinforceValue"]>
    ) => ReturnType<IdentityService["reinforceValue"]>;
    listReinforcementEvents: (
      ...args: Parameters<ValuesRepository["listReinforcementEvents"]>
    ) => ReturnType<ValuesRepository["listReinforcementEvents"]>;
    listContradictionEvents: (
      ...args: Parameters<ValuesRepository["listContradictionEvents"]>
    ) => ReturnType<ValuesRepository["listContradictionEvents"]>;
  };
  goals: {
    get: (...args: Parameters<GoalsRepository["get"]>) => ReturnType<GoalsRepository["get"]>;
    list: (...args: Parameters<GoalsRepository["list"]>) => ReturnType<GoalsRepository["list"]>;
    add: (
      ...args: Parameters<IdentityService["addGoal"]>
    ) => ReturnType<IdentityService["addGoal"]>;
    update: (
      ...args: Parameters<IdentityService["updateGoal"]>
    ) => ReturnType<IdentityService["updateGoal"]>;
    updateStatus: (
      ...args: Parameters<IdentityService["updateGoalStatus"]>
    ) => ReturnType<IdentityService["updateGoalStatus"]>;
    updateProgress: (
      ...args: Parameters<IdentityService["updateGoalProgress"]>
    ) => ReturnType<IdentityService["updateGoalProgress"]>;
  };
  traits: {
    get: (...args: Parameters<TraitsRepository["get"]>) => ReturnType<TraitsRepository["get"]>;
    list: (...args: Parameters<TraitsRepository["list"]>) => ReturnType<TraitsRepository["list"]>;
    add: (
      ...args: Parameters<IdentityService["addTrait"]>
    ) => ReturnType<IdentityService["addTrait"]>;
    update: (
      ...args: Parameters<IdentityService["updateTrait"]>
    ) => ReturnType<IdentityService["updateTrait"]>;
    reinforce: (
      ...args: Parameters<IdentityService["reinforceTrait"]>
    ) => ReturnType<IdentityService["reinforceTrait"]>;
    listReinforcementEvents: (
      ...args: Parameters<TraitsRepository["listReinforcementEvents"]>
    ) => ReturnType<TraitsRepository["listReinforcementEvents"]>;
    listContradictionEvents: (
      ...args: Parameters<TraitsRepository["listContradictionEvents"]>
    ) => ReturnType<TraitsRepository["listContradictionEvents"]>;
  };
  autobiographical: {
    currentPeriod: () => ReturnType<AutobiographicalRepository["currentPeriod"]>;
    listPeriods: (
      ...args: Parameters<AutobiographicalRepository["listPeriods"]>
    ) => ReturnType<AutobiographicalRepository["listPeriods"]>;
    upsertPeriod: BorgAutobiographicalUpsertPeriod;
    closePeriod: (
      ...args: Parameters<IdentityService["closePeriod"]>
    ) => ReturnType<IdentityService["closePeriod"]>;
    getPeriod: (
      ...args: Parameters<AutobiographicalRepository["getPeriod"]>
    ) => ReturnType<AutobiographicalRepository["getPeriod"]>;
    getByLabel: (
      ...args: Parameters<AutobiographicalRepository["getByLabel"]>
    ) => ReturnType<AutobiographicalRepository["getByLabel"]>;
  };
  growthMarkers: {
    list: (
      ...args: Parameters<GrowthMarkersRepository["list"]>
    ) => ReturnType<GrowthMarkersRepository["list"]>;
    add: (
      ...args: Parameters<IdentityService["addGrowthMarker"]>
    ) => ReturnType<IdentityService["addGrowthMarker"]>;
    summarize: (
      ...args: Parameters<GrowthMarkersRepository["summarize"]>
    ) => ReturnType<GrowthMarkersRepository["summarize"]>;
  };
  openQuestions: {
    list: (
      ...args: Parameters<OpenQuestionsRepository["list"]>
    ) => ReturnType<OpenQuestionsRepository["list"]>;
    add: (
      ...args: Parameters<IdentityService["addOpenQuestion"]>
    ) => ReturnType<IdentityService["addOpenQuestion"]>;
    resolve: (
      ...args: Parameters<IdentityService["resolveOpenQuestion"]>
    ) => ReturnType<IdentityService["resolveOpenQuestion"]>;
    abandon: (
      ...args: Parameters<IdentityService["abandonOpenQuestion"]>
    ) => ReturnType<IdentityService["abandonOpenQuestion"]>;
    bumpUrgency: (
      ...args: Parameters<IdentityService["bumpOpenQuestionUrgency"]>
    ) => ReturnType<IdentityService["bumpOpenQuestionUrgency"]>;
  };
};

export type BorgSkillsFacade = {
  list: (...args: Parameters<SkillRepository["list"]>) => ReturnType<SkillRepository["list"]>;
  add: (...args: Parameters<SkillRepository["add"]>) => ReturnType<SkillRepository["add"]>;
  get: (...args: Parameters<SkillRepository["get"]>) => ReturnType<SkillRepository["get"]>;
  searchByContext: (
    ...args: Parameters<SkillRepository["searchByContext"]>
  ) => ReturnType<SkillRepository["searchByContext"]>;
  recordOutcome: (
    ...args: Parameters<SkillRepository["recordOutcome"]>
  ) => ReturnType<SkillRepository["recordOutcome"]>;
  select: (...args: Parameters<SkillSelector["select"]>) => ReturnType<SkillSelector["select"]>;
};

export type BorgMoodFacade = {
  current: (
    ...args: Parameters<MoodRepository["current"]>
  ) => ReturnType<MoodRepository["current"]>;
  history: (
    ...args: Parameters<MoodRepository["history"]>
  ) => ReturnType<MoodRepository["history"]>;
  update: (...args: Parameters<MoodRepository["update"]>) => ReturnType<MoodRepository["update"]>;
};

export type BorgSocialFacade = {
  getProfile: (entity: string) => ReturnType<SocialRepository["getProfile"]>;
  list: (...args: Parameters<SocialRepository["list"]>) => ReturnType<SocialRepository["list"]>;
  upsertProfile: (entity: string) => ReturnType<SocialRepository["upsertProfile"]>;
  recordInteraction: (
    entity: string,
    interaction: Parameters<SocialRepository["recordInteraction"]>[1],
  ) => ReturnType<SocialRepository["recordInteraction"]>;
  adjustTrust: (
    entity: string,
    delta: number,
    provenance: Provenance,
  ) => ReturnType<SocialRepository["adjustTrust"]>;
};

export type BorgEntitiesFacade = {
  resolve: (
    ...args: Parameters<EntityRepository["resolve"]>
  ) => ReturnType<EntityRepository["resolve"]>;
  get: (...args: Parameters<EntityRepository["get"]>) => ReturnType<EntityRepository["get"]>;
  list: (...args: Parameters<EntityRepository["list"]>) => ReturnType<EntityRepository["list"]>;
  getCreator: () => ReturnType<EntityRepository["getCreator"]>;
  setBorgRole: (id: EntityId, role: BorgRole | null) => ReturnType<EntityRepository["setBorgRole"]>;
  find: (
    name: string,
    options?: Parameters<EntityRepository["findByName"]>[1],
  ) => ReturnType<EntityRepository["get"]>;
};

export type BorgSharedStateFacade = {
  getForAudience: (
    audience: string,
  ) => ReturnType<BorgDependencies["sharedStateRepository"]["get"]>;
  listEntriesForAudience: (
    audience: string,
  ) => NonNullable<ReturnType<BorgDependencies["sharedStateRepository"]["get"]>>["entries"];
};

export type BorgAttachmentBytesResult = {
  attachment: NonNullable<ReturnType<BorgDependencies["attachmentRepository"]["get"]>>;
  mediaType: string;
  bytes: Uint8Array;
};

export type BorgAttachmentMetadataResult = {
  attachment: NonNullable<ReturnType<BorgDependencies["attachmentRepository"]["get"]>>;
  perception: ReturnType<BorgDependencies["imagePerceptionRepository"]["get"]>;
  status: {
    active: boolean;
    quarantined: boolean;
    stream_active?: boolean;
    parent_active?: boolean;
  };
};

export type BorgAttachmentsFacade = {
  get: (attachmentId: AttachmentId) => BorgAttachmentMetadataResult | null;
  getBytes: (
    attachmentId: AttachmentId,
    options?: { audience?: string | null },
  ) => BorgAttachmentBytesResult | null;
};

export type BorgSemanticFacade = {
  nodes: {
    add: (input: {
      kind: SemanticNode["kind"];
      label: string;
      description: string;
      domain?: string | null;
      aliases?: string[];
      confidence?: number;
      sourceEpisodeIds: SemanticNode["source_episode_ids"];
    }) => Promise<SemanticNode>;
    get: (id: SemanticNode["id"]) => Promise<SemanticNode | null>;
    list: (
      ...args: Parameters<SemanticNodeRepository["list"]>
    ) => ReturnType<SemanticNodeRepository["list"]>;
    listPage: (
      ...args: Parameters<SemanticNodeRepository["listPage"]>
    ) => ReturnType<SemanticNodeRepository["listPage"]>;
    countByStatus: () => ReturnType<SemanticNodeRepository["countByStatus"]>;
    search: (
      query: string,
      options?: Omit<RetrievalSearchOptions, "temporalCue" | "attentionWeights" | "asOf"> & {
        limit?: number;
      },
    ) => Promise<SemanticNodeSearchCandidate[]>;
  };
  edges: {
    add: (
      input: Parameters<SemanticEdgeRepository["addEdge"]>[0],
    ) => ReturnType<SemanticEdgeRepository["addEdge"]>;
    get: (
      id: Parameters<SemanticEdgeRepository["getEdge"]>[0],
    ) => ReturnType<SemanticEdgeRepository["getEdge"]>;
    list: (
      ...args: Parameters<BorgDependencies["semanticEdgeRepository"]["listEdges"]>
    ) => ReturnType<BorgDependencies["semanticEdgeRepository"]["listEdges"]>;
  };
  walk: (
    fromId: SemanticNode["id"],
    ...args: Parameters<SemanticGraph["walk"]> extends [unknown, ...infer Rest] ? Rest : never
  ) => ReturnType<SemanticGraph["walk"]>;
  extract: (
    episodes: readonly Parameters<SemanticExtractor["extractFromEpisodes"]>[0][number][],
  ) => Promise<Awaited<ReturnType<SemanticExtractor["extractFromEpisodes"]>>>;
};

export type BorgCommitmentsFacade = {
  add: (input: {
    type: Parameters<CommitmentRepository["add"]>[0]["type"];
    kind?: Parameters<CommitmentRepository["add"]>[0]["kind"];
    enforcementClass?: Parameters<CommitmentRepository["add"]>[0]["enforcementClass"];
    criticalDomain?: Parameters<CommitmentRepository["add"]>[0]["criticalDomain"];
    directiveFamily: string;
    directive: string;
    priority: number;
    madeTo?: string | null;
    audience?: string | null;
    about?: string | null;
    provenance: Provenance;
    expiresAt?: number | null;
  }) => ReturnType<CommitmentRepository["add"]>;
  revoke: (
    ...args: Parameters<CommitmentRepository["revoke"]>
  ) => ReturnType<CommitmentRepository["revoke"]>;
  list: (options?: {
    activeOnly?: boolean;
    audience?: string | null;
    aboutEntity?: string | null;
  }) => ReturnType<CommitmentRepository["list"]>;
  countActive: () => ReturnType<CommitmentRepository["countActive"]>;
  countActiveByKind: () => ReturnType<CommitmentRepository["countActiveByKind"]>;
  countActiveByEnforcementClass: () => ReturnType<
    CommitmentRepository["countActiveByEnforcementClass"]
  >;
  countSuperseded: () => ReturnType<CommitmentRepository["countSuperseded"]>;
  countRevoked: () => ReturnType<CommitmentRepository["countRevoked"]>;
  countExpired: () => ReturnType<CommitmentRepository["countExpired"]>;
  countCanonicalized: () => ReturnType<CommitmentRepository["countCanonicalized"]>;
};

export type BorgCreatorDirectivesFacade = {
  queue: (
    ...args: Parameters<CreatorDirectiveRepository["queue"]>
  ) => ReturnType<CreatorDirectiveRepository["queue"]>;
  get: (
    ...args: Parameters<CreatorDirectiveRepository["get"]>
  ) => ReturnType<CreatorDirectiveRepository["get"]>;
  list: (
    ...args: Parameters<CreatorDirectiveRepository["list"]>
  ) => ReturnType<CreatorDirectiveRepository["list"]>;
  listApplicable: (
    ...args: Parameters<CreatorDirectiveRepository["listApplicable"]>
  ) => ReturnType<CreatorDirectiveRepository["listApplicable"]>;
  supersede: (
    ...args: Parameters<CreatorDirectiveRepository["supersede"]>
  ) => ReturnType<CreatorDirectiveRepository["supersede"]>;
  supersedeFamilyAtomic: (
    ...args: Parameters<CreatorDirectiveRepository["supersedeFamilyAtomic"]>
  ) => ReturnType<CreatorDirectiveRepository["supersedeFamilyAtomic"]>;
  revoke: (
    ...args: Parameters<CreatorDirectiveRepository["revoke"]>
  ) => ReturnType<CreatorDirectiveRepository["revoke"]>;
};

export type BorgIdentityFacade = {
  updateValue: (
    ...args: Parameters<IdentityService["updateValue"]>
  ) => ReturnType<IdentityService["updateValue"]>;
  updateGoal: (
    ...args: Parameters<IdentityService["updateGoal"]>
  ) => ReturnType<IdentityService["updateGoal"]>;
  updateTrait: (
    ...args: Parameters<IdentityService["updateTrait"]>
  ) => ReturnType<IdentityService["updateTrait"]>;
  addCommitment: (
    ...args: Parameters<IdentityService["addCommitment"]>
  ) => ReturnType<IdentityService["addCommitment"]>;
  updateCommitment: (
    ...args: Parameters<IdentityService["updateCommitment"]>
  ) => ReturnType<IdentityService["updateCommitment"]>;
  updatePeriod: (
    ...args: Parameters<IdentityService["updatePeriod"]>
  ) => ReturnType<IdentityService["updatePeriod"]>;
  updateGrowthMarker: (
    ...args: Parameters<IdentityService["updateGrowthMarker"]>
  ) => ReturnType<IdentityService["updateGrowthMarker"]>;
  updateOpenQuestion: (
    ...args: Parameters<IdentityService["updateOpenQuestion"]>
  ) => ReturnType<IdentityService["updateOpenQuestion"]>;
  listEvents: (
    ...args: Parameters<IdentityService["listEvents"]>
  ) => ReturnType<IdentityService["listEvents"]>;
};

export type BorgCorrectionFacade = {
  forget: (
    ...args: Parameters<CorrectionService["forget"]>
  ) => ReturnType<CorrectionService["forget"]>;
  why: (...args: Parameters<CorrectionService["why"]>) => ReturnType<CorrectionService["why"]>;
  invalidateSemanticEdge: (
    ...args: Parameters<CorrectionService["invalidateSemanticEdge"]>
  ) => ReturnType<CorrectionService["invalidateSemanticEdge"]>;
  correct: (
    ...args: Parameters<CorrectionService["correct"]>
  ) => ReturnType<CorrectionService["correct"]>;
  rememberAboutMe: (
    ...args: Parameters<CorrectionService["rememberAboutMe"]>
  ) => ReturnType<CorrectionService["rememberAboutMe"]>;
  listIdentityEvents: (
    ...args: Parameters<CorrectionService["listIdentityEvents"]>
  ) => ReturnType<CorrectionService["listIdentityEvents"]>;
};

export type BorgActionsFacade = ActionRepository;

export type BorgRelationalSlotsFacade = {
  list: (
    ...args: Parameters<RelationalSlotRepository["list"]>
  ) => ReturnType<RelationalSlotRepository["list"]>;
  countByState: () => ReturnType<RelationalSlotRepository["countByState"]>;
};

export type BorgReviewFacade = {
  list: (options?: { kind?: ReviewKind; openOnly?: boolean }) => ReviewQueueItem[];
  resolve: (
    id: number,
    decision: ReviewResolutionInput,
    options?: ReviewResolveOptions,
  ) => Promise<ReviewQueueItem | null>;
  revalidate: (options: ReviewRevalidationOptions) => Promise<ReviewRevalidationResult>;
};

export type BorgAuditFacade = {
  list: (options?: {
    runId?: MaintenanceRunId;
    process?: OfflineProcessName;
    reverted?: boolean;
  }) => ReturnType<BorgDependencies["auditLog"]["list"]>;
  revert: (id: AuditId, revertedBy?: string) => ReturnType<BorgDependencies["auditLog"]["revert"]>;
};

export type BorgDreamFacade = BorgDreamRunner;

export type BorgAutonomyFacade = {
  scheduler: AutonomyScheduler;
  wakes: AutonomyWakesRepository;
};

export type BorgMaintenanceFacade = {
  scheduler: MaintenanceScheduler;
  config: () => {
    enabled: boolean;
    lightIntervalMs: number;
    heavyIntervalMs: number;
    lightProcesses: readonly OfflineProcessName[];
    heavyProcesses: readonly OfflineProcessName[];
    processBudgets: Partial<Record<OfflineProcessName, number | null>>;
  };
};

export type BorgInboxFacade = {
  catchUp: ChatResponseCatchUpWorker;
};

export type BorgWorkmemFacade = {
  load: (sessionId?: SessionId) => WorkingMemory;
  clear: (sessionId?: SessionId) => void;
  getPendingActionMergeCount: () => ReturnType<WorkingMemoryStore["getPendingActionMergeCount"]>;
};

export type BorgPromptBlockView = {
  key: PromptKey;
  label: string;
  description: string;
  default_text: string;
  current_text: string;
  overridden: boolean;
  updated_at: number | null;
};

export type BorgAssembledFramingPromptPreview = {
  text: string;
  sections: readonly string[];
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
