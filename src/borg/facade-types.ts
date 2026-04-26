// Public facade property types exposed by the Borg class.

import type { AutonomyScheduler, AutonomyWakesRepository } from "../autonomy/index.js";
import type { MaintenanceScheduler } from "../offline/index.js";
import type { CorrectionService } from "../correction/index.js";
import type { MoodRepository } from "../memory/affective/index.js";
import type { CommitmentRepository } from "../memory/commitments/index.js";
import type { Provenance } from "../memory/common/index.js";
import type { EpisodicRepository, ExtractFromStreamResult } from "../memory/episodic/index.js";
import type { IdentityService } from "../memory/identity/index.js";
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
  ReviewKind,
  ReviewQueueItem,
  ReviewResolutionInput,
  SemanticEdge,
  SemanticEdgeRepository,
  SemanticGraph,
  SemanticNode,
  SemanticNodeRepository,
  SemanticNodeSearchCandidate,
} from "../memory/semantic/index.js";
import type { SemanticExtractor } from "../memory/semantic/index.js";
import type { SocialRepository } from "../memory/social/index.js";
import type { WorkingMemory } from "../memory/working/index.js";
import type { OfflineProcessName } from "../offline/index.js";
import type { RetrievedEpisode, RetrievalSearchOptions } from "../retrieval/index.js";
import type { StreamCursor, StreamEntry, StreamEntryInput, StreamReader } from "../stream/index.js";
import type { AuditId, EpisodeId, MaintenanceRunId, SessionId } from "../util/ids.js";
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
};

export type BorgSelfFacade = {
  values: ValuesRepository;
  goals: GoalsRepository;
  traits: TraitsRepository;
  autobiographical: {
    currentPeriod: () => ReturnType<AutobiographicalRepository["currentPeriod"]>;
    listPeriods: (
      ...args: Parameters<AutobiographicalRepository["listPeriods"]>
    ) => ReturnType<AutobiographicalRepository["listPeriods"]>;
    upsertPeriod: (
      ...args: Parameters<AutobiographicalRepository["upsertPeriod"]>
    ) => ReturnType<AutobiographicalRepository["upsertPeriod"]>;
    closePeriod: (
      ...args: Parameters<AutobiographicalRepository["closePeriod"]>
    ) => ReturnType<AutobiographicalRepository["closePeriod"]>;
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
      ...args: Parameters<GrowthMarkersRepository["add"]>
    ) => ReturnType<GrowthMarkersRepository["add"]>;
    summarize: (
      ...args: Parameters<GrowthMarkersRepository["summarize"]>
    ) => ReturnType<GrowthMarkersRepository["summarize"]>;
  };
  openQuestions: {
    list: (
      ...args: Parameters<OpenQuestionsRepository["list"]>
    ) => ReturnType<OpenQuestionsRepository["list"]>;
    add: (
      ...args: Parameters<OpenQuestionsRepository["add"]>
    ) => ReturnType<OpenQuestionsRepository["add"]>;
    resolve: (
      ...args: Parameters<OpenQuestionsRepository["resolve"]>
    ) => ReturnType<OpenQuestionsRepository["resolve"]>;
    abandon: (
      ...args: Parameters<OpenQuestionsRepository["abandon"]>
    ) => ReturnType<OpenQuestionsRepository["abandon"]>;
    bumpUrgency: (
      ...args: Parameters<OpenQuestionsRepository["bumpUrgency"]>
    ) => ReturnType<OpenQuestionsRepository["bumpUrgency"]>;
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

export type BorgFacades = {
  stream: BorgStreamFacade;
  episodic: BorgEpisodicFacade;
  self: BorgSelfFacade;
  skills: BorgSkillsFacade;
  mood: BorgMoodFacade;
  social: BorgSocialFacade;
  semantic: BorgSemanticFacade;
  commitments: BorgCommitmentsFacade;
  identity: BorgIdentityFacade;
  correction: BorgCorrectionFacade;
  review: {
    list: (options?: { kind?: ReviewKind; openOnly?: boolean }) => ReviewQueueItem[];
    resolve: (id: number, decision: ReviewResolutionInput) => Promise<ReviewQueueItem | null>;
  };
  audit: {
    list: (options?: {
      runId?: MaintenanceRunId;
      process?: OfflineProcessName;
      reverted?: boolean;
    }) => ReturnType<BorgDependencies["auditLog"]["list"]>;
    revert: (
      id: AuditId,
      revertedBy?: string,
    ) => ReturnType<BorgDependencies["auditLog"]["revert"]>;
  };
  dream: BorgDreamRunner;
  autonomy: {
    scheduler: AutonomyScheduler;
    wakes: AutonomyWakesRepository;
  };
  maintenance: {
    scheduler: MaintenanceScheduler;
  };
  workmem: {
    load: (sessionId?: SessionId) => WorkingMemory;
    clear: (sessionId?: SessionId) => void;
  };
};
