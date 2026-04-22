import { join } from "node:path";

import { TurnOrchestrator, type TurnInput, type TurnResult } from "./cognition/index.js";
import { TurnContextCompiler } from "./cognition/recency/index.js";
import { DEFAULT_CONFIG, configSchema, loadConfig, type Config } from "./config/index.js";
import { CorrectionService } from "./correction/index.js";
import { OpenAICompatibleEmbeddingClient, type EmbeddingClient } from "./embeddings/index.js";
import { AnthropicLLMClient, type LLMClient } from "./llm/index.js";
import { MoodRepository, affectiveMigrations } from "./memory/affective/index.js";
import type { Provenance } from "./memory/common/index.js";
import {
  CommitmentRepository,
  EntityRepository,
  commitmentMigrations,
} from "./memory/commitments/index.js";
import {
  EpisodicExtractor,
  EpisodicRepository,
  type ExtractFromStreamResult,
  createEpisodesTableSchema,
  episodicMigrations,
} from "./memory/episodic/index.js";
import {
  SkillRepository,
  SkillSelector,
  createSkillsTableSchema,
  proceduralMigrations,
  type SkillSelectionResult,
} from "./memory/procedural/index.js";
import {
  IdentityEventRepository,
  IdentityService,
  identityMigrations,
} from "./memory/identity/index.js";
import {
  ReviewQueueRepository,
  SemanticEdgeRepository,
  SemanticExtractor,
  SemanticGraph,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
  semanticMigrations,
  type ExtractSemanticResult,
  type ReviewKind,
  type ReviewQueueItem,
  type ReviewResolution,
  type SemanticEdge,
  type SemanticNode,
  type SemanticNodeSearchCandidate,
} from "./memory/semantic/index.js";
import type {
  RetrievedEpisode,
  RetrievalGetEpisodeOptions,
  RetrievalSearchOptions,
} from "./retrieval/index.js";
import {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
  selfMigrations,
} from "./memory/self/index.js";
import { SocialRepository, socialMigrations } from "./memory/social/index.js";
import {
  appendOpenQuestionHookFailureEvent,
  enqueueOpenQuestionForReview,
} from "./memory/self/review-open-question-hook.js";
import { WorkingMemoryStore, type WorkingMemory } from "./memory/working/index.js";
import {
  AuditLog,
  ConsolidatorProcess,
  CuratorProcess,
  MaintenanceOrchestrator,
  OverseerProcess,
  ReflectorProcess,
  ReverserRegistry,
  RuminatorProcess,
  SelfNarratorProcess,
  type MaintenancePlan,
  offlineMigrations,
  type OfflineProcess,
  type OfflineProcessName,
  type OrchestratorResult,
} from "./offline/index.js";
import { retrievalMigrations, RetrievalPipeline } from "./retrieval/index.js";
import {
  StreamReader,
  StreamWatermarkRepository,
  StreamWriter,
  streamWatermarkMigrations,
  type StreamCursor,
  type StreamEntry,
  type StreamEntryInput,
} from "./stream/index.js";
import { StreamIngestionCoordinator } from "./cognition/ingestion/index.js";
import { LanceDbStore } from "./storage/lancedb/index.js";
import { openDatabase, type Migration, type SqliteDatabase } from "./storage/sqlite/index.js";
import { SystemClock, type Clock } from "./util/clock.js";
import {
  DEFAULT_SESSION_ID,
  type AuditId,
  type EntityId,
  type MaintenanceRunId,
  createSemanticNodeId,
  type EpisodeId,
  type SessionId,
} from "./util/ids.js";

type BorgDependencies = {
  config: Config;
  sqlite: SqliteDatabase;
  lance: LanceDbStore;
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
  /**
   * When true, every completed turn triggers watermark-based episodic
   * extraction so the next turn's retrieval sees material from the turn
   * that just ran. Defaults to false: the existing test suite uses fake
   * LLMs with scripted response queues, and live extraction would consume
   * responses out of band. Production callers (scripts/chat.ts,
   * scripts/debug.ts when using real clients) opt in explicitly.
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

function createEmbeddingClient(config: Config): EmbeddingClient {
  return new OpenAICompatibleEmbeddingClient({
    baseUrl: config.embedding.baseUrl,
    apiKey: config.embedding.apiKey,
    model: config.embedding.model,
    dims: config.embedding.dims,
  });
}

function createLlmFactory(
  config: Config,
  llmClient: LLMClient | undefined,
  env: NodeJS.ProcessEnv | undefined,
  clock: Clock,
): () => LLMClient {
  if (llmClient !== undefined) {
    return () => llmClient;
  }

  let cached: LLMClient | undefined;

  return () => {
    cached ??= new AnthropicLLMClient({
      authMode: config.anthropic.auth,
      apiKey: config.anthropic.apiKey,
      env,
      clock,
    });
    return cached;
  };
}

function createLazyLlmClient(factory: () => LLMClient): LLMClient {
  return {
    complete(options) {
      return factory().complete(options);
    },
  };
}

function createMigrations(): Migration[] {
  return [
    ...episodicMigrations,
    ...selfMigrations,
    ...identityMigrations,
    ...affectiveMigrations,
    ...retrievalMigrations,
    ...semanticMigrations,
    ...commitmentMigrations,
    ...socialMigrations,
    ...proceduralMigrations,
    ...offlineMigrations,
    ...streamWatermarkMigrations,
  ];
}

function quarterLabel(timestamp: number): string {
  const date = new Date(timestamp);
  const year = date.getUTCFullYear();
  const quarter = Math.floor(date.getUTCMonth() / 3) + 1;
  return `${year}-Q${quarter}`;
}

async function closeBestEffort(
  sqlite: SqliteDatabase | undefined,
  lance: LanceDbStore | undefined,
): Promise<void> {
  if (sqlite !== undefined) {
    try {
      sqlite.close();
    } catch {
      // Best-effort cleanup after a partial Borg.open failure.
    }
  }

  if (lance !== undefined) {
    try {
      await lance.close();
    } catch {
      // Best-effort cleanup after a partial Borg.open failure.
    }
  }
}

export class Borg {
  readonly stream: {
    append: (input: StreamEntryInput, options?: { session?: SessionId }) => Promise<StreamEntry>;
    tail: (n: number, options?: { session?: SessionId }) => StreamEntry[];
    reader: (options?: { session?: SessionId }) => StreamReader;
  };

  readonly episodic: {
    get: (id: EpisodeId, options?: BorgEpisodeGetOptions) => Promise<RetrievedEpisode | null>;
    search: (query: string, options?: BorgEpisodeSearchOptions) => Promise<RetrievedEpisode[]>;
    extract: (options?: {
      sinceTs?: number;
      sinceCursor?: StreamCursor;
      untilTs?: number;
      session?: SessionId;
    }) => Promise<ExtractFromStreamResult>;
    list: (
      ...args: Parameters<EpisodicRepository["list"]>
    ) => ReturnType<EpisodicRepository["list"]>;
  };

  readonly self: {
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
  readonly skills: {
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
  readonly mood: {
    current: (
      ...args: Parameters<MoodRepository["current"]>
    ) => ReturnType<MoodRepository["current"]>;
    history: (
      ...args: Parameters<MoodRepository["history"]>
    ) => ReturnType<MoodRepository["history"]>;
    update: (...args: Parameters<MoodRepository["update"]>) => ReturnType<MoodRepository["update"]>;
  };
  readonly social: {
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
  readonly semantic: {
    nodes: {
      add: (input: {
        kind: SemanticNode["kind"];
        label: string;
        description: string;
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
        options?: Omit<RetrievalSearchOptions, "temporalCue" | "attentionWeights"> & {
          limit?: number;
        },
      ) => Promise<SemanticNodeSearchCandidate[]>;
    };
    edges: {
      add: (input: Omit<SemanticEdge, "id"> & { id?: SemanticEdge["id"] }) => SemanticEdge;
      list: (
        ...args: Parameters<SemanticEdgeRepository["listEdges"]>
      ) => ReturnType<SemanticEdgeRepository["listEdges"]>;
    };
    walk: (
      fromId: SemanticNode["id"],
      ...args: Parameters<SemanticGraph["walk"]> extends [unknown, ...infer Rest] ? Rest : never
    ) => ReturnType<SemanticGraph["walk"]>;
    extract: (
      episodes: readonly Parameters<SemanticExtractor["extractFromEpisodes"]>[0][number][],
    ) => Promise<ExtractSemanticResult>;
  };
  readonly commitments: {
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
  readonly identity: {
    updateValue: (...args: Parameters<IdentityService["updateValue"]>) => ReturnType<IdentityService["updateValue"]>;
    updateTrait: (...args: Parameters<IdentityService["updateTrait"]>) => ReturnType<IdentityService["updateTrait"]>;
    updateCommitment: (
      ...args: Parameters<IdentityService["updateCommitment"]>
    ) => ReturnType<IdentityService["updateCommitment"]>;
    listEvents: (...args: Parameters<IdentityService["listEvents"]>) => ReturnType<IdentityService["listEvents"]>;
  };
  readonly correction: {
    forget: (...args: Parameters<CorrectionService["forget"]>) => ReturnType<CorrectionService["forget"]>;
    why: (...args: Parameters<CorrectionService["why"]>) => ReturnType<CorrectionService["why"]>;
    correct: (...args: Parameters<CorrectionService["correct"]>) => ReturnType<CorrectionService["correct"]>;
    rememberAboutMe: (
      ...args: Parameters<CorrectionService["rememberAboutMe"]>
    ) => ReturnType<CorrectionService["rememberAboutMe"]>;
    listIdentityEvents: (
      ...args: Parameters<CorrectionService["listIdentityEvents"]>
    ) => ReturnType<CorrectionService["listIdentityEvents"]>;
  };
  readonly review: {
    list: (options?: { kind?: ReviewKind; openOnly?: boolean }) => ReviewQueueItem[];
    resolve: (id: number, decision: ReviewResolution) => Promise<ReviewQueueItem | null>;
  };
  readonly audit: {
    list: (options?: {
      runId?: MaintenanceRunId;
      process?: OfflineProcessName;
      reverted?: boolean;
    }) => ReturnType<AuditLog["list"]>;
    revert: (id: AuditId, revertedBy?: string) => ReturnType<AuditLog["revert"]>;
  };
  readonly dream: BorgDreamRunner;
  readonly workmem: {
    load: (sessionId?: SessionId) => WorkingMemory;
    clear: (sessionId?: SessionId) => void;
  };

  private constructor(private readonly deps: BorgDependencies) {
    const resolveEpisodeAudienceEntityId = (
      options:
        | {
            audience?: string | null;
            audienceEntityId?: EntityId | null;
          }
        | undefined,
    ): EntityId | null | undefined => {
      if (options?.audienceEntityId !== undefined) {
        return options.audienceEntityId;
      }

      if (options?.audience === undefined) {
        return undefined;
      }

      if (options.audience === null) {
        return null;
      }

      return this.deps.entityRepository.resolve(options.audience);
    };

    const resolveEpisodeAudienceTerms = (
      options: BorgEpisodeSearchOptions | undefined,
      audienceEntityId: EntityId | null | undefined,
    ): readonly string[] | undefined => {
      if (options?.audienceTerms !== undefined) {
        return options.audienceTerms;
      }

      if (audienceEntityId === null || audienceEntityId === undefined) {
        return typeof options?.audience === "string" ? [options.audience] : undefined;
      }

      const audienceEntity = this.deps.entityRepository.get(audienceEntityId);

      if (audienceEntity === null) {
        return typeof options?.audience === "string" ? [options.audience] : undefined;
      }

      return [
        audienceEntity.canonical_name,
        ...audienceEntity.aliases,
        ...(typeof options?.audience === "string" ? [options.audience] : []),
      ];
    };

    const resolveEpisodeSearchOptions = (
      options: BorgEpisodeSearchOptions | undefined,
    ): RetrievalSearchOptions => {
      const audienceEntityId = resolveEpisodeAudienceEntityId(options);
      const audienceProfile =
        options?.audienceProfile !== undefined
          ? options.audienceProfile
          : audienceEntityId === null || audienceEntityId === undefined
            ? undefined
            : this.deps.socialRepository.getProfile(audienceEntityId) ?? undefined;
      const audienceTerms = resolveEpisodeAudienceTerms(options, audienceEntityId);
      const hasTemporalSignal =
        options?.temporalCue !== undefined ||
        options?.timeRange !== undefined;
      const hasEntitySignal =
        options?.entityTerms !== undefined && options.entityTerms.length > 0;

      return {
        ...options,
        audienceEntityId,
        audienceProfile,
        audienceTerms,
        strictTimeRange: options?.strictTimeRange ?? (options?.timeRange !== undefined),
        attentionWeights:
          options?.attentionWeights ??
          (options?.scoreWeights !== undefined
            ? undefined
            : {
                semantic: 0.35,
                goal_relevance:
                  options?.goalDescriptions !== undefined && options.goalDescriptions.length > 0
                    ? 0.1
                    : 0,
                mood: 0,
                time: hasTemporalSignal ? 0.2 : 0,
                social: audienceTerms !== undefined && audienceTerms.length > 0 ? 0.15 : 0,
                entity: hasEntitySignal ? 0.2 : 0,
                heat: 0.45,
                suppression_penalty: 0.5,
              }),
      };
    };

    const defaultDreamProcesses = (): OfflineProcessName[] =>
      Object.entries({
        consolidator: this.deps.config.offline.consolidator.enabled,
        reflector: this.deps.config.offline.reflector.enabled,
        curator: this.deps.config.offline.curator.enabled,
        overseer: this.deps.config.offline.overseer.enabled,
        ruminator: this.deps.config.offline.ruminator.enabled,
        "self-narrator": this.deps.config.offline.selfNarrator.enabled,
      })
        .filter(([, enabled]) => enabled)
        .map(([name]) => name as OfflineProcessName);

    this.stream = {
      append: async (input, options = {}) => {
        const writer = new StreamWriter({
          dataDir: this.deps.config.dataDir,
          sessionId: options.session ?? DEFAULT_SESSION_ID,
          clock: this.deps.clock,
        });

        try {
          return await writer.append(input);
        } finally {
          writer.close();
        }
      },
      tail: (n, options = {}) =>
        new StreamReader({
          dataDir: this.deps.config.dataDir,
          sessionId: options.session ?? DEFAULT_SESSION_ID,
        }).tail(n),
      reader: (options = {}) =>
        new StreamReader({
          dataDir: this.deps.config.dataDir,
          sessionId: options.session ?? DEFAULT_SESSION_ID,
        }),
    };
    this.episodic = {
      get: (id, options = {}) =>
        this.deps.retrievalPipeline.getEpisode(id, {
          audienceEntityId: resolveEpisodeAudienceEntityId(options),
          crossAudience: options.crossAudience,
        }),
      search: (query, options = {}) =>
        this.deps.retrievalPipeline.search(query, resolveEpisodeSearchOptions(options)),
      extract: async (options = {}) => {
        const extractor = new EpisodicExtractor({
          dataDir: this.deps.config.dataDir,
          episodicRepository: this.deps.episodicRepository,
          embeddingClient: this.deps.embeddingClient,
          llmClient: this.deps.llmFactory(),
          model: this.deps.config.anthropic.models.extraction,
          entityRepository: this.deps.entityRepository,
          clock: this.deps.clock,
        });

        return extractor.extractFromStream({
          session: options.session ?? DEFAULT_SESSION_ID,
          sinceTs: options.sinceTs,
          sinceCursor: options.sinceCursor,
          untilTs: options.untilTs,
        });
      },
      list: (...args) => this.deps.episodicRepository.list(...args),
    };
    this.self = {
      values: this.deps.valuesRepository,
      goals: this.deps.goalsRepository,
      traits: this.deps.traitsRepository,
      autobiographical: {
        currentPeriod: () => this.deps.autobiographicalRepository.currentPeriod(),
        listPeriods: (...args) => this.deps.autobiographicalRepository.listPeriods(...args),
        upsertPeriod: (...args) => this.deps.autobiographicalRepository.upsertPeriod(...args),
        closePeriod: (...args) => this.deps.autobiographicalRepository.closePeriod(...args),
        getPeriod: (...args) => this.deps.autobiographicalRepository.getPeriod(...args),
        getByLabel: (...args) => this.deps.autobiographicalRepository.getByLabel(...args),
      },
      growthMarkers: {
        list: (...args) => this.deps.growthMarkersRepository.list(...args),
        add: (...args) => this.deps.growthMarkersRepository.add(...args),
        summarize: (...args) => this.deps.growthMarkersRepository.summarize(...args),
      },
      openQuestions: {
        list: (...args) => this.deps.openQuestionsRepository.list(...args),
        add: (...args) => this.deps.openQuestionsRepository.add(...args),
        resolve: (...args) => this.deps.openQuestionsRepository.resolve(...args),
        abandon: (...args) => this.deps.openQuestionsRepository.abandon(...args),
        bumpUrgency: (...args) => this.deps.openQuestionsRepository.bumpUrgency(...args),
      },
    };
    this.skills = {
      list: (...args) => this.deps.skillRepository.list(...args),
      add: (...args) => this.deps.skillRepository.add(...args),
      get: (...args) => this.deps.skillRepository.get(...args),
      searchByContext: (...args) => this.deps.skillRepository.searchByContext(...args),
      recordOutcome: (...args) => this.deps.skillRepository.recordOutcome(...args),
      select: (...args) => this.deps.skillSelector.select(...args),
    };
    this.mood = {
      current: (...args) => this.deps.moodRepository.current(...args),
      history: (...args) => this.deps.moodRepository.history(...args),
      update: (...args) => this.deps.moodRepository.update(...args),
    };
    this.social = {
      getProfile: (entity) =>
        this.deps.socialRepository.getProfile(this.deps.entityRepository.resolve(entity)),
      upsertProfile: (entity) =>
        this.deps.socialRepository.upsertProfile(this.deps.entityRepository.resolve(entity)),
      recordInteraction: (entity, interaction) =>
        this.deps.socialRepository.recordInteraction(
          this.deps.entityRepository.resolve(entity),
          interaction,
        ),
      adjustTrust: (entity, delta, provenance) =>
        this.deps.socialRepository.adjustTrust(
          this.deps.entityRepository.resolve(entity),
          delta,
          provenance,
        ),
    };
    this.semantic = {
      nodes: {
        add: async (input) => {
          const nowMs = this.deps.clock.now();
          const embedding = await this.deps.embeddingClient.embed(
            `${input.label}\n${input.description}\n${input.aliases?.join(" ") ?? ""}`,
          );

          return this.deps.semanticNodeRepository.insert({
            id: createSemanticNodeId(),
            kind: input.kind,
            label: input.label,
            description: input.description,
            aliases: input.aliases ?? [],
            confidence: input.confidence ?? 0.6,
            source_episode_ids: input.sourceEpisodeIds,
            created_at: nowMs,
            updated_at: nowMs,
            last_verified_at: nowMs,
            embedding,
            archived: false,
            superseded_by: null,
          });
        },
        get: (id) => this.deps.semanticNodeRepository.get(id),
        list: (...args) => this.deps.semanticNodeRepository.list(...args),
        search: async (query, options = {}) => {
          const vector = await this.deps.embeddingClient.embed(query);
          return this.deps.semanticNodeRepository.searchByVector(vector, {
            limit: options.limit,
          });
        },
      },
      edges: {
        add: (input) => this.deps.semanticEdgeRepository.addEdge(input),
        list: (...args) => this.deps.semanticEdgeRepository.listEdges(...args),
      },
      walk: (fromId, ...args) => this.deps.semanticGraph.walk(fromId, ...args),
      extract: async (episodes) => {
        const extractor = new SemanticExtractor({
          nodeRepository: this.deps.semanticNodeRepository,
          edgeRepository: this.deps.semanticEdgeRepository,
          embeddingClient: this.deps.embeddingClient,
          llmClient: this.deps.llmFactory(),
          model: this.deps.config.anthropic.models.extraction,
          clock: this.deps.clock,
        });

        return extractor.extractFromEpisodes(episodes);
      },
    };
    this.commitments = {
      add: (input) =>
        this.deps.commitmentRepository.add({
          type: input.type,
          directive: input.directive,
          priority: input.priority,
          madeToEntity:
            input.madeTo === undefined || input.madeTo === null
              ? null
              : this.deps.entityRepository.resolve(input.madeTo),
          restrictedAudience:
            input.audience === undefined || input.audience === null
              ? null
              : this.deps.entityRepository.resolve(input.audience),
          aboutEntity:
            input.about === undefined || input.about === null
              ? null
              : this.deps.entityRepository.resolve(input.about),
          provenance: input.provenance,
          expiresAt: input.expiresAt ?? null,
        }),
      revoke: (...args) => this.deps.commitmentRepository.revoke(...args),
      list: (options = {}) =>
        this.deps.commitmentRepository.list({
          activeOnly: options.activeOnly,
          audience:
            options.audience === undefined || options.audience === null
              ? null
              : this.deps.entityRepository.resolve(options.audience),
          aboutEntity:
            options.aboutEntity === undefined || options.aboutEntity === null
              ? null
              : this.deps.entityRepository.resolve(options.aboutEntity),
        }),
    };
    this.identity = {
      updateValue: (...args) => this.deps.identityService.updateValue(...args),
      updateTrait: (...args) => this.deps.identityService.updateTrait(...args),
      updateCommitment: (...args) => this.deps.identityService.updateCommitment(...args),
      listEvents: (...args) => this.deps.identityService.listEvents(...args),
    };
    this.correction = {
      forget: (...args) => this.deps.correctionService.forget(...args),
      why: (...args) => this.deps.correctionService.why(...args),
      correct: (...args) => this.deps.correctionService.correct(...args),
      rememberAboutMe: (...args) => this.deps.correctionService.rememberAboutMe(...args),
      listIdentityEvents: (...args) => this.deps.correctionService.listIdentityEvents(...args),
    };
    this.review = {
      list: (options = {}) => this.deps.reviewQueueRepository.list(options),
      resolve: (id, decision) => this.deps.reviewQueueRepository.resolve(id, decision),
    };
    this.audit = {
      list: (options = {}) =>
        this.deps.auditLog.list({
          run_id: options.runId,
          process: options.process,
          reverted: options.reverted,
        }),
      revert: (id, revertedBy) => this.deps.auditLog.revert(id, revertedBy),
    };
    const runDream = async (
      processNames: readonly OfflineProcessName[],
      options: BorgDreamOptions = {},
    ): Promise<OrchestratorResult> => {
      const processes = processNames.map((name) => this.deps.offlineProcesses[name]);

      return this.deps.maintenanceOrchestrator.run({
        processes,
        opts: {
          dryRun: options.dryRun,
          budget: options.budget,
          processOverrides: options.processOverrides,
        },
      });
    };
    const planDream = (
      processNames: readonly OfflineProcessName[],
      options: BorgDreamOptions = {},
    ) =>
      this.deps.maintenanceOrchestrator.plan({
        processes: processNames.map((name) => this.deps.offlineProcesses[name]),
        opts: {
          budget: options.budget,
          processOverrides: options.processOverrides,
        },
      });
    this.dream = Object.assign(
      async (options: BorgDreamOptions = {}) =>
        runDream(options.processes ?? defaultDreamProcesses(), options),
      {
        plan: (options: Omit<BorgDreamOptions, "dryRun"> = {}) =>
          planDream(options.processes ?? defaultDreamProcesses(), options),
        preview: (plan: MaintenancePlan) => this.deps.maintenanceOrchestrator.preview(plan),
        apply: (plan: MaintenancePlan) => this.deps.maintenanceOrchestrator.apply(plan),
        consolidate: (options = {}) => runDream(["consolidator"], options),
        reflect: (options = {}) => runDream(["reflector"], options),
        curate: (options = {}) => runDream(["curator"], options),
        oversee: (options = {}) => runDream(["overseer"], options),
        ruminate: (
          options: {
            dryRun?: boolean;
            budget?: number;
            maxQuestionsPerRun?: number;
          } = {},
        ) =>
          runDream(["ruminator"], {
            ...options,
            processOverrides: {
              ruminator: {
                dryRun: options.dryRun,
                budget: options.budget,
                params:
                  options.maxQuestionsPerRun === undefined
                    ? undefined
                    : {
                        maxQuestionsPerRun: options.maxQuestionsPerRun,
                      },
              },
            },
          }),
        narrate: (
          options: {
            dryRun?: boolean;
            budget?: number;
            label?: string;
          } = {},
        ) =>
          runDream(["self-narrator"], {
            ...options,
            processOverrides: {
              "self-narrator": {
                dryRun: options.dryRun,
                budget: options.budget,
                params:
                  options.label === undefined
                    ? undefined
                    : {
                        label: options.label,
                      },
              },
            },
          }),
      },
    ) satisfies BorgDreamRunner;
    this.workmem = {
      load: (sessionId = DEFAULT_SESSION_ID) => this.deps.workingMemoryStore.load(sessionId),
      clear: (sessionId = DEFAULT_SESSION_ID) => {
        this.deps.turnOrchestrator.clearWorkingMemory(sessionId);
      },
    };
  }

  turn(input: TurnInput): Promise<TurnResult> {
    return this.deps.turnOrchestrator.run(input);
  }

  static async open(options: BorgOpenOptions = {}): Promise<Borg> {
    const clock = options.clock ?? new SystemClock();
    let sqlite: SqliteDatabase | undefined;
    let lance: LanceDbStore | undefined;

    try {
      const rawConfig =
        options.config ?? loadConfig({ env: options.env, dataDir: options.dataDir });
      const config = configSchema.parse({
        ...DEFAULT_CONFIG,
        ...rawConfig,
        dataDir: options.dataDir ?? rawConfig.dataDir ?? DEFAULT_CONFIG.dataDir,
        defaultUser: rawConfig.defaultUser ?? DEFAULT_CONFIG.defaultUser,
        perception: {
          ...DEFAULT_CONFIG.perception,
          ...rawConfig.perception,
        },
        affective: {
          ...DEFAULT_CONFIG.affective,
          ...(rawConfig as Partial<Config>).affective,
        },
        embedding: {
          ...DEFAULT_CONFIG.embedding,
          ...rawConfig.embedding,
        },
        anthropic: {
          ...DEFAULT_CONFIG.anthropic,
          ...rawConfig.anthropic,
          models: {
            ...DEFAULT_CONFIG.anthropic.models,
            ...rawConfig.anthropic?.models,
          },
        },
        self: {
          ...DEFAULT_CONFIG.self,
          ...(rawConfig as Partial<Config>).self,
        },
        offline: {
          ...DEFAULT_CONFIG.offline,
          ...rawConfig.offline,
          consolidator: {
            ...DEFAULT_CONFIG.offline.consolidator,
            ...rawConfig.offline?.consolidator,
          },
          reflector: {
            ...DEFAULT_CONFIG.offline.reflector,
            ...rawConfig.offline?.reflector,
          },
          curator: {
            ...DEFAULT_CONFIG.offline.curator,
            ...rawConfig.offline?.curator,
          },
          overseer: {
            ...DEFAULT_CONFIG.offline.overseer,
            ...rawConfig.offline?.overseer,
          },
          ruminator: {
            ...DEFAULT_CONFIG.offline.ruminator,
            ...rawConfig.offline?.ruminator,
          },
          selfNarrator: {
            ...DEFAULT_CONFIG.offline.selfNarrator,
            ...rawConfig.offline?.selfNarrator,
          },
        },
      });
      sqlite = openDatabase(join(config.dataDir, "borg.db"), {
        migrations: createMigrations(),
      });
      lance = new LanceDbStore({
        uri: join(config.dataDir, "lancedb"),
      });
      const embeddingDimensions = options.embeddingDimensions ?? config.embedding.dims;
      const episodesTable = await lance.openTable({
        name: "episodes",
        schema: createEpisodesTableSchema(embeddingDimensions),
      });
      const semanticNodesTable = await lance.openTable({
        name: "semantic_nodes",
        schema: createSemanticNodesTableSchema(embeddingDimensions),
      });
      const skillsTable = await lance.openTable({
        name: "skills",
        schema: createSkillsTableSchema(embeddingDimensions),
      });
      const embeddingClient = options.embeddingClient ?? createEmbeddingClient(config);
      const episodicRepository = new EpisodicRepository({
        table: episodesTable,
        db: sqlite,
        clock,
      });
      const createDefaultStreamWriter = () =>
        new StreamWriter({
          dataDir: config.dataDir,
          sessionId: DEFAULT_SESSION_ID,
          clock,
        });
      let reviewQueueRepository: ReviewQueueRepository | undefined;
      let applyCorrectionReview: ((item: ReviewQueueItem) => Promise<void>) | undefined;
      const openQuestionsRepository = new OpenQuestionsRepository({
        db: sqlite,
        clock,
      });
      const moodRepository = new MoodRepository({
        db: sqlite,
        clock,
        defaultHalfLifeHours: config.affective.moodHalfLifeHours,
        incomingWeight: config.affective.incomingMoodWeight,
      });
      const enqueueReview = (input: Parameters<ReviewQueueRepository["enqueue"]>[0]) => {
        return reviewQueueRepository?.enqueue(input);
      };
      // llmFactory is declared later in this scope (line ~947) because other
      // construction depends on it, but the semantic repo only uses the LLM
      // lazily -- at duplicate-review time. We can defer creation by passing
      // options.llmClient if present, else undefined. If not present, the
      // later factory will fabricate one from config, but for near-dup
      // contradiction judging we only need the SAME LLM client later. Use
      // a closure that will resolve once llmFactory exists.
      let deferredLlm: LLMClient | undefined;
      const semanticNodeRepository = new SemanticNodeRepository({
        table: semanticNodesTable,
        db: sqlite,
        clock,
        enqueueReview,
        // Accessed lazily -- deferredLlm is populated below once llmFactory
        // is built. Until then the field remains undefined and the repo
        // simply skips duplicate-review on inserts (safe default).
        get llmClient(): LLMClient | undefined {
          return deferredLlm;
        },
        contradictionJudgeModel: config.anthropic.models.background,
      });
      reviewQueueRepository = new ReviewQueueRepository({
        db: sqlite,
        clock,
        semanticNodeRepository,
        applyCorrection: (item) => {
          if (applyCorrectionReview === undefined) {
            throw new Error("Correction service not initialized");
          }

          return applyCorrectionReview(item);
        },
        onEnqueue: (item) => enqueueOpenQuestionForReview(openQuestionsRepository, item),
        onEnqueueError: (error) => {
          const writer = createDefaultStreamWriter();
          void appendOpenQuestionHookFailureEvent(
            writer,
            "review_queue_open_question",
            error,
          ).finally(() => {
            writer.close();
          });
        },
      });
      const semanticEdgeRepository = new SemanticEdgeRepository({
        db: sqlite,
        clock,
        enqueueReview,
      });
      const semanticGraph = new SemanticGraph({
        nodeRepository: semanticNodeRepository,
        edgeRepository: semanticEdgeRepository,
      });
      const identityEventRepository = new IdentityEventRepository({
        db: sqlite,
        clock,
      });
      const valuesRepository = new ValuesRepository({
        db: sqlite,
        clock,
        identityEventRepository,
      });
      const goalsRepository = new GoalsRepository({
        db: sqlite,
        clock,
        identityEventRepository,
      });
      const traitsRepository = new TraitsRepository({
        db: sqlite,
        clock,
        identityEventRepository,
      });
      const autobiographicalRepository = new AutobiographicalRepository({
        db: sqlite,
        clock,
      });
      if (config.self.autoBootstrapPeriod && autobiographicalRepository.currentPeriod() === null) {
        const nowMs = clock.now();
        autobiographicalRepository.upsertPeriod({
          label: quarterLabel(nowMs),
          start_ts: nowMs,
          end_ts: null,
          narrative: "",
          key_episode_ids: [],
          themes: [],
          provenance: {
            kind: "system",
          },
        });
      }
      const growthMarkersRepository = new GrowthMarkersRepository({
        db: sqlite,
        clock,
      });
      const entityRepository = new EntityRepository({
        db: sqlite,
        clock,
      });
      const socialRepository = new SocialRepository({
        db: sqlite,
        clock,
      });
      const commitmentRepository = new CommitmentRepository({
        db: sqlite,
        clock,
        identityEventRepository,
      });
      const identityService = new IdentityService({
        valuesRepository,
        traitsRepository,
        commitmentRepository,
        identityEventRepository,
      });
      const skillRepository = new SkillRepository({
        table: skillsTable,
        db: sqlite,
        embeddingClient,
        clock,
      });
      const skillSelector = new SkillSelector({
        repository: skillRepository,
      });
      const retrievalPipeline = new RetrievalPipeline({
        embeddingClient,
        episodicRepository,
        semanticNodeRepository,
        semanticGraph,
        openQuestionsRepository,
        dataDir: config.dataDir,
        clock,
      });
      const correctionService = new CorrectionService({
        config,
        retrievalPipeline,
        episodicRepository,
        semanticNodeRepository,
        semanticEdgeRepository,
        semanticGraph,
        valuesRepository,
        goalsRepository,
        traitsRepository,
        openQuestionsRepository,
        socialRepository,
        entityRepository,
        commitmentRepository,
        reviewQueueRepository,
        identityService,
        identityEventRepository,
      });
      applyCorrectionReview = (item) => correctionService.applyCorrectionReview(item);
      const workingMemoryStore = new WorkingMemoryStore({
        dataDir: config.dataDir,
        clock,
      });
      const streamWatermarkRepository = new StreamWatermarkRepository({
        db: sqlite,
        clock,
      });
      const createStreamWriter = (sessionId: SessionId) =>
        new StreamWriter({
          dataDir: config.dataDir,
          sessionId,
          clock,
        });
      const llmFactory = createLlmFactory(config, options.llmClient, options.env, clock);
      const lazyLlmClient = createLazyLlmClient(llmFactory);
      // Resolve the deferred LLM client for semantic-repo duplicate review
      // now that llmFactory exists. Before this point the repo's getter
      // returns undefined and near-dup inserts simply skip review.
      deferredLlm = llmFactory();
      const reverserRegistry = new ReverserRegistry();
      const auditLog = new AuditLog({
        db: sqlite,
        clock,
        registry: reverserRegistry,
      });
      const offlineProcesses = {
        consolidator: new ConsolidatorProcess({
          episodicRepository,
          registry: reverserRegistry,
        }),
        reflector: new ReflectorProcess({
          semanticNodeRepository,
          semanticEdgeRepository,
          reviewQueueRepository,
          registry: reverserRegistry,
        }),
        curator: new CuratorProcess({
          episodicRepository,
          moodRepository,
          socialRepository,
          registry: reverserRegistry,
        }),
        overseer: new OverseerProcess({
          reviewQueueRepository,
          registry: reverserRegistry,
        }),
        ruminator: new RuminatorProcess({
          openQuestionsRepository,
          growthMarkersRepository,
          registry: reverserRegistry,
        }),
        "self-narrator": new SelfNarratorProcess({
          autobiographicalRepository,
          growthMarkersRepository,
          registry: reverserRegistry,
        }),
      } satisfies Record<OfflineProcessName, OfflineProcess>;
      const maintenanceOrchestrator = new MaintenanceOrchestrator({
        baseContext: {
          config,
          clock,
          embeddingClient,
          llm: {
            cognition: lazyLlmClient,
            background: lazyLlmClient,
            extraction: lazyLlmClient,
          },
          episodicRepository,
          semanticNodeRepository,
          semanticEdgeRepository,
          reviewQueueRepository,
          valuesRepository,
          goalsRepository,
          traitsRepository,
          autobiographicalRepository,
          growthMarkersRepository,
          openQuestionsRepository,
          moodRepository,
          socialRepository,
          entityRepository,
          commitmentRepository,
          skillRepository,
          retrievalPipeline,
        },
        auditLog,
        createStreamWriter: () => createStreamWriter(DEFAULT_SESSION_ID),
        processRegistry: offlineProcesses,
      });
      const liveExtractionEnabled = options.liveExtraction ?? false;
      // Live extractor shares the same embedding + LLM wiring as the offline
      // consolidator process. It runs after each turn inside the ingestion
      // coordinator so the next turn's retrieval sees this turn's material.
      const streamIngestionCoordinator = liveExtractionEnabled
        ? new StreamIngestionCoordinator({
            extractor: new EpisodicExtractor({
              dataDir: config.dataDir,
              episodicRepository,
              embeddingClient,
              llmClient: lazyLlmClient,
              model: config.anthropic.models.extraction,
              entityRepository,
              clock,
            }),
            watermarkRepository: streamWatermarkRepository,
            dataDir: config.dataDir,
            clock,
            onError: (error) => {
              // Use a fresh writer: the turn's writer closes before
              // ingestion resolves, and we must not hold onto stream handles
              // across fire-and-forget boundaries.
              const writer = createStreamWriter(DEFAULT_SESSION_ID);
              void writer
                .append({
                  kind: "internal_event",
                  content: `Live episodic extraction failed: ${
                    error instanceof Error ? error.message : String(error)
                  }`,
                })
                .catch(() => undefined)
                .finally(() => {
                  writer.close();
                });
            },
          })
        : undefined;
      const turnOrchestrator = new TurnOrchestrator({
        config,
        retrievalPipeline,
        episodicRepository,
        entityRepository,
        commitmentRepository,
        reviewQueueRepository,
        valuesRepository,
        goalsRepository,
        traitsRepository,
        autobiographicalRepository,
        growthMarkersRepository,
        openQuestionsRepository,
        moodRepository,
        socialRepository,
        skillRepository,
        skillSelector,
        workingMemoryStore,
        llmFactory,
        clock,
        createStreamWriter,
        // Explicit so borg.ts wires a single compiler instance per process;
        // turn-orchestrator.ts falls back to defaults if omitted, but doing
        // it here makes the configuration visible at the composition root.
        turnContextCompiler: new TurnContextCompiler(),
        ...(streamIngestionCoordinator === undefined
          ? {}
          : { streamIngestionCoordinator }),
      });

      return new Borg({
        config,
        sqlite,
        lance,
        episodicRepository,
        semanticNodeRepository,
        semanticEdgeRepository,
        semanticGraph,
        reviewQueueRepository,
        identityEventRepository,
        identityService,
        valuesRepository,
        goalsRepository,
        traitsRepository,
        autobiographicalRepository,
        growthMarkersRepository,
        openQuestionsRepository,
        moodRepository,
        socialRepository,
        entityRepository,
        commitmentRepository,
        correctionService,
        skillRepository,
        skillSelector,
        retrievalPipeline,
        workingMemoryStore,
        turnOrchestrator,
        auditLog,
        maintenanceOrchestrator,
        offlineProcesses,
        llmFactory,
        embeddingClient,
        clock,
      });
    } catch (error) {
      await closeBestEffort(sqlite, lance);
      throw error;
    }
  }

  async close(): Promise<void> {
    this.deps.sqlite.close();
    await this.deps.lance.close();
  }
}
