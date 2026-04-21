import { join } from "node:path";

import { TurnOrchestrator, type TurnInput, type TurnResult } from "./cognition/index.js";
import { loadConfig, type Config } from "./config/index.js";
import { OpenAICompatibleEmbeddingClient, type EmbeddingClient } from "./embeddings/index.js";
import { AnthropicLLMClient, type LLMClient } from "./llm/index.js";
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
import type { RetrievedEpisode, RetrievalSearchOptions } from "./retrieval/index.js";
import {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
  selfMigrations,
} from "./memory/self/index.js";
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
  StreamWriter,
  type StreamEntry,
  type StreamEntryInput,
} from "./stream/index.js";
import { LanceDbStore } from "./storage/lancedb/index.js";
import { openDatabase, type Migration, type SqliteDatabase } from "./storage/sqlite/index.js";
import { SystemClock, type Clock } from "./util/clock.js";
import {
  DEFAULT_SESSION_ID,
  type AuditId,
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
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  autobiographicalRepository: AutobiographicalRepository;
  growthMarkersRepository: GrowthMarkersRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  entityRepository: EntityRepository;
  commitmentRepository: CommitmentRepository;
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

function createEmbeddingClient(config: Config): EmbeddingClient {
  return new OpenAICompatibleEmbeddingClient({
    baseUrl: config.embedding.baseUrl,
    apiKey: config.embedding.apiKey,
    model: config.embedding.model,
    dims: config.embedding.dims,
  });
}

function createLlmFactory(config: Config, llmClient: LLMClient | undefined): () => LLMClient {
  if (llmClient !== undefined) {
    return () => llmClient;
  }

  let cached: LLMClient | undefined;

  return () => {
    cached ??= new AnthropicLLMClient({
      apiKey: config.anthropic.apiKey,
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
    ...retrievalMigrations,
    ...semanticMigrations,
    ...commitmentMigrations,
    ...offlineMigrations,
  ];
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
    get: (id: EpisodeId) => Promise<RetrievedEpisode | null>;
    search: (query: string, options?: RetrievalSearchOptions) => Promise<RetrievedEpisode[]>;
    extract: (options?: {
      sinceTs?: number;
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
      sourceEpisodeIds?: EpisodeId[];
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
      get: (id) => this.deps.retrievalPipeline.getEpisode(id),
      search: (query, options) => this.deps.retrievalPipeline.search(query, options),
      extract: async (options = {}) => {
        const extractor = new EpisodicExtractor({
          dataDir: this.deps.config.dataDir,
          episodicRepository: this.deps.episodicRepository,
          embeddingClient: this.deps.embeddingClient,
          llmClient: this.deps.llmFactory(),
          model: this.deps.config.anthropic.models.extraction,
          clock: this.deps.clock,
        });

        return extractor.extractFromStream({
          session: options.session ?? DEFAULT_SESSION_ID,
          sinceTs: options.sinceTs,
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
          sourceEpisodeIds: input.sourceEpisodeIds,
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
      const config = options.config ?? loadConfig({ env: options.env, dataDir: options.dataDir });
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
      const openQuestionsRepository = new OpenQuestionsRepository({
        db: sqlite,
        clock,
      });
      const enqueueReview = (input: Parameters<ReviewQueueRepository["enqueue"]>[0]) => {
        return reviewQueueRepository?.enqueue(input);
      };
      const semanticNodeRepository = new SemanticNodeRepository({
        table: semanticNodesTable,
        db: sqlite,
        clock,
        enqueueReview,
      });
      reviewQueueRepository = new ReviewQueueRepository({
        db: sqlite,
        clock,
        semanticNodeRepository,
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
      const valuesRepository = new ValuesRepository({
        db: sqlite,
        clock,
      });
      const goalsRepository = new GoalsRepository({
        db: sqlite,
        clock,
      });
      const traitsRepository = new TraitsRepository({
        db: sqlite,
        clock,
      });
      const autobiographicalRepository = new AutobiographicalRepository({
        db: sqlite,
        clock,
      });
      const growthMarkersRepository = new GrowthMarkersRepository({
        db: sqlite,
        clock,
      });
      const entityRepository = new EntityRepository({
        db: sqlite,
        clock,
      });
      const commitmentRepository = new CommitmentRepository({
        db: sqlite,
        clock,
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
      const workingMemoryStore = new WorkingMemoryStore({
        dataDir: config.dataDir,
        clock,
      });
      const createStreamWriter = (sessionId: SessionId) =>
        new StreamWriter({
          dataDir: config.dataDir,
          sessionId,
          clock,
        });
      const llmFactory = createLlmFactory(config, options.llmClient);
      const lazyLlmClient = createLazyLlmClient(llmFactory);
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
          entityRepository,
          commitmentRepository,
          retrievalPipeline,
        },
        auditLog,
        createStreamWriter: () => createStreamWriter(DEFAULT_SESSION_ID),
        processRegistry: offlineProcesses,
      });
      const turnOrchestrator = new TurnOrchestrator({
        config,
        retrievalPipeline,
        episodicRepository,
        entityRepository,
        commitmentRepository,
        valuesRepository,
        goalsRepository,
        traitsRepository,
        openQuestionsRepository,
        workingMemoryStore,
        llmFactory,
        clock,
        createStreamWriter,
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
        valuesRepository,
        goalsRepository,
        traitsRepository,
        autobiographicalRepository,
        growthMarkersRepository,
        openQuestionsRepository,
        entityRepository,
        commitmentRepository,
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
