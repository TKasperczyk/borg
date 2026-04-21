import { join } from "node:path";

import { loadConfig, type Config } from "./config/index.js";
import { OpenAICompatibleEmbeddingClient, type EmbeddingClient } from "./embeddings/index.js";
import { AnthropicLLMClient, type LLMClient } from "./llm/index.js";
import {
  EpisodicExtractor,
  EpisodicRepository,
  type ExtractFromStreamResult,
  createEpisodesTableSchema,
  episodicMigrations,
} from "./memory/episodic/index.js";
import type { RetrievedEpisode, RetrievalSearchOptions } from "./retrieval/index.js";
import {
  GoalsRepository,
  TraitsRepository,
  ValuesRepository,
  selfMigrations,
} from "./memory/self/index.js";
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
import { DEFAULT_SESSION_ID, type EpisodeId, type SessionId } from "./util/ids.js";

type BorgDependencies = {
  config: Config;
  sqlite: SqliteDatabase;
  lance: LanceDbStore;
  episodicRepository: EpisodicRepository;
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  retrievalPipeline: RetrievalPipeline;
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

function createMigrations(): Migration[] {
  return [...episodicMigrations, ...selfMigrations, ...retrievalMigrations];
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
  };

  private constructor(private readonly deps: BorgDependencies) {
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
    };
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
      const embeddingClient = options.embeddingClient ?? createEmbeddingClient(config);
      const episodicRepository = new EpisodicRepository({
        table: episodesTable,
        db: sqlite,
        clock,
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
      const retrievalPipeline = new RetrievalPipeline({
        embeddingClient,
        episodicRepository,
        dataDir: config.dataDir,
        clock,
      });

      return new Borg({
        config,
        sqlite,
        lance,
        episodicRepository,
        valuesRepository,
        goalsRepository,
        traitsRepository,
        retrievalPipeline,
        llmFactory: createLlmFactory(config, options.llmClient),
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
