// Borg -- public library entry point.
// See ARCHITECTURE.md for the design. Public API surface will grow
// sprint by sprint; this file re-exports stable entry points as they land.

export const VERSION = "0.1.0";

export {
  DEFAULT_CONFIG,
  configSchema,
  expandPath,
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
  type LLMCompleteOptions,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolCall,
  type LLMToolDefinition,
  type TokenUsageEvent,
  type TokenUsageSink,
} from "./llm/index.js";
export {
  DEFAULT_SESSION_ID,
  STREAM_ENTRY_KINDS,
  StreamReader,
  StreamWriter,
  getSessionStreamPath,
  getStreamDirectory,
  streamEntryInputSchema,
  streamEntryKindSchema,
  streamEntrySchema,
  type SessionId,
  type StreamEntry,
  type StreamEntryInput,
  type StreamEntryKind,
  type StreamIterateOptions,
} from "./stream/index.js";
export {
  LanceDbStore,
  LanceDbTable,
  booleanField,
  float64Field,
  int32Field,
  int64Field,
  schema,
  timestampMsField,
  utf8Field,
  vectorField,
  type LanceDbListOptions,
  type LanceDbOpenTableOptions,
  type LanceDbRow,
  type LanceDbSearchOptions,
  type LanceDbStoreOptions,
  type LanceDbUpsertOptions,
} from "./storage/lancedb/index.js";
export {
  openDatabase,
  SqliteDatabase,
  type AppliedMigration,
  type Migration,
  type OpenDatabaseOptions,
} from "./storage/sqlite/index.js";
export { FixedClock, ManualClock, SystemClock, type Clock } from "./util/clock.js";
export {
  ConfigError,
  BorgError,
  EmbeddingError,
  LLMError,
  StorageError,
  StreamError,
  type BorgErrorJSON,
  type BorgErrorOptions,
} from "./util/errors.js";
export {
  DEFAULT_SESSION_ID as DEFAULT_LIBRARY_SESSION_ID,
  createSessionId,
  createStreamEntryId,
  parseSessionId,
  sessionIdHelpers,
  streamEntryIdHelpers,
  type BrandedId,
  type SessionId as IdSessionId,
  type StreamEntryId,
} from "./util/ids.js";
