// Creates the optional live stream ingestion coordinator used after turns.

import type { Config } from "../config/index.js";
import {
  StreamIngestionCoordinator,
  type ChatResponseWatermarkCoordinator,
} from "../cognition/ingestion/index.js";
import { CorrectivePreferenceTurnService } from "../cognition/commitments/corrective-preference-service.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { LLMClient } from "../llm/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import { EpisodicExtractor, type EpisodicRepository } from "../memory/episodic/index.js";
import type { IdentityEventRepository, IdentityService } from "../memory/identity/index.js";
import type { RelationalSlotRepository } from "../memory/relational-slots/index.js";
import { appendInternalFailureEvent } from "../memory/self/index.js";
import type { WorkingMemoryStore } from "../memory/working/index.js";
import type { StreamEntryIndexRepository, StreamWatermarkRepository } from "../stream/index.js";
import type { Clock } from "../util/clock.js";
import type { BorgStreamWriterFactory } from "./types.js";
import { CorrectivePreferenceIngestion } from "./corrective-preference-ingestion.js";

export type BuildIngestionCoordinatorOptions = {
  enabled: boolean;
  config: Config;
  episodicRepository: EpisodicRepository;
  embeddingClient: EmbeddingClient;
  lazyLlmClient: LLMClient;
  entityRepository: EntityRepository;
  commitmentRepository: CommitmentRepository;
  identityService: IdentityService;
  identityEventRepository: Pick<IdentityEventRepository, "runInTransaction">;
  relationalSlotRepository: RelationalSlotRepository;
  workingMemoryStore: WorkingMemoryStore;
  entryIndex: StreamEntryIndexRepository;
  streamWatermarkRepository: StreamWatermarkRepository;
  chatResponseWatermarkCoordinator?: ChatResponseWatermarkCoordinator;
  createStreamWriter: BorgStreamWriterFactory;
  correctivePreferenceExtraction?: {
    budget: number | null;
  };
  tracer: TurnTracer;
  clock: Clock;
};

export function buildStreamIngestionCoordinator(
  options: BuildIngestionCoordinatorOptions,
): StreamIngestionCoordinator | undefined {
  if (!options.enabled) {
    return undefined;
  }

  const appendHookFailure = async (
    sessionId: Parameters<BorgStreamWriterFactory>[0],
    hook: string,
    error: unknown,
    details?: Record<string, unknown>,
  ): Promise<void> => {
    const writer = options.createStreamWriter(sessionId);

    try {
      await appendInternalFailureEvent(writer, hook, error, details);
    } catch {
      // Best-effort observability on the detached ingestion path.
    } finally {
      writer.close();
    }
  };
  const entryProcessor =
    options.correctivePreferenceExtraction === undefined
      ? undefined
      : new CorrectivePreferenceIngestion({
          service: new CorrectivePreferenceTurnService({
            model: options.config.anthropic.models.recallExpansion,
            commitmentRepository: options.commitmentRepository,
            identityService: options.identityService,
            relationalSlotRepository: options.relationalSlotRepository,
            workingMemoryStore: options.workingMemoryStore,
            clock: options.clock,
            tracer: options.tracer,
            strictFailurePropagation: true,
            runPersistenceTransaction: (callback) =>
              options.identityEventRepository.runInTransaction(callback),
          }),
          llmClient: options.lazyLlmClient,
          dataDir: options.config.dataDir,
          entryIndex: options.entryIndex,
          entityRepository: options.entityRepository,
          relationalSlotRepository: options.relationalSlotRepository,
          tracer: options.tracer,
          budget: options.correctivePreferenceExtraction.budget,
          clock: options.clock,
          onHookFailure: appendHookFailure,
        });

  // HTTP operator writes, maintenance writes, and this processor all share the
  // same synchronous SQLite repository connection. SQLite transaction
  // serialization plus repository CAS checks order conflicting mutations;
  // strict sidecar propagation turns a CAS/write conflict into receipt-backed
  // retry instead of allowing a partially applied supersession.

  // Lifecycle fidelity note: unlike the full cognition extraction phase, the
  // sidecar has no actionable frame-anomaly detector, so it cannot skip this
  // processor on that signal. It intentionally classifies every active,
  // sender-bound user entry, including entries the episodic salience gate skips.
  // The episodic extractor must also retain whole-window grouping (notably for
  // OUTCOME episodes), so it runs before per-entry commitments; classifiers see
  // the relational-slot snapshot after that window. This documented ordering
  // deviation is covered by sidecar ingestion tests.
  //
  // Live extraction shares the same embedding + LLM wiring as the offline
  // consolidator process. Retryable commitment failures leave the shared
  // watermark in place; durable per-entry receipts prevent successful entries
  // from being replayed while the poison entry approaches its retry ceiling.
  return new StreamIngestionCoordinator({
    extractor: new EpisodicExtractor({
      dataDir: options.config.dataDir,
      episodicRepository: options.episodicRepository,
      embeddingClient: options.embeddingClient,
      llmClient: options.lazyLlmClient,
      model: options.config.anthropic.models.extraction,
      entityRepository: options.entityRepository,
      relationalSlotRepository: options.relationalSlotRepository,
      workingMemoryStore: options.workingMemoryStore,
      defaultUser: options.config.defaultUser,
      salienceGateEnabled: options.config.episodic.salienceGateEnabled,
      tracer: options.tracer,
      clock: options.clock,
    }),
    ...(entryProcessor === undefined ? {} : { entryProcessor }),
    watermarkRepository: options.streamWatermarkRepository,
    chatResponseWatermarkCoordinator: options.chatResponseWatermarkCoordinator,
    dataDir: options.config.dataDir,
    maxEntries: options.config.streamIngestion.preTurnCatchup.maxEntries,
    settleMs: options.config.streamIngestion.settle.settleMs,
    maxSettleMs: options.config.streamIngestion.settle.maxSettleMs,
    clock: options.clock,
    onError: (error, sessionId) => {
      // Use a fresh writer: the turn's writer closes before ingestion
      // resolves, and we must not hold onto stream handles across
      // fire-and-forget boundaries.
      const writer = options.createStreamWriter(sessionId);
      void writer
        .append({
          kind: "internal_event",
          content: `Live stream ingestion failed: ${
            error instanceof Error ? error.message : String(error)
          }`,
        })
        .catch(() => undefined)
        .finally(() => {
          writer.close();
        });
    },
  });
}
