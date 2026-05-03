// Creates the optional live stream ingestion coordinator used after turns.

import type { Config } from "../config/index.js";
import { StreamIngestionCoordinator } from "../cognition/ingestion/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { LLMClient } from "../llm/index.js";
import type { EntityRepository } from "../memory/commitments/index.js";
import { EpisodicExtractor, type EpisodicRepository } from "../memory/episodic/index.js";
import type { RelationalSlotRepository } from "../memory/relational-slots/index.js";
import type { WorkingMemoryStore } from "../memory/working/index.js";
import type { StreamWatermarkRepository } from "../stream/index.js";
import type { Clock } from "../util/clock.js";
import type { BorgStreamWriterFactory } from "./types.js";

export type BuildIngestionCoordinatorOptions = {
  enabled: boolean;
  config: Config;
  episodicRepository: EpisodicRepository;
  embeddingClient: EmbeddingClient;
  lazyLlmClient: LLMClient;
  entityRepository: EntityRepository;
  relationalSlotRepository: RelationalSlotRepository;
  workingMemoryStore: WorkingMemoryStore;
  streamWatermarkRepository: StreamWatermarkRepository;
  createStreamWriter: BorgStreamWriterFactory;
  clock: Clock;
};

export function buildStreamIngestionCoordinator(
  options: BuildIngestionCoordinatorOptions,
): StreamIngestionCoordinator | undefined {
  if (!options.enabled) {
    return undefined;
  }

  // Live extraction shares the same embedding + LLM wiring as the offline
  // consolidator process. It runs after each turn on a best-effort path;
  // pre-turn catch-up retries missed stream backlog before retrieval.
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
      clock: options.clock,
    }),
    watermarkRepository: options.streamWatermarkRepository,
    dataDir: options.config.dataDir,
    clock: options.clock,
    onError: (error, sessionId) => {
      // Use a fresh writer: the turn's writer closes before ingestion
      // resolves, and we must not hold onto stream handles across
      // fire-and-forget boundaries.
      const writer = options.createStreamWriter(sessionId);
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
  });
}
