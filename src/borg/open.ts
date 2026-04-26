// Borg.open composition root: orders storage, repositories, tools, offline work, turns, and autonomy.

import { SystemClock } from "../util/clock.js";
import { SessionLock } from "../cognition/index.js";
import { createTurnTracer } from "../cognition/tracing/tracer.js";
import type { LLMClient } from "../llm/index.js";
import type { LanceDbStore } from "../storage/lancedb/index.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { StreamWatermarkRepository } from "../stream/index.js";
import { buildAutonomyScheduler } from "./autonomy-setup.js";
import { buildMaintenanceScheduler } from "./maintenance-setup.js";
import { createEmbeddingClient, createLazyLlmClient, createLlmFactory } from "./clients.js";
import { buildStreamIngestionCoordinator } from "./ingestion-setup.js";
import { closeBestEffort } from "./lifecycle.js";
import { buildOfflineSetup } from "./offline-setup.js";
import { buildBorgRepositories } from "./repositories.js";
import {
  openBorgLanceTables,
  openBorgStorage,
  resolveBorgConfig,
  type BorgLanceTables,
} from "./storage-setup.js";
import { buildToolDispatcher } from "./tools-setup.js";
import { buildTurnOrchestrator } from "./turn-setup.js";
import type { BorgDependencies, BorgOpenOptions } from "./types.js";

export async function openBorgDependencies(
  options: BorgOpenOptions = {},
): Promise<BorgDependencies> {
  const clock = options.clock ?? new SystemClock();
  let sqlite: SqliteDatabase | undefined;
  let lance: LanceDbStore | undefined;

  try {
    const config = resolveBorgConfig(options);
    const tracer = createTurnTracer({
      tracerPath: options.tracerPath,
      env: options.env ?? process.env,
      clock,
    });
    const storage = openBorgStorage(config);
    sqlite = storage.sqlite;
    lance = storage.lance;
    const tables: BorgLanceTables = await openBorgLanceTables({
      lance,
      embeddingDimensions: options.embeddingDimensions ?? config.embedding.dims,
    });
    const embeddingClient = options.embeddingClient ?? createEmbeddingClient(config);
    let deferredLlm: LLMClient | undefined;
    const repositories = await buildBorgRepositories({
      config,
      sqlite,
      episodesTable: tables.episodesTable,
      semanticNodesTable: tables.semanticNodesTable,
      skillsTable: tables.skillsTable,
      embeddingClient,
      clock,
      tracer,
      getDeferredLlm: () => deferredLlm,
    });
    const sessionLock = new SessionLock({
      dataDir: config.dataDir,
    });
    const streamWatermarkRepository = new StreamWatermarkRepository({
      db: sqlite,
      clock,
    });
    const toolDispatcher = buildToolDispatcher({
      retrievalPipeline: repositories.retrievalPipeline,
      semanticGraph: repositories.semanticGraph,
      commitmentRepository: repositories.commitmentRepository,
      openQuestionsRepository: repositories.openQuestionsRepository,
      identityService: repositories.identityService,
      skillRepository: repositories.skillRepository,
      createStreamWriter: repositories.createStreamWriter,
      clock,
    });
    const llmFactory = createLlmFactory(config, options.llmClient, options.env, clock);
    const lazyLlmClient = createLazyLlmClient(llmFactory);
    // Resolve the deferred LLM client for semantic-repo duplicate review
    // now that llmFactory exists. Before this point the repo's getter
    // returns undefined and near-dup inserts simply skip review.
    deferredLlm = llmFactory();
    const offline = buildOfflineSetup({
      config,
      sqlite,
      clock,
      embeddingClient,
      lazyLlmClient,
      episodicRepository: repositories.episodicRepository,
      semanticNodeRepository: repositories.semanticNodeRepository,
      semanticEdgeRepository: repositories.semanticEdgeRepository,
      reviewQueueRepository: repositories.reviewQueueRepository,
      identityService: repositories.identityService,
      valuesRepository: repositories.valuesRepository,
      goalsRepository: repositories.goalsRepository,
      traitsRepository: repositories.traitsRepository,
      autobiographicalRepository: repositories.autobiographicalRepository,
      growthMarkersRepository: repositories.growthMarkersRepository,
      openQuestionsRepository: repositories.openQuestionsRepository,
      moodRepository: repositories.moodRepository,
      socialRepository: repositories.socialRepository,
      entityRepository: repositories.entityRepository,
      commitmentRepository: repositories.commitmentRepository,
      skillRepository: repositories.skillRepository,
      retrievalPipeline: repositories.retrievalPipeline,
      createStreamWriter: repositories.createStreamWriter,
    });
    const streamIngestionCoordinator = buildStreamIngestionCoordinator({
      enabled: options.liveExtraction ?? true,
      config,
      episodicRepository: repositories.episodicRepository,
      embeddingClient,
      lazyLlmClient,
      entityRepository: repositories.entityRepository,
      streamWatermarkRepository,
      createStreamWriter: repositories.createStreamWriter,
      clock,
    });
    const turnOrchestrator = buildTurnOrchestrator({
      config,
      retrievalPipeline: repositories.retrievalPipeline,
      episodicRepository: repositories.episodicRepository,
      entityRepository: repositories.entityRepository,
      commitmentRepository: repositories.commitmentRepository,
      reviewQueueRepository: repositories.reviewQueueRepository,
      identityService: repositories.identityService,
      valuesRepository: repositories.valuesRepository,
      goalsRepository: repositories.goalsRepository,
      traitsRepository: repositories.traitsRepository,
      autobiographicalRepository: repositories.autobiographicalRepository,
      growthMarkersRepository: repositories.growthMarkersRepository,
      openQuestionsRepository: repositories.openQuestionsRepository,
      moodRepository: repositories.moodRepository,
      socialRepository: repositories.socialRepository,
      skillRepository: repositories.skillRepository,
      skillSelector: repositories.skillSelector,
      workingMemoryStore: repositories.workingMemoryStore,
      llmFactory,
      toolDispatcher,
      sessionLock,
      streamIngestionCoordinator,
      createStreamWriter: repositories.createStreamWriter,
      clock,
      tracer,
    });
    const autonomyScheduler = buildAutonomyScheduler({
      config,
      commitmentRepository: repositories.commitmentRepository,
      goalsRepository: repositories.goalsRepository,
      openQuestionsRepository: repositories.openQuestionsRepository,
      moodRepository: repositories.moodRepository,
      streamWatermarkRepository,
      autonomyWakesRepository: repositories.autonomyWakesRepository,
      turnOrchestrator,
      toolDispatcher,
      createStreamWriter: repositories.createStreamWriter,
      clock,
    });
    const maintenanceScheduler = buildMaintenanceScheduler({
      config,
      orchestrator: offline.maintenanceOrchestrator,
      processRegistry: offline.offlineProcesses,
      clock,
      isBusy: () => sessionLock.isHeld(),
    });

    return {
      config,
      sqlite,
      lance,
      entryIndex: repositories.entryIndex,
      episodicRepository: repositories.episodicRepository,
      semanticNodeRepository: repositories.semanticNodeRepository,
      semanticEdgeRepository: repositories.semanticEdgeRepository,
      semanticGraph: repositories.semanticGraph,
      reviewQueueRepository: repositories.reviewQueueRepository,
      identityEventRepository: repositories.identityEventRepository,
      identityService: repositories.identityService,
      valuesRepository: repositories.valuesRepository,
      goalsRepository: repositories.goalsRepository,
      traitsRepository: repositories.traitsRepository,
      autobiographicalRepository: repositories.autobiographicalRepository,
      growthMarkersRepository: repositories.growthMarkersRepository,
      openQuestionsRepository: repositories.openQuestionsRepository,
      moodRepository: repositories.moodRepository,
      socialRepository: repositories.socialRepository,
      entityRepository: repositories.entityRepository,
      commitmentRepository: repositories.commitmentRepository,
      correctionService: repositories.correctionService,
      skillRepository: repositories.skillRepository,
      skillSelector: repositories.skillSelector,
      retrievalPipeline: repositories.retrievalPipeline,
      workingMemoryStore: repositories.workingMemoryStore,
      autonomyWakesRepository: repositories.autonomyWakesRepository,
      turnOrchestrator,
      autonomyScheduler,
      maintenanceScheduler,
      streamIngestionCoordinator,
      auditLog: offline.auditLog,
      maintenanceOrchestrator: offline.maintenanceOrchestrator,
      offlineProcesses: offline.offlineProcesses,
      llmFactory,
      embeddingClient,
      clock,
    };
  } catch (error) {
    await closeBestEffort(sqlite, lance);
    throw error;
  }
}
