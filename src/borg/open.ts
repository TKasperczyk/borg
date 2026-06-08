// Borg.open composition root: orders storage, repositories, tools, offline work, turns, and autonomy.

import { SystemClock } from "../util/clock.js";
import {
  AttachmentBlobStore,
  AttachmentRepository,
  AttachmentService,
  ImageAttachmentLifecycleService,
  ImagePerceptionService,
} from "../attachments/index.js";
import { SessionLock } from "../cognition/index.js";
import {
  ChatResponseBacklogPrefixBuilder,
  ChatResponseCatchUpWorker,
  ChatResponseWatermarkCoordinator,
  MessageEnqueuer,
  type ChatResponseCatchUpWorkerConfig,
} from "../cognition/ingestion/index.js";
import { compositeTracer, createTurnTracer } from "../cognition/tracing/tracer.js";
import type { LanceDbStore } from "../storage/lancedb/index.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import {
  StreamEntryIndexRepository,
  StreamReader,
  StreamWatermarkRepository,
} from "../stream/index.js";
import { createOutboundPostTool } from "../tools/index.js";
import {
  AutonomousOutboundPolicy,
  MessageConnectorRegistry,
  OutboundDelivery,
  runDirectedOutboundTurn,
} from "../outbound/index.js";
import { withDerivedOutboundCapabilities } from "../cognition/prompts/host-capabilities.js";
import { buildAutonomyScheduler } from "./autonomy-setup.js";
import { buildMaintenanceScheduler } from "./maintenance-setup.js";
import { createEmbeddingClient, createLazyLlmClient, createLlmFactory } from "./clients.js";
import { buildStreamIngestionCoordinator } from "./ingestion-setup.js";
import { closeBestEffort } from "./lifecycle.js";
import { buildOfflineSetup } from "./offline-setup.js";
import { backfillSessionStreamEntryIndexAndAttachments } from "./reconciliation.js";
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
import type { SessionId } from "../util/ids.js";

const CHAT_RESPONSE_CATCH_UP_BACKOFF_CONFIG = {
  backoffBaseMs: 1_000,
  maxBackoffMs: 60_000,
} satisfies Pick<ChatResponseCatchUpWorkerConfig, "backoffBaseMs" | "maxBackoffMs">;

export async function openBorgDependencies(
  options: BorgOpenOptions = {},
): Promise<BorgDependencies> {
  const clock = options.clock ?? new SystemClock();
  let sqlite: SqliteDatabase | undefined;
  let lance: LanceDbStore | undefined;
  let catchUpWorker: ChatResponseCatchUpWorker | undefined;

  try {
    const resolvedConfig = resolveBorgConfig(options);
    const outboundConnectorRegistry = new MessageConnectorRegistry(
      options.outboundConnectors ?? [],
    );
    const config = {
      ...resolvedConfig,
      host_capabilities: withDerivedOutboundCapabilities({
        hostCapabilities: resolvedConfig.host_capabilities,
        outboundSourceTypes: outboundConnectorRegistry.sourceTypes(),
      }),
    };
    const tracer = compositeTracer([
      createTurnTracer({
        tracerPath: options.tracerPath,
        env: options.env ?? process.env,
        clock,
      }),
      options.tracer,
    ]);
    const storage = openBorgStorage(config);
    sqlite = storage.sqlite;
    lance = storage.lance;
    const tables: BorgLanceTables = await openBorgLanceTables({
      lance,
      embeddingDimensions: options.embeddingDimensions ?? config.embedding.dims,
    });
    const attachmentRepository = new AttachmentRepository(sqlite);
    const entryIndex = new StreamEntryIndexRepository({
      db: sqlite,
      dataDir: config.dataDir,
    });
    const imageAttachmentLifecycleService = new ImageAttachmentLifecycleService({
      db: sqlite,
      attachmentRepository,
      tracer,
    });
    const attachmentService = new AttachmentService({
      repository: attachmentRepository,
      blobStore: new AttachmentBlobStore(config.dataDir),
      config: config.attachments,
      entryIndex,
      createStreamReader: (sessionId) =>
        new StreamReader({
          dataDir: config.dataDir,
          sessionId,
          entryIndex,
        }),
      lifecycle: imageAttachmentLifecycleService,
      tracer,
    });
    const embeddingClient = options.embeddingClient ?? createEmbeddingClient(config);
    const llmFactory = createLlmFactory(
      config,
      options.llmClient,
      options.env,
      clock,
      attachmentService,
    );
    const lazyLlmClient = createLazyLlmClient(llmFactory);
    const onStreamAppend: NonNullable<BorgOpenOptions["onStreamAppend"]> = (entries) => {
      try {
        options.onStreamAppend?.(entries);
      } finally {
        catchUpWorker?.onAppend(entries);
      }
    };
    const repositories = await buildBorgRepositories({
      config,
      sqlite,
      episodesTable: tables.episodesTable,
      semanticNodesTable: tables.semanticNodesTable,
      openQuestionsTable: tables.openQuestionsTable,
      skillsTable: tables.skillsTable,
      actionRecordsTable: tables.actionRecordsTable,
      imagePerceptionsTable: tables.imagePerceptionsTable,
      observedEventsTable: tables.observedEventsTable,
      embeddingClient,
      llmClient: lazyLlmClient,
      clock,
      tracer,
      attachmentRepository,
      entryIndex,
      onStreamAppend,
    });
    const imagePerceptionService = new ImagePerceptionService({
      repository: repositories.imagePerceptionRepository,
      attachmentRepository,
      llmClient: lazyLlmClient,
      embeddingClient,
      model: config.anthropic.models.imagePerception,
      promptVersion: config.attachments.perceptionPromptVersion,
      clock,
      tracer,
    });
    const sessionLock = new SessionLock({
      dataDir: config.dataDir,
    });
    const streamWatermarkRepository = new StreamWatermarkRepository({
      db: sqlite,
      clock,
    });
    const chatResponseWatermarkCoordinator = new ChatResponseWatermarkCoordinator({
      watermarkRepository: streamWatermarkRepository,
      entryIndex: repositories.entryIndex,
    });
    const toolDispatcher = buildToolDispatcher({
      retrievalPipeline: repositories.retrievalPipeline,
      episodicRepository: repositories.episodicRepository,
      semanticNodeRepository: repositories.semanticNodeRepository,
      semanticGraph: repositories.semanticGraph,
      commitmentRepository: repositories.commitmentRepository,
      identityService: repositories.identityService,
      skillRepository: repositories.skillRepository,
      scheduledWakesRepository: repositories.scheduledWakesRepository,
      createStreamWriter: repositories.createStreamWriter,
      clock,
    });
    const outboundDelivery = new OutboundDelivery({
      connectorRegistry: outboundConnectorRegistry,
      createStreamWriter: repositories.createStreamWriter,
      clock,
    });
    const autonomousOutboundPolicy = new AutonomousOutboundPolicy({
      config: config.autonomy.proactiveOutbound,
      sessionsRepository: repositories.sessionsRepository,
      creatorDirectiveRepository: repositories.creatorDirectiveRepository,
      createStreamReader: (sessionId) =>
        new StreamReader({
          dataDir: config.dataDir,
          sessionId,
          entryIndex,
        }),
      transportSourceTypes: outboundConnectorRegistry.sourceTypes(),
      clock,
    });
    const offline = buildOfflineSetup({
      config,
      sqlite,
      clock,
      embeddingClient,
      lazyLlmClient,
      entryIndex: repositories.entryIndex,
      episodicRepository: repositories.episodicRepository,
      semanticNodeRepository: repositories.semanticNodeRepository,
      semanticEdgeRepository: repositories.semanticEdgeRepository,
      semanticBeliefDependencyRepository: repositories.semanticBeliefDependencyRepository,
      semanticReviewService: repositories.semanticReviewService,
      reviewQueueRepository: repositories.reviewQueueRepository,
      identityService: repositories.identityService,
      valuesRepository: repositories.valuesRepository,
      goalsRepository: repositories.goalsRepository,
      traitsRepository: repositories.traitsRepository,
      autobiographicalRepository: repositories.autobiographicalRepository,
      growthMarkersRepository: repositories.growthMarkersRepository,
      openQuestionsRepository: repositories.openQuestionsRepository,
      moodRepository: repositories.moodRepository,
      actionRepository: repositories.actionRepository,
      socialRepository: repositories.socialRepository,
      entityRepository: repositories.entityRepository,
      relationalSlotRepository: repositories.relationalSlotRepository,
      commitmentRepository: repositories.commitmentRepository,
      creatorDirectiveRepository: repositories.creatorDirectiveRepository,
      skillRepository: repositories.skillRepository,
      proceduralEvidenceRepository: repositories.proceduralEvidenceRepository,
      workingMemoryStore: repositories.workingMemoryStore,
      retrievalPipeline: repositories.retrievalPipeline,
      createStreamWriter: repositories.createStreamWriter,
      tracer,
    });
    const streamIngestionCoordinator = buildStreamIngestionCoordinator({
      enabled: options.liveExtraction ?? true,
      config,
      episodicRepository: repositories.episodicRepository,
      embeddingClient,
      lazyLlmClient,
      entityRepository: repositories.entityRepository,
      relationalSlotRepository: repositories.relationalSlotRepository,
      workingMemoryStore: repositories.workingMemoryStore,
      streamWatermarkRepository,
      chatResponseWatermarkCoordinator,
      createStreamWriter: repositories.createStreamWriter,
      clock,
    });
    const repairSessionStreamEntryIndex = (sessionId: SessionId) =>
      backfillSessionStreamEntryIndexAndAttachments({
        dataDir: config.dataDir,
        sessionId,
        entryIndex: repositories.entryIndex,
        attachmentRepository,
      });
    const messageEnqueuer = new MessageEnqueuer({
      sessionsRepository: repositories.sessionsRepository,
      entityRepository: repositories.entityRepository,
      activityRepository: repositories.activityRepository,
      attachmentService,
      imagePerceptionService,
      entryIndex: repositories.entryIndex,
      createReceiptStreamWriter: repositories.createNonNotifyingStreamWriter,
      repairSessionStreamEntryIndex,
      isDuplicatePendingResponse: (record) => {
        if (record.receipt_pending === true) {
          return true;
        }

        if (record.kind !== "user_msg" || record.turn_id !== null || record.entry_index === null) {
          return false;
        }

        const watermark = chatResponseWatermarkCoordinator.getWatermark(record.session_id);

        if (watermark === null) {
          return true;
        }

        return (
          record.entry_index >
          chatResponseWatermarkCoordinator.cursorEntryIndex(
            record.session_id,
            watermark,
            "duplicate response watermark",
          )
        );
      },
      onReceiptReady: (event) => {
        try {
          if (event.entries.length > 0) {
            options.onStreamAppend?.(event.entries);
          }
        } catch (error) {
          console.error("Stream append observer failed", {
            sessionId: event.sessionId,
            entryIds: event.entries.map((entry) => entry.id),
            cause: error instanceof Error ? error.message : String(error),
          });
        } finally {
          catchUpWorker?.onPendingSession(event.sessionId, event.pendingAt);
        }
      },
      clock,
    });
    const turnOrchestrator = buildTurnOrchestrator({
      config,
      retrievalPipeline: repositories.retrievalPipeline,
      embeddingClient,
      episodicRepository: repositories.episodicRepository,
      semanticNodeRepository: repositories.semanticNodeRepository,
      entityRepository: repositories.entityRepository,
      commitmentRepository: repositories.commitmentRepository,
      creatorDirectiveRepository: repositories.creatorDirectiveRepository,
      sharedStateRepository: repositories.sharedStateRepository,
      activityRepository: repositories.activityRepository,
      selfDecisionRepository: repositories.selfDecisionRepository,
      trainOfThoughtRepository: repositories.trainOfThoughtRepository,
      observedEventRepository: repositories.observedEventRepository,
      reviewQueueRepository: repositories.reviewQueueRepository,
      identityService: repositories.identityService,
      valuesRepository: repositories.valuesRepository,
      goalsRepository: repositories.goalsRepository,
      traitsRepository: repositories.traitsRepository,
      autobiographicalRepository: repositories.autobiographicalRepository,
      growthMarkersRepository: repositories.growthMarkersRepository,
      openQuestionsRepository: repositories.openQuestionsRepository,
      executiveStepsRepository: repositories.executiveStepsRepository,
      moodRepository: repositories.moodRepository,
      actionRepository: repositories.actionRepository,
      socialRepository: repositories.socialRepository,
      skillRepository: repositories.skillRepository,
      proceduralEvidenceRepository: repositories.proceduralEvidenceRepository,
      relationalSlotRepository: repositories.relationalSlotRepository,
      skillSelector: repositories.skillSelector,
      workingMemoryStore: repositories.workingMemoryStore,
      llmFactory,
      toolDispatcher,
      sessionLock,
      streamIngestionCoordinator,
      chatResponseWatermarkCoordinator,
      outboundDelivery,
      autonomousOutboundPolicy,
      outboundSourceTypes: outboundConnectorRegistry.sourceTypes(),
      createStreamWriter: repositories.createStreamWriter,
      entryIndex: repositories.entryIndex,
      attachmentService,
      attachmentRepository,
      imagePerceptionRepository: repositories.imagePerceptionRepository,
      imagePerceptionService,
      clock,
      tracer,
      promptOverrideRepository: repositories.promptOverrideRepository,
      sessionsRepository: repositories.sessionsRepository,
    });
    toolDispatcher.register(
      createOutboundPostTool({
        sessionsRepository: repositories.sessionsRepository,
        connectorRegistry: outboundConnectorRegistry,
        autonomousOutboundPolicy,
        postOutbound: (input) =>
          runDirectedOutboundTurn(
            {
              turnOrchestrator,
            },
            input,
          ),
      }),
    );
    const chatResponseBacklogPrefixBuilder = new ChatResponseBacklogPrefixBuilder({
      entryIndex: repositories.entryIndex,
      createStreamReader: (sessionId) =>
        new StreamReader({
          dataDir: config.dataDir,
          sessionId,
          entryIndex: repositories.entryIndex,
        }),
    });
    catchUpWorker = new ChatResponseCatchUpWorker({
      coordinator: chatResponseWatermarkCoordinator,
      prefixBuilder: chatResponseBacklogPrefixBuilder,
      entryIndex: repositories.entryIndex,
      repairSessionStreamEntryIndex,
      turnOrchestrator,
      clock,
      config: {
        quietWindowMs: config.streamIngestion.settle.settleMs,
        maxWaitMs: config.streamIngestion.settle.maxSettleMs,
        ...CHAT_RESPONSE_CATCH_UP_BACKOFF_CONFIG,
      },
    });
    const autonomyScheduler = buildAutonomyScheduler({
      config,
      commitmentRepository: repositories.commitmentRepository,
      episodicRepository: repositories.episodicRepository,
      embeddingClient,
      goalsRepository: repositories.goalsRepository,
      executiveStepsRepository: repositories.executiveStepsRepository,
      openQuestionsRepository: repositories.openQuestionsRepository,
      moodRepository: repositories.moodRepository,
      streamWatermarkRepository,
      autonomyWakesRepository: repositories.autonomyWakesRepository,
      scheduledWakesRepository: repositories.scheduledWakesRepository,
      selfDecisionRepository: repositories.selfDecisionRepository,
      trainOfThoughtRepository: repositories.trainOfThoughtRepository,
      turnOrchestrator,
      toolDispatcher,
      createStreamWriter: repositories.createStreamWriter,
      clock,
      tracer,
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
      attachmentRepository,
      attachmentService,
      imagePerceptionRepository: repositories.imagePerceptionRepository,
      imageAttachmentLifecycleService,
      promptOverrideRepository: repositories.promptOverrideRepository,
      outboundConnectorRegistry,
      outboundDelivery,
      autonomousOutboundPolicy,
      sessionsRepository: repositories.sessionsRepository,
      episodicRepository: repositories.episodicRepository,
      semanticNodeRepository: repositories.semanticNodeRepository,
      semanticEdgeRepository: repositories.semanticEdgeRepository,
      semanticBeliefDependencyRepository: repositories.semanticBeliefDependencyRepository,
      semanticReviewService: repositories.semanticReviewService,
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
      executiveStepsRepository: repositories.executiveStepsRepository,
      moodRepository: repositories.moodRepository,
      actionRepository: repositories.actionRepository,
      socialRepository: repositories.socialRepository,
      entityRepository: repositories.entityRepository,
      commitmentRepository: repositories.commitmentRepository,
      creatorDirectiveRepository: repositories.creatorDirectiveRepository,
      sharedStateRepository: repositories.sharedStateRepository,
      activityRepository: repositories.activityRepository,
      correctionService: repositories.correctionService,
      skillRepository: repositories.skillRepository,
      proceduralContextStatsRepository: repositories.proceduralContextStatsRepository,
      proceduralEvidenceRepository: repositories.proceduralEvidenceRepository,
      relationalSlotRepository: repositories.relationalSlotRepository,
      skillSelector: repositories.skillSelector,
      retrievalPipeline: repositories.retrievalPipeline,
      workingMemoryStore: repositories.workingMemoryStore,
      autonomyWakesRepository: repositories.autonomyWakesRepository,
      scheduledWakesRepository: repositories.scheduledWakesRepository,
      selfDecisionRepository: repositories.selfDecisionRepository,
      trainOfThoughtRepository: repositories.trainOfThoughtRepository,
      observedEventRepository: repositories.observedEventRepository,
      turnOrchestrator,
      autonomyScheduler,
      maintenanceScheduler,
      streamIngestionCoordinator,
      chatResponseWatermarkCoordinator,
      chatResponseCatchUpWorker: catchUpWorker,
      messageEnqueuer,
      auditLog: offline.auditLog,
      maintenanceOrchestrator: offline.maintenanceOrchestrator,
      offlineProcesses: offline.offlineProcesses,
      createStreamWriter: repositories.createStreamWriter,
      llmFactory,
      embeddingClient,
      tracer,
      clock,
    };
  } catch (error) {
    await closeBestEffort(sqlite, lance);
    throw error;
  }
}
