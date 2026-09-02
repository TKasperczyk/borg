// Public Borg facade adapters over the internal repository and service graph.

import { EpisodicExtractor } from "../memory/episodic/index.js";
import {
  SemanticExtractor,
  createUserStreamEntryRelationshipEvidenceTrustValidator,
} from "../memory/semantic/index.js";
import { buildParticipantRosterFromRepositories } from "../cognition/perception/index.js";
import { resolveEpisodeSourceParticipants } from "../cognition/participants.js";
import {
  PROMPT_BLOCKS,
  getPromptBlockSpec,
  type PromptKey,
} from "../cognition/prompts/registry.js";
import {
  buildCacheableBaseSystemPromptParts,
  buildBaseSystemPromptSections,
  createAssembledFramingPreviewContext,
  createBasePromptSurfaceRenderContext,
} from "../cognition/deliberation/prompt/system-prompt.js";
import {
  PROMPT_SURFACES,
  promptSurfaceBlocksForSurface,
} from "../cognition/prompts/prompt-surface-registry.js";
import type { BorgPromptBlockView, BorgPromptsFacade } from "./facade-types.js";
import {
  OFFLINE_PROCESS_NAMES,
  revalidateReviewQueue,
  runStorageOptimization,
} from "../offline/index.js";
import type { MaintenancePlan, OfflineProcessName, OrchestratorResult } from "../offline/index.js";
import type { DisclosureRetrievalOptions } from "../retrieval/index.js";
import {
  mapWithDisclosureConcurrency,
  memoryDisclosureLabelForEpisodeIds,
  resolveMemoryDisclosureLabelForEpisodeIds,
  resolveMemoryDisclosureLabelsByEpisodeId,
  type MemoryDisclosureLabel,
} from "../retrieval/index.js";
import { StreamReader } from "../stream/index.js";
import { AttachmentError, StorageError } from "../util/errors.js";
import {
  DEFAULT_SESSION_ID,
  createSemanticNodeId,
  type EntityId,
  type ImagePerceptionId,
} from "../util/ids.js";
import type { BorgFacades } from "./facade-types.js";
import type {
  SemanticEdge,
  SemanticNode,
  SemanticNodeSearchCandidate,
  SemanticWalkStep,
} from "../memory/semantic/index.js";
import type {
  BorgDependencies,
  BorgDreamOptions,
  BorgDreamRunner,
  BorgEpisodeSearchOptions,
} from "./types.js";

function errorCode(error: unknown): unknown {
  return error !== null && typeof error === "object" && "code" in error
    ? (error as { code?: unknown }).code
    : undefined;
}

function isCommittedStreamIndexUpdateFailure(error: unknown): boolean {
  const code = errorCode(error);

  return code === "STREAM_INDEX_UPDATE_FAILED" || code === "STREAM_INDEX_POISONED";
}

function describeCaughtError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function participationPolicyRollbackFailedError(
  originalError: unknown,
  rollbackError: unknown,
): StorageError {
  return new StorageError(
    [
      "Failed to roll back participation policy after audit append failed.",
      `Original append error: ${describeCaughtError(originalError)}.`,
      `Rollback error: ${describeCaughtError(rollbackError)}.`,
    ].join(" "),
    {
      cause: originalError,
    },
  );
}

function createActionsFacade(deps: BorgDependencies): BorgFacades["actions"] {
  return new Proxy(deps.actionRepository, {
    get(target, property) {
      if (property === "add") {
        return (...args: Parameters<typeof target.add>) =>
          target.add(args[0], {
            ...args[1],
            creationSource: "api",
          });
      }

      const value = Reflect.get(target, property, target);

      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

async function countPendingSemanticExtractionEpisodes(
  deps: Pick<
    BorgDependencies,
    "auditLog" | "episodicRepository" | "semanticEdgeRepository" | "semanticNodeRepository"
  >,
): Promise<number> {
  const episodeIds = await deps.episodicRepository.listUnarchivedEpisodeIds();
  const processedEpisodeIds = new Set([
    ...deps.semanticNodeRepository.listAllSourceEpisodeIds(),
    ...deps.semanticEdgeRepository.listAllEvidenceEpisodeIds(),
  ]);

  for (const audit of deps.auditLog.list({ process: "semantic-extractor", reverted: false })) {
    const auditEpisodeIds = audit.targets.episode_ids;

    if (!Array.isArray(auditEpisodeIds)) {
      continue;
    }

    for (const episodeId of auditEpisodeIds) {
      if (typeof episodeId === "string") {
        processedEpisodeIds.add(episodeId as (typeof episodeIds)[number]);
      }
    }
  }

  return episodeIds.reduce(
    (pending, episodeId) => pending + (processedEpisodeIds.has(episodeId) ? 0 : 1),
    0,
  );
}

type CreatorDirectivesFacadeDeps = Pick<BorgDependencies, "creatorDirectiveRepository">;

function createActivityFacade(deps: BorgDependencies): BorgFacades["activity"] {
  return {
    record: (...args) => deps.activityRepository.record(...args),
    projectCompletedTurn: (input) => {
      const project = deps.sqlite.transaction(() => {
        const userContactAlreadyStored =
          deps.activityRepository.getByKindAndSource(
            input.userContact.kind,
            input.userContact.sourceStreamEntryIds,
          ) !== null;
        const borgReplyAlreadyStored =
          deps.activityRepository.getByKindAndSource(
            input.borgReplied.kind,
            input.borgReplied.sourceStreamEntryIds,
          ) !== null;
        const session = deps.sessionsRepository.ensure(input.session);
        const userContact = deps.activityRepository.record(input.userContact);
        const borgReplied = deps.activityRepository.record(input.borgReplied);
        const touchedSession =
          userContactAlreadyStored && borgReplyAlreadyStored
            ? session
            : deps.sessionsRepository.touch(input.session.session_id, input.touch);

        if (touchedSession === null) {
          throw new StorageError(`Session ${input.session.session_id} was not stored`, {
            code: "SESSION_TURN_PROJECTION_FAILED",
          });
        }

        return { userContact, borgReplied, session: touchedSession };
      });

      return project.immediate();
    },
    listObservedGroupAudienceEntityIdsForSpeaker: (...args) =>
      deps.activityRepository.listObservedGroupAudienceEntityIdsForSpeaker(...args),
    listRecentVisibleOtherSessionEvents: (...args) =>
      deps.activityRepository.listRecentVisibleOtherSessionEvents(...args),
  };
}

export function createCreatorDirectivesFacade(
  deps: CreatorDirectivesFacadeDeps,
): BorgFacades["creatorDirectives"] {
  return {
    queue: (...args) => deps.creatorDirectiveRepository.queue(...args),
    get: (...args) => deps.creatorDirectiveRepository.get(...args),
    list: (...args) => deps.creatorDirectiveRepository.list(...args),
    listApplicable: (...args) => deps.creatorDirectiveRepository.listApplicable(...args),
    supersede: (...args) => deps.creatorDirectiveRepository.supersede(...args),
    supersedeFamilyAtomic: (...args) =>
      deps.creatorDirectiveRepository.supersedeFamilyAtomic(...args),
    revoke: (...args) => deps.creatorDirectiveRepository.revoke(...args),
  };
}

async function semanticDisclosureLabel(
  deps: BorgDependencies,
  episodeIds: readonly SemanticNode["source_episode_ids"][number][],
): Promise<MemoryDisclosureLabel> {
  return resolveMemoryDisclosureLabelForEpisodeIds(deps.episodicRepository, episodeIds);
}

async function semanticNodeWithDisclosure(
  deps: BorgDependencies,
  node: SemanticNode,
): Promise<SemanticNode & { disclosureLabel: MemoryDisclosureLabel }> {
  return {
    ...node,
    disclosureLabel: await semanticDisclosureLabel(deps, node.source_episode_ids),
  };
}

function semanticNodeWithBatchedDisclosure(
  node: SemanticNode,
  labelsByEpisodeId: ReadonlyMap<string, MemoryDisclosureLabel>,
): SemanticNode & { disclosureLabel: MemoryDisclosureLabel } {
  return {
    ...node,
    disclosureLabel: memoryDisclosureLabelForEpisodeIds(node.source_episode_ids, labelsByEpisodeId),
  };
}

async function semanticNodesWithDisclosure(
  deps: BorgDependencies,
  nodes: readonly SemanticNode[],
): Promise<Array<SemanticNode & { disclosureLabel: MemoryDisclosureLabel }>> {
  const labelsByEpisodeId = await resolveMemoryDisclosureLabelsByEpisodeId(
    deps.episodicRepository,
    nodes.flatMap((node) => node.source_episode_ids),
  );

  return mapWithDisclosureConcurrency(nodes, async (node) =>
    semanticNodeWithBatchedDisclosure(node, labelsByEpisodeId),
  );
}

async function semanticEdgeWithDisclosure(
  deps: BorgDependencies,
  edge: SemanticEdge,
): Promise<SemanticEdge & { disclosureLabel: MemoryDisclosureLabel }> {
  return {
    ...edge,
    disclosureLabel: await semanticDisclosureLabel(deps, edge.evidence_episode_ids),
  };
}

function semanticEdgeWithBatchedDisclosure(
  edge: SemanticEdge,
  labelsByEpisodeId: ReadonlyMap<string, MemoryDisclosureLabel>,
): SemanticEdge & { disclosureLabel: MemoryDisclosureLabel } {
  return {
    ...edge,
    disclosureLabel: memoryDisclosureLabelForEpisodeIds(
      edge.evidence_episode_ids,
      labelsByEpisodeId,
    ),
  };
}

async function semanticEdgesWithDisclosure(
  deps: BorgDependencies,
  edges: readonly SemanticEdge[],
): Promise<Array<SemanticEdge & { disclosureLabel: MemoryDisclosureLabel }>> {
  const labelsByEpisodeId = await resolveMemoryDisclosureLabelsByEpisodeId(
    deps.episodicRepository,
    edges.flatMap((edge) => edge.evidence_episode_ids),
  );

  return mapWithDisclosureConcurrency(edges, async (edge) =>
    semanticEdgeWithBatchedDisclosure(edge, labelsByEpisodeId),
  );
}

async function semanticSearchCandidatesWithDisclosure(
  deps: BorgDependencies,
  candidates: readonly SemanticNodeSearchCandidate[],
): Promise<
  Array<
    SemanticNodeSearchCandidate & {
      node: SemanticNode & { disclosureLabel: MemoryDisclosureLabel };
    }
  >
> {
  const labelsByEpisodeId = await resolveMemoryDisclosureLabelsByEpisodeId(
    deps.episodicRepository,
    candidates.flatMap((candidate) => candidate.node.source_episode_ids),
  );

  return mapWithDisclosureConcurrency(candidates, async (candidate) => ({
    ...candidate,
    node: semanticNodeWithBatchedDisclosure(candidate.node, labelsByEpisodeId),
  }));
}

async function semanticWalkStepsWithDisclosure(
  deps: BorgDependencies,
  steps: readonly SemanticWalkStep[],
): Promise<
  Array<
    Omit<SemanticWalkStep, "node" | "edgePath"> & {
      node: SemanticNode & { disclosureLabel: MemoryDisclosureLabel };
      edgePath: Array<SemanticEdge & { disclosureLabel: MemoryDisclosureLabel }>;
    }
  >
> {
  const labelsByEpisodeId = await resolveMemoryDisclosureLabelsByEpisodeId(
    deps.episodicRepository,
    steps.flatMap((step) => [
      ...step.node.source_episode_ids,
      ...step.edgePath.flatMap((edge) => edge.evidence_episode_ids),
    ]),
  );

  return mapWithDisclosureConcurrency(steps, async (step) => ({
    ...step,
    node: semanticNodeWithBatchedDisclosure(step.node, labelsByEpisodeId),
    edgePath: step.edgePath.map((edge) =>
      semanticEdgeWithBatchedDisclosure(edge, labelsByEpisodeId),
    ),
  }));
}

export function createBorgFacades(deps: BorgDependencies): BorgFacades {
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

    return deps.entityRepository.resolve(options.audience, {
      provenance: "transport_audience_label",
    });
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

    const audienceEntity = deps.entityRepository.get(audienceEntityId);

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
  ): DisclosureRetrievalOptions => {
    const audienceEntityId = resolveEpisodeAudienceEntityId(options);
    const audienceProfile =
      options?.audienceProfile !== undefined
        ? options.audienceProfile
        : audienceEntityId === null || audienceEntityId === undefined
          ? undefined
          : (deps.socialRepository.getProfile(audienceEntityId) ?? undefined);
    const audienceTerms = resolveEpisodeAudienceTerms(options, audienceEntityId);
    const hasTemporalSignal =
      options?.temporalCue !== undefined || options?.timeRange !== undefined;
    const hasEntitySignal = options?.entityTerms !== undefined && options.entityTerms.length > 0;
    const configuredAttentionWeights = deps.config.retrieval.attentionWeights;

    return {
      ...options,
      audienceEntityId,
      audienceProfile,
      audienceTerms,
      strictTimeRange: options?.strictTimeRange ?? options?.timeRange !== undefined,
      attentionWeights:
        options?.attentionWeights ??
        (options?.scoreWeights !== undefined
          ? undefined
          : // Similarity-forward defaults (2026-08 scoring rebalance): semantic
            // 0.65 gives similarity 0.65 / salience 0.35, heat 0.15. The old
            // 0.35/0.65/0.45 split saturated ~90% of returned scores at the
            // 1.0 clamp and let query-independent salience+heat crown one
            // "greatest hit" episode as top-1 for over half the query suite.
            //
            // Those figures are corpus-dependent, not universal -- `semantic`
            // is fused against a raw, un-normalized cosine whose spread varies
            // per bank -- so every value is deployment-tunable via
            // config.retrieval.attentionWeights. See that schema, and measure
            // with `pnpm retrieval:signal-report` before changing them.
            {
              semantic: configuredAttentionWeights.semantic,
              goal_relevance:
                options?.goalDescriptions !== undefined && options.goalDescriptions.length > 0
                  ? configuredAttentionWeights.goal_relevance
                  : 0,
              value_alignment: configuredAttentionWeights.value_alignment,
              mood: configuredAttentionWeights.mood,
              time: hasTemporalSignal ? configuredAttentionWeights.time : 0,
              social:
                audienceTerms !== undefined && audienceTerms.length > 0
                  ? configuredAttentionWeights.social
                  : 0,
              entity: hasEntitySignal ? configuredAttentionWeights.entity : 0,
              heat: configuredAttentionWeights.heat,
              suppression_penalty: configuredAttentionWeights.suppression_penalty,
            }),
    };
  };

  // List membership (light ∪ heavy) is the single enablement authority, but
  // emit the default dream order in canonical OFFLINE_PROCESS_NAMES order so
  // unifying the gate does not also change manual full-dream process sequencing.
  const defaultDreamProcesses = (): OfflineProcessName[] => {
    const selected = new Set<OfflineProcessName>([
      ...deps.config.maintenance.lightProcesses,
      ...deps.config.maintenance.heavyProcesses,
    ]);
    return OFFLINE_PROCESS_NAMES.filter((name) => selected.has(name));
  };

  const maintenanceConfigSnapshot = () => ({
    enabled: deps.config.maintenance.enabled,
    lightIntervalMs: deps.config.maintenance.lightIntervalMs,
    heavyIntervalMs: deps.config.maintenance.heavyIntervalMs,
    optimizeStorage: deps.config.maintenance.optimizeStorage,
    lightBudget: deps.config.maintenance.lightBudget,
    heavyBudget: deps.config.maintenance.heavyBudget,
    lightProcesses: deps.config.maintenance.lightProcesses,
    heavyProcesses: deps.config.maintenance.heavyProcesses,
    processBudgets: {
      consolidator: deps.config.offline.consolidator.budget,
      reflector: deps.config.offline.reflector.budget,
      associator: deps.config.offline.associator.budget,
      "semantic-extractor": deps.config.offline.semanticExtractor.budget,
      curator: null,
      overseer: deps.config.offline.overseer.budget,
      "review-resolver": deps.config.offline.reviewResolver.budget,
      ruminator: deps.config.offline.ruminator.budget,
      "self-narrator": deps.config.offline.selfNarrator.budget,
      "lived-experience-day-summarizer": deps.config.offline.livedExperienceDaySummarizer.budget,
      "procedural-synthesizer": deps.config.offline.proceduralSynthesizer.budget,
      "belief-reviser": null,
      "creator-directive-reconciler": deps.config.offline.creatorDirectiveReconciler.budget,
      "commitment-reconciler": deps.config.offline.commitmentReconciler.budget,
    } satisfies Partial<Record<OfflineProcessName, number | null>>,
  });

  const upsertAutobiographicalPeriod = ((
    input: Parameters<typeof deps.identityService.addPeriod>[0],
  ) => {
    if (input.id === undefined || deps.autobiographicalRepository.getPeriod(input.id) === null) {
      return deps.identityService.addPeriod(input);
    }

    return deps.identityService.updatePeriod(
      input.id,
      {
        label: input.label,
        start_ts: input.start_ts,
        end_ts: input.end_ts ?? null,
        narrative: input.narrative,
        key_episode_ids: [...(input.key_episode_ids ?? [])],
        themes: [...(input.themes ?? [])],
        provenance: input.provenance,
      },
      input.provenance,
    );
  }) as BorgFacades["self"]["autobiographical"]["upsertPeriod"];

  const runDream = async (
    processNames: readonly OfflineProcessName[],
    options: BorgDreamOptions = {},
  ): Promise<OrchestratorResult> => {
    const processes = processNames.map((name) => deps.offlineProcesses[name]);

    return deps.maintenanceOrchestrator.run({
      runId: options.runId,
      processes,
      opts: {
        dryRun: options.dryRun,
        budget: options.budget,
        processOverrides: options.processOverrides,
      },
    });
  };
  const planDream = (processNames: readonly OfflineProcessName[], options: BorgDreamOptions = {}) =>
    deps.maintenanceOrchestrator.plan({
      runId: options.runId,
      processes: processNames.map((name) => deps.offlineProcesses[name]),
      opts: {
        budget: options.budget,
        processOverrides: options.processOverrides,
      },
    });

  return {
    stream: {
      append: async (input, options = {}) => {
        const writer = deps.createStreamWriter(options.session ?? DEFAULT_SESSION_ID);

        try {
          return await writer.append(input);
        } finally {
          writer.close();
        }
      },
      appendMany: async (inputs, options = {}) => {
        const writer = deps.createStreamWriter(options.session ?? DEFAULT_SESSION_ID);

        try {
          return await writer.appendMany(inputs);
        } finally {
          writer.close();
        }
      },
      tail: (n, options = {}) =>
        new StreamReader({
          dataDir: deps.config.dataDir,
          sessionId: options.session ?? DEFAULT_SESSION_ID,
          entryIndex: deps.entryIndex,
        }).tail(n),
      reader: (options = {}) =>
        new StreamReader({
          dataDir: deps.config.dataDir,
          sessionId: options.session ?? DEFAULT_SESSION_ID,
          entryIndex: deps.entryIndex,
        }),
    },
    episodic: {
      get: (id, options = {}) =>
        deps.retrievalPipeline.getEpisode(id, {
          audienceEntityId: resolveEpisodeAudienceEntityId(options),
          visibleAudienceEntityIds: options.visibleAudienceEntityIds,
          crossAudience: options.crossAudience,
        }),
      inspect: (id) => deps.episodicRepository.get(id, { includeArchived: true }),
      search: (query, options = {}) =>
        deps.retrievalPipeline.searchEpisodesForDisclosure(
          query,
          resolveEpisodeSearchOptions(options),
        ),
      extract: async (options = {}) => {
        const extractor = new EpisodicExtractor({
          dataDir: deps.config.dataDir,
          episodicRepository: deps.episodicRepository,
          embeddingClient: deps.embeddingClient,
          llmClient: deps.llmFactory(),
          model: deps.config.anthropic.models.extraction,
          entityRepository: deps.entityRepository,
          relationalSlotRepository: deps.relationalSlotRepository,
          workingMemoryStore: deps.workingMemoryStore,
          defaultUser: deps.config.defaultUser,
          salienceGateEnabled: deps.config.episodic.salienceGateEnabled,
          tracer: deps.tracer,
          clock: deps.clock,
        });

        return extractor.extractFromStream({
          session: options.session ?? DEFAULT_SESSION_ID,
          sinceTs: options.sinceTs,
          sinceCursor: options.sinceCursor,
          untilTs: options.untilTs,
          bypassSalienceGate: options.bypassSalienceGate,
        });
      },
      ingest: (options = {}) =>
        deps.streamIngestionCoordinator?.ingest(options.session ?? DEFAULT_SESSION_ID) ??
        Promise.resolve({ ran: false, processedEntries: 0 }),
      list: (...args) => deps.episodicRepository.list(...args),
      listAll: () => deps.episodicRepository.listAll(),
      getStats: (...args) => deps.episodicRepository.getStats(...args),
    },
    self: {
      values: {
        get: (...args) => deps.valuesRepository.get(...args),
        list: (...args) => deps.valuesRepository.list(...args),
        add: (...args) => deps.identityService.addValue(...args),
        update: (...args) => deps.identityService.updateValue(...args),
        reinforce: (...args) => deps.identityService.reinforceValue(...args),
        listReinforcementEvents: (...args) =>
          deps.valuesRepository.listReinforcementEvents(...args),
        listContradictionEvents: (...args) =>
          deps.valuesRepository.listContradictionEvents(...args),
      },
      goals: {
        get: (...args) => deps.goalsRepository.get(...args),
        list: (...args) => deps.goalsRepository.list(...args),
        add: (...args) => deps.identityService.addGoal(...args),
        update: (...args) => deps.identityService.updateGoal(...args),
        updateStatus: (...args) => deps.identityService.updateGoalStatus(...args),
        updateProgress: (...args) => deps.identityService.updateGoalProgress(...args),
      },
      traits: {
        get: (...args) => deps.traitsRepository.get(...args),
        list: (...args) => deps.traitsRepository.list(...args),
        add: (...args) => deps.identityService.addTrait(...args),
        update: (...args) => deps.identityService.updateTrait(...args),
        reinforce: (...args) => deps.identityService.reinforceTrait(...args),
        listReinforcementEvents: (...args) =>
          deps.traitsRepository.listReinforcementEvents(...args),
        listContradictionEvents: (...args) =>
          deps.traitsRepository.listContradictionEvents(...args),
      },
      autobiographical: {
        currentPeriod: () => deps.autobiographicalRepository.currentPeriod(),
        listPeriods: (...args) => deps.autobiographicalRepository.listPeriods(...args),
        upsertPeriod: upsertAutobiographicalPeriod,
        closePeriod: (...args) => deps.identityService.closePeriod(...args),
        getPeriod: (...args) => deps.autobiographicalRepository.getPeriod(...args),
        getByLabel: (...args) => deps.autobiographicalRepository.getByLabel(...args),
      },
      growthMarkers: {
        list: (...args) => deps.growthMarkersRepository.list(...args),
        add: (...args) => deps.identityService.addGrowthMarker(...args),
        summarize: (...args) => deps.growthMarkersRepository.summarize(...args),
      },
      journal: {
        latest: () => deps.trainOfThoughtRepository.latest(),
        list: (...args) => deps.trainOfThoughtRepository.list(...args),
      },
      openQuestions: {
        list: (...args) => deps.openQuestionsRepository.list(...args),
        add: (...args) => deps.identityService.addOpenQuestion(...args),
        resolve: (...args) => deps.identityService.resolveOpenQuestion(...args),
        abandon: (...args) => deps.identityService.abandonOpenQuestion(...args),
        bumpUrgency: (...args) => deps.identityService.bumpOpenQuestionUrgency(...args),
      },
    },
    skills: {
      list: (...args) => deps.skillRepository.list(...args),
      add: (...args) => deps.skillRepository.add(...args),
      get: (...args) => deps.skillRepository.get(...args),
      searchByContext: (...args) => deps.skillRepository.searchByContext(...args),
      recordOutcome: (...args) => deps.skillRepository.recordOutcome(...args),
      select: (...args) => deps.skillSelector.select(...args),
    },
    mood: {
      current: (...args) => deps.moodRepository.current(...args),
      history: (...args) => deps.moodRepository.history(...args),
      update: (...args) => deps.moodRepository.update(...args),
    },
    actions: createActionsFacade(deps),
    social: {
      getProfile: (entity) =>
        deps.socialRepository.getProfile(deps.entityRepository.resolve(entity)),
      list: (...args) => deps.socialRepository.list(...args),
      upsertProfile: (entity) =>
        deps.socialRepository.upsertProfile(deps.entityRepository.resolve(entity)),
      recordInteraction: (entity, interaction) =>
        deps.socialRepository.recordInteraction(deps.entityRepository.resolve(entity), interaction),
      adjustTrust: (entity, delta, provenance) =>
        deps.socialRepository.adjustTrust(deps.entityRepository.resolve(entity), delta, provenance),
    },
    entities: {
      resolve: (...args) => deps.entityRepository.resolve(...args),
      resolveExternal: (...args) => deps.entityRepository.resolveExternal(...args),
      findByExternalId: (...args) => deps.entityRepository.findByExternalId(...args),
      get: (...args) => deps.entityRepository.get(...args),
      list: (...args) => deps.entityRepository.list(...args),
      getCreator: () => deps.entityRepository.getCreator(),
      getSelf: () => deps.entityRepository.getSelf(),
      ensureSelf: (...args) => deps.entityRepository.ensureSelf(...args),
      setBorgRole: (...args) => deps.entityRepository.setBorgRole(...args),
      find: (name, options) => {
        const entityId = deps.entityRepository.findByName(name, options);
        return entityId === null ? null : deps.entityRepository.get(entityId);
      },
    },
    sharedState: {
      getForAudience: (audience) => {
        const entityId = deps.entityRepository.findByName(audience);
        return entityId === null ? null : deps.sharedStateRepository.get(entityId);
      },
      listEntriesForAudience: (audience) => {
        const entityId = deps.entityRepository.findByName(audience);
        return entityId === null ? [] : (deps.sharedStateRepository.get(entityId)?.entries ?? []);
      },
    },
    attachments: {
      get: (attachmentId) => {
        const attachment = deps.attachmentRepository.get(attachmentId);

        if (attachment === null) {
          return null;
        }

        const streamFacts =
          attachment.stream_entry_id === null
            ? null
            : deps.entryIndex.lookup(attachment.stream_entry_id);
        const parentFacts = deps.entryIndex.lookup(attachment.parent_entry_id);

        return {
          attachment,
          perception:
            attachment.perception_id === null
              ? null
              : deps.imagePerceptionRepository.get(attachment.perception_id as ImagePerceptionId),
          status: {
            active:
              attachment.active && streamFacts?.active !== false && parentFacts?.active !== false,
            quarantined:
              !attachment.active || streamFacts?.active === false || parentFacts?.active === false,
            ...(streamFacts === null ? {} : { stream_active: streamFacts.active }),
            ...(parentFacts === null ? {} : { parent_active: parentFacts.active }),
          },
        };
      },
      getBytes: (attachmentId, options = {}) => {
        const attachment = deps.attachmentRepository.get(attachmentId);

        if (attachment === null) {
          return null;
        }

        if (options.audience !== undefined && attachment.audience !== options.audience) {
          return null;
        }

        try {
          const image = deps.attachmentService.fetchImageForLlm(attachmentId);

          return {
            attachment,
            mediaType: attachment.media_type,
            bytes: image.bytes,
          };
        } catch (error) {
          if (error instanceof AttachmentError) {
            return null;
          }

          throw error;
        }
      },
    },
    semantic: {
      nodes: {
        add: async (input) => {
          const nowMs = deps.clock.now();
          const embedding = await deps.embeddingClient.embed(
            `${input.label}\n${input.description}\n${input.aliases?.join(" ") ?? ""}`,
          );

          return deps.semanticNodeRepository.insert({
            id: createSemanticNodeId(),
            kind: input.kind,
            label: input.label,
            description: input.description,
            domain: input.domain ?? null,
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
        get: async (id) => {
          const node = await deps.semanticNodeRepository.get(id);
          return node === null ? null : semanticNodeWithDisclosure(deps, node);
        },
        list: async (...args) => {
          const nodes = await deps.semanticNodeRepository.list(...args);
          return semanticNodesWithDisclosure(deps, nodes);
        },
        listPage: async (...args) => {
          const page = await deps.semanticNodeRepository.listPage(...args);
          return {
            ...page,
            items: await semanticNodesWithDisclosure(deps, page.items),
          };
        },
        countByStatus: () => deps.semanticNodeRepository.countByStatus(),
        search: async (query, options = {}) => {
          const vector = await deps.embeddingClient.embed(query);
          const results = await deps.semanticNodeRepository.searchByVector(vector, {
            limit: options.limit,
          });
          return semanticSearchCandidatesWithDisclosure(deps, results);
        },
      },
      edges: {
        add: (input) => deps.semanticEdgeRepository.addEdge(input),
        get: async (id) => {
          const edge = deps.semanticEdgeRepository.getEdge(id);
          return edge === null ? null : semanticEdgeWithDisclosure(deps, edge);
        },
        list: async (...args) =>
          semanticEdgesWithDisclosure(deps, deps.semanticEdgeRepository.listEdges(...args)),
      },
      walk: async (fromId, ...args) => {
        const steps = await deps.semanticGraph.walk(fromId, ...args);
        return semanticWalkStepsWithDisclosure(deps, steps);
      },
      extract: async (episodes) => {
        const selfEntity = deps.entityRepository.getSelf();
        const sourceParticipants = resolveEpisodeSourceParticipants({
          episodes,
          entryIndex: deps.entryIndex,
          entityRepository: deps.entityRepository,
        });
        const extractor = new SemanticExtractor({
          nodeRepository: deps.semanticNodeRepository,
          edgeRepository: deps.semanticEdgeRepository,
          embeddingClient: deps.embeddingClient,
          episodicRepository: deps.episodicRepository,
          llmClient: deps.llmFactory(),
          model: deps.config.anthropic.models.extraction,
          semanticReviewService: deps.semanticReviewService,
          reviewEnqueue: (input) => deps.reviewQueueRepository.enqueue(input),
          participantRoster: buildParticipantRosterFromRepositories({
            activeParticipants: [
              ...(selfEntity === null
                ? []
                : [
                    {
                      entityId: selfEntity.id,
                      displayName: selfEntity.canonical_name,
                      role: "participant" as const,
                    },
                  ]),
              ...sourceParticipants.filter(
                (participant) => participant.entityId !== selfEntity?.id,
              ),
            ],
            entityRepository: deps.entityRepository,
            relationalSlotRepository: deps.relationalSlotRepository,
          }),
          selfEntityId: selfEntity?.id ?? null,
          entityRepository: deps.entityRepository,
          relationshipEvidenceStreamEntryTrust:
            createUserStreamEntryRelationshipEvidenceTrustValidator({
              entryIndex: deps.entryIndex,
              createStreamReader: (sessionId) =>
                new StreamReader({
                  dataDir: deps.config.dataDir,
                  sessionId,
                  entryIndex: deps.entryIndex,
                }),
              isActiveAttachmentStreamEntry: (streamEntryId) =>
                deps.attachmentRepository.isActiveForStreamEntry(streamEntryId),
            }),
          clock: deps.clock,
        });

        return extractor.extractFromEpisodes(episodes);
      },
    },
    relationalSlots: {
      list: (...args) => deps.relationalSlotRepository.list(...args),
      countByState: () => deps.relationalSlotRepository.countByState(),
    },
    commitments: {
      add: (input) =>
        deps.identityService.addCommitment({
          type: input.type,
          kind: input.kind,
          enforcementClass: input.enforcementClass,
          criticalDomain: input.criticalDomain,
          directiveFamily: input.directiveFamily,
          directive: input.directive,
          priority: input.priority,
          madeToEntity:
            input.madeTo === undefined || input.madeTo === null
              ? null
              : deps.entityRepository.resolve(input.madeTo),
          restrictedAudience:
            input.audience === undefined || input.audience === null
              ? null
              : deps.entityRepository.resolve(input.audience),
          aboutEntity:
            input.about === undefined || input.about === null
              ? null
              : deps.entityRepository.resolve(input.about),
          provenance: input.provenance,
          expiresAt: input.expiresAt ?? null,
        }),
      get: (id) => deps.commitmentRepository.get(id),
      revoke: (...args) => deps.commitmentRepository.revoke(...args),
      list: (options = {}) =>
        deps.commitmentRepository.list({
          activeOnly: options.activeOnly,
          audience:
            options.audienceEntityId !== undefined
              ? options.audienceEntityId
              : options.audience === undefined
                ? undefined
                : options.audience === null
                  ? null
                  : deps.entityRepository.resolve(options.audience),
          aboutEntity:
            options.aboutEntity === undefined
              ? undefined
              : options.aboutEntity === null
                ? null
                : deps.entityRepository.resolve(options.aboutEntity),
        }),
      countActive: () => deps.commitmentRepository.countActive(),
      countActiveByKind: () => deps.commitmentRepository.countActiveByKind(),
      countActiveByEnforcementClass: () =>
        deps.commitmentRepository.countActiveByEnforcementClass(),
      countSuperseded: () => deps.commitmentRepository.countSuperseded(),
      countRevoked: () => deps.commitmentRepository.countRevoked(),
      countExpired: () => deps.commitmentRepository.countExpired(),
      countCanonicalized: () => deps.commitmentRepository.countCanonicalized(),
    },
    activity: createActivityFacade(deps),
    creatorDirectives: createCreatorDirectivesFacade(deps),
    identity: {
      updateValue: (...args) => deps.identityService.updateValue(...args),
      updateGoal: (...args) => deps.identityService.updateGoal(...args),
      updateTrait: (...args) => deps.identityService.updateTrait(...args),
      addCommitment: (...args) => deps.identityService.addCommitment(...args),
      updateCommitment: (...args) => deps.identityService.updateCommitment(...args),
      updatePeriod: (...args) => deps.identityService.updatePeriod(...args),
      updateGrowthMarker: (...args) => deps.identityService.updateGrowthMarker(...args),
      updateOpenQuestion: (...args) => deps.identityService.updateOpenQuestion(...args),
      listEvents: (...args) => deps.identityService.listEvents(...args),
    },
    correction: {
      forget: (...args) => deps.correctionService.forget(...args),
      why: (...args) => deps.correctionService.why(...args),
      invalidateSemanticEdge: (...args) => deps.correctionService.invalidateSemanticEdge(...args),
      correct: (...args) => deps.correctionService.correct(...args),
      rememberAboutMe: (...args) => deps.correctionService.rememberAboutMe(...args),
      listIdentityEvents: (...args) => deps.correctionService.listIdentityEvents(...args),
    },
    review: {
      list: (options = {}) => deps.reviewQueueRepository.list(options),
      resolve: (id, decision, options) => deps.reviewQueueRepository.resolve(id, decision, options),
      revalidate: (options) =>
        revalidateReviewQueue(
          {
            clock: deps.clock,
            episodicRepository: deps.episodicRepository,
            retrievalPipeline: deps.retrievalPipeline,
            reviewQueueRepository: deps.reviewQueueRepository,
          },
          options,
        ),
    },
    audit: {
      list: (options = {}) =>
        deps.auditLog.list({
          run_id: options.runId,
          process: options.process,
          reverted: options.reverted,
        }),
      revert: (id, revertedBy) => deps.auditLog.revert(id, revertedBy),
    },
    dream: Object.assign(
      async (options: BorgDreamOptions = {}) =>
        runDream(options.processes ?? defaultDreamProcesses(), options),
      {
        plan: (options: Omit<BorgDreamOptions, "dryRun"> = {}) =>
          planDream(options.processes ?? defaultDreamProcesses(), options),
        preview: (plan: MaintenancePlan) => deps.maintenanceOrchestrator.preview(plan),
        apply: (plan: MaintenancePlan) => deps.maintenanceOrchestrator.apply(plan),
        consolidate: (options = {}) => runDream(["consolidator"], options),
        reflect: (options = {}) => runDream(["reflector"], options),
        associate: (options = {}) => runDream(["associator"], options),
        extractSemantics: (options = {}) => runDream(["semantic-extractor"], options),
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
    ) satisfies BorgDreamRunner,
    autonomy: {
      scheduler: deps.autonomyScheduler,
      wakes: deps.autonomyWakesRepository,
    },
    maintenance: {
      scheduler: deps.maintenanceScheduler,
      config: maintenanceConfigSnapshot,
      // Manual heavy maintenance always optimizes. The config flag gates only
      // automatic scheduler ticks; callers decide whether a manual run is dry.
      optimizeStorage: (options = {}) =>
        deps.maintenanceOrchestrator.runMechanicalMaintenance(() =>
          runStorageOptimization({
            optimizer: () => deps.lance.optimizeStorage(),
            ts: deps.clock.now(),
            runId: options.runId,
            tracer: deps.tracer,
          }),
        ),
      countPendingSemanticExtractionEpisodes: () => countPendingSemanticExtractionEpisodes(deps),
    },
    inbox: {
      catchUp: deps.chatResponseCatchUpWorker,
    },
    workmem: {
      load: (sessionId = DEFAULT_SESSION_ID) => deps.workingMemoryStore.load(sessionId),
      clear: (sessionId = DEFAULT_SESSION_ID) => {
        deps.turnOrchestrator.clearWorkingMemory(sessionId);
      },
      getPendingActionMergeCount: () => deps.workingMemoryStore.getPendingActionMergeCount(),
    },
    prompts: createPromptsFacade(deps),
    sessions: createSessionsFacade(deps),
  };
}

function createSessionsFacade(deps: BorgDependencies): BorgFacades["sessions"] {
  return {
    ensure: (...args) => deps.sessionsRepository.ensure(...args),
    touch: (...args) => deps.sessionsRepository.touch(...args),
    setParticipationPolicy: async (sessionId, policy, opts) => {
      const current = deps.sessionsRepository.get(sessionId);

      if (current === null) {
        throw new StorageError(`Session ${sessionId} not found`, {
          code: "SESSION_NOT_FOUND",
        });
      }

      if (current.audience_role === "operator" && policy !== "active") {
        throw new StorageError("Operator sessions are always active", {
          code: "SESSION_OPERATOR_POLICY_LOCKED",
        });
      }

      const previousPolicy = current.participation_policy;
      const updated = deps.sessionsRepository.setParticipationPolicy(sessionId, policy);

      if (updated === null) {
        throw new StorageError(`Session ${sessionId} not found`, {
          code: "SESSION_NOT_FOUND",
        });
      }

      const writer = deps.createStreamWriter(sessionId);
      try {
        await writer.append({
          kind: "internal_event",
          content: {
            event: "participation_policy.changed",
            session_id: sessionId,
            previous: previousPolicy,
            next: policy,
            reason: opts?.reason ?? null,
            operator: true,
          },
        });
      } catch (error) {
        if (isCommittedStreamIndexUpdateFailure(error)) {
          throw error;
        }

        try {
          deps.sessionsRepository.setParticipationPolicy(sessionId, previousPolicy);
        } catch (rollbackError) {
          throw participationPolicyRollbackFailedError(error, rollbackError);
        }

        throw error;
      } finally {
        writer.close();
      }

      return updated;
    },
    get: (...args) => deps.sessionsRepository.get(...args),
    list: (...args) => deps.sessionsRepository.list(...args),
  };
}

function createPromptsFacade(deps: BorgDependencies): BorgPromptsFacade {
  const repo = deps.promptOverrideRepository;

  function promptBlockOverrides(): Partial<Record<PromptKey, string>> | undefined {
    const records = repo.list();
    if (records.length === 0) {
      return undefined;
    }

    return Object.fromEntries(
      records.map((record) => [record.prompt_key, record.override_text]),
    ) as Partial<Record<PromptKey, string>>;
  }

  function view(key: PromptKey): BorgPromptBlockView {
    const spec = getPromptBlockSpec(key);
    const override = repo.get(key);
    const records = repo.list();
    const record = records.find((row) => row.prompt_key === key);
    const fallback = key === "host_capabilities" ? deps.config.host_capabilities : spec.default;
    const currentText = override ?? fallback;
    const currentTextKind =
      override !== null
        ? "stored_override"
        : currentText === spec.default
          ? "static_default"
          : "runtime_composed";
    return {
      key: spec.key,
      label: spec.label,
      description: spec.description,
      default_text: spec.default,
      current_text: currentText,
      current_text_kind: currentTextKind,
      overridden: override !== null,
      updated_at: record?.updated_at ?? null,
    };
  }

  return {
    list: () => PROMPT_BLOCKS.map((spec) => view(spec.key)),
    set: (key, text) => {
      repo.set(key, text);
      return view(key);
    },
    clear: (key) => {
      repo.clear(key);
      return view(key);
    },
    previewAssembledFraming: () => {
      const promptBlocks = promptBlockOverrides();
      const options = {
        retrievalContextBudget: 0,
        semanticContextBudget: 0,
        hostCapabilities: deps.config.host_capabilities,
        nowMs: deps.clock.now(),
        ...(promptBlocks === undefined ? {} : { promptBlocks }),
      };
      const parts = buildCacheableBaseSystemPromptParts(
        createAssembledFramingPreviewContext(options.nowMs),
        options,
      );
      const context = createAssembledFramingPreviewContext(options.nowMs);
      const sections = buildBaseSystemPromptSections(context, options);
      const renderContext = createBasePromptSurfaceRenderContext(context, sections);
      const renderedBlocks = promptSurfaceBlocksForSurface(PROMPT_SURFACES.cacheableStaticPrefix)
        .map((block) => {
          const content = block.render(renderContext);
          return content === null ? null : { block, content };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null);
      let offset = 0;
      const segments = renderedBlocks.map(({ block, content }, index) => {
        const start = offset;
        const end = start + content.length;
        offset = end + (index === renderedBlocks.length - 1 ? 0 : 2);
        return {
          id: block.id,
          label: block.id,
          editable_key: block.editableKey ?? null,
          start,
          end,
        };
      });

      return {
        text: parts.staticPrefix,
        sections: [...parts.staticPrefixSections],
        segments,
      };
    },
  };
}
