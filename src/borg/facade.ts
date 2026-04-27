// Public Borg facade adapters over the internal repository and service graph.

import { EpisodicExtractor } from "../memory/episodic/index.js";
import { SemanticExtractor } from "../memory/semantic/index.js";
import type { MaintenancePlan, OfflineProcessName, OrchestratorResult } from "../offline/index.js";
import type { RetrievalSearchOptions } from "../retrieval/index.js";
import { StreamReader, StreamWriter } from "../stream/index.js";
import { DEFAULT_SESSION_ID, createSemanticNodeId, type EntityId } from "../util/ids.js";
import type { BorgFacades } from "./facade-types.js";
import type {
  BorgDependencies,
  BorgDreamOptions,
  BorgDreamRunner,
  BorgEpisodeSearchOptions,
} from "./types.js";

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

    return deps.entityRepository.resolve(options.audience);
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
  ): RetrievalSearchOptions => {
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
          : {
              semantic: 0.35,
              goal_relevance:
                options?.goalDescriptions !== undefined && options.goalDescriptions.length > 0
                  ? 0.1
                  : 0,
              value_alignment: 0,
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
      consolidator: deps.config.offline.consolidator.enabled,
      reflector: deps.config.offline.reflector.enabled,
      curator: deps.config.offline.curator.enabled,
      overseer: deps.config.offline.overseer.enabled,
      ruminator: deps.config.offline.ruminator.enabled,
      "self-narrator": deps.config.offline.selfNarrator.enabled,
      "procedural-synthesizer": deps.config.offline.proceduralSynthesizer.enabled,
    })
      .filter(([, enabled]) => enabled)
      .map(([name]) => name as OfflineProcessName);

  const runDream = async (
    processNames: readonly OfflineProcessName[],
    options: BorgDreamOptions = {},
  ): Promise<OrchestratorResult> => {
    const processes = processNames.map((name) => deps.offlineProcesses[name]);

    return deps.maintenanceOrchestrator.run({
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
      processes: processNames.map((name) => deps.offlineProcesses[name]),
      opts: {
        budget: options.budget,
        processOverrides: options.processOverrides,
      },
    });

  return {
    stream: {
      append: async (input, options = {}) => {
        const writer = new StreamWriter({
          dataDir: deps.config.dataDir,
          sessionId: options.session ?? DEFAULT_SESSION_ID,
          clock: deps.clock,
          entryIndex: deps.entryIndex,
        });

        try {
          return await writer.append(input);
        } finally {
          writer.close();
        }
      },
      tail: (n, options = {}) =>
        new StreamReader({
          dataDir: deps.config.dataDir,
          sessionId: options.session ?? DEFAULT_SESSION_ID,
        }).tail(n),
      reader: (options = {}) =>
        new StreamReader({
          dataDir: deps.config.dataDir,
          sessionId: options.session ?? DEFAULT_SESSION_ID,
        }),
    },
    episodic: {
      get: (id, options = {}) =>
        deps.retrievalPipeline.getEpisode(id, {
          audienceEntityId: resolveEpisodeAudienceEntityId(options),
          crossAudience: options.crossAudience,
        }),
      search: (query, options = {}) =>
        deps.retrievalPipeline.search(query, resolveEpisodeSearchOptions(options)),
      extract: async (options = {}) => {
        const extractor = new EpisodicExtractor({
          dataDir: deps.config.dataDir,
          episodicRepository: deps.episodicRepository,
          embeddingClient: deps.embeddingClient,
          llmClient: deps.llmFactory(),
          model: deps.config.anthropic.models.extraction,
          entityRepository: deps.entityRepository,
          clock: deps.clock,
        });

        return extractor.extractFromStream({
          session: options.session ?? DEFAULT_SESSION_ID,
          sinceTs: options.sinceTs,
          sinceCursor: options.sinceCursor,
          untilTs: options.untilTs,
        });
      },
      list: (...args) => deps.episodicRepository.list(...args),
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
        upsertPeriod: (...args) => deps.identityService.addPeriod(...args),
        closePeriod: (...args) => deps.identityService.closePeriod(...args),
        getPeriod: (...args) => deps.autobiographicalRepository.getPeriod(...args),
        getByLabel: (...args) => deps.autobiographicalRepository.getByLabel(...args),
      },
      growthMarkers: {
        list: (...args) => deps.growthMarkersRepository.list(...args),
        add: (...args) => deps.identityService.addGrowthMarker(...args),
        summarize: (...args) => deps.growthMarkersRepository.summarize(...args),
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
    social: {
      getProfile: (entity) =>
        deps.socialRepository.getProfile(deps.entityRepository.resolve(entity)),
      upsertProfile: (entity) =>
        deps.socialRepository.upsertProfile(deps.entityRepository.resolve(entity)),
      recordInteraction: (entity, interaction) =>
        deps.socialRepository.recordInteraction(deps.entityRepository.resolve(entity), interaction),
      adjustTrust: (entity, delta, provenance) =>
        deps.socialRepository.adjustTrust(deps.entityRepository.resolve(entity), delta, provenance),
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
        get: (id) => deps.semanticNodeRepository.get(id),
        list: (...args) => deps.semanticNodeRepository.list(...args),
        search: async (query, options = {}) => {
          const vector = await deps.embeddingClient.embed(query);
          return deps.semanticNodeRepository.searchByVector(vector, {
            limit: options.limit,
          });
        },
      },
      edges: {
        add: (input) => deps.semanticEdgeRepository.addEdge(input),
        list: (...args) => deps.semanticEdgeRepository.listEdges(...args),
      },
      walk: (fromId, ...args) => deps.semanticGraph.walk(fromId, ...args),
      extract: async (episodes) => {
        const extractor = new SemanticExtractor({
          nodeRepository: deps.semanticNodeRepository,
          edgeRepository: deps.semanticEdgeRepository,
          embeddingClient: deps.embeddingClient,
          episodicRepository: deps.episodicRepository,
          llmClient: deps.llmFactory(),
          model: deps.config.anthropic.models.extraction,
          semanticReviewService: deps.semanticReviewService,
          clock: deps.clock,
        });

        return extractor.extractFromEpisodes(episodes);
      },
    },
    commitments: {
      add: (input) =>
        deps.commitmentRepository.add({
          type: input.type,
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
      revoke: (...args) => deps.commitmentRepository.revoke(...args),
      list: (options = {}) =>
        deps.commitmentRepository.list({
          activeOnly: options.activeOnly,
          audience:
            options.audience === undefined || options.audience === null
              ? null
              : deps.entityRepository.resolve(options.audience),
          aboutEntity:
            options.aboutEntity === undefined || options.aboutEntity === null
              ? null
              : deps.entityRepository.resolve(options.aboutEntity),
        }),
    },
    identity: {
      updateValue: (...args) => deps.identityService.updateValue(...args),
      updateGoal: (...args) => deps.identityService.updateGoal(...args),
      updateTrait: (...args) => deps.identityService.updateTrait(...args),
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
      resolve: (id, decision) => deps.reviewQueueRepository.resolve(id, decision),
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
    },
    workmem: {
      load: (sessionId = DEFAULT_SESSION_ID) => deps.workingMemoryStore.load(sessionId),
      clear: (sessionId = DEFAULT_SESSION_ID) => {
        deps.turnOrchestrator.clearWorkingMemory(sessionId);
      },
    },
  };
}
