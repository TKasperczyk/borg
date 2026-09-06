// Registers Borg's built-in read and memory tools with a dispatcher.

import type { Clock } from "../util/clock.js";
import type { ScheduledWakesRepository } from "../autonomy/index.js";
import type { PromptSurfaceHistoryRepository } from "../cognition/prompts/prompt-surface-history.js";
import {
  combineMemoryDisclosureLabels,
  type SourceStreamAudienceDisclosureResolver,
  unknownMemoryDisclosureLabel,
} from "../memory/common/index.js";
import {
  goalMemoryDisclosureLabel,
  identityEventMemoryDisclosureLabel,
} from "../memory/common/disclosure-serializers.js";
import type { CommitmentRepository } from "../memory/commitments/index.js";
import type { EntityRepository } from "../memory/commitments/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import {
  parseIdentityEventDisclosureSources,
  type IdentityEvent,
  type IdentityService,
} from "../memory/identity/index.js";
import type { SkillRepository } from "../memory/procedural/index.js";
import type { GoalsRepository, OpenQuestionsRepository } from "../memory/self/index.js";
import type { TrainOfThoughtRepository } from "../memory/train-of-thought/index.js";
import {
  SELF_RECALL_SCOPE,
  mapWithDisclosureConcurrency,
  memoryDisclosureLabelForEpisodeIds,
  resolveMemoryDisclosureLabelForEpisodeIds,
  resolveMemoryDisclosureLabelsByEpisodeId,
  type MemoryDisclosureLabel,
  type RetrievalPipeline,
} from "../retrieval/index.js";
import type {
  SemanticEdge,
  SemanticGraph,
  SemanticNode,
  SemanticNodeRepository,
  SemanticWalkStep,
} from "../memory/semantic/index.js";
import { readStreamEntryAtOffset, type StreamEntryIndexRepository } from "../stream/index.js";
import {
  ToolDispatcher,
  createCommitmentsListTool,
  createEpisodicRecentTool,
  createEpisodicSearchTool,
  createGoalsRetireTool,
  createGoalsBlockTool,
  createGoalsUnblockTool,
  createIdentityEventsListForCognitionTool,
  createJournalAppendTool,
  createOpenQuestionsCreateTool,
  createOpenQuestionsResolveTool,
  createOpenQuestionsRuminationsTool,
  createOwnRecordsListTool,
  createPromptSurfaceChangesTool,
  createScheduledWakesCancelTool,
  createScheduledWakesCreateTool,
  createScheduledWakesListTool,
  createSemanticWalkTool,
  createSkillsListTool,
} from "../tools/index.js";
import type { BorgStreamWriterFactory } from "./types.js";

export type BuildToolDispatcherOptions = {
  dataDir: string;
  entryIndex: StreamEntryIndexRepository;
  sourceStreamAudienceDisclosureResolver: SourceStreamAudienceDisclosureResolver;
  retrievalPipeline: RetrievalPipeline;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository: SemanticNodeRepository;
  semanticGraph: SemanticGraph;
  commitmentRepository: CommitmentRepository;
  entityRepository: EntityRepository;
  goalsRepository: GoalsRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  identityService: IdentityService;
  skillRepository: SkillRepository;
  trainOfThoughtRepository: TrainOfThoughtRepository;
  scheduledWakesRepository: ScheduledWakesRepository;
  promptSurfaceHistoryRepository: PromptSurfaceHistoryRepository;
  createStreamWriter: BorgStreamWriterFactory;
  clock: Clock;
};

async function identityEventDisclosureLabelsForCollection(
  events: readonly IdentityEvent[],
  episodicRepository: EpisodicRepository,
  sourceStreamAudienceDisclosureResolver: SourceStreamAudienceDisclosureResolver,
): Promise<ReadonlyMap<IdentityEvent["id"], MemoryDisclosureLabel>> {
  const parsedSources = events.map((event) => parseIdentityEventDisclosureSources(event));
  const episodeIds = [...new Set(parsedSources.flatMap((sources) => sources.sourceEpisodeIds))];
  const commitmentLabels = sourceStreamAudienceDisclosureResolver.resolveLabels({
    commitments: parsedSources.flatMap((sources) => sources.commitmentAccesses),
  }).commitmentLabels;
  let commitmentLabelOffset = 0;
  const commitmentDisclosureLabelsByEvent = parsedSources.map((sources) => {
    const labels = commitmentLabels.slice(
      commitmentLabelOffset,
      commitmentLabelOffset + sources.commitmentAccesses.length,
    );
    commitmentLabelOffset += sources.commitmentAccesses.length;
    return labels;
  });
  const episodes = episodeIds.length === 0 ? [] : await episodicRepository.getMany(episodeIds);
  const episodesById = new Map(episodes.map((episode) => [episode.id, episode]));
  const cachedEpisodicRepository: Pick<EpisodicRepository, "getMany"> = {
    async getMany(ids) {
      return ids.map((id) => episodesById.get(id)).filter((episode) => episode !== undefined);
    },
  };
  const labels = await mapWithDisclosureConcurrency(
    events,
    async (event, index) =>
      [
        event.id,
        await identityEventMemoryDisclosureLabel(event, {
          episodicRepository: cachedEpisodicRepository,
          commitmentDisclosureLabels: commitmentDisclosureLabelsByEvent[index],
        }),
      ] as const,
  );

  return new Map(labels);
}

async function annotateSemanticWalkSteps(
  steps: readonly SemanticWalkStep[],
  episodicRepository: EpisodicRepository,
): Promise<
  Array<
    Omit<SemanticWalkStep, "node" | "edgePath"> & {
      node: SemanticNode & { disclosureLabel: MemoryDisclosureLabel };
      edgePath: Array<SemanticEdge & { disclosureLabel: MemoryDisclosureLabel }>;
    }
  >
> {
  const labelsByEpisodeId = await resolveMemoryDisclosureLabelsByEpisodeId(
    episodicRepository,
    steps.flatMap((step) => [
      ...step.node.source_episode_ids,
      ...step.edgePath.flatMap((edge) => edge.evidence_episode_ids),
    ]),
  );

  return mapWithDisclosureConcurrency(steps, async (step) => ({
    ...step,
    node: {
      ...step.node,
      disclosureLabel: memoryDisclosureLabelForEpisodeIds(
        step.node.source_episode_ids,
        labelsByEpisodeId,
      ),
    },
    edgePath: step.edgePath.map((edge) => ({
      ...edge,
      disclosureLabel: memoryDisclosureLabelForEpisodeIds(
        edge.evidence_episode_ids,
        labelsByEpisodeId,
      ),
    })),
  }));
}

export function buildToolDispatcher(options: BuildToolDispatcherOptions): ToolDispatcher {
  const toolDispatcher = new ToolDispatcher({
    createStreamWriter: options.createStreamWriter,
    clock: options.clock,
  });

  toolDispatcher
    .register(
      createEpisodicSearchTool({
        searchEpisodes: async (query, limit, context) => {
          const currentAudienceEntityId = context.audienceEntityId ?? null;
          const retrieved = await options.retrievalPipeline.recallEpisodesForCognition(query, {
            limit,
            recallContext: {
              reader: SELF_RECALL_SCOPE,
              currentSessionId: context.sessionId,
              currentAudienceEntityId,
              currentParticipantEntityIds:
                currentAudienceEntityId === null ? [] : [currentAudienceEntityId],
            },
            rankingAudienceEntityId: currentAudienceEntityId,
            sessionId: context.sessionId,
            traceTurnId: context.turnId,
          });

          return retrieved.episodes;
        },
      }),
    )
    .register(
      createEpisodicRecentTool({
        listRecentEpisodes: (limit) => options.episodicRepository.listRecentForCognition({ limit }),
      }),
    )
    .register(
      createOwnRecordsListTool({
        listThoughtRecords: (input) =>
          options.entryIndex.listActiveEntriesByKindRange({
            kinds: ["thought"],
            sinceTs: input.sinceMs,
            untilTs: input.untilMs,
            limit: input.limit,
            ...(input.sessionId === undefined ? {} : { sessionId: input.sessionId }),
            ...(input.cursor === undefined ? {} : { cursor: input.cursor }),
          }),
        readThoughtRecord: (record) =>
          readStreamEntryAtOffset({
            dataDir: options.dataDir,
            sessionId: record.session_id,
            byteOffset: record.byte_offset,
          }),
        listJournalRecords: (input) => options.trainOfThoughtRepository.listForRange(input),
        clock: options.clock,
      }),
    )
    .register(
      createSemanticWalkTool({
        walkGraph: async (fromId, walkOptions) => {
          const root = await options.semanticNodeRepository.get(fromId);

          if (root === null) {
            return [];
          }

          return annotateSemanticWalkSteps(
            await options.semanticGraph.walk(fromId, walkOptions),
            options.episodicRepository,
          );
        },
      }),
    )
    .register(
      createPromptSurfaceChangesTool({
        current: () => options.promptSurfaceHistoryRepository.current(),
        listChanges: (input) => options.promptSurfaceHistoryRepository.listChanges(input),
      }),
    )
    .register(
      createCommitmentsListTool({
        listCommitments: () =>
          options.commitmentRepository.list({
            activeOnly: true,
          }),
        disclosureLabelsForCommitments: (commitments) =>
          options.sourceStreamAudienceDisclosureResolver.resolveLabels({ commitments })
            .commitmentLabelsById,
      }),
    )
    .register(
      createOpenQuestionsCreateTool({
        createOpenQuestion: (input) => options.identityService.addOpenQuestion(input),
      }),
    )
    .register(
      createOpenQuestionsResolveTool({
        identityService: options.identityService,
        disclosureLabelForEvidence: async (episodeIds, streamEntryIds) =>
          combineMemoryDisclosureLabels([
            ...(episodeIds.length === 0
              ? []
              : [
                  await resolveMemoryDisclosureLabelForEpisodeIds(
                    options.episodicRepository,
                    episodeIds,
                  ),
                ]),
            ...streamEntryIds.map(() => unknownMemoryDisclosureLabel()),
          ]),
      }),
    )
    .register(
      createOpenQuestionsRuminationsTool({
        listRuminations: (listOptions) =>
          options.openQuestionsRepository.listRuminationsInRange(listOptions),
        getOpenQuestion: (id) => options.openQuestionsRepository.get(id),
      }),
    )
    .register(
      createGoalsRetireTool({
        goalsRepository: options.goalsRepository,
        disclosureLabelForGoal: (goal) =>
          options.sourceStreamAudienceDisclosureResolver
            .resolveLabels({ goals: [goal] })
            .goalLabelsById.get(goal.id) ?? goalMemoryDisclosureLabel(goal),
      }),
    )
    .register(
      createGoalsBlockTool({
        goalsRepository: options.goalsRepository,
        disclosureLabelForGoal: (goal) =>
          options.sourceStreamAudienceDisclosureResolver
            .resolveLabels({ goals: [goal] })
            .goalLabelsById.get(goal.id) ?? goalMemoryDisclosureLabel(goal),
      }),
    )
    .register(
      createGoalsUnblockTool({
        goalsRepository: options.goalsRepository,
        disclosureLabelForGoal: (goal) =>
          options.sourceStreamAudienceDisclosureResolver
            .resolveLabels({ goals: [goal] })
            .goalLabelsById.get(goal.id) ?? goalMemoryDisclosureLabel(goal),
      }),
    )
    .register(
      createJournalAppendTool({
        resolveSelfEntityId: () =>
          options.entityRepository.resolve("self", {
            kind: "self",
            provenance: "assistant_seeded",
          }),
        appendJournalEntry: (input) => options.trainOfThoughtRepository.append(input),
      }),
    )
    .register(
      createScheduledWakesCreateTool({
        scheduleWake: (input) => options.scheduledWakesRepository.schedule(input),
      }),
    )
    .register(
      createScheduledWakesListTool({
        listScheduledWakes: (input) => options.scheduledWakesRepository.list(input),
      }),
    )
    .register(
      createScheduledWakesCancelTool({
        cancelScheduledWake: (id) => options.scheduledWakesRepository.cancel(id),
      }),
    )
    .register(
      createIdentityEventsListForCognitionTool({
        listEvents: (listOptions) => options.identityService.listEvents(listOptions),
        disclosureLabelsForEvents: (events) =>
          identityEventDisclosureLabelsForCollection(
            events,
            options.episodicRepository,
            options.sourceStreamAudienceDisclosureResolver,
          ),
      }),
    )
    .register(
      createSkillsListTool({
        listSkills: (limit: number) => options.skillRepository.list(limit),
        listContextStatsForSkill: (skillId) =>
          options.skillRepository.listContextStatsForSkill(skillId),
      }),
    );

  return toolDispatcher;
}
