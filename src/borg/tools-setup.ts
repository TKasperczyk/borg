// Registers Borg's built-in read and memory tools with a dispatcher.

import type { Clock } from "../util/clock.js";
import type { ScheduledWakesRepository } from "../autonomy/index.js";
import {
  commitmentMemoryDisclosureLabel,
  identityEventMemoryDisclosureLabel,
} from "../cognition/disclosure-labels.js";
import type { CommitmentRepository } from "../memory/commitments/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import type { IdentityService } from "../memory/identity/index.js";
import type { SkillRepository } from "../memory/procedural/index.js";
import {
  resolveMemoryDisclosureLabelForEpisodeIds,
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
import {
  ToolDispatcher,
  createCommitmentsListTool,
  createEpisodicSearchTool,
  createIdentityEventsListForCognitionTool,
  createOpenQuestionsCreateTool,
  createScheduledWakesCancelTool,
  createScheduledWakesCreateTool,
  createScheduledWakesListTool,
  createSemanticWalkTool,
  createSkillsListTool,
} from "../tools/index.js";
import type { BorgStreamWriterFactory } from "./types.js";

export type BuildToolDispatcherOptions = {
  retrievalPipeline: RetrievalPipeline;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository: SemanticNodeRepository;
  semanticGraph: SemanticGraph;
  commitmentRepository: CommitmentRepository;
  identityService: IdentityService;
  skillRepository: SkillRepository;
  scheduledWakesRepository: ScheduledWakesRepository;
  createStreamWriter: BorgStreamWriterFactory;
  clock: Clock;
};

async function annotateSemanticWalkStep(
  step: SemanticWalkStep,
  episodicRepository: EpisodicRepository,
): Promise<
  Omit<SemanticWalkStep, "node" | "edgePath"> & {
    node: SemanticNode & { disclosureLabel: MemoryDisclosureLabel };
    edgePath: Array<SemanticEdge & { disclosureLabel: MemoryDisclosureLabel }>;
  }
> {
  return {
    ...step,
    node: {
      ...step.node,
      disclosureLabel: await resolveMemoryDisclosureLabelForEpisodeIds(
        episodicRepository,
        step.node.source_episode_ids,
      ),
    },
    edgePath: await Promise.all(
      step.edgePath.map(async (edge) => ({
        ...edge,
        disclosureLabel: await resolveMemoryDisclosureLabelForEpisodeIds(
          episodicRepository,
          edge.evidence_episode_ids,
        ),
      })),
    ),
  };
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
              reader: "sol",
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
      createSemanticWalkTool({
        walkGraph: async (fromId, walkOptions) => {
          const root = await options.semanticNodeRepository.get(fromId);

          if (root === null) {
            return [];
          }

          return Promise.all(
            (await options.semanticGraph.walk(fromId, walkOptions)).map((step) =>
              annotateSemanticWalkStep(step, options.episodicRepository),
            ),
          );
        },
      }),
    )
    .register(
      createCommitmentsListTool({
        listCommitments: () =>
          options.commitmentRepository.list({
            activeOnly: true,
          }),
        disclosureLabelForCommitment: (commitment) => commitmentMemoryDisclosureLabel(commitment),
      }),
    )
    .register(
      createOpenQuestionsCreateTool({
        createOpenQuestion: (input) => options.identityService.addOpenQuestion(input),
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
        disclosureLabelForEvent: (event) =>
          identityEventMemoryDisclosureLabel(event, {
            episodicRepository: options.episodicRepository,
          }),
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
