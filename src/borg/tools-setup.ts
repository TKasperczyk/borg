// Registers Borg's built-in read and memory tools with a dispatcher.

import type { Clock } from "../util/clock.js";
import type { ScheduledWakesRepository } from "../autonomy/index.js";
import type { CommitmentRepository } from "../memory/commitments/index.js";
import { legacyCommitmentSchema } from "../memory/commitments/index.js";
import { isEpisodeVisibleToAudience, type EpisodicRepository } from "../memory/episodic/index.js";
import type { IdentityEvent } from "../memory/identity/index.js";
import type { IdentityService } from "../memory/identity/index.js";
import type { SkillRepository } from "../memory/procedural/index.js";
import type { SemanticGraph, SemanticNodeRepository } from "../memory/semantic/index.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import {
  filterSemanticWalkStepsByAudience,
  isSemanticNodeVisibleToAudience,
} from "../retrieval/semantic-retrieval.js";
import type { EntityId } from "../util/ids.js";
import {
  ToolDispatcher,
  createCommitmentsListTool,
  createEpisodicSearchTool,
  createIdentityEventsListTool,
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

function visibleCommitmentAudience(
  restrictedAudience: EntityId | null,
  audienceEntityId: EntityId | null | undefined,
): boolean {
  if (restrictedAudience === null) {
    return true;
  }

  return (
    audienceEntityId !== null &&
    audienceEntityId !== undefined &&
    restrictedAudience === audienceEntityId
  );
}

function eventValueHasKey(value: unknown, key: string): value is Record<string, unknown> {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.prototype.hasOwnProperty.call(value, key)
  );
}

function isIdentityEventValueVisible(
  value: unknown,
  audienceEntityId: EntityId | null | undefined,
): boolean {
  if (eventValueHasKey(value, "restricted_audience")) {
    const parsed = legacyCommitmentSchema.safeParse(value);

    return parsed.success
      ? visibleCommitmentAudience(parsed.data.restricted_audience, audienceEntityId)
      : false;
  }

  if (eventValueHasKey(value, "audience_entity_id")) {
    return isEpisodeVisibleToAudience(
      {
        audience_entity_id:
          typeof value.audience_entity_id === "string"
            ? (value.audience_entity_id as EntityId)
            : null,
        shared: typeof value.shared === "boolean" ? value.shared : undefined,
      },
      audienceEntityId,
    );
  }

  return true;
}

function isIdentityEventVisible(
  event: IdentityEvent,
  audienceEntityId: EntityId | null | undefined,
): boolean {
  return (
    isIdentityEventValueVisible(event.old_value, audienceEntityId) &&
    isIdentityEventValueVisible(event.new_value, audienceEntityId)
  );
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
        walkGraph: async (fromId, walkOptions, context) => {
          const root = await options.semanticNodeRepository.get(fromId);
          const visibility = {
            audienceEntityId: context?.audienceEntityId,
          };

          if (
            root === null ||
            !(await isSemanticNodeVisibleToAudience(root, visibility, {
              episodicRepository: options.episodicRepository,
            }))
          ) {
            return [];
          }

          return filterSemanticWalkStepsByAudience(
            await options.semanticGraph.walk(fromId, walkOptions),
            visibility,
            {
              episodicRepository: options.episodicRepository,
            },
          );
        },
      }),
    )
    .register(
      createCommitmentsListTool({
        listCommitments: (context) =>
          options.commitmentRepository.list({
            activeOnly: true,
            audience: context.audienceEntityId ?? null,
          }),
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
      createIdentityEventsListTool({
        listEvents: (listOptions, context) =>
          options.identityService
            .listEvents(listOptions)
            .filter((event) => isIdentityEventVisible(event, context.audienceEntityId ?? null)),
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
