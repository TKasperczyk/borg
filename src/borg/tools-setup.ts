// Registers Borg's built-in read and memory tools with a dispatcher.

import type { Clock } from "../util/clock.js";
import type { ScheduledWakesRepository } from "../autonomy/index.js";
import type { CommitmentRepository } from "../memory/commitments/index.js";
import { legacyCommitmentSchema } from "../memory/commitments/index.js";
import {
  isEpisodeAccessVisible,
  type EpisodeAccessLike,
  type EpisodicRepository,
} from "../memory/episodic/index.js";
import type { IdentityEvent } from "../memory/identity/index.js";
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
  recordType: IdentityEvent["record_type"],
): boolean {
  if (value === null || value === undefined) {
    return true;
  }

  if (eventValueHasKey(value, "restricted_audience")) {
    const parsed = legacyCommitmentSchema.safeParse(value);

    return parsed.success
      ? visibleCommitmentAudience(parsed.data.restricted_audience, audienceEntityId)
      : false;
  }

  const episodeAccess = identityEventEpisodeAccess(value, {
    allowNestedEpisode: recordType === "episode",
  });

  if (episodeAccess !== undefined) {
    return episodeAccess === null ? false : isEpisodeAccessVisible(episodeAccess, audienceEntityId);
  }

  if (recordType === "episode") {
    return false;
  }

  return true;
}

function identityEventEpisodeAccess(
  value: unknown,
  options: {
    allowNestedEpisode: boolean;
  },
): EpisodeAccessLike | null | undefined {
  if (
    !eventValueHasKey(value, "audience_entity_id") &&
    !eventValueHasKey(value, "origin_audience_entity_ids")
  ) {
    if (options.allowNestedEpisode && eventValueHasKey(value, "episode")) {
      return (
        identityEventEpisodeAccess(value.episode, {
          allowNestedEpisode: false,
        }) ?? null
      );
    }

    return undefined;
  }

  const audienceEntityId =
    !eventValueHasKey(value, "audience_entity_id") ||
    value.audience_entity_id === null ||
    value.audience_entity_id === undefined
      ? null
      : typeof value.audience_entity_id === "string"
        ? (value.audience_entity_id as EntityId)
        : undefined;

  if (audienceEntityId === undefined) {
    return null;
  }

  const originAudienceEntityIds =
    !eventValueHasKey(value, "origin_audience_entity_ids") ||
    value.origin_audience_entity_ids === undefined
      ? undefined
      : Array.isArray(value.origin_audience_entity_ids) &&
          value.origin_audience_entity_ids.every((item) => typeof item === "string")
        ? (value.origin_audience_entity_ids as EntityId[])
        : null;

  if (originAudienceEntityIds === null) {
    return null;
  }

  const shared =
    !eventValueHasKey(value, "shared") || value.shared === undefined
      ? undefined
      : typeof value.shared === "boolean"
        ? value.shared
        : null;

  if (shared === null) {
    return null;
  }

  if (originAudienceEntityIds === undefined && audienceEntityId === null && shared === false) {
    return null;
  }

  return {
    audience_entity_id: audienceEntityId,
    ...(originAudienceEntityIds === undefined
      ? {}
      : { origin_audience_entity_ids: originAudienceEntityIds }),
    ...(shared === undefined ? {} : { shared }),
  };
}

function isIdentityEventVisible(
  event: IdentityEvent,
  audienceEntityId: EntityId | null | undefined,
): boolean {
  return (
    isIdentityEventValueVisible(event.old_value, audienceEntityId, event.record_type) &&
    isIdentityEventValueVisible(event.new_value, audienceEntityId, event.record_type)
  );
}

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
