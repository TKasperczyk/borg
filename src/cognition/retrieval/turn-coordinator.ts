import { createNeutralAffectiveSignal, type MoodRepository } from "../../memory/affective/index.js";
import type {
  CommitmentRecord,
  CommitmentRepository,
  EntityRecord,
} from "../../memory/commitments/index.js";
import type { ExecutiveFocus } from "../../executive/index.js";
import type { ReviewQueueItem, ReviewQueueRepository } from "../../memory/semantic/index.js";
import type {
  ProceduralContext,
  SkillSelectionResult,
  SkillSelector,
} from "../../memory/procedural/index.js";
import type { SocialProfile } from "../../memory/social/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type {
  RetrievedContext,
  RetrievedEpisode,
  RetrievalPipeline,
  RetrievalSearchOptions,
} from "../../retrieval/index.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import { computeRetrievalLimit, computeWeights, type SuppressionSet } from "../attention/index.js";
import type { SelfSnapshot } from "../deliberation/deliberator.js";
import { deriveProceduralContext } from "../procedural/context-derivation.js";
import type { PerceptionResult } from "../types.js";

function selectActiveValues(values: readonly SelfSnapshot["values"][number][], candidateLimit = 2) {
  const established = values.filter((value) => value.state === "established");
  const candidates = values
    .filter((value) => value.state !== "established")
    .sort((left, right) => right.priority - left.priority || left.created_at - right.created_at)
    .slice(0, candidateLimit);

  return [...established, ...candidates];
}

function buildSkillSelectionQuery(userMessage: string, entities: readonly string[]): string {
  return [userMessage, ...entities]
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .join(" ");
}

function selectGoalDescriptions(
  goals: readonly SelfSnapshot["goals"][number][],
  executiveFocus: ExecutiveFocus | null | undefined,
): {
  goalDescriptions: string[];
  primaryGoalDescription: string | undefined;
} {
  const selectedGoal = executiveFocus?.selected_goal ?? null;

  if (selectedGoal === null) {
    return {
      goalDescriptions: goals.map((goal) => goal.description),
      primaryGoalDescription: undefined,
    };
  }

  return {
    goalDescriptions: [
      selectedGoal.description,
      ...goals.filter((goal) => goal.id !== selectedGoal.id).map((goal) => goal.description),
    ],
    primaryGoalDescription: selectedGoal.description,
  };
}

export type TurnRetrievalCoordinatorOptions = {
  commitmentRepository: Pick<CommitmentRepository, "getApplicable">;
  reviewQueueRepository: Pick<ReviewQueueRepository, "list">;
  moodRepository: Pick<MoodRepository, "current" | "history">;
  retrievalPipeline: Pick<RetrievalPipeline, "searchWithContext" | "search">;
  skillSelector: Pick<SkillSelector, "select">;
  clock: Clock;
};

export type TurnRetrievalCoordinatorInput = {
  sessionId: SessionId;
  turnId: string;
  userMessage: string;
  cognitionInput: string;
  inputAudience?: string;
  isSelfAudience: boolean;
  audienceEntityId: EntityId | null;
  audienceEntity: EntityRecord | null;
  audienceProfile: SocialProfile | null;
  perception: PerceptionResult;
  workingMemory: WorkingMemory;
  selfSnapshot: SelfSnapshot;
  executiveFocus?: ExecutiveFocus | null;
  suppressionSet: SuppressionSet;
  findEntityByName: (name: string) => EntityId | null;
};

export type TurnRetrievalCoordinatorResult = {
  applicableCommitments: CommitmentRecord[];
  pendingCorrections: ReviewQueueItem[];
  affectiveTrajectory: ReturnType<MoodRepository["history"]>;
  retrieval: RetrievedContext;
  retrievedEpisodes: RetrievedContext["episodes"];
  retrievedSemantic: RetrievedContext["semantic"];
  proceduralContext: ProceduralContext | null;
  selectedSkill: SkillSelectionResult | null;
  retrievalOptions: RetrievalSearchOptions;
  reRetrieve: (query: string, overrides?: RetrievalSearchOptions) => Promise<RetrievedEpisode[]>;
};

export class TurnRetrievalCoordinator {
  constructor(private readonly options: TurnRetrievalCoordinatorOptions) {}

  private collectApplicableCommitments(
    audienceEntityId: EntityId | null,
    perceivedEntities: readonly string[],
    findEntityByName: (name: string) => EntityId | null,
  ): CommitmentRecord[] {
    const aboutEntityIds: Array<EntityId | null> = [];
    const seenEntities = new Set<string>();

    for (const entity of perceivedEntities) {
      const normalized = entity.trim();

      if (normalized.length === 0) {
        continue;
      }

      const key = normalized.toLowerCase();

      if (seenEntities.has(key)) {
        continue;
      }

      seenEntities.add(key);
      const entityId = findEntityByName(normalized);

      if (entityId !== null) {
        aboutEntityIds.push(entityId);
      }
    }

    if (aboutEntityIds.length === 0) {
      aboutEntityIds.push(null);
    }

    const byId = new Map<string, CommitmentRecord>();

    for (const aboutEntityId of aboutEntityIds) {
      const applicable = this.options.commitmentRepository.getApplicable({
        audience: audienceEntityId,
        aboutEntity: aboutEntityId,
        nowMs: this.options.clock.now(),
      });

      for (const commitment of applicable) {
        byId.set(commitment.id, commitment);
      }
    }

    return [...byId.values()].sort(
      (left, right) => right.priority - left.priority || left.created_at - right.created_at,
    );
  }

  async coordinate(input: TurnRetrievalCoordinatorInput): Promise<TurnRetrievalCoordinatorResult> {
    const applicableCommitments = this.collectApplicableCommitments(
      input.audienceEntityId,
      input.perception.entities,
      input.findEntityByName,
    );
    const pendingCorrections = this.options.reviewQueueRepository
      .list({
        kind: "correction",
        openOnly: true,
      })
      .filter((item) => {
        const correctionAudience =
          typeof item.refs.audience_entity_id === "string" ? item.refs.audience_entity_id : null;

        if (input.audienceEntityId === null) {
          return correctionAudience === null;
        }

        return correctionAudience === null || correctionAudience === input.audienceEntityId;
      });
    const perceivedMood = input.workingMemory.mood ?? createNeutralAffectiveSignal();
    const perceivedMoodActive =
      Math.abs(perceivedMood.valence) + Math.abs(perceivedMood.arousal) > 0.3;
    const retrievalMood = perceivedMoodActive
      ? perceivedMood
      : this.options.moodRepository.current(input.sessionId);
    const affectiveTrajectory = this.options.moodRepository.history(input.sessionId, {
      limit: 5,
    });
    const activeValues = selectActiveValues(input.selfSnapshot.values);
    const goalSelection = selectGoalDescriptions(input.selfSnapshot.goals, input.executiveFocus);

    const attentionWeights = computeWeights(input.perception.mode, {
      currentGoals: input.selfSnapshot.goals,
      hasActiveValues: activeValues.length > 0,
      hasTemporalCue: input.perception.temporalCue !== null,
      moodActive: Math.abs(retrievalMood.valence) + Math.abs(retrievalMood.arousal) > 0.3,
      audienceTrust: input.audienceProfile?.trust ?? null,
    });
    const retrievalOptions: RetrievalSearchOptions = {
      limit: computeRetrievalLimit(input.perception.mode),
      audienceEntityId: input.audienceEntityId,
      attentionWeights,
      goalDescriptions: goalSelection.goalDescriptions,
      primaryGoalDescription: goalSelection.primaryGoalDescription,
      activeValues,
      temporalCue: input.perception.temporalCue,
      strictTimeRange: input.perception.temporalCue !== null,
      moodState: retrievalMood,
      audienceProfile: input.audienceProfile,
      audienceTerms: input.isSelfAudience
        ? []
        : input.audienceEntity === null
          ? input.inputAudience === undefined
            ? []
            : [input.inputAudience]
          : [
              input.audienceEntity.canonical_name,
              ...input.audienceEntity.aliases,
              ...(input.inputAudience === undefined ? [] : [input.inputAudience]),
            ],
      entityTerms: input.perception.entities,
      suppressionSet: input.suppressionSet,
      includeOpenQuestions: input.perception.mode === "reflective",
      traceTurnId: input.turnId,
    };
    const retrieval = await this.options.retrievalPipeline.searchWithContext(
      input.cognitionInput,
      retrievalOptions,
    );
    const retrievedEpisodes = retrieval.episodes;
    const retrievedSemantic = retrieval.semantic;
    const skillSelectionQuery = buildSkillSelectionQuery(
      input.userMessage,
      input.perception.entities,
    );
    const proceduralContext =
      input.perception.mode === "problem_solving"
        ? deriveProceduralContext({
            userMessage: input.userMessage,
            perception: input.perception,
            isSelfAudience: input.isSelfAudience,
            audienceEntityId: input.audienceEntityId,
            audienceProfile: input.audienceProfile,
            inputAudience: input.inputAudience,
          })
        : null;
    const selectedSkill =
      input.perception.mode === "problem_solving"
        ? await this.options.skillSelector.select(skillSelectionQuery, {
            k: 5,
            ...(proceduralContext === null ? {} : { proceduralContext }),
          })
        : null;

    return {
      applicableCommitments,
      pendingCorrections,
      affectiveTrajectory,
      retrieval,
      retrievedEpisodes,
      retrievedSemantic,
      proceduralContext,
      selectedSkill,
      retrievalOptions,
      reRetrieve: (query, overrides = {}) =>
        this.options.retrievalPipeline.search(query, {
          ...retrievalOptions,
          ...overrides,
        }),
    };
  }
}
