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
  RetrievalPipeline,
  RetrievalSearchOptions,
} from "../../retrieval/index.js";
import {
  selectActiveScoringValues,
  type RetrievalScoringFeatures,
} from "../../retrieval/scoring-features.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import type { LLMClient } from "../../llm/index.js";
import { NOOP_TRACER, type TurnTracer } from "../tracing/tracer.js";
import { computeRetrievalLimit, computeWeights, type SuppressionSet } from "../attention/index.js";
import type { SelfSnapshot } from "../deliberation/deliberator.js";
import { deriveProceduralContext } from "../procedural/context-derivation.js";
import type { PerceptionResult } from "../types.js";

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
  retrievalPipeline: Pick<RetrievalPipeline, "searchWithContext">;
  skillSelector: Pick<SkillSelector, "select">;
  clock: Clock;
  tracer?: TurnTracer;
};

export type TurnRetrievalCoordinatorInput = {
  sessionId: SessionId;
  turnId: string;
  userMessage: string;
  recentMessages: readonly { role: "user" | "assistant"; content: string }[];
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
  activeValues?: readonly SelfSnapshot["values"][number][];
  scoringFeatures?: RetrievalScoringFeatures;
  suppressionSet: SuppressionSet;
  findEntityByName: (name: string) => EntityId | null;
  llmClient?: LLMClient;
  proceduralContextModel?: string;
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
  reRetrieve: (query: string, overrides?: RetrievalSearchOptions) => Promise<RetrievedContext>;
};

export class TurnRetrievalCoordinator {
  private readonly tracer: TurnTracer;

  constructor(private readonly options: TurnRetrievalCoordinatorOptions) {
    this.tracer = options.tracer ?? NOOP_TRACER;
  }

  private collectApplicableCommitments(audienceEntityId: EntityId | null): CommitmentRecord[] {
    return this.options.commitmentRepository
      .getApplicable({
        audience: audienceEntityId,
        nowMs: this.options.clock.now(),
      })
      .sort((left, right) => right.priority - left.priority || left.created_at - right.created_at);
  }

  async coordinate(input: TurnRetrievalCoordinatorInput): Promise<TurnRetrievalCoordinatorResult> {
    const applicableCommitments = this.collectApplicableCommitments(input.audienceEntityId);
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
    const activeValues = input.activeValues ?? selectActiveScoringValues(input.selfSnapshot.values);
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
      ...(input.scoringFeatures === undefined ? {} : { scoringFeatures: input.scoringFeatures }),
      temporalCue: input.perception.temporalCue,
      strictTimeRange: false,
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
      sessionId: input.sessionId,
      turnCounter: input.workingMemory.turn_counter,
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
        ? await deriveProceduralContext(
            {
              userMessage: input.userMessage,
              recentMessages: input.recentMessages,
              perception: input.perception,
              isSelfAudience: input.isSelfAudience,
              audienceEntityId: input.audienceEntityId,
              audienceProfile: input.audienceProfile,
              inputAudience: input.inputAudience,
            },
            {
              llmClient: input.llmClient,
              model: input.proceduralContextModel,
              onDegraded: (reason) => {
                if (this.tracer.enabled) {
                  this.tracer.emit("perception_classifier_degraded", {
                    turnId: input.turnId,
                    classifier: "procedural_context",
                    reason,
                  });
                }
              },
            },
          )
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
        this.options.retrievalPipeline.searchWithContext(query, {
          ...retrievalOptions,
          ...overrides,
        }),
    };
  }
}
