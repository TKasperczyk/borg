import {
  MOOD_ACTIVITY_THRESHOLD,
  createNeutralAffectiveSignal,
  type MoodRepository,
} from "../../memory/affective/index.js";
import type {
  CommitmentRecord,
  CommitmentRepository,
  EntityRepository,
  EntityRecord,
} from "../../memory/commitments/index.js";
import type { ExecutiveFocus } from "../../executive/index.js";
import type {
  OpenCommitmentReconciliationStatus,
  ReviewQueueItem,
  ReviewQueueRepository,
} from "../../memory/review-queue/index.js";
import type {
  ProceduralContext,
  SkillSelectionResult,
  SkillSelector,
} from "../../memory/procedural/index.js";
import type { SocialProfile } from "../../memory/social/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type {
  CognitionRecallContext,
  DisclosureContext,
  CognitionRetrievalOptions,
  RetrievedContext,
  RetrievalPipeline,
} from "../../retrieval/index.js";
import {
  selectActiveScoringValues,
  type RetrievalScoringFeatures,
} from "../../retrieval/scoring-features.js";
import type { Clock } from "../../util/clock.js";
import type { AttachmentId, EntityId } from "../../util/ids.js";
import type { LLMClient } from "../../llm/index.js";
import { NOOP_TRACER, type TurnTracer } from "../../tracing/tracer.js";
import { computeRetrievalLimit, computeWeights, type SuppressionSet } from "../attention/index.js";
import type { SelfSnapshot } from "../deliberation/deliberator.js";
import { deriveProceduralContext } from "../procedural/context-derivation.js";
import type { PerceptionResult } from "../types.js";
import { correctionMemoryDisclosureLabel } from "../../memory/common/disclosure-serializers.js";
import type { SourceStreamAudienceDisclosureResolver } from "../../memory/common/index.js";
import type { StreamConversation } from "../../stream/index.js";
import type { LivedExperienceDaySummaryRepository } from "../../memory/activity/lived-experience-day-summary.js";

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

function retrievalOptionsFromRecallDisclosureContext(input: {
  recallContext: CognitionRecallContext;
  disclosureContext: DisclosureContext;
}): Pick<
  CognitionRetrievalOptions,
  "recallContext" | "disclosureContext" | "rankingAudienceEntityId" | "sessionId"
> {
  return {
    recallContext: input.recallContext,
    disclosureContext: input.disclosureContext,
    rankingAudienceEntityId: input.disclosureContext.currentAudienceEntityId,
    sessionId: input.recallContext.currentSessionId,
  };
}

function coordinatorContextFromRecallDisclosureContext(input: {
  recallContext: CognitionRecallContext;
  disclosureContext: DisclosureContext;
}): {
  sessionId: CognitionRecallContext["currentSessionId"];
  audienceEntityId: EntityId | null;
} {
  return {
    sessionId: input.recallContext.currentSessionId,
    audienceEntityId: input.disclosureContext.currentAudienceEntityId,
  };
}

export type TurnRetrievalCoordinatorOptions = {
  commitmentRepository: Pick<CommitmentRepository, "getApplicable" | "list">;
  sourceStreamAudienceDisclosureResolver?: SourceStreamAudienceDisclosureResolver;
  entityRepository: Pick<EntityRepository, "getSelf">;
  reviewQueueRepository: Pick<ReviewQueueRepository, "list"> &
    Partial<Pick<ReviewQueueRepository, "listOpenCommitmentReconciliationsForCognition">>;
  moodRepository: Pick<MoodRepository, "current" | "history">;
  retrievalPipeline: Pick<RetrievalPipeline, "recallEpisodesForCognition">;
  skillSelector: Pick<SkillSelector, "select">;
  // Optional: the owner's closed-day summaries reach the recall planner as OWNER_LIVED_EXPERIENCE.
  livedExperienceDaySummaryRepository?: Pick<LivedExperienceDaySummaryRepository, "listForWindow">;
  clock: Clock;
  tracer?: TurnTracer;
};

const PLANNER_LIVED_EXPERIENCE_WINDOW_MS = 7 * 24 * 60 * 60_000;
const PLANNER_LIVED_EXPERIENCE_LIMIT = 7;

export type TurnRetrievalCoordinatorInput = {
  turnId: string;
  userMessage: string;
  recentMessages: readonly { role: "user" | "assistant"; content: string }[];
  cognitionInput: string;
  inputAudience?: string;
  isSelfAudience: boolean;
  recallContext: CognitionRecallContext;
  disclosureContext: DisclosureContext;
  audienceEntity: EntityRecord | null;
  currentSenderName?: string;
  currentVenue?: StreamConversation;
  audienceProfile: SocialProfile | null;
  perception: PerceptionResult;
  workingMemory: WorkingMemory;
  selfSnapshot: SelfSnapshot;
  executiveFocus?: ExecutiveFocus | null;
  activeValues?: readonly SelfSnapshot["values"][number][];
  scoringFeatures?: RetrievalScoringFeatures;
  suppressionSet: SuppressionSet;
  currentTurnAttachmentIds?: readonly AttachmentId[];
  llmClient?: LLMClient;
  proceduralContextModel?: string;
};

export type TurnRetrievalCoordinatorResult = {
  applicableCommitments: CommitmentRecord[];
  applicableCommitmentsReadAtMs?: number;
  actionApplicableCommitments: CommitmentRecord[];
  pendingCorrections: ReviewQueueItem[];
  pendingCommitmentReviews: OpenCommitmentReconciliationStatus[];
  affectiveTrajectory: ReturnType<MoodRepository["history"]>;
  retrieval: RetrievedContext;
  retrievedEpisodes: RetrievedContext["episodes"];
  retrievedSemantic: RetrievedContext["semantic"];
  proceduralContext: ProceduralContext | null;
  selectedSkill: SkillSelectionResult | null;
  retrievalOptions: CognitionRetrievalOptions;
  reRetrieve: (
    query: string,
    overrides?: Partial<CognitionRetrievalOptions>,
  ) => Promise<RetrievedContext>;
};

export class TurnRetrievalCoordinator {
  private readonly tracer: TurnTracer;

  constructor(private readonly options: TurnRetrievalCoordinatorOptions) {
    this.tracer = options.tracer ?? NOOP_TRACER;
  }

  private sortCommitmentsForCognition(
    commitments: readonly CommitmentRecord[],
  ): CommitmentRecord[] {
    return [...commitments].sort(
      (left, right) => right.priority - left.priority || left.created_at - right.created_at,
    );
  }

  private recallActiveCommitmentsForCognition(nowMs: number): CommitmentRecord[] {
    return this.sortCommitmentsForCognition(
      this.options.commitmentRepository.list({
        activeOnly: true,
        nowMs,
      }),
    );
  }

  private collectApplicableCommitmentsForActionAuthorization(
    audienceEntityId: EntityId | null,
    nowMs: number,
  ): CommitmentRecord[] {
    return this.options.commitmentRepository
      .getApplicable({
        audience: audienceEntityId,
        nowMs,
      })
      .sort((left, right) => right.priority - left.priority || left.created_at - right.created_at);
  }

  async coordinate(input: TurnRetrievalCoordinatorInput): Promise<TurnRetrievalCoordinatorResult> {
    const coordinatorContext = coordinatorContextFromRecallDisclosureContext(input);
    const nowMs = this.options.clock.now();
    const rawApplicableCommitments = this.recallActiveCommitmentsForCognition(nowMs);
    const rawActionApplicableCommitments = this.collectApplicableCommitmentsForActionAuthorization(
      coordinatorContext.audienceEntityId,
      nowMs,
    );
    const commitmentsById = new Map(
      [...rawApplicableCommitments, ...rawActionApplicableCommitments].map((commitment) => [
        commitment.id,
        commitment,
      ]),
    );
    const resolvedCommitments = this.options.sourceStreamAudienceDisclosureResolver?.resolve({
      commitments: [...commitmentsById.values()],
    }).commitments ?? [...commitmentsById.values()];
    const resolvedCommitmentsById = new Map(
      resolvedCommitments.map((commitment) => [commitment.id, commitment]),
    );
    const applicableCommitments = rawApplicableCommitments.map(
      (commitment) => resolvedCommitmentsById.get(commitment.id) ?? commitment,
    );
    const actionApplicableCommitments = rawActionApplicableCommitments.map(
      (commitment) => resolvedCommitmentsById.get(commitment.id) ?? commitment,
    );
    const pendingCorrections = this.options.reviewQueueRepository
      .list({
        kind: "correction",
        openOnly: true,
      })
      .map((item) => ({
        ...item,
        disclosureLabel: correctionMemoryDisclosureLabel(item.refs),
      }));
    const pendingCommitmentReviews =
      this.options.reviewQueueRepository.listOpenCommitmentReconciliationsForCognition?.({
        subkinds: ["cross_scope_conflict", "cross_scope_redundancy"],
      }) ?? [];
    // Three quantities pass through the `mood` slot; only the first is ever
    // rendered. `perceivedMood` is this turn's classifier reading of the
    // INBOUND text (see perception/gateway.ts) -- the value the working-memory
    // line prints as `mood=V/A`. It is the operative value only while its
    // magnitude clears MOOD_ACTIVITY_THRESHOLD. Below that line retrieval
    // silently substitutes `moodRepository.current()`, which is (a) the EMA
    // blend written by reflection, not a reading, and (b) half-life decayed by
    // elapsed wall-clock time at read (mood.ts `decayFactor`, 24h default). So
    // a long enough silence shrinks the fallback toward neutral and switches
    // `moodActive` off by the clock alone -- no event, no mood_history row,
    // nothing on the rendered line to mark it. Reading `mood=` as "the value
    // steering retrieval" is exact only above the threshold.
    //
    // The two stale-mood regimes age in opposite directions, and which one is
    // in force is invisible. A carried-forward `perceivedMood` (autonomous-like
    // origin, or a degraded classifier -- perception/gateway.ts) is copied
    // verbatim through working memory and never decays, so if its original
    // magnitude cleared the threshold it keeps the mood term on indefinitely,
    // ranking on an arbitrarily old reading while the decayed fallback below it
    // has long since gone inactive and unused.
    //
    // What the term does when on: a ranking weight (ACTIVE_MOOD_WEIGHT 0.2 in
    // an un-normalised linear fusion, scoring.ts), never a filter, so recall
    // stays global. It scores each episode by distance from that episode's
    // `emotional_arc`; when the episodic extractor emits none, the fallback
    // `buildEmotionalArc` synthesises one from the affective signals of the
    // episode's USER entries -- the same sender-side readings. On an inbound
    // turn both sides of the comparison are therefore estimates of how the
    // other party sounded, and the term ranks past episodes by how closely the
    // sender's affect then matches the sender's affect now.
    const perceivedMood = input.workingMemory.mood ?? createNeutralAffectiveSignal();
    const perceivedMoodActive =
      Math.abs(perceivedMood.valence) + Math.abs(perceivedMood.arousal) > MOOD_ACTIVITY_THRESHOLD;
    const retrievalMood = perceivedMoodActive
      ? perceivedMood
      : this.options.moodRepository.current(coordinatorContext.sessionId);
    const affectiveTrajectory = this.options.moodRepository.history(coordinatorContext.sessionId, {
      limit: 5,
    });
    const activeValues = input.activeValues ?? selectActiveScoringValues(input.selfSnapshot.values);
    const goalSelection = selectGoalDescriptions(input.selfSnapshot.goals, input.executiveFocus);
    const memoryOwner = this.options.entityRepository.getSelf();
    const nowMsForPlanner = this.options.clock.now();
    const ownerLivedExperience =
      this.options.livedExperienceDaySummaryRepository === undefined
        ? undefined
        : this.options.livedExperienceDaySummaryRepository
            .listForWindow({
              fromMs: nowMsForPlanner - PLANNER_LIVED_EXPERIENCE_WINDOW_MS,
              toMs: nowMsForPlanner,
              limit: PLANNER_LIVED_EXPERIENCE_LIMIT,
              ...(memoryOwner === null ? {} : { selfEntityId: memoryOwner.id }),
            })
            .sort((a, b) => b.day_start_ms - a.day_start_ms)
            .map((summary) => ({
              day: summary.utc_day,
              gist: summary.gist,
              salience: summary.salience,
              disclosure: {
                class: summary.disclosure_label.disclosureClass,
                origin_audience_entity_ids: [...summary.disclosure_label.originAudienceEntityIds],
                private_to_entity_ids: [...summary.disclosure_label.privateToEntityIds],
                public_to_entity_ids: [...summary.disclosure_label.publicToEntityIds],
              },
            }));
    const attentionWeights = computeWeights(input.perception.mode, {
      currentGoals: input.selfSnapshot.goals,
      hasActiveValues: activeValues.length > 0,
      hasTemporalCue: input.perception.temporalCue !== null,
      moodActive:
        Math.abs(retrievalMood.valence) + Math.abs(retrievalMood.arousal) > MOOD_ACTIVITY_THRESHOLD,
      audienceTrust: input.audienceProfile?.trust ?? null,
    });
    const retrievalOptions: CognitionRetrievalOptions = {
      ...retrievalOptionsFromRecallDisclosureContext({
        recallContext: input.recallContext,
        disclosureContext: input.disclosureContext,
      }),
      limit: computeRetrievalLimit(input.perception.mode),
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
      recallQueryPlannerContext: {
        contextTurns: input.recentMessages,
        ...(ownerLivedExperience === undefined ? {} : { ownerLivedExperience }),
        identity: {
          ...(memoryOwner === null ? {} : { memoryOwnerName: memoryOwner.canonical_name }),
          ...(input.currentSenderName === undefined
            ? {}
            : { currentSenderName: input.currentSenderName }),
          ...(input.audienceEntity === null
            ? {}
            : { currentAudienceName: input.audienceEntity.canonical_name }),
          ...(input.currentVenue === undefined ? {} : { currentVenue: input.currentVenue }),
          entityTerms: input.perception.entities,
        },
      },
      ...(input.currentTurnAttachmentIds === undefined ||
      input.currentTurnAttachmentIds.length === 0
        ? {}
        : { currentTurnAttachmentIds: input.currentTurnAttachmentIds }),
      suppressionSet: input.suppressionSet,
      includeOpenQuestions: input.perception.mode === "reflective",
      turnCounter: input.workingMemory.turn_counter,
      traceTurnId: input.turnId,
    };
    const retrieval = await this.options.retrievalPipeline.recallEpisodesForCognition(
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
              audienceEntityId: coordinatorContext.audienceEntityId,
              audienceProfile: input.audienceProfile,
              inputAudience: input.inputAudience,
            },
            {
              llmClient: input.llmClient,
              model: input.proceduralContextModel,
              onDegraded: (reason) => {
                if (this.tracer.enabled) {
                  this.tracer.emit("perception.classifier.degraded", {
                    turnId: input.turnId,
                    session_id: coordinatorContext.sessionId,
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
      applicableCommitmentsReadAtMs: nowMs,
      actionApplicableCommitments,
      pendingCorrections,
      pendingCommitmentReviews,
      affectiveTrajectory,
      retrieval,
      retrievedEpisodes,
      retrievedSemantic,
      proceduralContext,
      selectedSkill,
      retrievalOptions,
      reRetrieve: (query, overrides = {}) =>
        this.options.retrievalPipeline.recallEpisodesForCognition(query, {
          ...retrievalOptions,
          ...overrides,
        }),
    };
  }
}
