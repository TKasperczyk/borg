import type { LLMClient } from "../../llm/index.js";
import type { AffectiveSignal, MoodRepository } from "../../memory/affective/index.js";
import type { ProceduralContext, SkillSelectionResult } from "../../memory/procedural/index.js";
import type { OpenQuestionsRepository } from "../../memory/self/index.js";
import type { SocialRepository } from "../../memory/social/index.js";
import type {
  PendingSocialAttribution,
  WorkingMemory,
  WorkingMemoryStore,
} from "../../memory/working/index.js";
import type { StreamEntry, StreamWriter } from "../../stream/index.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, SessionId, StreamEntryId } from "../../util/ids.js";
import type { ActionResult } from "../turn-action/index.js";
import type { SuppressionSet } from "../attention/index.js";
import type { DeliberationResult, SelfSnapshot } from "../deliberation/deliberator.js";
import type { ExecutiveFocus } from "../../executive/index.js";
import type { PendingProceduralAttemptTracker } from "../procedural/pending-attempt-tracker.js";
import { runsReflectionPersistence, type PerceptionResult, type TurnOrigin } from "../types.js";
import type { ReflectionEffects, ReflectionResult, Reflector } from "./index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { ActualFrameAnomalyClassification } from "../frame-anomaly/index.js";

const ACTION_RESPONSE_SUMMARY_LIMIT = 240;
const OPEN_QUESTIONS_REFLECTION_LIMIT = 20;

function emptyReflectionEffects(): ReflectionEffects {
  return {
    createdActionIds: [],
    createdExecutiveStepIds: [],
    createdOpenQuestionIds: [],
    updatedExecutiveSteps: [],
    updatedGoals: [],
    resolvedOpenQuestions: [],
    updatedEpisodeStats: [],
  };
}

export type TurnReflectionCoordinatorOptions = {
  moodRepository: Pick<MoodRepository, "update">;
  socialRepository: Pick<SocialRepository, "recordInteractionWithId">;
  openQuestionsRepository: Pick<OpenQuestionsRepository, "list">;
  workingMemoryStore: Pick<WorkingMemoryStore, "save">;
  pendingProceduralAttemptTracker: Pick<PendingProceduralAttemptTracker, "update">;
  createReflector: (llmClient: LLMClient) => Reflector;
  clock: Clock;
  tracer: TurnTracer;
};

export type RunTurnReflectionInput = {
  llmClient: LLMClient;
  sessionId: SessionId;
  turnId: string;
  actionLifecycleTurnCounter?: number | null;
  origin?: TurnOrigin;
  userMessage: string;
  perception: PerceptionResult;
  workingMood: AffectiveSignal;
  postActionWorkingMemory: WorkingMemory;
  selfSnapshot: SelfSnapshot;
  deliberation: DeliberationResult;
  actionResult: ActionResult;
  retrievedEpisodes: DeliberationResult["retrievedEpisodes"];
  retrievalConfidence: Parameters<Reflector["reflect"]>[0]["retrievalConfidence"];
  executiveFocus: ExecutiveFocus;
  selectedSkill: SkillSelectionResult | null;
  proceduralContext: ProceduralContext | null;
  audienceEntityId: EntityId | null;
  audienceIsGroup?: boolean;
  senderEntityId?: EntityId | null;
  socialInteractionEntityId: EntityId | null;
  pendingSocialAttribution: PendingSocialAttribution | null;
  suppressionSet: SuppressionSet;
  persistedUserEntryId?: StreamEntryId;
  sourceUserEntryIds?: readonly StreamEntryId[];
  persistedPerceptionEntry?: StreamEntry;
  persistedAgentEntry: StreamEntry;
  isUserTurn: boolean;
  frameAnomaly?: ActualFrameAnomalyClassification | null;
  streamWriter: StreamWriter;
  onHookFailure: (hook: string, error: unknown) => Promise<void>;
  trackReflectionEffects: (effects: ReflectionEffects) => void;
};

export class TurnReflectionCoordinator {
  constructor(private readonly options: TurnReflectionCoordinatorOptions) {}

  async run(input: RunTurnReflectionInput): Promise<ReflectionResult> {
    if (!runsReflectionPersistence(input.origin)) {
      const effects = emptyReflectionEffects();
      input.trackReflectionEffects(effects);

      return {
        workingMemory: input.postActionWorkingMemory,
        effects,
      };
    }

    let moodSnapshot = input.workingMood;

    if (input.isUserTurn && input.perception.affectiveSignalDegraded !== true) {
      try {
        const nextMood = this.options.moodRepository.update(input.sessionId, {
          valence: input.perception.affectiveSignal.valence,
          arousal: input.perception.affectiveSignal.arousal,
          reason: input.userMessage.slice(0, 120),
          provenance: {
            kind: "system",
          },
        });
        moodSnapshot = {
          valence: nextMood.valence,
          arousal: nextMood.arousal,
          dominant_emotion: input.perception.affectiveSignal.dominant_emotion,
        };
      } catch (error) {
        await input.onHookFailure("mood_update", error);
      }
    }

    let interactionRecord: ReturnType<SocialRepository["recordInteractionWithId"]> | null = null;
    if (input.socialInteractionEntityId !== null) {
      try {
        // The lifecycle passes the current speaker for group channels, so
        // participant sentiment does not collapse onto the abstract group.
        interactionRecord = this.options.socialRepository.recordInteractionWithId(
          input.socialInteractionEntityId,
          {
            now: this.options.clock.now(),
            provenance: {
              kind: "system",
            },
          },
        );
      } catch (error) {
        await input.onHookFailure("social_update", error);
      }
    }

    const reflector = this.options.createReflector(input.llmClient);
    const activeOpenQuestions = this.options.openQuestionsRepository.list({
      status: "open",
      visibleToAudienceEntityId: input.audienceEntityId,
      limit: OPEN_QUESTIONS_REFLECTION_LIMIT,
    });
    const reflection = await reflector.reflect(
      {
        turnId: input.turnId,
        sessionId: input.sessionId,
        actionLifecycleTurnCounter: input.actionLifecycleTurnCounter ?? null,
        origin: input.origin ?? "user",
        userMessage: input.userMessage,
        perception: input.perception,
        workingMemory: {
          ...input.postActionWorkingMemory,
          mood: moodSnapshot,
        },
        selfSnapshot: input.selfSnapshot,
        deliberationResult: input.deliberation,
        actionResult: {
          ...input.actionResult,
          workingMemory: input.postActionWorkingMemory,
        },
        retrievedEpisodes: input.retrievedEpisodes,
        retrievalConfidence: input.retrievalConfidence,
        executiveFocus: input.executiveFocus,
        selectedSkillId: input.selectedSkill?.skill.id ?? null,
        audienceEntityId: input.audienceEntityId,
        audienceIsGroup: input.audienceIsGroup ?? false,
        senderEntityId: input.senderEntityId ?? null,
        activeOpenQuestions,
        suppressionSet: input.suppressionSet,
        frameAnomaly: input.frameAnomaly ?? null,
        currentTurnStreamEntryIds: [
          ...(input.sourceUserEntryIds ??
            (input.persistedUserEntryId === undefined ? [] : [input.persistedUserEntryId])),
          ...(input.persistedUserEntryId === undefined &&
          (input.sourceUserEntryIds === undefined || input.sourceUserEntryIds.length === 0)
            ? [input.persistedPerceptionEntry?.id]
            : []),
          input.persistedAgentEntry.id,
        ].filter((entryId): entryId is StreamEntryId => entryId !== undefined),
      },
      input.streamWriter,
    );
    input.trackReflectionEffects(reflection.effects);

    const reflectedWorkingMemory = reflection.workingMemory;
    const nextPendingSocialAttribution =
      input.socialInteractionEntityId !== null &&
      interactionRecord !== null &&
      input.pendingSocialAttribution === null
        ? {
            entity_id: input.socialInteractionEntityId,
            interaction_id: interactionRecord.interaction_id,
            agent_response_summary:
              input.actionResult.response.trim().length === 0
                ? null
                : input.actionResult.response
                    .replace(/\s+/g, " ")
                    .trim()
                    .slice(0, ACTION_RESPONSE_SUMMARY_LIMIT),
            turn_completed_ts: input.persistedAgentEntry.timestamp,
          }
        : input.pendingSocialAttribution;
    const nextPendingProceduralAttempts = this.options.pendingProceduralAttemptTracker.update({
      isUserTurn: input.isUserTurn,
      userMessage: input.userMessage,
      perception: input.perception,
      actionResult: {
        ...input.actionResult,
        workingMemory: input.postActionWorkingMemory,
      },
      selectedSkill: input.selectedSkill,
      proceduralContext: input.proceduralContext,
      reflectedWorkingMemory,
      persistedUserEntryId: input.persistedUserEntryId,
      sourceUserEntryIds: input.sourceUserEntryIds,
      persistedAgentEntryId: input.persistedAgentEntry.id,
      audienceEntityId: input.audienceEntityId,
    });

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("reflection.completed", {
        turnId: input.turnId,
        ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
        attributions: {
          pending_social: nextPendingSocialAttribution !== null,
          pending_trait: reflectedWorkingMemory.pending_trait_attribution !== null,
          pending_procedural: nextPendingProceduralAttempts.length > 0,
          pending_actions: reflectedWorkingMemory.pending_actions.length,
        },
      });
    }

    this.options.workingMemoryStore.save({
      ...reflectedWorkingMemory,
      mood: moodSnapshot,
      pending_social_attribution: nextPendingSocialAttribution,
      pending_procedural_attempts: nextPendingProceduralAttempts,
      suppressed: input.suppressionSet.snapshot(),
      updated_at: this.options.clock.now(),
    });

    return reflection;
  }
}
