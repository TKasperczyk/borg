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
import type { ActionResult } from "../action/index.js";
import type { SuppressionSet } from "../attention/index.js";
import type { DeliberationResult, SelfSnapshot } from "../deliberation/deliberator.js";
import type { ExecutiveFocus } from "../../executive/index.js";
import type { PendingProceduralAttemptTracker } from "../procedural/pending-attempt-tracker.js";
import type { PerceptionResult } from "../types.js";
import type { ReflectionEffects, ReflectionResult, Reflector } from "./index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { isFrameAnomaly, type FrameAnomalyClassification } from "../frame-anomaly/index.js";

const ACTION_RESPONSE_SUMMARY_LIMIT = 240;
const OPEN_QUESTIONS_REFLECTION_LIMIT = 20;

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
  origin?: "user" | "autonomous";
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
  pendingSocialAttribution: PendingSocialAttribution | null;
  suppressionSet: SuppressionSet;
  persistedUserEntryId?: StreamEntryId;
  persistedPerceptionEntry: StreamEntry;
  persistedAgentEntry: StreamEntry;
  isUserTurn: boolean;
  frameAnomaly?: FrameAnomalyClassification | null;
  streamWriter: StreamWriter;
  onHookFailure: (hook: string, error: unknown) => Promise<void>;
  trackReflectionEffects: (effects: ReflectionEffects) => void;
};

export class TurnReflectionCoordinator {
  constructor(private readonly options: TurnReflectionCoordinatorOptions) {}

  async run(input: RunTurnReflectionInput): Promise<ReflectionResult> {
    let moodSnapshot = input.workingMood;

    if (input.origin !== "autonomous" && input.perception.affectiveSignalDegraded !== true) {
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
    if (input.audienceEntityId !== null) {
      try {
        interactionRecord = this.options.socialRepository.recordInteractionWithId(
          input.audienceEntityId,
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
        activeOpenQuestions,
        suppressionSet: input.suppressionSet,
        frameAnomaly: input.frameAnomaly ?? null,
        currentTurnStreamEntryIds:
          input.persistedUserEntryId === undefined
            ? [input.persistedPerceptionEntry.id, input.persistedAgentEntry.id]
            : isFrameAnomaly(input.frameAnomaly)
              ? [input.persistedAgentEntry.id]
              : [input.persistedUserEntryId, input.persistedAgentEntry.id],
      },
      input.streamWriter,
    );
    input.trackReflectionEffects(reflection.effects);

    const reflectedWorkingMemory = reflection.workingMemory;
    const nextPendingSocialAttribution =
      input.audienceEntityId !== null &&
      interactionRecord !== null &&
      input.pendingSocialAttribution === null
        ? {
            entity_id: input.audienceEntityId,
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
      persistedAgentEntryId: input.persistedAgentEntry.id,
      audienceEntityId: input.audienceEntityId,
    });

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("reflection_emitted", {
        turnId: input.turnId,
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
