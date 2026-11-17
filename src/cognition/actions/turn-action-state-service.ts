import type { SimilarityConfigSource } from "../../config/similarity.js";
import type { LLMClient } from "../../llm/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import { ACTIVE_ACTION_STATES, type ActionRepository } from "../../memory/actions/index.js";
import type { SharedStateEntry } from "../../memory/shared-state/index.js";
import type { Clock } from "../../util/clock.js";
import type {
  ActionId,
  EntityId,
  GoalId,
  OpenQuestionId,
  SessionId,
  StreamEntryId,
} from "../../util/ids.js";
import type { ExtractCorrectivePreferenceInput } from "../commitments/corrective-preference-extractor.js";
import { retrieveActionCandidates } from "./action-candidates.js";
import type { GoalRecord } from "../../memory/self/index.js";
import type { ActualFrameAnomalyClassification } from "../frame-anomaly/index.js";
import type { FrameAnomalyDisposition } from "../lifecycle/turn-phase/perception-phase.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import type { CurrentTurnUserInputSenderAttribution } from "../turn-input.js";
import { ActionStateExtractor } from "./action-state-extractor.js";

export type TurnActionStateServiceOptions = {
  similarityConfig?: SimilarityConfigSource;
  model: string;
  actionRepository: ActionRepository;
  embeddingClient: EmbeddingClient;
  clock: Clock;
  tracer: TurnTracer;
};

export type ExtractTurnActionStatesInput = {
  llmClient: LLMClient;
  turnId: string;
  isUserTurn: boolean;
  userMessage: string;
  persistedUserEntryId?: StreamEntryId;
  sourceUserEntryIds?: readonly StreamEntryId[];
  senderAttribution?: readonly CurrentTurnUserInputSenderAttribution[];
  recentHistory: ExtractCorrectivePreferenceInput["recentHistory"];
  audienceEntityId: EntityId | null;
  sessionId?: SessionId | null;
  speakerEntityId?: EntityId | null;
  speakerDisplayName?: string | null;
  senderDisplayNameById?: (entityId: EntityId) => string | null | undefined;
  /** @deprecated Ignored; supply goal records for the model instead. */
  goalId?: GoalId | null;
  suppliedGoals?: readonly GoalRecord[];
  participantEntityIds?: readonly EntityId[];
  openQuestionId?: OpenQuestionId | null;
  turnCounter?: number | null;
  frameAnomaly?: ActualFrameAnomalyClassification | null;
  frameAnomalyDisposition?: FrameAnomalyDisposition;
};

export type CloseBorgSelfPerformedActionsInput = {
  llmClient: LLMClient;
  turnId: string;
  userMessage: string;
  persistedUserEntryId: StreamEntryId;
  sourceUserEntryIds?: readonly StreamEntryId[];
  persistedAgentEntryId: StreamEntryId;
  agentResponse: string;
  recentHistory: ExtractCorrectivePreferenceInput["recentHistory"];
  audienceEntityId: EntityId | null;
  sessionId?: SessionId | null;
  speakerEntityId?: EntityId | null;
  speakerDisplayName?: string | null;
  currentTurnSharedStateEntries: readonly SharedStateEntry[];
  turnCounter?: number | null;
};

export class TurnActionStateService {
  constructor(private readonly options: TurnActionStateServiceOptions) {}

  async extract(input: ExtractTurnActionStatesInput): Promise<ActionId[]> {
    const sessionId = input.sessionId ?? undefined;

    const sourceUserEntryIds =
      input.sourceUserEntryIds === undefined || input.sourceUserEntryIds.length === 0
        ? input.persistedUserEntryId === undefined
          ? []
          : [input.persistedUserEntryId]
        : [...input.sourceUserEntryIds];
    const currentUserStreamEntryId = input.persistedUserEntryId ?? sourceUserEntryIds[0];

    if (!input.isUserTurn || currentUserStreamEntryId === undefined) {
      return [];
    }

    if (
      input.frameAnomaly !== null &&
      input.frameAnomaly !== undefined &&
      input.frameAnomalyDisposition !== "annotate"
    ) {
      return [];
    }

    const actionStateExtractor = new ActionStateExtractor({
      similarityConfig: this.options.similarityConfig,
      llmClient: input.llmClient,
      model: this.options.model,
      actionRepository: this.options.actionRepository,
      embeddingClient: this.options.embeddingClient,
      clock: this.options.clock,
      tracer: this.options.tracer,
      turnId: input.turnId,
      sessionId,
      onDegraded: (reason, error) => {
        if (!this.options.tracer.enabled) {
          return;
        }

        this.options.tracer.emit("extraction.actions.degraded", {
          turnId: input.turnId,
          ...(sessionId !== undefined ? { session_id: sessionId } : {}),
          reason,
          ...(this.options.tracer.includePayloads && error !== undefined
            ? { error: error instanceof Error ? error.message : String(error) }
            : {}),
        });
      },
    });
    // An empty store needs neither an embedding nor an asynchronous recall barrier
    // before this extractor starts alongside the other early extractors.
    const candidates =
      this.options.actionRepository.count() === 0
        ? { records: [], total: 0 }
        : await retrieveActionCandidates({
            repository: this.options.actionRepository,
            text: input.userMessage,
            audienceEntityId: input.audienceEntityId,
            speakerEntityId: input.speakerEntityId,
            participantEntityIds: input.participantEntityIds,
            onSearchFailure: (error) =>
              this.options.tracer.emit("extraction.actions.candidate_search_failed", {
                turnId: input.turnId,
                error: error instanceof Error ? error.message : String(error),
              }),
          });
    const actionStateRecords = await actionStateExtractor.extract({
      userMessage: input.userMessage,
      currentUserStreamEntryId,
      currentUserStreamEntryIds: sourceUserEntryIds,
      recentHistory: input.recentHistory,
      audienceEntityId: input.audienceEntityId,
      sessionId: input.sessionId ?? null,
      speakerEntityId: input.speakerEntityId ?? null,
      speakerDisplayName: input.speakerDisplayName ?? null,
      senderDisplayNameById: input.senderDisplayNameById,
      senderAttribution: input.senderAttribution,
      suppliedGoals: input.suppliedGoals,
      openQuestionId: input.openQuestionId ?? null,
      turnCounter: input.turnCounter ?? null,
      activeActionsForReference: candidates.records,
      actionCandidateTotal: candidates.total,
    });

    return actionStateRecords.map((record) => record.id);
  }

  async closeBorgSelfPerformedActions(
    input: CloseBorgSelfPerformedActionsInput,
  ): Promise<ActionId[]> {
    const sessionId = input.sessionId ?? undefined;

    const activeBorgActions = this.options.actionRepository.list({
      states: ACTIVE_ACTION_STATES,
      actor: "borg",
      limit: 80,
    });

    if (activeBorgActions.length === 0) {
      return [];
    }

    const sourceUserEntryIds =
      input.sourceUserEntryIds === undefined || input.sourceUserEntryIds.length === 0
        ? [input.persistedUserEntryId]
        : [...input.sourceUserEntryIds];
    const currentTurnBorgActions = activeBorgActions.filter((action) =>
      (action.origin_stream_entry_ids ?? action.provenance_stream_entry_ids).some((entryId) =>
        sourceUserEntryIds.includes(entryId),
      ),
    );

    if (currentTurnBorgActions.length === 0 && input.currentTurnSharedStateEntries.length === 0) {
      return [];
    }

    const actionStateExtractor = new ActionStateExtractor({
      similarityConfig: this.options.similarityConfig,
      llmClient: input.llmClient,
      model: this.options.model,
      actionRepository: this.options.actionRepository,
      embeddingClient: this.options.embeddingClient,
      clock: this.options.clock,
      tracer: this.options.tracer,
      turnId: input.turnId,
      sessionId,
      onDegraded: (reason, error) => {
        if (!this.options.tracer.enabled) {
          return;
        }

        this.options.tracer.emit("extraction.actions.degraded", {
          turnId: input.turnId,
          ...(sessionId !== undefined ? { session_id: sessionId } : {}),
          reason,
          ...(this.options.tracer.includePayloads && error !== undefined
            ? { error: error instanceof Error ? error.message : String(error) }
            : {}),
        });
      },
    });

    const inserted = await actionStateExtractor.extract({
      userMessage: input.userMessage,
      currentUserStreamEntryId: input.persistedUserEntryId,
      currentUserStreamEntryIds: sourceUserEntryIds,
      currentAgentStreamEntryId: input.persistedAgentEntryId,
      recentHistory: input.recentHistory,
      audienceEntityId: input.audienceEntityId,
      sessionId: input.sessionId ?? null,
      speakerEntityId: input.speakerEntityId ?? null,
      speakerDisplayName: input.speakerDisplayName ?? null,
      turnCounter: input.turnCounter ?? null,
      activeActionsForReference: currentTurnBorgActions,
      actionCandidateTotal: currentTurnBorgActions.length,
      postTurnSelfPerformance: {
        activeBorgActions: currentTurnBorgActions,
        currentTurnSharedStateEntries: input.currentTurnSharedStateEntries,
        agentResponse: input.agentResponse,
      },
    });
    return inserted.map((record) => record.id);
  }
}
