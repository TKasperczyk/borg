import type { LLMClient } from "../../llm/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import type { ActionRecord, ActionRepository, ActionState } from "../../memory/actions/index.js";
import type { SharedStateEntry } from "../../memory/decision-artifacts/index.js";
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
import { isFrameAnomaly, type FrameAnomalyClassification } from "../frame-anomaly/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { ActionStateExtractor } from "./action-state-extractor.js";

export type TurnActionStateServiceOptions = {
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
  recentHistory: ExtractCorrectivePreferenceInput["recentHistory"];
  audienceEntityId: EntityId | null;
  sessionId?: SessionId | null;
  speakerEntityId?: EntityId | null;
  speakerDisplayName?: string | null;
  goalId?: GoalId | null;
  openQuestionId?: OpenQuestionId | null;
  turnCounter?: number | null;
  frameAnomaly?: FrameAnomalyClassification | null;
};

export type CloseBorgSelfPerformedActionsInput = {
  llmClient: LLMClient;
  turnId: string;
  userMessage: string;
  persistedUserEntryId: StreamEntryId;
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

const ACTIVE_ACTION_STATES: readonly ActionState[] = [
  "considering",
  "committed_to_do",
  "scheduled",
  "unknown",
];

function uniqueActions(actions: readonly ActionRecord[]): ActionRecord[] {
  return [...new Map(actions.map((action) => [action.id, action])).values()];
}

export class TurnActionStateService {
  constructor(private readonly options: TurnActionStateServiceOptions) {}

  async extract(input: ExtractTurnActionStatesInput): Promise<ActionId[]> {
    if (!input.isUserTurn || input.persistedUserEntryId === undefined) {
      return [];
    }

    if (isFrameAnomaly(input.frameAnomaly)) {
      return [];
    }

    const actionStateExtractor = new ActionStateExtractor({
      llmClient: input.llmClient,
      model: this.options.model,
      actionRepository: this.options.actionRepository,
      embeddingClient: this.options.embeddingClient,
      clock: this.options.clock,
      tracer: this.options.tracer,
      turnId: input.turnId,
      sessionId: input.sessionId ?? undefined,
      onDegraded: (reason, error) => {
        if (!this.options.tracer.enabled) {
          return;
        }

        this.options.tracer.emit("extraction.actions.degraded", {
          turnId: input.turnId,
          reason,
          ...(this.options.tracer.includePayloads && error !== undefined
            ? { error: error instanceof Error ? error.message : String(error) }
            : {}),
        });
      },
    });
    const activeActionsForReference = uniqueActions([
      ...this.options.actionRepository.list({
        states: ACTIVE_ACTION_STATES,
        audienceEntityId: null,
        limit: 40,
      }),
      ...(input.audienceEntityId === null
        ? []
        : this.options.actionRepository.list({
            states: ACTIVE_ACTION_STATES,
            audienceEntityId: input.audienceEntityId,
            limit: 40,
          })),
      ...(input.speakerEntityId === null || input.speakerEntityId === undefined
        ? []
        : this.options.actionRepository.list({
            states: ACTIVE_ACTION_STATES,
            actor: input.speakerEntityId,
            limit: 40,
          })),
    ]).slice(0, 80);
    const actionStateRecords = await actionStateExtractor.extract({
      userMessage: input.userMessage,
      currentUserStreamEntryId: input.persistedUserEntryId,
      recentHistory: input.recentHistory,
      audienceEntityId: input.audienceEntityId,
      sessionId: input.sessionId ?? null,
      speakerEntityId: input.speakerEntityId ?? null,
      speakerDisplayName: input.speakerDisplayName ?? null,
      goalId: input.goalId ?? null,
      openQuestionId: input.openQuestionId ?? null,
      turnCounter: input.turnCounter ?? null,
      activeActionsForReference,
    });

    return actionStateRecords.map((record) => record.id);
  }

  async closeBorgSelfPerformedActions(input: CloseBorgSelfPerformedActionsInput): Promise<void> {
    const activeBorgActions = this.options.actionRepository.list({
      states: ACTIVE_ACTION_STATES,
      actor: "borg",
      limit: 80,
    });

    if (activeBorgActions.length === 0) {
      return;
    }

    const currentTurnBorgActions = activeBorgActions.filter((action) =>
      action.provenance_stream_entry_ids.includes(input.persistedUserEntryId),
    );

    if (currentTurnBorgActions.length === 0 && input.currentTurnSharedStateEntries.length === 0) {
      return;
    }

    const actionStateExtractor = new ActionStateExtractor({
      llmClient: input.llmClient,
      model: this.options.model,
      actionRepository: this.options.actionRepository,
      embeddingClient: this.options.embeddingClient,
      clock: this.options.clock,
      tracer: this.options.tracer,
      turnId: input.turnId,
      sessionId: input.sessionId ?? undefined,
      onDegraded: (reason, error) => {
        if (!this.options.tracer.enabled) {
          return;
        }

        this.options.tracer.emit("extraction.actions.degraded", {
          turnId: input.turnId,
          reason,
          ...(this.options.tracer.includePayloads && error !== undefined
            ? { error: error instanceof Error ? error.message : String(error) }
            : {}),
        });
      },
    });

    await actionStateExtractor.extract({
      userMessage: input.userMessage,
      currentUserStreamEntryId: input.persistedUserEntryId,
      currentAgentStreamEntryId: input.persistedAgentEntryId,
      recentHistory: input.recentHistory,
      audienceEntityId: input.audienceEntityId,
      sessionId: input.sessionId ?? null,
      speakerEntityId: input.speakerEntityId ?? null,
      speakerDisplayName: input.speakerDisplayName ?? null,
      turnCounter: input.turnCounter ?? null,
      activeActionsForReference: currentTurnBorgActions,
      postTurnSelfPerformance: {
        activeBorgActions: currentTurnBorgActions,
        currentTurnSharedStateEntries: input.currentTurnSharedStateEntries,
        agentResponse: input.agentResponse,
      },
      persistNewActions: false,
    });
  }
}
