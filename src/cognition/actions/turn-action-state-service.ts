import type { LLMClient } from "../../llm/index.js";
import type { ActionRepository } from "../../memory/actions/index.js";
import type { Clock } from "../../util/clock.js";
import type { ActionId, EntityId, StreamEntryId } from "../../util/ids.js";
import type { ExtractCorrectivePreferenceInput } from "../commitments/corrective-preference-extractor.js";
import { isFrameAnomaly, type FrameAnomalyClassification } from "../frame-anomaly/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { ActionStateExtractor } from "./action-state-extractor.js";

export type TurnActionStateServiceOptions = {
  model: string;
  actionRepository: ActionRepository;
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
  frameAnomaly?: FrameAnomalyClassification | null;
};

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
      clock: this.options.clock,
      tracer: this.options.tracer,
      turnId: input.turnId,
      onDegraded: (reason, error) => {
        if (!this.options.tracer.enabled) {
          return;
        }

        this.options.tracer.emit("action_state_extractor_degraded", {
          turnId: input.turnId,
          reason,
          ...(this.options.tracer.includePayloads && error !== undefined
            ? { error: error instanceof Error ? error.message : String(error) }
            : {}),
        });
      },
    });
    const actionStateRecords = await actionStateExtractor.extract({
      userMessage: input.userMessage,
      currentUserStreamEntryId: input.persistedUserEntryId,
      recentHistory: input.recentHistory,
      audienceEntityId: input.audienceEntityId,
    });

    return actionStateRecords.map((record) => record.id);
  }
}
