import type { Config } from "../../config/index.js";
import {
  createNeutralAffectiveSignal,
  type AffectiveSignal,
} from "../../memory/affective/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { LLMClient } from "../../llm/index.js";
import { StreamReader } from "../../stream/index.js";
import type { Clock } from "../../util/clock.js";
import { ConfigError } from "../../util/errors.js";
import type { SessionId } from "../../util/ids.js";
import { TurnContextCompiler, type RecencyWindow } from "../recency/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { detectAffectiveSignal } from "./affective-signal.js";
import { Perceiver } from "./index.js";
import type { PerceptionResult } from "../types.js";

export type PerceptionGatewayOptions = {
  config: Config;
  llmFactory: () => LLMClient;
  clock: Clock;
  tracer: TurnTracer;
  getAffectiveSignalDetector?: () => typeof detectAffectiveSignal | undefined;
  turnContextCompiler: Pick<TurnContextCompiler, "compile">;
  createStreamReader: (sessionId: SessionId) => StreamReader;
};

export type PerceptionGatewayInput = {
  sessionId: SessionId;
  isSelfAudience: boolean;
  origin?: "user" | "autonomous";
  cognitionInput: string;
  workingMemory: WorkingMemory;
};

export type PerceptionGatewayBeginInput = {
  turnId: string;
  onHookFailure: (hook: string, error: unknown, details?: Record<string, unknown>) => Promise<void>;
};

export type PerceptionGatewayResult = {
  perception: PerceptionResult;
  recencyWindow: RecencyWindow;
  workingMood: AffectiveSignal;
  workingMemory: WorkingMemory;
};

export type PreparedPerceptionGateway = {
  perceive(input: PerceptionGatewayInput): Promise<PerceptionGatewayResult>;
};

export class PerceptionGateway {
  constructor(private readonly options: PerceptionGatewayOptions) {}

  private getOptionalLlmClient(): LLMClient | undefined {
    try {
      return this.options.llmFactory();
    } catch (error) {
      if (error instanceof ConfigError) {
        return undefined;
      }

      throw error;
    }
  }

  beginTurn(input: PerceptionGatewayBeginInput): PreparedPerceptionGateway {
    const optionalPerceptionLlm =
      this.options.config.perception.useLlmFallback === true
        ? this.getOptionalLlmClient()
        : undefined;
    const perceiver = new Perceiver({
      llmClient: optionalPerceptionLlm,
      model: this.options.config.anthropic.models.background,
      // entity_extractor and temporal_cue judge from the current message
      // alone with bounded extraction rubrics. Run them on the fast slot
      // (Haiku) instead of background (Opus) to cut perception latency.
      fastModel: this.options.config.anthropic.models.recallExpansion,
      useLlmFallback: this.options.config.perception.useLlmFallback,
      modeWhenLlmAbsent: this.options.config.perception.modeWhenLlmAbsent,
      affectiveUseLlmFallback: this.options.config.affective.useLlmFallback,
      // Temporal cue uses the same LLM gate as mode detection: both rely
      // on the perception-bound LLM client. Turning off perception LLM
      // fallback turns off temporal extraction too (degrades to null).
      temporalCueUseLlmFallback: this.options.config.perception.useLlmFallback,
      detectAffectiveSignal: this.options.getAffectiveSignalDetector?.(),
      onAffectiveError: (error) => input.onHookFailure("affective_extraction", error),
      onClassifierFailure: ({ classifier, error }) =>
        input.onHookFailure("perception_classifier", error, { classifier }),
      clock: this.options.clock,
      tracer: this.options.tracer,
      turnId: input.turnId,
    });

    return {
      perceive: (perceptionInput) =>
        this.perceiveWithPreparedPerceiver(perceiver, input.turnId, perceptionInput),
    };
  }

  private async perceiveWithPreparedPerceiver(
    perceiver: Perceiver,
    turnId: string,
    input: PerceptionGatewayInput,
  ): Promise<PerceptionGatewayResult> {
    // Compile recent dialogue BEFORE appending the current user message,
    // so the window contains prior turns only. The compiler guarantees
    // the window starts with a user role and ends with an assistant
    // role, making it safe to concatenate with a trailing
    // {role:"user", content: currentUserMessage}.
    const recencyWindow: RecencyWindow = this.options.turnContextCompiler.compile(
      this.options.createStreamReader(input.sessionId),
      {
        includeSelfTurns: input.isSelfAudience,
      },
    );
    if (this.options.tracer.enabled) {
      this.options.tracer.emit("recency_compiled", {
        turnId,
        messageCount: recencyWindow.messages.length,
        sourceEntryIds: recencyWindow.messages.map((message) => message.stream_entry_id),
      });
    }
    const recentHistoryStrings = recencyWindow.messages.map(
      (message) => `${message.role}: ${message.content}`,
    );
    const perception = await perceiver.perceive(input.cognitionInput, recentHistoryStrings);
    const workingMood =
      input.origin === "autonomous" || perception.affectiveSignalDegraded === true
        ? (input.workingMemory.mood ?? createNeutralAffectiveSignal())
        : perception.affectiveSignal;
    const workingMemory = {
      ...input.workingMemory,
      turn_counter: input.workingMemory.turn_counter + 1,
      hot_entities: perception.entities,
      mood: workingMood,
      mode: perception.mode,
      updated_at: this.options.clock.now(),
    };

    return {
      perception,
      recencyWindow,
      workingMood,
      workingMemory,
    };
  }
}
