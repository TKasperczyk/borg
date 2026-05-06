import type { LLMClient } from "../../llm/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { RetrievedEpisode } from "../../retrieval/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import type { AutonomyTriggerContext } from "../autonomy-trigger.js";
import type { CommitmentGuardRunner } from "../commitments/guard-runner.js";
import type { DeliberationResult } from "../deliberation/deliberator.js";
import type { PendingTurnEmission } from "../generation/types.js";
import type { TurnRelationalGuardRunner } from "../generation/turn-relational-guard.js";
import { toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";
import type { PerceptionResult } from "../types.js";
import {
  LLMPendingActionJudge,
  performAction,
  type ActionResult,
  type PendingActionRejection,
} from "./index.js";

export type TurnActionCoordinatorOptions = {
  commitmentGuardRunner: Pick<CommitmentGuardRunner, "run">;
  relationalGuardRunner: Pick<TurnRelationalGuardRunner, "run">;
  embeddingClient: EmbeddingClient;
  pendingActionJudgeModel: string;
  clock: Clock;
  tracer: TurnTracer;
};

export type RunTurnActionInput = {
  llmClient: LLMClient;
  turnId: string;
  sessionId: SessionId;
  deliberation: DeliberationResult;
  workingMemory: WorkingMemory;
  userMessage: string;
  cognitionInput: string;
  origin?: "user" | "autonomous";
  autonomyTrigger?: AutonomyTriggerContext | null;
  applicableCommitments: readonly CommitmentRecord[];
  perceptionEntities: PerceptionResult["entities"];
  persistedUserEntry?: Parameters<TurnRelationalGuardRunner["run"]>[0]["persistedUserEntry"];
  retrievedEpisodes: readonly RetrievedEpisode[];
  audienceEntityId: EntityId | null;
};

export type TurnActionCoordinatorResult = {
  actionResult: ActionResult;
  actionEmission: PendingTurnEmission;
};

export class TurnActionCoordinator {
  constructor(private readonly options: TurnActionCoordinatorOptions) {}

  async run(input: RunTurnActionInput): Promise<TurnActionCoordinatorResult> {
    const deliberationEmission: PendingTurnEmission =
      input.deliberation.emissionRecommendation === "no_output"
        ? {
            kind: "suppressed",
            reason: "s2_planner_no_output",
          }
        : (input.deliberation.emission ?? {
            kind: "message",
            content: input.deliberation.response,
          });
    const pendingActionJudge = new LLMPendingActionJudge({
      llmClient: input.llmClient,
      model: this.options.pendingActionJudgeModel,
    });
    const onPendingActionRejected = (event: PendingActionRejection) => {
      if (!this.options.tracer.enabled) {
        return;
      }

      this.options.tracer.emit("working_memory_degraded", {
        turnId: input.turnId,
        subsystem: "pending_actions",
        reason: event.reason,
        confidence: event.confidence,
        degraded: event.degraded,
        ...(this.options.tracer.includePayloads
          ? {
              record: toTraceJsonValue(event.record),
            }
          : {}),
      });
    };
    const actionResult =
      deliberationEmission.kind === "suppressed"
        ? await performAction({
            response: "",
            emission: deliberationEmission,
            toolCalls: input.deliberation.tool_calls,
            intents: [],
            workingMemory: input.workingMemory,
          })
        : await this.performGuardedAction({
            ...input,
            deliberationEmission,
            pendingActionJudge,
            onPendingActionRejected,
          });
    const actionEmission: PendingTurnEmission = actionResult.emission ?? {
      kind: "message",
      content: actionResult.response,
    };

    return {
      actionResult,
      actionEmission,
    };
  }

  private async performGuardedAction(
    input: RunTurnActionInput & {
      deliberationEmission: Extract<PendingTurnEmission, { kind: "message" }>;
      pendingActionJudge: LLMPendingActionJudge;
      onPendingActionRejected: (event: PendingActionRejection) => void;
    },
  ): Promise<ActionResult> {
    const commitmentCheck = await this.options.commitmentGuardRunner.run({
      llmClient: input.llmClient,
      turnId: input.turnId,
      response: input.deliberation.response,
      userMessage: input.userMessage,
      cognitionInput: input.cognitionInput,
      origin: input.origin,
      autonomyTrigger: input.autonomyTrigger,
      commitments: input.applicableCommitments,
      relevantEntities: input.perceptionEntities,
    });
    const commitmentEmission = commitmentCheck.emission;
    const guardedEmission =
      commitmentEmission.kind === "suppressed"
        ? commitmentEmission
        : await this.options.relationalGuardRunner.run({
            llmClient: input.llmClient,
            turnId: input.turnId,
            response: commitmentEmission.content,
            userMessage: input.userMessage,
            sessionId: input.sessionId,
            persistedUserEntry: input.persistedUserEntry,
            retrievedEpisodes: input.retrievedEpisodes,
            activeCommitments: input.applicableCommitments,
            closureLoop: input.workingMemory.discourse_state?.closure_loop ?? null,
            audienceEntityId: input.audienceEntityId,
          });

    return performAction({
      response: guardedEmission.kind === "message" ? guardedEmission.content : "",
      emission: guardedEmission,
      toolCalls: input.deliberation.tool_calls,
      intents: input.deliberation.intents,
      workingMemory: input.workingMemory,
      pendingActionJudge: input.pendingActionJudge,
      pendingActionEmbeddingClient: this.options.embeddingClient,
      pendingActionTimestamp: this.options.clock.now(),
      onPendingActionRejected: input.onPendingActionRejected,
    });
  }
}
