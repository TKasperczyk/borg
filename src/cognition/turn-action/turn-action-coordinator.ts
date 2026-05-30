import type { LLMClient } from "../../llm/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { RetrievedEpisode } from "../../retrieval/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import type { AutonomyTriggerContext } from "../autonomy-trigger.js";
import type { CommitmentGuardResult, CommitmentGuardRunner } from "../commitments/guard-runner.js";
import type { DeliberationResult } from "../deliberation/deliberator.js";
import type { ClosureLoopDialogueAct } from "../generation/closure-loop.js";
import type { PendingTurnEmission } from "../generation/types.js";
import type { TurnPostGenerationGuardRunner } from "../generation/turn-post-generation-guard.js";
import { traceTurnPhase } from "../lifecycle/turn-phase/phase-trace.js";
import { toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";
import type { PerceptionResult, TurnOrigin } from "../types.js";
import {
  LLMPendingActionJudge,
  performAction,
  type ActionResult,
  type PendingActionRejection,
} from "./index.js";

export type TurnActionCoordinatorOptions = {
  commitmentGuardRunner: Pick<CommitmentGuardRunner, "run">;
  postGenerationGuardRunner: Pick<TurnPostGenerationGuardRunner, "run">;
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
  origin?: TurnOrigin;
  autonomyTrigger?: AutonomyTriggerContext | null;
  applicableCommitments: readonly CommitmentRecord[];
  perceptionEntities: PerceptionResult["entities"];
  persistedUserEntry?: Parameters<TurnPostGenerationGuardRunner["run"]>[0]["persistedUserEntry"];
  persistedUserEntries?: Parameters<
    TurnPostGenerationGuardRunner["run"]
  >[0]["persistedUserEntries"];
  retrievedEpisodes: readonly RetrievedEpisode[];
  currentUserClosureKind?: ClosureLoopDialogueAct | null;
  audienceEntityId: EntityId | null;
  knownInternalIdentifiers?: readonly string[];
};

export type TurnActionCoordinatorResult = {
  actionResult: ActionResult;
  actionEmission: PendingTurnEmission;
  deliberation: DeliberationResult;
};

function withMessageMetadata<T extends PendingTurnEmission>(
  emission: T,
  source: Extract<PendingTurnEmission, { kind: "message" }>,
): T {
  if (emission.kind !== "message") {
    return emission;
  }

  return {
    ...emission,
    ...(source.reply_target === undefined ? {} : { reply_target: source.reply_target }),
    ...(source.persistence_class === undefined
      ? {}
      : { persistence_class: source.persistence_class }),
    ...(source.discourse_control === undefined
      ? {}
      : { discourse_control: source.discourse_control }),
  } as T;
}

function pendingEmissionFromDeliberation(deliberation: DeliberationResult): PendingTurnEmission {
  return deliberation.emissionRecommendation === "no_output"
    ? {
        kind: "suppressed",
        reason: "s2_planner_no_output",
      }
    : (deliberation.emission ?? {
        kind: "message",
        content: deliberation.response,
      });
}

function suppressUnsupportedRegeneration(result: CommitmentGuardResult): CommitmentGuardResult {
  return {
    ...result,
    emission: {
      kind: "suppressed",
      reason: "commitment_violation",
    },
  };
}

type RegenerationRequest = Extract<
  CommitmentGuardResult["emission"],
  { kind: "requires_regeneration" }
>["regeneration"];

function commitmentKindsForRegeneration(
  commitments: readonly CommitmentRecord[],
  commitmentIds: readonly string[],
): CommitmentRecord["kind"][] {
  const requestedIds = new Set(commitmentIds);

  return [
    ...new Set(
      commitments
        .filter((commitment) => requestedIds.has(commitment.id))
        .map((commitment) => commitment.kind),
    ),
  ];
}

function emitRegenerationFailedTrace(input: {
  tracer: TurnTracer;
  turnId: string;
  sessionId: SessionId;
  reason: "regeneration_not_supported" | "regenerated_non_message_emission";
  regeneration: RegenerationRequest;
  commitments: readonly CommitmentRecord[];
  regeneratedEmissionKind?: PendingTurnEmission["kind"];
  regeneratedEmissionReason?: string;
}): void {
  if (!input.tracer.enabled) {
    return;
  }

  input.tracer.emit("commitment_guard.regeneration_failed", {
    turnId: input.turnId,
    session_id: input.sessionId,
    mode: "enforce",
    verdict: "suppressed",
    reason: input.reason,
    violationCount: input.regeneration.violationCount,
    commitmentIds: input.regeneration.commitmentIds,
    commitmentKinds: commitmentKindsForRegeneration(
      input.commitments,
      input.regeneration.commitmentIds,
    ),
    ...(input.regeneratedEmissionKind === undefined
      ? {}
      : { regeneratedEmissionKind: input.regeneratedEmissionKind }),
    ...(input.regeneratedEmissionReason === undefined
      ? {}
      : { regeneratedEmissionReason: input.regeneratedEmissionReason }),
  });
}

function commitmentEmissionForAction(
  result: CommitmentGuardResult,
):
  | Extract<PendingTurnEmission, { kind: "message" }>
  | Extract<PendingTurnEmission, { kind: "suppressed" }> {
  return result.emission.kind === "requires_regeneration"
    ? {
        kind: "suppressed",
        reason: "commitment_violation",
      }
    : result.emission;
}

export class TurnActionCoordinator {
  constructor(private readonly options: TurnActionCoordinatorOptions) {}

  async run(input: RunTurnActionInput): Promise<TurnActionCoordinatorResult> {
    const deliberationEmission = pendingEmissionFromDeliberation(input.deliberation);
    const pendingActionJudge = new LLMPendingActionJudge({
      llmClient: input.llmClient,
      model: this.options.pendingActionJudgeModel,
    });
    const onPendingActionRejected = (event: PendingActionRejection) => {
      if (!this.options.tracer.enabled) {
        return;
      }

      this.options.tracer.emit("working_memory.degraded", {
        turnId: input.turnId,
        session_id: input.sessionId,
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
    const guarded =
      deliberationEmission.kind !== "message"
        ? {
            deliberation: input.deliberation,
            actionResult: await performAction({
              response: "",
              emission: deliberationEmission,
              toolCalls: input.deliberation.tool_calls,
              intents: [],
              workingMemory: input.workingMemory,
            }),
          }
        : await this.performGuardedAction({
            ...input,
            deliberationEmission,
            pendingActionJudge,
            onPendingActionRejected,
          });
    const actionResult = guarded.actionResult;
    const actionEmission: PendingTurnEmission = actionResult.emission ?? {
      kind: "message",
      content: actionResult.response,
    };

    return {
      actionResult,
      actionEmission,
      deliberation: guarded.deliberation,
    };
  }

  private async performGuardedAction(
    input: RunTurnActionInput & {
      deliberationEmission: Extract<PendingTurnEmission, { kind: "message" }>;
      pendingActionJudge: LLMPendingActionJudge;
      onPendingActionRejected: (event: PendingActionRejection) => void;
    },
  ): Promise<{ actionResult: ActionResult; deliberation: DeliberationResult }> {
    const guarded = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId: input.turnId,
      sessionId: input.sessionId,
      phase: "guards",
      sub: "commitment_post_generation",
      run: async () => {
        let currentDeliberation = input.deliberation;
        let currentDeliberationEmission: Extract<PendingTurnEmission, { kind: "message" }> =
          input.deliberationEmission;
        let commitmentCheck = await this.options.commitmentGuardRunner.run({
          llmClient: input.llmClient,
          turnId: input.turnId,
          sessionId: input.sessionId,
          response: currentDeliberation.response,
          userMessage: input.userMessage,
          cognitionInput: input.cognitionInput,
          origin: input.origin,
          autonomyTrigger: input.autonomyTrigger,
          commitments: input.applicableCommitments,
          relevantEntities: input.perceptionEntities,
        });

        if (commitmentCheck.emission.kind === "requires_regeneration") {
          if (currentDeliberation.regenerateFinalResponse === undefined) {
            emitRegenerationFailedTrace({
              tracer: this.options.tracer,
              turnId: input.turnId,
              sessionId: input.sessionId,
              reason: "regeneration_not_supported",
              regeneration: commitmentCheck.emission.regeneration,
              commitments: input.applicableCommitments,
            });
            commitmentCheck = suppressUnsupportedRegeneration(commitmentCheck);
          } else {
            currentDeliberation = await currentDeliberation.regenerateFinalResponse({
              additionalPromptSections: [commitmentCheck.emission.regeneration.promptSection],
            });
            const regeneratedEmission = pendingEmissionFromDeliberation(currentDeliberation);

            if (regeneratedEmission.kind !== "message") {
              emitRegenerationFailedTrace({
                tracer: this.options.tracer,
                turnId: input.turnId,
                sessionId: input.sessionId,
                reason: "regenerated_non_message_emission",
                regeneration: commitmentCheck.emission.regeneration,
                commitments: input.applicableCommitments,
                regeneratedEmissionKind: regeneratedEmission.kind,
                regeneratedEmissionReason: regeneratedEmission.reason,
              });

              return {
                deliberation: currentDeliberation,
                guardedEmission: regeneratedEmission,
                performActionInsideGuard: true,
              };
            }

            currentDeliberationEmission = regeneratedEmission;
            commitmentCheck = await this.options.commitmentGuardRunner.run({
              llmClient: input.llmClient,
              turnId: input.turnId,
              sessionId: input.sessionId,
              response: currentDeliberation.response,
              userMessage: input.userMessage,
              cognitionInput: input.cognitionInput,
              origin: input.origin,
              autonomyTrigger: input.autonomyTrigger,
              commitments: input.applicableCommitments,
              relevantEntities: input.perceptionEntities,
              regenerationAttempted: true,
            });
          }
        }

        if (commitmentCheck.emission.kind === "requires_regeneration") {
          commitmentCheck = suppressUnsupportedRegeneration(commitmentCheck);
        }

        const rawCommitmentEmission = commitmentEmissionForAction(commitmentCheck);
        const commitmentEmission =
          rawCommitmentEmission.kind === "message"
            ? withMessageMetadata(rawCommitmentEmission, currentDeliberationEmission)
            : rawCommitmentEmission;
        const guardedEmission =
          commitmentEmission.kind === "suppressed"
            ? commitmentEmission
            : withMessageMetadata(
                await this.options.postGenerationGuardRunner.run({
                  llmClient: input.llmClient,
                  turnId: input.turnId,
                  response: commitmentEmission.content,
                  sessionId: input.sessionId,
                  persistedUserEntry: input.persistedUserEntry,
                  retrievedEpisodes: input.retrievedEpisodes,
                  activeCommitments: input.applicableCommitments,
                  closureLoop: input.workingMemory.discourse_state?.closure_loop ?? null,
                  closurePressureHistory:
                    input.workingMemory.discourse_state?.closure_pressure_history ?? [],
                  recentSuppressions:
                    input.workingMemory.discourse_state?.recent_suppressions ?? [],
                  currentUserClosureKind: input.currentUserClosureKind,
                  currentTurn: input.workingMemory.turn_counter,
                  audienceEntityId: input.audienceEntityId,
                  knownInternalIdentifiers: input.knownInternalIdentifiers,
                  persistedUserEntries: input.persistedUserEntries,
                }),
                commitmentEmission,
              );

        return {
          deliberation: currentDeliberation,
          guardedEmission,
          performActionInsideGuard: false,
        };
      },
      completedSub: (result) => `emission=${result.guardedEmission.kind}`,
    });

    if (guarded.performActionInsideGuard) {
      return {
        deliberation: guarded.deliberation,
        actionResult: await performAction({
          response: "",
          emission: guarded.guardedEmission,
          toolCalls: guarded.deliberation.tool_calls,
          intents: guarded.deliberation.intents,
          workingMemory: input.workingMemory,
        }),
      };
    }

    return {
      deliberation: guarded.deliberation,
      actionResult: await performAction({
        response: guarded.guardedEmission.kind === "message" ? guarded.guardedEmission.content : "",
        emission: guarded.guardedEmission,
        toolCalls: guarded.deliberation.tool_calls,
        intents: guarded.deliberation.intents,
        workingMemory: input.workingMemory,
        pendingActionJudge: input.pendingActionJudge,
        pendingActionEmbeddingClient: this.options.embeddingClient,
        pendingActionTimestamp: this.options.clock.now(),
        onPendingActionRejected: input.onPendingActionRejected,
      }),
    };
  }
}
