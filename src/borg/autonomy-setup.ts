// Builds autonomy wake sources and the scheduler that runs autonomous turns.

import {
  AutonomyScheduler,
  type AutonomyWakesRepository,
  type ScheduledWakesRepository,
  createCommitmentExpiringTrigger,
  createCommitmentRevokedCondition,
  createExecutiveFocusDueTrigger,
  createGoalFollowupDueTrigger,
  createMoodValenceDropCondition,
  createOpenQuestionDormantTrigger,
  createOpenQuestionUrgencyBumpCondition,
  createScheduledReflectionTrigger,
  createScheduledWakeTrigger,
} from "../autonomy/index.js";
import type { TurnOrchestrator } from "../cognition/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { Config } from "../config/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { ExecutiveStepsRepository } from "../executive/index.js";
import type { MoodRepository } from "../memory/affective/index.js";
import type { CommitmentRepository } from "../memory/commitments/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import type { SelfDecisionRepository } from "../memory/self-decisions/index.js";
import type { TrainOfThoughtRepository } from "../memory/train-of-thought/index.js";
import type { GoalsRepository, OpenQuestionsRepository } from "../memory/self/index.js";
import type { AutonomousOutboundPolicy } from "../outbound/autonomous-policy.js";
import { autonomousOutboundActionAvailabilityKey } from "../outbound/outbound-prompt.js";
import type { StreamWatermarkRepository } from "../stream/index.js";
import type { ToolDispatcher } from "../tools/index.js";
import { OUTBOUND_POST_TOOL_NAME } from "../tools/internal/outbound-post-name.js";
import type { Clock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import type { BorgStreamWriterFactory } from "./types.js";

export type BuildAutonomySchedulerOptions = {
  config: Config;
  commitmentRepository: CommitmentRepository;
  episodicRepository: EpisodicRepository;
  embeddingClient: EmbeddingClient;
  goalsRepository: GoalsRepository;
  executiveStepsRepository: ExecutiveStepsRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  moodRepository: MoodRepository;
  streamWatermarkRepository: StreamWatermarkRepository;
  autonomyWakesRepository: AutonomyWakesRepository;
  scheduledWakesRepository: ScheduledWakesRepository;
  selfDecisionRepository: SelfDecisionRepository;
  trainOfThoughtRepository: TrainOfThoughtRepository;
  turnOrchestrator: TurnOrchestrator;
  toolDispatcher: ToolDispatcher;
  autonomousOutboundPolicy: Pick<AutonomousOutboundPolicy, "promptContext" | "actionRouteTopology">;
  createStreamWriter: BorgStreamWriterFactory;
  clock: Clock;
  tracer?: TurnTracer;
};

export function buildAutonomyScheduler(options: BuildAutonomySchedulerOptions): AutonomyScheduler {
  const goalStaleBackoffActionAvailabilityKey = () => {
    const outboundTool = options.toolDispatcher.getDefinition(OUTBOUND_POST_TOOL_NAME);
    const outboundToolAvailable = outboundTool?.allowedOrigins.includes("autonomous") === true;

    if (!outboundToolAvailable) {
      return null;
    }

    const context = options.autonomousOutboundPolicy.promptContext(DEFAULT_SESSION_ID);

    if (context === null) {
      return null;
    }

    return autonomousOutboundActionAvailabilityKey({
      context,
      routeTopology: options.autonomousOutboundPolicy.actionRouteTopology(DEFAULT_SESSION_ID),
      outboundToolAvailable,
    });
  };
  const autonomySources = [
    ...(options.config.autonomy.triggers.commitmentExpiring.enabled
      ? [
          createCommitmentExpiringTrigger({
            commitmentRepository: options.commitmentRepository,
            watermarkRepository: options.streamWatermarkRepository,
            lookaheadMs: options.config.autonomy.triggers.commitmentExpiring.lookaheadMs,
            clock: options.clock,
          }),
        ]
      : []),
    ...(options.config.autonomy.triggers.goalFollowupDue.enabled
      ? [
          createGoalFollowupDueTrigger({
            goalsRepository: options.goalsRepository,
            watermarkRepository: options.streamWatermarkRepository,
            lookaheadMs: options.config.autonomy.triggers.goalFollowupDue.lookaheadMs,
            staleMs: options.config.autonomy.triggers.goalFollowupDue.staleMs,
            // goal_followup_due and executive_focus_due intentionally share
            // one per-goal empty-wake policy and its historical watermark.
            staleBackoff: {
              baseCooldownMs: options.config.autonomy.executiveFocus.wakeCooldownSec * 1_000,
              multiplier: options.config.autonomy.executiveFocus.emptyWakeBackoffMultiplier,
              maxCooldownMs: options.config.autonomy.executiveFocus.wakeCooldownMaxSec * 1_000,
              dormancyCount: options.config.autonomy.executiveFocus.emptyWakeDormancyCount,
            },
            respectStaleBackoff:
              options.config.autonomy.triggers.goalFollowupDue.respectStaleBackoff,
            executiveScoring: {
              embeddingClient: options.embeddingClient,
              threshold: options.config.executive.goalFocusThreshold,
              deadlineLookaheadMs: options.config.autonomy.triggers.goalFollowupDue.lookaheadMs,
              staleMs: options.config.autonomy.triggers.goalFollowupDue.staleMs,
              tracer: options.tracer,
            },
            goalStaleBackoffActionAvailabilityKey,
            clock: options.clock,
          }),
        ]
      : []),
    ...(options.config.autonomy.executiveFocus.enabled
      ? [
          createExecutiveFocusDueTrigger({
            enabled: options.config.autonomy.executiveFocus.enabled,
            goalsRepository: options.goalsRepository,
            executiveStepsRepository: options.executiveStepsRepository,
            episodicRepository: options.episodicRepository,
            embeddingClient: options.embeddingClient,
            watermarkRepository: options.streamWatermarkRepository,
            threshold: options.config.executive.goalFocusThreshold,
            stalenessMs: options.config.autonomy.executiveFocus.stalenessSec * 1_000,
            progressDebtStaleMs: options.config.autonomy.triggers.goalFollowupDue.staleMs,
            dueLeadMs: options.config.autonomy.executiveFocus.dueLeadSec * 1_000,
            wakeCooldownMs: options.config.autonomy.executiveFocus.wakeCooldownSec * 1_000,
            wakeCooldownBackoffMultiplier:
              options.config.autonomy.executiveFocus.emptyWakeBackoffMultiplier,
            wakeCooldownMaxMs: options.config.autonomy.executiveFocus.wakeCooldownMaxSec * 1_000,
            wakeEmptyDormancyCount: options.config.autonomy.executiveFocus.emptyWakeDormancyCount,
            deadlineLookaheadMs: options.config.autonomy.triggers.goalFollowupDue.lookaheadMs,
            goalFollowupDue: {
              enabled: options.config.autonomy.triggers.goalFollowupDue.enabled,
              lookaheadMs: options.config.autonomy.triggers.goalFollowupDue.lookaheadMs,
              staleMs: options.config.autonomy.triggers.goalFollowupDue.staleMs,
            },
            clock: options.clock,
            tracer: options.tracer,
            goalStaleBackoffActionAvailabilityKey,
          }),
        ]
      : []),
    ...(options.config.autonomy.triggers.openQuestionDormant.enabled
      ? [
          createOpenQuestionDormantTrigger({
            openQuestionsRepository: options.openQuestionsRepository,
            watermarkRepository: options.streamWatermarkRepository,
            dormantMs: options.config.autonomy.triggers.openQuestionDormant.dormantMs,
            clock: options.clock,
          }),
        ]
      : []),
    ...(options.config.autonomy.triggers.scheduledReflection.enabled
      ? [
          createScheduledReflectionTrigger({
            watermarkRepository: options.streamWatermarkRepository,
            intervalMs: options.config.autonomy.triggers.scheduledReflection.intervalMs,
            clock: options.clock,
          }),
        ]
      : []),
    ...(options.config.autonomy.triggers.scheduledWake.enabled
      ? [
          createScheduledWakeTrigger({
            scheduledWakesRepository: options.scheduledWakesRepository,
            watermarkRepository: options.streamWatermarkRepository,
            clock: options.clock,
          }),
        ]
      : []),
    ...(options.config.autonomy.conditions.commitmentRevoked.enabled
      ? [
          createCommitmentRevokedCondition({
            commitmentRepository: options.commitmentRepository,
            watermarkRepository: options.streamWatermarkRepository,
            clock: options.clock,
          }),
        ]
      : []),
    ...(options.config.autonomy.conditions.moodValenceDrop.enabled
      ? [
          createMoodValenceDropCondition({
            moodRepository: options.moodRepository,
            watermarkRepository: options.streamWatermarkRepository,
            threshold: options.config.autonomy.conditions.moodValenceDrop.threshold,
            windowN: options.config.autonomy.conditions.moodValenceDrop.windowN,
            activationPeriodMs:
              options.config.autonomy.conditions.moodValenceDrop.activationPeriodMs,
            clock: options.clock,
          }),
        ]
      : []),
    ...(options.config.autonomy.conditions.openQuestionUrgencyBump.enabled
      ? [
          createOpenQuestionUrgencyBumpCondition({
            openQuestionsRepository: options.openQuestionsRepository,
            watermarkRepository: options.streamWatermarkRepository,
            threshold: options.config.autonomy.conditions.openQuestionUrgencyBump.threshold,
            clock: options.clock,
          }),
        ]
      : []),
  ];

  return new AutonomyScheduler({
    enabled: options.config.autonomy.enabled,
    intervalMs: options.config.autonomy.intervalMs,
    prepToolTimeoutMs: options.config.autonomy.prepToolTimeoutMs,
    maxWakesPerWindow: options.config.autonomy.maxWakesPerWindow,
    goalWakeBatchMax: options.config.autonomy.goalWakeBatchMax,
    budgetWindowMs: options.config.autonomy.budgetWindowMs,
    reservedContemplativeWakesPerWindow:
      options.config.autonomy.reservedContemplativeWakesPerWindow,
    fleetBrake: options.config.autonomy.fleetBrake,
    respectGoalFollowupStaleBackoff:
      options.config.autonomy.triggers.goalFollowupDue.respectStaleBackoff,
    clock: options.clock,
    createStreamWriter: options.createStreamWriter,
    watermarkRepository: options.streamWatermarkRepository,
    wakeRepository: options.autonomyWakesRepository,
    selfDecisionRepository: options.selfDecisionRepository,
    trainOfThoughtRepository: options.trainOfThoughtRepository,
    goalsRepository: options.goalsRepository,
    turnOrchestrator: options.turnOrchestrator,
    toolDispatcher: options.toolDispatcher,
    sources: autonomySources,
  });
}
