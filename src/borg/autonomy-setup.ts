// Builds autonomy wake sources and the scheduler that runs autonomous turns.

import {
  AutonomyScheduler,
  type AutonomyWakesRepository,
  createCommitmentExpiringTrigger,
  createCommitmentRevokedCondition,
  createExecutiveFocusDueTrigger,
  createGoalFollowupDueTrigger,
  createMoodValenceDropCondition,
  createOpenQuestionDormantTrigger,
  createOpenQuestionUrgencyBumpCondition,
  createScheduledReflectionTrigger,
} from "../autonomy/index.js";
import type { TurnOrchestrator } from "../cognition/index.js";
import type { Config } from "../config/index.js";
import type { ExecutiveStepsRepository } from "../executive/index.js";
import type { MoodRepository } from "../memory/affective/index.js";
import type { CommitmentRepository } from "../memory/commitments/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import type { GoalsRepository, OpenQuestionsRepository } from "../memory/self/index.js";
import type { StreamWatermarkRepository } from "../stream/index.js";
import type { ToolDispatcher } from "../tools/index.js";
import type { Clock } from "../util/clock.js";
import type { BorgStreamWriterFactory } from "./types.js";

export type BuildAutonomySchedulerOptions = {
  config: Config;
  commitmentRepository: CommitmentRepository;
  episodicRepository: EpisodicRepository;
  goalsRepository: GoalsRepository;
  executiveStepsRepository: ExecutiveStepsRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  moodRepository: MoodRepository;
  streamWatermarkRepository: StreamWatermarkRepository;
  autonomyWakesRepository: AutonomyWakesRepository;
  turnOrchestrator: TurnOrchestrator;
  toolDispatcher: ToolDispatcher;
  createStreamWriter: BorgStreamWriterFactory;
  clock: Clock;
};

export function buildAutonomyScheduler(options: BuildAutonomySchedulerOptions): AutonomyScheduler {
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
            watermarkRepository: options.streamWatermarkRepository,
            threshold: options.config.executive.goalFocusThreshold,
            stalenessMs: options.config.autonomy.executiveFocus.stalenessSec * 1_000,
            dueLeadMs: options.config.autonomy.executiveFocus.dueLeadSec * 1_000,
            wakeCooldownMs: options.config.autonomy.executiveFocus.wakeCooldownSec * 1_000,
            deadlineLookaheadMs: options.config.autonomy.triggers.goalFollowupDue.lookaheadMs,
            goalFollowupDue: {
              enabled: options.config.autonomy.triggers.goalFollowupDue.enabled,
              lookaheadMs: options.config.autonomy.triggers.goalFollowupDue.lookaheadMs,
              staleMs: options.config.autonomy.triggers.goalFollowupDue.staleMs,
            },
            clock: options.clock,
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
    maxWakesPerWindow: options.config.autonomy.maxWakesPerWindow,
    budgetWindowMs: options.config.autonomy.budgetWindowMs,
    clock: options.clock,
    createStreamWriter: options.createStreamWriter,
    watermarkRepository: options.streamWatermarkRepository,
    wakeRepository: options.autonomyWakesRepository,
    turnOrchestrator: options.turnOrchestrator,
    toolDispatcher: options.toolDispatcher,
    sources: autonomySources,
  });
}
