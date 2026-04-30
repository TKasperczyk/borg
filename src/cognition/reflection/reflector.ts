import type { RetrievalConfidence, RetrievedEpisode } from "../../retrieval/index.js";
import { type LLMClient, type LLMToolDefinition, toToolInputSchema } from "../../llm/index.js";
import { StreamWriter } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { ExecutiveFocus, ExecutiveStepsRepository } from "../../executive/index.js";
import {
  executiveStepGoalIdSchema,
  executiveStepIdSchema,
  executiveStepKindSchema,
  executiveStepStatusSchema,
} from "../../executive/types.js";
import { GoalsRepository, TraitsRepository, type GoalRecord } from "../../memory/self/index.js";
import type { IdentityService } from "../../memory/identity/index.js";
import type { ReviewQueueRepository } from "../../memory/semantic/index.js";
import { ProceduralEvidenceRepository, SkillRepository } from "../../memory/procedural/index.js";
import {
  appendInternalFailureEvent,
  appendOpenQuestionHookFailureEvent,
} from "../../memory/self/review-open-question-hook.js";
import { EpisodicRepository, episodeIdSchema } from "../../memory/episodic/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { EntityId, EpisodeId, GoalId, SkillId, StreamEntryId } from "../../util/ids.js";
import { z } from "zod";

import type { ActionResult } from "../action/index.js";
import type { DeliberationResult, SelfSnapshot } from "../deliberation/deliberator.js";
import { SuppressionSet } from "../attention/index.js";
import { intentRecordSchema, type IntentRecord, type PerceptionResult } from "../types.js";
export type ReflectionContext = {
  origin?: "user" | "autonomous";
  userMessage: string;
  perception?: PerceptionResult;
  workingMemory: WorkingMemory;
  selfSnapshot: SelfSnapshot;
  deliberationResult: DeliberationResult;
  actionResult: ActionResult;
  retrievedEpisodes: RetrievedEpisode[];
  retrievalConfidence: RetrievalConfidence;
  executiveFocus?: ExecutiveFocus | null;
  selectedSkillId?: SkillId | null;
  audienceEntityId?: EntityId | null;
  suppressionSet: SuppressionSet;
  // Sprint 56: stream entries persisted by the just-completed turn
  // (user_msg + agent_msg). Used as evidence for trait demonstrations
  // since the episode hasn't been extracted yet at reflection time.
  currentTurnStreamEntryIds?: readonly StreamEntryId[];
};

const SURFACED_TTL_TURNS = 4;
const NOISE_TTL_TURNS = 2;
const REFLECTION_TOOL_NAME = "EmitTurnReflection";
const DEFAULT_REFLECTION_MAX_TOKENS = 768;

const traitDemonstrationSchema = z.object({
  trait_label: z.string().min(1),
  evidence: z.string().min(1),
  strength_delta: z.number().min(0).max(0.2),
});
const executiveStepOutcomeStatusSchema = executiveStepStatusSchema.exclude(["queued"]);

const executiveStepOutcomeSchema = z.object({
  step_id: executiveStepIdSchema,
  new_status: executiveStepOutcomeStatusSchema,
  evidence: z.string(),
});

const proposedExecutiveStepSchema = z.object({
  goal_id: executiveStepGoalIdSchema,
  description: z.string().min(1),
  kind: executiveStepKindSchema,
  due_at: z.number().finite().nullable().optional(),
  rationale: z.string().min(1),
});

const reflectionOpenQuestionSchema = z.object({
  question: z.string().min(1),
  urgency: z.number().min(0).max(1),
  related_episode_ids: z.array(episodeIdSchema),
});

const strictReflectionOutputSchema = z.object({
  advanced_goals: z
    .array(
      z.object({
        goal_id: z.string().min(1),
        evidence: z.string().min(1),
      }),
    )
    .describe(
      "Goals advanced by this turn. Mark only if the turn took a concrete step toward the goal, not just discussed it.",
    )
    .default([]),
  procedural_outcomes: z
    .array(
      z.object({
        attempt_turn_counter: z
          .number()
          .int()
          .nonnegative()
          .describe(
            "Identifier matching the turn_counter of the pending procedural attempt being graded.",
          ),
        classification: z.enum(["success", "failure", "unclear"]),
        evidence: z.string().min(1),
        grounded: z
          .boolean()
          .describe(
            "True only when evidence is grounded in an actual user signal about the pending attempt, not assistant self-narration.",
          ),
        skill_actually_applied: z
          .boolean()
          .describe(
            "True only if the prior assistant turn's response actually executed the attempt's approach_summary. False if the model ignored or substituted a different approach. Drives whether the skill posterior is credited or blamed.",
          ),
      }),
    )
    .describe(
      "Outcomes for prior pending_procedural_attempts. Judge success only from the user's follow-up signal, never from the assistant's wording. Use attempt_turn_counter to identify which pending attempt each outcome refers to. Omit attempts that the current turn does not provide evidence about.",
    )
    .default([]),
  trait_demonstrations: z
    .array(traitDemonstrationSchema)
    .describe(
      "Traits the completed assistant turn actually demonstrated through its content or actions. Do not infer a trait from cognitive mode alone.",
    )
    .default([]),
  intent_updates: z
    .array(
      intentRecordSchema.extend({
        status: z.enum(["completed", "abandoned"]),
        evidence: z.string().min(1),
      }),
    )
    .describe(
      "Prior pending_intents resolved by this completed turn. Include only exact prior intents with clear evidence, marked completed or abandoned.",
    )
    .default([]),
  step_outcomes: z
    .array(executiveStepOutcomeSchema)
    .describe(
      "Status outcomes for executive steps directly affected by the completed turn. Include evidence tied to the turn.",
    )
    .default([]),
  proposed_steps: z
    .array(proposedExecutiveStepSchema)
    .describe(
      "Small next-step proposals for the selected executive goal when it has no open step after this turn.",
    )
    .default([]),
  open_questions: z
    .array(reflectionOpenQuestionSchema)
    .max(5)
    .describe(
      "Durable unresolved questions from this completed turn that should be remembered in self-memory. Emit zero items unless the turn reveals a real question worth revisiting. Write the question in the user's language and attach only related episode ids present in the reflection input.",
    )
    .default([]),
});

const reflectionOutputParseSchema = strictReflectionOutputSchema.extend({
  step_outcomes: z.array(z.unknown()).default([]),
  proposed_steps: z.array(z.unknown()).default([]),
});

type ReflectionOutput = z.infer<typeof strictReflectionOutputSchema>;
type RawReflectionOutput = z.infer<typeof reflectionOutputParseSchema>;

const REFLECTION_TOOL: LLMToolDefinition = {
  name: REFLECTION_TOOL_NAME,
  description:
    "Emit structured post-turn reflection. Mark advanced_goals only for concrete progress, procedural_outcomes only from user follow-up evidence with grounded set explicitly, trait_demonstrations only from turn content, intent_updates only for prior pending intents resolved by the completed turn, executive step outcomes/proposals only when the turn directly supports them, and open_questions only for durable unresolved questions worth remembering.",
  inputSchema: toToolInputSchema(strictReflectionOutputSchema),
};

function emptyReflectionOutput(): ReflectionOutput {
  return {
    advanced_goals: [],
    procedural_outcomes: [],
    trait_demonstrations: [],
    intent_updates: [],
    step_outcomes: [],
    proposed_steps: [],
    open_questions: [],
  };
}

function buildReflectionProvenance(retrievedEpisodes: readonly RetrievedEpisode[]) {
  const episodeIds = [...new Set(retrievedEpisodes.slice(0, 3).map((result) => result.episode.id))];
  return episodeIds.length > 0
    ? {
        kind: "episodes" as const,
        episode_ids: episodeIds,
      }
    : {
        kind: "online" as const,
        process: "reflector",
      };
}

function buildIdentityPatchReviewRefs(
  targetId: string,
  patch: Record<string, unknown>,
  provenance: ReturnType<typeof buildReflectionProvenance>,
) {
  return {
    target_type: "goal" as const,
    target_id: targetId,
    repair_op: "patch" as const,
    patch,
    proposed_provenance: provenance,
    ...(provenance.kind === "episodes"
      ? {
          evidence_episode_ids: provenance.episode_ids,
        }
      : {}),
  };
}

function isProgressOnlyGoalPatch(patch: Record<string, unknown>): boolean {
  const keys = Object.keys(patch).filter((key) => patch[key] !== undefined);

  return (
    keys.length === 2 && keys.every((key) => key === "progress_notes" || key === "last_progress_ts")
  );
}

function intentKey(intent: IntentRecord): string {
  return JSON.stringify([
    intent.description.trim().toLowerCase(),
    intent.next_action === null ? null : intent.next_action.trim().toLowerCase(),
  ]);
}

function selectResolvedIntentKeys(
  pendingIntents: readonly IntentRecord[],
  updates: readonly ReflectionOutput["intent_updates"][number][],
): Set<string> {
  const pendingKeys = new Set(pendingIntents.map((intent) => intentKey(intent)));
  const resolved = new Set<string>();

  for (const update of updates) {
    const key = intentKey(update);

    if (pendingKeys.has(key)) {
      resolved.add(key);
    }
  }

  return resolved;
}

async function resolveAttemptEpisodeIds(
  context: ReflectionContext,
  episodicRepository: EpisodicRepository,
  sourceStreamIds: readonly StreamEntryId[],
): Promise<EpisodeId[]> {
  const retrievedMatches = context.retrievedEpisodes
    .filter((result) =>
      sourceStreamIds.every((streamId) => result.episode.source_stream_ids.includes(streamId)),
    )
    .map((result) => result.episode.id);

  const repositoryMatch = await episodicRepository.findBySourceStreamIdsContaining(sourceStreamIds);
  const repositoryMatchIds = repositoryMatch === null ? [] : [repositoryMatch.id];

  return [...new Set([...retrievedMatches, ...repositoryMatchIds])];
}

function isAutonomousStepOutcomeAllowed(input: {
  currentStatus: string;
  nextStatus: string;
}): boolean {
  return (
    (input.currentStatus === "queued" &&
      (input.nextStatus === "doing" || input.nextStatus === "abandoned")) ||
    (input.currentStatus === "doing" &&
      (input.nextStatus === "blocked" || input.nextStatus === "abandoned"))
  );
}

function isClosingExecutiveStepStatus(
  status: ReflectionOutput["step_outcomes"][number]["new_status"],
) {
  return status === "done" || status === "blocked" || status === "abandoned";
}

function summarizeExecutiveFocusForReflection(focus: ExecutiveFocus | null | undefined) {
  if (focus?.selected_goal === null || focus?.selected_goal === undefined) {
    return null;
  }

  const nextStep = focus.next_step ?? null;

  return {
    selected_goal: {
      goal_id: focus.selected_goal.id,
      description: focus.selected_goal.description,
      progress_notes: focus.selected_goal.progress_notes,
    },
    next_step:
      nextStep === null
        ? null
        : {
            step_id: nextStep.id,
            description: nextStep.description,
            status: nextStep.status,
            kind: nextStep.kind,
            due_at: nextStep.due_at,
          },
  };
}

export type ReflectorOptions = {
  clock?: Clock;
  llmClient?: LLMClient;
  model?: string;
  maxTokens?: number;
  episodicRepository: EpisodicRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  identityService?: Pick<
    IdentityService,
    "updateGoal" | "updateGoalProgressFromReflection" | "addOpenQuestion"
  >;
  reviewQueueRepository?: Pick<ReviewQueueRepository, "enqueue">;
  skillRepository?: SkillRepository;
  proceduralEvidenceRepository?: ProceduralEvidenceRepository;
  executiveStepsRepository?: ExecutiveStepsRepository;
};

export class Reflector {
  private readonly clock: Clock;
  private readonly llmClient?: LLMClient;
  private readonly model?: string;
  private readonly maxTokens: number;

  constructor(private readonly options: ReflectorOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.llmClient = options.llmClient;
    this.model = options.model;
    this.maxTokens = options.maxTokens ?? DEFAULT_REFLECTION_MAX_TOKENS;
  }

  async reflect(context: ReflectionContext, streamWriter: StreamWriter): Promise<WorkingMemory> {
    const reflectionProvenance = buildReflectionProvenance(context.retrievedEpisodes);
    let reflectionOutput = emptyReflectionOutput();

    if (
      context.deliberationResult.thoughts.length > 0 &&
      !context.deliberationResult.thoughtsPersisted
    ) {
      await streamWriter.appendMany(
        context.deliberationResult.thoughts.map((thought) => ({
          kind: "thought",
          content: thought,
        })),
      );
    }

    try {
      reflectionOutput = await this.runReflectionJudgment(context, streamWriter);
    } catch (error) {
      await appendInternalFailureEvent(streamWriter, "reflection_judgment", error);
    }

    const activeGoalsById = new Map(context.selfSnapshot.goals.map((goal) => [goal.id, goal]));
    const reflectionOrigin = context.origin ?? "user";
    const isAutonomousTurn = reflectionOrigin === "autonomous";

    const autonomouslyClosedStepGoalIds = await this.applyExecutiveStepOutcomes(
      context,
      reflectionOutput.step_outcomes,
      streamWriter,
    );
    const proposedSteps = await this.dropAutonomousCloseAndReplaceProposals(
      context,
      reflectionOutput.proposed_steps,
      autonomouslyClosedStepGoalIds,
      streamWriter,
    );
    await this.applyProposedExecutiveSteps(context, proposedSteps, streamWriter);

    const advancedGoals = isAutonomousTurn ? [] : reflectionOutput.advanced_goals;

    for (const advancedGoal of advancedGoals) {
      const goal = activeGoalsById.get(advancedGoal.goal_id as GoalRecord["id"]);

      if (goal === undefined) {
        continue;
      }

      const evidence = advancedGoal.evidence.trim();
      const note = `[${this.clock.now()}] ${evidence}`;
      const nextProgress = goal.progress_notes === null ? note : `${goal.progress_notes}\n${note}`;
      const patch = {
        progress_notes: nextProgress,
        last_progress_ts: this.clock.now(),
      };

      if (
        this.options.identityService === undefined ||
        this.options.reviewQueueRepository === undefined
      ) {
        this.options.goalsRepository.update(goal.id, patch, reflectionProvenance);
        continue;
      }

      const result =
        reflectionOrigin === "user" && isProgressOnlyGoalPatch(patch)
          ? this.options.identityService.updateGoalProgressFromReflection(
              goal.id,
              patch,
              reflectionProvenance,
              { origin: reflectionOrigin },
            )
          : this.options.identityService.updateGoal(goal.id, patch, reflectionProvenance);

      if (result.status === "requires_review") {
        this.options.reviewQueueRepository.enqueue({
          kind: "identity_inconsistency",
          refs: buildIdentityPatchReviewRefs(goal.id, patch, reflectionProvenance),
          reason: `reflector proposed updating goal ${goal.id}`,
        });
      }
    }

    const referencedEpisodeIds = selectReferencedRetrievedEpisodeIds(
      context.deliberationResult,
      context.retrievedEpisodes,
    );
    const referencedEpisodeIdSet = new Set(referencedEpisodeIds);

    for (const result of context.retrievedEpisodes) {
      const used = referencedEpisodeIdSet.has(result.episode.id);

      if (used) {
        const stats = this.options.episodicRepository.getStats(result.episode.id);

        if (stats !== null) {
          this.options.episodicRepository.updateStats(result.episode.id, {
            use_count: stats.use_count + 1,
          });
        }

        context.suppressionSet.suppress(result.episode.id, "already surfaced", SURFACED_TTL_TURNS);
        continue;
      }

      if (context.deliberationResult.path === "system_2") {
        context.suppressionSet.suppress(result.episode.id, "noise this session", NOISE_TTL_TURNS);
      }
    }

    await this.applyReflectionOpenQuestions(
      context,
      reflectionOutput.open_questions,
      referencedEpisodeIdSet,
      streamWriter,
    );

    context.suppressionSet.tickTurn();

    let nextWorkingMemory: WorkingMemory = {
      ...context.actionResult.workingMemory,
      updated_at: this.clock.now(),
    };
    const intentUpdates = isAutonomousTurn ? [] : reflectionOutput.intent_updates;

    if (isAutonomousTurn && reflectionOutput.intent_updates.length > 0) {
      await this.appendReflectorInternalEvent(streamWriter, {
        hook: "reflector_intent_update_dropped",
        reason: "autonomous_turn",
        count: reflectionOutput.intent_updates.length,
      });
    }

    const resolvedIntentKeys = selectResolvedIntentKeys(
      context.workingMemory.pending_intents,
      intentUpdates,
    );

    if (resolvedIntentKeys.size > 0) {
      nextWorkingMemory = {
        ...nextWorkingMemory,
        pending_intents: nextWorkingMemory.pending_intents.filter(
          (intent) => !resolvedIntentKeys.has(intentKey(intent)),
        ),
      };
    }

    // Sprint 56: trait evidence is the assistant turn that demonstrated
    // the trait, not arbitrary memories the planner referenced. The
    // episode hasn't been extracted yet, so capture the current turn's
    // stream entries; the orchestrator resolves them to the extracted
    // episode at consumption time.
    const traitEvidenceStreamIds = context.currentTurnStreamEntryIds ?? [];
    const traitDemonstration = selectTraitDemonstration(reflectionOutput.trait_demonstrations);

    if (
      // Lagged trait attribution must only come from user-visible turns.
      // Autonomous/self-talk turns are not shown to the user, so a later
      // user reply must not be treated as evidence about them.
      context.origin !== "autonomous" &&
      nextWorkingMemory.pending_trait_attribution === null &&
      traitDemonstration !== null &&
      traitEvidenceStreamIds.length > 0
    ) {
      nextWorkingMemory = {
        ...nextWorkingMemory,
        pending_trait_attribution: {
          trait_label: traitDemonstration.trait_label,
          strength_delta: traitDemonstration.strength_delta,
          source_stream_entry_ids: [...traitEvidenceStreamIds],
          source_episode_ids: [],
          turn_completed_ts: this.clock.now(),
          audience_entity_id: context.audienceEntityId ?? null,
        },
      };
    }

    const pendingProceduralAttempts = context.workingMemory.pending_procedural_attempts ?? [];
    // Sprint 56: only user turns produce procedural feedback. An
    // autonomous wake's "user message" is internal and can't legitimately
    // grade what the prior user turn attempted; skipping here prevents
    // self-grading from retiring real attempts or biasing posteriors.
    if (pendingProceduralAttempts.length > 0 && !isAutonomousTurn) {
      const retiredTurnCounters = new Set<number>();
      const outcomesByTurn = new Map(
        reflectionOutput.procedural_outcomes.map((outcome) => [
          outcome.attempt_turn_counter,
          outcome,
        ]),
      );

      for (const attempt of pendingProceduralAttempts) {
        const outcome = outcomesByTurn.get(attempt.turn_counter);

        if (outcome === undefined || !outcome.grounded) {
          // Ungrounded outcomes and silence both leave the attempt
          // pending for a later turn (or TTL expiry).
          continue;
        }

        try {
          if (this.options.proceduralEvidenceRepository !== undefined) {
            const resolvedEpisodeIds = await resolveAttemptEpisodeIds(
              context,
              this.options.episodicRepository,
              attempt.source_stream_ids,
            );

            this.options.proceduralEvidenceRepository.insert({
              pendingAttemptSnapshot: attempt,
              classification: outcome.classification,
              evidenceText: outcome.evidence,
              grounded: true,
              skillActuallyApplied: outcome.skill_actually_applied,
              resolvedEpisodeIds,
              audienceEntityId: attempt.audience_entity_id,
            });

            if (
              attempt.selected_skill_id !== null &&
              (outcome.classification === "success" || outcome.classification === "failure") &&
              outcome.skill_actually_applied &&
              this.options.skillRepository !== undefined
            ) {
              // Only credit/blame the posterior when the model actually
              // executed the suggested approach. If it ignored the skill,
              // the user's success or failure feedback isn't evidence
              // about the skill itself.
              const updatedSkill = this.options.skillRepository.recordOutcome(
                attempt.selected_skill_id,
                outcome.classification === "success",
                resolvedEpisodeIds,
                attempt.procedural_context ?? null,
              );

              if (updatedSkill.status === "superseded") {
                await this.appendReflectorInternalEvent(streamWriter, {
                  hook: "record_outcome_skipped_superseded",
                  skill_id: attempt.selected_skill_id,
                  attempt_turn_counter: attempt.turn_counter,
                  classification: outcome.classification,
                });
              }
            }
          }

          // Only actionable signals retire the attempt; an "unclear" but
          // grounded outcome stays pending in case later turns clarify.
          if (outcome.classification === "success" || outcome.classification === "failure") {
            retiredTurnCounters.add(attempt.turn_counter);
          }
        } catch (error) {
          await appendInternalFailureEvent(streamWriter, "procedural_evidence_record", error);
        }
      }

      const survivingAttempts = pendingProceduralAttempts.filter(
        (attempt) => !retiredTurnCounters.has(attempt.turn_counter),
      );

      nextWorkingMemory = {
        ...nextWorkingMemory,
        pending_procedural_attempts: survivingAttempts,
      };
    }

    return nextWorkingMemory;
  }

  private async applyExecutiveStepOutcomes(
    context: ReflectionContext,
    outcomes: readonly ReflectionOutput["step_outcomes"][number][],
    streamWriter: StreamWriter,
  ): Promise<Set<GoalId>> {
    const autonomouslyClosedStepGoalIds = new Set<GoalId>();

    if (outcomes.length === 0) {
      return autonomouslyClosedStepGoalIds;
    }

    const repository = this.options.executiveStepsRepository;

    if (repository === undefined) {
      await this.appendReflectorInternalEvent(streamWriter, {
        hook: "reflector_step_outcome_dropped",
        reason: "executive_steps_repository_unavailable",
        count: outcomes.length,
      });
      return autonomouslyClosedStepGoalIds;
    }

    const visibleGoalIds = new Set(context.selfSnapshot.goals.map((goal) => goal.id));
    const selectedGoalId = context.executiveFocus?.selected_goal?.id ?? null;

    for (const outcome of outcomes) {
      const evidence = outcome.evidence.trim();

      if (evidence.length === 0) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_outcome_dropped",
          reason: "empty_evidence",
          step_id: outcome.step_id,
          new_status: outcome.new_status,
        });
        continue;
      }

      const current = repository.get(outcome.step_id);

      if (current === null) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_outcome_dropped",
          reason: "missing_step",
          step_id: outcome.step_id,
          new_status: outcome.new_status,
        });
        continue;
      }

      if (selectedGoalId === null) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_outcome_dropped",
          reason: "no_selected_goal",
          step_id: outcome.step_id,
          new_status: outcome.new_status,
        });
        continue;
      }

      if (!visibleGoalIds.has(selectedGoalId)) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_outcome_dropped",
          reason: "selected_goal_not_visible",
          step_id: outcome.step_id,
          goal_id: selectedGoalId,
          new_status: outcome.new_status,
        });
        continue;
      }

      if (!visibleGoalIds.has(current.goal_id)) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_outcome_dropped",
          reason: "step_goal_not_visible",
          step_id: outcome.step_id,
          goal_id: current.goal_id,
          new_status: outcome.new_status,
        });
        continue;
      }

      if (current.goal_id !== selectedGoalId) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_outcome_dropped",
          reason: "step_goal_mismatch",
          step_id: outcome.step_id,
          goal_id: current.goal_id,
          selected_goal_id: selectedGoalId,
          new_status: outcome.new_status,
        });
        continue;
      }

      if (
        context.origin === "autonomous" &&
        !isAutonomousStepOutcomeAllowed({
          currentStatus: current.status,
          nextStatus: outcome.new_status,
        })
      ) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_outcome_dropped",
          reason: "autonomous_transition_forbidden",
          step_id: outcome.step_id,
          current_status: current.status,
          new_status: outcome.new_status,
        });
        continue;
      }

      try {
        repository.update(outcome.step_id, {
          status: outcome.new_status,
          last_attempt_ts: this.clock.now(),
        });

        if (context.origin === "autonomous" && isClosingExecutiveStepStatus(outcome.new_status)) {
          autonomouslyClosedStepGoalIds.add(current.goal_id);
        }
      } catch (error) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_outcome_dropped",
          reason: "repository_update_failed",
          step_id: outcome.step_id,
          current_status: current.status,
          new_status: outcome.new_status,
          error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
        });
      }
    }

    return autonomouslyClosedStepGoalIds;
  }

  private async dropAutonomousCloseAndReplaceProposals(
    context: ReflectionContext,
    proposals: readonly ReflectionOutput["proposed_steps"][number][],
    autonomouslyClosedStepGoalIds: ReadonlySet<GoalId>,
    streamWriter: StreamWriter,
  ): Promise<ReflectionOutput["proposed_steps"]> {
    if (
      context.origin !== "autonomous" ||
      proposals.length === 0 ||
      autonomouslyClosedStepGoalIds.size === 0
    ) {
      return [...proposals];
    }

    const kept: ReflectionOutput["proposed_steps"] = [];

    for (const proposal of proposals) {
      if (!autonomouslyClosedStepGoalIds.has(proposal.goal_id)) {
        kept.push(proposal);
        continue;
      }

      await this.appendReflectorInternalEvent(streamWriter, {
        hook: "reflector_proposal_dropped_close_and_replace",
        goal_id: proposal.goal_id,
        description: proposal.description,
      });
    }

    return kept;
  }

  private async applyProposedExecutiveSteps(
    context: ReflectionContext,
    proposals: readonly ReflectionOutput["proposed_steps"][number][],
    streamWriter: StreamWriter,
  ): Promise<void> {
    if (proposals.length === 0) {
      return;
    }

    const repository = this.options.executiveStepsRepository;

    if (repository === undefined) {
      await this.appendReflectorInternalEvent(streamWriter, {
        hook: "reflector_step_proposal_dropped",
        reason: "executive_steps_repository_unavailable",
        count: proposals.length,
      });
      return;
    }

    const visibleGoalIds = new Set(context.selfSnapshot.goals.map((goal) => goal.id));
    const selectedGoal = context.executiveFocus?.selected_goal ?? null;

    if (selectedGoal === null) {
      await this.appendReflectorInternalEvent(streamWriter, {
        hook: "reflector_step_proposal_dropped",
        reason: "no_selected_goal",
        count: proposals.length,
      });
      return;
    }

    if (!visibleGoalIds.has(selectedGoal.id)) {
      await this.appendReflectorInternalEvent(streamWriter, {
        hook: "reflector_step_proposal_dropped",
        reason: "selected_goal_not_visible",
        goal_id: selectedGoal.id,
        count: proposals.length,
      });
      return;
    }

    if (repository.listOpen(selectedGoal.id).length > 0) {
      return;
    }

    const provenance = await this.buildStepReflectionProvenance(context);

    for (const proposal of proposals) {
      if (!visibleGoalIds.has(proposal.goal_id)) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_proposal_dropped",
          reason: "goal_not_visible",
          goal_id: proposal.goal_id,
        });
        continue;
      }

      if (proposal.goal_id !== selectedGoal.id) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_proposal_dropped",
          reason: "non_selected_goal",
          goal_id: proposal.goal_id,
          selected_goal_id: selectedGoal.id,
        });
        continue;
      }

      if (proposal.kind === "wait" && (proposal.due_at === null || proposal.due_at === undefined)) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_proposal_dropped",
          reason: "wait_without_due_at",
          goal_id: selectedGoal.id,
          description: proposal.description,
        });
        continue;
      }

      if (repository.listOpen(selectedGoal.id).length >= 3) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_proposal_dropped",
          reason: "open_step_cap",
          goal_id: selectedGoal.id,
          description: proposal.description,
        });
        continue;
      }

      try {
        repository.add({
          goalId: selectedGoal.id,
          description: proposal.description,
          kind: proposal.kind,
          dueAt: proposal.due_at ?? null,
          provenance,
        });
      } catch (error) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_step_proposal_dropped",
          reason: "repository_add_failed",
          goal_id: selectedGoal.id,
          description: proposal.description,
          error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
        });
      }
    }
  }

  private async buildStepReflectionProvenance(
    context: ReflectionContext,
  ): Promise<ReturnType<typeof buildReflectionProvenance>> {
    const sourceStreamIds = context.currentTurnStreamEntryIds ?? [];

    if (sourceStreamIds.length > 0) {
      const currentTurnEpisode =
        await this.options.episodicRepository.findBySourceStreamIdsContaining(sourceStreamIds);

      if (currentTurnEpisode !== null) {
        return {
          kind: "episodes",
          episode_ids: [currentTurnEpisode.id],
        };
      }
    }

    const reflectionProvenance = buildReflectionProvenance(context.retrievedEpisodes);

    if (reflectionProvenance.kind === "episodes") {
      return reflectionProvenance;
    }

    return {
      kind: "online",
      process: "turn-reflection",
    };
  }

  private async appendReflectorInternalEvent(
    streamWriter: StreamWriter,
    content: Record<string, unknown>,
  ): Promise<void> {
    try {
      await streamWriter.append({
        kind: "internal_event",
        content,
      });
    } catch {
      // Best-effort logging only.
    }
  }

  private async applyReflectionOpenQuestions(
    context: ReflectionContext,
    proposals: readonly ReflectionOutput["open_questions"][number][],
    referencedEpisodeIdSet: ReadonlySet<EpisodeId>,
    streamWriter: StreamWriter,
  ): Promise<void> {
    if (proposals.length === 0) {
      return;
    }

    const identityService = this.options.identityService;

    if (identityService === undefined) {
      await appendOpenQuestionHookFailureEvent(
        streamWriter,
        "reflection_open_question",
        new Error("identity_service_unavailable"),
      );
      return;
    }

    for (const proposal of proposals) {
      const question = proposal.question.trim();

      if (question.length === 0) {
        continue;
      }

      const proposedEpisodeIds = [...new Set(proposal.related_episode_ids)];
      const relatedEpisodeIds = proposedEpisodeIds.filter((id) => referencedEpisodeIdSet.has(id));
      const droppedEpisodeIds = proposedEpisodeIds.filter((id) => !referencedEpisodeIdSet.has(id));

      if (droppedEpisodeIds.length > 0) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflection_open_question_filtered_episode_ids",
          dropped_episode_ids: droppedEpisodeIds,
          kept_episode_ids: relatedEpisodeIds,
        });
      }

      const provenance =
        relatedEpisodeIds.length > 0
          ? {
              kind: "episodes" as const,
              episode_ids: relatedEpisodeIds,
            }
          : {
              kind: "online" as const,
              process: "reflector",
            };

      try {
        identityService.addOpenQuestion({
          question,
          urgency: proposal.urgency,
          audience_entity_id: context.audienceEntityId ?? null,
          related_episode_ids: relatedEpisodeIds,
          provenance,
          source: "reflection",
        });
      } catch (error) {
        await appendOpenQuestionHookFailureEvent(streamWriter, "reflection_open_question", error);
      }
    }
  }

  private async parseExecutiveReflectionItems(
    raw: RawReflectionOutput,
    streamWriter: StreamWriter,
  ): Promise<ReflectionOutput> {
    const stepOutcomes: ReflectionOutput["step_outcomes"] = [];
    const proposedSteps: ReflectionOutput["proposed_steps"] = [];

    for (const [index, item] of raw.step_outcomes.entries()) {
      const parsed = executiveStepOutcomeSchema.safeParse(item);

      if (!parsed.success) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_executive_item_dropped",
          reason: "malformed_step_outcome",
          index,
          error: parsed.error.message,
        });
        continue;
      }

      stepOutcomes.push(parsed.data);
    }

    for (const [index, item] of raw.proposed_steps.entries()) {
      const parsed = proposedExecutiveStepSchema.safeParse(item);

      if (!parsed.success) {
        await this.appendReflectorInternalEvent(streamWriter, {
          hook: "reflector_executive_item_dropped",
          reason: "malformed_proposed_step",
          index,
          error: parsed.error.message,
        });
        continue;
      }

      proposedSteps.push(parsed.data);
    }

    return {
      ...raw,
      step_outcomes: stepOutcomes,
      proposed_steps: proposedSteps,
    };
  }

  private async runReflectionJudgment(
    context: ReflectionContext,
    streamWriter: StreamWriter,
  ): Promise<ReflectionOutput> {
    const pendingProceduralAttempts = context.workingMemory.pending_procedural_attempts ?? [];
    const pendingIntents = context.workingMemory.pending_intents;
    const referencedEpisodeIds = selectReferencedRetrievedEpisodeIds(
      context.deliberationResult,
      context.retrievedEpisodes,
    );
    const isAutonomousTurn = context.origin === "autonomous";
    const hasUserVisibleTurnPayload =
      !isAutonomousTurn &&
      (context.userMessage.trim().length > 0 || context.actionResult.response.trim().length > 0);
    const executiveFocusForReflection = summarizeExecutiveFocusForReflection(
      context.executiveFocus,
    );
    const hasExecutiveWork = executiveFocusForReflection !== null;
    const hasReflectionWork =
      context.selfSnapshot.goals.length > 0 ||
      pendingProceduralAttempts.length > 0 ||
      pendingIntents.length > 0 ||
      referencedEpisodeIds.length > 0 ||
      hasExecutiveWork ||
      hasUserVisibleTurnPayload;

    if (
      this.llmClient === undefined ||
      this.model === undefined ||
      (isAutonomousTurn && !hasExecutiveWork) ||
      !hasReflectionWork
    ) {
      return emptyReflectionOutput();
    }

    const response = await this.llmClient.complete({
      model: this.model,
      system: [
        "You are Borg's post-turn reflector. Read the completed turn and active goals, then emit only the structured reflection tool.",
        "Mark advanced_goals only if the turn took a concrete step toward the goal, not just discussed it.",
        "Apply common-sense task linkage: when a turn describes the user completing a recognizable sub-task of an active goal, mark advanced_goals for that goal even if the user doesn't name the goal explicitly.",
        "For step_outcomes, update only executive steps the completed turn directly started, blocked, abandoned, or externally confirmed as done, and include concrete evidence.",
        "For autonomous turns, never mark an executive step done; autonomous turns may only start, block, or abandon a step.",
        "If executive_focus has a selected goal and next_step is null, proposed_steps may include a small concrete next step only when the completed turn revealed one for that selected goal. Otherwise omit proposed_steps.",
        "If pending_procedural_attempts has any entries, emit a procedural_outcome per attempt the current turn provides evidence about. Identify each by its attempt_turn_counter and classify success, failure, or unclear.",
        "Omit attempts the current turn says nothing about -- they will stay pending and may be graded on a later turn.",
        "For every procedural_outcome, set grounded=false when the evidence is assistant self-narration rather than an actual user signal.",
        "For every procedural_outcome, set skill_actually_applied=true only if the prior assistant response visibly executed the attempt's approach_summary. If the response ignored or substituted a different approach, set it false so the skill posterior is not credited or blamed for an outcome it didn't earn.",
        "Do not infer procedural success or failure from the assistant response, confidence, phrasing, or intentions.",
        "Emit trait_demonstrations only for traits actually shown by the completed assistant turn. Do not map from cognitive mode labels.",
        "Use strength_delta 0.01-0.1 for grounded trait demonstrations, and omit weak or generic traits.",
        "If pending_intents are present, mark only prior pending intents completed or abandoned when the current user message and agent response give clear evidence. Otherwise omit them.",
        "For open_questions, emit only questions the completed turn actually leaves unresolved and worth remembering. Retrieval confidence is context, not a trigger. Preserve the user's language in the question text.",
      ].join("\n"),
      messages: [
        {
          role: "user",
          content: JSON.stringify({
            user_message: context.userMessage,
            agent_response: context.actionResult.response,
            retrieval_confidence: context.retrievalConfidence,
            active_goals: context.selfSnapshot.goals.map((goal) => ({
              goal_id: goal.id,
              description: goal.description,
              progress_notes: goal.progress_notes,
            })),
            executive_focus: executiveFocusForReflection,
            origin: context.origin ?? "user",
            pending_procedural_attempts: pendingProceduralAttempts,
            pending_intents: pendingIntents,
            referenced_episodes: referencedEpisodeIds.map((episodeId) => {
              const result = context.retrievedEpisodes.find(
                (item) => item.episode.id === episodeId,
              );

              return {
                id: episodeId,
                title: result?.episode.title,
                narrative: result?.episode.narrative,
              };
            }),
          }),
        },
      ],
      tools: [REFLECTION_TOOL],
      tool_choice: { type: "tool", name: REFLECTION_TOOL_NAME },
      max_tokens: this.maxTokens,
      budget: "reflection",
    });
    const toolCall = response.tool_calls.find((call) => call.name === REFLECTION_TOOL_NAME);

    if (toolCall === undefined) {
      return emptyReflectionOutput();
    }

    const parsed = reflectionOutputParseSchema.safeParse(toolCall.input);

    if (!parsed.success) {
      throw parsed.error;
    }

    return this.parseExecutiveReflectionItems(parsed.data, streamWriter);
  }
}

function selectReferencedRetrievedEpisodeIds(
  deliberationResult: DeliberationResult,
  retrievedEpisodes: readonly RetrievedEpisode[],
): RetrievedEpisode["episode"]["id"][] {
  if (
    deliberationResult.referencedEpisodeIds === null ||
    deliberationResult.referencedEpisodeIds.length === 0
  ) {
    return [];
  }

  const retrievedIds = new Map(
    retrievedEpisodes.map((result) => [result.episode.id, result.episode.id]),
  );
  const selected: RetrievedEpisode["episode"]["id"][] = [];
  const seen = new Set<string>();

  for (const referencedId of deliberationResult.referencedEpisodeIds) {
    const retrievedId = retrievedIds.get(referencedId as RetrievedEpisode["episode"]["id"]);

    if (retrievedId === undefined || seen.has(retrievedId)) {
      continue;
    }

    seen.add(retrievedId);
    selected.push(retrievedId);
  }

  return selected;
}

function selectTraitDemonstration(
  demonstrations: readonly ReflectionOutput["trait_demonstrations"][number][],
): ReflectionOutput["trait_demonstrations"][number] | null {
  const ranked = demonstrations
    .map((demonstration) => ({
      ...demonstration,
      trait_label: demonstration.trait_label.trim(),
      evidence: demonstration.evidence.trim(),
    }))
    .filter(
      (demonstration) =>
        demonstration.trait_label.length > 0 &&
        demonstration.evidence.length > 0 &&
        demonstration.strength_delta > 0,
    )
    .sort((left, right) => right.strength_delta - left.strength_delta);

  return ranked[0] ?? null;
}
