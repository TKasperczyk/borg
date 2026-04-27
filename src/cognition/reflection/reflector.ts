import type { RetrievalConfidence, RetrievedEpisode } from "../../retrieval/index.js";
import { type LLMClient, type LLMToolDefinition, toToolInputSchema } from "../../llm/index.js";
import { StreamWriter } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import {
  GoalsRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  type GoalRecord,
} from "../../memory/self/index.js";
import type { IdentityService } from "../../memory/identity/index.js";
import type { ReviewQueueRepository } from "../../memory/semantic/index.js";
import { ProceduralEvidenceRepository, SkillRepository } from "../../memory/procedural/index.js";
import {
  appendInternalFailureEvent,
  appendOpenQuestionHookFailureEvent,
} from "../../memory/self/review-open-question-hook.js";
import { EpisodicRepository } from "../../memory/episodic/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { EntityId, EpisodeId, SkillId, StreamEntryId } from "../../util/ids.js";
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
  episodicRepository: EpisodicRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  identityService?: Pick<IdentityService, "updateGoal">;
  reviewQueueRepository?: Pick<ReviewQueueRepository, "enqueue">;
  skillRepository?: SkillRepository;
  proceduralEvidenceRepository?: ProceduralEvidenceRepository;
  selectedSkillId?: SkillId | null;
  audienceEntityId?: EntityId | null;
  suppressionSet: SuppressionSet;
};

const SURFACED_TTL_TURNS = 4;
const NOISE_TTL_TURNS = 2;
// RetrievalConfidence is calibrated epistemic confidence, not the relevance
// ranking score. Keep this aligned with the S1/S2 low-confidence route.
const OPEN_QUESTION_CONFIDENCE_THRESHOLD = 0.45;
const REFLECTION_TOOL_NAME = "EmitTurnReflection";
const DEFAULT_REFLECTION_MAX_TOKENS = 768;

const traitDemonstrationSchema = z.object({
  trait_label: z.string().min(1),
  evidence: z.string().min(1),
  strength_delta: z.number().min(0).max(0.2),
});

const reflectionOutputSchema = z.object({
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
});

type ReflectionOutput = z.infer<typeof reflectionOutputSchema>;

const REFLECTION_TOOL: LLMToolDefinition = {
  name: REFLECTION_TOOL_NAME,
  description:
    "Emit structured post-turn reflection. Mark advanced_goals only for concrete progress, procedural_outcomes only from user follow-up evidence with grounded set explicitly, trait_demonstrations only from turn content, and intent_updates only for prior pending intents resolved by the completed turn.",
  inputSchema: toToolInputSchema(reflectionOutputSchema),
};

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

function buildReflectionQuestion(userMessage: string, entities: readonly string[]): string {
  const anchor = entities
    .map((entity) => entity.trim())
    .filter((entity) => entity.length > 0)
    .slice(0, 2)
    .join(" and ");
  const prompt = userMessage.trim().replace(/\s+/g, " ");

  if (anchor.length > 0) {
    return `What am I missing about ${anchor} in this situation: ${prompt}?`;
  }

  return `What am I missing here: ${prompt}?`;
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
  sourceStreamIds: readonly StreamEntryId[],
): Promise<EpisodeId[]> {
  const retrievedMatches = context.retrievedEpisodes
    .filter((result) =>
      sourceStreamIds.every((streamId) => result.episode.source_stream_ids.includes(streamId)),
    )
    .map((result) => result.episode.id);

  const exact = await context.episodicRepository.findBySourceStreamIds(sourceStreamIds);
  const exactIds = exact === null ? [] : [exact.id];

  return [...new Set([...retrievedMatches, ...exactIds])];
}

export type ReflectorOptions = {
  clock?: Clock;
  llmClient?: LLMClient;
  model?: string;
  maxTokens?: number;
};

export class Reflector {
  private readonly clock: Clock;
  private readonly llmClient?: LLMClient;
  private readonly model?: string;
  private readonly maxTokens: number;

  constructor(options: ReflectorOptions = {}) {
    this.clock = options.clock ?? new SystemClock();
    this.llmClient = options.llmClient;
    this.model = options.model;
    this.maxTokens = options.maxTokens ?? DEFAULT_REFLECTION_MAX_TOKENS;
  }

  async reflect(context: ReflectionContext, streamWriter: StreamWriter): Promise<WorkingMemory> {
    const reflectionProvenance = buildReflectionProvenance(context.retrievedEpisodes);
    let reflectionOutput: ReflectionOutput = {
      advanced_goals: [],
      procedural_outcomes: [],
      trait_demonstrations: [],
      intent_updates: [],
    };

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
      reflectionOutput = await this.runReflectionJudgment(context);
    } catch (error) {
      await appendInternalFailureEvent(streamWriter, "reflection_judgment", error);
    }

    const activeGoalsById = new Map(context.selfSnapshot.goals.map((goal) => [goal.id, goal]));

    for (const advancedGoal of reflectionOutput.advanced_goals) {
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

      if (context.identityService === undefined || context.reviewQueueRepository === undefined) {
        context.goalsRepository.update(goal.id, patch, reflectionProvenance);
        continue;
      }

      const result = context.identityService.updateGoal(goal.id, patch, reflectionProvenance);

      if (result.status === "requires_review") {
        context.reviewQueueRepository.enqueue({
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
        const stats = context.episodicRepository.getStats(result.episode.id);

        if (stats !== null) {
          context.episodicRepository.updateStats(result.episode.id, {
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

    if (
      context.deliberationResult.path === "system_2" &&
      context.retrievalConfidence.overall < OPEN_QUESTION_CONFIDENCE_THRESHOLD
    ) {
      try {
        const relatedEpisodeIds =
          reflectionProvenance.kind === "episodes" ? reflectionProvenance.episode_ids : [];

        context.openQuestionsRepository.add({
          question: buildReflectionQuestion(
            context.userMessage,
            context.workingMemory.hot_entities,
          ),
          urgency: 0.45,
          related_episode_ids: relatedEpisodeIds,
          provenance: reflectionProvenance,
          source: "reflection",
        });
      } catch (error) {
        await appendOpenQuestionHookFailureEvent(streamWriter, "reflection_open_question", error);
      }
    }

    context.suppressionSet.tickTurn();

    let nextWorkingMemory: WorkingMemory = {
      ...context.actionResult.workingMemory,
      updated_at: this.clock.now(),
    };
    const resolvedIntentKeys = selectResolvedIntentKeys(
      context.workingMemory.pending_intents,
      reflectionOutput.intent_updates,
    );

    if (resolvedIntentKeys.size > 0) {
      nextWorkingMemory = {
        ...nextWorkingMemory,
        pending_intents: nextWorkingMemory.pending_intents.filter(
          (intent) => !resolvedIntentKeys.has(intentKey(intent)),
        ),
      };
    }

    const traitEvidenceEpisodeIds = referencedEpisodeIds;
    const traitDemonstration = selectTraitDemonstration(reflectionOutput.trait_demonstrations);

    if (
      // Lagged trait attribution must only come from user-visible turns.
      // Autonomous/self-talk turns are not shown to the user, so a later
      // user reply must not be treated as evidence about them.
      context.origin !== "autonomous" &&
      nextWorkingMemory.pending_trait_attribution === null &&
      traitDemonstration !== null &&
      traitEvidenceEpisodeIds.length > 0
    ) {
      nextWorkingMemory = {
        ...nextWorkingMemory,
        pending_trait_attribution: {
          trait_label: traitDemonstration.trait_label,
          strength_delta: traitDemonstration.strength_delta,
          source_episode_ids: traitEvidenceEpisodeIds,
          turn_completed_ts: this.clock.now(),
          audience_entity_id: context.audienceEntityId ?? null,
        },
      };
    }

    const pendingProceduralAttempts = context.workingMemory.pending_procedural_attempts ?? [];

    if (pendingProceduralAttempts.length > 0) {
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
          if (context.proceduralEvidenceRepository !== undefined) {
            const resolvedEpisodeIds = await resolveAttemptEpisodeIds(
              context,
              attempt.source_stream_ids,
            );

            context.proceduralEvidenceRepository.insert({
              pendingAttemptSnapshot: attempt,
              classification: outcome.classification,
              evidenceText: outcome.evidence,
              grounded: true,
              resolvedEpisodeIds,
              audienceEntityId: attempt.audience_entity_id,
            });

            if (
              attempt.selected_skill_id !== null &&
              (outcome.classification === "success" ||
                outcome.classification === "failure") &&
              context.skillRepository !== undefined
            ) {
              context.skillRepository.recordOutcome(
                attempt.selected_skill_id,
                outcome.classification === "success",
                resolvedEpisodeIds,
              );
            }
          }

          // Only actionable signals retire the attempt; an "unclear" but
          // grounded outcome stays pending in case later turns clarify.
          if (
            outcome.classification === "success" ||
            outcome.classification === "failure"
          ) {
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
        last_selected_skill_id: null,
        last_selected_skill_turn: null,
      };
    }

    return nextWorkingMemory;
  }

  private async runReflectionJudgment(context: ReflectionContext): Promise<ReflectionOutput> {
    const pendingProceduralAttempts =
      context.workingMemory.pending_procedural_attempts ?? [];
    const pendingIntents = context.workingMemory.pending_intents;

    if (
      this.llmClient === undefined ||
      this.model === undefined ||
      (context.selfSnapshot.goals.length === 0 &&
        pendingProceduralAttempts.length === 0 &&
        pendingIntents.length === 0 &&
        selectReferencedRetrievedEpisodeIds(context.deliberationResult, context.retrievedEpisodes)
          .length === 0)
    ) {
      return {
        advanced_goals: [],
        procedural_outcomes: [],
        trait_demonstrations: [],
        intent_updates: [],
      };
    }

    const response = await this.llmClient.complete({
      model: this.model,
      system: [
        "You are Borg's post-turn reflector. Read the completed turn and active goals, then emit only the structured reflection tool.",
        "Mark advanced_goals only if the turn took a concrete step toward the goal, not just discussed it.",
        "If pending_procedural_attempts has any entries, emit a procedural_outcome per attempt the current turn provides evidence about. Identify each by its attempt_turn_counter and classify success, failure, or unclear.",
        "Omit attempts the current turn says nothing about -- they will stay pending and may be graded on a later turn.",
        "For every procedural_outcome, set grounded=false when the evidence is assistant self-narration rather than an actual user signal.",
        "Do not infer procedural success or failure from the assistant response, confidence, phrasing, or intentions.",
        "Emit trait_demonstrations only for traits actually shown by the completed assistant turn. Do not map from cognitive mode labels.",
        "Use strength_delta 0.01-0.1 for grounded trait demonstrations, and omit weak or generic traits.",
        "If pending_intents are present, mark only prior pending intents completed or abandoned when the current user message and agent response give clear evidence. Otherwise omit them.",
      ].join("\n"),
      messages: [
        {
          role: "user",
          content: JSON.stringify({
            user_message: context.userMessage,
            agent_response: context.actionResult.response,
            active_goals: context.selfSnapshot.goals.map((goal) => ({
              goal_id: goal.id,
              description: goal.description,
              progress_notes: goal.progress_notes,
            })),
            pending_procedural_attempts: pendingProceduralAttempts,
            pending_intents: pendingIntents,
            referenced_episodes: selectReferencedRetrievedEpisodeIds(
              context.deliberationResult,
              context.retrievedEpisodes,
            ).map((episodeId) => {
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
      return {
        advanced_goals: [],
        procedural_outcomes: [],
        trait_demonstrations: [],
        intent_updates: [],
      };
    }

    const parsed = reflectionOutputSchema.safeParse(toolCall.input);

    if (!parsed.success) {
      throw parsed.error;
    }

    return parsed.data;
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
