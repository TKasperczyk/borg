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
import {
  isProceduralOutcomeEvidenceGrounded,
  ProceduralEvidenceRepository,
  SkillRepository,
} from "../../memory/procedural/index.js";
import {
  appendInternalFailureEvent,
  appendOpenQuestionHookFailureEvent,
} from "../../memory/self/review-open-question-hook.js";
import { EpisodicRepository } from "../../memory/episodic/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import { tokenizeText } from "../../util/text/tokenize.js";
import type { EntityId, EpisodeId, SkillId, StreamEntryId } from "../../util/ids.js";
import { z } from "zod";

import type { ActionResult } from "../action/index.js";
import type { DeliberationResult, SelfSnapshot } from "../deliberation/deliberator.js";
import { SuppressionSet } from "../attention/index.js";
import { intentRecordSchema, type IntentRecord, type PerceptionResult } from "../types.js";
import { MODE_TRAIT_MAP } from "./trait-signals.js";

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

const TOKEN_STOPWORDS = ["the", "and", "with", "this", "that", "from", "into", "after", "before"];
const SURFACED_TTL_TURNS = 4;
const NOISE_TTL_TURNS = 2;
// RetrievalConfidence is calibrated epistemic confidence, not the relevance
// ranking score. Keep this aligned with the S1/S2 low-confidence route.
const OPEN_QUESTION_CONFIDENCE_THRESHOLD = 0.45;
const REFLECTION_TOOL_NAME = "EmitTurnReflection";
const DEFAULT_REFLECTION_MAX_TOKENS = 768;

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
        classification: z.enum(["success", "failure", "unclear"]),
        evidence: z.string().min(1),
      }),
    )
    .describe(
      "Outcomes for the prior pending_procedural_attempt. Judge success only from the user's follow-up signal, never from the assistant's wording.",
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
    "Emit structured post-turn reflection. Mark advanced_goals only if the turn took a concrete step toward the goal, procedural_outcomes only from user follow-up evidence, and intent_updates only for prior pending intents resolved by the completed turn.",
  inputSchema: toToolInputSchema(reflectionOutputSchema),
};

function episodeUsed(result: RetrievedEpisode, response: string): boolean {
  const normalized = response.toLowerCase();
  const normalizedTitle = result.episode.title.trim().toLowerCase();

  if (normalizedTitle.length > 0 && normalized.includes(normalizedTitle)) {
    return true;
  }

  if (
    result.episode.tags.some((tag) => normalized.includes(tag.toLowerCase())) ||
    result.episode.participants.some((participant) =>
      normalized.includes(participant.toLowerCase()),
    )
  ) {
    return true;
  }

  const responseTokens = tokenizeText(response, {
    stopwords: TOKEN_STOPWORDS,
  });
  const episodeTokens = tokenizeText(
    [
      result.episode.title,
      result.episode.narrative,
      result.episode.tags.join(" "),
      result.episode.participants.join(" "),
    ].join(" "),
    {
      stopwords: TOKEN_STOPWORDS,
    },
  );

  if (responseTokens.size === 0 || episodeTokens.size === 0) {
    return false;
  }

  let overlap = 0;

  for (const token of responseTokens) {
    if (episodeTokens.has(token)) {
      overlap += 1;
    }
  }

  const unionSize = new Set([...responseTokens, ...episodeTokens]).size;
  return unionSize > 0 && overlap / unionSize >= 0.15;
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
    const effectiveMode = context.perception?.mode ?? context.workingMemory.mode ?? "idle";
    const reflectedTrait = MODE_TRAIT_MAP[effectiveMode];
    let reflectionOutput: ReflectionOutput = {
      advanced_goals: [],
      procedural_outcomes: [],
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

    for (const result of context.retrievedEpisodes) {
      const used = episodeUsed(result, context.actionResult.response);

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

    const traitEvidenceEpisodeIds = selectTraitEvidenceEpisodeIds(
      context.deliberationResult,
      context.retrievedEpisodes,
    );

    if (
      // Lagged trait attribution must only come from user-visible turns.
      // Autonomous/self-talk turns are not shown to the user, so a later
      // user reply must not be treated as evidence about them.
      context.origin !== "autonomous" &&
      nextWorkingMemory.pending_trait_attribution === null &&
      reflectedTrait !== null &&
      traitEvidenceEpisodeIds.length > 0
    ) {
      nextWorkingMemory = {
        ...nextWorkingMemory,
        pending_trait_attribution: {
          trait_label: reflectedTrait,
          source_episode_ids: traitEvidenceEpisodeIds,
          turn_completed_ts: this.clock.now(),
          audience_entity_id: context.audienceEntityId ?? null,
        },
      };
    }

    const pendingProceduralAttempt = context.workingMemory.pending_procedural_attempt;
    const proceduralOutcome = reflectionOutput.procedural_outcomes[0];

    if (pendingProceduralAttempt !== null) {
      try {
        if (proceduralOutcome !== undefined && context.proceduralEvidenceRepository !== undefined) {
          const grounded = isProceduralOutcomeEvidenceGrounded({
            classification: proceduralOutcome.classification,
            evidence_text: proceduralOutcome.evidence,
          });
          const resolvedEpisodeIds = grounded
            ? await resolveAttemptEpisodeIds(context, pendingProceduralAttempt.source_stream_ids)
            : [];

          if (grounded) {
            context.proceduralEvidenceRepository.insert({
              pendingAttemptSnapshot: pendingProceduralAttempt,
              classification: proceduralOutcome.classification,
              evidenceText: proceduralOutcome.evidence,
              resolvedEpisodeIds,
              audienceEntityId: pendingProceduralAttempt.audience_entity_id,
            });

            if (
              pendingProceduralAttempt.selected_skill_id !== null &&
              (proceduralOutcome.classification === "success" ||
                proceduralOutcome.classification === "failure") &&
              context.skillRepository !== undefined
            ) {
              context.skillRepository.recordOutcome(
                pendingProceduralAttempt.selected_skill_id,
                proceduralOutcome.classification === "success",
                resolvedEpisodeIds,
              );
            }
          }
        }
      } catch (error) {
        await appendInternalFailureEvent(streamWriter, "procedural_evidence_record", error);
      }

      nextWorkingMemory = {
        ...nextWorkingMemory,
        pending_procedural_attempt: null,
        last_selected_skill_id: null,
        last_selected_skill_turn: null,
      };
    }

    return nextWorkingMemory;
  }

  private async runReflectionJudgment(context: ReflectionContext): Promise<ReflectionOutput> {
    const pendingProceduralAttempt = context.workingMemory.pending_procedural_attempt;
    const pendingIntents = context.workingMemory.pending_intents;

    if (
      this.llmClient === undefined ||
      this.model === undefined ||
      (context.selfSnapshot.goals.length === 0 &&
        pendingProceduralAttempt === null &&
        pendingIntents.length === 0)
    ) {
      return {
        advanced_goals: [],
        procedural_outcomes: [],
        intent_updates: [],
      };
    }

    const response = await this.llmClient.complete({
      model: this.model,
      system: [
        "You are Borg's post-turn reflector. Read the completed turn and active goals, then emit only the structured reflection tool.",
        "Mark advanced_goals only if the turn took a concrete step toward the goal, not just discussed it.",
        "If pending_procedural_attempt is present, classify whether the user's current message shows that attempt succeeded, failed, or remains unclear.",
        "Do not infer procedural success or failure from the assistant response, confidence, phrasing, or intentions.",
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
            pending_procedural_attempt: pendingProceduralAttempt,
            pending_intents: pendingIntents,
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

function selectTraitEvidenceEpisodeIds(
  deliberationResult: DeliberationResult,
  retrievedEpisodes: readonly RetrievedEpisode[],
): RetrievedEpisode["episode"]["id"][] {
  if (
    deliberationResult.path !== "system_2" ||
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
