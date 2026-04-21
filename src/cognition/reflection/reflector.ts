import type { RetrievedEpisode } from "../../retrieval/index.js";
import { StreamWriter } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import {
  GoalsRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  type GoalRecord,
} from "../../memory/self/index.js";
import { SkillRepository } from "../../memory/procedural/index.js";
import {
  appendInternalFailureEvent,
  appendOpenQuestionHookFailureEvent,
} from "../../memory/self/review-open-question-hook.js";
import { EpisodicRepository } from "../../memory/episodic/index.js";
import { pushRecentThought, type WorkingMemory } from "../../memory/working/index.js";
import { tokenizeText } from "../../util/text/tokenize.js";
import type { SkillId } from "../../util/ids.js";

import type { ActionResult } from "../action/index.js";
import type { DeliberationResult, SelfSnapshot } from "../deliberation/deliberator.js";
import { SuppressionSet } from "../attention/index.js";
import type { PerceptionResult } from "../types.js";

export type ReflectionContext = {
  userMessage: string;
  perception?: PerceptionResult;
  workingMemory: WorkingMemory;
  selfSnapshot: SelfSnapshot;
  deliberationResult: DeliberationResult;
  actionResult: ActionResult;
  retrievedEpisodes: RetrievedEpisode[];
  episodicRepository: EpisodicRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  skillRepository?: SkillRepository;
  selectedSkillId?: SkillId | null;
  suppressionSet: SuppressionSet;
};

const TOKEN_STOPWORDS = ["the", "and", "with", "this", "that", "from", "into", "after", "before"];
const SURFACED_TTL_TURNS = 4;
const NOISE_TTL_TURNS = 2;
const OPEN_QUESTION_CONFIDENCE_THRESHOLD = 0.4;
const SKILL_OUTCOME_WINDOW_TURNS = 2;
const EXPLICIT_SUCCESS_MARKERS = [
  /\bit works\b/i,
  /\bit worked\b/i,
  /\bworks now\b/i,
  /\bfixed\b/i,
  /\bsolved\b/i,
  /\bthanks,\s*that did it\b/i,
  /\bperfect\b/i,
  /\bgreat,\s*that fixed it\b/i,
  /^[\s]*[✅✔]/u,
] as const;
const EXPLICIT_FAILURE_MARKERS = [
  /\bdidn'?t work\b/i,
  /\bstill broken\b/i,
  /\bsame error\b/i,
  /\bthat didn'?t help\b/i,
  /\bno,\s*still\b/i,
] as const;

function goalMentioned(goal: GoalRecord, userMessage: string, response: string): boolean {
  const goalTokens = tokenizeText(goal.description);
  const userTokens = tokenizeText(userMessage);
  const responseTokens = tokenizeText(response);
  const inUser = [...goalTokens].some((token) => userTokens.has(token));
  const inResponse = [...goalTokens].some((token) => responseTokens.has(token));

  return inUser && inResponse;
}

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

function averageRetrievalConfidence(results: readonly RetrievedEpisode[]): number {
  if (results.length === 0) {
    return 0;
  }

  return results.reduce((sum, result) => sum + result.score, 0) / results.length;
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

function inferSkillOutcomeFromUserFollowUp(userMessage: string): boolean | null {
  if (EXPLICIT_SUCCESS_MARKERS.some((pattern) => pattern.test(userMessage))) {
    return true;
  }

  if (EXPLICIT_FAILURE_MARKERS.some((pattern) => pattern.test(userMessage))) {
    return false;
  }

  return null;
}

export type ReflectorOptions = {
  clock?: Clock;
};

export class Reflector {
  private readonly clock: Clock;

  constructor(options: ReflectorOptions = {}) {
    this.clock = options.clock ?? new SystemClock();
  }

  async reflect(context: ReflectionContext, streamWriter: StreamWriter): Promise<WorkingMemory> {
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

    context.traitsRepository.reinforce("engaged", 0.05, this.clock.now());

    for (const goal of context.selfSnapshot.goals) {
      if (!goalMentioned(goal, context.userMessage, context.actionResult.response)) {
        continue;
      }

      const note = `[${this.clock.now()}] Heuristic turn progress from response overlap`;
      const nextProgress = goal.progress_notes === null ? note : `${goal.progress_notes}\n${note}`;
      context.goalsRepository.updateProgress(goal.id, nextProgress);
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
      averageRetrievalConfidence(context.retrievedEpisodes) < OPEN_QUESTION_CONFIDENCE_THRESHOLD
    ) {
      try {
        context.openQuestionsRepository.add({
          question: buildReflectionQuestion(
            context.userMessage,
            context.workingMemory.hot_entities,
          ),
          urgency: 0.45,
          related_episode_ids: context.retrievedEpisodes
            .slice(0, 3)
            .map((result) => result.episode.id),
          source: "reflection",
        });
      } catch (error) {
        await appendOpenQuestionHookFailureEvent(streamWriter, "reflection_open_question", error);
      }
    }

    context.suppressionSet.tickTurn();

    let nextWorkingMemory: WorkingMemory = {
      ...context.actionResult.workingMemory,
      scratchpad: "",
      updated_at: this.clock.now(),
    };

    for (const thought of context.deliberationResult.thoughts) {
      nextWorkingMemory = pushRecentThought(nextWorkingMemory, thought);
    }

    const carrySkillId = context.workingMemory.last_selected_skill_id;
    const carrySkillTurn = context.workingMemory.last_selected_skill_turn;
    const currentTurn = nextWorkingMemory.turn_counter;
    let consumedCarryOutcome = false;

    if (
      carrySkillId !== null &&
      carrySkillTurn !== null &&
      currentTurn - carrySkillTurn <= SKILL_OUTCOME_WINDOW_TURNS
    ) {
      const carryOutcome = inferSkillOutcomeFromUserFollowUp(context.userMessage);

      if (carryOutcome !== null && context.skillRepository !== undefined) {
        try {
          context.skillRepository.recordOutcome(carrySkillId, carryOutcome);
          consumedCarryOutcome = true;
          nextWorkingMemory = {
            ...nextWorkingMemory,
            last_selected_skill_id: null,
            last_selected_skill_turn: null,
          };
        } catch (error) {
          await appendInternalFailureEvent(streamWriter, "skill_outcome_record", error);
        }
      }
    } else if (carrySkillId !== null && carrySkillTurn !== null) {
      nextWorkingMemory = {
        ...nextWorkingMemory,
        last_selected_skill_id: null,
        last_selected_skill_turn: null,
      };
    }

    if (context.selectedSkillId !== null && context.selectedSkillId !== undefined) {
      if (consumedCarryOutcome) {
        return {
          ...nextWorkingMemory,
          last_selected_skill_id: null,
          last_selected_skill_turn: null,
        };
      }

      // Outcome attribution is follow-up-only to avoid treating the current problem statement
      // or the assistant's own wording as evidence that a suggested skill succeeded or failed.
      nextWorkingMemory = {
        ...nextWorkingMemory,
        last_selected_skill_id: context.selectedSkillId,
        last_selected_skill_turn: currentTurn,
      };
    }

    return nextWorkingMemory;
  }
}
