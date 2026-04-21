import type { RetrievedEpisode } from "../../retrieval/index.js";
import { StreamWriter } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { GoalsRepository, TraitsRepository, type GoalRecord } from "../../memory/self/index.js";
import { EpisodicRepository } from "../../memory/episodic/index.js";
import { pushRecentThought, type WorkingMemory } from "../../memory/working/index.js";
import { tokenizeText } from "../../util/text/tokenize.js";

import type { ActionResult } from "../action/index.js";
import type { DeliberationResult, SelfSnapshot } from "../deliberation/deliberator.js";
import { SuppressionSet } from "../attention/index.js";

export type ReflectionContext = {
  userMessage: string;
  workingMemory: WorkingMemory;
  selfSnapshot: SelfSnapshot;
  deliberationResult: DeliberationResult;
  actionResult: ActionResult;
  retrievedEpisodes: RetrievedEpisode[];
  episodicRepository: EpisodicRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  suppressionSet: SuppressionSet;
};

const TOKEN_STOPWORDS = ["the", "and", "with", "this", "that", "from", "into", "after", "before"];
const SURFACED_TTL_TURNS = 4;
const NOISE_TTL_TURNS = 2;

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

    context.suppressionSet.tickTurn();

    let nextWorkingMemory: WorkingMemory = {
      ...context.actionResult.workingMemory,
      scratchpad: "",
      updated_at: this.clock.now(),
    };

    for (const thought of context.deliberationResult.thoughts) {
      nextWorkingMemory = pushRecentThought(nextWorkingMemory, thought);
    }

    return nextWorkingMemory;
  }
}
