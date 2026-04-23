import type { OpenQuestion, OpenQuestionsRepository } from "../../memory/self/index.js";
import type { StreamWatermarkRepository } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../../util/ids.js";
import type { AutonomyTrigger, DueEvent } from "../types.js";

const TRIGGER_NAME = "open_question_dormant" as const;
const WATERMARK_PREFIX = "autonomy:open-question-dormant";

type EpisodicSearchHit = {
  id: string;
  title: string;
  score: number;
};

type OpenQuestionDormantPayload = {
  open_question_id: OpenQuestion["id"];
  question: string;
  urgency: number;
  last_touched: number;
  related_episodes?: EpisodicSearchHit[];
};

export type OpenQuestionDormantTriggerOptions = {
  openQuestionsRepository: OpenQuestionsRepository;
  watermarkRepository: StreamWatermarkRepository;
  dormantMs: number;
  clock?: Clock;
  sessionId?: SessionId;
};

function formatEpisodes(episodes: readonly EpisodicSearchHit[] | undefined): string {
  if (episodes === undefined || episodes.length === 0) {
    return "Fresh episodic context: none found.";
  }

  return [
    "Fresh episodic context:",
    ...episodes.map((episode) => `- ${episode.title} (score ${episode.score.toFixed(2)})`),
  ].join("\n");
}

export function createOpenQuestionDormantTrigger(
  options: OpenQuestionDormantTriggerOptions,
): AutonomyTrigger<OpenQuestionDormantPayload> {
  const clock = options.clock ?? new SystemClock();
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;

  return {
    name: TRIGGER_NAME,
    type: "trigger",
    async scan() {
      const nowMs = clock.now();
      const openQuestions = options.openQuestionsRepository
        .list({
          status: "open",
          limit: 10_000,
        })
        .filter((question) => question.last_touched + options.dormantMs < nowMs)
        .sort((left, right) => left.last_touched - right.last_touched);

      return openQuestions
        .map<DueEvent<OpenQuestionDormantPayload> | null>((question) => {
          const watermarkProcessName = `${WATERMARK_PREFIX}:${question.id}:${question.last_touched}`;

          if (options.watermarkRepository.get(watermarkProcessName, sessionId) !== null) {
            return null;
          }

          return {
            id: `${question.id}:${question.last_touched}`,
            sourceName: TRIGGER_NAME,
            sourceType: "trigger",
            watermarkProcessName,
            sortTs: question.last_touched,
            payload: {
              open_question_id: question.id,
              question: question.question,
              urgency: question.urgency,
              last_touched: question.last_touched,
            },
          };
        })
        .filter((event): event is DueEvent<OpenQuestionDormantPayload> => event !== null);
    },
    buildTurn(event) {
      return {
        audience: "self",
        stakes: "low",
        userMessage: [
          "An open question has gone dormant.",
          `Question: ${event.payload.question}`,
          `Urgency: ${event.payload.urgency.toFixed(2)}`,
          `Last touched: ${new Date(event.payload.last_touched).toISOString()}`,
          formatEpisodes(event.payload.related_episodes),
          "Review whether anything new has emerged or what follow-up should happen next.",
        ].join("\n"),
      };
    },
  };
}
