import type { OpenQuestion, OpenQuestionsRepository } from "../../memory/self/index.js";
import type { StreamWatermarkRepository } from "../../stream/index.js";
import type { Clock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../../util/ids.js";
import type { AutonomyCondition, DueEvent } from "../types.js";

const CONDITION_NAME = "open_question_urgency_bump" as const;
const WATERMARK_PREFIX = "autonomy:open-question-urgency";

export type OpenQuestionUrgencyBumpPayload = {
  open_question_id: OpenQuestion["id"];
  question: string;
  urgency: number;
};

export type OpenQuestionUrgencyBumpConditionOptions = {
  openQuestionsRepository: OpenQuestionsRepository;
  watermarkRepository: StreamWatermarkRepository;
  threshold: number;
  clock?: Clock;
  sessionId?: SessionId;
};

function urgencyFloor(value: number): string {
  return value.toFixed(2);
}

export function createOpenQuestionUrgencyBumpCondition(
  options: OpenQuestionUrgencyBumpConditionOptions,
): AutonomyCondition<OpenQuestionUrgencyBumpPayload> {
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;

  return {
    name: CONDITION_NAME,
    type: "condition",
    async scan() {
      return options.openQuestionsRepository
        .list({
          status: "open",
          minUrgency: options.threshold,
          limit: 10_000,
        })
        .sort((left, right) => left.last_touched - right.last_touched)
        .map<DueEvent<OpenQuestionUrgencyBumpPayload> | null>((question) => {
          const watermarkProcessName = `${WATERMARK_PREFIX}:${question.id}:${urgencyFloor(question.urgency)}`;

          if (options.watermarkRepository.get(watermarkProcessName, sessionId) !== null) {
            return null;
          }

          return {
            id: `${question.id}:${urgencyFloor(question.urgency)}`,
            sourceName: CONDITION_NAME,
            sourceType: "condition",
            watermarkProcessName,
            sortTs: question.last_touched,
            payload: {
              open_question_id: question.id,
              question: question.question,
              urgency: question.urgency,
            },
          };
        })
        .filter((event): event is DueEvent<OpenQuestionUrgencyBumpPayload> => event !== null);
    },
    buildTurn(event) {
      return {
        audience: "self",
        stakes: "low",
        userMessage: "",
        autonomyTrigger: {
          source_name: event.sourceName,
          source_type: event.sourceType,
          event_id: event.id,
          sort_ts: event.sortTs,
          payload: event.payload,
        },
      };
    },
  };
}
