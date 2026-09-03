import type { OpenQuestion, OpenQuestionsRepository } from "../../memory/self/index.js";
import {
  memoryDisclosurePayloadFields,
  openQuestionMemoryDisclosureLabel,
} from "../../memory/common/disclosure-serializers.js";
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
  // The offline rumination loop selects questions on its own criteria, and this
  // trigger neither feeds it nor reads its result. Carrying its bookkeeping here
  // is the only way the wake says whether the question it is handing over has
  // ever been worked offline -- otherwise a question no offline pass has reached
  // arrives looking exactly like one that has been ruminated a dozen times.
  unresolved_rumination_ticks: number;
  last_ruminated_at: number | null;
  related_episodes?: EpisodicSearchHit[];
} & ReturnType<typeof memoryDisclosurePayloadFields>;

export type OpenQuestionDormantTriggerOptions = {
  openQuestionsRepository: OpenQuestionsRepository;
  watermarkRepository: StreamWatermarkRepository;
  dormantMs: number;
  clock?: Clock;
  sessionId?: SessionId;
};

export function createOpenQuestionDormantTrigger(
  options: OpenQuestionDormantTriggerOptions,
): AutonomyTrigger<OpenQuestionDormantPayload> {
  const clock = options.clock ?? new SystemClock();
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;

  return {
    name: TRIGGER_NAME,
    type: "trigger",
    sourceCategory: "operational",
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
            stateTs: question.last_touched,
            payload: {
              open_question_id: question.id,
              question: question.question,
              urgency: question.urgency,
              last_touched: question.last_touched,
              unresolved_rumination_ticks: question.unresolved_rumination_ticks,
              last_ruminated_at: question.last_ruminated_at,
              ...memoryDisclosurePayloadFields(openQuestionMemoryDisclosureLabel(question)),
            },
          };
        })
        .filter((event): event is DueEvent<OpenQuestionDormantPayload> => event !== null);
    },
    async nextDueAt() {
      const nowMs = clock.now();
      const candidates = options.openQuestionsRepository
        .list({
          status: "open",
          limit: 10_000,
        })
        .flatMap((question): number[] => {
          const watermarkProcessName = `${WATERMARK_PREFIX}:${question.id}:${question.last_touched}`;

          if (options.watermarkRepository.get(watermarkProcessName, sessionId) !== null) {
            return [];
          }

          return [Math.max(question.last_touched + options.dormantMs + 1, nowMs)];
        });

      return candidates.length === 0 ? null : Math.min(...candidates);
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
