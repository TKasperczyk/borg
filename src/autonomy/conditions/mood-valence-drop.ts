import type { MoodRepository } from "../../memory/affective/index.js";
import type { StreamWatermarkRepository } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../../util/ids.js";
import type { AutonomyCondition, DueEvent } from "../types.js";

const CONDITION_NAME = "mood_valence_drop" as const;
const WATERMARK_PREFIX = "autonomy:mood-valence-drop";
const DEFAULT_ACTIVATION_PERIOD_MS = 24 * 60 * 60 * 1_000;

export type MoodValenceDropPayload = {
  session_id: SessionId;
  average_valence: number;
  threshold: number;
  window_n: number;
  latest_ts: number;
};

export type MoodValenceDropConditionOptions = {
  moodRepository: MoodRepository;
  watermarkRepository: StreamWatermarkRepository;
  threshold: number;
  windowN: number;
  activationPeriodMs?: number;
  clock?: Clock;
  sessionId?: SessionId;
};

export function createMoodValenceDropCondition(
  options: MoodValenceDropConditionOptions,
): AutonomyCondition<MoodValenceDropPayload> {
  const clock = options.clock ?? new SystemClock();
  // Single-session scan. `mood_history` is keyed per session, and the wiring in
  // borg/autonomy-setup.ts passes no `sessionId`, so this falls to
  // DEFAULT_SESSION_ID -- a session that carries no conversational traffic in a
  // multi-session deployment. Conversational sessions can then run deeply
  // negative for days without this condition seeing a single row, and the
  // `history.length < windowN` guard below turns that into a silent no-op
  // rather than an error. Enabling the condition is not enough to make it
  // observe anything; it also has to be pointed at the sessions that have mood.
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;
  const activationPeriodMs = options.activationPeriodMs ?? DEFAULT_ACTIVATION_PERIOD_MS;

  return {
    name: CONDITION_NAME,
    type: "condition",
    sourceCategory: "operational",
    async scan() {
      const history = options.moodRepository.history(sessionId, {
        limit: options.windowN,
      });

      if (history.length < options.windowN) {
        return [];
      }

      const averageValence =
        history.reduce((sum, entry) => sum + entry.valence, 0) / history.length;

      if (averageValence >= options.threshold) {
        return [];
      }

      const watermarkProcessName = `${WATERMARK_PREFIX}:${sessionId}`;
      const watermark = options.watermarkRepository.get(watermarkProcessName, sessionId);
      const nowMs = clock.now();

      if (watermark !== null && nowMs - watermark.lastTs < activationPeriodMs) {
        return [];
      }

      return [
        {
          id: `${sessionId}:${history[0]!.ts}`,
          sourceName: CONDITION_NAME,
          sourceType: "condition",
          watermarkProcessName,
          sortTs: history[0]!.ts,
          stateTs: history[0]!.ts,
          payload: {
            session_id: sessionId,
            average_valence: averageValence,
            threshold: options.threshold,
            window_n: options.windowN,
            latest_ts: history[0]!.ts,
          },
        } satisfies DueEvent<MoodValenceDropPayload>,
      ];
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
