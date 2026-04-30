import type { IdentityEvent } from "../../memory/identity/index.js";
import type { StreamWatermarkRepository } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../../util/ids.js";
import type { AutonomyTrigger, DueEvent } from "../types.js";

const TRIGGER_NAME = "scheduled_reflection" as const;
const WATERMARK_PROCESS_NAME = "autonomy:scheduled-reflection";

type ScheduledReflectionPayload = {
  interval_ms: number;
  recent_identity_events?: IdentityEvent[];
};

export type ScheduledReflectionTriggerOptions = {
  watermarkRepository: StreamWatermarkRepository;
  intervalMs: number;
  clock?: Clock;
  sessionId?: SessionId;
};

export function createScheduledReflectionTrigger(
  options: ScheduledReflectionTriggerOptions,
): AutonomyTrigger<ScheduledReflectionPayload> {
  const clock = options.clock ?? new SystemClock();
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;
  const initialWindowStartMs = clock.now();

  return {
    name: TRIGGER_NAME,
    type: "trigger",
    async scan() {
      const nowMs = clock.now();
      const watermark = options.watermarkRepository.get(WATERMARK_PROCESS_NAME, sessionId);

      if (watermark !== null && nowMs - watermark.lastTs < options.intervalMs) {
        return [];
      }

      const dueWindowStartMs =
        watermark === null
          ? initialWindowStartMs +
            Math.floor(Math.max(0, nowMs - initialWindowStartMs) / options.intervalMs) *
              options.intervalMs
          : watermark.lastTs +
            options.intervalMs +
            Math.floor(
              Math.max(0, nowMs - (watermark.lastTs + options.intervalMs)) / options.intervalMs,
            ) *
              options.intervalMs;

      return [
        {
          id: `scheduled-reflection:${dueWindowStartMs}`,
          sourceName: TRIGGER_NAME,
          sourceType: "trigger",
          watermarkProcessName: WATERMARK_PROCESS_NAME,
          sortTs: dueWindowStartMs,
          payload: {
            interval_ms: options.intervalMs,
          },
        } satisfies DueEvent<ScheduledReflectionPayload>,
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
