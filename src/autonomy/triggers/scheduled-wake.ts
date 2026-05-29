import { SystemClock, type Clock } from "../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  parseScheduledWakeId,
  type ScheduledWakeId,
  type SessionId,
} from "../../util/ids.js";
import type { StreamWatermarkRepository } from "../../stream/index.js";
import type { ScheduledWakesRepository } from "../scheduled-wakes-repository.js";
import type { AutonomyTrigger, DueEvent } from "../types.js";

const TRIGGER_NAME = "scheduled_wake" as const;
const WATERMARK_PREFIX = "autonomy:scheduled-wake";

type ScheduledWakePayload = {
  note: string;
  scheduled_at: number;
  fire_at: number;
};

export type ScheduledWakeTriggerOptions = {
  scheduledWakesRepository: ScheduledWakesRepository;
  watermarkRepository: StreamWatermarkRepository;
  clock?: Clock;
  sessionId?: SessionId;
};

export function createScheduledWakeTrigger(
  options: ScheduledWakeTriggerOptions,
): AutonomyTrigger<ScheduledWakePayload> {
  const clock = options.clock ?? new SystemClock();
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;

  return {
    name: TRIGGER_NAME,
    type: "trigger",
    async scan() {
      const nowMs = clock.now();
      const duePending = options.scheduledWakesRepository.listDuePending(nowMs);
      const dueEvents: DueEvent<ScheduledWakePayload>[] = [];
      const alreadyFiredIds: ScheduledWakeId[] = [];

      for (const wake of duePending) {
        const watermarkProcessName = `${WATERMARK_PREFIX}:${wake.id}`;

        // A watermark means the scheduler already ran this wake successfully.
        // Reconcile the row to `fired` so it leaves the pending working set and
        // is never emitted twice (one-time semantics).
        if (options.watermarkRepository.get(watermarkProcessName, sessionId) !== null) {
          alreadyFiredIds.push(wake.id);
          continue;
        }

        dueEvents.push({
          id: wake.id,
          sourceName: TRIGGER_NAME,
          sourceType: "trigger",
          watermarkProcessName,
          sortTs: wake.fire_at,
          payload: {
            note: wake.note,
            scheduled_at: wake.created_at,
            fire_at: wake.fire_at,
          },
        });
      }

      if (alreadyFiredIds.length > 0) {
        options.scheduledWakesRepository.markFired(alreadyFiredIds, nowMs);
      }

      return dueEvents;
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
    onFired(event) {
      // Make the row authoritative at fire-time so list/cancel reflect the fire
      // immediately, not one scan later. scan()'s watermark reconcile remains a
      // backstop for the crash-between-watermark-and-onFired case.
      options.scheduledWakesRepository.markFired([parseScheduledWakeId(event.id)], clock.now());
    },
  };
}
