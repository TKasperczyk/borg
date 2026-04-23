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

function formatIdentityEvents(events: readonly IdentityEvent[] | undefined): string {
  if (events === undefined || events.length === 0) {
    return "Recent identity changes: none.";
  }

  return [
    "Recent identity changes:",
    ...events.map(
      (event) =>
        `- [${event.record_type}] ${event.action} ${event.record_id} at ${new Date(event.ts).toISOString()}`,
    ),
  ].join("\n");
}

export function createScheduledReflectionTrigger(
  options: ScheduledReflectionTriggerOptions,
): AutonomyTrigger<ScheduledReflectionPayload> {
  const clock = options.clock ?? new SystemClock();
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;

  return {
    name: TRIGGER_NAME,
    type: "trigger",
    async scan() {
      const nowMs = clock.now();
      const watermark = options.watermarkRepository.get(WATERMARK_PROCESS_NAME, sessionId);

      if (watermark !== null && nowMs - watermark.lastTs < options.intervalMs) {
        return [];
      }

      return [
        {
          id: `scheduled-reflection:${nowMs}`,
          sourceName: TRIGGER_NAME,
          sourceType: "trigger",
          watermarkProcessName: WATERMARK_PROCESS_NAME,
          sortTs: nowMs,
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
        userMessage: [
          "Pause and reflect: what changed in the last few turns?",
          `Reflection cadence: every ${Math.round(event.payload.interval_ms / 60_000)} minutes.`,
          formatIdentityEvents(event.payload.recent_identity_events),
        ].join("\n"),
      };
    },
  };
}
