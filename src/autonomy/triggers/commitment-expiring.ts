import type { CommitmentRecord, CommitmentRepository } from "../../memory/commitments/index.js";
import type { StreamWatermarkRepository } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../../util/ids.js";
import type { AutonomyTrigger, DueEvent } from "../types.js";

const TRIGGER_NAME = "commitment_expiring" as const;
const WATERMARK_PREFIX = "autonomy:commitment-expiring";

type CommitmentExpiringPayload = {
  commitment_id: CommitmentRecord["id"];
  type: CommitmentRecord["type"];
  directive: string;
  priority: number;
  expires_at: number;
  active_commitments?: CommitmentRecord[];
};

export type CommitmentExpiringTriggerOptions = {
  commitmentRepository: CommitmentRepository;
  watermarkRepository: StreamWatermarkRepository;
  lookaheadMs: number;
  clock?: Clock;
  sessionId?: SessionId;
};

export function createCommitmentExpiringTrigger(
  options: CommitmentExpiringTriggerOptions,
): AutonomyTrigger<CommitmentExpiringPayload> {
  const clock = options.clock ?? new SystemClock();
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;

  return {
    name: TRIGGER_NAME,
    type: "trigger",
    async scan() {
      const nowMs = clock.now();
      const dueEvents = options.commitmentRepository
        .list({
          activeOnly: true,
          nowMs,
        })
        .filter(
          (commitment) =>
            commitment.expires_at !== null &&
            commitment.expires_at > nowMs &&
            commitment.expires_at - nowMs < options.lookaheadMs,
        )
        .sort(
          (left, right) =>
            (left.expires_at ?? Number.MAX_SAFE_INTEGER) -
              (right.expires_at ?? Number.MAX_SAFE_INTEGER) || right.priority - left.priority,
        );

      return dueEvents
        .map<DueEvent<CommitmentExpiringPayload> | null>((commitment) => {
          const expiresAt = commitment.expires_at;

          if (expiresAt === null) {
            return null;
          }

          const watermarkProcessName = `${WATERMARK_PREFIX}:${commitment.id}:${expiresAt}`;

          if (options.watermarkRepository.get(watermarkProcessName, sessionId) !== null) {
            return null;
          }

          return {
            id: `${commitment.id}:${expiresAt}`,
            sourceName: TRIGGER_NAME,
            sourceType: "trigger",
            watermarkProcessName,
            sortTs: expiresAt,
            payload: {
              commitment_id: commitment.id,
              type: commitment.type,
              directive: commitment.directive,
              priority: commitment.priority,
              expires_at: expiresAt,
            },
          };
        })
        .filter((event): event is DueEvent<CommitmentExpiringPayload> => event !== null);
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
