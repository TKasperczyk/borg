import type { CommitmentRecord, CommitmentRepository } from "../../memory/commitments/index.js";
import type { StreamWatermarkRepository } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../../util/ids.js";
import { AUTONOMOUS_WAKE_USER_MESSAGE } from "../../cognition/autonomy-trigger.js";
import type { AutonomyCondition, DueEvent } from "../types.js";

const CONDITION_NAME = "commitment_revoked" as const;
const WATERMARK_PREFIX = "autonomy:commitment-revoked";

export type CommitmentRevokedPayload = {
  commitment_id: CommitmentRecord["id"];
  directive: string;
  reason: string | null;
  revoked_at: number;
};

export type CommitmentRevokedConditionOptions = {
  commitmentRepository: CommitmentRepository;
  watermarkRepository: StreamWatermarkRepository;
  clock?: Clock;
  sessionId?: SessionId;
};

export function createCommitmentRevokedCondition(
  options: CommitmentRevokedConditionOptions,
): AutonomyCondition<CommitmentRevokedPayload> {
  const clock = options.clock ?? new SystemClock();
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;

  return {
    name: CONDITION_NAME,
    type: "condition",
    async scan() {
      return options.commitmentRepository
        .list()
        .filter(
          (commitment): commitment is CommitmentRecord & { revoked_at: number } =>
            commitment.revoked_at !== null,
        )
        .sort((left, right) => left.revoked_at - right.revoked_at)
        .map<DueEvent<CommitmentRevokedPayload> | null>((commitment) => {
          const watermarkProcessName =
            `${WATERMARK_PREFIX}:${commitment.id}:${commitment.revoked_at}`;

          if (options.watermarkRepository.get(watermarkProcessName, sessionId) !== null) {
            return null;
          }

          return {
            id: `${commitment.id}:${commitment.revoked_at}`,
            sourceName: CONDITION_NAME,
            sourceType: "condition",
            watermarkProcessName,
            sortTs: commitment.revoked_at,
            payload: {
              commitment_id: commitment.id,
              directive: commitment.directive,
              reason: commitment.revoked_reason,
              revoked_at: commitment.revoked_at,
            },
          };
        })
        .filter((event): event is DueEvent<CommitmentRevokedPayload> => event !== null);
    },
    buildTurn(event) {
      return {
        audience: "self",
        stakes: "low",
        userMessage: AUTONOMOUS_WAKE_USER_MESSAGE,
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
