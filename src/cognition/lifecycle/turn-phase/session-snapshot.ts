import type {
  ConversationKind,
  SessionParticipationPolicy,
  SessionRecord,
} from "../../../sessions/index.js";
import type { SessionId } from "../../../util/ids.js";
import { formatRelativeAge } from "../../deliberation/prompt/system-prompt.js";

export const OPERATOR_SESSION_SNAPSHOT_CAP = 12;

export type OperatorSessionSnapshotSession = {
  alias: string;
  session_id: SessionId;
  outbound_targetable: boolean;
  audience_label: string;
  conversation_kind: ConversationKind;
  participation_policy: SessionParticipationPolicy;
  last_activity: string;
  message_count: number;
  recent_state: "last_turn_available" | "no_recent_turn";
};

export type OperatorSessionSnapshot = {
  generated_at: string;
  sessions: OperatorSessionSnapshotSession[];
  omitted_count?: number;
};

export type BuildOperatorSessionSnapshotInput = {
  sessions: readonly SessionRecord[];
  currentSessionId: SessionId;
  nowMs: number;
  cap?: number;
  totalActiveOtherSessionCount?: number;
  /**
   * Session ids that are outbound-targetable on this turn (creator-in-operator
   * with a wired connector for the session's source_type). Only these expose
   * their session_id to the model; awareness rendering for everyone else stays
   * alias-only so the snapshot does not leak internal ids on non-outbound turns.
   */
  outboundTargetableSessionIds?: ReadonlySet<SessionId>;
};

export function buildOperatorSessionSnapshot(
  input: BuildOperatorSessionSnapshotInput,
): OperatorSessionSnapshot | null {
  if (!Number.isFinite(input.nowMs)) {
    return null;
  }

  const cap = Math.max(0, Math.floor(input.cap ?? OPERATOR_SESSION_SNAPSHOT_CAP));
  const eligibleSessions = input.sessions
    .filter(
      (session) => session.status === "active" && session.session_id !== input.currentSessionId,
    )
    .sort(
      (left, right) =>
        right.last_activity_at - left.last_activity_at ||
        left.session_id.localeCompare(right.session_id),
    );
  const visibleSessions = eligibleSessions.slice(0, cap);
  const totalCount = input.totalActiveOtherSessionCount ?? eligibleSessions.length;
  const omittedCount = Math.max(0, totalCount - visibleSessions.length);

  return {
    generated_at: new Date(input.nowMs).toISOString(),
    sessions: visibleSessions.map((session, index) => ({
      alias: `session_${index + 1}`,
      session_id: session.session_id,
      outbound_targetable: input.outboundTargetableSessionIds?.has(session.session_id) ?? false,
      audience_label: session.audience_label,
      conversation_kind: session.conversation_kind,
      participation_policy: session.participation_policy,
      last_activity: formatRelativeAge(session.last_activity_at, input.nowMs),
      message_count: session.message_count,
      recent_state: session.last_turn_id === null ? "no_recent_turn" : "last_turn_available",
    })),
    ...(omittedCount > 0 ? { omitted_count: omittedCount } : {}),
  };
}
