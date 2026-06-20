import {
  combineMemoryDisclosureLabels,
  selfPrivateMemoryDisclosureLabel,
} from "../common/disclosure-label.js";
import { formatRelativeAge } from "../../util/relative-time.js";
import {
  formatUtcDaySpanLabel,
  formatUtcTimeSpan,
  isUtcDayBefore,
  utcDayKey,
} from "../../util/utc-day.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import type {
  SelfDecisionDailyDensityRow,
  SelfDecisionIntrospectionRow,
} from "../self-decisions/index.js";
import type { ActivityDailyDensityRow, ActivityEventKindCounts } from "./repository.js";
import type { CrossSessionSelfActivityRow } from "./projection.js";

export const DEFAULT_RECENT_LIVED_EXPERIENCE_RECENCY_WINDOW_MS = 3 * 24 * 60 * 60_000;
export const DEFAULT_RECENT_LIVED_EXPERIENCE_CAP = 64;
export const DEFAULT_RECENT_LIVED_EXPERIENCE_DENSITY_CAP = 48;
// Surface the lived-experience band only when returning to a session after at
// least this much silence (default 3h). Suppresses noise on rapid same-session
// back-and-forth while still firing on any real return gap (overnight, days).
export const DEFAULT_RECENT_LIVED_EXPERIENCE_GAP_THRESHOLD_MS = 3 * 60 * 60_000;
export const RECENT_LIVED_EXPERIENCE_INDIVIDUAL_WINDOW_MS = 24 * 60 * 60_000;

export type RecentLivedExperienceKind =
  | "cross_session_activity"
  | "self_decision_introspection"
  | "cross_session_activity_density"
  | "self_decision_density";

export type RecentLivedExperienceRow = {
  kind: RecentLivedExperienceKind;
  occurredAt: number;
  relativeAge: string;
  text: string;
  sourceStreamEntryIds: readonly StreamEntryId[];
  originAudienceEntityIds: readonly EntityId[];
  metadata: Record<string, unknown>;
};

export type RecentLivedExperienceProjectionInput = {
  nowMs: number;
  crossSessionSelfActivity: readonly CrossSessionSelfActivityRow[];
  selfDecisionIntrospection: readonly SelfDecisionIntrospectionRow[];
  activityDensity: readonly ActivityDailyDensityRow[];
  selfDecisionDensity: readonly SelfDecisionDailyDensityRow[];
  individualWindowMs?: number;
};

function promptSafeLabel(value: string): string {
  const normalized = value.replaceAll("\n", " ").replaceAll("\r", " ").replaceAll("\t", " ").trim();

  if (normalized.length === 0) {
    return "a participant";
  }

  return normalized.slice(0, 120).trimEnd();
}

function activityKindBreakdown(counts: ActivityEventKindCounts): string {
  return [
    `user_contact=${counts.userContact}`,
    `borg_replied=${counts.borgReplied}`,
    `turn_completed=${counts.turnCompleted}`,
  ].join(" ");
}

function activityCollapseKey(dayKey: string, sessionId: string): string {
  return `${dayKey}/${sessionId}`;
}

function activityDensityText(row: ActivityDailyDensityRow): string {
  const span = formatUtcDaySpanLabel(row.firstOccurredAt, row.lastOccurredAt);
  const timeSpan = formatUtcTimeSpan(row.firstOccurredAt, row.lastOccurredAt);
  const label = promptSafeLabel(row.audienceLabel);
  const turnLabel = row.conversationTurnCount === 1 ? "conversation turn" : "conversation turns";

  return `[${span}] ${row.conversationTurnCount} ${turnLabel} with ${label} (${timeSpan}; activity_events=${row.eventCount}; ${activityKindBreakdown(
    row.kindCounts,
  )}).`;
}

function selfDecisionDensityText(row: SelfDecisionDailyDensityRow): string {
  const span = formatUtcDaySpanLabel(row.firstOccurredAt, row.lastOccurredAt);
  const timeSpan = formatUtcTimeSpan(row.firstOccurredAt, row.lastOccurredAt);

  return `[${span}] ${row.decisionCount} autonomous reflections (${timeSpan}).`;
}

function privateOriginLabel(originAudienceEntityIds: readonly EntityId[]) {
  return combineMemoryDisclosureLabels([selfPrivateMemoryDisclosureLabel(originAudienceEntityIds)]);
}

export function selectRecentLivedExperienceRows(
  input: RecentLivedExperienceProjectionInput,
): RecentLivedExperienceRow[] {
  const individualWindowMs = Math.max(
    0,
    input.individualWindowMs ?? RECENT_LIVED_EXPERIENCE_INDIVIDUAL_WINDOW_MS,
  );
  const individualCutoffMs = input.nowMs - individualWindowMs;
  const collapsedActivityDayKeys = new Set<string>();
  const collapsedSelfDecisionDayKeys = new Set<string>();

  for (const row of input.activityDensity) {
    if (isUtcDayBefore(row.lastOccurredAt, individualCutoffMs)) {
      collapsedActivityDayKeys.add(activityCollapseKey(row.dayKey, row.sessionId));
    }
  }

  for (const row of input.selfDecisionDensity) {
    if (isUtcDayBefore(row.lastOccurredAt, individualCutoffMs)) {
      collapsedSelfDecisionDayKeys.add(row.dayKey);
    }
  }

  const rows: RecentLivedExperienceRow[] = [];

  for (const row of input.crossSessionSelfActivity) {
    if (
      collapsedActivityDayKeys.has(activityCollapseKey(utcDayKey(row.occurredAt), row.sessionId))
    ) {
      continue;
    }

    rows.push({
      kind: "cross_session_activity",
      occurredAt: row.occurredAt,
      relativeAge: row.relativeAge,
      text: row.text,
      sourceStreamEntryIds: row.sourceStreamEntryIds,
      originAudienceEntityIds: row.originAudienceEntityIds,
      metadata: {
        event_kind: row.kind,
        session_id: row.sessionId,
        occurred_at: row.occurredAt,
        relative_age: row.relativeAge,
        source_stream_ids: [...row.sourceStreamEntryIds],
      },
    });
  }

  for (const row of input.selfDecisionIntrospection) {
    if (collapsedSelfDecisionDayKeys.has(utcDayKey(row.occurredAt))) {
      continue;
    }

    rows.push({
      kind: "self_decision_introspection",
      occurredAt: row.occurredAt,
      relativeAge: row.relativeAge,
      text: row.text,
      sourceStreamEntryIds: row.sourceStreamEntryIds,
      originAudienceEntityIds: [],
      metadata: {
        trigger_name: row.triggerName,
        trigger_type: row.triggerType,
        occurred_at: row.occurredAt,
        relative_age: row.relativeAge,
        disclosure_class: "self_private",
        source_stream_ids: [...row.sourceStreamEntryIds],
      },
    });
  }

  for (const row of input.activityDensity) {
    if (!isUtcDayBefore(row.lastOccurredAt, individualCutoffMs)) {
      continue;
    }

    rows.push({
      kind: "cross_session_activity_density",
      occurredAt: row.lastOccurredAt,
      relativeAge: formatRelativeAge(row.lastOccurredAt, input.nowMs),
      text: activityDensityText(row),
      sourceStreamEntryIds: [],
      originAudienceEntityIds: row.audienceEntityId === null ? [] : [row.audienceEntityId],
      metadata: {
        day_key: row.dayKey,
        session_id: row.sessionId,
        session_label: row.sessionLabel,
        audience_label: row.audienceLabel,
        audience_entity_id: row.audienceEntityId,
        event_count: row.eventCount,
        conversation_turn_count: row.conversationTurnCount,
        kind_counts: row.kindCounts,
        first_occurred_at: row.firstOccurredAt,
        last_occurred_at: row.lastOccurredAt,
        relative_age: formatRelativeAge(row.lastOccurredAt, input.nowMs),
        disclosure_class: "self_private",
      },
    });
  }

  for (const row of input.selfDecisionDensity) {
    if (!isUtcDayBefore(row.lastOccurredAt, individualCutoffMs)) {
      continue;
    }

    rows.push({
      kind: "self_decision_density",
      occurredAt: row.lastOccurredAt,
      relativeAge: formatRelativeAge(row.lastOccurredAt, input.nowMs),
      text: selfDecisionDensityText(row),
      sourceStreamEntryIds: [],
      originAudienceEntityIds: [],
      metadata: {
        day_key: row.dayKey,
        decision_count: row.decisionCount,
        first_occurred_at: row.firstOccurredAt,
        last_occurred_at: row.lastOccurredAt,
        relative_age: formatRelativeAge(row.lastOccurredAt, input.nowMs),
        disclosure_class: "self_private",
      },
    });
  }

  return rows.sort((left, right) => {
    if (left.occurredAt !== right.occurredAt) {
      return left.occurredAt - right.occurredAt;
    }

    return left.kind.localeCompare(right.kind);
  });
}

export function recentLivedExperienceDisclosureLabel(
  row: Pick<RecentLivedExperienceRow, "originAudienceEntityIds">,
) {
  return privateOriginLabel(row.originAudienceEntityIds);
}
