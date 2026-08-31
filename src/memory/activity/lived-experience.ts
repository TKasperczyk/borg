import {
  combineMemoryDisclosureLabels,
  selfPrivateMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../common/disclosure-label.js";
import { formatRelativeAge } from "../../util/relative-time.js";
import {
  formatUtcDaySpanLabel,
  formatUtcTimeSpan,
  isUtcDayBefore,
  utcDayKey,
  utcDayStartMs,
} from "../../util/utc-day.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import type { AutobiographicalPeriod } from "../self/index.js";
import type {
  SelfDecisionDailyDensityRow,
  SelfDecisionIntrospectionRow,
} from "../self-decisions/index.js";
import type { LivedExperienceDaySummary } from "./lived-experience-day-summary.js";
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
export const RECENT_LIVED_EXPERIENCE_DAILY_SPINE_WINDOW_MS = 7 * 24 * 60 * 60_000;
export const DEFAULT_RECENT_LIVED_EXPERIENCE_PERIOD_CAP = 16;

export type RecentLivedExperienceKind =
  | "return_silence_delta"
  | "cross_session_activity"
  | "self_decision_introspection"
  | "cross_session_activity_density"
  | "self_decision_density"
  | "lived_experience_day_summary"
  | "autobiographical_period";

export const RECENT_LIVED_EXPERIENCE_SPINE_KINDS = [
  "return_silence_delta",
  "cross_session_activity_density",
  "self_decision_density",
  "lived_experience_day_summary",
  "autobiographical_period",
] as const satisfies readonly RecentLivedExperienceKind[];

export type RecentLivedExperienceSpineKind = (typeof RECENT_LIVED_EXPERIENCE_SPINE_KINDS)[number];

const RECENT_LIVED_EXPERIENCE_SPINE_KIND_SET = new Set<string>(RECENT_LIVED_EXPERIENCE_SPINE_KINDS);

export function isRecentLivedExperienceSpineKind(
  kind: string | null | undefined,
): kind is RecentLivedExperienceSpineKind {
  return kind !== null && kind !== undefined && RECENT_LIVED_EXPERIENCE_SPINE_KIND_SET.has(kind);
}

export type RecentLivedExperienceRow = {
  kind: RecentLivedExperienceKind;
  occurredAt: number;
  relativeAge: string;
  text: string;
  sourceStreamEntryIds: readonly StreamEntryId[];
  originAudienceEntityIds: readonly EntityId[];
  disclosureLabel?: MemoryDisclosureLabel;
  plannerDecision?: {
    outcomeReference: string;
    summary: string;
    rationale: string | null;
  };
  metadata: Record<string, unknown>;
};

export type RecentLivedExperienceProjectionInput = {
  nowMs: number;
  crossSessionSelfActivity: readonly CrossSessionSelfActivityRow[];
  selfDecisionIntrospection: readonly SelfDecisionIntrospectionRow[];
  activityDensity: readonly ActivityDailyDensityRow[];
  selfDecisionDensity: readonly SelfDecisionDailyDensityRow[];
  daySummaries?: readonly LivedExperienceDaySummary[];
  autobiographicalPeriods?: readonly AutobiographicalPeriod[];
  returnSilence?: {
    currentAudienceLabel?: string | null;
    currentSessionPreviousTurnAt: number | null;
  };
  individualWindowMs?: number;
  dailySpineWindowMs?: number;
  periodCap?: number;
  windowStartMs?: number;
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

  return `[${span}] ${row.conversationTurnCount} ${turnLabel} with ${label} (${timeSpan}; ${activityKindBreakdown(
    row.kindCounts,
  )}).`;
}

function selfDecisionDensityText(row: SelfDecisionDailyDensityRow): string {
  const span = formatUtcDaySpanLabel(row.firstOccurredAt, row.lastOccurredAt);
  const timeSpan = formatUtcTimeSpan(row.firstOccurredAt, row.lastOccurredAt);

  return `[${span}] ${row.decisionCount} autonomous reflections (${timeSpan}).`;
}

function daySummaryText(row: LivedExperienceDaySummary): string {
  const span = formatUtcDaySpanLabel(row.day_start_ms, row.day_end_ms);

  return `[${span}] ${row.gist}`;
}

function privateOriginLabel(originAudienceEntityIds: readonly EntityId[]) {
  return combineMemoryDisclosureLabels([selfPrivateMemoryDisclosureLabel(originAudienceEntityIds)]);
}

function periodEndAt(period: AutobiographicalPeriod, nowMs: number): number {
  return period.end_ts === null ? nowMs : Math.min(period.end_ts, nowMs);
}

type EmittedAutobiographicalPeriodRow = {
  period: AutobiographicalPeriod;
  startAt: number;
  endAt: number;
};

function clampPeriodToWindow(input: {
  period: AutobiographicalPeriod;
  fromMs: number;
  toMs: number;
  nowMs: number;
}): EmittedAutobiographicalPeriodRow | null {
  const startAt = Math.max(input.period.start_ts, input.fromMs);
  const endAt = Math.min(periodEndAt(input.period, input.nowMs), input.toMs);

  return startAt <= endAt
    ? {
        period: input.period,
        startAt,
        endAt,
      }
    : null;
}

function densityDayCoveredByPeriod(
  dayStartMs: number,
  periods: readonly EmittedAutobiographicalPeriodRow[],
): boolean {
  return periods.some((row) => {
    const periodStartDay = utcDayStartMs(row.startAt);
    const periodEndDay = utcDayStartMs(row.endAt);

    return dayStartMs >= periodStartDay && dayStartMs <= periodEndDay;
  });
}

function autobiographicalPeriodText(row: EmittedAutobiographicalPeriodRow): string {
  const span = formatUtcDaySpanLabel(row.startAt, row.endAt);
  const label = promptSafeLabel(row.period.label);

  return `[${span}] autobiographical period: ${label}.`;
}

function returnSilenceText(input: {
  currentAudienceLabel?: string | null;
  currentSessionPreviousTurnAt: number;
  nowMs: number;
}): string {
  const label = promptSafeLabel(input.currentAudienceLabel ?? "current audience");
  const relativeAge = formatRelativeAge(input.currentSessionPreviousTurnAt, input.nowMs);

  return `Returned to ${label}; last engaged this current session ${relativeAge}.`;
}

export function selectRecentLivedExperienceRows(
  input: RecentLivedExperienceProjectionInput,
): RecentLivedExperienceRow[] {
  const individualWindowMs = Math.max(
    0,
    input.individualWindowMs ?? RECENT_LIVED_EXPERIENCE_INDIVIDUAL_WINDOW_MS,
  );
  const dailySpineWindowMs = Math.max(
    0,
    input.dailySpineWindowMs ?? RECENT_LIVED_EXPERIENCE_DAILY_SPINE_WINDOW_MS,
  );
  const periodCap = Math.max(
    0,
    Math.floor(input.periodCap ?? DEFAULT_RECENT_LIVED_EXPERIENCE_PERIOD_CAP),
  );
  const individualCutoffMs = input.nowMs - individualWindowMs;
  const dailySpineCutoffMs = input.nowMs - dailySpineWindowMs;
  const collapsedActivityDayKeys = new Set<string>();
  const collapsedSelfDecisionDayKeys = new Set<string>();
  const summaryDayKeys = new Set((input.daySummaries ?? []).map((summary) => summary.utc_day));
  const olderSpineRows = [...input.activityDensity, ...input.selfDecisionDensity].filter((row) =>
    isUtcDayBefore(row.lastOccurredAt, dailySpineCutoffMs),
  );
  const olderSpineFromMs =
    olderSpineRows.length === 0
      ? dailySpineCutoffMs
      : Math.min(...olderSpineRows.map((row) => row.firstOccurredAt));
  const olderSpineWindowStartMs = input.windowStartMs ?? olderSpineFromMs;
  const olderSpineWindowEndMs = utcDayStartMs(dailySpineCutoffMs) - 1;
  const emittedAutobiographicalPeriods =
    olderSpineRows.length > 0 &&
    input.autobiographicalPeriods !== undefined &&
    periodCap > 0 &&
    olderSpineWindowEndMs >= olderSpineWindowStartMs
      ? input.autobiographicalPeriods
          .map((period) =>
            clampPeriodToWindow({
              period,
              fromMs: olderSpineWindowStartMs,
              toMs: olderSpineWindowEndMs,
              nowMs: input.nowMs,
            }),
          )
          .filter((period): period is EmittedAutobiographicalPeriodRow => period !== null)
          .slice(0, periodCap)
      : [];

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

  if (
    input.returnSilence !== undefined &&
    input.returnSilence.currentSessionPreviousTurnAt !== null
  ) {
    const previousTurnAt = input.returnSilence.currentSessionPreviousTurnAt;

    rows.push({
      kind: "return_silence_delta",
      occurredAt: previousTurnAt,
      relativeAge: formatRelativeAge(previousTurnAt, input.nowMs),
      text: returnSilenceText({
        currentAudienceLabel: input.returnSilence.currentAudienceLabel,
        currentSessionPreviousTurnAt: previousTurnAt,
        nowMs: input.nowMs,
      }),
      sourceStreamEntryIds: [],
      originAudienceEntityIds: [],
      metadata: {
        current_session_previous_turn_at: previousTurnAt,
        returned_at: input.nowMs,
        relative_age: formatRelativeAge(previousTurnAt, input.nowMs),
        disclosure_class: "self_private",
      },
    });
  }

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
      plannerDecision: {
        outcomeReference: row.decisionOutcomeReference,
        summary: row.decisionSummary,
        rationale: row.decisionRationale,
      },
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
    if (summaryDayKeys.has(row.dayKey)) {
      continue;
    }

    if (!isUtcDayBefore(row.lastOccurredAt, individualCutoffMs)) {
      continue;
    }

    if (
      isUtcDayBefore(row.lastOccurredAt, dailySpineCutoffMs) &&
      densityDayCoveredByPeriod(row.dayStartMs, emittedAutobiographicalPeriods)
    ) {
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
    if (summaryDayKeys.has(row.dayKey)) {
      continue;
    }

    if (!isUtcDayBefore(row.lastOccurredAt, individualCutoffMs)) {
      continue;
    }

    if (
      isUtcDayBefore(row.lastOccurredAt, dailySpineCutoffMs) &&
      densityDayCoveredByPeriod(row.dayStartMs, emittedAutobiographicalPeriods)
    ) {
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
        distinct_decision_shape_count: row.distinctDecisionShapeCount,
        first_occurred_at: row.firstOccurredAt,
        last_occurred_at: row.lastOccurredAt,
        relative_age: formatRelativeAge(row.lastOccurredAt, input.nowMs),
        disclosure_class: "self_private",
      },
    });
  }

  for (const summary of input.daySummaries ?? []) {
    rows.push({
      kind: "lived_experience_day_summary",
      occurredAt: summary.day_end_ms,
      relativeAge: formatRelativeAge(summary.day_end_ms, input.nowMs),
      text: daySummaryText(summary),
      sourceStreamEntryIds: summary.source_stream_entry_ids,
      originAudienceEntityIds: summary.disclosure_label.originAudienceEntityIds,
      disclosureLabel: summary.disclosure_label,
      metadata: {
        summary_id: summary.id,
        self_entity_id: summary.self_entity_id,
        utc_day: summary.utc_day,
        day_start_ms: summary.day_start_ms,
        day_end_ms: summary.day_end_ms,
        salience: summary.salience,
        counts_snapshot: summary.counts_snapshot,
        source_episode_ids: [...summary.source_episode_ids],
        source_stream_ids: [...summary.source_stream_entry_ids],
        relative_age: formatRelativeAge(summary.day_end_ms, input.nowMs),
      },
    });
  }

  for (const periodRow of emittedAutobiographicalPeriods) {
    rows.push({
      kind: "autobiographical_period",
      occurredAt: periodRow.endAt,
      relativeAge: formatRelativeAge(periodRow.endAt, input.nowMs),
      text: autobiographicalPeriodText(periodRow),
      sourceStreamEntryIds: [],
      originAudienceEntityIds: [],
      disclosureLabel: periodRow.period.disclosure_label,
      metadata: {
        autobiographical_period_id: periodRow.period.id,
        period_label: periodRow.period.label,
        period_start_ts: periodRow.period.start_ts,
        period_end_ts: periodRow.period.end_ts,
        clamped_period_start_ts: periodRow.startAt,
        clamped_period_end_ts: periodRow.endAt,
        relative_age: formatRelativeAge(periodRow.endAt, input.nowMs),
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
  row: Pick<RecentLivedExperienceRow, "originAudienceEntityIds" | "disclosureLabel">,
) {
  return row.disclosureLabel === undefined
    ? privateOriginLabel(row.originAudienceEntityIds)
    : combineMemoryDisclosureLabels([
        privateOriginLabel(row.originAudienceEntityIds),
        row.disclosureLabel,
      ]);
}
