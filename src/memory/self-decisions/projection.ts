import { formatRelativeAge } from "../../util/relative-time.js";
import type { StreamEntryId } from "../../util/ids.js";
import type { SelfDecisionProjectionSourceEvent, SelfDecisionRepository } from "./repository.js";
import type { SelfDecisionTriggerType } from "./types.js";

export const DEFAULT_SELF_DECISION_INTROSPECTION_RECENCY_WINDOW_MS = 3 * 24 * 60 * 60_000;
export const DEFAULT_SELF_DECISION_INTROSPECTION_CAP = 8;

export type SelfDecisionIntrospectionRow = {
  occurredAt: number;
  decisionOutcomeReference: string;
  relativeAge: string;
  triggerName: string;
  triggerType: SelfDecisionTriggerType;
  decisionSummary: string;
  decisionRationale: string | null;
  sourceStreamEntryIds: readonly StreamEntryId[];
  text: string;
};

export type SelfDecisionIntrospectionProjectionInput = {
  repository: Pick<SelfDecisionRepository, "listRecentAutonomousSelfPrivate">;
  nowMs: number;
  recencyWindowMs?: number;
  cap?: number;
};

function promptSafeTriggerName(value: string): string {
  const normalized = value.replaceAll("\n", " ").replaceAll("\r", " ").replaceAll("\t", " ").trim();

  return normalized.length === 0 ? "autonomous_wake" : normalized.slice(0, 120).trimEnd();
}

function rowText(event: SelfDecisionProjectionSourceEvent, relativeAge: string): string {
  const decisionText =
    event.decisionRationale === null
      ? event.decisionSummary
      : `${event.decisionSummary} because ${event.decisionRationale}`;

  return `Autonomous ${event.triggerType} ${promptSafeTriggerName(event.triggerName)} completed ${relativeAge}: ${decisionText}`;
}

export function selectSelfDecisionIntrospection(
  input: SelfDecisionIntrospectionProjectionInput,
): SelfDecisionIntrospectionRow[] {
  const cap = Math.max(1, Math.floor(input.cap ?? DEFAULT_SELF_DECISION_INTROSPECTION_CAP));
  const recencyWindowMs = Math.max(
    0,
    input.recencyWindowMs ?? DEFAULT_SELF_DECISION_INTROSPECTION_RECENCY_WINDOW_MS,
  );
  const events = input.repository.listRecentAutonomousSelfPrivate({
    sinceMs: input.nowMs - recencyWindowMs,
    limit: cap,
  });

  return events.map((event) => {
    const relativeAge = formatRelativeAge(event.occurredAt, input.nowMs);

    return {
      occurredAt: event.occurredAt,
      // The scheduler's source-event id is a machine-generated structural
      // reference. Recurring derivations of the same due outcome therefore
      // share an identity without comparing their natural-language content.
      decisionOutcomeReference: event.sourceEventId,
      relativeAge,
      triggerName: event.triggerName,
      triggerType: event.triggerType,
      decisionSummary: event.decisionSummary,
      decisionRationale: event.decisionRationale,
      sourceStreamEntryIds: event.sourceStreamEntryIds,
      text: rowText(event, relativeAge),
    };
  });
}
