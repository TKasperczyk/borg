import type { BorgRole } from "../commitments/index.js";
import { isCreatorInOperatorContext } from "../../cognition/authority.js";
import type { SessionAudienceRole } from "../../sessions/index.js";
import { formatRelativeAge } from "../../util/relative-time.js";
import type { SessionId } from "../../util/ids.js";
import type { SelfDecisionProjectionSourceEvent, SelfDecisionRepository } from "./repository.js";
import type { SelfDecisionTriggerType } from "./types.js";

export const DEFAULT_SELF_DECISION_INTROSPECTION_RECENCY_WINDOW_MS = 3 * 24 * 60 * 60_000;
export const DEFAULT_SELF_DECISION_INTROSPECTION_CAP = 8;

export type SelfDecisionIntrospectionRow = {
  occurredAt: number;
  relativeAge: string;
  triggerName: string;
  triggerType: SelfDecisionTriggerType;
  decisionSummary: string;
  text: string;
};

export type SelfDecisionIntrospectionProjectionInput = {
  repository: Pick<SelfDecisionRepository, "listRecentForSession">;
  sessionId: SessionId;
  sessionAudienceRole: SessionAudienceRole;
  currentSenderBorgRole: BorgRole | null;
  nowMs: number;
  recencyWindowMs?: number;
  cap?: number;
};

function promptSafeTriggerName(value: string): string {
  const normalized = value.replaceAll("\n", " ").replaceAll("\r", " ").replaceAll("\t", " ").trim();

  return normalized.length === 0 ? "autonomous_wake" : normalized.slice(0, 120).trimEnd();
}

function rowText(event: SelfDecisionProjectionSourceEvent, relativeAge: string): string {
  return `Autonomous ${event.triggerType} ${promptSafeTriggerName(event.triggerName)} completed ${relativeAge}: ${event.decisionSummary}`;
}

export function selectSelfDecisionIntrospection(
  input: SelfDecisionIntrospectionProjectionInput,
): SelfDecisionIntrospectionRow[] {
  if (
    !isCreatorInOperatorContext({
      currentSenderBorgRole: input.currentSenderBorgRole,
      sessionAudienceRole: input.sessionAudienceRole,
    })
  ) {
    return [];
  }

  const cap = Math.max(1, Math.floor(input.cap ?? DEFAULT_SELF_DECISION_INTROSPECTION_CAP));
  const recencyWindowMs = Math.max(
    0,
    input.recencyWindowMs ?? DEFAULT_SELF_DECISION_INTROSPECTION_RECENCY_WINDOW_MS,
  );
  const events = input.repository.listRecentForSession({
    sessionId: input.sessionId,
    sinceMs: input.nowMs - recencyWindowMs,
    limit: cap,
  });

  return events.map((event) => {
    const relativeAge = formatRelativeAge(event.occurredAt, input.nowMs);

    return {
      occurredAt: event.occurredAt,
      relativeAge,
      triggerName: event.triggerName,
      triggerType: event.triggerType,
      decisionSummary: event.decisionSummary,
      text: rowText(event, relativeAge),
    };
  });
}
