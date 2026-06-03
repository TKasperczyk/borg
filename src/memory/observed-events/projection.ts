import type { BorgRole } from "../commitments/index.js";
import { isCreatorInOperatorContext } from "../../cognition/authority.js";
import type { SessionAudienceRole } from "../../sessions/index.js";
import { formatRelativeAge } from "../../util/relative-time.js";
import type { SessionId } from "../../util/ids.js";
import type { ObservedEventProjectionSourceEvent, ObservedEventRepository } from "./repository.js";
import type { ObservedEventDisclosureClass } from "./types.js";

export const DEFAULT_OBSERVED_EVENT_INTROSPECTION_RECENCY_WINDOW_MS = 3 * 24 * 60 * 60_000;
export const DEFAULT_OBSERVED_EVENT_INTROSPECTION_CAP = 8;

export type ObservedEventIntrospectionRow = {
  occurredAt: number;
  lastSeenAt: number;
  relativeAge: string;
  stance: string;
  taint: string;
  beliefEffect: string;
  disclosureClass: ObservedEventDisclosureClass;
  interactionText: string;
  recurrenceCount: number;
  text: string;
};

export type ObservedEventIntrospectionProjectionInput = {
  repository: Pick<ObservedEventRepository, "listRecentForSession">;
  sessionId: SessionId;
  sessionAudienceRole: SessionAudienceRole;
  currentSenderBorgRole: BorgRole | null;
  nowMs: number;
  recencyWindowMs?: number;
  cap?: number;
};

function disclosureClassesForAudience(
  input: Pick<
    ObservedEventIntrospectionProjectionInput,
    "currentSenderBorgRole" | "sessionAudienceRole"
  >,
): ObservedEventDisclosureClass[] {
  const disclosureClasses: ObservedEventDisclosureClass[] = ["social_observed"];

  if (
    isCreatorInOperatorContext({
      currentSenderBorgRole: input.currentSenderBorgRole,
      sessionAudienceRole: input.sessionAudienceRole,
    })
  ) {
    disclosureClasses.push("self_private");
  }

  return disclosureClasses;
}

function rowText(event: ObservedEventProjectionSourceEvent, relativeAge: string): string {
  const recurrencePrefix =
    event.recurrenceCount > 1 ? `Observed ${event.recurrenceCount} times` : "Observed";

  return `${recurrencePrefix} ${event.stance} ${relativeAge}: ${event.interactionText}`;
}

export function selectObservedEventIntrospection(
  input: ObservedEventIntrospectionProjectionInput,
): ObservedEventIntrospectionRow[] {
  const cap = Math.max(1, Math.floor(input.cap ?? DEFAULT_OBSERVED_EVENT_INTROSPECTION_CAP));
  const recencyWindowMs = Math.max(
    0,
    input.recencyWindowMs ?? DEFAULT_OBSERVED_EVENT_INTROSPECTION_RECENCY_WINDOW_MS,
  );
  const sinceMs = input.nowMs - recencyWindowMs;
  const events = disclosureClassesForAudience(input)
    .flatMap((disclosureClass) =>
      input.repository.listRecentForSession({
        sessionId: input.sessionId,
        disclosureClass,
        sinceMs,
        limit: cap,
      }),
    )
    .sort((left, right) => right.lastSeenAt - left.lastSeenAt)
    .slice(0, cap);

  return events.map((event) => {
    const relativeAge = formatRelativeAge(event.lastSeenAt, input.nowMs);

    return {
      occurredAt: event.occurredAt,
      lastSeenAt: event.lastSeenAt,
      relativeAge,
      stance: event.stance,
      taint: event.taint,
      beliefEffect: event.beliefEffect,
      disclosureClass: event.disclosureClass,
      interactionText: event.interactionText,
      recurrenceCount: event.recurrenceCount,
      text: rowText(event, relativeAge),
    };
  });
}
