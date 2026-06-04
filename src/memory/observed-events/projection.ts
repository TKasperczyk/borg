import { formatRelativeAge } from "../../util/relative-time.js";
import type { EntityId } from "../../util/ids.js";
import type { ObservedEventProjectionSourceEvent, ObservedEventRepository } from "./repository.js";
import type { ObservedEventDisclosureClass } from "./types.js";

export const DEFAULT_OBSERVED_EVENT_INTROSPECTION_RECENCY_WINDOW_MS = 90 * 24 * 60 * 60_000;
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
  speakerEntityId: EntityId | null;
  audienceEntityId: EntityId | null;
  text: string;
};

export type ObservedEventIntrospectionProjectionInput = {
  repository: Pick<ObservedEventRepository, "listRecentBySpeakers">;
  speakerEntityIds: readonly EntityId[];
  nowMs: number;
  recencyWindowMs?: number;
  cap?: number;
};

function rowText(event: ObservedEventProjectionSourceEvent, relativeAge: string): string {
  const recurrencePrefix =
    event.recurrenceCount > 1 ? `Observed ${event.recurrenceCount} times` : "Observed";

  return `${recurrencePrefix} ${event.stance} ${relativeAge}: ${event.interactionText}`;
}

export function selectObservedEventIntrospection(
  input: ObservedEventIntrospectionProjectionInput,
): ObservedEventIntrospectionRow[] {
  if (input.speakerEntityIds.length === 0) {
    return [];
  }

  const cap = Math.max(1, Math.floor(input.cap ?? DEFAULT_OBSERVED_EVENT_INTROSPECTION_CAP));
  const recencyWindowMs = Math.max(
    0,
    input.recencyWindowMs ?? DEFAULT_OBSERVED_EVENT_INTROSPECTION_RECENCY_WINDOW_MS,
  );
  const sinceMs = input.nowMs - recencyWindowMs;
  const events = (["social_observed", "self_private"] satisfies ObservedEventDisclosureClass[])
    .flatMap((disclosureClass) =>
      input.repository.listRecentBySpeakers({
        speakerEntityIds: input.speakerEntityIds,
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
      speakerEntityId: event.speakerEntityId,
      audienceEntityId: event.audienceEntityId,
      text: rowText(event, relativeAge),
    };
  });
}
