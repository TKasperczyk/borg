import { formatRelativeAge } from "../../util/relative-time.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import { clamp } from "../../util/math.js";
import type { ObservedEventProjectionSourceEvent, ObservedEventRepository } from "./repository.js";
import type { ObservedEventDisclosureClass } from "./types.js";

export const DEFAULT_OBSERVED_EVENT_INTROSPECTION_RECENCY_WINDOW_MS = 90 * 24 * 60 * 60_000;
export const DEFAULT_OBSERVED_EVENT_INTROSPECTION_CAP = 8;
export const DEFAULT_OBSERVED_EVENT_TOPIC_MIN_SIMILARITY = 0.45;

export type ObservedEventRecallReason = "topic" | "recent" | "recurring" | "person";

export type ObservedEventIntrospectionRow = {
  id: string;
  occurredAt: number;
  lastSeenAt: number;
  relativeAge: string;
  recallScore: number;
  recallReasons: readonly ObservedEventRecallReason[];
  stance: string;
  taint: string;
  beliefEffect: string;
  disclosureClass: ObservedEventDisclosureClass;
  interactionText: string;
  recurrenceCount: number;
  speakerEntityId: EntityId | null;
  audienceEntityId: EntityId | null;
  sourceStreamEntryIds: readonly StreamEntryId[];
  text: string;
};

export type ObservedEventIntrospectionProjectionInput = {
  repository: Pick<
    ObservedEventRepository,
    "listRecentGlobal" | "listRecurringGlobal" | "listRecentBySpeakers" | "searchByVector"
  >;
  speakerEntityIds: readonly EntityId[];
  queryVector?: Float32Array | null;
  topicMinSimilarity?: number;
  nowMs: number;
  recencyWindowMs?: number;
  cap?: number;
};

type ObservedEventCandidate = {
  event: ObservedEventProjectionSourceEvent;
  score: number;
  reasons: Set<ObservedEventRecallReason>;
};

const OBSERVED_EVENT_DISCLOSURE_CLASSES_FOR_RECALL = [
  "social_observed",
  "self_private",
] as const satisfies readonly ObservedEventDisclosureClass[];

function observedEventStanceNarrative(event: ObservedEventProjectionSourceEvent): string {
  const stance = `stance=${event.stance}`;
  const taint = `taint=${event.taint}`;
  const belief = `belief_effect=${event.beliefEffect}`;

  if (event.stance === "rejected_frame" || event.taint === "quarantined") {
    return `${stance}; ${taint}; ${belief}; not accepted as true`;
  }

  if (event.stance === "accepted_frame") {
    return `${stance}; ${taint}; ${belief}; accepted as an observed social memory`;
  }

  return `${stance}; ${taint}; ${belief}; noted without treating the source text as direct truth`;
}

function rowText(event: ObservedEventProjectionSourceEvent, relativeAge: string): string {
  const recurrencePrefix =
    event.recurrenceCount > 1 ? `Observed ${event.recurrenceCount} times` : "Observed";

  return `${recurrencePrefix} ${relativeAge}: ${event.interactionText}. ${observedEventStanceNarrative(event)}`;
}

function recencyScore(
  event: ObservedEventProjectionSourceEvent,
  input: {
    nowMs: number;
    recencyWindowMs: number;
  },
): number {
  if (input.recencyWindowMs <= 0) {
    return 0;
  }

  const age = Math.max(0, input.nowMs - event.lastSeenAt);
  return clamp(1 - age / input.recencyWindowMs, 0, 1);
}

function recurrenceScore(event: ObservedEventProjectionSourceEvent): number {
  return clamp(Math.log2(Math.max(1, event.recurrenceCount)) / 5, 0, 1);
}

function presentSpeakerBoost(
  event: ObservedEventProjectionSourceEvent,
  speakerEntityIds: ReadonlySet<EntityId>,
): number {
  return event.speakerEntityId !== null && speakerEntityIds.has(event.speakerEntityId) ? 0.12 : 0;
}

function baseEventScore(
  event: ObservedEventProjectionSourceEvent,
  input: {
    nowMs: number;
    recencyWindowMs: number;
    speakerEntityIds: ReadonlySet<EntityId>;
  },
): number {
  return clamp(
    0.2 +
      recencyScore(event, input) * 0.24 +
      recurrenceScore(event) * 0.18 +
      presentSpeakerBoost(event, input.speakerEntityIds),
    0,
    1,
  );
}

function addCandidate(
  candidates: Map<string, ObservedEventCandidate>,
  event: ObservedEventProjectionSourceEvent,
  reason: ObservedEventRecallReason,
  score: number,
): void {
  const existing = candidates.get(event.id);

  if (existing === undefined) {
    candidates.set(event.id, {
      event,
      score,
      reasons: new Set([reason]),
    });
    return;
  }

  existing.score = Math.max(existing.score, score);
  existing.reasons.add(reason);
}

export async function selectObservedEventIntrospection(
  input: ObservedEventIntrospectionProjectionInput,
): Promise<ObservedEventIntrospectionRow[]> {
  const cap = Math.max(1, Math.floor(input.cap ?? DEFAULT_OBSERVED_EVENT_INTROSPECTION_CAP));
  const recencyWindowMs = Math.max(
    0,
    input.recencyWindowMs ?? DEFAULT_OBSERVED_EVENT_INTROSPECTION_RECENCY_WINDOW_MS,
  );
  const sinceMs = input.nowMs - recencyWindowMs;
  const speakerEntityIds = new Set(input.speakerEntityIds);
  const candidates = new Map<string, ObservedEventCandidate>();

  if (input.queryVector !== undefined && input.queryVector !== null) {
    const topicHits = await input.repository.searchByVector(input.queryVector, {
      minSimilarity: input.topicMinSimilarity ?? DEFAULT_OBSERVED_EVENT_TOPIC_MIN_SIMILARITY,
      limit: Math.max(cap * 3, 12),
    });

    for (const hit of topicHits) {
      addCandidate(
        candidates,
        hit.event,
        "topic",
        clamp(
          0.5 +
            hit.similarity * 0.3 +
            presentSpeakerBoost(hit.event, speakerEntityIds) +
            recurrenceScore(hit.event) * 0.1,
          0,
          1,
        ),
      );
    }
  }

  for (const disclosureClass of OBSERVED_EVENT_DISCLOSURE_CLASSES_FOR_RECALL) {
    const recentEvents = input.repository.listRecentGlobal({
      disclosureClass,
      sinceMs,
      limit: Math.max(cap * 2, 12),
    });
    const recurringEvents = input.repository.listRecurringGlobal({
      disclosureClass,
      sinceMs,
      limit: Math.max(cap * 2, 12),
    });

    for (const event of recentEvents) {
      addCandidate(
        candidates,
        event,
        "recent",
        baseEventScore(event, { nowMs: input.nowMs, recencyWindowMs, speakerEntityIds }),
      );
    }

    for (const event of recurringEvents) {
      addCandidate(
        candidates,
        event,
        "recurring",
        clamp(
          baseEventScore(event, { nowMs: input.nowMs, recencyWindowMs, speakerEntityIds }) + 0.14,
          0,
          1,
        ),
      );
    }

    if (input.speakerEntityIds.length > 0) {
      const personEvents = input.repository.listRecentBySpeakers({
        speakerEntityIds: input.speakerEntityIds,
        disclosureClass,
        sinceMs,
        limit: Math.max(cap * 2, 12),
      });

      for (const event of personEvents) {
        addCandidate(
          candidates,
          event,
          "person",
          clamp(
            baseEventScore(event, { nowMs: input.nowMs, recencyWindowMs, speakerEntityIds }) + 0.18,
            0,
            1,
          ),
        );
      }
    }
  }

  const events = [...candidates.values()]
    .sort((left, right) => {
      const scoreDelta = right.score - left.score;
      return scoreDelta !== 0 ? scoreDelta : right.event.lastSeenAt - left.event.lastSeenAt;
    })
    .slice(0, cap);

  return events.map((candidate) => {
    const event = candidate.event;
    const relativeAge = formatRelativeAge(event.lastSeenAt, input.nowMs);

    return {
      id: event.id,
      occurredAt: event.occurredAt,
      lastSeenAt: event.lastSeenAt,
      relativeAge,
      recallScore: candidate.score,
      recallReasons: [...candidate.reasons].sort(),
      stance: event.stance,
      taint: event.taint,
      beliefEffect: event.beliefEffect,
      disclosureClass: event.disclosureClass,
      interactionText: event.interactionText,
      recurrenceCount: event.recurrenceCount,
      speakerEntityId: event.speakerEntityId,
      audienceEntityId: event.audienceEntityId,
      sourceStreamEntryIds: event.sourceStreamEntryIds,
      text: rowText(event, relativeAge),
    };
  });
}
