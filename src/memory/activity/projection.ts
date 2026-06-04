import { formatRelativeAge } from "../../util/relative-time.js";
import type { SessionId, StreamEntryId } from "../../util/ids.js";
import type { ActivityEventKind } from "./types.js";
import type { ActivityRepository, ActivityProjectionSourceEvent } from "./repository.js";

export const DEFAULT_CROSS_SESSION_ACTIVITY_RECENCY_WINDOW_MS = 60 * 60_000;
export const DEFAULT_CROSS_SESSION_ACTIVITY_CAP = 8;

export type CrossSessionSelfActivityRow = {
  kind: ActivityEventKind;
  occurredAt: number;
  relativeAge: string;
  text: string;
  sourceStreamEntryIds: readonly StreamEntryId[];
};

export type CrossSessionSelfActivityProjectionInput = {
  repository: Pick<ActivityRepository, "listRecentOtherActiveSessionEvents">;
  currentSessionId: SessionId;
  nowMs: number;
  recencyWindowMs?: number;
  cap?: number;
};

function promptSafeLabel(value: string): string {
  const normalized = value.replaceAll("\n", " ").replaceAll("\r", " ").replaceAll("\t", " ").trim();

  if (normalized.length === 0) {
    return "A participant";
  }

  return normalized.slice(0, 120).trimEnd();
}

function rowText(event: ActivityProjectionSourceEvent, relativeAge: string): string {
  const label = promptSafeLabel(event.participantLabel);

  switch (event.kind) {
    case "user_contact":
      return `${label} contacted Borg ${relativeAge} in another active session.`;
    case "borg_replied":
      return `Borg replied to ${label} ${relativeAge} in another active session.`;
    case "turn_completed":
      return `Borg completed a turn with ${label} ${relativeAge} in another active session.`;
  }
}

export function selectCrossSessionSelfActivity(
  input: CrossSessionSelfActivityProjectionInput,
): CrossSessionSelfActivityRow[] {
  const cap = Math.max(1, Math.floor(input.cap ?? DEFAULT_CROSS_SESSION_ACTIVITY_CAP));
  const recencyWindowMs = Math.max(
    0,
    input.recencyWindowMs ?? DEFAULT_CROSS_SESSION_ACTIVITY_RECENCY_WINDOW_MS,
  );
  const events = input.repository.listRecentOtherActiveSessionEvents({
    currentSessionId: input.currentSessionId,
    sinceMs: input.nowMs - recencyWindowMs,
    limit: cap,
  });

  return events.map((event) => {
    const relativeAge = formatRelativeAge(event.occurredAt, input.nowMs);

    return {
      kind: event.kind,
      occurredAt: event.occurredAt,
      relativeAge,
      text: rowText(event, relativeAge),
      sourceStreamEntryIds: event.sourceStreamEntryIds,
    };
  });
}
