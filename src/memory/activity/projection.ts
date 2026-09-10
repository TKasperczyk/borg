import { formatRelativeAge } from "../../util/relative-time.js";
import type { EntityId, SessionId, StreamEntryId } from "../../util/ids.js";
import type { ActivityEventKind, ActivityRelevanceInput } from "./types.js";
import type { ActivityRepository, ActivityProjectionSourceEvent } from "./repository.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";
import { utf16SafePrefix } from "../../util/utf16-boundary.js";
import { halfLifeDecay } from "../../util/math.js";
import { EmbeddingError } from "../../util/errors.js";
import { similarityThresholds, type SimilarityConfigSource } from "../../config/similarity.js";
import {
  DEFAULT_RECENT_LIVED_EXPERIENCE_CAP,
  DEFAULT_RECENT_LIVED_EXPERIENCE_RECENCY_WINDOW_MS,
} from "./lived-experience.js";

export const DEFAULT_CROSS_SESSION_ACTIVITY_RECENCY_WINDOW_MS =
  DEFAULT_RECENT_LIVED_EXPERIENCE_RECENCY_WINDOW_MS;
export const DEFAULT_CROSS_SESSION_ACTIVITY_CAP = DEFAULT_RECENT_LIVED_EXPERIENCE_CAP;

/** The embedding model interprets content; code only combines numeric similarity and age. */
export async function rankActivityByRelevance(
  input: ActivityRelevanceInput,
  embeddingClient: EmbeddingClient,
  similarityConfig?: SimilarityConfigSource,
): Promise<string[]> {
  if (input.candidates.length === 0) return [];
  // Keep the focus under the single-query stall guard, independent of the slower
  // activity batch. Episodic recall reuses this same query cache entry.
  const [focusVector, vectors] = await Promise.all([
    embeddingClient.embed(input.focus),
    embeddingClient.embedBatch(input.candidates.map((candidate) => candidate.text)),
  ]);
  if (vectors.length !== input.candidates.length) {
    throw new EmbeddingError("Activity relevance embedding batch is incomplete", {
      code: "ACTIVITY_EMBEDDING_BATCH_INCOMPLETE",
    });
  }
  const auxiliaryScale = similarityThresholds(
    similarityConfig ?? { embedding: embeddingClient.profile },
  ).recallAuxiliaryScoreScale;
  return input.candidates
    .map((candidate, index) => ({
      candidate,
      score:
        cosineSimilarity(focusVector, vectors[index]!) +
        0.15 *
          auxiliaryScale *
          halfLifeDecay(Math.max(0, input.nowMs - candidate.occurredAt) / (60 * 60_000), 36),
    }))
    .sort(
      (left, right) =>
        right.score - left.score || right.candidate.occurredAt - left.candidate.occurredAt,
    )
    .map(({ candidate }) => candidate.key);
}

export type CrossSessionSelfActivityRow = {
  kind: ActivityEventKind;
  occurredAt: number;
  sessionId: SessionId;
  relativeAge: string;
  text: string;
  originAudienceEntityIds: readonly EntityId[];
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

  return utf16SafePrefix(normalized, 120).trimEnd();
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
      sessionId: event.sessionId,
      relativeAge,
      text: rowText(event, relativeAge),
      originAudienceEntityIds:
        event.audienceEntityId === null || event.audienceEntityId === undefined
          ? []
          : [event.audienceEntityId],
      sourceStreamEntryIds: event.sourceStreamEntryIds,
    };
  });
}
