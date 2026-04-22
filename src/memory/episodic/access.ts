import type { EntityId } from "../../util/ids.js";

export type EpisodeAccessLike = {
  audience_entity_id?: EntityId | null;
  shared?: boolean;
};

export function normalizeEpisodeAccess<T extends EpisodeAccessLike>(
  input: T,
): T & {
  audience_entity_id: EntityId | null;
  shared: boolean;
} {
  const audienceEntityId = input.audience_entity_id ?? null;

  return {
    ...input,
    audience_entity_id: audienceEntityId,
    shared: input.shared ?? audienceEntityId === null,
  };
}

export function episodeAccessScopeKey(input: EpisodeAccessLike): string {
  const normalized = normalizeEpisodeAccess(input);
  return `${normalized.audience_entity_id ?? "public"}:${normalized.shared ? "shared" : "private"}`;
}

export function hasSameEpisodeAccessScope(
  left: EpisodeAccessLike,
  right: EpisodeAccessLike,
): boolean {
  return episodeAccessScopeKey(left) === episodeAccessScopeKey(right);
}

export function isEpisodeVisibleToAudience(
  input: EpisodeAccessLike,
  audienceEntityId: EntityId | null | undefined,
  options: {
    crossAudience?: boolean;
  } = {},
): boolean {
  if (options.crossAudience === true) {
    return true;
  }

  const normalized = normalizeEpisodeAccess(input);

  if (normalized.audience_entity_id === null || normalized.shared) {
    return true;
  }

  if (audienceEntityId === null || audienceEntityId === undefined) {
    return false;
  }

  return normalized.audience_entity_id === audienceEntityId;
}
