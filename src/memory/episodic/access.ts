import type { EntityId } from "../../util/ids.js";
import {
  isEpisodeAccessVisible,
  normalizeEpisodeAccess,
  type EpisodeAccessLike,
} from "./audience-filter.js";

export { normalizeEpisodeAccess, type EpisodeAccessLike } from "./audience-filter.js";

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

  return isEpisodeAccessVisible(input, audienceEntityId);
}

export function isEpisodeInGlobalIdentityScope(
  input: EpisodeAccessLike,
  selfAudienceEntityId?: EntityId | null,
): boolean {
  const normalized = normalizeEpisodeAccess(input);

  return (
    normalized.audience_entity_id === null ||
    (selfAudienceEntityId !== null &&
      selfAudienceEntityId !== undefined &&
      normalized.audience_entity_id === selfAudienceEntityId)
  );
}
