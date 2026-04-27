import type { EntityId, EpisodeId } from "../../util/ids.js";

export type EpisodeAccessLike = {
  audience_entity_id?: EntityId | null;
  shared?: boolean;
};

export type AudienceEpisodeAccess = EpisodeAccessLike & {
  id: EpisodeId;
};

export type AudiencePolicy = "filter" | "reject_if_mixed" | "all_or_nothing";

export type AudienceFilterResult = {
  visibleEpisodeIds: EpisodeId[];
  hiddenEpisodeIds: EpisodeId[];
  hasPrivateMix: boolean;
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

export function isEpisodeAccessVisible(
  input: EpisodeAccessLike,
  audienceEntityId: EntityId | null | undefined,
): boolean {
  const normalized = normalizeEpisodeAccess(input);

  if (normalized.audience_entity_id === null || normalized.shared) {
    return true;
  }

  if (audienceEntityId === null || audienceEntityId === undefined) {
    return false;
  }

  return normalized.audience_entity_id === audienceEntityId;
}

function isPrivateToDifferentAudience(
  input: EpisodeAccessLike,
  audienceEntityId: EntityId | null | undefined,
): boolean {
  const normalized = normalizeEpisodeAccess(input);

  return (
    normalized.shared !== true &&
    normalized.audience_entity_id !== null &&
    audienceEntityId !== null &&
    audienceEntityId !== undefined &&
    normalized.audience_entity_id !== audienceEntityId
  );
}

function hasAnyPrivateEpisode(input: EpisodeAccessLike): boolean {
  const normalized = normalizeEpisodeAccess(input);
  return normalized.shared !== true && normalized.audience_entity_id !== null;
}

export function inferSinglePrivateAudience(
  episodes: readonly EpisodeAccessLike[],
): EntityId | null | "multiple" {
  const privateAudiences = new Set<EntityId>();

  for (const episode of episodes) {
    const normalized = normalizeEpisodeAccess(episode);

    if (normalized.shared || normalized.audience_entity_id === null) {
      continue;
    }

    privateAudiences.add(normalized.audience_entity_id);
  }

  if (privateAudiences.size > 1) {
    return "multiple";
  }

  return [...privateAudiences][0] ?? null;
}

export function filterEpisodesByAudience(
  episodes: readonly AudienceEpisodeAccess[],
  audienceEntityId: EntityId | null,
  policy: AudiencePolicy,
): AudienceFilterResult {
  if (episodes.length === 0) {
    return {
      visibleEpisodeIds: [],
      hiddenEpisodeIds: [],
      hasPrivateMix: false,
    };
  }

  const visibleEpisodeIds = episodes
    .filter((episode) => isEpisodeAccessVisible(episode, audienceEntityId))
    .map((episode) => episode.id);
  const hiddenEpisodeIds = episodes
    .filter((episode) => !isEpisodeAccessVisible(episode, audienceEntityId))
    .map((episode) => episode.id);

  if (policy === "filter") {
    return {
      visibleEpisodeIds,
      hiddenEpisodeIds,
      hasPrivateMix: episodes.some((episode) =>
        audienceEntityId === null
          ? hasAnyPrivateEpisode(episode)
          : isPrivateToDifferentAudience(episode, audienceEntityId),
      ),
    };
  }

  const hasPrivateMix =
    policy === "all_or_nothing"
      ? episodes.some((episode) =>
          audienceEntityId === null
            ? hasAnyPrivateEpisode(episode)
            : isPrivateToDifferentAudience(episode, audienceEntityId),
        )
      : episodes.some((episode) => isPrivateToDifferentAudience(episode, audienceEntityId));

  if (hasPrivateMix) {
    return {
      visibleEpisodeIds: [],
      hiddenEpisodeIds: episodes.map((episode) => episode.id),
      hasPrivateMix: true,
    };
  }

  return {
    visibleEpisodeIds,
    hiddenEpisodeIds,
    hasPrivateMix: false,
  };
}
