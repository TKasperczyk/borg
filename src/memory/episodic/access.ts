import type { EntityId } from "../../util/ids.js";
import {
  isEpisodeAccessVisible,
  isEpisodeAccessVisibleToAnyAudience,
  normalizeEpisodeAccess,
  type EpisodeAccessLike,
} from "./audience-filter.js";

export { normalizeEpisodeAccess, type EpisodeAccessLike } from "./audience-filter.js";

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

// Disclosure/admin-only viewer capabilities for audience-filtered episodic reads. Cognition
// recall is global and must not route through this resolver; use EpisodeCognitionRecallOptions
// / CognitionRecallSearchOptions for cognition paths. For explicit disclosure/export paths there
// are three ways to read:
//   - audience: public/shared plus one exact-origin audience match
//   - audience_set: public/shared plus any exact-origin audience match in a caller-derived set
//   - unrestricted: see everything (admin/correction/export read paths ONLY)
export type ViewerCapability =
  | { readonly kind: "audience"; readonly audienceEntityId: EntityId | null }
  | { readonly kind: "audience_set"; readonly audienceEntityIds: readonly EntityId[] }
  | { readonly kind: "unrestricted" };

export type ViewerCapabilityOptions = {
  // Disclosure/admin option shape. These are intentionally not part of cognition recall.
  readonly audienceEntityId?: EntityId | null;
  readonly visibleAudienceEntityIds?: readonly EntityId[];
  readonly crossAudience?: boolean;
};

// Collapse the disclosure/admin options into ONE capability. FAIL-CLOSED by construction: the
// fallthrough is the restrictive `audience` arm, and `unrestricted` is produced ONLY by an
// explicit `crossAudience === true`. A caller that passes nothing resolves to the most restrictive
// disclosure scope (public/shared only), never see-all.
export function resolveViewerCapability(options: ViewerCapabilityOptions): ViewerCapability {
  if (options.crossAudience === true) {
    return { kind: "unrestricted" };
  }

  if (options.visibleAudienceEntityIds !== undefined) {
    return {
      kind: "audience_set",
      audienceEntityIds: [
        ...new Set([
          ...options.visibleAudienceEntityIds,
          ...(options.audienceEntityId === null || options.audienceEntityId === undefined
            ? []
            : [options.audienceEntityId]),
        ]),
      ],
    };
  }

  return { kind: "audience", audienceEntityId: options.audienceEntityId ?? null };
}

// In-memory visibility decision for a resolved capability. Exhaustive and fail-closed: an
// unrecognized capability kind throws rather than silently admitting the episode.
export function isEpisodeVisibleToCapability(
  input: EpisodeAccessLike,
  capability: ViewerCapability,
): boolean {
  switch (capability.kind) {
    case "unrestricted":
      return true;
    case "audience":
      return isEpisodeAccessVisible(input, capability.audienceEntityId);
    case "audience_set":
      return isEpisodeAccessVisibleToAnyAudience(input, capability.audienceEntityIds);
    default: {
      const exhaustive: never = capability;
      throw new Error(`unhandled ViewerCapability kind: ${JSON.stringify(exhaustive)}`);
    }
  }
}
