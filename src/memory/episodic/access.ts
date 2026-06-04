import type { EntityId } from "../../util/ids.js";
import {
  isEpisodeAccessVisible,
  normalizeEpisodeAccess,
  type EpisodeAccessLike,
} from "./audience-filter.js";

export { normalizeEpisodeAccess, type EpisodeAccessLike } from "./audience-filter.js";

export function episodeAccessScopeKey(input: EpisodeAccessLike): string {
  const normalized = normalizeEpisodeAccess(input);
  const origins =
    normalized.origin_audience_entity_ids.length === 0
      ? "public"
      : [...normalized.origin_audience_entity_ids].sort().join("+");

  return `${origins}:${normalized.shared ? "shared" : "private"}`;
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
    normalized.origin_audience_entity_ids.length === 0 ||
    (selfAudienceEntityId !== null &&
      selfAudienceEntityId !== undefined &&
      normalized.origin_audience_entity_ids.includes(selfAudienceEntityId))
  );
}

// The single, total description of how a reader is allowed to see episodes. Replaces the
// implicit precedence between the audienceEntityId / crossAudience /
// globalIdentitySelfAudienceEntityId option triple that was previously re-derived at every
// visibility site. There are exactly three ways to read:
//   - audience: the normal firewall path (exact-audience match + public/shared)
//   - self_continuity: public episodes plus the self/identity entity's episodes
//   - operator_introspection: public plus self-scope episodes for operator introspection
//   - unrestricted: see everything (admin/correction read paths ONLY)
export type ViewerCapability =
  | { readonly kind: "audience"; readonly audienceEntityId: EntityId | null }
  | { readonly kind: "self_continuity"; readonly selfAudienceEntityId: EntityId | null }
  | { readonly kind: "operator_introspection"; readonly selfAudienceEntityId: EntityId | null }
  | { readonly kind: "unrestricted" };

export type ViewerCapabilityOptions = {
  readonly audienceEntityId?: EntityId | null;
  readonly crossAudience?: boolean;
  readonly globalIdentitySelfAudienceEntityId?: EntityId | null;
  readonly operatorIntrospectionSelfAudienceEntityId?: EntityId | null;
};

// Collapse the legacy option triple into ONE capability, applying the visibility precedence
// exactly once: self_continuity wins over unrestricted wins over audience. FAIL-CLOSED by
// construction -- the fallthrough is the restrictive `audience` arm, and `unrestricted` is
// produced ONLY by an explicit `crossAudience === true`. A caller that passes nothing
// resolves to the most restrictive audience scope (public/shared only), never see-all.
export function resolveViewerCapability(options: ViewerCapabilityOptions): ViewerCapability {
  if (options.globalIdentitySelfAudienceEntityId !== undefined) {
    return {
      kind: "self_continuity",
      selfAudienceEntityId: options.globalIdentitySelfAudienceEntityId,
    };
  }

  if (options.operatorIntrospectionSelfAudienceEntityId !== undefined) {
    return {
      kind: "operator_introspection",
      selfAudienceEntityId: options.operatorIntrospectionSelfAudienceEntityId,
    };
  }

  if (options.crossAudience === true) {
    return { kind: "unrestricted" };
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
    case "self_continuity":
      return isEpisodeInGlobalIdentityScope(input, capability.selfAudienceEntityId);
    case "operator_introspection":
      return isEpisodeInGlobalIdentityScope(input, capability.selfAudienceEntityId);
    case "audience":
      return isEpisodeAccessVisible(input, capability.audienceEntityId);
    default: {
      const exhaustive: never = capability;
      throw new Error(`unhandled ViewerCapability kind: ${JSON.stringify(exhaustive)}`);
    }
  }
}
