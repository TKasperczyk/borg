import { describe, expect, it } from "vitest";

import type { EntityId } from "../../util/ids.js";

import type { EpisodeAccessLike } from "./audience-filter.js";
import {
  isEpisodeVisibleToCapability,
  resolveViewerCapability,
  type ViewerCapability,
} from "./access.js";

const SELF = "ent_aaaaaaaaaaaaaaaa" as EntityId;
const OTHER = "ent_bbbbbbbbbbbbbbbb" as EntityId;

const PUBLIC: EpisodeAccessLike = { audience_entity_id: null, shared: true };
const PRIVATE_SELF: EpisodeAccessLike = { audience_entity_id: SELF, shared: false };
const PRIVATE_OTHER: EpisodeAccessLike = { audience_entity_id: OTHER, shared: false };
const SHARED_OTHER: EpisodeAccessLike = { audience_entity_id: OTHER, shared: true };
const PRIVATE_SELF_AND_OTHER: EpisodeAccessLike = {
  origin_audience_entity_ids: [SELF, OTHER],
  shared: false,
};

describe("resolveViewerCapability", () => {
  it("resolves an absent/under-specified viewer to the restrictive audience scope, never unrestricted", () => {
    expect(resolveViewerCapability({})).toEqual({ kind: "audience", audienceEntityId: null });
    expect(resolveViewerCapability({ audienceEntityId: undefined })).toEqual({
      kind: "audience",
      audienceEntityId: null,
    });
    expect(resolveViewerCapability({ crossAudience: false })).toEqual({
      kind: "audience",
      audienceEntityId: null,
    });
  });

  it("maps an audience id to the audience arm", () => {
    expect(resolveViewerCapability({ audienceEntityId: SELF })).toEqual({
      kind: "audience",
      audienceEntityId: SELF,
    });
  });

  it("produces unrestricted ONLY for an explicit crossAudience=true", () => {
    expect(resolveViewerCapability({ crossAudience: true })).toEqual({ kind: "unrestricted" });
  });

  it("maps operator introspection to a narrow self-introspection capability", () => {
    expect(resolveViewerCapability({ operatorIntrospectionSelfAudienceEntityId: SELF })).toEqual({
      kind: "operator_introspection",
      selfAudienceEntityId: SELF,
    });
    expect(resolveViewerCapability({ operatorIntrospectionSelfAudienceEntityId: null })).toEqual({
      kind: "operator_introspection",
      selfAudienceEntityId: null,
    });
  });

  it("maps a defined globalIdentity self-audience to self_continuity", () => {
    expect(resolveViewerCapability({ globalIdentitySelfAudienceEntityId: SELF })).toEqual({
      kind: "self_continuity",
      selfAudienceEntityId: SELF,
    });
    expect(resolveViewerCapability({ globalIdentitySelfAudienceEntityId: null })).toEqual({
      kind: "self_continuity",
      selfAudienceEntityId: null,
    });
  });

  it("applies the precedence self_continuity > unrestricted > audience", () => {
    // globalIdentity wins over crossAudience and audienceEntityId
    expect(
      resolveViewerCapability({
        globalIdentitySelfAudienceEntityId: SELF,
        crossAudience: true,
        audienceEntityId: OTHER,
      }),
    ).toEqual({ kind: "self_continuity", selfAudienceEntityId: SELF });
    // crossAudience wins over audienceEntityId when globalIdentity is absent
    expect(resolveViewerCapability({ crossAudience: true, audienceEntityId: OTHER })).toEqual({
      kind: "unrestricted",
    });
    // operator introspection is not widened by crossAudience.
    expect(
      resolveViewerCapability({
        operatorIntrospectionSelfAudienceEntityId: SELF,
        crossAudience: true,
        audienceEntityId: OTHER,
      }),
    ).toEqual({ kind: "operator_introspection", selfAudienceEntityId: SELF });
  });
});

describe("isEpisodeVisibleToCapability", () => {
  const audience = (id: EntityId | null): ViewerCapability => ({
    kind: "audience",
    audienceEntityId: id,
  });

  it("audience arm: sees public/shared and own-audience, never another audience's private", () => {
    expect(isEpisodeVisibleToCapability(PUBLIC, audience(SELF))).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF, audience(SELF))).toBe(true);
    expect(isEpisodeVisibleToCapability(SHARED_OTHER, audience(SELF))).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF_AND_OTHER, audience(SELF))).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF_AND_OTHER, audience(OTHER))).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_OTHER, audience(SELF))).toBe(false);
  });

  it("audience arm with null id: only public/shared, no private", () => {
    expect(isEpisodeVisibleToCapability(PUBLIC, audience(null))).toBe(true);
    expect(isEpisodeVisibleToCapability(SHARED_OTHER, audience(null))).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF, audience(null))).toBe(false);
    expect(isEpisodeVisibleToCapability(PRIVATE_OTHER, audience(null))).toBe(false);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF_AND_OTHER, audience(null))).toBe(false);
  });

  it("self_continuity arm: public (null-audience) plus the self entity's episodes only", () => {
    const cap: ViewerCapability = { kind: "self_continuity", selfAudienceEntityId: SELF };
    expect(isEpisodeVisibleToCapability(PUBLIC, cap)).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF, cap)).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF_AND_OTHER, cap)).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_OTHER, cap)).toBe(false);
    // self_continuity keys on the origin audience set (empty or self) and deliberately
    // IGNORES `shared`, so a shared episode owned by ANOTHER audience is NOT in scope.
    expect(isEpisodeVisibleToCapability(SHARED_OTHER, cap)).toBe(false);
  });

  it("self_continuity arm with null self: only public, no private at all", () => {
    const cap: ViewerCapability = { kind: "self_continuity", selfAudienceEntityId: null };
    expect(isEpisodeVisibleToCapability(PUBLIC, cap)).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF, cap)).toBe(false);
    expect(isEpisodeVisibleToCapability(PRIVATE_OTHER, cap)).toBe(false);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF_AND_OTHER, cap)).toBe(false);
  });

  it("operator_introspection arm: fail-closed and not equivalent to unrestricted", () => {
    const cap: ViewerCapability = { kind: "operator_introspection", selfAudienceEntityId: SELF };
    expect(isEpisodeVisibleToCapability(PUBLIC, cap)).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF, cap)).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF_AND_OTHER, cap)).toBe(true);
    expect(isEpisodeVisibleToCapability(SHARED_OTHER, cap)).toBe(false);
    expect(isEpisodeVisibleToCapability(PRIVATE_OTHER, cap)).toBe(false);
  });

  it("unrestricted arm: sees everything", () => {
    const cap: ViewerCapability = { kind: "unrestricted" };
    expect(isEpisodeVisibleToCapability(PUBLIC, cap)).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF, cap)).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_OTHER, cap)).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF_AND_OTHER, cap)).toBe(true);
  });

  it("throws (fail-closed) on an unknown capability kind rather than admitting the episode", () => {
    const bogus = { kind: "everyone" } as unknown as ViewerCapability;
    expect(() => isEpisodeVisibleToCapability(PRIVATE_OTHER, bogus)).toThrow(
      /unhandled ViewerCapability/,
    );
  });
});
