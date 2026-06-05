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
const CONFLICT_SHARED_OTHER: EpisodeAccessLike = { audience_entity_id: OTHER, shared: true };
const PRIVATE_SELF_AND_OTHER: EpisodeAccessLike = {
  origin_audience_entity_ids: [SELF, OTHER],
  shared: false,
};
const UNKNOWN_ORIGIN: EpisodeAccessLike = {
  audience_entity_id: null,
  origin_audience_entity_ids: [],
  shared: false,
};
const LEGACY_PUBLIC: EpisodeAccessLike = {
  audience_entity_id: null,
  origin_audience_entity_ids: [],
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

  it("applies explicit admin/export all-audience precedence over audienceEntityId", () => {
    expect(resolveViewerCapability({ crossAudience: true, audienceEntityId: OTHER })).toEqual({
      kind: "unrestricted",
    });
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
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF_AND_OTHER, audience(SELF))).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF_AND_OTHER, audience(OTHER))).toBe(true);
    expect(isEpisodeVisibleToCapability(PRIVATE_OTHER, audience(SELF))).toBe(false);
    expect(isEpisodeVisibleToCapability(CONFLICT_SHARED_OTHER, audience(SELF))).toBe(false);
    expect(isEpisodeVisibleToCapability(CONFLICT_SHARED_OTHER, audience(OTHER))).toBe(true);
  });

  it("audience arm with null id: only public/shared, no private", () => {
    expect(isEpisodeVisibleToCapability(PUBLIC, audience(null))).toBe(true);
    expect(isEpisodeVisibleToCapability(LEGACY_PUBLIC, audience(null))).toBe(true);
    expect(isEpisodeVisibleToCapability(UNKNOWN_ORIGIN, audience(null))).toBe(false);
    expect(isEpisodeVisibleToCapability(CONFLICT_SHARED_OTHER, audience(null))).toBe(false);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF, audience(null))).toBe(false);
    expect(isEpisodeVisibleToCapability(PRIVATE_OTHER, audience(null))).toBe(false);
    expect(isEpisodeVisibleToCapability(PRIVATE_SELF_AND_OTHER, audience(null))).toBe(false);
  });

  it("fails closed for unknown-origin consolidated episodes in disclosure reads", () => {
    expect(isEpisodeVisibleToCapability(UNKNOWN_ORIGIN, audience(SELF))).toBe(false);
    expect(isEpisodeVisibleToCapability(UNKNOWN_ORIGIN, audience(OTHER))).toBe(false);
    expect(isEpisodeVisibleToCapability(PUBLIC, audience(SELF))).toBe(true);
    expect(isEpisodeVisibleToCapability(LEGACY_PUBLIC, audience(SELF))).toBe(true);
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
