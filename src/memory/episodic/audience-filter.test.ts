import { describe, expect, it } from "vitest";

import type { EntityId, EpisodeId } from "../../util/ids.js";

import {
  filterEpisodesByAudience,
  inferSinglePrivateAudience,
  type AudienceEpisodeAccess,
} from "./audience-filter.js";

const AUDIENCE_A = "ent_aaaaaaaaaaaaaaaa" as EntityId;
const AUDIENCE_B = "ent_bbbbbbbbbbbbbbbb" as EntityId;
const PUBLIC_EPISODE = "ep_aaaaaaaaaaaaaaaa" as EpisodeId;
const PRIVATE_A_EPISODE = "ep_bbbbbbbbbbbbbbbb" as EpisodeId;
const PRIVATE_B_EPISODE = "ep_cccccccccccccccc" as EpisodeId;
const SHARED_B_EPISODE = "ep_dddddddddddddddd" as EpisodeId;

function episode(
  id: EpisodeId,
  audienceEntityId: EntityId | null,
  shared?: boolean,
): AudienceEpisodeAccess {
  return {
    id,
    audience_entity_id: audienceEntityId,
    ...(shared === undefined ? {} : { shared }),
  };
}

describe("filterEpisodesByAudience", () => {
  describe("filter policy", () => {
    it("returns empty output for an empty episode list", () => {
      expect(filterEpisodesByAudience([], AUDIENCE_A, "filter")).toEqual({
        visibleEpisodeIds: [],
        hiddenEpisodeIds: [],
        hasPrivateMix: false,
      });
    });

    it("keeps all episodes visible to the requested audience", () => {
      expect(
        filterEpisodesByAudience(
          [
            episode(PUBLIC_EPISODE, null),
            episode(PRIVATE_A_EPISODE, AUDIENCE_A),
            episode(SHARED_B_EPISODE, AUDIENCE_B, true),
          ],
          AUDIENCE_A,
          "filter",
        ),
      ).toEqual({
        visibleEpisodeIds: [PUBLIC_EPISODE, PRIVATE_A_EPISODE, SHARED_B_EPISODE],
        hiddenEpisodeIds: [],
        hasPrivateMix: false,
      });
    });

    it("keeps visible episodes and reports hidden private episodes from other audiences", () => {
      expect(
        filterEpisodesByAudience(
          [
            episode(PUBLIC_EPISODE, null),
            episode(PRIVATE_A_EPISODE, AUDIENCE_A),
            episode(PRIVATE_B_EPISODE, AUDIENCE_B),
          ],
          AUDIENCE_A,
          "filter",
        ),
      ).toEqual({
        visibleEpisodeIds: [PUBLIC_EPISODE, PRIVATE_A_EPISODE],
        hiddenEpisodeIds: [PRIVATE_B_EPISODE],
        hasPrivateMix: true,
      });
    });

    it("treats null audience as public-only visibility", () => {
      expect(
        filterEpisodesByAudience(
          [episode(PUBLIC_EPISODE, null), episode(PRIVATE_A_EPISODE, AUDIENCE_A)],
          null,
          "filter",
        ),
      ).toEqual({
        visibleEpisodeIds: [PUBLIC_EPISODE],
        hiddenEpisodeIds: [PRIVATE_A_EPISODE],
        hasPrivateMix: true,
      });
    });
  });

  describe("reject_if_mixed policy", () => {
    it("keeps all episodes visible to the requested audience", () => {
      expect(
        filterEpisodesByAudience(
          [episode(PUBLIC_EPISODE, null), episode(PRIVATE_A_EPISODE, AUDIENCE_A)],
          AUDIENCE_A,
          "reject_if_mixed",
        ),
      ).toEqual({
        visibleEpisodeIds: [PUBLIC_EPISODE, PRIVATE_A_EPISODE],
        hiddenEpisodeIds: [],
        hasPrivateMix: false,
      });
    });

    it("hides every episode when a private episode belongs to a different audience", () => {
      expect(
        filterEpisodesByAudience(
          [episode(PUBLIC_EPISODE, null), episode(PRIVATE_B_EPISODE, AUDIENCE_B)],
          AUDIENCE_A,
          "reject_if_mixed",
        ),
      ).toEqual({
        visibleEpisodeIds: [],
        hiddenEpisodeIds: [PUBLIC_EPISODE, PRIVATE_B_EPISODE],
        hasPrivateMix: true,
      });
    });
  });

  describe("all_or_nothing policy", () => {
    it("matches reject_if_mixed for explicit cross-audience private episodes", () => {
      expect(
        filterEpisodesByAudience(
          [episode(PUBLIC_EPISODE, null), episode(PRIVATE_B_EPISODE, AUDIENCE_B)],
          AUDIENCE_A,
          "all_or_nothing",
        ),
      ).toEqual({
        visibleEpisodeIds: [],
        hiddenEpisodeIds: [PUBLIC_EPISODE, PRIVATE_B_EPISODE],
        hasPrivateMix: true,
      });
    });

    it("is stricter than reject_if_mixed for null audience requests", () => {
      const episodes = [episode(PUBLIC_EPISODE, null), episode(PRIVATE_A_EPISODE, AUDIENCE_A)];

      expect(filterEpisodesByAudience(episodes, null, "reject_if_mixed")).toEqual({
        visibleEpisodeIds: [PUBLIC_EPISODE],
        hiddenEpisodeIds: [PRIVATE_A_EPISODE],
        hasPrivateMix: false,
      });
      expect(filterEpisodesByAudience(episodes, null, "all_or_nothing")).toEqual({
        visibleEpisodeIds: [],
        hiddenEpisodeIds: [PUBLIC_EPISODE, PRIVATE_A_EPISODE],
        hasPrivateMix: true,
      });
    });
  });
});

describe("inferSinglePrivateAudience", () => {
  it("returns null for public-only episode sets", () => {
    expect(inferSinglePrivateAudience([episode(PUBLIC_EPISODE, null)])).toBeNull();
  });

  it("returns the single private audience when one is present", () => {
    expect(
      inferSinglePrivateAudience([
        episode(PUBLIC_EPISODE, null),
        episode(PRIVATE_A_EPISODE, AUDIENCE_A),
      ]),
    ).toBe(AUDIENCE_A);
  });

  it("returns multiple when private audiences differ", () => {
    expect(
      inferSinglePrivateAudience([
        episode(PRIVATE_A_EPISODE, AUDIENCE_A),
        episode(PRIVATE_B_EPISODE, AUDIENCE_B),
      ]),
    ).toBe("multiple");
  });
});
