import { describe, expect, it } from "vitest";

import { IdentityGuard } from "./guard.js";

describe("IdentityGuard", () => {
  const guard = new IdentityGuard();

  it("allows new records and review-approved writes", () => {
    expect(
      guard.evaluateChange({
        current: null,
        provenance: {
          kind: "offline",
          process: "reflector",
        },
      }),
    ).toEqual({
      allowed: true,
      requires_review: false,
      overwrite_without_review: false,
    });

    expect(
      guard.evaluateChange({
        current: {
          state: "established",
          provenance: {
            kind: "episodes",
            episode_ids: ["ep_aaaaaaaaaaaaaaaa" as const],
          },
        },
        provenance: {
          kind: "offline",
          process: "reflector",
        },
        throughReview: true,
      }),
    ).toEqual({
      allowed: true,
      requires_review: false,
      overwrite_without_review: false,
    });
  });

  it("allows non-established or non-episode-backed records without review", () => {
    expect(
      guard.evaluateChange({
        current: {
          state: "candidate",
          provenance: {
            kind: "episodes",
            episode_ids: ["ep_aaaaaaaaaaaaaaaa" as const],
          },
        },
        provenance: {
          kind: "manual",
        },
      }),
    ).toEqual({
      allowed: true,
      requires_review: false,
      overwrite_without_review: false,
    });

    expect(
      guard.evaluateChange({
        current: {
          state: "established",
          provenance: {
            kind: "system",
          },
        },
        provenance: {
          kind: "offline",
          process: "reflector",
        },
      }),
    ).toEqual({
      allowed: true,
      requires_review: false,
      overwrite_without_review: false,
    });
  });

  it("allows episode-backed evidence to update established episode-backed identity silently", () => {
    expect(
      guard.evaluateChange({
        current: {
          state: "established",
          provenance: {
            kind: "episodes",
            episode_ids: ["ep_aaaaaaaaaaaaaaaa" as const],
          },
        },
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_bbbbbbbbbbbbbbbb" as const],
        },
      }),
    ).toEqual({
      allowed: true,
      requires_review: false,
      overwrite_without_review: false,
    });
  });

  it("requires review for manual, system, and offline overwrites of established episode-backed identity", () => {
    const current = {
      state: "established" as const,
      provenance: {
        kind: "episodes" as const,
        episode_ids: ["ep_aaaaaaaaaaaaaaaa" as const],
      },
    };

    expect(
      guard.evaluateChange({
        current,
        provenance: {
          kind: "manual",
        },
      }),
    ).toEqual({
      allowed: false,
      requires_review: true,
      overwrite_without_review: false,
    });

    expect(
      guard.evaluateChange({
        current,
        provenance: {
          kind: "system",
        },
      }),
    ).toEqual({
      allowed: false,
      requires_review: true,
      overwrite_without_review: false,
    });

    expect(
      guard.evaluateChange({
        current,
        provenance: {
          kind: "offline",
          process: "reflector",
        },
      }),
    ).toEqual({
      allowed: false,
      requires_review: true,
      overwrite_without_review: false,
    });
  });
});
