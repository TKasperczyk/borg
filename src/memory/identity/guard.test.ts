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
    });

    expect(
      guard.evaluateChange({
        current: {
          state: "established",
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
    });
  });

  it("allows non-established records without review", () => {
    expect(
      guard.evaluateChange({
        current: {
          state: "candidate",
        },
        provenance: {
          kind: "manual",
        },
      }),
    ).toEqual({
      allowed: true,
      requires_review: false,
    });
  });

  it("allows episode-backed evidence to update established identity silently", () => {
    expect(
      guard.evaluateChange({
        current: {
          state: "established",
        },
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_bbbbbbbbbbbbbbbb" as const],
        },
      }),
    ).toEqual({
      allowed: true,
      requires_review: false,
    });
  });

  it("requires review for manual, system, and offline overwrites of established identity", () => {
    const current = {
      state: "established" as const,
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
    });
  });
});
