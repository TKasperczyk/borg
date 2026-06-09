import { afterEach, describe, expect, it, vi } from "vitest";

import type { ReviewKind, ReviewResolution, ReviewRow } from "../api/types";
import { GENERIC_REVIEW_ACTIONS, resolveReviewAction } from "./review-actions";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function requestBody(init: RequestInit | undefined): Record<string, unknown> {
  return JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
}

function reviewRow(kind: ReviewKind): ReviewRow {
  return {
    id: 42,
    kind,
    refs:
      kind === "creator_directive_reconciliation"
        ? { directive_ids: ["cdir_dispatch111111", "cdir_dispatch222222"] }
        : { node_ids: ["semn_dispatch111111", "semn_dispatch222222"] },
    reason: `${kind} dispatch`,
    created_at: 1,
    resolved_at: null,
    resolution: null,
  };
}

type RoutingCase = {
  kind: ReviewKind;
  action: ReviewResolution;
};

const DISPATCH_CASES: RoutingCase[] = [
  ...Object.entries(GENERIC_REVIEW_ACTIONS).flatMap(([kind, actions]) =>
    actions.map((action) => ({ kind: kind as ReviewKind, action })),
  ),
  { kind: "creator_directive_reconciliation", action: "keep" },
  { kind: "creator_directive_reconciliation", action: "supersede" },
];

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("resolveReviewAction", () => {
  it("covers every review kind with its sanctioned routing cases", () => {
    expect(new Set(DISPATCH_CASES.map((item) => item.kind))).toEqual(
      new Set<ReviewKind>([
        "belief_revision",
        "commitment_reconciliation",
        "contradiction",
        "correction",
        "creator_directive_reconciliation",
        "duplicate",
        "identity_inconsistency",
        "misattribution",
        "new_insight",
        "skill_split",
        "temporal_drift",
      ]),
    );
  });

  it.each(DISPATCH_CASES)(
    "routes $kind $action through the sanctioned endpoint",
    async ({ kind, action }) => {
      const row = reviewRow(kind);
      const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
        expect(init?.method).toBeDefined();
        return Promise.resolve(jsonResponse({ ...row, resolved_at: 2, resolution: action }));
      });
      vi.stubGlobal("fetch", fetchMock);

      await resolveReviewAction({
        row,
        action,
        note: "operator note",
        winnerNodeId:
          (kind === "contradiction" || kind === "duplicate") &&
          (action === "supersede" || action === "invalidate")
            ? "semn_dispatch111111"
            : undefined,
        survivorId:
          kind === "creator_directive_reconciliation" && action === "supersede"
            ? "cdir_dispatch111111"
            : undefined,
      });

      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [request, init] = fetchMock.mock.calls[0]!;
      const body = requestBody(init);

      if (kind === "correction") {
        expect(requestPath(request)).toBe("/api/correction/reviews/42");
        expect(init?.method).toBe("PATCH");
        expect(body).toEqual({ action, note: "operator note" });
        return;
      }

      if (kind === "belief_revision") {
        expect(requestPath(request)).toBe("/api/dream/review/42");
        expect(init?.method).toBe("PATCH");
        expect(body).toEqual({ action: "dismiss", note: "operator note" });
        return;
      }

      if (kind === "creator_directive_reconciliation") {
        expect(requestPath(request)).toBe("/api/reviews/42/creator-directive-reconciliation");
        expect(init?.method).toBe("POST");
        expect(body).toEqual(
          action === "supersede"
            ? {
                action: "supersede",
                survivor_id: "cdir_dispatch111111",
                reason: "operator note",
              }
            : { action: "keep", reason: "operator note" },
        );
        return;
      }

      expect(requestPath(request)).toBe("/api/reviews/42");
      expect(init?.method).toBe("PATCH");
      expect(body).toEqual({
        action,
        note: "operator note",
        ...((kind === "contradiction" || kind === "duplicate") &&
        (action === "supersede" || action === "invalidate")
          ? { winner_node_id: "semn_dispatch111111" }
          : {}),
      });
    },
  );
});
