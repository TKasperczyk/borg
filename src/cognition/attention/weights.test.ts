import { describe, expect, it } from "vitest";

import { SuppressionSet } from "./suppression-set.js";
import { computeWeights } from "./weights.js";

describe("attention weights", () => {
  it("computes mode-conditioned weights", () => {
    const reflective = computeWeights("reflective", {
      currentGoals: [
        {
          id: "goal_aaaaaaaaaaaaaaaa" as never,
          description: "Ship release",
          priority: 1,
          parent_goal_id: null,
          status: "active",
          progress_notes: null,
          last_progress_ts: null,
          created_at: 0,
          target_at: null,
          audience_entity_id: null,
          provenance: { kind: "system" },
        },
      ],
      hasActiveValues: true,
      hasTemporalCue: true,
    });
    const idle = computeWeights("idle", {
      currentGoals: [],
      hasActiveValues: false,
      hasTemporalCue: false,
    });

    expect(reflective.goal_relevance).toBeGreaterThan(0);
    expect(reflective.value_alignment).toBeGreaterThan(0);
    expect(reflective.time).toBeGreaterThan(0);
    expect(reflective.entity).toBeGreaterThan(idle.entity);
    expect(idle.semantic).toBeLessThan(reflective.semantic);
  });

  it("expires suppression entries by turn ttl", () => {
    const suppression = new SuppressionSet(0);

    suppression.suppress("ep_1", "duplicate", 2);
    expect(suppression.isSuppressed("ep_1")).toBe(true);
    suppression.tickTurn();
    expect(suppression.isSuppressed("ep_1")).toBe(true);
    suppression.tickTurn();
    expect(suppression.isSuppressed("ep_1")).toBe(true);
    suppression.tickTurn();
    expect(suppression.isSuppressed("ep_1")).toBe(false);
  });
});
