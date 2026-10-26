import { describe, expect, it } from "vitest";

import { classifyActionArchiveCandidate } from "./archive-classifier.js";

describe("scheduled action inactivity", () => {
  it.each([
    { state: "scheduled" as const, scheduled_at: 1_000 },
    { state: "scheduled" as const, scheduled_at: null },
    { state: "committed_to_do" as const, scheduled_at: 1_000 },
  ])(
    "ages $state rows with scheduled_at=$scheduled_at on the inactive-turn boundary",
    (schedule) => {
      const action = {
        ...schedule,
        actor: "user" as const,
        audience_entity_id: null,
        last_referenced_turn_counter: 99,
        last_referenced_turn_global: 10,
      };
      expect(
        classifyActionArchiveCandidate(action, { turnCounter: 29, archiveAfterTurns: 20 }),
      ).toEqual({
        status: "skipped",
        reason: "below_inactive_threshold",
        inactiveTurns: 19,
      });
      expect(
        classifyActionArchiveCandidate(action, { turnCounter: 30, archiveAfterTurns: 20 }),
      ).toEqual({
        status: "eligible",
        inactiveTurns: 20,
      });
    },
  );
});
