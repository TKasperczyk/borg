import { describe, expect, it } from "vitest";

import { parseRecallPlannerCases } from "./cases.js";

function validCase(id = "case-one") {
  return {
    id,
    focus: "A co ona wtedy wybrała?",
    context_turns: [
      { role: "user" as const, content: "Przypomnij rozmowę z Mają o Atlasie." },
      { role: "assistant" as const, content: "Maja porównała dwa sposoby wdrożenia." },
    ],
    identity: {
      memory_owner_name: "team-agent",
      current_sender_name: "Tomasz",
      current_venue: { type: "personal" as const, name: "Tomasz" },
      entity_terms: ["Maja", "Atlas"],
    },
    owner_recent_activity: [],
    expected_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
  };
}

describe("recall planner case parsing", () => {
  it("accepts the documented referential case shape", () => {
    expect(parseRecallPlannerCases([validCase()])).toEqual([validCase()]);
  });

  it("accepts a pinned clock with an explicit offset and rejects one without", () => {
    const pinned = { ...validCase(), now: "2026-09-05T12:00:00+02:00" };
    expect(parseRecallPlannerCases([pinned])).toEqual([pinned]);

    expect(() => parseRecallPlannerCases([{ ...validCase(), now: "2026-09-05T12:00:00" }])).toThrow(
      /explicit offset/,
    );
    expect(() => parseRecallPlannerCases([{ ...validCase(), now: "yesterday" }])).toThrow(
      /explicit offset/,
    );
  });

  it("rejects duplicate case IDs", () => {
    expect(() => parseRecallPlannerCases([validCase(), validCase()])).toThrow(
      /Duplicate recall-planner case id/,
    );
  });

  it("rejects an invalid conversation role and episode ID", () => {
    const malformed = {
      ...validCase(),
      context_turns: [{ role: "system", content: "not allowed" }],
      expected_episode_ids: ["not-an-episode"],
    };

    expect(() => parseRecallPlannerCases([malformed])).toThrow();
  });
});
