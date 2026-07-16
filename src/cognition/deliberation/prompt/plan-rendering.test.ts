import { describe, expect, it } from "vitest";

import type { TurnPlan } from "../s2-planner.js";
import { formatTurnPlanForThought } from "../thoughts.js";
import { formatTurnPlanForPrompt } from "./plan-rendering.js";

function plan(overrides: Partial<TurnPlan> = {}): TurnPlan {
  return {
    uncertainty: "",
    verification_steps: [],
    tensions: [],
    voice_note: "",
    emission_recommendation: "emit",
    intents: [],
    ...overrides,
  };
}

function requireRendered(text: string | null): string {
  if (text === null) {
    throw new Error("Expected rendered plan text");
  }

  return text;
}

function extractTagContent(text: string, tag: string): string {
  const openTag = `<${tag}>`;
  const closeTag = `</${tag}>`;
  const start = text.indexOf(openTag);
  const end = text.indexOf(closeTag, start);

  expect(start).toBeGreaterThanOrEqual(0);
  expect(end).toBeGreaterThan(start);

  return text.slice(start + openTag.length, end).trim();
}

describe("S2 plan rendering", () => {
  it("renders a want-only plan with Named want as the first borg_s2_plan line", () => {
    const wantOnlyPlan = plan({ want: "write down the unresolved question" });
    const rendered = requireRendered(formatTurnPlanForPrompt(wantOnlyPlan));
    const content = extractTagContent(rendered, "borg_s2_plan");

    expect(content.split("\n")[0]).toBe("Named want: write down the unresolved question");
    expect(content).toContain("S2 planner advisory:");
    expect(formatTurnPlanForThought(wantOnlyPlan)).toBe(
      "plan: want: write down the unresolved question",
    );
  });

  it("omits the want line when want is empty", () => {
    const rendered = requireRendered(
      formatTurnPlanForPrompt(plan({ want: "", uncertainty: "Which source changed?" })),
    );
    const content = extractTagContent(rendered, "borg_s2_plan");

    expect(content).not.toContain("Named want:");
    expect(content.split("\n")[0]).toBe("S2 planner advisory:");
    expect(formatTurnPlanForPrompt(plan({ want: "" }))).toBeNull();
    expect(formatTurnPlanForThought(plan({ want: "" }))).toBe("plan: (no changes needed)");
  });
});
