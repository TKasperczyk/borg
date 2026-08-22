import { describe, expect, it } from "vitest";

import { formatAutonomyTriggerContext } from "./autonomy-trigger.js";

const BASE = {
  source_name: "goal_followup_due",
  source_type: "trigger",
  event_id: "goal_aaaaaaaaaaaaaaaa:1787050000000:1785840551552:deadline",
  sort_ts: 1_787_050_000_000,
} as const;

describe("formatAutonomyTriggerContext epoch annotation", () => {
  it("adds a calendar sibling for payload timestamps without touching the raw field", () => {
    const rendered = formatAutonomyTriggerContext({
      ...BASE,
      payload: { target_at: 1_787_050_000_000, last_progress_ts: null, priority: 10 },
    });

    expect(rendered).toContain('"target_at": 1787050000000');
    expect(rendered).toContain('"target_at_iso": "2026-08-18T10:46:40.000Z"');
    expect(rendered).toContain('"last_progress_ts": null');
    expect(rendered).not.toContain("last_progress_ts_iso");
    expect(rendered).not.toContain("priority_iso");
  });

  it("annotates nested objects and the secondary goal batch", () => {
    const rendered = formatAutonomyTriggerContext({
      ...BASE,
      payload: {
        selected_goal: { id: "goal_aaaaaaaaaaaaaaaa", created_at: 1_785_840_551_552 },
        secondary_due_goals: [{ goal_id: "goal_bbbbbbbbbbbbbbbb", sort_ts: 1_786_556_400_000 }],
      },
    });

    expect(rendered).toContain('"created_at_iso": "2026-08-04T10:49:11.552Z"');
    expect(rendered).toContain('"sort_ts_iso": "2026-08-12T17:40:00.000Z"');
  });

  it("leaves an existing sibling and an unrepresentable instant alone", () => {
    const rendered = formatAutonomyTriggerContext({
      ...BASE,
      payload: {
        target_at: 1_787_050_000_000,
        target_at_iso: "already supplied",
        due_at: 9e15,
      },
    });

    expect(rendered).toContain('"target_at_iso": "already supplied"');
    expect(rendered).not.toContain('"target_at_iso": "2026-08-18T10:46:40.000Z"');
    expect(rendered).not.toContain("due_at_iso");
  });
});
