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

  it("names the identity-event log's domain whenever the payload carries that reader", () => {
    const rendered = formatAutonomyTriggerContext({
      ...BASE,
      source_name: "scheduled_reflection",
      payload: {
        interval_ms: 14_400_000,
        recent_identity_events: [
          { id: 1, record_type: "trait", record_id: "trt_aaaaaaaaaaaaaaaa", action: "decay" },
        ],
      },
    });

    expect(rendered).toContain("not every write a record received");
    expect(rendered).toContain("record_version that write produced");
    expect(rendered).toContain("not evidence that the record did not change");
    expect(rendered.indexOf("recent_identity_events: this is the log")).toBeGreaterThan(
      rendered.indexOf('"record_id": "trt_aaaaaaaaaaaaaaaa"'),
    );
  });

  it("still names the domain when the reader returned nothing, and stays silent when it did not run", () => {
    const empty = formatAutonomyTriggerContext({
      ...BASE,
      source_name: "scheduled_reflection",
      payload: { interval_ms: 14_400_000, recent_identity_events: [] },
    });
    const absent = formatAutonomyTriggerContext({
      ...BASE,
      payload: { interval_ms: 14_400_000 },
    });

    expect(empty).toContain("not every write a record received");
    expect(absent).not.toContain("not every write a record received");
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
