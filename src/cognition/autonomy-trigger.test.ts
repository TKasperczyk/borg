import { describe, expect, it } from "vitest";

import { createIdentityEventsListForCognitionTool } from "../tools/index.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";

import {
  AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS,
  formatAutonomyTriggerContext,
} from "./autonomy-trigger.js";

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

  it.each([false, true])(
    "preserves selected event metadata under pressure (oversized metadata=%s)",
    async (oversizedMetadata) => {
      const tool = createIdentityEventsListForCognitionTool({
        listEvents: () =>
          Array.from({ length: 10 }, (_, index) => ({
            id: index + 1,
            record_type: "goal" as const,
            record_id: `goal_${String(index).padStart(16, "a")}`,
            action: "update",
            old_value: {
              progress_notes: `OLD_${index}_${"o".repeat(110_000)}`,
              record_version: index + 1,
            },
            new_value: {
              progress_notes: `${"n".repeat(115_000)}_NEW_${index}`,
              record_version: index + 2,
            },
            reason: oversizedMetadata
              ? `Reason ${index}: ${"r".repeat(6_000)}`
              : `Scheduled identity maintenance ${index}.`,
            provenance: { kind: "offline" as const, process: "test-reflector" },
            review_item_id: null,
            overwrite_without_review: false,
            ts: 1_787_050_000_000 + index,
          })),
      });
      const output = await tool.invoke(
        { limit: 10 },
        { sessionId: DEFAULT_SESSION_ID, origin: "autonomous" },
      );
      const rendered = formatAutonomyTriggerContext({
        ...BASE,
        source_name: "scheduled_reflection",
        payload: {
          interval_ms: 14_400_000,
          recent_identity_events: output.events,
          prior_self_thought: { text: "thought".repeat(2_000), updated_at: BASE.sort_ts },
          oversized_structural_tail: "z".repeat(100_000),
        },
      });

      expect(rendered.length).toBeLessThanOrEqual(AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS);
      expect(rendered.match(/excerpt_notice:/g)).toHaveLength(1);
      expect(rendered).toContain("mechanically bounded to 32000 chars");
      expect(rendered).toContain("old-to-new change excerpt is bounded to 1500 chars");
      expect(rendered).toContain('"record_type": "goal"');
      expect(rendered).toContain('"excerpt_exact": false');
      expect(rendered).toContain("not every write a record received");
      expect(rendered).not.toContain("o".repeat(10_000));
      expect(rendered).not.toContain("n".repeat(10_000));

      const payloadStart = rendered.indexOf("payload:\n") + "payload:\n".length;
      const payloadEnd = rendered.indexOf("\nnote on recent_identity_events:");
      const payload = JSON.parse(rendered.slice(payloadStart, payloadEnd));
      const selected = payload.recent_identity_events as typeof output.events;
      if (oversizedMetadata) {
        expect(selected.length).toBeGreaterThan(0);
        expect(selected.length).toBeLessThan(output.events.length);
      } else {
        expect(selected).toHaveLength(output.events.length);
      }
      expect(rendered).toContain(
        `recent_identity_events_omitted: ${output.events.length - selected.length}`,
      );
      for (const [index, event] of selected.entries()) {
        const { change: _change, ...metadata } = output.events[index]!;
        expect(event).toMatchObject(metadata);
      }
    },
  );

  it("names what an old sort_ts on a dormant-question wake does and does not mean", () => {
    const rendered = formatAutonomyTriggerContext({
      source_name: "open_question_dormant",
      source_type: "trigger",
      event_id: "oq_aaaaaaaaaaaaaaaa:1787050000000",
      sort_ts: 1_787_050_000_000,
      payload: {
        open_question_id: "oq_aaaaaaaaaaaaaaaa",
        question: "What is the right autonomy cadence?",
        urgency: 0.6,
        last_touched: 1_787_050_000_000,
        unresolved_rumination_ticks: 0,
        last_ruminated_at: null,
      },
    });

    expect(rendered).toContain("cannot wake me a second time");
    expect(rendered).toContain("never a count of how often it has already woken me");
    expect(rendered).toContain("leaves the pair unlatched and can return");
    expect(rendered).toContain("which this wake neither feeds nor writes");
    expect(rendered.indexOf("note on this dormant-question wake")).toBeGreaterThan(
      rendered.indexOf('"last_ruminated_at": null'),
    );
  });

  it("stays silent on dormancy for a wake that carries no open question", () => {
    const rendered = formatAutonomyTriggerContext({
      ...BASE,
      payload: { goal_id: "goal_aaaaaaaaaaaaaaaa", last_touched: 1_787_050_000_000 },
    });

    expect(rendered).not.toContain("note on this dormant-question wake");
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
