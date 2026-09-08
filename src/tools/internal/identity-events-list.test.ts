import { describe, expect, it } from "vitest";

import {
  IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS,
  type IdentityEvent,
} from "../../memory/identity/index.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";

import { createIdentityEventsListForCognitionTool } from "./identity-events-list.js";

describe("identity-events cognition presentation", () => {
  it("replaces oversized record snapshots with a bounded mechanical old-to-new diff", async () => {
    const event: IdentityEvent = {
      id: 1,
      record_type: "goal",
      record_id: "goal_aaaaaaaaaaaaaaaa",
      action: "update",
      old_value: {
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "Keep the service healthy.",
        progress_notes: `OLD_HEAD_${"o".repeat(110_000)}_OLD_TAIL`,
        record_version: 8,
      },
      new_value: {
        id: "goal_aaaaaaaaaaaaaaaa",
        description: "Keep the service healthy.",
        progress_notes: `NEW_HEAD_${"n".repeat(115_000)}_NEW_TAIL`,
        record_version: 9,
      },
      reason: "Recorded structural progress.",
      provenance: { kind: "offline", process: "test-reflector" },
      review_item_id: null,
      overwrite_without_review: false,
      ts: 1_787_050_000_000,
    };
    const tool = createIdentityEventsListForCognitionTool({
      listEvents: () => [event],
    });

    const output = await tool.invoke(
      { limit: 10 },
      { sessionId: DEFAULT_SESSION_ID, origin: "autonomous" },
    );
    const parsed = tool.outputSchema.parse(output);
    const renderedEvent = parsed.events[0];

    expect(renderedEvent).not.toHaveProperty("old_value");
    expect(renderedEvent).not.toHaveProperty("new_value");
    expect(renderedEvent).toMatchObject({
      id: 1,
      record_type: "goal",
      record_id: "goal_aaaaaaaaaaaaaaaa",
      action: "update",
      reason: "Recorded structural progress.",
      provenance: { kind: "offline", process: "test-reflector" },
      ts: 1_787_050_000_000,
      change: {
        format: "top_level_fields_old_to_new",
        changed_fields: ["progress_notes", "record_version"],
        excerpt_exact: false,
      },
    });
    expect(
      (renderedEvent?.change.excerpt_head.length ?? 0) +
        (renderedEvent?.change.excerpt_tail?.length ?? 0),
    ).toBeLessThanOrEqual(IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS);
    expect(renderedEvent?.change.excerpt_chars).toBe(
      (renderedEvent?.change.excerpt_head.length ?? 0) +
        (renderedEvent?.change.excerpt_tail?.length ?? 0),
    );
    expect(renderedEvent?.change.source_chars).toBeGreaterThan(220_000);
    expect(renderedEvent?.change.excerpt_head).toContain("OLD_HEAD_");
    expect(
      (renderedEvent?.change.excerpt_head ?? "") + (renderedEvent?.change.excerpt_tail ?? ""),
    ).toContain("_NEW_TAIL");
    expect(JSON.stringify(renderedEvent).length).toBeLessThan(5_000);
  });

  it.each(["x", '\n"\\😀'])(
    "preserves small scalar comparisons between oversized changed field values (%j)",
    async (padding) => {
      const largeFields = ["description", "progress_notes", "terminal_condition"];
      const oldValue = {
        ...Object.fromEntries(
          largeFields.map((field) => [field, `OLD_HEAD_${padding.repeat(100_000)}_OLD_TAIL`]),
        ),
        priority: 1,
        record_version: 8,
        status: "active",
      };
      const newValue = {
        ...Object.fromEntries(
          largeFields.map((field) => [field, `NEW_HEAD_${padding.repeat(100_000)}_NEW_TAIL`]),
        ),
        priority: 10,
        record_version: 9,
        status: "blocked",
      };
      const tool = createIdentityEventsListForCognitionTool({
        listEvents: () => [
          {
            id: 1,
            record_type: "goal",
            record_id: "goal_aaaaaaaaaaaaaaaa",
            action: "update",
            old_value: oldValue,
            new_value: newValue,
            reason: "Updated the goal.",
            provenance: { kind: "manual" },
            review_item_id: null,
            overwrite_without_review: false,
            ts: 1_787_050_000_000,
          },
        ],
      });
      const output = tool.outputSchema.parse(
        await tool.invoke({}, { sessionId: DEFAULT_SESSION_ID, origin: "autonomous" }),
      );
      const change = output.events[0]!.change;
      const excerpt = change.excerpt_head + (change.excerpt_tail ?? "");
      expect(excerpt.length).toBeLessThanOrEqual(IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS);
      expect(change.excerpt_chars).toBe(excerpt.length);
      expect(change.excerpt_exact).toBe(false);
      expect(change.changed_fields).toEqual(Object.keys(oldValue).sort());
      const { field_changes: fields } = JSON.parse(excerpt);
      for (const field of ["priority", "record_version", "status"] as const) {
        expect(fields[field]).toEqual({
          old: { present: true, value: oldValue[field] },
          new: { present: true, value: newValue[field] },
        });
      }
      for (const field of largeFields) {
        for (const side of ["old", "new"] as const) {
          const value = fields[field][side];
          expect(value.present).toBe(true);
          expect(value).not.toHaveProperty("value");
          expect(value.value_excerpt.head).toContain(`${side.toUpperCase()}_HEAD_`);
          expect(value.value_excerpt.tail).toContain(`_${side.toUpperCase()}_TAIL`);
          expect(value.value_excerpt.source_chars).toBeGreaterThan(100_000);
        }
      }
    },
  );

  it("reports whole field omissions when change structure alone exceeds the budget", async () => {
    const fields = Array.from(
      { length: 40 },
      (_, index) => `field_${String(index).padStart(2, "0")}`,
    );
    const tool = createIdentityEventsListForCognitionTool({
      listEvents: () => [
        {
          id: 1,
          record_type: "goal",
          record_id: "goal_aaaaaaaaaaaaaaaa",
          action: "update",
          old_value: Object.fromEntries(fields.map((field) => [field, 1])),
          new_value: Object.fromEntries(fields.map((field) => [field, 10])),
          reason: "Updated fields.",
          provenance: { kind: "manual" },
          review_item_id: null,
          overwrite_without_review: false,
          ts: 1_787_050_000_000,
        },
      ],
    });
    const output = tool.outputSchema.parse(
      await tool.invoke({}, { sessionId: DEFAULT_SESSION_ID, origin: "autonomous" }),
    );
    const change = output.events[0]!.change;
    const excerpt = JSON.parse(change.excerpt_head);
    expect(change.changed_fields).toEqual(fields);
    expect(change.excerpt_chars).toBeLessThanOrEqual(
      IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS,
    );
    expect(excerpt.omitted_fields).toBeGreaterThan(0);
    expect(excerpt.omitted_fields).toBe(fields.length - excerpt.changed_fields.length);
    expect(excerpt.changed_fields).toEqual(fields.slice(0, excerpt.changed_fields.length));
    expect(Object.keys(excerpt.field_changes)).toEqual(excerpt.changed_fields);
    for (const field of excerpt.changed_fields) {
      expect(excerpt.field_changes[field]).toEqual({
        old: { present: true, value: 1 },
        new: { present: true, value: 10 },
      });
    }
  });
});
