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
    expect(renderedEvent?.change.excerpt_tail).toContain("_NEW_TAIL");
    expect(JSON.stringify(renderedEvent).length).toBeLessThan(5_000);
  });
});
