import { describe, expect, it } from "vitest";
import { createStreamEntryId } from "../util/ids.js";
import { streamResponseToSchema, taskEventSchema } from "./types.js";

describe("task event contracts", () => {
  const stamp = {
    kind: "task_event",
    event_id: "event-1",
    event_entry_id: createStreamEntryId(),
    task_id: "task-1",
    task_version: 2,
  };
  it("round-trips a stamp answering exactly one event", () => {
    expect(streamResponseToSchema.parse(stamp)).toEqual(stamp);
    expect(
      streamResponseToSchema.safeParse({ ...stamp, source_entry_ids: [createStreamEntryId()] })
        .success,
    ).toBe(false);
    expect(streamResponseToSchema.safeParse({ ...stamp, task_version: 1.5 }).success).toBe(false);
    expect(
      streamResponseToSchema.safeParse({ ...stamp, event_entry_id: "not-a-stream-id" }).success,
    ).toBe(false);
  });
  it("preserves the existing stream_backlog stamp", () => {
    const id = createStreamEntryId();
    const backlog = {
      kind: "stream_backlog",
      from_cursor_exclusive: null,
      through_cursor_inclusive: { ts: 100, entryId: id },
      source_entry_ids: [id],
      count: 1,
    };
    expect(streamResponseToSchema.parse(backlog)).toEqual(backlog);
  });
  it("requires an offset and bounded summary, accepts all terminal outcomes", () => {
    const event = {
      schema_version: 1,
      event_id: "e",
      task_id: "t",
      task_version: 0,
      kind: "task_completed",
      occurred_at: "2026-09-06T10:30:00+02:00",
      outcome: {
        status: "succeeded",
        summary: "x".repeat(8000),
        artifacts: [{ label: "Report", url: "https://example.com/report" }],
      },
      origin: { source_entry_ids: [] },
    };
    for (const status of ["succeeded", "failed", "timed_out", "cancelled"]) {
      expect(
        taskEventSchema.safeParse({ ...event, outcome: { ...event.outcome, status } }).success,
      ).toBe(true);
    }
    expect(
      taskEventSchema.safeParse({ ...event, occurred_at: "2026-09-06T10:30:00" }).success,
    ).toBe(false);
    expect(
      taskEventSchema.safeParse({
        ...event,
        outcome: { status: "succeeded", summary: "x".repeat(8001) },
      }).success,
    ).toBe(false);
  });
});
