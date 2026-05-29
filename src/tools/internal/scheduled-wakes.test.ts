import { describe, expect, it } from "vitest";

import type { ScheduledWake } from "../../autonomy/index.js";
import { createScheduledWakeId, DEFAULT_SESSION_ID, type ScheduledWakeId } from "../../util/ids.js";
import type { ToolInvocationContext } from "../dispatcher.js";
import { createScheduledWakesCancelTool } from "./scheduled-wakes-cancel.js";
import { createScheduledWakesCreateTool } from "./scheduled-wakes-create.js";
import { createScheduledWakesListTool } from "./scheduled-wakes-list.js";

function sampleWake(overrides: Partial<ScheduledWake> = {}): ScheduledWake {
  return {
    id: createScheduledWakeId(),
    fire_at: 2_000_000,
    note: "n",
    origin_session_id: null,
    status: "pending",
    created_at: 1_000_000,
    updated_at: 1_000_000,
    fired_at: null,
    cancelled_at: null,
    ...overrides,
  };
}

const context: ToolInvocationContext = {
  sessionId: DEFAULT_SESSION_ID,
  origin: "deliberator",
};

describe("scheduled wake tools", () => {
  it("create passes delay, note, and origin session to the callback", async () => {
    const calls: unknown[] = [];
    const wake = sampleWake();
    const tool = createScheduledWakesCreateTool({
      scheduleWake: (input) => {
        calls.push(input);
        return wake;
      },
    });

    const result = await tool.invoke({ delay_seconds: 120, note: "revisit X" }, context);

    expect(calls).toEqual([
      { delaySeconds: 120, note: "revisit X", originSessionId: DEFAULT_SESSION_ID },
    ]);
    expect(result.scheduledWake.id).toBe(wake.id);
    expect(tool.writeScope).toBe("write");
    expect(tool.allowedOrigins).toContain("deliberator");
  });

  it("create rejects a non-positive delay or empty note at the schema", () => {
    const tool = createScheduledWakesCreateTool({ scheduleWake: () => sampleWake() });
    expect(tool.inputSchema.safeParse({ delay_seconds: 0, note: "x" }).success).toBe(false);
    expect(tool.inputSchema.safeParse({ delay_seconds: -1, note: "x" }).success).toBe(false);
    expect(tool.inputSchema.safeParse({ delay_seconds: 5, note: "" }).success).toBe(false);
    expect(tool.inputSchema.safeParse({ delay_seconds: 5, note: "ok" }).success).toBe(true);
  });

  it("list defaults to pending", async () => {
    const calls: unknown[] = [];
    const tool = createScheduledWakesListTool({
      listScheduledWakes: (input) => {
        calls.push(input);
        return [sampleWake()];
      },
    });

    await tool.invoke({}, context);

    expect(calls).toEqual([{ status: "pending", limit: 20 }]);
    expect(tool.writeScope).toBe("read");
  });

  it("cancel parses the id and returns null when nothing pending", async () => {
    const id = createScheduledWakeId();
    const seen: ScheduledWakeId[] = [];
    const tool = createScheduledWakesCancelTool({
      cancelScheduledWake: (cancelId) => {
        seen.push(cancelId);
        return null;
      },
    });

    const result = await tool.invoke({ scheduled_wake_id: id }, context);

    expect(seen).toEqual([id]);
    expect(result.scheduledWake).toBeNull();
  });

  it("cancel rejects an invalid id at the schema", () => {
    const tool = createScheduledWakesCancelTool({ cancelScheduledWake: () => null });
    expect(tool.inputSchema.safeParse({ scheduled_wake_id: "not-an-id" }).success).toBe(false);
  });
});
