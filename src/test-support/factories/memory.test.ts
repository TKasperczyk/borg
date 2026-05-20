import { describe, expect, it } from "vitest";

import { makeActionRecord } from "./memory.js";

describe("memory test factories", () => {
  it("preserves explicit null overrides for nullable action timestamps", () => {
    const action = makeActionRecord({
      state: "scheduled",
      scheduled_at: null,
    });

    expect(action.state).toBe("scheduled");
    expect(action.scheduled_at).toBeNull();
  });
});
