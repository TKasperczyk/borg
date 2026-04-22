import { describe, expect, it } from "vitest";

import { commitmentPatchSchema } from "./types.js";

describe("commitment patch schema", () => {
  it("rejects immutable commitment fields in patches", () => {
    expect(() =>
      commitmentPatchSchema.parse({
        created_at: 123,
      }),
    ).toThrow();
  });
});
