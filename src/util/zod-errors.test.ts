import { z } from "zod";
import { describe, expect, it } from "vitest";

import { parseErrorMessage } from "./zod-errors.js";

describe("parseErrorMessage", () => {
  it("formats Zod issue paths and root issues", () => {
    const schema = z.object({
      nested: z.object({
        value: z.string(),
      }),
    });

    const result = schema.safeParse({ nested: { value: 1 } });

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(parseErrorMessage(result.error)).toContain("nested.value:");
    }

    const rootResult = z.string().safeParse(1);
    expect(rootResult.success).toBe(false);
    if (!rootResult.success) {
      expect(parseErrorMessage(rootResult.error)).toContain("(root):");
    }
  });

  it("falls back to Error.message and String(error)", () => {
    expect(parseErrorMessage(new Error("plain error"))).toBe("plain error");
    expect(parseErrorMessage(42)).toBe("42");
  });
});
