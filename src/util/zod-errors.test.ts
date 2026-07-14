import { z } from "zod";
import { describe, expect, it } from "vitest";

import { formatZodErrorIssues, parseErrorMessage } from "./zod-errors.js";

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

  it("finds nested zod causes and bounds the issue count", () => {
    const result = z
      .object({
        first: z.string(),
        second: z.string(),
        third: z.string(),
      })
      .safeParse({ first: 1, second: 2, third: 3 });

    expect(result.success).toBe(false);
    if (!result.success) {
      const wrapped = new Error("wrapped", { cause: result.error });
      expect(formatZodErrorIssues(wrapped, { maxIssues: 2 })).toMatch(
        /^first: .*; second: .*; \(\+1 more issues\)$/,
      );
      expect(parseErrorMessage(wrapped, { maxIssues: 1 })).toMatch(
        /^first: .*; \(\+2 more issues\)$/,
      );
    }
  });

  it("bounds long issue messages without cutting the omitted-issue suffix", () => {
    const result = z
      .object({ first: z.string(), second: z.string() })
      .safeParse({ first: 1, second: 2 });

    expect(result.success).toBe(false);
    if (!result.success) {
      const message = formatZodErrorIssues(result.error, {
        maxIssues: 2,
        maxCharacters: 60,
      });

      expect(message).toMatch(/\(\+1 more issues\)$/);
      expect(message?.length).toBeLessThanOrEqual(60);
    }
  });
});
