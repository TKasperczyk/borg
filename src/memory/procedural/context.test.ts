import { describe, expect, it } from "vitest";

import { deriveProceduralContextKey, proceduralContextSchema } from "./context.js";

describe("procedural context schema", () => {
  it("accepts canonical v2 context keys that match normalized metadata", () => {
    const input = {
      problem_kind: "code_debugging" as const,
      domain_tags: ["TypeScript", "typescript", "deploy"],
      audience_scope: "self" as const,
    };
    const parsed = proceduralContextSchema.parse({
      ...input,
      context_key: deriveProceduralContextKey(input),
    });

    expect(parsed).toEqual({
      problem_kind: "code_debugging",
      domain_tags: ["typescript", "deploy"],
      audience_scope: "self",
      context_key: deriveProceduralContextKey({
        ...input,
        domain_tags: ["typescript", "deploy"],
      }),
    });
  });

  it("rejects non-v2 and mismatched context keys", () => {
    const input = {
      problem_kind: "code_debugging" as const,
      domain_tags: ["typescript"],
      audience_scope: "self" as const,
    };

    expect(() =>
      proceduralContextSchema.parse({
        ...input,
        context_key: "code_debugging:typescript:self",
      }),
    ).toThrow();
    expect(() =>
      proceduralContextSchema.parse({
        ...input,
        context_key: deriveProceduralContextKey({
          ...input,
          audience_scope: "known_other",
        }),
      }),
    ).toThrow(/does not match metadata/u);
  });
});
