import { describe, expect, it } from "vitest";

import { goalPatchSchema, traitPatchSchema, valuePatchSchema } from "./types.js";

describe("self patch schemas", () => {
  it("rejects evidence field mutation in value patches", () => {
    expect(() =>
      valuePatchSchema.parse({
        evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      }),
    ).toThrow();
  });

  it("rejects evidence field mutation in trait patches", () => {
    expect(() =>
      traitPatchSchema.parse({
        evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      }),
    ).toThrow();
  });

  it("rejects immutable goal fields in patches", () => {
    expect(() =>
      goalPatchSchema.parse({
        created_at: 123,
      }),
    ).toThrow();
  });
});
