import { describe, expect, it } from "vitest";

import {
  DEFAULT_SESSION_ID,
  autonomyWakeIdHelpers,
  correctionTargetIdKindDescriptors,
  createAutonomyWakeId,
  createSessionId,
  createStreamEntryId,
  parseAuditId,
  parseSessionId,
  sessionIdHelpers,
  streamEntryIdHelpers,
} from "./ids.js";

describe("ids", () => {
  it("creates branded ids with stable prefixes", () => {
    const sessionId = createSessionId();
    const streamEntryId = createStreamEntryId();
    const autonomyWakeId = createAutonomyWakeId();

    expect(sessionIdHelpers.is(sessionId)).toBe(true);
    expect(streamEntryIdHelpers.is(streamEntryId)).toBe(true);
    expect(autonomyWakeIdHelpers.is(autonomyWakeId)).toBe(true);
    expect(sessionId).toMatch(/^sess_[a-z0-9]{16}$/);
    expect(streamEntryId).toMatch(/^strm_[a-z0-9]{16}$/);
    expect(autonomyWakeId).toMatch(/^autonomy_wake_[a-f0-9]{16}$/);
  });

  it("parses default and generated session ids", () => {
    expect(parseSessionId("default")).toBe(DEFAULT_SESSION_ID);

    const generated = createSessionId();
    expect(parseSessionId(generated)).toBe(generated);
    expect(() => parseSessionId("bad-session")).toThrow(/Invalid sess identifier/);
  });

  it("rejects audit ids with trailing junk", () => {
    expect(parseAuditId("12")).toBe(12);
    expect(() => parseAuditId("12junk")).toThrow(/Invalid audit identifier/);
  });

  it("registers correction target id kinds", () => {
    expect(
      correctionTargetIdKindDescriptors.map((descriptor) => ({
        kind: descriptor.kind,
        prefix: descriptor.prefix,
      })),
    ).toEqual([
      { kind: "episode", prefix: "ep_" },
      { kind: "semantic_node", prefix: "semn_" },
      { kind: "semantic_edge", prefix: "seme_" },
      { kind: "value", prefix: "val_" },
      { kind: "goal", prefix: "goal_" },
      { kind: "trait", prefix: "trt_" },
      { kind: "commitment", prefix: "cmt_" },
      { kind: "open_question", prefix: "oq_" },
    ]);
  });
});
