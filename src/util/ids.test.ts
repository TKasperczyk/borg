import { describe, expect, it } from "vitest";

import {
  DEFAULT_SESSION_ID,
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

    expect(sessionIdHelpers.is(sessionId)).toBe(true);
    expect(streamEntryIdHelpers.is(streamEntryId)).toBe(true);
    expect(sessionId).toMatch(/^sess_[a-z0-9]{16}$/);
    expect(streamEntryId).toMatch(/^strm_[a-z0-9]{16}$/);
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
});
