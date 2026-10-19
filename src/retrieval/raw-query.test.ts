import { describe, expect, it } from "vitest";

import { boundRawRecallQuery, DEFAULT_RAW_QUERY_MAX_BYTES } from "./raw-query.js";

describe("boundRawRecallQuery", () => {
  it.each(["Short query", "  Zażółć gęślą jaźń\n", "中文", "🦉🚀", "a\ud800b"])(
    "preserves short queries byte-for-byte: %j",
    (query) => {
      expect(
        boundRawRecallQuery(query, Buffer.byteLength(query), ["different", "messages"]),
      ).toEqual({
        query,
        bounded: false,
        mode: "unchanged",
        originalBytes: Buffer.byteLength(query),
        queryBytes: Buffer.byteLength(query),
        selectedMessageCount: 0,
      });
    },
  );

  it("defaults to a 6 KiB tail for a long single query", () => {
    const query = "old".repeat(20_000) + "newest";
    const result = boundRawRecallQuery(query);
    expect(DEFAULT_RAW_QUERY_MAX_BYTES).toBe(6_144);
    expect(result.query).toBe(query.slice(-6_144));
    expect(result).toMatchObject({ bounded: true, mode: "query_tail", queryBytes: 6_144 });
  });

  it.each([
    { query: "older-tail", limit: 4, expected: "tail" },
    { query: "olderZażółć", limit: 8, expected: "żółć" },
    { query: "前面中文", limit: 6, expected: "中文" },
    { query: "前面中文", limit: 5, expected: "文" },
    { query: "prefix🦉x", limit: 5, expected: "🦉x" },
    { query: "prefix🦉x", limit: 4, expected: "x" },
    { query: "prefix🦉", limit: 3, expected: "" },
    { query: "prefix👩‍💻", limit: 7, expected: "‍💻" },
    { query: "old\ud83c", limit: 3, expected: "\ufffd" },
    { query: "old\ud83c", limit: 2, expected: "" },
    { query: "old\udd95x", limit: 4, expected: "\ufffdx" },
  ])("keeps a code-point-safe tail within $limit bytes of $query", ({ query, limit, expected }) => {
    const result = boundRawRecallQuery(query, limit);
    expect(result.query).toBe(expected);
    expect(Buffer.byteLength(result.query)).toBeLessThanOrEqual(limit);
    expect(Buffer.from(result.query).toString("utf8")).toBe(result.query);
  });

  it("selects whole structured messages newest-first and counts separator bytes", () => {
    const messages = Object.freeze(["oldest", "older", "newest"]);
    const query = messages.join("\n\n");
    expect(boundRawRecallQuery(query, 13, messages)).toMatchObject({
      query: "newest\n\nolder",
      mode: "messages",
      queryBytes: 13,
      selectedMessageCount: 2,
    });
    expect(boundRawRecallQuery(query, 12, messages)).toMatchObject({
      query: "newest",
      selectedMessageCount: 1,
    });
    expect(messages).toEqual(["oldest", "older", "newest"]);
  });

  it("stops at the first older message that does not fit", () => {
    const messages = ["small", "long middle message", "newest"];
    expect(boundRawRecallQuery(messages.join("\n\n"), 13, messages).query).toBe("newest");
  });

  it("tails only the newest message when that message alone exceeds the cap", () => {
    const messages = ["older", "long newest 前面中文🦉"];
    expect(boundRawRecallQuery(messages.join("\n\n"), 10, messages)).toMatchObject({
      query: "中文🦉",
      mode: "message_tail",
      queryBytes: 10,
      selectedMessageCount: 1,
    });
  });

  it("retains an exactly-at-limit newest message whole", () => {
    const messages = ["earlier", "Zażółć"];
    expect(boundRawRecallQuery(messages.join("\n\n"), 10, messages)).toMatchObject({
      query: "Zażółć",
      mode: "messages",
      queryBytes: 10,
    });
  });

  it("repairs lone surrogates in bounded whole-message copies without mutating sources", () => {
    const messages = Object.freeze(["older message", "x\ud800"]);
    expect(boundRawRecallQuery(messages.join("\n\n"), 4, messages).query).toBe("x\ufffd");
    expect(messages[1]).toBe("x\ud800");
  });

  it("uses the string tail when no structured messages are available", () => {
    expect(boundRawRecallQuery("longer query", 5, []).query).toBe("query");
  });
});
