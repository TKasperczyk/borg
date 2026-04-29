import { describe, expect, it } from "vitest";

import { createStreamEntryId } from "../../util/ids.js";
import type { RecencyMessage } from "../recency/index.js";
import { buildDialogueMessages, toContentBlockMessages } from "./dialogue.js";

function makeRecency(
  role: "user" | "assistant",
  content: string,
  index: number,
): RecencyMessage {
  return {
    role,
    content,
    stream_entry_id: createStreamEntryId(),
    ts: 1_000 + index,
  };
}

describe("buildDialogueMessages", () => {
  it("filters out empty recency entries that would produce empty text blocks", () => {
    const messages = buildDialogueMessages(
      [
        makeRecency("user", "hello", 1),
        makeRecency("assistant", "", 2),
        makeRecency("user", "   ", 3),
        makeRecency("assistant", "ack", 4),
      ],
      "current",
    );

    expect(messages).toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "ack" },
      { role: "user", content: "current" },
    ]);
  });

  it("substitutes a placeholder when the current user message is empty", () => {
    const messages = buildDialogueMessages([], "");

    expect(messages).toEqual([{ role: "user", content: "(no content)" }]);
  });

  it("preserves non-empty content as-is", () => {
    const messages = buildDialogueMessages([makeRecency("user", "first", 1)], "second");

    expect(messages).toEqual([
      { role: "user", content: "first" },
      { role: "user", content: "second" },
    ]);
  });
});

describe("toContentBlockMessages", () => {
  it("substitutes a placeholder for empty content blocks", () => {
    const blocks = toContentBlockMessages([
      { role: "user", content: "" },
      { role: "assistant", content: "real" },
    ]);

    expect(blocks).toEqual([
      { role: "user", content: [{ type: "text", text: "(no content)" }] },
      { role: "assistant", content: [{ type: "text", text: "real" }] },
    ]);
  });
});
