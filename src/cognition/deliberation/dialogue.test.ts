import { describe, expect, it } from "vitest";

import { createStreamEntryId } from "../../util/ids.js";
import type { RecencyMessage } from "../recency/index.js";
import { buildDialogueMessages, toContentBlockMessages } from "./dialogue.js";

function makeRecency(role: "user" | "assistant", content: string, index: number): RecencyMessage {
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

  it("merges adjacent user messages without inventing assistant output", () => {
    const messages = buildDialogueMessages([makeRecency("user", "first", 1)], "second");

    expect(messages).toEqual([{ role: "user", content: "first\n\nsecond" }]);
  });

  it("merges observed-turn markers with the surrounding user-role run", () => {
    const messages = buildDialogueMessages(
      [
        makeRecency("user", "Alice: Tuesday works.", 1),
        makeRecency(
          "user",
          "[system: borg observed turn turn-2 silently -- reason: Peer coordination.]",
          2,
        ),
      ],
      "Bob: Tuesday works for me too.",
    );

    expect(messages).toEqual([
      {
        role: "user",
        content:
          "Alice: Tuesday works.\n\n[system: borg observed turn turn-2 silently -- reason: Peer coordination.]\n\nBob: Tuesday works for me too.",
      },
    ]);
  });

  it("keeps the current user message unchanged", () => {
    const messages = buildDialogueMessages([], "Please check Atlas.");

    expect(messages).toEqual([{ role: "user", content: "Please check Atlas." }]);
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
