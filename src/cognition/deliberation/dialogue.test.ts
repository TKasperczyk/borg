import { describe, expect, it } from "vitest";

import { createStreamEntryId, type AttachmentId } from "../../util/ids.js";
import { renderEvidenceLedger } from "../evidence-ledger/index.js";
import type { RecencyMessage } from "../recency/index.js";
import {
  buildDialogueMessages,
  toContentBlockMessages,
  withFinalizerImageBudget,
  withCurrentUserContentBlocks,
  withLedgerImageContentBlocks,
} from "./dialogue.js";

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
        makeRecency("user", "[borg observation: Peer coordination.]", 2),
      ],
      "Bob: Tuesday works for me too.",
    );

    expect(messages).toEqual([
      {
        role: "user",
        content:
          "Alice: Tuesday works.\n\n[borg observation: Peer coordination.]\n\nBob: Tuesday works for me too.",
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

describe("withCurrentUserContentBlocks", () => {
  it("preserves adjacent prior user recency when current turn has images", () => {
    const messages = toContentBlockMessages(
      buildDialogueMessages([makeRecency("user", "prior user turn", 1)], "current user turn"),
    );
    const attachmentId = "att_aaaaaaaaaaaaaaaa" as never;

    expect(
      withCurrentUserContentBlocks(messages, [
        { type: "text", text: "current user turn" },
        { type: "image_ref", attachment_id: attachmentId },
      ]),
    ).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "prior user turn" },
          { type: "image_ref", attachment_id: attachmentId },
          { type: "text", text: "current user turn" },
        ],
      },
    ]);
  });

  it("orders current-turn images before text at the caller boundary", () => {
    const messages = toContentBlockMessages(buildDialogueMessages([], "compare these"));
    const firstAttachmentId = "att_aaaaaaaaaaaaaaaa" as AttachmentId;
    const secondAttachmentId = "att_bbbbbbbbbbbbbbbb" as AttachmentId;

    expect(
      withCurrentUserContentBlocks(messages, [
        { type: "text", text: "compare these" },
        { type: "image_ref", attachment_id: firstAttachmentId },
        { type: "image_ref", attachment_id: secondAttachmentId },
      ]),
    ).toEqual([
      {
        role: "user",
        content: [
          { type: "image_ref", attachment_id: firstAttachmentId },
          { type: "image_ref", attachment_id: secondAttachmentId },
          { type: "text", text: "compare these" },
        ],
      },
    ]);
  });
});

describe("withLedgerImageContentBlocks", () => {
  it("appends retrieved image refs with stable labels for the finalizer", () => {
    const attachmentId = "att_aaaaaaaaaaaaaaaa";
    const messages = toContentBlockMessages(buildDialogueMessages([], "What was in it?"));
    const withImages = withLedgerImageContentBlocks(messages, {
      sections: [],
      transcriptIncluded: true,
      transcriptCompacted: false,
      originalTranscriptTokenEstimate: 0,
      compactedTranscriptEntryCount: 0,
      rawPreservedUserTranscriptEntryCount: 0,
      estimatedTokens: 0,
      imageAttachments: [
        {
          label: "Image A: user-uploaded screenshot from turn 42 (audience: Alice)",
          attachment_id: attachmentId,
          citation_type: "original_image",
        },
      ],
    });

    expect(withImages.at(-1)).toEqual({
      role: "user",
      content: [
        {
          type: "text",
          text: "Image A: user-uploaded screenshot from turn 42 (audience: Alice)",
        },
        {
          type: "image_ref",
          attachment_id: attachmentId,
        },
      ],
    });
  });

  it("applies a combined finalizer image cap and prefers current user images", () => {
    const currentAttachmentId = "att_aaaaaaaaaaaaaaaa" as AttachmentId;
    const retrievedAttachmentId = "att_bbbbbbbbbbbbbbbb";
    const messages = withCurrentUserContentBlocks(
      toContentBlockMessages(buildDialogueMessages([], "Look at this")),
      [
        { type: "text", text: "Look at this" },
        { type: "image_ref", attachment_id: currentAttachmentId },
      ],
    );
    const withImages = withLedgerImageContentBlocks(
      messages,
      {
        sections: [],
        transcriptIncluded: true,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
        imageAttachments: [
          {
            label: "Image A: retrieved",
            attachment_id: retrievedAttachmentId,
            citation_type: "original_image",
          },
        ],
      },
      { maxImagesPerLlmCall: 1 },
    );

    expect(withImages).toEqual(messages);
  });

  it("downgrades ledger image citations when current images exhaust the call cap", () => {
    const firstCurrentAttachmentId = "att_aaaaaaaaaaaaaaaa" as AttachmentId;
    const secondCurrentAttachmentId = "att_bbbbbbbbbbbbbbbb" as AttachmentId;
    const retrievedAttachmentId = "att_cccccccccccccccc";
    const messages = withCurrentUserContentBlocks(
      toContentBlockMessages(buildDialogueMessages([], "Compare these to the old diagram")),
      [
        { type: "text", text: "Compare these to the old diagram" },
        { type: "image_ref", attachment_id: firstCurrentAttachmentId },
        { type: "image_ref", attachment_id: secondCurrentAttachmentId },
      ],
    );
    const ledger = {
      sections: [
        {
          id: "retrieved_memory_evidence" as const,
          label: "11. Retrieved Memory Evidence",
          entries: [
            {
              id: "retrieved_evidence:image",
              source_type: "image_attachment" as const,
              session_scope: "current_session" as const,
              actor: "memory" as const,
              trust_rank: 6,
              text: "Caption: old diagram",
              value: "image_perception",
              state: "score=0.91",
              state_metadata: { attachment_id: retrievedAttachmentId },
              citation_type: "original_image" as const,
            },
          ],
        },
      ],
      transcriptIncluded: true,
      transcriptCompacted: false,
      originalTranscriptTokenEstimate: 0,
      compactedTranscriptEntryCount: 0,
      rawPreservedUserTranscriptEntryCount: 0,
      estimatedTokens: 0,
      imageAttachments: [
        {
          label: "Image A: old diagram",
          attachment_id: retrievedAttachmentId,
          citation_type: "original_image" as const,
        },
      ],
    };

    const budgetedLedger = withFinalizerImageBudget(messages, ledger, {
      maxImagesPerLlmCall: 2,
    });
    const withImages = withLedgerImageContentBlocks(messages, budgetedLedger, {
      maxImagesPerLlmCall: 2,
    });

    expect(budgetedLedger?.imageAttachments).toBeUndefined();
    expect(budgetedLedger?.sections[0]?.entries[0]).toMatchObject({
      citation_type: "generated_perception_text",
      state: "score=0.91 image_unavailable=call_budget",
    });
    expect(renderEvidenceLedger(budgetedLedger!)).not.toContain(
      "Retrieved images are reattached below",
    );
    expect(withImages).toEqual(messages);
  });
});
