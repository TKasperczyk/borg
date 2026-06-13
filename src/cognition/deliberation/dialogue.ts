// Converts recency-window dialogue and the current turn into LLM message shapes.
import type { BorgUserContentBlock } from "../../attachments/index.js";
import type { AttachmentId } from "../../util/ids.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "../evidence-ledger/types.js";
import type { LLMContentBlock, LLMContentBlockMessage, LLMMessage } from "../../llm/index.js";
import type { RecencyMessage } from "../recency/index.js";

// Anthropic rejects requests where any text content block has empty
// content with 'messages: text content blocks must be non-empty'.
// Defense in depth: filter empty stream entries from recency at the
// source, AND substitute a placeholder when serializing to content
// blocks. Empty content can leak in via empty agent_msg entries
// written from a prior failed turn (Borg's pipeline can return an
// empty response in edge cases) -- those poison subsequent turns'
// recency window without this guard.
const EMPTY_CONTENT_PLACEHOLDER = "(no content)";
const ADJACENT_MESSAGE_SEPARATOR = "\n\n";

function appendMessage(messages: LLMMessage[], next: LLMMessage): void {
  const previous = messages[messages.length - 1];

  if (previous?.role === next.role) {
    messages[messages.length - 1] = {
      role: previous.role,
      content: [previous.content, next.content].join(ADJACENT_MESSAGE_SEPARATOR),
    };
    return;
  }

  messages.push(next);
}

/**
 * Assemble the Anthropic `messages` array from recent dialogue + the current
 * user message. Recency can contain adjacent user-role messages when Borg
 * observed instead of speaking, so this step merges adjacent same-role entries
 * rather than inventing assistant output.
 */
export function buildDialogueMessages(
  recency: readonly RecencyMessage[] | undefined,
  currentUserMessage: string,
): LLMMessage[] {
  const messages: LLMMessage[] = [];

  if (recency !== undefined) {
    for (const item of recency) {
      if (item.content.trim().length === 0) {
        continue;
      }
      appendMessage(messages, { role: item.role, content: item.content });
    }
  }

  const trimmed = currentUserMessage.trim();
  const currentContent = trimmed.length === 0 ? EMPTY_CONTENT_PLACEHOLDER : currentUserMessage;
  appendMessage(messages, {
    role: "user",
    content: currentContent,
  });
  return messages;
}

export function toContentBlockMessages(messages: readonly LLMMessage[]): LLMContentBlockMessage[] {
  return messages.map((message) => ({
    role: message.role,
    content: [
      {
        type: "text",
        text: message.content.trim().length === 0 ? EMPTY_CONTENT_PLACEHOLDER : message.content,
      },
    ],
  }));
}

export function withCurrentUserContentBlocks(
  messages: readonly LLMContentBlockMessage[],
  currentUserContent: readonly BorgUserContentBlock[] | undefined,
): LLMContentBlockMessage[] {
  if (currentUserContent === undefined || currentUserContent.length <= 1 || messages.length === 0) {
    return [...messages];
  }

  const next = messages.map((message) => ({
    role: message.role,
    content: [...message.content],
  }));
  const last = next[next.length - 1];

  if (last === undefined || last.role !== "user") {
    return next;
  }

  const mappedContent: LLMContentBlock[] = currentUserContent.map((block) =>
    block.type === "text"
      ? {
          type: "text",
          text: block.text.trim().length === 0 ? EMPTY_CONTENT_PLACEHOLDER : block.text,
        }
      : {
          type: "image_ref",
          attachment_id: block.attachment_id,
        },
  );
  const imageContent = mappedContent.filter((block) => block.type === "image_ref");
  const textContent = mappedContent.filter((block) => block.type !== "image_ref");
  const content = [...imageContent, ...textContent];

  const currentText = textContent[0]?.type === "text" ? textContent[0].text : undefined;
  const existingText =
    last.content.length === 1 && last.content[0]?.type === "text"
      ? last.content[0].text
      : undefined;

  if (
    currentText !== undefined &&
    existingText !== undefined &&
    existingText !== currentText &&
    existingText.endsWith(`${ADJACENT_MESSAGE_SEPARATOR}${currentText}`)
  ) {
    last.content = [
      {
        type: "text",
        text: existingText.slice(0, -`${ADJACENT_MESSAGE_SEPARATOR}${currentText}`.length),
      },
      ...content,
    ];
    return next;
  }

  last.content = content;
  return next;
}

/**
 * Append a final user-role text message after the transcript. The system prompt always renders
 * before every message, so important per-turn directives placed there sit tens of thousands of
 * tokens upstream of the generation point, behind the whole transcript. A trailing message is
 * the only position truly adjacent to generation. This reuses the same shape
 * withLedgerImageContentBlocks already uses to append trailing image messages, so the
 * conversation shape is not novel. The caller frames the text as harness scaffolding (tagged,
 * self-disclaiming) rather than conversation, so it does not read as the current speaker's turn.
 */
export function withTrailingUserMessage(
  messages: readonly LLMContentBlockMessage[],
  text: string,
): LLMContentBlockMessage[] {
  const trimmed = text.trim();
  if (trimmed.length === 0) {
    return [...messages];
  }
  return [...messages, { role: "user", content: [{ type: "text", text: trimmed }] }];
}

function imageRefCount(messages: readonly LLMContentBlockMessage[]): number {
  return messages.reduce(
    (count, message) =>
      count + message.content.filter((block) => block.type === "image_ref").length,
    0,
  );
}

function appendedImageBudgetState(state: string | undefined): string {
  const marker = "image_unavailable=call_budget";

  if (state === undefined || state.length === 0) {
    return marker;
  }

  return state.includes(marker) ? state : `${state} ${marker}`;
}

function downgradeOmittedImageEntry(
  entry: EvidenceLedgerEntry,
  omittedAttachmentIds: ReadonlySet<string>,
  allLedgerImagesOmitted: boolean,
): EvidenceLedgerEntry {
  if (entry.citation_type !== "original_image") {
    return entry;
  }

  const attachmentId = entry.state_metadata?.attachment_id;
  const omitted =
    (typeof attachmentId === "string" && omittedAttachmentIds.has(attachmentId)) ||
    (attachmentId === undefined &&
      allLedgerImagesOmitted &&
      entry.source_type === "image_attachment");

  if (!omitted) {
    return entry;
  }

  return {
    ...entry,
    citation_type: "generated_perception_text",
    state: appendedImageBudgetState(entry.state),
  };
}

export function withFinalizerImageBudget(
  messages: readonly LLMContentBlockMessage[],
  ledger: EvidenceLedger | null | undefined,
  options: { maxImagesPerLlmCall?: number } = {},
): EvidenceLedger | null | undefined {
  if (ledger?.imageAttachments === undefined || ledger.imageAttachments.length === 0) {
    return ledger;
  }

  const remaining =
    options.maxImagesPerLlmCall === undefined
      ? ledger.imageAttachments.length
      : Math.max(0, options.maxImagesPerLlmCall - imageRefCount(messages));

  if (remaining >= ledger.imageAttachments.length) {
    return ledger;
  }

  const imageAttachments = ledger.imageAttachments.slice(0, remaining);
  const omittedAttachmentIds = new Set(
    ledger.imageAttachments.slice(remaining).map((image) => image.attachment_id),
  );
  const allLedgerImagesOmitted = imageAttachments.length === 0;
  const next: EvidenceLedger = {
    ...ledger,
    sections: ledger.sections.map((section) => ({
      ...section,
      entries: section.entries.map((entry) =>
        downgradeOmittedImageEntry(entry, omittedAttachmentIds, allLedgerImagesOmitted),
      ),
    })),
    imageAttachments,
  };

  if (next.imageAttachments?.length === 0) {
    delete next.imageAttachments;
  }

  return next;
}

export function withLedgerImageContentBlocks(
  messages: readonly LLMContentBlockMessage[],
  ledger: EvidenceLedger | null | undefined,
  options: { maxImagesPerLlmCall?: number } = {},
): LLMContentBlockMessage[] {
  const budgetedLedger = withFinalizerImageBudget(messages, ledger, options);

  if (
    budgetedLedger?.imageAttachments === undefined ||
    budgetedLedger.imageAttachments.length === 0
  ) {
    return [...messages];
  }

  return [
    ...messages,
    {
      role: "user",
      content: budgetedLedger.imageAttachments.flatMap((image): LLMContentBlock[] => [
        {
          type: "text",
          text: image.label,
        },
        {
          type: "image_ref",
          attachment_id: image.attachment_id as AttachmentId,
        },
      ]),
    },
  ];
}
