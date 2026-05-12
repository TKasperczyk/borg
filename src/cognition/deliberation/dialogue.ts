// Converts recency-window dialogue and the current turn into LLM message shapes.
import type { LLMContentBlockMessage, LLMMessage } from "../../llm/index.js";
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
