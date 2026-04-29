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

/**
 * Assemble the Anthropic `messages` array from recent dialogue + the current
 * user message. The recency window is already shaped to satisfy Anthropic's
 * ordering constraints (starts with user, ends with assistant), so we can
 * concatenate and append the current user message safely.
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
      messages.push({ role: item.role, content: item.content });
    }
  }

  const trimmed = currentUserMessage.trim();
  messages.push({
    role: "user",
    content: trimmed.length === 0 ? EMPTY_CONTENT_PLACEHOLDER : currentUserMessage,
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
