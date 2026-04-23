// Converts recency-window dialogue and the current turn into LLM message shapes.
import type { LLMContentBlockMessage, LLMMessage } from "../../llm/index.js";
import type { RecencyMessage } from "../recency/index.js";

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
      messages.push({ role: item.role, content: item.content });
    }
  }

  messages.push({ role: "user", content: currentUserMessage });
  return messages;
}

export function toContentBlockMessages(messages: readonly LLMMessage[]): LLMContentBlockMessage[] {
  return messages.map((message) => ({
    role: message.role,
    content: [
      {
        type: "text",
        text: message.content,
      },
    ],
  }));
}
