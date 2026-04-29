import { StreamReader, type StreamEntry } from "../../stream/index.js";

import type { RecencyMessage, RecencyWindow } from "./types.js";

export type TurnContextCompilerOptions = {
  /**
   * Maximum number of user/assistant messages to include (raw message count,
   * NOT pairs). Defaults to 16 -- roughly the last eight user/agent turns.
   */
  maxMessages?: number;
  /**
   * Soft character cap across included messages. Newer messages are kept
   * first; older messages are dropped once the cap is reached. Defaults to
   * 24_000 characters (~6k tokens), well under any model's context window
   * but large enough that a typical multi-turn session survives intact.
   */
  maxChars?: number;
  /**
   * Include self-addressed autonomous turns (`audience === "self"`) in the
   * compiled recency window. Defaults to false so normal user turns do not
   * see scheduler-generated self-conversation in their dialogue context.
   */
  includeSelfTurns?: boolean;
};

const DEFAULT_MAX_MESSAGES = 16;
const DEFAULT_MAX_CHARS = 24_000;

/**
 * Number of trailing stream entries to read in order to find `maxMessages`
 * conversational entries. Inflated to skip past interleaved `thought`,
 * `internal_event`, and `tool_call` entries that belong between user/agent
 * messages but aren't part of the recency window itself.
 */
const TAIL_MULTIPLIER = 6;

function entryContentToString(entry: StreamEntry): string {
  if (typeof entry.content === "string") {
    return entry.content;
  }

  try {
    return JSON.stringify(entry.content);
  } catch {
    return String(entry.content);
  }
}

function streamKindToRole(kind: StreamEntry["kind"]): RecencyMessage["role"] | null {
  if (kind === "user_msg") {
    return "user";
  }

  if (kind === "agent_msg") {
    return "assistant";
  }

  return null;
}

/**
 * Compile the recent conversation window for a session from its append-only
 * stream. Call this BEFORE writing the current user message to the stream
 * so the window contains only prior turns.
 *
 * Guarantees on the returned `messages`:
 *   - First message (if any) has role `user` -- Anthropic requires `messages`
 *     to start with a user role when it's provided.
 *   - Last message does NOT have role `user`. The caller appends a
 *     `{role:"user", content: currentUserMessage}` entry, so a trailing
 *     user in the window would produce two consecutive user messages.
 *   - Roles alternate.
 *
 * Non-conversational entries (`thought`, `internal_event`, `tool_call`,
 * `tool_result`, `perception`, `agent_suppressed`, `dream_report`) are
 * ignored here. The stream keeps them, but they are not part of the dialogue
 * layer the LLM sees.
 */
export class TurnContextCompiler {
  private readonly maxMessages: number;
  private readonly maxChars: number;
  private readonly includeSelfTurns: boolean;

  constructor(options: TurnContextCompilerOptions = {}) {
    this.maxMessages = options.maxMessages ?? DEFAULT_MAX_MESSAGES;
    this.maxChars = options.maxChars ?? DEFAULT_MAX_CHARS;
    this.includeSelfTurns = options.includeSelfTurns ?? false;
  }

  compile(
    streamReader: StreamReader,
    options: Pick<TurnContextCompilerOptions, "includeSelfTurns"> = {},
  ): RecencyWindow {
    const includeSelfTurns = options.includeSelfTurns ?? this.includeSelfTurns;
    const tailSize = Math.max(this.maxMessages * TAIL_MULTIPLIER, this.maxMessages);
    const recent = streamReader.tail(tailSize);

    // Keep only conversation entries; preserve stream order.
    const conversational: Array<{
      entry: StreamEntry;
      role: RecencyMessage["role"];
    }> = [];

    for (const entry of recent) {
      const role = streamKindToRole(entry.kind);

      if (role === null) {
        continue;
      }

      if (!includeSelfTurns && entry.audience === "self") {
        continue;
      }

      conversational.push({ entry, role });
    }

    // Walk backwards from the newest conversational entry, accumulating until
    // we hit either the message-count cap or the char cap.
    const reversed: RecencyMessage[] = [];
    let totalChars = 0;

    for (let index = conversational.length - 1; index >= 0; index -= 1) {
      const item = conversational[index];

      if (item === undefined) {
        continue;
      }

      const content = entryContentToString(item.entry);
      const contentLength = content.length;

      if (reversed.length > 0 && totalChars + contentLength > this.maxChars) {
        break;
      }

      reversed.push({
        role: item.role,
        content,
        stream_entry_id: item.entry.id,
        ts: item.entry.timestamp,
      });

      totalChars += contentLength;

      if (reversed.length >= this.maxMessages) {
        break;
      }
    }

    const messages = reversed.reverse();

    // Drop a leading `assistant` -- Anthropic requires `messages` to start
    // with `user` when non-empty. This can happen if our window starts
    // mid-pair because of the max-messages cap.
    while (messages.length > 0 && messages[0]?.role === "assistant") {
      messages.shift();
    }

    // Drop a trailing `user` -- the caller is about to append the current
    // user message, and two consecutive user entries are rejected.
    while (messages.length > 0 && messages[messages.length - 1]?.role === "user") {
      messages.pop();
    }

    const alternating: RecencyMessage[] = [];

    for (const message of messages) {
      const previous = alternating[alternating.length - 1];

      if (previous?.role === message.role) {
        alternating[alternating.length - 1] = message;
        continue;
      }

      alternating.push(message);
    }

    const latestTs =
      alternating.length === 0 ? null : (alternating[alternating.length - 1]?.ts ?? null);
    const normalizedChars = alternating.reduce((sum, message) => sum + message.content.length, 0);

    return {
      messages: alternating,
      latest_ts: latestTs,
      total_chars: normalizedChars,
    };
  }
}
