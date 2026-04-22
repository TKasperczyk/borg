import type { StreamEntryId } from "../../util/ids.js";

/**
 * A single conversational message resolved from the stream, pre-formatted for
 * use as an Anthropic `messages` array entry. Role is derived from the stream
 * entry kind (`user_msg` -> "user", `agent_msg` -> "assistant").
 */
export type RecencyMessage = {
  role: "user" | "assistant";
  content: string;
  stream_entry_id: StreamEntryId;
  ts: number;
};

/**
 * Recent conversation window compiled from the session stream. `messages` is
 * safe to concatenate with `{role:"user", content: currentUserMessage}` and
 * hand to the LLM: it starts with a user role, alternates user/assistant,
 * and does NOT end with a user role (so it won't collide with the current
 * user message).
 */
export type RecencyWindow = {
  messages: RecencyMessage[];
  /** Timestamp of the newest entry included, or null if window is empty. */
  latest_ts: number | null;
  /**
   * Total character count across `messages`. Useful for downstream prompts
   * that need to know how much conversational context they already carry.
   */
  total_chars: number;
};
