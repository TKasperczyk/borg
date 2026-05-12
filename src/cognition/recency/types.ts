import type { StreamEntry } from "../../stream/index.js";
import type { StreamEntryId } from "../../util/ids.js";

/**
 * A single conversational message resolved from the stream, pre-formatted for
 * use as dialogue context. Role is derived from the stream entry kind. Observed
 * turns are rendered as user-role system markers so dialogue assembly can merge
 * them with adjacent participant messages without inventing assistant output.
 */
export type RecencyMessage = {
  role: "user" | "assistant";
  content: string;
  stream_entry_id: StreamEntryId;
  ts: number;
  kind?: StreamEntry["kind"];
};

/**
 * Recent conversation window compiled from the session stream. It starts with a
 * user role when non-empty, but may include adjacent user-role messages when
 * Borg observed intervening turns. Deliberation dialogue assembly normalizes
 * these runs before sending them to the LLM.
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
