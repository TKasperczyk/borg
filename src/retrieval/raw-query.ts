import { toWellFormedUtf16 } from "../util/utf16-boundary.js";

// A 6 KiB raw-query vector leaves room below the embedding transport cap while
// keeping the budget language-independent; expansion retains the full focus.
export const DEFAULT_RAW_QUERY_MAX_BYTES = 6 * 1024;

export type BoundedRawRecallQuery = {
  query: string;
  bounded: boolean;
  mode: "unchanged" | "query_tail" | "messages" | "message_tail";
  originalBytes: number;
  queryBytes: number;
  selectedMessageCount: number;
};

function utf8Tail(value: string, maxBytes: number): string {
  const bytes = Buffer.from(toWellFormedUtf16(value), "utf8");
  let start = Math.max(0, bytes.length - maxBytes);

  // A UTF-8 continuation byte cannot begin the retained suffix.
  while (start < bytes.length && (bytes[start]! & 0xc0) === 0x80) {
    start += 1;
  }

  return bytes.subarray(start).toString("utf8");
}

/** Message boundaries and oldest-first order come from the caller, never from parsing text. */
export function boundRawRecallQuery(
  query: string,
  maxBytes = DEFAULT_RAW_QUERY_MAX_BYTES,
  messages?: readonly string[],
): BoundedRawRecallQuery {
  const originalBytes = Buffer.byteLength(query, "utf8");
  if (originalBytes <= maxBytes) {
    return {
      query,
      bounded: false,
      mode: "unchanged",
      originalBytes,
      queryBytes: originalBytes,
      selectedMessageCount: 0,
    };
  }

  const sourceMessages = messages ?? [];
  const newest = sourceMessages.at(-1);
  if (newest === undefined) {
    const tail = utf8Tail(query, maxBytes);
    return {
      query: tail,
      bounded: true,
      mode: "query_tail",
      originalBytes,
      queryBytes: Buffer.byteLength(tail, "utf8"),
      selectedMessageCount: 0,
    };
  }

  if (Buffer.byteLength(newest, "utf8") > maxBytes) {
    const tail = utf8Tail(newest, maxBytes);
    return {
      query: tail,
      bounded: true,
      mode: "message_tail",
      originalBytes,
      queryBytes: Buffer.byteLength(tail, "utf8"),
      selectedMessageCount: 1,
    };
  }

  const selected: string[] = [];
  const separator = "\n\n";
  const separatorBytes = Buffer.byteLength(separator, "utf8");
  let queryBytes = 0;
  for (let index = sourceMessages.length - 1; index >= 0; index -= 1) {
    const message = toWellFormedUtf16(sourceMessages[index]!);
    const addedBytes =
      Buffer.byteLength(message, "utf8") + (selected.length === 0 ? 0 : separatorBytes);
    if (queryBytes + addedBytes > maxBytes) {
      break;
    }
    selected.push(message);
    queryBytes += addedBytes;
  }

  // Selection runs newest-first; restore conversation order so the newest
  // message ends the query (Qwen3 embeddings pool on the last token).
  return {
    query: selected.slice().reverse().join(separator),
    bounded: true,
    mode: "messages",
    originalBytes,
    queryBytes,
    selectedMessageCount: selected.length,
  };
}
