import { describe, expect, it } from "vitest";

import {
  DEFAULT_SESSION_ID,
  type SessionId,
  type StreamCursor,
  type StreamEntry,
  type StreamEntryIndexRecord,
  type StreamEntryKind,
  type StreamIterateOptions,
  type StreamReader,
} from "../../stream/index.js";
import { createSessionId, parseStreamEntryId, type StreamEntryId } from "../../util/ids.js";

import { ChatResponseBacklogPrefixBuilder, type BacklogPrefixCaps } from "./index.js";

function idAt(index: number): StreamEntryId {
  return parseStreamEntryId(`strm_${index.toString(36).padStart(16, "0")}`);
}

function literalId(value: string): StreamEntryId {
  return parseStreamEntryId(`strm_${value}`);
}

function makeEntry(input: {
  entryIndex: number;
  id?: StreamEntryId;
  timestamp?: number;
  kind?: StreamEntryKind;
  content?: unknown;
  sessionId?: SessionId;
  tokenEstimate?: number;
  turnId?: string;
}): StreamEntry {
  return {
    id: input.id ?? idAt(input.entryIndex),
    timestamp: input.timestamp ?? 1_000 + input.entryIndex,
    entry_index: input.entryIndex,
    kind: input.kind ?? "user_msg",
    content: input.content ?? `message-${input.entryIndex}`,
    ...(input.turnId === undefined ? {} : { turn_id: input.turnId }),
    ...(input.tokenEstimate === undefined ? {} : { token_estimate: input.tokenEstimate }),
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: input.sessionId ?? DEFAULT_SESSION_ID,
    compressed: false,
  };
}

function recordFor(
  entry: StreamEntry,
  overrides: Partial<StreamEntryIndexRecord> = {},
): StreamEntryIndexRecord {
  return {
    entry_id: entry.id,
    session_id: entry.session_id,
    byte_offset: (entry.entry_index ?? 0) * 100,
    entry_index: entry.entry_index ?? null,
    timestamp: entry.timestamp,
    kind: entry.kind,
    sender_entity_id: entry.sender_entity_id ?? null,
    turn_id: entry.turn_id ?? null,
    turn_status: entry.turn_status ?? "active",
    active: true,
    receipt_pending: false,
    source_message_key_source_type: null,
    source_message_key_source_external_id: null,
    source_message_key_external_message_id: null,
    response_to_kind: null,
    response_to_from_cursor_ts: null,
    response_to_from_cursor_entry_id: null,
    response_to_through_cursor_ts: null,
    response_to_through_cursor_entry_id: null,
    response_to_source_entry_ids: null,
    response_to_count: null,
    ...overrides,
  };
}

function cursorFor(entry: Pick<StreamEntry, "id" | "timestamp">): StreamCursor {
  return {
    ts: entry.timestamp,
    entryId: entry.id,
  };
}

function makeBuilder(input: {
  entries: readonly StreamEntry[];
  records?: readonly StreamEntryIndexRecord[];
  caps?: BacklogPrefixCaps;
}): ChatResponseBacklogPrefixBuilder {
  const records = input.records ?? input.entries.map((entry) => recordFor(entry));
  const recordsById = new Map(records.map((record) => [record.entry_id, record]));

  return new ChatResponseBacklogPrefixBuilder({
    entryIndex: {
      lookup: (entryId) => recordsById.get(entryId) ?? null,
      lookupSessionEntriesByKind: ({ sessionId, kind }) =>
        records
          .filter((record) => record.session_id === sessionId && record.kind === kind)
          .sort((left, right) => left.byte_offset - right.byte_offset),
    },
    createStreamReader: (sessionId) =>
      ({
        iterate: async function* (options: StreamIterateOptions = {}) {
          if (options.sinceCursor !== undefined) {
            throw new Error("backlog prefix tests forbid cursor-based stream iteration");
          }

          const kinds =
            options.kinds === undefined ? undefined : new Set<StreamEntryKind>(options.kinds);

          for (const entry of input.entries) {
            if (entry.session_id !== sessionId) {
              continue;
            }

            if (kinds !== undefined && !kinds.has(entry.kind)) {
              continue;
            }

            yield entry;
          }
        },
      }) satisfies Pick<StreamReader, "iterate">,
    ...(input.caps === undefined ? {} : { caps: input.caps }),
  });
}

describe("ChatResponseBacklogPrefixBuilder", () => {
  it("returns an empty result when the backlog is empty", async () => {
    const builder = makeBuilder({ entries: [] });

    await expect(
      builder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: null,
      }),
    ).resolves.toEqual({
      fromCursorExclusive: null,
      entryIds: [],
      throughCursorInclusive: null,
      includedCount: 0,
      remainingCount: 0,
      hasMore: false,
      estimatedTokens: 0,
      estimatedChars: 0,
    });
  });

  it("starts a null watermark at the first pending user message even when inactive", async () => {
    const inactive = makeEntry({ entryIndex: 0 });
    const firstActive = makeEntry({ entryIndex: 1 });
    const secondActive = makeEntry({ entryIndex: 2 });
    const builder = makeBuilder({
      entries: [inactive, firstActive, secondActive],
      records: [
        recordFor(inactive, { active: false }),
        recordFor(firstActive),
        recordFor(secondActive),
      ],
    });

    await expect(
      builder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: null,
        caps: { maxMessages: 1 },
      }),
    ).resolves.toMatchObject({
      entryIds: [inactive.id],
      throughCursorInclusive: cursorFor(inactive),
      includedCount: 1,
      remainingCount: 2,
      hasMore: true,
    });
  });

  it("filters strictly after a non-null watermark entry_index", async () => {
    const entries = [
      makeEntry({ entryIndex: 0 }),
      makeEntry({ entryIndex: 1 }),
      makeEntry({ entryIndex: 2 }),
    ];
    const builder = makeBuilder({ entries });

    await expect(
      builder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: cursorFor(entries[1]!),
      }),
    ).resolves.toMatchObject({
      fromCursorExclusive: cursorFor(entries[1]!),
      entryIds: [entries[2]!.id],
      throughCursorInclusive: cursorFor(entries[2]!),
      includedCount: 1,
      remainingCount: 0,
      hasMore: false,
    });
  });

  it("returns an oldest-first prefix for a backlog larger than maxMessages", async () => {
    const entries = Array.from({ length: 5 }, (_, entryIndex) => makeEntry({ entryIndex }));
    const builder = makeBuilder({ entries, caps: { maxMessages: 3 } });

    await expect(
      builder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: null,
      }),
    ).resolves.toMatchObject({
      entryIds: [entries[0]!.id, entries[1]!.id, entries[2]!.id],
      throughCursorInclusive: cursorFor(entries[2]!),
      includedCount: 3,
      remainingCount: 2,
      hasMore: true,
    });
  });

  it("second build after watermark advancement returns the remainder", async () => {
    const entries = Array.from({ length: 4 }, (_, entryIndex) => makeEntry({ entryIndex }));
    const builder = makeBuilder({ entries });
    const first = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      fromCursorExclusive: null,
      caps: { maxMessages: 2 },
    });

    expect(first.entryIds).toEqual([entries[0]!.id, entries[1]!.id]);

    const second = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      fromCursorExclusive: first.throughCursorInclusive,
      caps: { maxMessages: 16 },
    });

    expect(second).toMatchObject({
      fromCursorExclusive: cursorFor(entries[1]!),
      entryIds: [entries[2]!.id, entries[3]!.id],
      throughCursorInclusive: cursorFor(entries[3]!),
      includedCount: 2,
      remainingCount: 0,
      hasMore: false,
    });
  });

  it("stops before the next message when token and char caps would be exceeded", async () => {
    const first = makeEntry({
      entryIndex: 0,
      content: "12345",
      tokenEstimate: 4,
    });
    const second = makeEntry({
      entryIndex: 1,
      content: "123456",
      tokenEstimate: 3,
    });
    const builder = makeBuilder({ entries: [first, second] });

    const result = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      fromCursorExclusive: null,
      caps: {
        maxMessages: 16,
        maxTokens: 6,
        maxChars: 10,
      },
    });

    expect(result).toMatchObject({
      entryIds: [first.id],
      throughCursorInclusive: cursorFor(first),
      includedCount: 1,
      remainingCount: 1,
      hasMore: true,
      estimatedTokens: 4,
      estimatedChars: 5,
    });
  });

  it("includes an entry that lands exactly on the cap", async () => {
    const first = makeEntry({ entryIndex: 0, content: "a", tokenEstimate: 2 });
    const second = makeEntry({ entryIndex: 1, content: "bc", tokenEstimate: 3 });
    const builder = makeBuilder({ entries: [first, second] });

    const result = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      fromCursorExclusive: null,
      caps: {
        maxMessages: 2,
        maxTokens: 5,
        maxChars: 3,
      },
    });

    expect(result).toMatchObject({
      entryIds: [first.id, second.id],
      throughCursorInclusive: cursorFor(second),
      includedCount: 2,
      remainingCount: 0,
      hasMore: false,
      estimatedTokens: 5,
      estimatedChars: 3,
    });
  });

  it("includes the first pending message even when it alone exceeds the caps", async () => {
    const oversized = makeEntry({
      entryIndex: 0,
      content: "0123456789",
      tokenEstimate: 100,
    });
    const next = makeEntry({ entryIndex: 1, content: "next", tokenEstimate: 1 });
    const builder = makeBuilder({ entries: [oversized, next] });

    const result = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      fromCursorExclusive: null,
      caps: {
        maxMessages: 1,
        maxTokens: 1,
        maxChars: 1,
      },
    });

    expect(result).toMatchObject({
      entryIds: [oversized.id],
      throughCursorInclusive: cursorFor(oversized),
      includedCount: 1,
      remainingCount: 1,
      hasMore: true,
      estimatedTokens: 100,
      estimatedChars: 10,
    });
  });

  it("ignores interleaved non-user entries while keeping the pending user prefix contiguous", async () => {
    const first = makeEntry({ entryIndex: 0 });
    const agent = makeEntry({ entryIndex: 1, kind: "agent_msg" });
    const second = makeEntry({ entryIndex: 2 });
    const builder = makeBuilder({ entries: [first, agent, second] });

    await expect(
      builder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: null,
        caps: { maxMessages: 2 },
      }),
    ).resolves.toMatchObject({
      entryIds: [first.id, second.id],
      throughCursorInclusive: cursorFor(second),
      includedCount: 2,
      remainingCount: 0,
      hasMore: false,
    });
  });

  it("includes inactive user messages that still need a response", async () => {
    const first = makeEntry({ entryIndex: 0 });
    const inactive = makeEntry({ entryIndex: 1 });
    const second = makeEntry({ entryIndex: 2 });
    const builder = makeBuilder({
      entries: [first, inactive, second],
      records: [recordFor(first), recordFor(inactive, { active: false }), recordFor(second)],
    });

    await expect(
      builder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: null,
      }),
    ).resolves.toMatchObject({
      entryIds: [first.id, inactive.id, second.id],
      throughCursorInclusive: cursorFor(second),
      includedCount: 3,
      remainingCount: 0,
      hasMore: false,
    });
  });

  it("skips already-answered normal turn user messages with turn ids", async () => {
    const answeredNormal = makeEntry({ entryIndex: 0, turnId: "turn-normal" });
    const queued = makeEntry({ entryIndex: 1 });
    const builder = makeBuilder({ entries: [answeredNormal, queued] });

    await expect(
      builder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: null,
      }),
    ).resolves.toMatchObject({
      entryIds: [queued.id],
      throughCursorInclusive: cursorFor(queued),
      includedCount: 1,
      remainingCount: 0,
      hasMore: false,
    });
  });

  it("treats receipt-pending queued user messages as ordered prefix barriers", async () => {
    const waitingForReceipt = makeEntry({ entryIndex: 0 });
    const ready = makeEntry({ entryIndex: 1 });
    const blockedBuilder = makeBuilder({
      entries: [waitingForReceipt, ready],
      records: [recordFor(waitingForReceipt, { receipt_pending: true }), recordFor(ready)],
    });

    await expect(
      blockedBuilder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: null,
      }),
    ).resolves.toMatchObject({
      entryIds: [],
      throughCursorInclusive: null,
      includedCount: 0,
      remainingCount: 0,
      hasMore: false,
    });

    const readyBuilder = makeBuilder({
      entries: [waitingForReceipt, ready],
      records: [recordFor(waitingForReceipt, { receipt_pending: false }), recordFor(ready)],
    });

    await expect(
      readyBuilder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: null,
      }),
    ).resolves.toMatchObject({
      entryIds: [waitingForReceipt.id, ready.id],
      throughCursorInclusive: cursorFor(ready),
      includedCount: 2,
      remainingCount: 0,
      hasMore: false,
    });
  });

  it("orders same-timestamp and lexically misleading ids by entry_index", async () => {
    const firstByIndex = makeEntry({
      entryIndex: 1,
      id: literalId("zzzzzzzzzzzzzzzz"),
      timestamp: 5_000,
    });
    const secondByIndex = makeEntry({
      entryIndex: 2,
      id: literalId("aaaaaaaaaaaaaaaa"),
      timestamp: 5_000,
    });
    const builder = makeBuilder({
      entries: [secondByIndex, firstByIndex],
      records: [
        recordFor(secondByIndex, { byte_offset: 0 }),
        recordFor(firstByIndex, { byte_offset: 100 }),
      ],
    });

    await expect(
      builder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: null,
      }),
    ).resolves.toMatchObject({
      entryIds: [firstByIndex.id, secondByIndex.id],
      throughCursorInclusive: cursorFor(secondByIndex),
      includedCount: 2,
      remainingCount: 0,
      hasMore: false,
    });
  });

  it("throws when a non-null watermark is missing from the index", async () => {
    const pending = makeEntry({ entryIndex: 1 });
    const missingWatermark = makeEntry({ entryIndex: 0 });
    const builder = makeBuilder({ entries: [pending], records: [recordFor(pending)] });

    await expect(
      builder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: cursorFor(missingWatermark),
      }),
    ).rejects.toMatchObject({
      code: "CHAT_RESPONSE_BACKLOG_WATERMARK_NOT_INDEXED",
    });
  });

  it("throws when a non-null watermark has no durable order", async () => {
    const watermark = makeEntry({ entryIndex: 0 });
    const pending = makeEntry({ entryIndex: 1 });
    const builder = makeBuilder({
      entries: [watermark, pending],
      records: [recordFor(watermark, { entry_index: null }), recordFor(pending)],
    });

    await expect(
      builder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: cursorFor(watermark),
      }),
    ).rejects.toMatchObject({
      code: "CHAT_RESPONSE_BACKLOG_ENTRY_ORDER_MISSING",
    });
  });

  it("throws when a non-null watermark points at a non-user entry", async () => {
    const watermark = makeEntry({ entryIndex: 0, kind: "agent_msg" });
    const pending = makeEntry({ entryIndex: 1 });
    const builder = makeBuilder({
      entries: [watermark, pending],
      records: [recordFor(watermark), recordFor(pending)],
    });

    await expect(
      builder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: cursorFor(watermark),
      }),
    ).rejects.toMatchObject({
      code: "CHAT_RESPONSE_BACKLOG_WATERMARK_KIND_INVALID",
    });
  });

  it("throws when a non-null watermark identity mismatches the index", async () => {
    const otherSession = createSessionId();
    const watermark = makeEntry({ entryIndex: 0 });
    const pending = makeEntry({ entryIndex: 1 });
    const sessionMismatchBuilder = makeBuilder({
      entries: [watermark, pending],
      records: [recordFor(watermark, { session_id: otherSession }), recordFor(pending)],
    });
    const timestampMismatchBuilder = makeBuilder({
      entries: [watermark, pending],
      records: [recordFor(watermark, { timestamp: watermark.timestamp + 1 }), recordFor(pending)],
    });

    await expect(
      sessionMismatchBuilder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: cursorFor(watermark),
      }),
    ).rejects.toMatchObject({
      code: "CHAT_RESPONSE_BACKLOG_WATERMARK_SESSION_MISMATCH",
    });
    await expect(
      timestampMismatchBuilder.build({
        sessionId: DEFAULT_SESSION_ID,
        fromCursorExclusive: cursorFor(watermark),
      }),
    ).rejects.toMatchObject({
      code: "CHAT_RESPONSE_BACKLOG_WATERMARK_TS_MISMATCH",
    });
  });
});
