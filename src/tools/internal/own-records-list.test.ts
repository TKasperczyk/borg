import { describe, expect, it, vi } from "vitest";

import type { TrainOfThoughtJournalEntry } from "../../memory/train-of-thought/index.js";
import type { StreamEntry, StreamEntryIndexRecord } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import type { EntityId, SessionId, StreamEntryId } from "../../util/ids.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import {
  createOwnRecordsListTool,
  type OwnRecordKind,
  type OwnRecordsListRange,
} from "./own-records-list.js";

const CURRENT_SESSION_ID = "sess_aaaaaaaaaaaaaaaa" as SessionId;
const OTHER_SESSION_ID = "sess_bbbbbbbbbbbbbbbb" as SessionId;
const SELF_ENTITY_ID = "ent_aaaaaaaaaaaaaaaa" as EntityId;
const THOUGHT_A = "strm_aaaaaaaaaaaaaaaa" as StreamEntryId;
const THOUGHT_B = "strm_bbbbbbbbbbbbbbbb" as StreamEntryId;
const THOUGHT_C = "strm_cccccccccccccccc" as StreamEntryId;

function iso(timestamp: number): string {
  return new Date(timestamp).toISOString();
}

function thoughtRecord(input: {
  id: StreamEntryId;
  timestamp: number;
  sessionId?: SessionId;
  turnId?: string | null;
}): StreamEntryIndexRecord {
  return {
    entry_id: input.id,
    session_id: input.sessionId ?? DEFAULT_SESSION_ID,
    byte_offset: 0,
    entry_index: 0,
    timestamp: input.timestamp,
    kind: "thought",
    sender_entity_id: null,
    turn_id: input.turnId ?? null,
    turn_status: "active",
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
  };
}

function thoughtEntry(record: StreamEntryIndexRecord, content: unknown): StreamEntry {
  return {
    id: record.entry_id as StreamEntryId,
    timestamp: record.timestamp,
    entry_index: record.entry_index ?? undefined,
    kind: "thought",
    content,
    ...(record.turn_id === null ? {} : { turn_id: record.turn_id }),
    turn_status: "active",
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: record.session_id,
    compressed: false,
  };
}

function journalRecord(input: {
  id: number;
  createdAt: number;
  updatedAt?: number;
  text: string;
  sourceTurnId?: string | null;
  markerStreamEntryId?: StreamEntryId | null;
}): TrainOfThoughtJournalEntry {
  return {
    id: input.id,
    self_entity_id: SELF_ENTITY_ID,
    text: input.text,
    disclosure_class: "self_private",
    created_at: input.createdAt,
    updated_at: input.updatedAt ?? input.createdAt,
    source_turn_id: input.sourceTurnId ?? null,
    marker_stream_entry_id: input.markerStreamEntryId ?? null,
  };
}

function thoughtRangeLister(records: readonly StreamEntryIndexRecord[]) {
  return (
    input: OwnRecordsListRange & {
      cursor?: { timestamp: number; entryId: StreamEntryId };
    },
  ): StreamEntryIndexRecord[] =>
    records
      .filter(
        (record) =>
          record.timestamp >= input.sinceMs &&
          record.timestamp <= input.untilMs &&
          (input.sessionId === undefined || record.session_id === input.sessionId) &&
          (input.cursor === undefined ||
            record.timestamp < input.cursor.timestamp ||
            (record.timestamp === input.cursor.timestamp &&
              record.entry_id < input.cursor.entryId)),
      )
      .sort((left, right) =>
        left.timestamp === right.timestamp
          ? right.entry_id.localeCompare(left.entry_id)
          : right.timestamp - left.timestamp,
      )
      .slice(0, input.limit);
}

function journalRangeLister(records: readonly TrainOfThoughtJournalEntry[]) {
  return (
    input: OwnRecordsListRange & {
      cursor?: { createdAt: number; id: number };
    },
  ): TrainOfThoughtJournalEntry[] =>
    records
      .filter(
        (record) =>
          record.created_at >= input.sinceMs &&
          record.created_at <= input.untilMs &&
          (input.cursor === undefined ||
            record.created_at < input.cursor.createdAt ||
            (record.created_at === input.cursor.createdAt && record.id < input.cursor.id)),
      )
      .sort((left, right) =>
        left.created_at === right.created_at
          ? right.id - left.id
          : right.created_at - left.created_at,
      )
      .slice(0, input.limit);
}

function invocationContext(sessionId = CURRENT_SESSION_ID) {
  return {
    sessionId,
    origin: "deliberator" as const,
  };
}

describe("tool.ownRecords.list", () => {
  it("defaults recall to global and applies only an explicit model-chosen session filter", async () => {
    const current = thoughtRecord({
      id: THOUGHT_A,
      timestamp: 2_000,
      sessionId: CURRENT_SESSION_ID,
    });
    const crossSession = thoughtRecord({
      id: THOUGHT_B,
      timestamp: 1_000,
      sessionId: OTHER_SESSION_ID,
    });
    const entries = new Map([
      [current.entry_id, thoughtEntry(current, "current session")],
      [crossSession.entry_id, thoughtEntry(crossSession, "other session")],
    ]);
    const listThoughtRecords = vi.fn(thoughtRangeLister([current, crossSession]));
    const tool = createOwnRecordsListTool({
      listThoughtRecords,
      readThoughtRecord: (record) => entries.get(record.entry_id) ?? null,
      listJournalRecords: () => [],
      clock: new ManualClock(3_000),
    });

    const global = await tool.invoke(
      {
        since: iso(1_000),
        until: iso(2_000),
        kinds: ["thought"],
      },
      invocationContext(CURRENT_SESSION_ID),
    );

    expect(global.records.map((record) => record.session_id)).toEqual([
      CURRENT_SESSION_ID,
      OTHER_SESSION_ID,
    ]);
    expect(listThoughtRecords.mock.calls[0]?.[0]).not.toHaveProperty("sessionId");

    const filtered = await tool.invoke(
      {
        since: iso(1_000),
        until: iso(2_000),
        kinds: ["thought"],
        session_id: OTHER_SESSION_ID,
      },
      invocationContext(CURRENT_SESSION_ID),
    );

    expect(filtered.records.map((record) => record.session_id)).toEqual([OTHER_SESSION_ID]);
    expect(listThoughtRecords.mock.calls[1]?.[0]).toMatchObject({
      sessionId: OTHER_SESSION_ID,
    });
    expect(
      tool.inputSchema.safeParse({
        since: iso(1_000),
        until: iso(2_000),
        query: "not part of this structural tool",
      }).success,
    ).toBe(false);
    expect(
      tool.inputSchema.safeParse({
        since: "not-an-iso-time",
        until: iso(2_000),
      }).success,
    ).toBe(false);
    expect(
      tool.inputSchema.safeParse({
        since: iso(1_000),
        until: iso(2_000),
        limit: 51,
      }).success,
    ).toBe(false);
  });

  it("merges sources and paginates equal timestamps by source then durable id", async () => {
    const newerJournal = journalRecord({ id: 3, createdAt: 4_000, text: "newest journal" });
    const equalThoughtA = thoughtRecord({ id: THOUGHT_A, timestamp: 3_000 });
    const equalThoughtC = thoughtRecord({ id: THOUGHT_C, timestamp: 3_000 });
    const equalJournal = journalRecord({ id: 2, createdAt: 3_000, text: "equal journal" });
    const olderThought = thoughtRecord({ id: THOUGHT_B, timestamp: 2_000 });
    const thoughts = [equalThoughtA, equalThoughtC, olderThought];
    const entries = new Map(
      thoughts.map((record) => [record.entry_id, thoughtEntry(record, record.entry_id)]),
    );
    const tool = createOwnRecordsListTool({
      listThoughtRecords: thoughtRangeLister(thoughts),
      readThoughtRecord: (record) => entries.get(record.entry_id) ?? null,
      listJournalRecords: journalRangeLister([newerJournal, equalJournal]),
      clock: new ManualClock(5_000),
    });
    const handles: string[] = [];
    let cursor: string | undefined;

    do {
      const page = await tool.invoke(
        {
          since: iso(2_000),
          until: iso(4_000),
          limit: 2,
          ...(cursor === undefined ? {} : { cursor }),
        },
        invocationContext(),
      );
      handles.push(...page.records.map((record) => record.handle));
      cursor = page.next_cursor ?? undefined;

      if (!page.has_more) {
        break;
      }
    } while (cursor !== undefined);

    expect(handles).toEqual([
      "journal:3",
      `thought:${THOUGHT_C}`,
      `thought:${THOUGHT_A}`,
      "journal:2",
      `thought:${THOUGHT_B}`,
    ]);
    expect(new Set(handles).size).toBe(handles.length);
  });

  it("distinguishes a page the limit ended from one the token budget ended, which has_more cannot", async () => {
    const newest = journalRecord({ id: 3, createdAt: 3_000, text: "newest" });
    const oldest = journalRecord({ id: 1, createdAt: 1_000, text: "oldest" });
    const listAll = (middle: TrainOfThoughtJournalEntry) =>
      createOwnRecordsListTool({
        listThoughtRecords: () => [],
        readThoughtRecord: () => null,
        listJournalRecords: journalRangeLister([newest, middle, oldest]),
        clock: new ManualClock(4_000),
      });
    const range = {
      since: iso(1_000),
      until: iso(3_000),
      kinds: ["journal"] as OwnRecordKind[],
    };
    const limitEnded = await listAll(
      journalRecord({ id: 2, createdAt: 2_000, text: "middle" }),
    ).invoke({ ...range, limit: 2 }, invocationContext());
    const budgetEnded = await listAll(
      journalRecord({ id: 2, createdAt: 2_000, text: "序".repeat(150_000) }),
    ).invoke({ ...range, limit: 2 }, invocationContext());
    const exhausted = await listAll(
      journalRecord({ id: 2, createdAt: 2_000, text: "middle" }),
    ).invoke({ ...range, limit: 3 }, invocationContext());

    // Identical has_more, identical requested limit, different cause: the short
    // page is short because its records are long, not because the range is.
    expect(limitEnded.has_more).toBe(true);
    expect(budgetEnded.has_more).toBe(true);
    expect(limitEnded.records).toHaveLength(2);
    expect(budgetEnded.records).toHaveLength(1);
    expect(limitEnded.page_end_reason).toBe("limit_reached");
    expect(budgetEnded.page_end_reason).toBe("context_budget");
    expect(exhausted.has_more).toBe(false);
    expect(exhausted.page_end_reason).toBe("range_exhausted");

    for (const page of [limitEnded, budgetEnded, exhausted]) {
      expect(page.page_end_reason === "range_exhausted").toBe(!page.has_more);
    }
  });

  it("returns exact multilingual content, origin times, nullable anchors, and row-local labels", async () => {
    const multilingual = "我在火车上注意到了它。 لاحظتُ ذلك أيضًا. Zażółć gęślą jaźń. 🧠";
    const exactThought = thoughtRecord({
      id: THOUGHT_B,
      timestamp: 2_000,
      sessionId: OTHER_SESSION_ID,
      turnId: "turn-exact-thought",
    });
    const unavailableThought = thoughtRecord({
      id: THOUGHT_A,
      timestamp: 1_500,
      sessionId: CURRENT_SESSION_ID,
    });
    const journal = journalRecord({
      id: 1,
      createdAt: 1_000,
      updatedAt: 99_000,
      text: "日記の内容 — без изменений",
      sourceTurnId: null,
      markerStreamEntryId: null,
    });
    const tool = createOwnRecordsListTool({
      listThoughtRecords: thoughtRangeLister([exactThought, unavailableThought]),
      readThoughtRecord: (record) =>
        record.entry_id === exactThought.entry_id ? thoughtEntry(exactThought, multilingual) : null,
      listJournalRecords: journalRangeLister([journal]),
      clock: new ManualClock(3_000),
    });

    const output = await tool.invoke({ since: iso(1_000), until: iso(2_000) }, invocationContext());
    const exactRow = output.records[0]!;
    const unavailableRow = output.records[1]!;
    const journalRow = output.records[2]!;

    expect(Buffer.from(exactRow.content ?? "", "utf8")).toEqual(Buffer.from(multilingual, "utf8"));
    expect(exactRow).toMatchObject({
      payload_status: "exact",
      oversized_anchors_omitted: false,
      occurred_at: 2_000,
      occurred_at_iso: iso(2_000),
      origin_time_basis: "stream_timestamp",
      relative_age: "~1s ago",
      disclosure_label: { disclosure_class: "self_private" },
    });
    expect(unavailableRow).toMatchObject({
      content: null,
      payload_status: "check_not_completed_retrieval_unavailable",
      oversized_anchors_omitted: false,
      disclosure_label: { disclosure_class: "unknown" },
    });
    expect(journalRow).toMatchObject({
      content: journal.text,
      occurred_at: journal.created_at,
      occurred_at_iso: iso(journal.created_at),
      origin_time_basis: "journal_created_at",
      relative_age: "~2s ago",
      session_id: null,
      turn_id: null,
      marker_stream_entry_id: null,
      disclosure_label: { disclosure_class: "self_private" },
    });
    expect(journalRow.occurred_at).not.toBe(journal.updated_at);
  });

  it("degrades a throwing thought hydration per row and continues the page", async () => {
    const unavailable = thoughtRecord({ id: THOUGHT_B, timestamp: 2_000 });
    const available = thoughtRecord({ id: THOUGHT_A, timestamp: 1_000 });
    const readThoughtRecord = vi.fn(async (record: StreamEntryIndexRecord) => {
      if (record.entry_id === unavailable.entry_id) {
        throw new Error("stream file unavailable");
      }

      return thoughtEntry(available, "the later row still hydrates exactly");
    });
    const tool = createOwnRecordsListTool({
      listThoughtRecords: thoughtRangeLister([unavailable, available]),
      readThoughtRecord,
      listJournalRecords: () => [],
      clock: new ManualClock(3_000),
    });

    const output = await tool.invoke(
      { since: iso(1_000), until: iso(2_000), kinds: ["thought"] },
      invocationContext(),
    );

    expect(output.records).toHaveLength(2);
    expect(output.records[0]).toMatchObject({
      handle: `thought:${THOUGHT_B}`,
      content: null,
      payload_status: "check_not_completed_retrieval_unavailable",
      oversized_anchors_omitted: false,
      disclosure_label: { disclosure_class: "unknown" },
    });
    expect(output.records[1]).toMatchObject({
      handle: `thought:${THOUGHT_A}`,
      content: "the later row still hydrates exactly",
      payload_status: "exact",
      disclosure_label: { disclosure_class: "self_private" },
    });
    expect(readThoughtRecord).toHaveBeenCalledTimes(2);
  });

  it("pages before budget overflow and returns an unsliced handle-only row on the next page", async () => {
    const small = journalRecord({ id: 2, createdAt: 2_000, text: "small exact payload" });
    const oversizedText = "界".repeat(150_000);
    const oversized = journalRecord({ id: 1, createdAt: 1_000, text: oversizedText });
    const tool = createOwnRecordsListTool({
      listThoughtRecords: () => [],
      readThoughtRecord: () => null,
      listJournalRecords: journalRangeLister([small, oversized]),
      clock: new ManualClock(3_000),
    });

    const firstPage = await tool.invoke(
      {
        since: iso(1_000),
        until: iso(2_000),
        kinds: ["journal"],
        limit: 2,
      },
      invocationContext(),
    );

    expect(firstPage).toMatchObject({
      records: [
        {
          handle: "journal:2",
          content: small.text,
          payload_status: "exact",
          payload_included_chars: small.text.length,
          payload_total_chars: small.text.length,
        },
      ],
      has_more: true,
      page_end_reason: "context_budget",
    });
    expect(firstPage.records.length).toBeLessThan(2);
    expect(firstPage.next_cursor).not.toBeNull();

    const secondPage = await tool.invoke(
      {
        since: iso(1_000),
        until: iso(2_000),
        kinds: ["journal"],
        limit: 2,
        cursor: firstPage.next_cursor!,
      },
      invocationContext(),
    );

    expect(secondPage).toMatchObject({
      records: [
        {
          handle: "journal:1",
          content: null,
          payload_status: "check_not_completed_budget",
          payload_included_chars: 0,
          payload_total_chars: oversizedText.length,
          oversized_anchors_omitted: false,
        },
      ],
      has_more: false,
      page_end_reason: "range_exhausted",
      next_cursor: null,
    });
    expect(JSON.stringify(secondPage)).not.toContain(oversizedText.slice(0, 1_000));
  });

  it("omits oversized anchors when even the payload-less row cannot fit", async () => {
    const oversizedTurnId = `turn-${"锚".repeat(150_000)}`;
    const record = thoughtRecord({
      id: THOUGHT_A,
      timestamp: 1_000,
      turnId: oversizedTurnId,
    });
    const tool = createOwnRecordsListTool({
      listThoughtRecords: thoughtRangeLister([record]),
      readThoughtRecord: () => thoughtEntry(record, "small exact payload"),
      listJournalRecords: () => [],
      clock: new ManualClock(2_000),
    });

    const output = await tool.invoke(
      { since: iso(1_000), until: iso(1_000), kinds: ["thought"] },
      invocationContext(),
    );

    expect(output).toMatchObject({
      records: [
        {
          handle: `thought:${THOUGHT_A}`,
          kind: "thought",
          content: null,
          payload_status: "check_not_completed_budget",
          payload_included_chars: 0,
          payload_total_chars: "small exact payload".length,
          session_id: null,
          turn_id: null,
          stream_entry_id: null,
          journal_entry_id: null,
          marker_stream_entry_id: null,
          self_entity_id: null,
          oversized_anchors_omitted: true,
        },
      ],
      has_more: false,
      next_cursor: null,
    });
    expect(JSON.stringify(output)).not.toContain(oversizedTurnId.slice(0, 1_000));
    expect(JSON.stringify(output).length).toBeLessThan(5_000);
  });
});
