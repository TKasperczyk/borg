import { describe, expect, it } from "vitest";

import type { StreamEntry } from "../api/types";
import {
  UNCLAIMED_STREAM_GROUP_ID,
  UNCLAIMED_STREAM_GROUP_LABEL,
  applyStreamStructuralFilters,
  groupStreamEntriesByTurn,
  hasStreamAttachment,
  matchesStreamStructuralFilters,
} from "./stream-grouping";

function entry(input: Partial<StreamEntry> & Pick<StreamEntry, "id" | "kind">): StreamEntry {
  const { id, kind, ...rest } = input;
  return {
    id,
    timestamp: 1,
    kind,
    content: {},
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: "default",
    compressed: false,
    ...rest,
  };
}

describe("stream grouping", () => {
  it("groups loaded entries by turn id and puts null-turn rows in the maintenance lane", () => {
    const groups = groupStreamEntriesByTurn([
      entry({ id: "strm_turn_a", kind: "user_msg", turn_id: "turn_a", timestamp: 10 }),
      entry({ id: "strm_maintenance", kind: "internal_event", timestamp: 11 }),
      entry({ id: "strm_turn_b", kind: "agent_msg", turn_id: "turn_b", timestamp: 12 }),
    ]);

    expect(groups.map((group) => group.id)).toEqual([
      "turn_b",
      UNCLAIMED_STREAM_GROUP_ID,
      "turn_a",
    ]);
    expect(groups[1]).toMatchObject({
      turnId: null,
      label: UNCLAIMED_STREAM_GROUP_LABEL,
      status: "maintenance",
      entryCount: 1,
    });
  });

  it("sorts inside a turn by session-global entry index before timestamp", () => {
    const groups = groupStreamEntriesByTurn([
      entry({
        id: "strm_newer_ts",
        kind: "agent_msg",
        turn_id: "turn_a",
        timestamp: 30,
        entry_index: 2,
      }),
      entry({
        id: "strm_newer_index",
        kind: "tool_result",
        turn_id: "turn_a",
        timestamp: 20,
        entry_index: 3,
      }),
    ]);

    expect(groups).toHaveLength(1);
    expect(groups[0]?.entries.map((item) => item.id)).toEqual([
      "strm_newer_index",
      "strm_newer_ts",
    ]);
  });

  it("falls back to timestamp ordering inside legacy mixed-index groups", () => {
    const groups = groupStreamEntriesByTurn([
      entry({
        id: "strm_with_index",
        kind: "agent_msg",
        turn_id: "turn_a",
        timestamp: 10,
        entry_index: 3,
      }),
      entry({ id: "strm_without_index", kind: "user_msg", turn_id: "turn_a", timestamp: 20 }),
    ]);

    expect(groups[0]?.entries.map((item) => item.id)).toEqual([
      "strm_without_index",
      "strm_with_index",
    ]);
  });

  it("keeps partial turn groups as loaded-window groups", () => {
    const groups = groupStreamEntriesByTurn([
      entry({ id: "strm_turn_a_tail", kind: "agent_msg", turn_id: "turn_a", timestamp: 20 }),
      entry({ id: "strm_turn_a_mid", kind: "thought", turn_id: "turn_a", timestamp: 10 }),
    ]);

    expect(groups).toHaveLength(1);
    expect(groups[0]).toMatchObject({
      id: "turn_a",
      startTimestamp: 10,
      endTimestamp: 20,
      entryCount: 2,
      status: "active",
    });
  });

  it("orders groups newest-first using max entry index when both groups have indexes", () => {
    const groups = groupStreamEntriesByTurn([
      entry({
        id: "strm_old_ts_high_index",
        kind: "agent_msg",
        turn_id: "turn_later",
        timestamp: 10,
        entry_index: 9,
      }),
      entry({
        id: "strm_new_ts_low_index",
        kind: "agent_msg",
        turn_id: "turn_earlier",
        timestamp: 99,
        entry_index: 3,
      }),
    ]);

    expect(groups.map((group) => group.id)).toEqual(["turn_later", "turn_earlier"]);
  });

  it("exposes structural filters without reading text semantics", () => {
    const rows = [
      entry({
        id: "strm_aborted",
        kind: "agent_suppressed",
        turn_id: "turn_a",
        turn_status: "aborted",
        compressed: true,
        source_message_key: {
          source_type: "demo",
          source_external_id: "thread",
          external_message_id: "msg_1",
        },
      }),
      entry({
        id: "strm_attachment",
        kind: "tool_result",
        content: { attachment_id: "att_1" },
        turn_id: "turn_b",
      }),
      entry({ id: "strm_maintenance", kind: "internal_event" }),
    ];

    expect(hasStreamAttachment(rows[1]!)).toBe(true);
    expect(matchesStreamStructuralFilters(rows[0]!, { aborted: true, compressed: true })).toBe(
      true,
    );
    expect(applyStreamStructuralFilters(rows, { hasTurnId: true }).map((item) => item.id)).toEqual([
      "strm_aborted",
      "strm_attachment",
    ]);
    expect(
      applyStreamStructuralFilters(rows, { hasAttachment: true }).map((item) => item.id),
    ).toEqual(["strm_attachment"]);
    expect(
      applyStreamStructuralFilters(rows, { hasSourceMessageKey: true }).map((item) => item.id),
    ).toEqual(["strm_aborted"]);
  });
});
