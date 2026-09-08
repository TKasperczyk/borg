import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { hydrateTurnMechanismEvidence } from "../cognition/mechanism-evidence.js";
import { createWorkingMemory } from "../memory/working/index.js";
import { openDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import { StreamEntryIndexRepository, streamEntryIndexMigrations } from "./entry-index.js";
import { renderAnsweredWindowEvidence, type AnsweredStreamWindow } from "./answered-window.js";
import { StreamReader } from "./stream-reader.js";
import { StreamWriter } from "./stream-writer.js";
import type { StreamEntry } from "./types.js";

const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const cleanup of cleanups.splice(0)) cleanup();
});

function harness() {
  const dataDir = mkdtempSync(join(tmpdir(), "borg-answered-window-"));
  const db = openDatabase(":memory:", { migrations: streamEntryIndexMigrations });
  const clock = new ManualClock(1_000);
  const index = new StreamEntryIndexRepository({ db, dataDir });
  const writer = new StreamWriter({ dataDir, clock, entryIndex: index });
  cleanups.push(() => {
    db.close();
    rmSync(dataDir, { recursive: true, force: true });
  });
  async function respond(
    entries: StreamEntry[],
    kind: "agent_msg" | "agent_suppressed" | "agent_observed" | "internal_event" = "agent_msg",
  ) {
    const last = entries.at(-1)!;
    return writer.append({
      kind,
      content: "回答",
      turn_id: "answer-turn",
      response_to: {
        kind: "stream_backlog",
        from_cursor_exclusive: null,
        through_cursor_inclusive: { ts: last.timestamp, entryId: last.id },
        source_entry_ids: entries.map((entry) => entry.id),
        count: entries.length,
      },
    });
  }
  return {
    dataDir,
    db,
    clock,
    index,
    writer,
    respond,
    describe: () => index.describeAnsweredWindow(DEFAULT_SESSION_ID, clock.now()),
  };
}

describe("answered-window edge evidence", () => {
  it("distinguishes no edge with no inputs from no edge with recorded inputs", async () => {
    const h = harness();
    expect(h.describe()).toMatchObject({
      state: "no_answered_edge",
      basis: null,
      outside: { without_edge: 0, arrived_after_edge: null },
    });
    await h.writer.append({ kind: "user_msg", content: "¿Hola?" });
    expect(h.describe().outside.without_edge).toBe(1);
  });

  it.each(["agent_msg", "agent_suppressed", "agent_observed", "internal_event"] as const)(
    "pins the %s terminal basis and renders exact none outside",
    async (kind) => {
      const h = harness();
      const inbound = await h.writer.append({ kind: "user_msg", content: "你好" });
      h.clock.advance(50);
      const terminal = await h.respond([inbound], kind);
      const window: AnsweredStreamWindow = {
        responseTo: terminal.response_to!,
        terminalCursor: { ts: terminal.timestamp, entryId: terminal.id },
      };
      const evidence = h.describe();
      expect(evidence).toMatchObject({
        state: "recorded",
        basis: {
          turn_id: "answer-turn",
          response_entry_id: window.terminalCursor.entryId,
          response_at: 1_050,
          response_kind: kind,
          last_answered_entry_id: inbound.id,
          last_answered_at: 1_000,
          answered_entry_count: 1,
        },
        outside: {
          state: "none",
          arrived_after_edge: 0,
          unselected_within_window: 0,
          before_window: 0,
        },
      });
      const rendered = renderAnsweredWindowEvidence(evidence);
      expect(rendered).toContain(terminal.id);
      expect(rendered).toContain(inbound.id);
      expect(rendered).toContain('"state":"none"');
      expect(rendered).toContain("neither gate recall nor decide whether to respond");
      const mechanism = await hydrateTurnMechanismEvidence({
        dataDir: h.dataDir,
        sessionId: DEFAULT_SESSION_ID,
        nowMs: h.clock.now(),
        entryIndex: h.index,
        workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, h.clock.now()),
        createStreamReader: (sessionId) => new StreamReader({ dataDir: h.dataDir, sessionId }),
      });
      expect(mechanism.answeredWindow).toEqual(evidence);
      const reopenedIndex = new StreamEntryIndexRepository({ db: h.db, dataDir: h.dataDir });
      expect(reopenedIndex.describeAnsweredWindow(DEFAULT_SESSION_ID, h.clock.now())).toEqual(
        evidence,
      );
    },
  );

  it("counts arrivals during generation and after the response, using record order at equal timestamps", async () => {
    const h = harness();
    const inbound = await h.writer.append({ kind: "user_msg", content: "first" });
    await h.writer.append({ kind: "user_msg", content: "same timestamp, later arrival" });
    await h.respond([inbound]);
    await h.writer.append({ kind: "thought", content: "internal record is not another inbound" });
    h.clock.advance(10);
    await h.writer.append({ kind: "user_msg", content: "after response" });
    expect(h.describe().outside).toMatchObject({
      state: "arrived_after_edge",
      arrived_after_edge: 2,
      unselected_within_window: 0,
      before_window: 0,
    });
    expect(renderAnsweredWindowEvidence(h.describe())).toContain('"arrived_after_edge":2');
  });

  it("labels a hole in the exact answered set separately from later arrivals", async () => {
    const h = harness();
    const first = await h.writer.append({ kind: "user_msg", content: "一" });
    await h.writer.append({ kind: "user_msg", content: "not in this response" });
    const last = await h.writer.append({ kind: "user_msg", content: "三" });
    await h.respond([first, last]);
    expect(h.describe()).toMatchObject({
      basis: { answered_entry_count: 2 },
      outside: {
        state: "outside_answered_set",
        arrived_after_edge: 0,
        unselected_within_window: 1,
      },
    });
  });

  it("reports missing basis records as unavailable, never zero", async () => {
    const h = harness();
    const input = await h.writer.append({ kind: "user_msg", content: "input" });
    await h.respond([input]);
    h.db.prepare("DELETE FROM stream_entry_index WHERE entry_id = ?").run(input.id);
    expect(h.describe()).toMatchObject({
      state: "basis_records_unavailable",
      outside: { state: "unavailable", arrived_after_edge: null },
    });
  });
});
