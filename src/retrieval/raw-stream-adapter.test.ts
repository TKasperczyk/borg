import { mkdtempSync, rmSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { ABORTED_TURN_EVENT, StreamWriter, StreamEntryIndexRepository } from "../stream/index.js";
import { createMigrations } from "../borg/storage-setup.js";
import { openDatabase } from "../storage/sqlite/index.js";
import { EntityRepository } from "../memory/commitments/repository.js";
import { ManualClock } from "../util/clock.js";
import { createSessionId } from "../util/ids.js";

import { RawStreamAdapter } from "./raw-stream-adapter.js";
import { CitationResolver } from "./citations.js";

const tempDirs: string[] = [];

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "borg-raw-stream-adapter-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

describe("RawStreamAdapter", () => {
  it("resolves historical entity handles in recency, indexed citations and fallback scans", async () => {
    const dataDir = tempDir();
    const db = openDatabase(join(dataDir, "borg.db"), { migrations: createMigrations() });
    const entities = new EntityRepository({ db });
    const source = entities.add({ canonicalName: "Earlier identity" });
    const target = entities.add({ canonicalName: "Surviving identity" });
    const entryIndex = new StreamEntryIndexRepository({ db, dataDir, entityRepository: entities });
    const writer = new StreamWriter({ dataDir, entryIndex });
    try {
      const entry = await writer.append({
        kind: "user_msg",
        content: "Historical message",
        sender_entity_id: source.id,
        reply_target_entity_id: source.id,
      });
      const bytes = readFileSync(join(dataDir, "stream", "default.jsonl"));
      entities.merge(source.id, target.id);
      const adapter = new RawStreamAdapter({ dataDir, entryIndex });
      expect(adapter.recent()[0]).toMatchObject({
        sender_entity_id: target.id,
        reply_target_entity_id: target.id,
      });
      expect((await adapter.resolveSourceIds([entry.id])).get(entry.id)).toMatchObject({
        sender_entity_id: target.id,
      });
      const lookup = vi.spyOn(entryIndex, "lookupMany").mockReturnValue(new Map());
      try {
        expect((await adapter.resolveSourceIds([entry.id])).get(entry.id)).toMatchObject({
          sender_entity_id: target.id,
        });
      } finally {
        lookup.mockRestore();
      }
      expect(readFileSync(join(dataDir, "stream", "default.jsonl"))).toEqual(bytes);
      expect(
        db
          .prepare("SELECT sender_entity_id FROM stream_entry_index WHERE entry_id = ?")
          .get(entry.id),
      ).toEqual({ sender_entity_id: source.id });
    } finally {
      writer.close();
      db.close();
    }
  });

  it("scopes recent entries to the requested session file", async () => {
    const dir = tempDir();
    const clock = new ManualClock();
    const firstSession = createSessionId();
    const secondSession = createSessionId();
    const firstWriter = new StreamWriter({
      dataDir: dir,
      sessionId: firstSession,
      clock,
    });
    const secondWriter = new StreamWriter({
      dataDir: dir,
      sessionId: secondSession,
      clock,
    });

    clock.set(100);
    const firstSessionOlder = await firstWriter.append({
      kind: "user_msg",
      content: "first session older",
    });
    clock.set(200);
    const secondSessionEntry = await secondWriter.append({
      kind: "user_msg",
      content: "second session recent",
    });
    clock.set(300);
    const firstSessionNewer = await firstWriter.append({
      kind: "agent_msg",
      content: "first session newer",
    });
    firstWriter.close();
    secondWriter.close();

    const adapter = new RawStreamAdapter({ dataDir: dir });

    expect(adapter.recent({ sessionId: firstSession, limit: 10 }).map((entry) => entry.id)).toEqual(
      [firstSessionNewer.id, firstSessionOlder.id],
    );
    expect(adapter.recent({ limit: 10 }).map((entry) => entry.id)).toEqual([
      firstSessionNewer.id,
      secondSessionEntry.id,
      firstSessionOlder.id,
    ]);
  });

  it("excludes aborted turn entries from recency and source-id resolution", async () => {
    const dir = tempDir();
    const clock = new ManualClock();
    const sessionId = createSessionId();
    const writer = new StreamWriter({
      dataDir: dir,
      sessionId,
      clock,
    });
    const activeContent = "active user";
    const abortedTurnId = "aborted-raw-stream-turn";

    clock.set(100);
    const active = await writer.append({
      kind: "user_msg",
      content: activeContent,
    });
    clock.set(200);
    const aborted = await writer.append({
      kind: "thought",
      content: "aborted plan",
      turn_id: abortedTurnId,
      turn_status: "active",
    });
    clock.set(300);
    await writer.append({
      kind: "internal_event",
      turn_id: abortedTurnId,
      turn_status: "aborted",
      content: {
        event: ABORTED_TURN_EVENT,
        turn_id: abortedTurnId,
        reason: "turn failed",
      },
    });
    writer.close();

    const adapter = new RawStreamAdapter({ dataDir: dir });
    const citationResolver = new CitationResolver({ dataDir: dir });
    const resolved = await adapter.resolveSourceIds([active.id, aborted.id]);
    const directlyResolved = await citationResolver.resolveCitationEntries([aborted.id]);

    expect(adapter.recent({ sessionId, limit: 10 }).map((entry) => entry.content)).toEqual([
      activeContent,
    ]);
    expect([...resolved.keys()]).toEqual([active.id]);
    expect(directlyResolved.size).toBe(0);
  });

  it("excludes diagnostic internal events from recent conversational evidence", async () => {
    const dir = tempDir();
    const clock = new ManualClock();
    const sessionId = createSessionId();
    const writer = new StreamWriter({
      dataDir: dir,
      sessionId,
      clock,
    });

    clock.set(100);
    const user = await writer.append({
      kind: "user_msg",
      content: "normal source",
    });
    clock.set(200);
    await writer.append({
      kind: "internal_event",
      content: {
        event: "frame_anomaly_gate",
        source_stream_entry_id: user.id,
      },
    });
    clock.set(300);
    const agent = await writer.append({
      kind: "agent_msg",
      content: "normal response",
    });
    writer.close();

    const adapter = new RawStreamAdapter({ dataDir: dir });

    expect(adapter.recent({ sessionId, limit: 10 }).map((entry) => entry.id)).toEqual([
      agent.id,
      user.id,
    ]);
  });
});
