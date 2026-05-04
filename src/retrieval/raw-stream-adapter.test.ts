import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { ABORTED_TURN_EVENT, StreamWriter } from "../stream/index.js";
import { ManualClock } from "../util/clock.js";
import { createSessionId } from "../util/ids.js";

import { RawStreamAdapter } from "./raw-stream-adapter.js";

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
    const resolved = await adapter.resolveSourceIds([active.id, aborted.id]);

    expect(adapter.recent({ sessionId, limit: 10 }).map((entry) => entry.content)).toEqual([
      activeContent,
    ]);
    expect([...resolved.keys()]).toEqual([active.id]);
  });
});
