import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { StreamWriter } from "../stream/index.js";
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
});
