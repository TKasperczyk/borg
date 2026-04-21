import { appendFileSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import { StreamError } from "../util/errors.js";
import { getSessionStreamPath, StreamReader, StreamWriter } from "./index.js";

describe("stream", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("appends and reads stream entries", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(100);
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    const first = await writer.append({
      kind: "user_msg",
      content: "hello",
    });

    clock.advance(50);

    await writer.appendMany([
      {
        kind: "thought",
        content: { note: "thinking" },
      },
      {
        kind: "internal_event",
        content: "done",
        compressed: true,
      },
    ]);

    writer.close();

    expect(first.timestamp).toBe(100);
    expect(first.compressed).toBe(false);

    const reader = new StreamReader({
      dataDir: tempDir,
    });

    const filtered = [];
    for await (const entry of reader.iterate({
      sinceTs: 120,
      kinds: ["thought", "internal_event"],
      limit: 1,
    })) {
      filtered.push(entry);
    }

    expect(filtered).toHaveLength(1);
    expect(filtered[0]?.kind).toBe("thought");
    expect(reader.tail(2).map((entry) => entry.kind)).toEqual(["thought", "internal_event"]);
  });

  it("skips invalid lines instead of crashing the reader", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const logger = {
      error: vi.fn(),
    };

    const streamPath = getSessionStreamPath(tempDir, DEFAULT_SESSION_ID);
    mkdirSync(join(tempDir, "stream"), { recursive: true });
    appendFileSync(streamPath, '{"broken"\n', { encoding: "utf8", flag: "a" });
    appendFileSync(
      streamPath,
      `${JSON.stringify({
        id: "strm_aaaaaaaaaaaaaaaa",
        timestamp: 1,
        kind: "user_msg",
        content: "ok",
        session_id: "default",
        compressed: false,
      })}\n`,
      { encoding: "utf8", flag: "a" },
    );

    const reader = new StreamReader({
      dataDir: tempDir,
      logger,
    });

    expect(reader.tail(10)).toHaveLength(1);
    expect(logger.error).toHaveBeenCalled();
  });

  it("skips corrupt trailing lines when tailing", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(50),
    });

    await writer.append({
      kind: "user_msg",
      content: "stable",
    });
    writer.close();

    const streamPath = getSessionStreamPath(tempDir, DEFAULT_SESSION_ID);
    appendFileSync(streamPath, '{"truncated"', { encoding: "utf8", flag: "a" });

    const reader = new StreamReader({
      dataDir: tempDir,
      logger: {
        error: vi.fn(),
      },
    });

    expect(reader.tail(1)).toMatchObject([
      {
        kind: "user_msg",
        content: "stable",
      },
    ]);
  });

  it("rejects non-json stream content at serialization time", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(1),
    });

    await expect(
      writer.append({
        kind: "user_msg",
        content: { bad: 1n },
      }),
    ).rejects.toMatchObject<Partial<StreamError>>({
      code: "STREAM_SERIALIZE_FAILED",
    });
  });
});
