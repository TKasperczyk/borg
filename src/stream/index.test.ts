import {
  appendFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { hostname } from "node:os";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import { StreamError } from "../util/errors.js";
import { getSessionStreamPath, streamEntrySchema, StreamReader, StreamWriter } from "./index.js";

describe("stream", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  function legacyTailFromDisk(streamPath: string, n: number) {
    if (n <= 0) {
      return [];
    }

    const lines = readFileSync(streamPath, "utf8").split("\n");
    const entries = [];

    for (let index = lines.length - 1; index >= 0 && entries.length < n; index -= 1) {
      const line = lines[index] ?? "";

      if (line.trim() === "") {
        continue;
      }

      try {
        const parsed = streamEntrySchema.safeParse(JSON.parse(line) as unknown);

        if (parsed.success) {
          entries.push(parsed.data);
        }
      } catch {
        // Legacy behavior ignored unreadable lines while searching for more.
      }
    }

    return entries.reverse();
  }

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
        kind: "agent_suppressed",
        content: { reason: "generation_gate" },
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
    expect(reader.tail(3).map((entry) => entry.kind)).toEqual([
      "thought",
      "agent_suppressed",
      "internal_event",
    ]);
  });

  it("assigns append timestamps after acquiring the stream lock", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const streamDir = join(tempDir, "stream");
    const lockPath = join(streamDir, "default.jsonl.lock");
    const clock = new ManualClock(100);

    mkdirSync(streamDir, { recursive: true });
    writeFileSync(
      lockPath,
      JSON.stringify({
        pid: process.pid,
        host: hostname(),
        timestamp: Date.now(),
      }),
    );

    const delayedWriter = new StreamWriter({
      dataDir: tempDir,
      clock,
      lockRetryDelayMs: 100,
    });
    const fastWriter = new StreamWriter({
      dataDir: tempDir,
      clock,
      lockRetryDelayMs: 1,
    });

    try {
      const delayedAppend = delayedWriter.append({
        kind: "user_msg",
        content: "waited for the lock",
      });

      await new Promise((resolve) => {
        setTimeout(resolve, 10);
      });

      unlinkSync(lockPath);
      clock.set(200);
      const firstPhysicalEntry = await fastWriter.append({
        kind: "agent_msg",
        content: "acquired first after manual release",
      });
      clock.set(300);
      const secondPhysicalEntry = await delayedAppend;

      const entries = new StreamReader({
        dataDir: tempDir,
      }).tail(2);

      expect(entries.map((entry) => entry.id)).toEqual([
        firstPhysicalEntry.id,
        secondPhysicalEntry.id,
      ]);
      expect(entries.map((entry) => entry.timestamp)).toEqual([200, 300]);
    } finally {
      delayedWriter.close();
      fastWriter.close();
    }
  });

  it("matches the legacy full-scan tail behavior on small files", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(100),
    });

    try {
      await writer.append({ kind: "user_msg", content: "alpha" });
      await writer.append({ kind: "agent_msg", content: "beta" });
      await writer.append({ kind: "thought", content: { note: "gamma" } });
      await writer.append({ kind: "internal_event", content: "delta" });
    } finally {
      writer.close();
    }

    const streamPath = getSessionStreamPath(tempDir, DEFAULT_SESSION_ID);
    appendFileSync(streamPath, "\n", { encoding: "utf8", flag: "a" });
    appendFileSync(streamPath, '{"broken"\n', { encoding: "utf8", flag: "a" });

    const reader = new StreamReader({
      dataDir: tempDir,
      logger: {
        error: vi.fn(),
      },
    });

    expect(reader.tail(3)).toEqual(legacyTailFromDisk(streamPath, 3));
  });

  it("tails the last entries from a very large synthetic stream", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const streamPath = getSessionStreamPath(tempDir, DEFAULT_SESSION_ID);

    mkdirSync(join(tempDir, "stream"), { recursive: true });

    const totalEntries = 100_000;
    const batchSize = 5_000;

    for (let start = 0; start < totalEntries; start += batchSize) {
      const batchLines: string[] = [];
      const end = Math.min(start + batchSize, totalEntries);

      for (let index = start; index < end; index += 1) {
        batchLines.push(
          `${JSON.stringify({
            id: `strm_${String(index).padStart(16, "0")}`,
            timestamp: index,
            kind: "user_msg",
            content: `entry-${index}`,
            session_id: "default",
            compressed: false,
          })}\n`,
        );
      }

      appendFileSync(streamPath, batchLines.join(""), { encoding: "utf8", flag: "a" });
    }

    const tail = new StreamReader({
      dataDir: tempDir,
    }).tail(10);

    expect(tail).toHaveLength(10);
    expect(tail.map((entry) => entry.content)).toEqual(
      Array.from({ length: 10 }, (_, offset) => `entry-${totalEntries - 10 + offset}`),
    );
  });

  it("parses entries that straddle the 64KB reverse-read chunk boundary", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const streamPath = getSessionStreamPath(tempDir, DEFAULT_SESSION_ID);

    mkdirSync(join(tempDir, "stream"), { recursive: true });
    appendFileSync(
      streamPath,
      [
        {
          id: "strm_0000000000000001",
          timestamp: 1,
          kind: "user_msg",
          content: "prefix",
          session_id: "default",
          compressed: false,
        },
        {
          id: "strm_0000000000000002",
          timestamp: 2,
          kind: "agent_msg",
          content: "x".repeat(70_000),
          session_id: "default",
          compressed: false,
        },
        {
          id: "strm_0000000000000003",
          timestamp: 3,
          kind: "internal_event",
          content: "suffix",
          session_id: "default",
          compressed: false,
        },
      ]
        .map((entry) => `${JSON.stringify(entry)}\n`)
        .join(""),
      { encoding: "utf8", flag: "a" },
    );

    expect(
      new StreamReader({
        dataDir: tempDir,
      }).tail(2),
    ).toMatchObject([
      {
        id: "strm_0000000000000002",
        content: "x".repeat(70_000),
      },
      {
        id: "strm_0000000000000003",
        content: "suffix",
      },
    ]);
  });

  it("tails across a 5MB line without rebuilding the full carry buffer", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const streamPath = getSessionStreamPath(tempDir, DEFAULT_SESSION_ID);
    const largeContent = "x".repeat(5 * 1024 * 1024);

    mkdirSync(join(tempDir, "stream"), { recursive: true });
    appendFileSync(
      streamPath,
      [
        {
          id: "strm_0000000000000100",
          timestamp: 100,
          kind: "user_msg",
          content: "prefix",
          session_id: "default",
          compressed: false,
        },
        {
          id: "strm_0000000000000101",
          timestamp: 101,
          kind: "agent_msg",
          content: largeContent,
          session_id: "default",
          compressed: false,
        },
        {
          id: "strm_0000000000000102",
          timestamp: 102,
          kind: "internal_event",
          content: "suffix",
          session_id: "default",
          compressed: false,
        },
      ]
        .map((entry) => `${JSON.stringify(entry)}\n`)
        .join(""),
      { encoding: "utf8", flag: "a" },
    );

    const startedAt = Date.now();
    const tail = new StreamReader({
      dataDir: tempDir,
    }).tail(2);
    const elapsedMs = Date.now() - startedAt;

    expect(tail).toHaveLength(2);
    expect(tail.map((entry) => entry.id)).toEqual([
      "strm_0000000000000101",
      "strm_0000000000000102",
    ]);
    expect(typeof tail[0]?.content).toBe("string");
    expect((tail[0]?.content as string).length).toBe(largeContent.length);
    expect(tail[1]?.content).toBe("suffix");
    expect(elapsedMs).toBeLessThan(5_000);
  });

  it("reads a final line even when the file has no trailing newline", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const streamPath = getSessionStreamPath(tempDir, DEFAULT_SESSION_ID);

    mkdirSync(join(tempDir, "stream"), { recursive: true });
    appendFileSync(
      streamPath,
      [
        JSON.stringify({
          id: "strm_0000000000000010",
          timestamp: 10,
          kind: "user_msg",
          content: "alpha",
          session_id: "default",
          compressed: false,
        }),
        JSON.stringify({
          id: "strm_0000000000000011",
          timestamp: 11,
          kind: "agent_msg",
          content: "omega",
          session_id: "default",
          compressed: false,
        }),
      ].join("\n"),
      { encoding: "utf8", flag: "a" },
    );

    expect(
      new StreamReader({
        dataDir: tempDir,
      }).tail(2),
    ).toMatchObject([
      {
        id: "strm_0000000000000010",
        content: "alpha",
      },
      {
        id: "strm_0000000000000011",
        content: "omega",
      },
    ]);
  });

  it("uses file order to resume within same-millisecond ties", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const streamPath = getSessionStreamPath(tempDir, DEFAULT_SESSION_ID);

    mkdirSync(join(tempDir, "stream"), { recursive: true });
    appendFileSync(
      streamPath,
      [
        {
          id: "strm_aaaaaaaaaaaaaaaa",
          timestamp: 100,
          kind: "user_msg",
          content: "A",
          session_id: "default",
          compressed: false,
        },
        {
          id: "strm_bbbbbbbbbbbbbbbb",
          timestamp: 100,
          kind: "user_msg",
          content: "B",
          session_id: "default",
          compressed: false,
        },
        {
          id: "strm_cccccccccccccccc",
          timestamp: 100,
          kind: "user_msg",
          content: "C",
          session_id: "default",
          compressed: false,
        },
      ]
        .map((entry) => `${JSON.stringify(entry)}\n`)
        .join(""),
      { encoding: "utf8", flag: "a" },
    );

    const reader = new StreamReader({
      dataDir: tempDir,
    });
    const afterA = [];
    const afterB = [];

    for await (const entry of reader.iterate({
      sinceCursor: {
        ts: 100,
        entryId: "strm_aaaaaaaaaaaaaaaa" as never,
      },
    })) {
      afterA.push(entry.content);
    }

    for await (const entry of reader.iterate({
      sinceCursor: {
        ts: 100,
        entryId: "strm_bbbbbbbbbbbbbbbb" as never,
      },
    })) {
      afterB.push(entry.content);
    }

    expect(afterA).toEqual(["B", "C"]);
    expect(afterB).toEqual(["C"]);
  });

  it("replays same-timestamp entries when the cursor id is not found", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const streamPath = getSessionStreamPath(tempDir, DEFAULT_SESSION_ID);

    mkdirSync(join(tempDir, "stream"), { recursive: true });
    appendFileSync(
      streamPath,
      [
        {
          id: "strm_aaaaaaaaaaaaaaaa",
          timestamp: 100,
          kind: "user_msg",
          content: "A",
          session_id: "default",
          compressed: false,
        },
        {
          id: "strm_bbbbbbbbbbbbbbbb",
          timestamp: 100,
          kind: "agent_msg",
          content: "B",
          session_id: "default",
          compressed: false,
        },
        {
          id: "strm_cccccccccccccccc",
          timestamp: 101,
          kind: "internal_event",
          content: "C",
          session_id: "default",
          compressed: false,
        },
      ]
        .map((entry) => `${JSON.stringify(entry)}\n`)
        .join(""),
      { encoding: "utf8", flag: "a" },
    );

    const reader = new StreamReader({
      dataDir: tempDir,
    });
    const replayed = [];

    for await (const entry of reader.iterate({
      sinceCursor: {
        ts: 100,
        entryId: "strm_zzzzzzzzzzzzzzzz" as never,
      },
    })) {
      replayed.push(entry.content);
    }

    expect(replayed).toEqual(["A", "B", "C"]);
  });

  it("returns later same-millisecond appends when resuming from a cursor", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const firstWriter = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(100),
    });
    const first = await firstWriter.append({
      kind: "user_msg",
      content: "A",
    });
    firstWriter.close();

    const reader = new StreamReader({
      dataDir: tempDir,
    });
    const beforeAppend = [];

    for await (const entry of reader.iterate({
      sinceCursor: {
        ts: first.timestamp,
        entryId: first.id,
      },
    })) {
      beforeAppend.push(entry.content);
    }

    const secondWriter = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(100),
    });
    const second = await secondWriter.append({
      kind: "agent_msg",
      content: "B",
    });
    secondWriter.close();

    const afterAppend = [];
    for await (const entry of reader.iterate({
      sinceCursor: {
        ts: first.timestamp,
        entryId: first.id,
      },
    })) {
      afterAppend.push(entry.content);
    }

    expect(beforeAppend).toEqual([]);
    expect(afterAppend).toEqual([second.content]);
  });

  it("keeps the cursor timestamp as a hard lower bound after the cursor point", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const streamPath = getSessionStreamPath(tempDir, DEFAULT_SESSION_ID);

    mkdirSync(join(tempDir, "stream"), { recursive: true });
    appendFileSync(
      streamPath,
      [
        {
          id: "strm_aaaaaaaaaaaaaaaa",
          timestamp: 200,
          kind: "user_msg",
          content: "A",
          session_id: "default",
          compressed: false,
        },
        {
          id: "strm_bbbbbbbbbbbbbbbb",
          timestamp: 250,
          kind: "agent_msg",
          content: "B",
          session_id: "default",
          compressed: false,
        },
        {
          id: "strm_cccccccccccccccc",
          timestamp: 150,
          kind: "internal_event",
          content: "C",
          session_id: "default",
          compressed: false,
        },
      ]
        .map((entry) => `${JSON.stringify(entry)}\n`)
        .join(""),
      { encoding: "utf8", flag: "a" },
    );

    const reader = new StreamReader({
      dataDir: tempDir,
    });
    const resumed = [];

    for await (const entry of reader.iterate({
      sinceCursor: {
        ts: 200,
        entryId: "strm_aaaaaaaaaaaaaaaa" as never,
      },
    })) {
      resumed.push(entry.content);
    }

    expect(resumed).toEqual(["B"]);
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
    ).rejects.toMatchObject({
      code: "STREAM_SERIALIZE_FAILED",
    });
  });
});
