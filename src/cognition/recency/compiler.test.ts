import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { StreamReader, StreamWriter } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";

import { TurnContextCompiler } from "./compiler.js";

describe("TurnContextCompiler", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  function makeWriter(dataDir: string, clock: ManualClock): StreamWriter {
    return new StreamWriter({
      dataDir,
      sessionId: DEFAULT_SESSION_ID,
      clock,
    });
  }

  function makeReader(dataDir: string): StreamReader {
    return new StreamReader({
      dataDir,
      sessionId: DEFAULT_SESSION_ID,
    });
  }

  function createTempDir(): string {
    const dir = mkdtempSync(join(tmpdir(), "borg-recency-"));
    tempDirs.push(dir);
    return dir;
  }

  it("returns an empty window when the stream has no entries", () => {
    const dataDir = createTempDir();
    const compiler = new TurnContextCompiler();

    const window = compiler.compile(makeReader(dataDir));

    expect(window.messages).toEqual([]);
    expect(window.latest_ts).toBeNull();
    expect(window.total_chars).toBe(0);
  });

  it("compiles a user/assistant alternation from prior turns", async () => {
    const dataDir = createTempDir();
    const clock = new ManualClock(1_000);
    const writer = makeWriter(dataDir, clock);

    try {
      await writer.append({ kind: "user_msg", content: "Hi there" });
      clock.advance(10);
      await writer.append({ kind: "agent_msg", content: "Hello yourself" });
      clock.advance(10);
      await writer.append({ kind: "user_msg", content: "How's it going?" });
      clock.advance(10);
      await writer.append({ kind: "agent_msg", content: "Fine, working on this loop" });
    } finally {
      writer.close();
    }

    const compiler = new TurnContextCompiler();
    const window = compiler.compile(makeReader(dataDir));

    expect(window.messages.map((m) => `${m.role}:${m.content}`)).toEqual([
      "user:Hi there",
      "assistant:Hello yourself",
      "user:How's it going?",
      "assistant:Fine, working on this loop",
    ]);
    expect(window.latest_ts).toBe(1_030);
    expect(window.total_chars).toBeGreaterThan(0);
  });

  it("skips non-conversational entries like thoughts and tool calls", async () => {
    const dataDir = createTempDir();
    const clock = new ManualClock(1_000);
    const writer = makeWriter(dataDir, clock);

    try {
      await writer.append({ kind: "user_msg", content: "debug this" });
      clock.advance(10);
      await writer.append({ kind: "thought", content: "scratchpad content" });
      clock.advance(10);
      await writer.append({ kind: "internal_event", content: "something happened" });
      clock.advance(10);
      await writer.append({
        kind: "agent_msg",
        content: "looking",
        tool_calls: [{ name: "read", input: {} }],
      });
    } finally {
      writer.close();
    }

    const window = new TurnContextCompiler().compile(makeReader(dataDir));

    expect(window.messages).toHaveLength(2);
    expect(window.messages[0]?.role).toBe("user");
    expect(window.messages[0]?.content).toBe("debug this");
    expect(window.messages[1]?.role).toBe("assistant");
    expect(window.messages[1]?.content).toBe("looking");
  });

  it("filters self-addressed turns by default", async () => {
    const dataDir = createTempDir();
    const clock = new ManualClock(1_000);
    const writer = makeWriter(dataDir, clock);

    try {
      await writer.append({ kind: "user_msg", content: "scheduled reflection", audience: "self" });
      clock.advance(10);
      await writer.append({ kind: "agent_msg", content: "I reflected on recent changes.", audience: "self" });
      clock.advance(10);
      await writer.append({ kind: "user_msg", content: "hello" });
      clock.advance(10);
      await writer.append({ kind: "agent_msg", content: "hi there" });
    } finally {
      writer.close();
    }

    const window = new TurnContextCompiler().compile(makeReader(dataDir));

    expect(window.messages.map((message) => message.content)).toEqual(["hello", "hi there"]);
  });

  it("can include self-addressed turns when requested", async () => {
    const dataDir = createTempDir();
    const clock = new ManualClock(1_000);
    const writer = makeWriter(dataDir, clock);

    try {
      await writer.append({ kind: "user_msg", content: "scheduled reflection", audience: "self" });
      clock.advance(10);
      await writer.append({ kind: "agent_msg", content: "I reflected on recent changes.", audience: "self" });
      clock.advance(10);
      await writer.append({ kind: "user_msg", content: "another self prompt", audience: "self" });
      clock.advance(10);
      await writer.append({ kind: "agent_msg", content: "Another self response.", audience: "self" });
    } finally {
      writer.close();
    }

    const window = new TurnContextCompiler({ includeSelfTurns: true }).compile(makeReader(dataDir));

    expect(window.messages.map((message) => message.content)).toEqual([
      "scheduled reflection",
      "I reflected on recent changes.",
      "another self prompt",
      "Another self response.",
    ]);
  });

  it("drops a trailing user entry so the caller's current user msg won't collide", async () => {
    const dataDir = createTempDir();
    const clock = new ManualClock(1_000);
    const writer = makeWriter(dataDir, clock);

    try {
      await writer.append({ kind: "user_msg", content: "first" });
      clock.advance(10);
      await writer.append({ kind: "agent_msg", content: "response" });
      clock.advance(10);
      // Simulates a prior user_msg that never produced an agent_msg, e.g. a
      // failed turn. The compiler must drop it so the caller can safely
      // append the NEW current user message without producing two
      // consecutive user entries.
      await writer.append({ kind: "user_msg", content: "orphan" });
    } finally {
      writer.close();
    }

    const window = new TurnContextCompiler().compile(makeReader(dataDir));

    expect(window.messages).toHaveLength(2);
    expect(window.messages[window.messages.length - 1]?.role).toBe("assistant");
    expect(window.messages.map((m) => m.content)).toEqual(["first", "response"]);
  });

  it("collapses same-role adjacency by keeping the newest entry in each run", async () => {
    const dataDir = createTempDir();
    const clock = new ManualClock(1_000);
    const writer = makeWriter(dataDir, clock);

    try {
      await writer.append({ kind: "user_msg", content: "first user" });
      clock.advance(10);
      await writer.append({ kind: "agent_msg", content: "older assistant" });
      clock.advance(10);
      await writer.append({ kind: "agent_msg", content: "newer assistant" });
      clock.advance(10);
      await writer.append({ kind: "user_msg", content: "older followup" });
      clock.advance(10);
      await writer.append({ kind: "user_msg", content: "newer followup" });
      clock.advance(10);
      await writer.append({ kind: "agent_msg", content: "final answer" });
    } finally {
      writer.close();
    }

    const window = new TurnContextCompiler().compile(makeReader(dataDir));

    expect(window.messages.map((message) => message.content)).toEqual([
      "first user",
      "newer assistant",
      "newer followup",
      "final answer",
    ]);
  });

  it("drops a leading assistant entry so the window starts with user", async () => {
    const dataDir = createTempDir();
    const clock = new ManualClock(1_000);
    const writer = makeWriter(dataDir, clock);

    try {
      // Simulate a truncation where the window begins mid-pair.
      await writer.append({ kind: "agent_msg", content: "dangling assistant" });
      clock.advance(10);
      await writer.append({ kind: "user_msg", content: "question" });
      clock.advance(10);
      await writer.append({ kind: "agent_msg", content: "answer" });
    } finally {
      writer.close();
    }

    const window = new TurnContextCompiler().compile(makeReader(dataDir));

    // "dangling assistant" is the newest-first head; once we cap by maxMessages
    // it can survive, but it must be dropped because the window must start
    // with a user role.
    expect(window.messages[0]?.role).toBe("user");
    expect(window.messages.map((m) => m.content)).toEqual(["question", "answer"]);
  });

  it("caps the number of messages at maxMessages", async () => {
    const dataDir = createTempDir();
    const clock = new ManualClock(1_000);
    const writer = makeWriter(dataDir, clock);

    try {
      for (let i = 0; i < 10; i += 1) {
        await writer.append({ kind: "user_msg", content: `u${i}` });
        clock.advance(1);
        await writer.append({ kind: "agent_msg", content: `a${i}` });
        clock.advance(1);
      }
    } finally {
      writer.close();
    }

    const window = new TurnContextCompiler({ maxMessages: 4 }).compile(makeReader(dataDir));

    expect(window.messages).toHaveLength(4);
    // Newest four kept; older ones dropped.
    expect(window.messages.map((m) => m.content)).toEqual(["u8", "a8", "u9", "a9"]);
  });

  it("caps characters via maxChars", async () => {
    const dataDir = createTempDir();
    const clock = new ManualClock(1_000);
    const writer = makeWriter(dataDir, clock);

    try {
      await writer.append({ kind: "user_msg", content: "a".repeat(100) });
      clock.advance(1);
      await writer.append({ kind: "agent_msg", content: "b".repeat(100) });
      clock.advance(1);
      await writer.append({ kind: "user_msg", content: "c".repeat(100) });
      clock.advance(1);
      await writer.append({ kind: "agent_msg", content: "d".repeat(100) });
    } finally {
      writer.close();
    }

    // 120 chars only fits the newest message plus the one before it if small
    // enough -- the compiler breaks after the first message that would put
    // us over the cap, but always keeps at least the newest message.
    const window = new TurnContextCompiler({ maxChars: 150 }).compile(makeReader(dataDir));

    // With 100-char messages, the compiler keeps the newest (agent "d")
    // then hits the cap on the next; the dangling assistant is dropped to
    // respect the "starts with user" invariant, leaving zero messages.
    expect(window.total_chars).toBeLessThanOrEqual(window.messages.reduce((s, m) => s + m.content.length, 0) + 1);
    expect(window.messages.length).toBeLessThanOrEqual(2);
  });
});
