import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import {
  ABORTED_TURN_EVENT,
  StreamReader,
  StreamWriter,
  filterActiveStreamEntries,
} from "../stream/index.js";
import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";

import { ToolDispatcher } from "./dispatcher.js";

describe("ToolDispatcher", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("writes tool_call and tool_result entries for successful dispatches", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000);
    const dispatcher = new ToolDispatcher({
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: tempDir,
          sessionId,
          clock,
        }),
    });

    dispatcher.register({
      name: "tool.test.echo",
      description: "Echo test input.",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "read",
      inputSchema: z.object({
        value: z.string().min(1),
      }),
      outputSchema: z.object({
        echoed: z.string().min(1),
      }),
      async invoke(input: { value: string }) {
        return {
          echoed: input.value,
        };
      },
    });

    const result = await dispatcher.dispatch({
      toolName: "tool.test.echo",
      input: {
        value: "hello",
      },
      origin: "autonomous",
    });

    expect(result).toMatchObject({
      toolName: "tool.test.echo",
      ok: true,
      output: {
        echoed: "hello",
      },
    });

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(2);
    expect(entries.map((entry) => entry.kind)).toEqual(["tool_call", "tool_result"]);
    expect(entries[0]?.content).toMatchObject({
      tool_name: "tool.test.echo",
      origin: "autonomous",
    });
    expect(entries[1]?.content).toMatchObject({
      ok: true,
      output: {
        echoed: "hello",
      },
    });
  });

  it("marks tool stream entries with turn_id so aborted turns filter them out", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_500);
    const turnId = "turn-tool-aborted";
    const dispatcher = new ToolDispatcher({
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: tempDir,
          sessionId,
          clock,
        }),
    });

    dispatcher.register({
      name: "tool.test.echo",
      description: "Echo test input.",
      allowedOrigins: ["deliberator"],
      writeScope: "read",
      inputSchema: z.object({
        value: z.string().min(1),
      }),
      outputSchema: z.object({
        echoed: z.string().min(1),
      }),
      async invoke(input: { value: string }) {
        return {
          echoed: input.value,
        };
      },
    });

    await dispatcher.dispatch({
      toolName: "tool.test.echo",
      input: {
        value: "hello",
      },
      origin: "deliberator",
      turnId,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock,
    });
    await writer.append({
      kind: "internal_event",
      turn_id: turnId,
      turn_status: "aborted",
      content: {
        event: ABORTED_TURN_EVENT,
        turn_id: turnId,
        reason: "test abort",
      },
    });
    writer.close();

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(3);

    expect(
      entries.filter((entry) => entry.kind === "tool_call" || entry.kind === "tool_result"),
    ).toEqual([
      expect.objectContaining({ turn_id: turnId }),
      expect.objectContaining({ turn_id: turnId }),
    ]);
    expect(filterActiveStreamEntries(entries)).toEqual([]);
  });

  it("returns a failed tool_result when input validation fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(2_000);
    const dispatcher = new ToolDispatcher({
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: tempDir,
          sessionId,
          clock,
        }),
    });

    dispatcher.register({
      name: "tool.test.echo",
      description: "Echo test input.",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "read",
      inputSchema: z.object({
        value: z.string().min(1),
      }),
      outputSchema: z.object({
        echoed: z.string().min(1),
      }),
      async invoke(input: { value: string }) {
        return {
          echoed: input.value,
        };
      },
    });

    const result = await dispatcher.dispatch({
      toolName: "tool.test.echo",
      input: {
        value: "",
      },
      origin: "autonomous",
    });

    expect(result.ok).toBe(false);

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(2);
    expect(entries.map((entry) => entry.kind)).toEqual(["tool_call", "tool_result"]);
    expect(entries[1]?.content).toMatchObject({
      ok: false,
    });
  });

  it("clears the timeout timer after a successful dispatch", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(3_000);
    const clearTimeoutSpy = vi.spyOn(globalThis, "clearTimeout");
    const dispatcher = new ToolDispatcher({
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: tempDir,
          sessionId,
          clock,
        }),
    });

    dispatcher.register({
      name: "tool.test.fast",
      description: "Fast tool.",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "read",
      inputSchema: z.object({
        value: z.string().min(1),
      }),
      outputSchema: z.object({
        echoed: z.string().min(1),
      }),
      async invoke(input: { value: string }) {
        return {
          echoed: input.value,
        };
      },
    });

    const result = await dispatcher.dispatch({
      toolName: "tool.test.fast",
      input: {
        value: "ok",
      },
      origin: "autonomous",
      timeoutMs: 100,
    });

    expect(result.ok).toBe(true);
    expect(clearTimeoutSpy).toHaveBeenCalled();
  });

  it("times out read-scoped tools and clears the timeout timer", async () => {
    vi.useFakeTimers();

    try {
      const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
      tempDirs.push(tempDir);
      const clock = new ManualClock(3_500);
      const clearTimeoutSpy = vi.spyOn(globalThis, "clearTimeout");
      const dispatcher = new ToolDispatcher({
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: tempDir,
            sessionId,
            clock,
          }),
      });

      dispatcher.register({
        name: "tool.test.slow-read",
        description: "Slow read tool.",
        allowedOrigins: ["autonomous", "deliberator"],
        writeScope: "read",
        inputSchema: z.object({}).strict(),
        outputSchema: z.object({
          ok: z.literal(true),
        }),
        async invoke() {
          await new Promise(() => undefined);
          return { ok: true } as const;
        },
      });

      const resultPromise = dispatcher.dispatch({
        toolName: "tool.test.slow-read",
        input: {},
        origin: "autonomous",
        timeoutMs: 10,
      });

      await vi.advanceTimersByTimeAsync(10);
      const result = await resultPromise;

      expect(result).toMatchObject({
        ok: false,
        error: "Timed out after 10ms",
      });
      expect(clearTimeoutSpy).toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not apply dispatcher timeouts to write-scoped tools", async () => {
    vi.useFakeTimers();

    try {
      const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
      tempDirs.push(tempDir);
      const clock = new ManualClock(3_750);
      const dispatcher = new ToolDispatcher({
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: tempDir,
            sessionId,
            clock,
          }),
      });

      dispatcher.register({
        name: "tool.test.slow-write",
        description: "Slow write tool.",
        allowedOrigins: ["autonomous", "deliberator"],
        writeScope: "write",
        inputSchema: z.object({}).strict(),
        outputSchema: z.object({
          ok: z.literal(true),
        }),
        async invoke() {
          await new Promise<void>((resolve) => {
            setTimeout(resolve, 25);
          });
          return { ok: true } as const;
        },
      });

      const resultPromise = dispatcher.dispatch({
        toolName: "tool.test.slow-write",
        input: {},
        origin: "autonomous",
        timeoutMs: 1,
      });

      await vi.advanceTimersByTimeAsync(1);
      await vi.advanceTimersByTimeAsync(24);
      await expect(resultPromise).resolves.toMatchObject({
        ok: true,
        output: {
          ok: true,
        },
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it("filters listed tools by origin and rejects disallowed origins at dispatch time", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(4_000);
    const dispatcher = new ToolDispatcher({
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: tempDir,
          sessionId,
          clock,
        }),
    });

    dispatcher.register({
      name: "tool.test.deliberator-only",
      description: "Deliberator-only tool.",
      allowedOrigins: ["deliberator"],
      writeScope: "write",
      inputSchema: z.object({}).strict(),
      outputSchema: z.object({
        ok: z.literal(true),
      }),
      async invoke() {
        return { ok: true } as const;
      },
    });

    expect(dispatcher.getDefinition("tool.test.deliberator-only")).toMatchObject({
      name: "tool.test.deliberator-only",
      writeScope: "write",
    });
    expect(dispatcher.listTools("deliberator").map((tool) => tool.name)).toEqual([
      "tool.test.deliberator-only",
    ]);
    expect(dispatcher.listTools("autonomous")).toEqual([]);

    const result = await dispatcher.dispatch({
      toolName: "tool.test.deliberator-only",
      input: {},
      origin: "autonomous",
    });

    expect(result).toMatchObject({
      ok: false,
      error: "Tool tool.test.deliberator-only is not allowed for origin autonomous",
    });
  });
});
