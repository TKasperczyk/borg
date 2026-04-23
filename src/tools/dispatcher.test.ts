import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { StreamReader, StreamWriter } from "../stream/index.js";
import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";

import { ToolDispatcher } from "./dispatcher.js";

describe("ToolDispatcher", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
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
      async invoke(input) {
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
      async invoke(input) {
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
      async invoke(input) {
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
