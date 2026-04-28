import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  groupTraceByPhase,
  groupTraceByTurn,
  phaseForTraceEvent,
  readTraceEvents,
  summarizeTraceFile,
} from "./trace-reader.js";

describe("trace-reader", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("reads JSONL, groups by turn and phase, and summarizes compactly", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-assessor-trace-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "trace.jsonl");

    writeFileSync(
      tracePath,
      [
        JSON.stringify({
          ts: 1,
          turnId: "turn-a",
          event: "perception_started",
          prompt: "large prompt",
        }),
        JSON.stringify({
          ts: 2,
          turnId: "turn-a",
          event: "tool_call_dispatched",
          toolName: "tool.episodic.search",
        }),
        JSON.stringify({
          ts: 3,
          turnId: "turn-b",
          event: "retrieval_completed",
          episodeCount: 2,
        }),
        "",
      ].join("\n"),
    );

    const records = readTraceEvents(tracePath);

    expect(records).toHaveLength(3);
    expect(groupTraceByTurn(records).get("turn-a")).toHaveLength(2);
    expect(groupTraceByPhase(records).get("action")?.[0]?.event).toBe("tool_call_dispatched");
    expect(phaseForTraceEvent("retrieval_completed")).toBe("retrieval");
    expect(summarizeTraceFile(tracePath, "turn-a")).toContain("tool.episodic.search");
    expect(summarizeTraceFile(tracePath, "turn-a")).toContain("prompt=[collapsed]");
  });

  it("skips malformed lines by default and throws in strict mode", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-assessor-trace-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "partial.trace.jsonl");

    writeFileSync(
      tracePath,
      [
        JSON.stringify({
          ts: 1,
          turnId: "turn-a",
          event: "perception_started",
        }),
        '{"ts":2,"turnId":',
        "",
      ].join("\n"),
    );

    expect(readTraceEvents(tracePath)).toHaveLength(1);
    expect(summarizeTraceFile(tracePath, "turn-a")).toContain("trace warnings: 1");
    expect(() => readTraceEvents(tracePath, { strict: true })).toThrow("Invalid JSON");
  });
});
