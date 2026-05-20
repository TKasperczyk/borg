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
          event: "perception.started",
          prompt: "large prompt",
        }),
        JSON.stringify({
          ts: 2,
          turnId: "turn-a",
          event: "tool_call.started",
          toolName: "tool.episodic.search",
        }),
        JSON.stringify({
          ts: 3,
          turnId: "turn-b",
          event: "retrieval.completed",
          episodeCount: 2,
        }),
        "",
      ].join("\n"),
    );

    const records = readTraceEvents(tracePath);

    expect(records).toHaveLength(3);
    expect(groupTraceByTurn(records).get("turn-a")).toHaveLength(2);
    expect(groupTraceByPhase(records).get("tools")?.[0]?.event).toBe("tool_call.started");
    expect(phaseForTraceEvent("retrieval.completed")).toBe("retrieval");
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
          event: "perception.started",
        }),
        '{"ts":2,"turnId":',
        "",
      ].join("\n"),
    );

    expect(readTraceEvents(tracePath)).toHaveLength(1);
    expect(summarizeTraceFile(tracePath, "turn-a")).toContain("trace warnings: 1");
    expect(() => readTraceEvents(tracePath, { strict: true })).toThrow("Invalid JSON");
  });

  it("routes normalized trace events through the shared taxonomy", () => {
    expect(phaseForTraceEvent("extraction.actions.completed")).toBe("extraction");
    expect(phaseForTraceEvent("extraction.commitments.transitioned")).toBe("extraction");
    expect(phaseForTraceEvent("review_resolver.completed")).toBe("review");
    expect(phaseForTraceEvent("turn.rejected")).toBe("session");
    expect(phaseForTraceEvent("frame_anomaly.completed")).toBe("perception");
    expect(phaseForTraceEvent("semantic_revision.completed")).toBe("retrieval");
    expect(phaseForTraceEvent("shared_state.compile.completed")).toBe("retrieval");
    expect(phaseForTraceEvent("shared_state.reconcile.completed")).toBe("retrieval");
  });
});
