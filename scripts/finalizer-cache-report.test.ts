import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  analyzeFinalizerCaptureFile,
  commonUtf8PrefixBytes,
  extractTopLevelTaggedSections,
  type FinalizerCachePairReport,
} from "./finalizer-cache-report.js";

function capture(captureId: string, turnOrigin: "user" | "autonomous", blocks: string[]) {
  return {
    capture_id: captureId,
    turn_origin: turnOrigin,
    surfaces: {
      compact: {
        system: blocks.map((text) => ({ type: "text", text })),
      },
    },
  };
}

describe("finalizer cache report", () => {
  const tempDirectories: string[] = [];

  afterEach(() => {
    for (const directory of tempDirectories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("measures UTF-8 prefix bytes rather than UTF-16 string positions", () => {
    expect(commonUtf8PrefixBytes("éx", "éy")).toBe(2);
    expect(commonUtf8PrefixBytes("🙂x", "🙂y")).toBe(4);
  });

  it("extracts only top-level machine-tagged sections with exact text", () => {
    const block = [
      "preamble",
      '<borg_availability state="available" />',
      "",
      "<borg_outer>",
      "  <nested>value</nested>",
      "</borg_outer>",
    ].join("\n");

    expect(extractTopLevelTaggedSections(block, 3)).toEqual([
      {
        blockIndex: 3,
        tag: "borg_availability",
        ordinal: 1,
        text: '<borg_availability state="available" />',
      },
      {
        blockIndex: 3,
        tag: "borg_outer",
        ordinal: 1,
        text: "<borg_outer>\n  <nested>value</nested>\n</borg_outer>",
      },
    ]);
  });

  it("streams consecutive autonomous pairs across intervening user captures", async () => {
    const directory = mkdtempSync(join(tmpdir(), "borg-finalizer-cache-report-"));
    tempDirectories.push(directory);
    const path = join(directory, "finalizer-contexts.jsonl");
    const stable = "<borg_stable>\nunchanged\n</borg_stable>";
    const firstBlocks = ["éx", `${stable}\n\n<borg_changing>\none\n</borg_changing>`];
    const secondBlocks = ["éy", `${stable}\n\n<borg_changing>\ntwo\n</borg_changing>`];
    const thirdBlocks = ["éy", `${stable}\n\n<borg_added>\nthree\n</borg_added>`];
    writeFileSync(
      path,
      [
        capture("autonomous-1", "autonomous", firstBlocks),
        capture("user-between", "user", ["ignored"]),
        capture("autonomous-2", "autonomous", secondBlocks),
        capture("autonomous-3", "autonomous", thirdBlocks),
      ]
        .map((record) => JSON.stringify(record))
        .join("\n") + "\n",
    );
    const reports: FinalizerCachePairReport[] = [];

    const summary = await analyzeFinalizerCaptureFile(path, (report) => reports.push(report));

    expect(summary).toEqual({ autonomousCaptures: 3, consecutivePairs: 2 });
    expect(
      reports.map((report) => [report.previous_capture_id, report.current_capture_id]),
    ).toEqual([
      ["autonomous-1", "autonomous-2"],
      ["autonomous-2", "autonomous-3"],
    ]);
    expect(reports[0]?.blocks).toEqual([
      { block_index: 0, common_prefix_bytes: 2, previous_chars: 2, current_chars: 2 },
      {
        block_index: 1,
        common_prefix_bytes: Buffer.byteLength(`${stable}\n\n<borg_changing>\n`),
        previous_chars: firstBlocks[1]!.length,
        current_chars: secondBlocks[1]!.length,
      },
    ]);
    expect(reports[0]?.sections).toEqual([
      {
        block_index: 1,
        tag: "borg_stable",
        ordinal: 1,
        byte_stable: true,
        previous_chars: stable.length,
        current_chars: stable.length,
      },
      expect.objectContaining({
        block_index: 1,
        tag: "borg_changing",
        byte_stable: false,
      }),
    ]);
    expect(reports[1]?.sections).toContainEqual(
      expect.objectContaining({
        tag: "borg_changing",
        byte_stable: false,
        current_chars: null,
      }),
    );
    expect(reports[1]?.sections).toContainEqual(
      expect.objectContaining({
        tag: "borg_added",
        byte_stable: false,
        previous_chars: null,
      }),
    );
  });

  it("compares user turns within their sessions without crossing intervening sessions", async () => {
    const directory = mkdtempSync(join(tmpdir(), "borg-finalizer-cache-report-"));
    tempDirectories.push(directory);
    const path = join(directory, "finalizer-contexts.jsonl");
    const records = [
      { ...capture("a1", "user", ["stable", "one"]), session_id: "a" },
      { ...capture("b1", "user", ["different", "two"]), session_id: "b" },
      { ...capture("a2", "user", ["stable", "three"]), session_id: "a" },
    ];
    writeFileSync(path, records.map((record) => JSON.stringify(record)).join("\n") + "\n");
    const reports: FinalizerCachePairReport[] = [];
    const summary = await analyzeFinalizerCaptureFile(path, (report) => reports.push(report), {
      sameSession: true,
    });
    expect(summary).toEqual({ autonomousCaptures: 0, consecutivePairs: 1 });
    expect(reports[0]).toMatchObject({ previous_capture_id: "a1", current_capture_id: "a2" });
    expect(reports[0]?.blocks[0]?.common_prefix_bytes).toBe(Buffer.byteLength("stable"));
  });
});
