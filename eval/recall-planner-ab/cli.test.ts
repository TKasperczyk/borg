import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import {
  MAX_RECALL_EXPANSION_SEMANTIC_VARIANTS,
  MIN_RECALL_EXPANSION_SEMANTIC_VARIANTS,
} from "../../src/retrieval/recall-expansion.js";

import { parseCliArgs } from "./cli.js";

describe("recall planner A/B CLI", () => {
  it("parses required inputs, de-duplicates variant counts, and supports judge auto-selection", () => {
    expect(
      parseCliArgs([
        "--data-dir",
        "bank",
        "--cases",
        "cases.json",
        "--out",
        "scratch",
        "--variant-counts",
        "1,3,1",
        "--baseline",
        "--judge-model",
        "--generate-cases",
        "6",
      ]),
    ).toEqual({
      help: false,
      dataDir: resolve("bank"),
      casesPath: resolve("cases.json"),
      outDir: resolve("scratch"),
      variantCounts: [1, 3],
      judgeRequested: true,
      baseline: true,
      generateCases: 6,
    });
  });

  it.each([
    String(MIN_RECALL_EXPANSION_SEMANTIC_VARIANTS - 1),
    String(MAX_RECALL_EXPANSION_SEMANTIC_VARIANTS + 1),
    "1.5",
    "one",
  ])("rejects unsupported semantic variant count %s", (count) => {
    expect(() =>
      parseCliArgs([
        "--data-dir",
        "bank",
        "--cases",
        "cases.json",
        "--out",
        "scratch",
        "--variant-counts",
        count,
      ]),
    ).toThrow(/--variant-counts/);
  });

  it("requires every primary path and variant counts", () => {
    expect(() => parseCliArgs([])).toThrow(/--data-dir/);
    expect(() => parseCliArgs(["--data-dir", "bank"])).toThrow(/--cases/);
  });
});
