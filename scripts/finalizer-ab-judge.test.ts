import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { resolveFinalizerAbJudgePaths } from "./finalizer-ab-judge.js";

describe("finalizer A/B judge filesystem boundary", () => {
  const temporaryDirectories: string[] = [];

  afterEach(() => {
    for (const directory of temporaryDirectories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("defaults private outputs beside the finalizer capture and replay cohort", () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-judge-"));
    temporaryDirectories.push(dataDir);
    const captures = join(dataDir, "captures");
    mkdirSync(captures);
    writeFileSync(join(captures, "finalizer-ab-results.jsonl"), "");
    writeFileSync(join(captures, "finalizer-contexts.jsonl"), "");

    const paths = resolveFinalizerAbJudgePaths({ dataDir });

    expect(paths.inputPath).toBe(join(captures, "finalizer-ab-results.jsonl"));
    expect(paths.sourceCapturesPath).toBe(join(captures, "finalizer-contexts.jsonl"));
    expect(paths.outputPath).toBe(join(captures, "finalizer-ab-judgments.jsonl"));
    expect(paths.summaryPath).toBe(join(captures, "finalizer-ab-judgment-summary.json"));
  });
});
