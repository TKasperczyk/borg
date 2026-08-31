import { mkdtempSync, mkdirSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { writePrivateContextCaptureJson } from "../src/cognition/deliberation/context-capture-storage.js";
import { appendDurableJsonl } from "../src/util/durable-jsonl.js";
import { parsePlannerAbJudgeArgs, resolvePlannerAbJudgePaths } from "./planner-ab-judge.js";

describe("planner A/B judge filesystem boundary", () => {
  const temporaryDirectories: string[] = [];

  function temp(name: string): string {
    const directory = mkdtempSync(join(tmpdir(), name));
    temporaryDirectories.push(directory);
    return directory;
  }

  afterEach(() => {
    for (const directory of temporaryDirectories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("parses a reduced cohort limit and explicit eval inputs", () => {
    expect(
      parsePlannerAbJudgeArgs([
        "--input",
        "/eval/results.jsonl",
        "--captures",
        "/eval/captures.jsonl",
        "--limit",
        "13",
      ]),
    ).toEqual({
      inputPath: "/eval/results.jsonl",
      capturesPath: "/eval/captures.jsonl",
      limit: 13,
    });
  });

  it("defaults both private outputs directly below dataDir/captures", () => {
    const dataDir = temp("borg-judge-data-");
    const external = temp("borg-judge-input-");
    const results = join(external, "results.jsonl");
    const captures = join(external, "contexts.jsonl");
    writeFileSync(results, "");
    writeFileSync(captures, "");

    const paths = resolvePlannerAbJudgePaths({
      dataDir,
      inputPath: results,
      capturesPath: captures,
    });

    expect(paths.outputPath).toBe(join(dataDir, "captures", "planner-ab-judgments.jsonl"));
    expect(paths.summaryPath).toBe(join(dataDir, "captures", "planner-ab-judgment-summary.json"));
  });

  it("resolves symlinks before rejecting an output outside captures", () => {
    const dataDir = temp("borg-judge-contained-");
    const external = temp("borg-judge-external-");
    const results = join(external, "results.jsonl");
    const captures = join(external, "contexts.jsonl");
    const privateElsewhere = join(dataDir, "stream", "judgments.jsonl");
    const alias = join(external, "judgments-link.jsonl");
    mkdirSync(join(dataDir, "stream"));
    writeFileSync(results, "");
    writeFileSync(captures, "");
    writeFileSync(privateElsewhere, "");
    symlinkSync(privateElsewhere, alias);

    expect(() =>
      resolvePlannerAbJudgePaths({
        dataDir,
        inputPath: results,
        capturesPath: captures,
        outputPath: alias,
      }),
    ).toThrow("must be a direct child of dataDir/captures");
  });

  it("creates and repairs judgment/summary privacy under a 0022 umask", async () => {
    const dataDir = temp("borg-judge-private-");
    const capturesDirectory = join(dataDir, "captures");
    const judgmentsPath = join(capturesDirectory, "planner-ab-judgments.jsonl");
    const previousUmask = process.umask(0o022);
    try {
      mkdirSync(capturesDirectory, { mode: 0o777 });
      writeFileSync(judgmentsPath, "", { mode: 0o666 });
      writeFileSync(join(capturesDirectory, "planner-ab-judgment-summary.json"), "{}\n", {
        mode: 0o666,
      });
      await appendDurableJsonl(
        judgmentsPath,
        { status: "completed" },
        {
          privateDirectory: capturesDirectory,
        },
      );
      const summaryPath = writePrivateContextCaptureJson({
        dataDir,
        fileName: "planner-ab-judgment-summary.json",
        value: { completed: 1 },
      });

      expect(statSync(capturesDirectory).mode & 0o777).toBe(0o700);
      expect(statSync(judgmentsPath).mode & 0o777).toBe(0o600);
      expect(statSync(summaryPath).mode & 0o777).toBe(0o600);
    } finally {
      process.umask(previousUmask);
    }
  });
});
