import { describe, expect, it } from "vitest";

import { parseAbJudgeCliArgs, parseAbReplayCliArgs } from "./ab-cli.js";

describe("shared A/B CLI parsing", () => {
  it("parses replay flags for both planner and finalizer callers", () => {
    expect(
      parseAbReplayCliArgs(
        ["--live", "--data-dir", "/data", "--input", "/in", "--output", "/out", "--limit", "3"],
        { command: "finalizer:ab-replay", includeNonCompleted: false },
      ),
    ).toEqual({
      mode: "live",
      includeNonCompleted: false,
      dataDir: "/data",
      inputPath: "/in",
      outputPath: "/out",
      limit: 3,
    });
    expect(
      parseAbReplayCliArgs(["--include-non-completed"], {
        command: "planner:ab-replay",
        includeNonCompleted: true,
      }),
    ).toMatchObject({ mode: "dry", includeNonCompleted: true });
  });

  it("parses the shared judge path flags", () => {
    expect(
      parseAbJudgeCliArgs(
        [
          "--data-dir",
          "/data",
          "--input",
          "/in",
          "--captures",
          "/captures",
          "--output",
          "/out",
          "--summary",
          "/summary",
          "--limit",
          "4",
        ],
        "finalizer:ab-judge",
      ),
    ).toEqual({
      dataDir: "/data",
      inputPath: "/in",
      capturesPath: "/captures",
      outputPath: "/out",
      summaryPath: "/summary",
      limit: 4,
    });
  });
});
