import { execFileSync } from "node:child_process";

import { describe, expect, it } from "vitest";

function runAssess(args: readonly string[]): string {
  return execFileSync("pnpm", ["tsx", "assessor/cli.ts", ...args], {
    cwd: process.cwd(),
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
}

describe("assessor CLI", () => {
  it("exits non-zero for failing scenarios unless --no-fail is passed", () => {
    let failure: unknown;

    try {
      runAssess(["--mock", "--scenario", "failing-mock-fixture"]);
    } catch (error) {
      failure = error;
    }

    expect((failure as { status?: number }).status).toBe(1);
    expect(String((failure as { stdout?: string }).stdout)).toContain("Fail: 1");

    const stdout = runAssess(["--mock", "--scenario", "failing-mock-fixture", "--no-fail"]);

    expect(stdout).toContain("Fail: 1");
  });
});
