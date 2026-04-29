import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { runSimulation } from "./runner.js";
import { tomPersona } from "./personas/tom.js";

const tempDirs: string[] = [];

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "borg-simulator-runner-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

describe("SimulatorRunner", () => {
  it("runs a 20-turn mock simulation with probes, overseer checkpoints, and metrics", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const report = await runSimulation({
      runId: "sim-runner-test",
      persona: tomPersona,
      totalTurns: 20,
      probeEvery: 5,
      checkEvery: 10,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
      probeRunner: async ({ scenarioName, transport, turnNumber }) => {
        const result = await transport.chat("What's my dog's name?");

        return {
          turn: turnNumber,
          scenarioName,
          passed: true,
          evidence: `Mock probe ${scenarioName} passed at ${result.turnId}.`,
          response: result.response,
          turnId: result.turnId,
        };
      },
      overseerRunner: async ({ turnCounter }) => ({
        ts: Date.now(),
        turn_counter: turnCounter,
        status: "healthy",
        observations: ["Mock overseer saw no degradation."],
        recommendation: "Continue.",
      }),
    });
    const metricsRows = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line) as { turn_counter: number });

    expect(report.totalTurns).toBe(20);
    expect(report.probes).toHaveLength(3);
    expect(report.probes.every((probe) => probe.passed)).toBe(true);
    expect(report.overseerCheckpoints).toHaveLength(2);
    expect(metricsRows).toHaveLength(20);
    expect(metricsRows.at(-1)?.turn_counter).toBe(20);
  });
});
