import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../src/index.js";
import { MaintenanceScheduler, type MaintenanceTickResult } from "../src/offline/scheduler.js";
import { runSimulation } from "./runner.js";
import { tomPersona } from "./personas/tom.js";

const tempDirs: string[] = [];

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "borg-simulator-runner-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  vi.restoreAllMocks();

  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

function spyMaintenanceTick() {
  return vi
    .spyOn(MaintenanceScheduler.prototype, "tick")
    .mockImplementation(async (cadence): Promise<MaintenanceTickResult> => {
      return {
        status: "ok",
        cadence,
        ts: Date.now(),
        processes: [],
        result: null,
      };
    });
}

describe("SimulatorRunner", () => {
  it("runs a 20-turn mock simulation with overseer checkpoints and metrics", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    spyMaintenanceTick();
    const report = await runSimulation({
      runId: "sim-runner-test",
      persona: tomPersona,
      totalTurns: 20,
      checkEvery: 10,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
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
    expect(Object.hasOwn(report, "probes")).toBe(false);
    expect(report.overseerCheckpoints).toHaveLength(2);
    expect(metricsRows).toHaveLength(20);
    expect(metricsRows.at(-1)?.turn_counter).toBe(20);
  });

  it("runs periodic maintenance ticks on cadence in mock mode", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const tickSpy = spyMaintenanceTick();

    await runSimulation({
      runId: "sim-runner-maintenance-test",
      persona: tomPersona,
      totalTurns: 20,
      checkEvery: 999,
      maintenanceEvery: 10,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(tickSpy).toHaveBeenCalledTimes(2);
    expect(tickSpy.mock.calls.map(([cadence]) => cadence)).toEqual(["light", "light"]);
  });

  it("stops self-play when Borg suppresses a turn", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    spyMaintenanceTick();
    const report = await runSimulation({
      runId: "sim-runner-suppression-test",
      persona: tomPersona,
      totalTurns: 5,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "toolu_plan",
                name: "EmitTurnPlan",
                input: {
                  uncertainty: "",
                  verification_steps: [],
                  tensions: [],
                  voice_note: "",
                  referenced_episode_ids: [],
                  intents: [],
                },
              },
            ],
          },
          {
            text: "Human: Done.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          {
            text: "Assistant: Still invalid.",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
    });
    const metricsRows = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line) as { turn_counter: number });

    expect(report.resultState).toBe("stopped_by_suppression");
    expect(report.stoppedTurn).toBe(1);
    expect(metricsRows).toHaveLength(1);
    expect(metricsRows[0]?.turn_counter).toBe(1);
  });
});
