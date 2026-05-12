import { describe, expect, it } from "vitest";

import { parseSimulatorCliOptions } from "./cli.js";
import { PIPELINE_C_DOUBLE_PRIME_INCOMPATIBLE_SHADOW_MESSAGE } from "./runner.js";

describe("simulator CLI", () => {
  it("parses --pipeline-c-double-prime", () => {
    const options = parseSimulatorCliOptions(["node", "simulate", "--pipeline-c-double-prime"]);

    expect(options.pipelineCDoublePrime).toBe(true);
  });

  it("parses flags after a pnpm-style -- separator", () => {
    const options = parseSimulatorCliOptions(["node", "cli.ts", "--", "--mock", "--turns", "1"]);

    expect(options.mock).toBe(true);
    expect(options.turns).toBe(1);
  });

  it("parses pipeline flags after a pnpm-style -- separator", () => {
    const options = parseSimulatorCliOptions([
      "node",
      "cli.ts",
      "--",
      "--real",
      "--pipeline-c-double-prime",
      "--turns",
      "70",
    ]);

    expect(options.real).toBe(true);
    expect(options.pipelineCDoublePrime).toBe(true);
    expect(options.turns).toBe(70);
  });

  it("parses output paths after a pnpm-style -- separator", () => {
    const options = parseSimulatorCliOptions([
      "node",
      "cli.ts",
      "--",
      "--metrics-out",
      "/tmp/x.jsonl",
      "--trace-out",
      "/tmp/y.jsonl",
    ]);

    expect(options.metricsOut).toBe("/tmp/x.jsonl");
    expect(options.traceOut).toBe("/tmp/y.jsonl");
  });

  it("parses --no-payloads", () => {
    const options = parseSimulatorCliOptions(["node", "simulate", "--no-payloads"]);

    expect(options.noPayloads).toBe(true);
  });

  it("parses comma-separated multi-persona keys", () => {
    const options = parseSimulatorCliOptions([
      "node",
      "simulate",
      "--personas",
      "alice-trip,ben-trip",
    ]);

    expect(options.personas).toBe("alice-trip,ben-trip");
  });

  it("rejects duplicate multi-persona keys", () => {
    expect(() =>
      parseSimulatorCliOptions([
        "node",
        "simulate",
        "--personas",
        "alice-trip,alice-trip,ben-trip",
      ]),
    ).toThrow("Duplicate persona key in --personas: alice-trip");
  });

  it("parses built-in simulator scenarios", () => {
    const options = parseSimulatorCliOptions(["node", "simulate", "--scenario", "trip-planning"]);

    expect(options.scenario).toBe("trip-planning");
  });

  it("preserves parsing without a -- separator", () => {
    const options = parseSimulatorCliOptions(["node", "cli.ts", "--mock", "--turns", "1"]);

    expect(options.mock).toBe(true);
    expect(options.turns).toBe(1);
  });

  it("merges flags before and after a pnpm-style -- separator", () => {
    const options = parseSimulatorCliOptions([
      "node",
      "cli.ts",
      "--mock",
      "--turns",
      "1",
      "--",
      "--check-every",
      "10",
      "--maintenance-every",
      "10",
      "--out",
      "./simulator-runs/v27-report.md",
      "--keep",
    ]);

    expect(options.mock).toBe(true);
    expect(options.turns).toBe(1);
    expect(options.checkEvery).toBe(10);
    expect(options.maintenanceEvery).toBe(10);
    expect(options.out).toBe("./simulator-runs/v27-report.md");
    expect(options.keep).toBe(true);
  });

  it("rejects --pipeline-c-double-prime with --shadow-post-gen-guards", () => {
    expect(() =>
      parseSimulatorCliOptions([
        "node",
        "simulate",
        "--pipeline-c-double-prime",
        "--shadow-post-gen-guards",
      ]),
    ).toThrow(PIPELINE_C_DOUBLE_PRIME_INCOMPATIBLE_SHADOW_MESSAGE);
  });

  it("rejects incompatible pipeline flags after a pnpm-style -- separator", () => {
    expect(() =>
      parseSimulatorCliOptions([
        "node",
        "cli.ts",
        "--",
        "--pipeline-c-double-prime",
        "--shadow-post-gen-guards",
      ]),
    ).toThrow(PIPELINE_C_DOUBLE_PRIME_INCOMPATIBLE_SHADOW_MESSAGE);
  });

  it("parses --shadow-post-gen-guards after a pnpm-style -- separator", () => {
    const options = parseSimulatorCliOptions(["node", "cli.ts", "--", "--shadow-post-gen-guards"]);

    expect(options.shadowPostGenGuards).toBe(true);
  });

  it("defaults --pipeline-c-double-prime to false", () => {
    const options = parseSimulatorCliOptions(["node", "simulate"]);

    expect(options.pipelineCDoublePrime === true).toBe(false);
  });
});
