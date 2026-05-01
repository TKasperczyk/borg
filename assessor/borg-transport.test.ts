import { afterEach, describe, expect, it, vi } from "vitest";

import { BorgTransport } from "./borg-transport.js";
import { recallScenario } from "./scenarios/recall.js";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("BorgTransport", () => {
  it("opens a mock Borg, runs a traced turn, and reads the trace summary", async () => {
    const transport = new BorgTransport({
      runId: "transport-test",
      scenario: recallScenario,
      mock: true,
    });

    try {
      await transport.open();
      const result = await transport.chat("What's my dog's name?");
      const summary = transport.readTrace(result.turnId);

      expect(result.response).toContain("Otto");
      expect(result.turnId).toHaveLength(36);
      expect(summary).toContain("tool.episodic.search");
      expect(transport.readTraceEvents().some((record) => record.turnId === result.turnId)).toBe(
        true,
      );
    } finally {
      await transport.close();
    }
  });

  it("keeps maintenance enabled when explicitly opted in", async () => {
    const disabledTransport = new BorgTransport({
      runId: "transport-maintenance-default-test",
      scenario: recallScenario,
      mock: true,
    });
    const enabledTransport = new BorgTransport({
      runId: "transport-maintenance-enabled-test",
      scenario: recallScenario,
      mock: true,
      maintenance: true,
    });

    try {
      await disabledTransport.open();
      await enabledTransport.open();

      expect(disabledTransport.getBorg().maintenance.scheduler.isEnabled()).toBe(false);
      expect(enabledTransport.getBorg().maintenance.scheduler.isEnabled()).toBe(true);
    } finally {
      await disabledTransport.close();
      await enabledTransport.close();
    }
  });

  it("passes an explicit audience through to Borg turns", async () => {
    const transport = new BorgTransport({
      runId: "transport-audience-test",
      scenario: recallScenario,
      mock: true,
    });

    try {
      await transport.open();
      const turnSpy = vi.spyOn(transport.getBorg(), "turn");

      await transport.chat("What's my dog's name?", { audience: "Tom" });

      expect(turnSpy.mock.calls[0]?.[0].audience).toBe("Tom");
    } finally {
      await transport.close();
    }
  });
});
