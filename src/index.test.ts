import { describe, it, expect } from "vitest";
import {
  COGNITIVE_MODES,
  DEFAULT_SESSION_ID,
  FakeLLMClient,
  VERSION,
  createTurnTracer,
  loadConfig,
  parseSessionId,
} from "./index.js";

describe("borg library entry", () => {
  it("exports a semver version string", () => {
    expect(VERSION).toMatch(/^\d+\.\d+\.\d+/);
  });

  it("re-exports key foundation APIs", () => {
    expect(DEFAULT_SESSION_ID).toBe("default");
    expect(typeof loadConfig).toBe("function");
    expect(typeof createTurnTracer).toBe("function");
    expect(COGNITIVE_MODES).toContain("problem_solving");
    expect(parseSessionId("default")).toBe(DEFAULT_SESSION_ID);
    expect(new FakeLLMClient()).toBeInstanceOf(FakeLLMClient);
  });
});
