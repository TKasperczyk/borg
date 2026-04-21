import { describe, it, expect } from "vitest";
import { DEFAULT_SESSION_ID, VERSION, loadConfig } from "./index.js";

describe("borg library entry", () => {
  it("exports a semver version string", () => {
    expect(VERSION).toMatch(/^\d+\.\d+\.\d+/);
  });

  it("re-exports key foundation APIs", () => {
    expect(DEFAULT_SESSION_ID).toBe("default");
    expect(typeof loadConfig).toBe("function");
  });
});
