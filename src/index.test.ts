import { describe, it, expect } from "vitest";
import { VERSION } from "./index.js";

describe("borg library entry", () => {
  it("exports a semver version string", () => {
    expect(VERSION).toMatch(/^\d+\.\d+\.\d+/);
  });
});
