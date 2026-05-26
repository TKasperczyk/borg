import { readFileSync } from "node:fs";

import { describe, it, expect } from "vitest";
import {
  Borg,
  COGNITIVE_MODES,
  DEFAULT_SESSION_ID,
  VERSION,
  loadConfig,
  parseSessionId,
} from "./index.js";

describe("borg library entry", () => {
  it("exports a semver version string", () => {
    expect(VERSION).toMatch(/^\d+\.\d+\.\d+/);
  });

  it("re-exports key foundation APIs", () => {
    expect(typeof Borg.open).toBe("function");
    expect(DEFAULT_SESSION_ID).toBe("default");
    expect(typeof loadConfig).toBe("function");
    expect(COGNITIVE_MODES).toContain("problem_solving");
    expect(parseSessionId("default")).toBe(DEFAULT_SESSION_ID);
  });

  it("keeps published root exports pointed at dist artifacts", () => {
    const packageJson = JSON.parse(
      readFileSync(new URL("../package.json", import.meta.url), "utf8"),
    ) as {
      exports?: {
        "."?: Record<string, string>;
      };
      files?: string[];
    };

    expect(packageJson.exports?.["."]?.development).toBeUndefined();
    expect(packageJson.exports?.["."]?.import).toBe("./dist/index.js");
    expect(packageJson.exports?.["."]?.types).toBe("./dist/index.d.ts");
    expect(packageJson.files).not.toContain("src");
  });
});
