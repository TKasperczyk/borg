import { homedir } from "node:os";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { expandPath } from "./path.js";

describe("expandPath", () => {
  it("expands the home directory marker", () => {
    expect(expandPath("~")).toBe(homedir());
    expect(expandPath("~/borg")).toBe(resolve(homedir(), "borg"));
  });

  it("resolves relative paths and preserves absolute paths", () => {
    expect(expandPath("relative/path")).toBe(resolve("relative/path"));
    expect(expandPath("/tmp/borg")).toBe("/tmp/borg");
  });
});
