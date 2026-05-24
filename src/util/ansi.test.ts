import { describe, expect, it } from "vitest";

import { createAnsi } from "./ansi.js";

describe("createAnsi", () => {
  it("emits plain text for non-TTY output", () => {
    const ansi = createAnsi({ isTTY: false });
    expect(ansi.red("error")).toBe("error");
  });

  it("wraps styled text for TTY output", () => {
    const ansi = createAnsi({ isTTY: true });
    expect(ansi.green("ok")).toBe("\u001b[32mok\u001b[0m");
  });
});
