import { afterEach, describe, expect, it, vi } from "vitest";

import { copyText } from "./clipboard";

function installExecCommand(result = true) {
  const execCommand = vi.fn(() => result);
  Object.defineProperty(document, "execCommand", {
    configurable: true,
    value: execCommand,
  });
  return execCommand;
}

afterEach(() => {
  Reflect.deleteProperty(window.navigator, "clipboard");
  Reflect.deleteProperty(document, "execCommand");
  document.body.innerHTML = "";
  vi.restoreAllMocks();
});

describe("copyText", () => {
  it("uses the execCommand fallback when the clipboard API is unavailable", async () => {
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: undefined,
    });
    const execCommand = installExecCommand();

    await copyText("fallback text");

    expect(execCommand).toHaveBeenCalledWith("copy");
    expect(document.querySelector("textarea")).toBeNull();
  });

  it("falls back to execCommand when clipboard.writeText rejects", async () => {
    const writeText = vi.fn<Clipboard["writeText"]>().mockRejectedValue(new Error("denied"));
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    const execCommand = installExecCommand();

    await copyText("fallback after rejection");

    expect(writeText).toHaveBeenCalledWith("fallback after rejection");
    expect(execCommand).toHaveBeenCalledWith("copy");
  });
});
