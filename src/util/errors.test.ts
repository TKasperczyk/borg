import { describe, expect, it } from "vitest";

import { ConfigError, findInErrorCauseChain } from "./errors.js";

describe("errors", () => {
  it("serializes borg errors with codes and causes", () => {
    const error = new ConfigError("invalid config", {
      cause: new Error("missing field"),
    });

    expect(error.code).toBe("BORG_CONFIG_ERROR");
    expect(error.toJSON()).toEqual({
      name: "ConfigError",
      code: "BORG_CONFIG_ERROR",
      message: "invalid config",
      cause: {
        name: "Error",
        message: "missing field",
      },
    });
  });

  it("finds a matching nested cause and terminates on cause cycles", () => {
    const target = Object.assign(new Error("target"), { status: 400 });
    const wrapped = new Error("wrapped", { cause: target });

    expect(
      findInErrorCauseChain(
        wrapped,
        (candidate): candidate is Error & { status: number } =>
          candidate instanceof Error && "status" in candidate,
      ),
    ).toBe(target);

    const cycle = new Error("cycle");
    Object.defineProperty(cycle, "cause", { value: cycle });
    expect(
      findInErrorCauseChain(cycle, (candidate): candidate is Date => candidate instanceof Date),
    ).toBeUndefined();
  });
});
