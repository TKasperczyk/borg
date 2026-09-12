import { describe, expect, it } from "vitest";

import { ConfigError, describeErrorWithCauses, findInErrorCauseChain } from "./errors.js";

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

  it("describes a cause chain outermost first, including non-error causes", () => {
    expect(describeErrorWithCauses(new Error("plain"))).toBe("Error: plain");
    expect(describeErrorWithCauses("not an error")).toBe("not an error");

    // The wrapper's message is a per-stage constant; the cause is the only part
    // that distinguishes one failure of that stage from another.
    expect(
      describeErrorWithCauses(new ConfigError("could not load config", { cause: "Unknown tool" })),
    ).toBe("ConfigError: could not load config; caused by Unknown tool");

    expect(
      describeErrorWithCauses(
        new ConfigError("outer", { cause: new Error("middle", { cause: new Error("inner") }) }),
      ),
    ).toBe("ConfigError: outer; caused by Error: middle; caused by Error: inner");
  });

  it("bounds the described cause chain and terminates on cause cycles", () => {
    const deep = [5, 4, 3, 2, 1].reduce<unknown>(
      (cause, depth) => new Error(`depth ${depth}`, { cause }),
      undefined,
    );

    expect(describeErrorWithCauses(deep)).toBe(
      "Error: depth 1; caused by Error: depth 2; caused by Error: depth 3; caused by Error: depth 4",
    );

    const cycle = new Error("cycle");
    Object.defineProperty(cycle, "cause", { value: cycle });
    expect(describeErrorWithCauses(cycle)).toBe("Error: cycle");
  });
});
