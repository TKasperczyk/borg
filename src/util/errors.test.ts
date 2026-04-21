import { describe, expect, it } from "vitest";

import { ConfigError } from "./errors.js";

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
});
