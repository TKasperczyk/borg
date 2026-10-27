import { describe, expect, it } from "vitest";

import { observatoryConfigFromEnv } from "./config.js";

describe("observatoryConfigFromEnv", () => {
  it("ignores other settings when disabled", () => {
    expect(observatoryConfigFromEnv({ OBSERVATORY_FLUSH_MS: "bad" })).toEqual({ enabled: false });
    expect(observatoryConfigFromEnv({ OBSERVATORY_BASE_URL: " " })).toEqual({ enabled: false });
  });

  it("uses the contract defaults and accepts overrides", () => {
    const env = { OBSERVATORY_BASE_URL: "http://observatory:8080", OBSERVATORY_TOKEN: "secret" };
    expect(observatoryConfigFromEnv(env)).toEqual({
      enabled: true, baseUrl: env.OBSERVATORY_BASE_URL, token: "secret",
      flushMs: 250, maxBatch: 100, queueCap: 5000, timeoutMs: 2000,
    });
    expect(observatoryConfigFromEnv({ ...env, OBSERVATORY_MAX_BATCH: "50", OBSERVATORY_FLUSH_MS: "300" }))
      .toMatchObject({ maxBatch: 50, flushMs: 300 });
  });

  it.each([
    { OBSERVATORY_TOKEN: "" },
    { OBSERVATORY_MAX_BATCH: "501" },
    { OBSERVATORY_FLUSH_MS: "0" },
    { OBSERVATORY_TIMEOUT_MS: "2147483648" },
    { OBSERVATORY_QUEUE_CAP: "NaN" },
    { OBSERVATORY_BASE_URL: "file:///tmp/events" },
    { OBSERVATORY_BASE_URL: "https://user:secret@example.test" },
  ])("rejects invalid enabled settings: %o", (override) => {
    expect(() => observatoryConfigFromEnv({
      OBSERVATORY_BASE_URL: "https://observatory.example", OBSERVATORY_TOKEN: "secret", ...override,
    })).toThrow("Invalid observatory configuration");
  });
});
