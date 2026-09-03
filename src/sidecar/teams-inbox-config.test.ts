import { describe, expect, it } from "vitest";

import { teamsInboxConfigFromEnv } from "./teams-inbox-config.js";

describe("teamsInboxConfigFromEnv", () => {
  it("is disabled without TEAM_AGENT_BASE_URL", () => {
    expect(teamsInboxConfigFromEnv({})).toEqual({ enabled: false });
  });

  it("parses defaults and explicit settle values", () => {
    expect(
      teamsInboxConfigFromEnv({
        TEAM_AGENT_BASE_URL: "http://team-agent:8080",
        TEAM_AGENT_API_TOKEN: "secret",
        TEAMS_INBOX_SETTLE_MS: "10",
        TEAMS_INBOX_MAX_SETTLE_MS: "20",
      }),
    ).toEqual({
      enabled: true,
      baseUrl: "http://team-agent:8080",
      apiToken: "secret",
      timeoutMs: 120_000,
      settleMs: 10,
      maxSettleMs: 20,
      staleMs: 600_000,
    });
  });

  it("rejects an enabled configuration without a token", () => {
    expect(() =>
      teamsInboxConfigFromEnv({ TEAM_AGENT_BASE_URL: "http://team-agent:8080" }),
    ).toThrow("Invalid Team Agent inbox configuration");
  });
});
