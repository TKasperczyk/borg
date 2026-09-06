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
      taskEventsEnabled: false,
      baseUrl: "http://team-agent:8080",
      apiToken: "secret",
      timeoutMs: 120_000,
      settleMs: 10,
      maxSettleMs: 20,
      staleMs: 600_000,
    });
  });

  it("enables task events only with an explicit true flag", () => {
    const env = { TEAM_AGENT_BASE_URL: "http://team-agent:8080", TEAM_AGENT_API_TOKEN: "secret" };
    expect(
      teamsInboxConfigFromEnv({ ...env, TEAMS_INBOX_TASK_EVENTS_ENABLED: "true" }),
    ).toMatchObject({ enabled: true, taskEventsEnabled: true });
    expect(
      teamsInboxConfigFromEnv({ ...env, TEAMS_INBOX_TASK_EVENTS_ENABLED: "false" }),
    ).toMatchObject({ enabled: true, taskEventsEnabled: false });
    expect(() =>
      teamsInboxConfigFromEnv({ ...env, TEAMS_INBOX_TASK_EVENTS_ENABLED: "yes" }),
    ).toThrow("Invalid Team Agent inbox configuration");
  });

  it("rejects an enabled configuration without a token", () => {
    expect(() =>
      teamsInboxConfigFromEnv({ TEAM_AGENT_BASE_URL: "http://team-agent:8080" }),
    ).toThrow("Invalid Team Agent inbox configuration");
  });

  it("rejects a stale threshold that a batch retried after a runner timeout would reach", () => {
    expect(() =>
      teamsInboxConfigFromEnv({
        TEAM_AGENT_BASE_URL: "http://team-agent:8080",
        TEAM_AGENT_API_TOKEN: "secret",
        TEAM_AGENT_TIMEOUT_MS: "600000",
      }),
    ).toThrow("Invalid Team Agent inbox configuration");
  });

  it("accepts a longer runner timeout once the stale threshold clears it", () => {
    expect(
      teamsInboxConfigFromEnv({
        TEAM_AGENT_BASE_URL: "http://team-agent:8080",
        TEAM_AGENT_API_TOKEN: "secret",
        TEAM_AGENT_TIMEOUT_MS: "600000",
        TEAMS_INBOX_STALE_MS: "900000",
      }),
    ).toMatchObject({ enabled: true, timeoutMs: 600_000, staleMs: 900_000 });
  });
});
