import { z } from "zod";

import { ConfigError } from "../util/errors.js";

export type TeamsInboxConfig =
  | { enabled: false }
  | {
      enabled: true;
      baseUrl: string;
      apiToken: string;
      timeoutMs: number;
      settleMs: number;
      maxSettleMs: number;
      staleMs: number;
    };

const enabledTeamsInboxEnvSchema = z
  .object({
    TEAM_AGENT_BASE_URL: z
      .url()
      .refine((value) => value.startsWith("http://") || value.startsWith("https://")),
    TEAM_AGENT_API_TOKEN: z.string().trim().min(1),
    TEAM_AGENT_TIMEOUT_MS: z.coerce.number().int().positive().default(120_000),
    TEAMS_INBOX_SETTLE_MS: z.coerce.number().int().nonnegative().default(3_000),
    TEAMS_INBOX_MAX_SETTLE_MS: z.coerce.number().int().positive().default(15_000),
    TEAMS_INBOX_STALE_MS: z.coerce.number().int().positive().default(600_000),
  })
  .refine((value) => value.TEAMS_INBOX_MAX_SETTLE_MS >= value.TEAMS_INBOX_SETTLE_MS, {
    message: "TEAMS_INBOX_MAX_SETTLE_MS must be at least TEAMS_INBOX_SETTLE_MS",
  })
  // A batch retried after a runner timeout is at least timeout + max settle old
  // when it is claimed again; a stale threshold below that age would seal it
  // unanswered instead of handing it to team-agent.
  .refine(
    (value) =>
      value.TEAMS_INBOX_STALE_MS > value.TEAM_AGENT_TIMEOUT_MS + value.TEAMS_INBOX_MAX_SETTLE_MS,
    {
      message:
        "TEAMS_INBOX_STALE_MS must exceed TEAM_AGENT_TIMEOUT_MS plus TEAMS_INBOX_MAX_SETTLE_MS",
    },
  );

export function teamsInboxConfigFromEnv(env: NodeJS.ProcessEnv): TeamsInboxConfig {
  if (env.TEAM_AGENT_BASE_URL === undefined || env.TEAM_AGENT_BASE_URL.trim() === "") {
    return { enabled: false };
  }

  const parsed = enabledTeamsInboxEnvSchema.safeParse({
    TEAM_AGENT_BASE_URL: env.TEAM_AGENT_BASE_URL,
    TEAM_AGENT_API_TOKEN: env.TEAM_AGENT_API_TOKEN,
    TEAM_AGENT_TIMEOUT_MS: env.TEAM_AGENT_TIMEOUT_MS ?? 120_000,
    TEAMS_INBOX_SETTLE_MS: env.TEAMS_INBOX_SETTLE_MS ?? 3_000,
    TEAMS_INBOX_MAX_SETTLE_MS: env.TEAMS_INBOX_MAX_SETTLE_MS ?? 15_000,
    TEAMS_INBOX_STALE_MS: env.TEAMS_INBOX_STALE_MS ?? 600_000,
  });
  if (!parsed.success) {
    throw new ConfigError("Invalid Team Agent inbox configuration", { cause: parsed.error });
  }

  return {
    enabled: true,
    baseUrl: parsed.data.TEAM_AGENT_BASE_URL,
    apiToken: parsed.data.TEAM_AGENT_API_TOKEN,
    timeoutMs: parsed.data.TEAM_AGENT_TIMEOUT_MS,
    settleMs: parsed.data.TEAMS_INBOX_SETTLE_MS,
    maxSettleMs: parsed.data.TEAMS_INBOX_MAX_SETTLE_MS,
    staleMs: parsed.data.TEAMS_INBOX_STALE_MS,
  };
}
