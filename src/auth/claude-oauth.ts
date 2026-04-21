import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";

import { z } from "zod";

import { withFileLock } from "../stream/file-lock.js";
import { readJsonFile, writeJsonFileAtomic } from "../util/atomic-write.js";
import type { Clock } from "../util/clock.js";
import { SystemClock } from "../util/clock.js";
import { AuthError } from "../util/errors.js";

const OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token";
const TOKEN_REFRESH_BUFFER_MS = 5 * 60 * 1000;
const DEFAULT_CREDENTIALS_PATH = "~/.claude/.credentials.json";

const oauthCredentialsSchema = z.object({
  accessToken: z.string().min(1),
  refreshToken: z.string().min(1),
  expiresAt: z.number().int().positive(),
  apiKey: z.string().min(1).optional(),
});

const credentialsFileSchema = z
  .object({
    claudeAiOauth: oauthCredentialsSchema.passthrough().optional(),
  })
  .passthrough();

export type OAuthCredentials = z.infer<typeof oauthCredentialsSchema>;

export type LoadedOAuthCredentials = {
  credentials: OAuthCredentials;
  raw: Record<string, unknown>;
};

export type ClaudeOAuthOptions = {
  env?: NodeJS.ProcessEnv;
  credentialsPath?: string;
};

export type GetFreshCredentialsOptions = ClaudeOAuthOptions & {
  clock?: Clock;
  forceRefresh?: boolean;
};

function expandPath(pathLike: string): string {
  if (pathLike === "~") {
    return homedir();
  }

  if (pathLike.startsWith("~/")) {
    return join(homedir(), pathLike.slice(2));
  }

  return isAbsolute(pathLike) ? pathLike : resolve(pathLike);
}

function resolveCredentialsPath(options: ClaudeOAuthOptions = {}): string {
  const env = options.env ?? process.env;
  return expandPath(
    options.credentialsPath ?? env.BORG_CLAUDE_CREDENTIALS_PATH ?? DEFAULT_CREDENTIALS_PATH,
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function getClaudeCredentialsPath(options: ClaudeOAuthOptions = {}): string {
  return resolveCredentialsPath(options);
}

function readCredentialsFile(options: ClaudeOAuthOptions = {}): LoadedOAuthCredentials {
  const credentialsPath = resolveCredentialsPath(options);
  let rawValue: unknown;

  try {
    rawValue = readJsonFile<unknown>(credentialsPath);
  } catch (error) {
    throw new AuthError("Claude OAuth credentials file is malformed", {
      cause: error,
      code: "AUTH_CREDENTIALS_MALFORMED",
    });
  }

  if (rawValue === undefined) {
    throw new AuthError("Claude OAuth credentials file was not found", {
      code: "AUTH_NO_CREDENTIALS",
    });
  }

  const parsed = credentialsFileSchema.safeParse(rawValue);

  if (!parsed.success || parsed.data.claudeAiOauth === undefined) {
    throw new AuthError("Claude OAuth credentials file is malformed", {
      cause: parsed.success ? undefined : parsed.error,
      code: "AUTH_CREDENTIALS_MALFORMED",
    });
  }

  const raw = isRecord(rawValue) ? rawValue : {};

  return {
    credentials: parsed.data.claudeAiOauth,
    raw,
  };
}

export function loadCredentials(options: ClaudeOAuthOptions = {}): LoadedOAuthCredentials | null {
  try {
    return readCredentialsFile(options);
  } catch {
    return null;
  }
}

export function saveCredentials(creds: OAuthCredentials, options: ClaudeOAuthOptions = {}): void {
  const credentialsPath = resolveCredentialsPath(options);
  const existing = loadCredentials({
    ...options,
    credentialsPath,
  });

  const root = isRecord(existing?.raw) ? { ...existing.raw } : {};
  const currentOauth = isRecord(root.claudeAiOauth)
    ? (root.claudeAiOauth as Record<string, unknown>)
    : {};

  root.claudeAiOauth = {
    ...currentOauth,
    ...creds,
  };

  writeJsonFileAtomic(credentialsPath, root);
}

export async function refreshAccessToken(refreshToken: string): Promise<OAuthCredentials> {
  let response: Response;

  try {
    response = await fetch(OAUTH_TOKEN_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        grant_type: "refresh_token",
        refresh_token: refreshToken,
        client_id: OAUTH_CLIENT_ID,
      }),
    });
  } catch (error) {
    throw new AuthError("Failed to refresh Claude OAuth access token", {
      cause: error,
      code: "AUTH_REFRESH_FAILED",
    });
  }

  if (!response.ok) {
    throw new AuthError("Failed to refresh Claude OAuth access token", {
      cause: await response.text(),
      code: "AUTH_REFRESH_FAILED",
    });
  }

  const parsed = z
    .object({
      access_token: z.string().min(1),
      refresh_token: z.string().min(1),
      expires_in: z.number().int().positive(),
    })
    .safeParse((await response.json()) as unknown);

  if (!parsed.success) {
    throw new AuthError("Claude OAuth token refresh returned an invalid payload", {
      cause: parsed.error,
      code: "AUTH_REFRESH_FAILED",
    });
  }

  return {
    accessToken: parsed.data.access_token,
    refreshToken: parsed.data.refresh_token,
    expiresAt: Date.now() + parsed.data.expires_in * 1000,
  };
}

export async function getFreshCredentials(
  options: GetFreshCredentialsOptions = {},
): Promise<OAuthCredentials | null> {
  const clock = options.clock ?? new SystemClock();
  const credentialsPath = resolveCredentialsPath(options);

  if (!existsSync(credentialsPath)) {
    return null;
  }

  const lockPath = `${credentialsPath}.lock`;

  try {
    return await withFileLock(lockPath, async () => {
      const loaded = loadCredentials({
        ...options,
        credentialsPath,
      });

      if (loaded === null) {
        return null;
      }

      const now = clock.now();
      const shouldRefresh =
        options.forceRefresh === true ||
        loaded.credentials.expiresAt < now + TOKEN_REFRESH_BUFFER_MS;

      if (!shouldRefresh) {
        return loaded.credentials;
      }

      const refreshed = await refreshAccessToken(loaded.credentials.refreshToken);
      const merged = {
        ...loaded.credentials,
        ...refreshed,
        apiKey: refreshed.apiKey ?? loaded.credentials.apiKey,
      } satisfies OAuthCredentials;

      saveCredentials(merged, {
        ...options,
        credentialsPath,
      });

      return merged;
    });
  } catch {
    return null;
  }
}

export function formatCredentialPathForDisplay(options: ClaudeOAuthOptions = {}): string {
  const resolvedPath = resolveCredentialsPath(options);
  const defaultPath = expandPath(DEFAULT_CREDENTIALS_PATH);
  return resolvedPath === defaultPath ? "~/.claude/.credentials.json" : resolvedPath;
}

export function getCredentialsLockPath(options: ClaudeOAuthOptions = {}): string {
  return `${resolveCredentialsPath(options)}.lock`;
}

export function getCredentialsDirectory(options: ClaudeOAuthOptions = {}): string {
  return dirname(resolveCredentialsPath(options));
}
