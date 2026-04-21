import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { writeJsonFileAtomic } from "../util/atomic-write.js";
import { FixedClock } from "../util/clock.js";
import {
  getFreshCredentials,
  loadCredentials,
  refreshAccessToken,
  saveCredentials,
} from "./claude-oauth.js";

function createTempCredentialsPath(tempDirs: string[]): string {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-auth-"));
  tempDirs.push(tempDir);
  return join(tempDir, "credentials.json");
}

describe("claude oauth auth", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("loads valid credentials, returns null for malformed files, and returns null when missing", () => {
    const credentialsPath = createTempCredentialsPath(tempDirs);

    expect(loadCredentials({ credentialsPath })).toBeNull();

    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "access-token",
        refreshToken: "refresh-token",
        expiresAt: 1234,
        scopes: ["user:inference"],
      },
      subscriptionType: "pro",
    });

    expect(loadCredentials({ credentialsPath })).toMatchObject({
      credentials: {
        accessToken: "access-token",
        refreshToken: "refresh-token",
        expiresAt: 1234,
      },
      raw: {
        subscriptionType: "pro",
      },
    });

    writeFileSync(credentialsPath, "{broken", "utf8");
    expect(loadCredentials({ credentialsPath })).toBeNull();
  });

  it("saves refreshed credentials by merging into the existing file", () => {
    const credentialsPath = createTempCredentialsPath(tempDirs);

    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "old-access",
        refreshToken: "old-refresh",
        expiresAt: 111,
        scopes: ["org:create_api_key", "user:inference"],
        subscriptionType: "pro",
        rateLimitTier: "plus",
      },
      anotherField: {
        keep: true,
      },
    });

    saveCredentials(
      {
        accessToken: "new-access",
        refreshToken: "new-refresh",
        expiresAt: 222,
      },
      { credentialsPath },
    );

    expect(loadCredentials({ credentialsPath })?.raw).toMatchObject({
      anotherField: {
        keep: true,
      },
      claudeAiOauth: {
        accessToken: "new-access",
        refreshToken: "new-refresh",
        expiresAt: 222,
        scopes: ["org:create_api_key", "user:inference"],
        subscriptionType: "pro",
        rateLimitTier: "plus",
      },
    });
  });

  it("posts the refresh request payload and parses a successful response", async () => {
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        return new Response(
          JSON.stringify({
            access_token: "new-access",
            refresh_token: "new-refresh",
            expires_in: 3600,
          }),
          {
            status: 200,
            headers: {
              "content-type": "application/json",
            },
          },
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await refreshAccessToken("refresh-token");

    expect(result.accessToken).toBe("new-access");
    expect(result.refreshToken).toBe("new-refresh");
    expect(result.expiresAt).toBeGreaterThan(Date.now());
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toEqual({
      grant_type: "refresh_token",
      refresh_token: "refresh-token",
      client_id: "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
    });
  });

  it("throws when token refresh fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("invalid_grant", { status: 401 })),
    );

    await expect(refreshAccessToken("bad-refresh")).rejects.toMatchObject({
      code: "AUTH_REFRESH_FAILED",
    });
  });

  it("returns unchanged credentials when they are not near expiry", async () => {
    const credentialsPath = createTempCredentialsPath(tempDirs);
    const clock = new FixedClock(1_000_000);

    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "access-token",
        refreshToken: "refresh-token",
        expiresAt: clock.now() + 10 * 60_000,
      },
    });

    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      getFreshCredentials({
        credentialsPath,
        clock,
      }),
    ).resolves.toEqual({
      accessToken: "access-token",
      refreshToken: "refresh-token",
      expiresAt: clock.now() + 10 * 60_000,
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refreshes credentials near expiry and saves the merged result", async () => {
    const credentialsPath = createTempCredentialsPath(tempDirs);
    const clock = new FixedClock(10_000);

    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "stale-access",
        refreshToken: "refresh-token",
        expiresAt: clock.now() + 60_000,
        scopes: ["user:inference"],
      },
      keep: true,
    });

    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              access_token: "fresh-access",
              refresh_token: "fresh-refresh",
              expires_in: 1800,
            }),
            {
              status: 200,
              headers: {
                "content-type": "application/json",
              },
            },
          ),
      ),
    );

    const refreshed = await getFreshCredentials({
      credentialsPath,
      clock,
    });

    expect(refreshed).toMatchObject({
      accessToken: "fresh-access",
      refreshToken: "fresh-refresh",
    });
    expect(loadCredentials({ credentialsPath })?.raw).toMatchObject({
      keep: true,
      claudeAiOauth: {
        accessToken: "fresh-access",
        refreshToken: "fresh-refresh",
        scopes: ["user:inference"],
      },
    });
  });

  it("serializes concurrent refreshes behind the shared lock", async () => {
    const credentialsPath = createTempCredentialsPath(tempDirs);
    const clock = new FixedClock(5_000);

    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "stale-access",
        refreshToken: "refresh-token",
        expiresAt: clock.now() + 1_000,
      },
    });

    let refreshCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        refreshCalls += 1;
        await new Promise((resolve) => setTimeout(resolve, 50));

        return new Response(
          JSON.stringify({
            access_token: "fresh-access",
            refresh_token: "fresh-refresh",
            expires_in: 1800,
          }),
          {
            status: 200,
            headers: {
              "content-type": "application/json",
            },
          },
        );
      }),
    );

    const [first, second] = await Promise.all([
      getFreshCredentials({
        credentialsPath,
        clock,
      }),
      getFreshCredentials({
        credentialsPath,
        clock,
      }),
    ]);

    expect(refreshCalls).toBe(1);
    expect(first).toMatchObject({
      accessToken: "fresh-access",
      refreshToken: "fresh-refresh",
    });
    expect(second).toMatchObject({
      accessToken: "fresh-access",
      refreshToken: "fresh-refresh",
    });
  });
});
