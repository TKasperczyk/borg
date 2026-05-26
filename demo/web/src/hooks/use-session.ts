import { useCallback, useEffect, useState } from "react";

export const DEFAULT_DEMO_SESSION_ID = "default";
const SESSION_ID_PATTERN = /^sess_[a-z0-9]{16}$/;

function normalizeSessionId(value: string | null): string {
  if (value === null || value.length === 0 || value === DEFAULT_DEMO_SESSION_ID) {
    return DEFAULT_DEMO_SESSION_ID;
  }

  return SESSION_ID_PATTERN.test(value) ? value : DEFAULT_DEMO_SESSION_ID;
}

function readSessionFromUrl(): string {
  const url = new URL(window.location.href);
  const session = normalizeSessionId(url.searchParams.get("session"));
  if (session === DEFAULT_DEMO_SESSION_ID && url.searchParams.has("session")) {
    url.searchParams.delete("session");
    window.history.replaceState(window.history.state, "", `${url.pathname}${url.search}${url.hash}`);
  }

  return session;
}

export function useSession(): { sessionId: string; setSessionId: (sessionId: string) => void } {
  const [sessionId, setSessionIdState] = useState(readSessionFromUrl);

  useEffect(() => {
    const onPopState = () => setSessionIdState(readSessionFromUrl());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const setSessionId = useCallback((nextSessionId: string) => {
    const normalized = normalizeSessionId(nextSessionId);
    const url = new URL(window.location.href);
    if (normalized === DEFAULT_DEMO_SESSION_ID) {
      url.searchParams.delete("session");
    } else {
      url.searchParams.set("session", normalized);
    }
    window.history.replaceState(window.history.state, "", `${url.pathname}${url.search}${url.hash}`);
    setSessionIdState(normalized);
  }, []);

  return { sessionId, setSessionId };
}
