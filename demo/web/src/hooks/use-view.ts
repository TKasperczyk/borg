import { useCallback, useEffect, useState } from "react";

import { DEFAULT_ROUTE_ID, isRouteId, type RouteId } from "../routes";

function normalizeView(value: string | null): RouteId {
  if (value === null || value.length === 0 || value === DEFAULT_ROUTE_ID) {
    return DEFAULT_ROUTE_ID;
  }

  return isRouteId(value) ? value : DEFAULT_ROUTE_ID;
}

function readViewFromUrl(): RouteId {
  const url = new URL(window.location.href);
  const view = normalizeView(url.searchParams.get("view"));
  if (view === DEFAULT_ROUTE_ID && url.searchParams.has("view")) {
    url.searchParams.delete("view");
    window.history.replaceState(window.history.state, "", `${url.pathname}${url.search}${url.hash}`);
  }

  return view;
}

export function useView(): { view: RouteId; setView: (view: RouteId) => void } {
  const [view, setViewState] = useState(readViewFromUrl);

  useEffect(() => {
    const onPopState = () => setViewState(readViewFromUrl());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const setView = useCallback((nextView: RouteId) => {
    const normalized = normalizeView(nextView);
    const url = new URL(window.location.href);
    if (normalized === DEFAULT_ROUTE_ID) {
      url.searchParams.delete("view");
    } else {
      url.searchParams.set("view", normalized);
    }
    window.history.replaceState(window.history.state, "", `${url.pathname}${url.search}${url.hash}`);
    setViewState(normalized);
  }, []);

  return { view, setView };
}
