import { useCallback, useEffect, useState } from "react";

import {
  DEFAULT_GOVERNANCE_TAB_ID,
  DEFAULT_ROUTE_ID,
  isGovernanceTabId,
  isRouteId,
  type GovernanceTabId,
  type RouteId,
  type RouteNavigationOptions,
} from "../routes";

type ViewState = {
  view: RouteId;
  governanceTab: GovernanceTabId;
};

export const LEGACY_VIEW_ALIASES: Readonly<Record<string, ViewState>> = {
  commit: { view: "governance", governanceTab: "commitments" },
  directives: { view: "governance", governanceTab: "shared_state" },
};

function normalizeView(value: string | null): RouteId {
  if (value === null || value.length === 0 || value === DEFAULT_ROUTE_ID) {
    return DEFAULT_ROUTE_ID;
  }

  const aliased = LEGACY_VIEW_ALIASES[value]?.view;
  if (aliased !== undefined) {
    return aliased;
  }

  return isRouteId(value) ? value : DEFAULT_ROUTE_ID;
}

function normalizeGovernanceTab(value: string | null): GovernanceTabId {
  if (value === null || value.length === 0 || value === DEFAULT_GOVERNANCE_TAB_ID) {
    return DEFAULT_GOVERNANCE_TAB_ID;
  }

  return isGovernanceTabId(value) ? value : DEFAULT_GOVERNANCE_TAB_ID;
}

function writeViewToUrl(state: ViewState): void {
  const url = new URL(window.location.href);
  const rawView = url.searchParams.get("view");
  const rawTab = url.searchParams.get("tab");

  if (state.view === DEFAULT_ROUTE_ID) {
    url.searchParams.delete("view");
    url.searchParams.delete("tab");
  } else {
    url.searchParams.set("view", state.view);
    if (state.view === "governance" && state.governanceTab !== DEFAULT_GOVERNANCE_TAB_ID) {
      url.searchParams.set("tab", state.governanceTab);
    } else {
      url.searchParams.delete("tab");
    }
  }

  const nextView = url.searchParams.get("view");
  const nextTab = url.searchParams.get("tab");
  if (rawView !== nextView || rawTab !== nextTab) {
    window.history.replaceState(
      window.history.state,
      "",
      `${url.pathname}${url.search}${url.hash}`,
    );
  }
}

function readViewFromUrl(): ViewState {
  const url = new URL(window.location.href);
  const rawView = url.searchParams.get("view");
  const aliased = rawView === null ? undefined : LEGACY_VIEW_ALIASES[rawView];
  const view = aliased?.view ?? normalizeView(rawView);
  const governanceTab =
    view === "governance"
      ? (aliased?.governanceTab ?? normalizeGovernanceTab(url.searchParams.get("tab")))
      : DEFAULT_GOVERNANCE_TAB_ID;

  const state: ViewState = { view, governanceTab };
  writeViewToUrl(state);

  return state;
}

export function useView(): {
  view: RouteId;
  governanceTab: GovernanceTabId;
  setView: (view: RouteId, options?: RouteNavigationOptions) => void;
  setGovernanceTab: (tab: GovernanceTabId) => void;
} {
  const [state, setViewState] = useState(readViewFromUrl);

  useEffect(() => {
    const onPopState = () => setViewState(readViewFromUrl());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const setView = useCallback((nextView: RouteId, options: RouteNavigationOptions = {}) => {
    const normalized = normalizeView(nextView);
    const nextState: ViewState = {
      view: normalized,
      governanceTab:
        normalized === "governance"
          ? (options.governanceTab ?? DEFAULT_GOVERNANCE_TAB_ID)
          : DEFAULT_GOVERNANCE_TAB_ID,
    };
    writeViewToUrl(nextState);
    setViewState(nextState);
  }, []);

  const setGovernanceTab = useCallback((tab: GovernanceTabId) => {
    const nextState: ViewState = {
      view: "governance",
      governanceTab: normalizeGovernanceTab(tab),
    };
    writeViewToUrl(nextState);
    setViewState(nextState);
  }, []);

  return { view: state.view, governanceTab: state.governanceTab, setView, setGovernanceTab };
}
