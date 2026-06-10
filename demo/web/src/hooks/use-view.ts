import { useCallback, useEffect, useRef, useState } from "react";

import type { DreamProcessName, MemoryBandId } from "../api/types";
import {
  DEFAULT_GOVERNANCE_TAB_ID,
  DEFAULT_ROUTE_ID,
  isGovernanceTabId,
  isDreamProcessName,
  isMemoryBandId,
  isRouteId,
  type GovernanceTabId,
  type RouteId,
  type RouteNavigationOptions,
} from "../routes";

export type ViewState = {
  view: RouteId;
  dreamProcess: DreamProcessName | null;
  governanceTab: GovernanceTabId;
  memoryBand: MemoryBandId | null;
};

type UseViewOptions = {
  onBlockedPopState?: (state: ViewState) => void;
  shouldBlockPopState?: (current: ViewState, next: ViewState) => boolean;
};

export const LEGACY_VIEW_ALIASES: Readonly<Record<string, ViewState>> = {
  commit: {
    view: "governance",
    dreamProcess: null,
    governanceTab: "commitments",
    memoryBand: null,
  },
  directives: {
    view: "governance",
    dreamProcess: null,
    governanceTab: "shared_state",
    memoryBand: null,
  },
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

function normalizeMemoryBand(value: string | null): MemoryBandId | null {
  return value !== null && isMemoryBandId(value) ? value : null;
}

function normalizeDreamProcess(value: string | null): DreamProcessName | null {
  return value !== null && isDreamProcessName(value) ? value : null;
}

function applyViewStateToUrl(url: URL, state: ViewState): void {
  if (state.view === DEFAULT_ROUTE_ID) {
    url.searchParams.delete("view");
    url.searchParams.delete("tab");
    url.searchParams.delete("band");
    url.searchParams.delete("process");
    return;
  }

  url.searchParams.set("view", state.view);
  if (state.view === "governance" && state.governanceTab !== DEFAULT_GOVERNANCE_TAB_ID) {
    url.searchParams.set("tab", state.governanceTab);
  } else {
    url.searchParams.delete("tab");
  }
  if (state.view === "memory" && state.memoryBand !== null) {
    url.searchParams.set("band", state.memoryBand);
  } else {
    url.searchParams.delete("band");
  }
  if (state.view === "dream" && state.dreamProcess !== null) {
    url.searchParams.set("process", state.dreamProcess);
  } else {
    url.searchParams.delete("process");
  }
}

function viewStatePath(state: ViewState): string {
  const url = new URL(window.location.href);
  applyViewStateToUrl(url, state);
  return `${url.pathname}${url.search}${url.hash}`;
}

function writeViewToUrl(state: ViewState): void {
  const url = new URL(window.location.href);
  const rawView = url.searchParams.get("view");
  const rawTab = url.searchParams.get("tab");
  const rawBand = url.searchParams.get("band");
  const rawProcess = url.searchParams.get("process");

  applyViewStateToUrl(url, state);

  const nextView = url.searchParams.get("view");
  const nextTab = url.searchParams.get("tab");
  const nextBand = url.searchParams.get("band");
  const nextProcess = url.searchParams.get("process");
  if (
    rawView !== nextView ||
    rawTab !== nextTab ||
    rawBand !== nextBand ||
    rawProcess !== nextProcess
  ) {
    window.history.replaceState(
      window.history.state,
      "",
      `${url.pathname}${url.search}${url.hash}`,
    );
  }
}

function readViewFromUrl({
  writeNormalizedUrl = true,
}: { writeNormalizedUrl?: boolean } = {}): ViewState {
  const url = new URL(window.location.href);
  const rawView = url.searchParams.get("view");
  const aliased = rawView === null ? undefined : LEGACY_VIEW_ALIASES[rawView];
  const view = aliased?.view ?? normalizeView(rawView);
  const governanceTab =
    view === "governance"
      ? (aliased?.governanceTab ?? normalizeGovernanceTab(url.searchParams.get("tab")))
      : DEFAULT_GOVERNANCE_TAB_ID;
  const memoryBand = view === "memory" ? normalizeMemoryBand(url.searchParams.get("band")) : null;
  const dreamProcess =
    view === "dream" ? normalizeDreamProcess(url.searchParams.get("process")) : null;

  const state: ViewState = {
    view,
    dreamProcess: aliased?.dreamProcess ?? dreamProcess,
    governanceTab,
    memoryBand: aliased?.memoryBand ?? memoryBand,
  };
  if (writeNormalizedUrl) {
    writeViewToUrl(state);
  }

  return state;
}

export function useView({ onBlockedPopState, shouldBlockPopState }: UseViewOptions = {}): {
  view: RouteId;
  dreamProcess: DreamProcessName | null;
  governanceTab: GovernanceTabId;
  memoryBand: MemoryBandId | null;
  setView: (view: RouteId, options?: RouteNavigationOptions) => void;
  setGovernanceTab: (tab: GovernanceTabId) => void;
} {
  const [state, setViewState] = useState(readViewFromUrl);
  const stateRef = useRef(state);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    const onPopState = () => {
      const nextState = readViewFromUrl({ writeNormalizedUrl: false });
      const currentState = stateRef.current;
      if (shouldBlockPopState?.(currentState, nextState) === true) {
        window.history.pushState(window.history.state, "", viewStatePath(currentState));
        onBlockedPopState?.(nextState);
        return;
      }
      writeViewToUrl(nextState);
      setViewState(nextState);
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [onBlockedPopState, shouldBlockPopState]);

  const setView = useCallback((nextView: RouteId, options: RouteNavigationOptions = {}) => {
    const normalized = normalizeView(nextView);
    const nextState: ViewState = {
      view: normalized,
      dreamProcess:
        normalized === "dream" && options.dreamProcess !== undefined
          ? normalizeDreamProcess(options.dreamProcess)
          : null,
      governanceTab:
        normalized === "governance"
          ? (options.governanceTab ?? DEFAULT_GOVERNANCE_TAB_ID)
          : DEFAULT_GOVERNANCE_TAB_ID,
      memoryBand:
        normalized === "memory" && options.memoryBand !== undefined
          ? normalizeMemoryBand(options.memoryBand)
          : null,
    };
    writeViewToUrl(nextState);
    setViewState(nextState);
  }, []);

  const setGovernanceTab = useCallback((tab: GovernanceTabId) => {
    const nextState: ViewState = {
      view: "governance",
      dreamProcess: null,
      governanceTab: normalizeGovernanceTab(tab),
      memoryBand: null,
    };
    writeViewToUrl(nextState);
    setViewState(nextState);
  }, []);

  return {
    view: state.view,
    dreamProcess: state.dreamProcess,
    governanceTab: state.governanceTab,
    memoryBand: state.memoryBand,
    setView,
    setGovernanceTab,
  };
}
