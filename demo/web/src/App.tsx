import { useEffect, useState } from "react";
import { Link, Route, Switch, useLocation } from "wouter";

import { useLive } from "./live/useLive";
import { ActivityPage } from "./pages/Activity";
import { ChatPage } from "./pages/Chat";
import { DreamPage } from "./pages/Dream";
import { MindPage } from "./pages/Mind";
import { ReviewsPage } from "./pages/Reviews";
import { SettingsPage } from "./pages/Settings";
import { useAppState } from "./state/app-state";
import { StateProvider } from "./state/app-state";
import { MoodProvider } from "./state/mood";

type NavItem = {
  n: string;
  label: string;
  path: string;
  page: string;
};

const NAV_ITEMS: NavItem[] = [
  { n: "01", label: "CHAT", path: "/", page: "Chat" },
  { n: "02", label: "MIND", path: "/mind", page: "Mind" },
  { n: "03", label: "REVIEWS", path: "/reviews", page: "Reviews" },
  { n: "04", label: "DREAM", path: "/dream", page: "Dream" },
  { n: "05", label: "SETTINGS", path: "/settings", page: "Settings" },
  { n: "06", label: "ACTIVITY", path: "/activity", page: "Activity" },
];

function activePage(path: string): string {
  return (
    NAV_ITEMS.find((item) =>
      item.path === "/" ? path === "/" : path === item.path || path.startsWith(`${item.path}/`),
    )?.page ?? "Chat"
  );
}

function UnknownPathRedirect() {
  const [, navigate] = useLocation();

  useEffect(() => {
    navigate("/", { replace: true });
  }, [navigate]);

  return null;
}

function AppShell({
  onActiveSessionChange,
}: {
  onActiveSessionChange: (sessionId: string | null) => void;
}) {
  const [location] = useLocation();
  const { status } = useLive();
  const state = useAppState();

  useEffect(() => {
    document.title = `BORG//CONSOLE - ${activePage(location)}`;
  }, [location]);

  return (
    <div className="app-shell">
      <nav className="nav-rail" aria-label="Primary">
        <div className="nav-header">
          <div className="nav-brand">BORG</div>
          <div className="nav-kicker">//console</div>
        </div>
        <div className="nav-links">
          {NAV_ITEMS.map((item) => {
            const active =
              item.path === "/"
                ? location === "/"
                : location === item.path || location.startsWith(`${item.path}/`);
            return (
              <Link
                key={item.path}
                className={active ? "nav-link nav-link-active" : "nav-link"}
                href={item.path}
                aria-current={active ? "page" : undefined}
              >
                <span>{item.n}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>
        <div className="nav-status">
          <div className="ws-line">
            <span
              className={status === "open" ? "ws-dot ws-dot-open pulse" : "ws-dot"}
              aria-label={`WebSocket ${status}`}
            />
            <span>WS /api/live</span>
          </div>
          {state.data?.version === undefined ? null : (
            <div className="version-line">demo · v{state.data.version}</div>
          )}
        </div>
      </nav>

      <Switch>
        <Route path="/mind/inspect/:section" component={MindPage} />
        <Route path="/mind/:tab" component={MindPage} />
        <Route path="/mind" component={MindPage} />
        <Route path="/reviews" component={ReviewsPage} />
        <Route path="/dream" component={DreamPage} />
        <Route path="/settings" component={SettingsPage} />
        <Route path="/activity" component={ActivityPage} />
        <Route path="/">
          <ChatPage onActiveSessionChange={onActiveSessionChange} />
        </Route>
        <Route component={UnknownPathRedirect} />
      </Switch>
    </div>
  );
}

export function App() {
  const [stateSessionId, setStateSessionId] = useState<string | null>(null);

  return (
    <StateProvider sessionId={stateSessionId}>
      <MoodProvider>
        <AppShell onActiveSessionChange={setStateSessionId} />
      </MoodProvider>
    </StateProvider>
  );
}
