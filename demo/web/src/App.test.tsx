import { render, screen } from "@testing-library/react";

import { installMockWebSocket } from "./__tests__/mock-websocket";
import { App } from "./App";
import type { ApiState } from "./api/types";
import { LiveProvider } from "./live/useLive";
import { StateProvider } from "./state/app-state";
import { MoodProvider } from "./state/mood";

function mockState(): ApiState {
  return {
    active_session: "default",
    audiences: [],
    counts: {
      turns: 0,
      commitments: 0,
      open_qs: 0,
      open_reviews: 0,
      dream_audit_rows: 0,
    },
    current_mood: {
      session_id: "default",
      valence: 0,
      arousal: 0,
      updated_at: 1,
      half_life_hours: 24,
      recent_triggers: [],
    },
    version: "0.1.0",
  };
}

function renderApp(path = "/") {
  window.history.pushState({}, "", path);
  installMockWebSocket();
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(JSON.stringify(mockState()), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  );

  return render(
    <LiveProvider>
      <StateProvider>
        <MoodProvider>
          <App />
        </MoodProvider>
      </StateProvider>
    </LiveProvider>,
  );
}

describe("App shell", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders six nav links and marks the active route", async () => {
    renderApp("/mind");

    expect(screen.getAllByRole("link")).toHaveLength(6);
    expect(screen.getByRole("link", { name: /02MIND/i }).getAttribute("aria-current")).toBe(
      "page",
    );
    expect(screen.getByRole("link", { name: /01CHAT/i }).getAttribute("aria-current")).toBeNull();
    expect(await screen.findByText("demo · v0.1.0")).toBeTruthy();
  });
});
