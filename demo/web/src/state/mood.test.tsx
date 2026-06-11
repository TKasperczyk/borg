import { render, waitFor } from "@testing-library/react";

import type { ApiState } from "../api/types";
import { StateProvider } from "./app-state";
import {
  accentVarsForMood,
  moodHue,
  moodLabel,
  MoodProvider,
  NEUTRAL_ACCENT,
  NEUTRAL_ACCENT_BACKDROP,
  NEUTRAL_ACCENT_DIM,
  useMood,
} from "./mood";

function stateWithMood(current_mood: ApiState["current_mood"]): ApiState {
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
    current_mood,
    version: "0.1.0",
  };
}

function MoodProbe() {
  const mood = useMood();
  return <div>{mood.label}</div>;
}

describe("mood", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    document.documentElement.removeAttribute("style");
  });

  it("maps valence to hue and clamps", () => {
    expect(moodHue(-1)).toBe(25);
    expect(moodHue(0)).toBe(95);
    expect(moodHue(1)).toBe(165);
    expect(moodHue(-2)).toBe(25);
    expect(moodHue(2)).toBe(165);
  });

  it("uses neutral accent vars when mood is absent", () => {
    expect(accentVarsForMood(null)).toEqual({
      ac: NEUTRAL_ACCENT,
      acD: NEUTRAL_ACCENT_DIM,
      acB: NEUTRAL_ACCENT_BACKDROP,
    });
  });

  it("sets neutral vars on documentElement for null mood", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(stateWithMood(null)), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    render(
      <StateProvider>
        <MoodProvider>
          <div />
        </MoodProvider>
      </StateProvider>,
    );

    await waitFor(() => {
      expect(document.documentElement.style.getPropertyValue("--ac")).toBe(NEUTRAL_ACCENT);
      expect(document.documentElement.style.getPropertyValue("--acD")).toBe(NEUTRAL_ACCENT_DIM);
      expect(document.documentElement.style.getPropertyValue("--acB")).toBe(
        NEUTRAL_ACCENT_BACKDROP,
      );
    });
  });

  it("requests state for the active session and exposes that session mood", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      return new Response(
        JSON.stringify(
          stateWithMood({
            session_id: url.includes("s_mood") ? "s_mood" : "default",
            valence: url.includes("s_mood") ? 0.5 : -0.5,
            arousal: 0.6,
            updated_at: 1,
            half_life_hours: 8,
            recent_triggers: [],
          }),
        ),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      );
    });

    const { getByText } = render(
      <StateProvider sessionId="s_mood">
        <MoodProvider>
          <MoodProbe />
        </MoodProvider>
      </StateProvider>,
    );

    await waitFor(() => expect(getByText("engaged")).toBeTruthy());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/state?session=s_mood",
      expect.objectContaining({ headers: expect.objectContaining({ Accept: "application/json" }) }),
    );
  });

  it("labels mood buckets", () => {
    expect(moodLabel(0.4, 0.6)).toBe("engaged");
    expect(moodLabel(0.4, 0.2)).toBe("settled");
    expect(moodLabel(-0.4, 0.6)).toBe("strained");
    expect(moodLabel(-0.4, 0.2)).toBe("subdued");
    expect(moodLabel(0, 0.6)).toBe("alert");
    expect(moodLabel(0, 0.2)).toBe("level");
  });
});
