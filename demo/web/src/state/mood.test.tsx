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

  it("labels mood buckets", () => {
    expect(moodLabel(0.4, 0.6)).toBe("engaged");
    expect(moodLabel(0.4, 0.2)).toBe("settled");
    expect(moodLabel(-0.4, 0.6)).toBe("strained");
    expect(moodLabel(-0.4, 0.2)).toBe("subdued");
    expect(moodLabel(0, 0.6)).toBe("alert");
    expect(moodLabel(0, 0.2)).toBe("level");
  });
});
