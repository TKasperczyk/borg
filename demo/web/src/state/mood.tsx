import { type ReactNode, useEffect } from "react";

import type { MoodState } from "../api/types";
import { useAppState } from "./app-state";

export const NEUTRAL_ACCENT = "oklch(0.78 0.02 95)";
export const NEUTRAL_ACCENT_DIM = "oklch(0.78 0.02 95 / 0.4)";
export const NEUTRAL_ACCENT_BACKDROP = "oklch(0.78 0.02 95 / 0.07)";

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function moodHue(valence: number): number {
  return 25 + (clamp(valence, -1, 1) + 1) * 70;
}

export function moodArousal(mood: MoodState | null | undefined): number {
  return clamp(mood?.arousal ?? 0, 0, 1);
}

export function moodLabel(valence: number, arousal: number): string {
  const v = clamp(valence, -1, 1);
  const a = clamp(arousal, 0, 1);

  if (v > 0.25 && a > 0.45) {
    return "engaged";
  }
  if (v > 0.25) {
    return "settled";
  }
  if (v < -0.25 && a > 0.45) {
    return "strained";
  }
  if (v < -0.25) {
    return "subdued";
  }
  if (a > 0.5) {
    return "alert";
  }

  return "level";
}

export function accentVarsForMood(mood: MoodState | null | undefined): {
  ac: string;
  acD: string;
  acB: string;
} {
  if (mood === null || mood === undefined) {
    return {
      ac: NEUTRAL_ACCENT,
      acD: NEUTRAL_ACCENT_DIM,
      acB: NEUTRAL_ACCENT_BACKDROP,
    };
  }

  const hue = moodHue(mood.valence);
  return {
    ac: `oklch(0.78 0.13 ${hue})`,
    acD: `oklch(0.78 0.13 ${hue} / 0.4)`,
    acB: `oklch(0.78 0.13 ${hue} / 0.07)`,
  };
}

export function MoodProvider({ children }: { children: ReactNode }) {
  const state = useAppState();
  const mood = state.data?.current_mood ?? null;

  useEffect(() => {
    const vars = accentVarsForMood(mood);
    const root = document.documentElement;
    root.style.setProperty("--ac", vars.ac);
    root.style.setProperty("--acD", vars.acD);
    root.style.setProperty("--acB", vars.acB);

    return () => {
      root.style.removeProperty("--ac");
      root.style.removeProperty("--acD");
      root.style.removeProperty("--acB");
    };
  }, [mood]);

  return <>{children}</>;
}
