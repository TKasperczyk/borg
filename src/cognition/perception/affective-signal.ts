import type { AffectiveSignal } from "../types.js";

export function detectAffectiveSignal(): AffectiveSignal {
  return {
    valence: 0,
    arousal: 0,
  };
}
