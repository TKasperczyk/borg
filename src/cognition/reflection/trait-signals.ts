import type { CognitiveMode } from "../types.js";

export const MODE_TRAIT_MAP: Record<CognitiveMode, string | null> = {
  reflective: "introspective",
  relational: "warm",
  problem_solving: "engaged",
  idle: null,
};
