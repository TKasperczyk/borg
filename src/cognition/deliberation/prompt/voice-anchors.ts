// Formats stable self-patterns into planner voice anchors.
import type { SelfSnapshot } from "../types.js";

export function summarizeVoiceAnchors(selfSnapshot: SelfSnapshot): string | null {
  const heldValues = selfSnapshot.values.filter((value) => value.state === "established");

  if (heldValues.length === 0) {
    return null;
  }

  return [
    `Active voice anchors (held values): ${heldValues.map((value) => value.label).join(", ")}.`,
    "Let voice_note reflect these where the turn allows.",
  ].join("\n");
}
