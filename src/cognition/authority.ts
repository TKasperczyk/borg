import type { BorgRole } from "../memory/commitments/index.js";
import type { SessionAudienceRole } from "../sessions/index.js";

export function isCreatorInOperatorContext(input: {
  currentSenderBorgRole?: BorgRole | null;
  sessionAudienceRole?: SessionAudienceRole | null;
}): boolean {
  return input.currentSenderBorgRole === "creator" && input.sessionAudienceRole === "operator";
}
