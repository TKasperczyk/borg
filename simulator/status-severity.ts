import type { OverseerFindingStatusImpact, OverseerVerdict } from "./types.js";

export function statusImpactSeverity(impact: OverseerFindingStatusImpact | undefined): number {
  switch (impact) {
    case "failing":
      return 2;
    case "concerning":
      return 1;
    case "minor":
      return 0.5;
    case "none":
    case undefined:
      return 0;
  }
}

export function statusSeverity(status: OverseerVerdict["status"]): number {
  switch (status) {
    case "failing":
      return 2;
    case "concerning":
      return 1;
    case "healthy":
      return 0;
  }
}

export function statusFromSeverity(severity: number): OverseerVerdict["status"] {
  if (severity >= 2) {
    return "failing";
  }

  if (severity >= 1) {
    return "concerning";
  }

  return "healthy";
}
