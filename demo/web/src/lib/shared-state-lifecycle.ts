import type { SharedStateEntryKind } from "../api/types";
import type { TagKind } from "../components/Tag";

export type PrimarySharedStateLifecycleKind = "locked" | "live" | "tentative" | "invalidated";

export function primaryLifecycleKind(
  kind: SharedStateEntryKind,
): PrimarySharedStateLifecycleKind {
  if (kind === "low_salience_live" || kind === "dormant_live") {
    return "live";
  }
  if (kind === "pending") {
    return "tentative";
  }
  return kind;
}

export function lifecycleLabel(
  kind: SharedStateEntryKind | PrimarySharedStateLifecycleKind,
): string {
  if (kind === "low_salience_live") {
    return "live - low salience";
  }
  if (kind === "dormant_live") {
    return "live - dormant";
  }
  if (kind === "pending") {
    return "pending (legacy)";
  }
  return kind;
}

export function tagKind(kind: SharedStateEntryKind): TagKind {
  if (kind === "locked") {
    return "acc";
  }
  if (kind === "live" || kind === "low_salience_live" || kind === "dormant_live") {
    return "info";
  }
  if (kind === "tentative" || kind === "pending") {
    return "warn";
  }
  if (kind === "invalidated") {
    return "bad";
  }
  return "";
}
