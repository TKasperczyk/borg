import type { TagKind } from "./Tag";
import { Tag } from "./Tag";

export type WebMemoryDisclosureClass =
  | "public"
  | "relationship_private"
  | "operator_private"
  | "self_private"
  | "sensitive"
  | "unknown";

export type DisclosureLabelClass = "public" | "private" | "operator" | "unknown";

const DISCLOSURE_CLASS_MAP = {
  public: "public",
  relationship_private: "private",
  operator_private: "operator",
  self_private: "private",
  sensitive: "unknown",
  unknown: "unknown",
} satisfies Record<WebMemoryDisclosureClass, DisclosureLabelClass>;

function isWebMemoryDisclosureClass(value: unknown): value is WebMemoryDisclosureClass {
  return typeof value === "string" && Object.hasOwn(DISCLOSURE_CLASS_MAP, value);
}

export function collapseDisclosureClass(value: unknown): DisclosureLabelClass {
  if (!isWebMemoryDisclosureClass(value)) {
    return "unknown";
  }
  return DISCLOSURE_CLASS_MAP[value];
}

function disclosureTagKind(disclosure: DisclosureLabelClass): TagKind {
  if (disclosure === "public") {
    return "info";
  }
  if (disclosure === "private") {
    return "purple";
  }
  if (disclosure === "operator") {
    return "warn";
  }
  return "solid";
}

export function DisclosureLabel({ value }: { value: unknown }) {
  const disclosure = collapseDisclosureClass(value);
  return <Tag kind={disclosureTagKind(disclosure)}>{disclosure}</Tag>;
}
