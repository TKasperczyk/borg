import type { PromptBlockView } from "../../api/types";

export function promptTextKindLabel(kind: PromptBlockView["current_text_kind"]): string {
  switch (kind) {
    case "runtime_composed":
      return "runtime composed";
    case "stored_override":
      return "stored override";
    case "static_default":
      return "static default";
  }
}
