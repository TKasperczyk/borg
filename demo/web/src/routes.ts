export type RouteId =
  | "mission"
  | "cognition"
  | "stream"
  | "memory"
  | "identity"
  | "governance"
  | "review"
  | "dream"
  | "prompts"
  | "admin";

export type GovernanceTabId = "commitments" | "shared_state" | "scope" | "sessions";

export type RouteNavigationOptions = {
  governanceTab?: GovernanceTabId;
};

export type RailItem = {
  id: RouteId;
  label: string;
  title?: string;
  short: string;
  glyph: string;
  num: number;
};

export const DEFAULT_ROUTE_ID: RouteId = "mission";
export const DEFAULT_GOVERNANCE_TAB_ID: GovernanceTabId = "commitments";

export const GOVERNANCE_TAB_IDS: readonly GovernanceTabId[] = [
  "commitments",
  "shared_state",
  "scope",
  "sessions",
];

export const RAIL_ITEMS: readonly RailItem[] = [
  { id: "mission", label: "mission", short: "MC", glyph: "⌂", num: 0 },
  {
    id: "cognition",
    label: "workbench",
    title: "Conversation Workbench",
    short: "COG",
    glyph: "ψ",
    num: 1,
  },
  { id: "stream", label: "stream", short: "STR", glyph: "≣", num: 2 },
  { id: "memory", label: "memory", short: "MEM", glyph: "◇", num: 3 },
  { id: "identity", label: "identity", short: "IDN", glyph: "◐", num: 4 },
  { id: "governance", label: "governance", short: "GOV", glyph: "§", num: 5 },
  { id: "review", label: "review", short: "REV", glyph: "?", num: 6 },
  { id: "dream", label: "dream", short: "DRM", glyph: "☾", num: 7 },
  { id: "prompts", label: "prompts", short: "PMT", glyph: "›", num: 8 },
  { id: "admin", label: "admin", short: "ADM", glyph: "⚙", num: 9 },
];

export function isRouteId(value: string): value is RouteId {
  return RAIL_ITEMS.some((item) => item.id === value);
}

export function isGovernanceTabId(value: string): value is GovernanceTabId {
  return GOVERNANCE_TAB_IDS.some((item) => item === value);
}
