export type RouteId =
  | "mission"
  | "cognition"
  | "stream"
  | "memory"
  | "identity"
  | "commit"
  | "directives"
  | "review"
  | "dream"
  | "prompts";

export type RailItem = {
  id: RouteId;
  label: string;
  short: string;
  glyph: string;
  num: number;
};

export const DEFAULT_ROUTE_ID: RouteId = "mission";

export const RAIL_ITEMS: readonly RailItem[] = [
  { id: "mission", label: "mission", short: "MC", glyph: "⌂", num: 0 },
  { id: "cognition", label: "cognition", short: "COG", glyph: "ψ", num: 1 },
  { id: "stream", label: "stream", short: "STR", glyph: "≣", num: 2 },
  { id: "memory", label: "memory", short: "MEM", glyph: "◇", num: 3 },
  { id: "identity", label: "identity", short: "IDN", glyph: "◐", num: 4 },
  { id: "commit", label: "commit", short: "CMT", glyph: "↵", num: 5 },
  { id: "directives", label: "directives", short: "DIR", glyph: "§", num: 6 },
  { id: "review", label: "review", short: "REV", glyph: "?", num: 7 },
  { id: "dream", label: "dream", short: "DRM", glyph: "☾", num: 8 },
  { id: "prompts", label: "prompts", short: "PMT", glyph: "›", num: 9 },
];

export function isRouteId(value: string): value is RouteId {
  return RAIL_ITEMS.some((item) => item.id === value);
}
