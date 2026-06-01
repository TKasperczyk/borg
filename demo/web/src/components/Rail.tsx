export type RouteId =
  | "cognition"
  | "stream"
  | "memory"
  | "graph"
  | "identity"
  | "commit"
  | "directives"
  | "shared"
  | "dream"
  | "prompts";

export type RailItem = {
  id: RouteId;
  label: string;
  short: string;
  glyph: string;
  num: number;
};

export const RAIL_ITEMS: readonly RailItem[] = [
  { id: "cognition", label: "cognition", short: "COG", glyph: "ψ", num: 1 },
  { id: "stream", label: "stream", short: "STR", glyph: "≣", num: 2 },
  { id: "memory", label: "memory", short: "MEM", glyph: "◇", num: 3 },
  { id: "graph", label: "graph", short: "GRF", glyph: "✦", num: 4 },
  { id: "identity", label: "identity", short: "IDN", glyph: "◐", num: 5 },
  { id: "shared", label: "shared", short: "SHR", glyph: "∞", num: 6 },
  { id: "commit", label: "commit", short: "CMT", glyph: "↵", num: 7 },
  { id: "directives", label: "directives", short: "DIR", glyph: "§", num: 8 },
  { id: "dream", label: "dream", short: "DRM", glyph: "☾", num: 9 },
  { id: "prompts", label: "prompts", short: "PMT", glyph: "›", num: 10 },
];

export type RailProps = {
  route: RouteId;
  setRoute: (route: RouteId) => void;
};

export function Rail({ route, setRoute }: RailProps) {
  return (
    <div className="rail">
      <div className="rail-brand">b</div>
      <div className="rail-list">
        {RAIL_ITEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`rail-btn ${route === item.id ? "active" : ""}`}
            onClick={() => setRoute(item.id)}
            title={`${item.label} (⌘${item.num})`}
            aria-label={item.label}
            aria-current={route === item.id ? "page" : undefined}
          >
            <span className="num">{item.num}</span>
            <span className="glyph" aria-hidden="true">
              {item.glyph}
            </span>
            <span className="label">{item.short}</span>
          </button>
        ))}
      </div>
      <div className="rail-spacer"></div>
    </div>
  );
}
