export type RouteId =
  | "cognition"
  | "stream"
  | "memory"
  | "graph"
  | "identity"
  | "commit"
  | "shared"
  | "dream";

export type RailItem = {
  id: RouteId;
  label: string;
  glyph: string;
};

export const RAIL_ITEMS: readonly RailItem[] = [
  { id: "cognition", label: "cognition", glyph: "◰" },
  { id: "stream", label: "stream", glyph: "≡" },
  { id: "memory", label: "memory", glyph: "⊞" },
  { id: "graph", label: "graph", glyph: "◌" },
  { id: "identity", label: "identity", glyph: "◉" },
  { id: "commit", label: "commit", glyph: "⌬" },
  { id: "shared", label: "shared", glyph: "◍" },
  { id: "dream", label: "dream", glyph: "☾" }
];

export type RailProps = {
  route: RouteId;
  setRoute: (route: RouteId) => void;
};

export function Rail({ route, setRoute }: RailProps) {
  return (
    <div className="rail">
      <div className="rail-brand">▣</div>
      {RAIL_ITEMS.map((item) => (
        <div
          key={item.id}
          className={`rail-btn ${route === item.id ? "active" : ""}`}
          onClick={() => setRoute(item.id)}
          role="button"
          tabIndex={0}
          title={item.label}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              setRoute(item.id);
            }
          }}
        >
          <span className="glyph">{item.glyph}</span>
          <span>{item.label}</span>
        </div>
      ))}
      <div className="rail-spacer"></div>
      <div className="rail-meta">
        <div className="row">
          <span className="k">tps</span>
          <span className="v">—</span>
        </div>
        <div className="row">
          <span className="k">p95</span>
          <span className="v">—</span>
        </div>
        <div className="row">
          <span className="k">emb</span>
          <span className="v">—</span>
        </div>
      </div>
    </div>
  );
}
