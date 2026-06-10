import { RAIL_ITEMS, routeChordLabel, type RouteId } from "../routes";
import { CountBadge } from "./CountBadge";
import type { SeverityRank } from "./SeverityChip";

export type { RouteId } from "../routes";

export type RailBadge = {
  count: number;
  severity?: SeverityRank;
  label?: string;
};

export type RailProps = {
  route: RouteId;
  setRoute: (route: RouteId) => void;
  badges?: Partial<Record<RouteId, RailBadge>>;
};

export function Rail({ route, setRoute, badges = {} }: RailProps) {
  return (
    <div className="rail">
      <div className="rail-brand">b</div>
      <div className="rail-list">
        {RAIL_ITEMS.map((item) => {
          const badge = badges[item.id];
          return (
            <button
              key={item.id}
              type="button"
              className={`rail-btn ${route === item.id ? "active" : ""}`}
              onClick={() => setRoute(item.id)}
              title={`${item.title ?? item.label} (${routeChordLabel(item)})`}
              aria-label={item.label}
              aria-current={route === item.id ? "page" : undefined}
            >
              <span className="num">{item.num}</span>
              {badge === undefined ? null : (
                <CountBadge
                  count={badge.count}
                  severity={badge.severity}
                  label={badge.label ?? item.label}
                />
              )}
              <span className="glyph" aria-hidden="true">
                {item.glyph}
              </span>
              <span className="label">{item.short}</span>
            </button>
          );
        })}
      </div>
      <div className="rail-spacer"></div>
    </div>
  );
}
